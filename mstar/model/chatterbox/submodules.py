"""Node submodules for Chatterbox: the compute behind each graph node.

``VoiceEncoderSubmodule``  reference clip -> speaker embedding + S3 prompt tokens
``T3Submodule``            autoregressive speech-token generation (paged KV,
                           continuous batching, batched classifier-free
                           guidance, captured decode graphs)
``S3GenSubmodule``         speech tokens -> mel -> waveform -> watermark

Reference behaviour these mirror: ``chatterbox/tts.py`` (``prepare_conditionals``,
``generate``), ``chatterbox/tts_turbo.py``, ``chatterbox/models/t3/t3.py``
(``inference`` / ``inference_turbo``).
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.model.chatterbox.config import (
    CFG_LABEL,
    COND_LABEL,
    S3_SR,
    S3GEN_SILENCE_TOKEN,
    T3_ATTN,
    T3_POS,
    T3_SAMPLER,
    UNCOND_LABEL,
    ChatterboxConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)

# Edge names (kept in sync with chatterbox_model.py)
TEXT_INPUTS = "text_inputs"
REF_AUDIO = "ref_audio"
VOICE_KEY = "voice_key"
SPEAKER_EMB = "speaker_emb"
PROMPT_TOKENS = "prompt_tokens"
SPEECH_TOKENS = "speech_tokens"
PREV_TOKEN = "prev_token"
AUDIO_CHUNK = "audio_chunk"
DECODE_LOOP = "decode_loop"

PREFILL_WALKS = ("prefill", "prefill_voice")


class VoiceCache:
    """Small LRU of per-voice conditioning, keyed by the clip's content hash,
    so a voice that is uploaded again (or a preset) is conditioned once."""

    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self._items: OrderedDict[int, Any] = OrderedDict()

    def get(self, key: int):
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: int, value) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)


# ===========================================================================
# 1. Voice encoder: reference audio -> T3 conditioning
# ===========================================================================


class VoiceEncoderSubmodule(NodeSubmodule):
    """Speaker embedding (LSTM voice encoder over the whole clip) and the S3
    prompt tokens (first ``enc_cond_seconds`` of the clip), as the reference
    ``prepare_conditionals`` computes them."""

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, encoder: nn.Module, s3_tokenizer: nn.Module, config: ChatterboxConfig):
        super().__init__()
        self.encoder = encoder
        self.s3_tokenizer = s3_tokenizer
        self.config = config
        self.cache = VoiceCache(config.voice_cache_size)

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> NodeInputs:
        del graph_walk, fwd_info, kwargs
        wav = inputs[REF_AUDIO][0].to(self.get_device(), torch.float32).reshape(-1)
        key = int(inputs[VOICE_KEY][0].reshape(-1)[0].item())
        return NodeInputs(tensor_inputs={REF_AUDIO: wav}, kwargs={VOICE_KEY: key})

    @torch.no_grad()
    def condition(self, wav24: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from mstar.model.chatterbox.components.audio_frontend import resample, trim_reference

        wav16 = resample(wav24, self.config.sample_rate, S3_SR)
        speaker_emb = self.encoder.embed_utterance(wav16)
        prompt_wav = trim_reference(wav16, S3_SR, self.config.enc_cond_seconds)
        tokens, lens = self.s3_tokenizer(
            [prompt_wav], max_len=self.config.t3.speech_cond_prompt_len,
        )
        return speaker_emb, tokens[0, : int(lens[0])]

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        ref_audio: torch.Tensor, voice_key: int, **kwargs,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        cached = self.cache.get(voice_key)
        if cached is None:
            cached = self.condition(ref_audio)
            self.cache.put(voice_key, cached)
        speaker_emb, prompt_tokens = cached
        return {SPEAKER_EMB: [speaker_emb], PROMPT_TOKENS: [prompt_tokens]}


# ===========================================================================
# 2. T3: text (+ voice, emotion) -> speech tokens
# ===========================================================================


@dataclass
class BuiltinT3Voice:
    """The T3 half of the checkpoint's ``conds.pt``."""

    speaker_emb: torch.Tensor   # [256]
    prompt_tokens: torch.Tensor  # [P]


class T3Submodule(ARNodeSubmodule):
    """One T3 step for a continuous batch.

    Prefill packs ``[cond | text | BOS]`` per request; decode embeds the
    previous speech token at its learned speech position. With guidance each
    request owns two KV streams, ``main`` and ``uncond``, that the step packs
    label-major into one plan (``CFG_LABEL``); the forward then sees ``2B``
    rows, combines the two logit halves with the per-request ``cfg_weight``
    and samples ``B`` tokens. Guidance on/off is the capture key, so a decode
    batch of either kind replays its own CUDA graph.
    """

    # Sampling, guidance and the min-p mask are plain tensor ops but the
    # forward branches on the guidance mode; the decode graph is captured
    # whole, which is where the time goes.
    disable_torch_compile = True
    MAX_BATCH_SIZE = 32
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(
        self, model: nn.Module, config: ChatterboxConfig, builtin_voice: BuiltinT3Voice | None,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.t3 = config.t3
        self.builtin_voice = builtin_voice
        self.supports_cfg = not config.is_turbo

    # -- request knobs -------------------------------------------------------

    def _knob(self, step_metadata: Mapping[str, Any] | None, key: str):
        step_metadata = step_metadata or {}
        return step_metadata.get(key, getattr(self.config.generation, key))

    def _requires_cfg(self, step_metadata: Mapping[str, Any] | None) -> bool:
        return self.supports_cfg and float(self._knob(step_metadata, "cfg_weight")) > 0.0

    def cg_key_info(self, graph_walk: str, per_request_info: dict[str, CurrentForwardPassInfo]):
        """Guidance on or off; a mixed batch matches no capture and runs eagerly."""
        del graph_walk
        flags = {self._requires_cfg(info.step_metadata) for info in per_request_info.values()}
        return flags.pop() if len(flags) == 1 else None

    def _scalar_inputs(self, fwd_info: CurrentForwardPassInfo, requires_cfg: bool) -> dict[str, torch.Tensor]:
        # sampling knobs (temperature, min_p, penalty) live in the request's
        # SamplingReqConfig and are applied by the sampler resource
        return {
            "cfg_weight": torch.tensor(
                [float(self._knob(fwd_info.step_metadata, "cfg_weight")) if requires_cfg else 0.0],
                dtype=torch.float32, device=self.get_device(),
            ),
        }

    # -- inputs ----------------------------------------------------------------

    def _voice(self, inputs: NameToTensorList) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.get_device()
        if SPEAKER_EMB in inputs:
            return (
                inputs[SPEAKER_EMB][0].to(device).reshape(-1),
                inputs[PROMPT_TOKENS][0].to(device, torch.long).reshape(-1),
            )
        if self.builtin_voice is None:
            raise ValueError("No reference voice given and the checkpoint ships no built-in voice")
        return self.builtin_voice.speaker_emb, self.builtin_voice.prompt_tokens

    @torch.no_grad()
    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> ARNodeInputs:
        del kwargs
        device = self.get_device()
        state = self.request_state(fwd_info.request_id)
        if graph_walk in PREFILL_WALKS:
            requires_cfg = self._requires_cfg(fwd_info.step_metadata)
            text_ids = inputs[TEXT_INPUTS][0].to(device, torch.long).reshape(-1)
            speaker_emb, prompt_tokens = self._voice(inputs)
            emotion = None
            if self.t3.emotion_adv:
                emotion = torch.tensor(
                    [float(self._knob(fwd_info.step_metadata, "exaggeration"))], device=device,
                )
            cond = self.model.conditioning(speaker_emb[None], prompt_tokens[None], emotion)[0]
            embeds = self.model.build_prefill_embeds(cond, text_ids, uncond=False)
            tensor_inputs = self._scalar_inputs(fwd_info, requires_cfg)
            if requires_cfg:
                tensor_inputs["uncond_embeds"] = self.model.build_prefill_embeds(
                    cond, text_ids, uncond=True,
                )
            state.add_all(speech_step=1, generated=0, requires_cfg=requires_cfg)
        elif graph_walk == "decode":
            requires_cfg = bool(state["requires_cfg"])
            step = int(state["speech_step"])
            token = inputs[PREV_TOKEN][0].to(device, torch.long).reshape(-1)
            embeds = self.model.embed_speech(token, torch.tensor([step], device=device))
            state.add("speech_step", step + 1)
            tensor_inputs = self._scalar_inputs(fwd_info, requires_cfg)
        else:
            raise ValueError(f"Unknown T3 walk {graph_walk!r}")
        return ARNodeInputs(
            input_embeds=embeds,
            input_seq_len=embeds.shape[0],
            tensor_inputs=tensor_inputs,
            resource_step_info=requires_cfg,
        )

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        """Pack the batch: all ``main`` rows, then (with guidance) all
        ``uncond`` rows, the order the combined KV plan lays the streams out."""
        del engine_inputs
        requires_cfg = bool(inputs[0].resource_step_info)
        main = torch.cat([inp.input_embeds for inp in inputs], dim=0)
        if requires_cfg:
            if graph_walk in PREFILL_WALKS:
                uncond = torch.cat([inp.tensor_inputs["uncond_embeds"] for inp in inputs], dim=0)
            else:
                uncond = main  # decode: the same token embedding feeds both streams
            embeds = torch.cat([main, uncond], dim=0)
        else:
            embeds = main
        cfg_weight = torch.cat([inp.tensor_inputs["cfg_weight"] for inp in inputs]).view(-1, 1)
        return {"input_embeds": embeds, "requires_cfg": requires_cfg, "cfg_weight": cfg_weight}

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        del slot_lease, piecewise_leases, kwargs
        requires_cfg = bool(inputs[0].resource_step_info)
        labels = (COND_LABEL, UNCOND_LABEL) if requires_cfg else (COND_LABEL,)
        segments = [
            Segment(request_id=rid, label=label, span=inp.input_seq_len)
            for label in labels
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        kv_step = (
            KVStep(combined_labels={labels: CFG_LABEL}) if requires_cfg else KVStep()
        )
        tracked = {}
        if graph_walk in PREFILL_WALKS:
            # the reference penalises repeats of everything after the text,
            # BOS included
            bos = torch.tensor([self.t3.start_speech_token], dtype=torch.long)
            tracked = {rid: bos for rid in request_ids}
        return SubmoduleStep(
            segments=segments,
            steps={
                self._kv_key(): kv_step,
                T3_ATTN: AttentionStep(causal=True),
                T3_POS: PositionStep(),
                T3_SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=tracked),
            },
            cg_key_info=requires_cfg,
        )

    @staticmethod
    def _kv_key() -> str:
        from mstar.model.chatterbox.config import T3_KV

        return T3_KV

    # -- compute ----------------------------------------------------------------

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        cfg_weight: torch.Tensor,
        requires_cfg: bool,
    ) -> torch.Tensor:
        attn = engine_inputs.resources[T3_ATTN]
        sampler = engine_inputs.resources[T3_SAMPLER]
        label = CFG_LABEL if requires_cfg else COND_LABEL
        position_ids = None
        if self.t3.backbone.is_gpt2:
            position_ids = engine_inputs.resources[T3_POS].pos_ids(label)[: input_embeds.shape[0]]
        hidden = self.model.hidden(input_embeds, label=label, position_ids=position_ids)
        if graph_walk in PREFILL_WALKS:
            hidden = attn.select_last_hidden(hidden, label=label)
        logits = self.model.logits(hidden).float()
        if requires_cfg:
            cond, uncond = logits.chunk(2, dim=0)
            logits = cond + cfg_weight * (cond - uncond)
        # penalty -> temperature -> min_p -> top-p happen in the sampler, in
        # the reference's order
        return sampler.sample(engine_inputs.request_ids, logits)

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
        cfg_weight: torch.Tensor, requires_cfg: bool = False, **kwargs,
    ) -> NameToTensorList:
        del kwargs
        tokens = self._forward(graph_walk, engine_inputs, input_embeds, cfg_weight, requires_cfg)
        return {SPEECH_TOKENS: [tokens.reshape(-1)[:1]]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
        cfg_weight: torch.Tensor, requires_cfg: bool = False, **kwargs,
    ) -> dict[str, NameToTensorList]:
        del kwargs
        tokens = self._forward(
            graph_walk, engine_inputs, input_embeds, cfg_weight, requires_cfg,
        ).reshape(-1)
        return {
            rid: {SPEECH_TOKENS: [tokens[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    # -- loop control ---------------------------------------------------------

    def postprocess(
        self, request_id: str, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]], **kwargs,
    ) -> None:
        del request_info, kwargs
        if SPEECH_TOKENS not in outputs:
            return
        # the sampled token is the next decode input; no value is read here
        outputs[PREV_TOKEN] = outputs[SPEECH_TOKENS]
        state = self.request_state(request_id)
        state.add("generated", int(state.get("generated", 0)) + 1)

    def check_stop(
        self, request_id: str, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if SPEECH_TOKENS not in outputs:
            return set()
        token = int(outputs[SPEECH_TOKENS][0].reshape(-1)[0].item())
        generated = int(self.request_state(request_id).get("generated", 0))
        max_new = int(request_info.step_metadata.get("max_new_tokens", request_info.max_tokens))
        sampling = request_info.resource_configs.get(T3_SAMPLER)
        ignore_eos = bool(getattr(sampling, "ignore_eos", False))
        if (not ignore_eos and token == self.t3.stop_speech_token) or generated >= max_new:
            return {DECODE_LOOP}
        return set()

    # -- batching and capture ---------------------------------------------------

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        if batch.graph_walk not in (*PREFILL_WALKS, "decode"):
            return False
        if not 0 < len(model_inputs) <= self.MAX_BATCH_SIZE:
            return False
        # one guidance mode per batch: the plan packs either one or two streams
        return len({bool(inp.resource_step_info) for inp in model_inputs}) == 1

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1) -> list[CudaGraphConfig]:
        del tp_world_size
        dtype = self.model.speech_emb.weight.dtype
        configs = []
        for requires_cfg in ((True, False) if self.supports_cfg else (False,)):
            configs.append(BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_embeds=torch.zeros(1, self.t3.hidden_size, dtype=dtype, device=device),
                    input_seq_len=1,
                    tensor_inputs={
                        "cfg_weight": torch.full((1,), 0.5 if requires_cfg else 0.0, device=device),
                    },
                    resource_step_info=requires_cfg,
                ),
                additional_key_info=requires_cfg,
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                # each request's step spans two streams packed into one plan
                total_tokens_multiplier=2 if requires_cfg else 1,
                compile=False,
            ))
        return configs

    def can_use_cuda_graphs(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        if batch.graph_walk != "decode" or not self.can_batch(batch, model_inputs):
            return False
        return super().can_use_cuda_graphs(batch, model_inputs)


# ===========================================================================
# 3. S3Gen: speech tokens -> waveform
# ===========================================================================


@dataclass
class StreamState:
    """One request's progress through chunked synthesis.

    ``tokens`` is every speech token received so far; ``token_offset`` how
    many of them have been turned into mel and vocoded. ``noise`` is the
    request's fixed flow-matching noise field (drawn once, indexed by frame),
    which is what keeps the mel of already-emitted frames stable when the
    decoder is re-run with more context. ``hift_mel`` / ``hift_source`` /
    ``hift_speech`` are the vocoder's held-back tail: the last mel frames,
    their harmonic excitation and the waveform that was not emitted yet; the
    next chunk re-vocodes those frames in context and crossfades them.
    """

    tokens: torch.Tensor
    token_offset: int = 0
    noise: torch.Tensor | None = None
    hift_mel: torch.Tensor | None = None
    hift_source: torch.Tensor | None = None
    hift_speech: torch.Tensor | None = None
    chunks_emitted: int = 0
    done: bool = False


class S3GenSubmodule(NodeSubmodule):
    """Flow-matching mel decoder + HiFT vocoder over a stream of speech tokens.

    A request whose whole token sequence arrives in one final chunk (the
    offline policy) is synthesised exactly as the reference does it. A
    streamed request follows CosyVoice 2's token2wav: every chunk re-runs the
    flow decoder over all tokens so far (fixed noise field, look-ahead tokens
    withheld until the end), vocodes only the new frames behind a small mel
    cache, continues the harmonic source from the previous chunk and
    crossfades the re-synthesised tail before emitting.

    The reference conditioning (prompt tokens, prompt mel, x-vector) comes
    from the checkpoint's built-in voice or from the request's reference clip,
    computed on first use and cached per voice.
    """

    disable_torch_compile = True
    disable_autocast = True

    def __init__(
        self, s3gen: nn.Module, s3_tokenizer: nn.Module, config: ChatterboxConfig,
        builtin_voice, watermarker=None,
    ):
        super().__init__()
        self.s3gen = s3gen
        self.s3_tokenizer = s3_tokenizer
        self.config = config
        self.builtin_voice = builtin_voice
        self.watermarker = watermarker
        self.cache = VoiceCache(config.voice_cache_size)
        s3 = config.s3gen
        self.frames_per_token = s3.token_mel_ratio
        self.samples_per_frame = s3.hift.upsample_factor
        self.lookahead_tokens = s3.encoder.pre_lookahead_len
        self.cache_frames = config.stream_mel_cache_frames
        self.cache_samples = self.cache_frames * self.samples_per_frame
        # crossfade of the re-synthesised tail: the second half of a Hamming
        # window fades the old tail out while the first half fades the new in
        self.register_buffer(
            "fade_window", torch.hamming_window(2 * self.cache_samples, periodic=False), persistent=False,
        )

    # -- reference conditioning ---------------------------------------------

    @torch.no_grad()
    def condition(self, wav24: torch.Tensor):
        from mstar.model.chatterbox.components.audio_frontend import resample, trim_reference

        wav24 = trim_reference(wav24, self.config.sample_rate, self.config.dec_cond_seconds)
        wav16 = resample(wav24, self.config.sample_rate, S3_SR)
        tokens, lens = self.s3_tokenizer([wav16])
        return self.s3gen.embed_reference(wav24[None], wav16[None], tokens[:, : int(lens[0])])

    def _reference(self, inputs: NameToTensorList):
        if not inputs.get(REF_AUDIO):
            if self.builtin_voice is None:
                raise ValueError("No reference voice given and the checkpoint ships no built-in voice")
            return self.builtin_voice
        key = int(inputs[VOICE_KEY][0].reshape(-1)[0].item())
        ref = self.cache.get(key)
        if ref is None:
            wav = inputs[REF_AUDIO][0].to(self.get_device(), torch.float32).reshape(-1)
            ref = self.condition(wav)
            self.cache.put(key, ref)
        return ref

    # -- inputs ----------------------------------------------------------------

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> NodeInputs:
        del graph_walk
        device = self.get_device()
        chunks = inputs.get(SPEECH_TOKENS) or []
        chunk = chunks[0] if chunks else None
        raw = (
            torch.empty(0, dtype=torch.long, device=device)
            if chunk is None or chunk.numel() == 0
            else chunk.to(device, torch.long).reshape(-1)
        )
        # The engine names the final chunk; without that (older engines) the
        # stream ends with T3's stop token or an empty flush.
        is_final = bool(kwargs.get("is_final_stream_chunk", False)) or chunk is None \
            or bool((raw == self.config.t3.stop_speech_token).any())
        # BOS/EOS and any other control id are not speech (reference
        # ``drop_invalid_tokens`` + ``< 6561`` filter)
        tokens = raw[raw < self.config.s3gen.vocab_size]
        meta = fwd_info.step_metadata
        return NodeInputs(
            tensor_inputs={SPEECH_TOKENS: tokens},
            kwargs={
                "request_id": fwd_info.request_id,
                "ref": self._reference(inputs),
                "n_timesteps": int(meta.get("n_cfm_timesteps", self.config.generation.n_cfm_timesteps)),
                "watermark": bool(meta.get("watermark", self.config.generation.watermark)),
                "seed": int(fwd_info.random_seed),
                "is_final": is_final,
            },
        )

    # -- synthesis ----------------------------------------------------------------

    def _generator(self, seed: int, device) -> torch.Generator:
        return torch.Generator(device=device).manual_seed(seed)

    def _with_trailing_silence(self, tokens: torch.Tensor) -> torch.Tensor:
        n = self.config.trailing_silence_tokens
        if n <= 0 or tokens.numel() == 0:
            return tokens
        silence = torch.full((n,), S3GEN_SILENCE_TOKEN, dtype=torch.long, device=tokens.device)
        return torch.cat([tokens, silence])

    @torch.no_grad()
    def synthesize(
        self, tokens: torch.Tensor, ref, n_timesteps: int, seed: int, watermark: bool,
    ) -> torch.Tensor:
        """Whole-utterance synthesis, PCM16: the reference path, one call."""
        tokens = self._with_trailing_silence(tokens)
        if tokens.numel() == 0:
            return torch.zeros(0, dtype=torch.int16, device=tokens.device)
        generator = self._generator(seed, tokens.device)
        lens = torch.tensor([tokens.numel()], dtype=torch.long, device=tokens.device)
        mel = self.s3gen.tokens_to_mel(tokens[None], lens, ref, n_timesteps=n_timesteps, generator=generator)
        wav = self.s3gen.mel_to_wav(mel, generator=generator)[0]
        return self._finish(wav, watermark)

    def _finish(self, wav: torch.Tensor, watermark: bool) -> torch.Tensor:
        if watermark and self.watermarker is not None and wav.numel() > 0:
            wav = self.watermarker.apply(wav, self.config.sample_rate)
        return (wav.clamp(-1, 1) * 32767).to(torch.int16)

    def _noise_field(self, state: StreamState, ref, generator: torch.Generator) -> torch.Tensor:
        if state.noise is None:
            prompt_frames = ref.num_prompt_tokens * self.frames_per_token
            max_tokens = self.config.t3.max_speech_tokens + self.config.trailing_silence_tokens
            frames = prompt_frames + max_tokens * self.frames_per_token
            state.noise = torch.randn(
                (1, self.config.s3gen.output_size, frames),
                dtype=self.s3gen.dtype, device=self.get_device(), generator=generator,
            )
        return state.noise

    @torch.no_grad()
    def synthesize_chunk(
        self, state: StreamState, new_tokens: torch.Tensor, is_final: bool, ref,
        n_timesteps: int, generator: torch.Generator,
    ) -> torch.Tensor:
        """Advance one request by a chunk of tokens; returns the waveform to
        emit now (float, possibly empty)."""
        device = self.get_device()
        if state.done:
            return torch.zeros(0, device=device)
        state.tokens = torch.cat([state.tokens, new_tokens.to(device)])
        if is_final:
            state.tokens = self._with_trailing_silence(state.tokens)
        n = state.tokens.numel()
        usable = n if is_final else max(n - self.lookahead_tokens, 0)
        if usable <= state.token_offset:
            if not is_final:
                return torch.zeros(0, device=device)  # not enough new tokens yet
            state.done = True
            # nothing new to decode: release the held-back tail as it is
            tail = state.hift_speech
            return tail[0] if tail is not None else torch.zeros(0, device=device)

        lens = torch.tensor([n], dtype=torch.long, device=device)
        noise = self._noise_field(state, ref, generator)
        mel = self.s3gen.tokens_to_mel(
            state.tokens[None], lens, ref, n_timesteps=n_timesteps, noise=noise, finalize=is_final,
        )
        new_mel = mel[:, :, state.token_offset * self.frames_per_token:]
        state.token_offset = usable

        first = state.hift_mel is None
        mel_in = new_mel if first else torch.cat([state.hift_mel, new_mel], dim=2)
        wav, source = self.s3gen.vocode(
            mel_in, generator=generator, cache_source=state.hift_source, fade_in=first,
        )
        if not first:
            n_fade = self.cache_samples
            wav[:, :n_fade] = (
                wav[:, :n_fade] * self.fade_window[:n_fade]
                + state.hift_speech[:, -n_fade:] * self.fade_window[n_fade:]
            )
        if is_final:
            state.done = True
            state.hift_mel = state.hift_source = state.hift_speech = None
            emitted = wav
        else:
            state.hift_mel = mel_in[:, :, -self.cache_frames:]
            state.hift_source = source[:, :, -self.cache_samples:]
            state.hift_speech = wav[:, -self.cache_samples:]
            emitted = wav[:, : -self.cache_samples]
        state.chunks_emitted += 1
        return emitted[0]

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, speech_tokens: torch.Tensor,
        request_id: str = "", ref=None, n_timesteps: int = 10, watermark: bool = True,
        seed: int = 0, is_final: bool = True, **kwargs,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        state = self.request_state(request_id)
        stream: StreamState | None = state.get("stream")
        if stream is None and is_final:
            # the whole utterance in one chunk: the offline reference path
            state.add("stream", StreamState(tokens=speech_tokens, done=True))
            return {AUDIO_CHUNK: [self.synthesize(speech_tokens, ref, n_timesteps, seed, watermark)]}
        if stream is None:
            device = self.get_device()
            stream = StreamState(tokens=torch.empty(0, dtype=torch.long, device=device))
            state.add_all(stream=stream, generator=self._generator(seed, device))
        wav = self.synthesize_chunk(
            stream, speech_tokens, is_final, ref, n_timesteps, state["generator"],
        )
        return {AUDIO_CHUNK: [self._finish(wav, watermark)]}

def audio_seconds(num_samples: int, sample_rate: int) -> float:
    return num_samples / float(sample_rate) if sample_rate else math.nan
