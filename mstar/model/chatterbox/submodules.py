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
        device = self.get_device()
        meta = fwd_info.step_metadata
        sampling = fwd_info.resource_configs.get(T3_SAMPLER)
        temperature = float(getattr(sampling, "temperature", self.config.generation.temperature))
        return {
            "cfg_weight": torch.tensor(
                [float(self._knob(meta, "cfg_weight")) if requires_cfg else 0.0],
                dtype=torch.float32, device=device,
            ),
            "min_p": torch.tensor([float(self._knob(meta, "min_p"))], dtype=torch.float32, device=device),
            "temperature": torch.tensor([temperature], dtype=torch.float32, device=device),
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
        scalars = {
            name: torch.cat([inp.tensor_inputs[name] for inp in inputs]).view(-1, 1)
            for name in ("cfg_weight", "min_p", "temperature")
        }
        return {"input_embeds": embeds, "requires_cfg": requires_cfg, **scalars}

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

    @staticmethod
    def _apply_min_p(
        logits: torch.Tensor, min_p: torch.Tensor, temperature: torch.Tensor,
    ) -> torch.Tensor:
        """HF ``MinPLogitsWarper`` on pre-temperature logits: keep ``z`` with
        ``softmax(z/T) >= min_p * max`` <=> ``z >= max(z) + T*log(min_p)``.
        ``min_p == 0`` keeps everything; greedy (``T == 0``) keeps the argmax."""
        margin = torch.where(
            min_p > 0,
            temperature * torch.log(min_p.clamp_min(1e-30)),
            torch.full_like(min_p, float("-inf")),
        )
        threshold = logits.amax(dim=-1, keepdim=True) + margin
        return logits.masked_fill(logits < threshold, float("-inf"))

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        cfg_weight: torch.Tensor,
        min_p: torch.Tensor,
        temperature: torch.Tensor,
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
        logits = self._apply_min_p(logits, min_p, temperature)
        return sampler.sample(engine_inputs.request_ids, logits)

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
        cfg_weight: torch.Tensor, min_p: torch.Tensor, temperature: torch.Tensor,
        requires_cfg: bool = False, **kwargs,
    ) -> NameToTensorList:
        del kwargs
        tokens = self._forward(
            graph_walk, engine_inputs, input_embeds, cfg_weight, min_p, temperature, requires_cfg,
        )
        return {SPEECH_TOKENS: [tokens.reshape(-1)[:1]]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
        cfg_weight: torch.Tensor, min_p: torch.Tensor, temperature: torch.Tensor,
        requires_cfg: bool = False, **kwargs,
    ) -> dict[str, NameToTensorList]:
        del kwargs
        tokens = self._forward(
            graph_walk, engine_inputs, input_embeds, cfg_weight, min_p, temperature, requires_cfg,
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
                        "min_p": torch.zeros(1, device=device),
                        "temperature": torch.ones(1, device=device),
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


class S3GenSubmodule(NodeSubmodule):
    """Flow-matching mel decoder + HiFT vocoder over one utterance's tokens.

    The reference conditioning (prompt tokens, prompt mel, x-vector) comes from
    the checkpoint's built-in voice or from the request's reference clip,
    computed on first use and cached per voice. The CFM noise and the vocoder's
    source excitation are drawn from a generator seeded with the request seed,
    so a request is reproducible.
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

    @torch.no_grad()
    def condition(self, wav24: torch.Tensor):
        from mstar.model.chatterbox.components.audio_frontend import resample, trim_reference

        wav24 = trim_reference(wav24, self.config.sample_rate, self.config.dec_cond_seconds)
        wav16 = resample(wav24, self.config.sample_rate, S3_SR)
        tokens, lens = self.s3_tokenizer([wav16])
        return self.s3gen.embed_reference(wav24[None], wav16[None], tokens[:, : int(lens[0])])

    def _reference(self, inputs: NameToTensorList):
        if REF_AUDIO not in inputs:
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

    def prepare_inputs(
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, **kwargs,
    ) -> NodeInputs:
        del graph_walk, kwargs
        device = self.get_device()
        chunk = inputs.get(SPEECH_TOKENS, [None])[0]
        tokens = (
            torch.empty(0, dtype=torch.long, device=device)
            if chunk is None
            else chunk.to(device, torch.long).reshape(-1)
        )
        # BOS/EOS and any other control id are not speech (reference
        # ``drop_invalid_tokens`` + ``< 6561`` filter)
        tokens = tokens[tokens < self.config.s3gen.vocab_size]
        if self.config.trailing_silence_tokens and tokens.numel() > 0:
            silence = torch.full(
                (self.config.trailing_silence_tokens,), S3GEN_SILENCE_TOKEN,
                dtype=torch.long, device=device,
            )
            tokens = torch.cat([tokens, silence])
        meta = fwd_info.step_metadata
        return NodeInputs(
            tensor_inputs={SPEECH_TOKENS: tokens},
            kwargs={
                "ref": self._reference(inputs),
                "n_timesteps": int(meta.get("n_cfm_timesteps", self.config.generation.n_cfm_timesteps)),
                "watermark": bool(meta.get("watermark", self.config.generation.watermark)),
                "seed": int(fwd_info.random_seed),
            },
        )

    @torch.no_grad()
    def synthesize(
        self, tokens: torch.Tensor, ref, n_timesteps: int, seed: int, watermark: bool,
    ) -> torch.Tensor:
        """PCM16 for one utterance's speech tokens."""
        if tokens.numel() == 0:
            return torch.zeros(0, dtype=torch.int16, device=tokens.device)
        generator = torch.Generator(device=tokens.device).manual_seed(seed)
        lens = torch.tensor([tokens.numel()], dtype=torch.long, device=tokens.device)
        mel = self.s3gen.tokens_to_mel(
            tokens[None], lens, ref, n_timesteps=n_timesteps, generator=generator,
        )
        wav = self.s3gen.mel_to_wav(mel, generator=generator)[0]
        if watermark and self.watermarker is not None:
            wav = self.watermarker.apply(wav, self.config.sample_rate)
        return (wav.clamp(-1, 1) * 32767).to(torch.int16)

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, speech_tokens: torch.Tensor,
        ref=None, n_timesteps: int = 10, watermark: bool = True, seed: int = 0, **kwargs,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        return {AUDIO_CHUNK: [self.synthesize(speech_tokens, ref, n_timesteps, seed, watermark)]}


def audio_seconds(num_samples: int, sample_rate: int) -> float:
    return num_samples / float(sample_rate) if sample_rate else math.nan
