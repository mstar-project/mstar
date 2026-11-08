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

import functools
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
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.state import (
    ArrayStep,
    SampleStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    at_last_token,
    host,
    positions,
    sample,
)
from mstar.model.chatterbox.components.s3_tokenizer import AudioEncoderV2
from mstar.model.chatterbox.components.s3gen import FlowRow, ReferenceConditioning
from mstar.model.chatterbox.components.s3gen_hift import HiFTNoise
from mstar.model.chatterbox.config import (
    COND_LABEL,
    S3_SR,
    S3GEN_SILENCE_TOKEN,
    S3GEN_STREAM,
    T3_KV,
    T3_SAMPLER,
    UNCOND_LABEL,
    ChatterboxConfig,
)
from mstar.model.submodule_base import (
    EMPTY_INPUT_METADATA,
    ARNodeInputs,
    ARNodeSubmodule,
    InputMetadata,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    TokenStop,
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


def _rid(fwd_info: CurrentForwardPassInfo) -> int:
    """The worker's handle for the request; the engine keys outputs and
    cleanup by it, so per-request state must be too."""
    if fwd_info.rid_handle == -1:
        raise ValueError(f"request {fwd_info.request_id!r} reached the worker without a rid handle")
    return fwd_info.rid_handle


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


def reference_tokens(samples: int, config: ChatterboxConfig) -> int:
    """How many S3 prompt tokens `VoiceEncoderSubmodule.condition` makes of a ``samples``-long
    reference clip. It asserts the same count."""
    wav16 = min(-(-samples * S3_SR // config.sample_rate), round(config.enc_cond_seconds * S3_SR))
    frames = min(wav16 // config.s3_tokenizer.hop_size, 4 * config.t3.speech_cond_prompt_len)
    return AudioEncoderV2._conv_out_len(AudioEncoderV2._conv_out_len(frames, 2), 2)


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
        count = int(lens[0])
        assert count == reference_tokens(wav24.numel(), self.config), (
            f"the S3 tokenizer made {count} prompt tokens of a {wav24.numel()}-sample clip, "
            "not the count `reference_tokens` gives a layout to size it by"
        )
        return speaker_emb, tokens[0, :count]

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


@functools.lru_cache(maxsize=64)
def _guidance_weight(weight: float, device: torch.device) -> torch.Tensor:
    """``weight`` as a ``[1]`` tensor on ``device``, made once, so a decode step copies nothing to the device."""
    return torch.tensor([weight], dtype=torch.float32, device=device)


@dataclass
class BuiltinT3Voice:
    """The T3 half of the checkpoint's ``conds.pt``."""

    speaker_emb: torch.Tensor   # [256]
    prompt_tokens: torch.Tensor  # [P]


class T3Submodule(ARNodeSubmodule):
    """One T3 step for a continuous batch.

    Prefill packs ``[cond | text | BOS]`` per request; decode embeds the
    previous speech token at its learned speech position. With guidance each
    request owns two KV streams, ``main`` and ``uncond``, that the step lists
    label-major, so they plan as one; the forward then sees ``2B``
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
    # prefill captures: total tokens of the batch (conditioning + text + BOS,
    # about 170-300 per request; Turbo's longer voice prompt about 400-550),
    # for up to eight requests at a time (a fast S3Gen lets the scheduler admit
    # that many together at concurrency 8-32); bigger admissions run eagerly
    PREFILL_TOKEN_BUCKETS = [256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]

    def __init__(
        self, model: nn.Module, config: ChatterboxConfig, builtin_voice: BuiltinT3Voice | None,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.t3 = config.t3
        self.builtin_voice = builtin_voice
        self.supports_cfg = not config.is_turbo
        self.token_stop = TokenStop(
            loop=DECODE_LOOP, buffer=SPEECH_TOKENS, stop_ids=frozenset({config.t3.stop_speech_token}),
            sampler=T3_SAMPLER, prefill_walks=frozenset(PREFILL_WALKS),
        )

    # -- request knobs -------------------------------------------------------

    def _knob(self, step_metadata: Mapping[str, Any] | None, key: str):
        step_metadata = step_metadata or {}
        return step_metadata.get(key, getattr(self.config.generation, key))

    def _requires_cfg(self, step_metadata: Mapping[str, Any] | None) -> bool:
        return self.supports_cfg and float(self._knob(step_metadata, "cfg_weight")) > 0.0

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv = handles[T3_KV]
        self.sampler = handles[T3_SAMPLER]

    def cg_key_info(
        self, graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
        per_request_input_metadata=None,
        **kwargs,
    ):
        """Guidance on or off; a mixed batch matches no capture and runs eagerly."""
        del graph_walk, per_request_input_metadata, kwargs
        flags = {self._requires_cfg(info.step_metadata) for info in per_request_info.values()}
        return flags.pop() if len(flags) == 1 else None

    def split_batches_by_capture_key(self, graph_walk: str) -> bool:
        """A mixed batch runs eager one request at a time, so batch each guidance mode apart."""
        del graph_walk
        return self.supports_cfg

    def _cfg_weight(self, fwd_info: CurrentForwardPassInfo, requires_cfg: bool) -> torch.Tensor:
        # sampling knobs (temperature, min_p, penalty) live in the request's
        # SampleSettings and are applied by the sampler
        weight = float(self._knob(fwd_info.step_metadata, "cfg_weight")) if requires_cfg else 0.0
        return _guidance_weight(weight, self.get_device())

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
        self, graph_walk: str, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList,
        input_metadata: InputMetadata = EMPTY_INPUT_METADATA, **kwargs,
    ) -> ARNodeInputs:
        del kwargs
        device = self.get_device()
        requires_cfg = self._requires_cfg(fwd_info.step_metadata)
        if graph_walk in PREFILL_WALKS:
            text_ids = inputs[TEXT_INPUTS][0].to(device, torch.long).reshape(-1)
            speaker_emb, prompt_tokens = self._voice(inputs)
            emotion = None
            if self.t3.emotion_adv:
                emotion = torch.tensor(
                    [float(self._knob(fwd_info.step_metadata, "exaggeration"))], device=device,
                )
            cond = self.model.conditioning(speaker_emb[None], prompt_tokens[None], emotion)[0]
            embeds = self.model.build_prefill_embeds(cond, text_ids, uncond=False)
            tensor_inputs = {"cfg_weight": self._cfg_weight(fwd_info, requires_cfg)}
            if requires_cfg:
                tensor_inputs["uncond_embeds"] = self.model.build_prefill_embeds(
                    cond, text_ids, uncond=True,
                )
        elif graph_walk == "decode":
            # The input token's speech position is how many tokens were
            # sampled: the prefill's, and one per decode step before this one.
            # It reads the step's own iteration, so a step prepared again after
            # a refused admit keeps it.
            step = 1 + input_metadata.dynamic_loop_iter_counts.get(DECODE_LOOP, 0)
            if step >= self.t3.speech_pos_table_size:
                raise ValueError(
                    f"speech position {step} is past the T3 table ({self.t3.speech_pos_table_size})"
                )
            token = inputs[PREV_TOKEN][0].to(device, torch.long).reshape(-1)
            embeds = self.model.embed_speech(token, step)
            tensor_inputs = {"cfg_weight": self._cfg_weight(fwd_info, requires_cfg)}
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
        # label-major, so the two streams plan as one
        segments = [
            Segment(request_id=rid, label=label, span=inp.input_seq_len)
            for label in labels
            for rid, inp in zip(request_ids, inputs, strict=True)
        ]
        tracked = {}
        if graph_walk in PREFILL_WALKS:
            # the reference penalises repeats of everything after the text,
            # BOS included
            bos = torch.tensor([self.t3.start_speech_token], dtype=torch.long)
            tracked = {rid: bos for rid in request_ids}
        return SubmoduleStep(
            segments=segments,
            state={T3_KV: ArrayStep(), T3_SAMPLER: SampleStep(apply_penalty=True, prompt_tokens=tracked)},
            cg_key_info=requires_cfg,
        )

    # -- compute ----------------------------------------------------------------

    def _forward(
        self, graph_walk: str, input_embeds: torch.Tensor, cfg_weight: torch.Tensor, requires_cfg: bool,
    ) -> torch.Tensor:
        position_ids = None
        if self.t3.backbone.is_gpt2:
            position_ids = positions(self.kv)[: input_embeds.shape[0]]
        hidden = self.model.hidden(input_embeds, position_ids=position_ids)
        if graph_walk in PREFILL_WALKS:
            # every request's main row, then every request's uncond row
            hidden = at_last_token(self.kv, hidden)
        logits = self.model.logits(hidden).float()
        if requires_cfg:
            cond, uncond = logits.chunk(2, dim=0)
            logits = cond + cfg_weight * (cond - uncond)
        # penalty -> temperature -> min_p -> top-p happen in the sampler, in
        # the reference's order
        return sample(self.sampler, logits)

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
        cfg_weight: torch.Tensor, requires_cfg: bool = False, **kwargs,
    ) -> NameToTensorList:
        del engine_inputs, kwargs
        tokens = self._forward(graph_walk, input_embeds, cfg_weight, requires_cfg)
        return {SPEECH_TOKENS: [tokens.reshape(-1)[:1]]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, input_embeds: torch.Tensor,
        cfg_weight: torch.Tensor, requires_cfg: bool = False, **kwargs,
    ) -> dict[str, NameToTensorList]:
        del kwargs
        tokens = self._forward(graph_walk, input_embeds, cfg_weight, requires_cfg).reshape(-1)
        return {
            rid: {SPEECH_TOKENS: [tokens[i : i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    # -- loop control ---------------------------------------------------------

    def postprocess(
        self, request_id: str, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]], **kwargs,
    ) -> None:
        del request_id, request_info, kwargs
        if SPEECH_TOKENS not in outputs:
            return
        # the sampled token is the next decode input; no value is read here
        outputs[PREV_TOKEN] = outputs[SPEECH_TOKENS]

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

    def _prefill_capture_input(
        self, requires_cfg: bool, device: torch.device, dtype: torch.dtype, n: int,
    ) -> ARNodeInputs:
        """A stand-in prefill of ``n`` tokens with the shape ``prepare_inputs`` yields."""
        tensor_inputs = {"cfg_weight": torch.full((1,), 0.5 if requires_cfg else 0.0, device=device)}
        if requires_cfg:
            tensor_inputs["uncond_embeds"] = torch.zeros(n, self.t3.hidden_size, dtype=dtype, device=device)
        return ARNodeInputs(
            input_embeds=torch.zeros(n, self.t3.hidden_size, dtype=dtype, device=device),
            input_seq_len=n,
            tensor_inputs=tensor_inputs,
            resource_step_info=requires_cfg,
        )

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        del tp_world_size
        dtype = self.model.speech_emb.weight.dtype
        configs = []
        for requires_cfg in ((True, False) if self.supports_cfg else (False,)):
            if self.config.t3_prefill_graphs:
                # both prefill walks replay the same captures; the packed
                # layout pads the batch's tokens up to the bucket
                configs.append(PackedAcceleratorGraphConfig(
                    capture_graph_walk="prefill",
                    replay_graph_walks=list(PREFILL_WALKS),
                    capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                    make_node_input=functools.partial(self._prefill_capture_input, requires_cfg, device, dtype),
                    additional_key_info=requires_cfg,
                    capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
                    caps_eager_batch_size=False,
                    compile=False,
                    # guidance packs the cond and uncond streams into one plan
                    total_tokens_multiplier=2 if requires_cfg else 1,
                ))
            configs.append(BatchedAcceleratorGraphConfig(
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

    def can_use_accelerator_graphs(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        if not self.can_batch(batch, model_inputs):
            return False
        if batch.graph_walk in PREFILL_WALKS and not self.config.t3_prefill_graphs:
            return False
        return super().can_use_accelerator_graphs(batch, model_inputs)


# ===========================================================================
# 3. S3Gen: speech tokens -> waveform
# ===========================================================================


def max_reference_tokens(config: ChatterboxConfig) -> int:
    """The most S3 prompt tokens `S3GenSubmodule.condition` keeps of a reference clip: those of its first
    ``dec_cond_seconds``, cut to the 24 kHz mel's frames over ``token_mel_ratio`` as `S3Gen.embed_reference` cuts
    them."""
    wav24 = round(config.dec_cond_seconds * config.sample_rate)
    wav16 = -(-wav24 * S3_SR // config.sample_rate)
    tokens = AudioEncoderV2._conv_out_len(AudioEncoderV2._conv_out_len(wav16 // config.s3_tokenizer.hop_size, 2), 2)
    return min(tokens, wav24 // config.s3gen.mel.hop_size // config.s3gen.token_mel_ratio)


@dataclass
class _ChunkPlan:
    """What one request needs from the current step, planned with every draw
    it takes before the step: a flow solve over ``tokens``, the held-back tail
    to release, or nothing yet. ``host`` is the stream's host fields once the
    step commits."""

    host: dict[str, Any]
    is_final: bool
    token_offset: int  # tokens vocoded before this chunk
    first: bool  # no chunk vocoded before this one
    window_start: int = 0  # first token of the flow solve (0 = whole history)
    tokens: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    vocoder_noise: HiFTNoise | None = None
    release: bool = False


class S3GenSubmodule(NodeSubmodule):
    """Flow-matching mel decoder + HiFT vocoder over a stream of speech tokens.

    A request whose whole token sequence arrives in one final chunk (the
    offline policy) is synthesised exactly as the reference does it. A
    streamed request follows CosyVoice 2's token2wav: every chunk re-runs the
    flow decoder over all tokens so far (fixed noise field, look-ahead tokens
    withheld until the end), vocodes only the new frames behind a small mel
    cache, continues the harmonic source from the previous chunk and
    crossfades the re-synthesised tail before emitting.

    Every chunk brings the whole token history. Between chunks the stream's
    vocoder tail and reference live in its ``S3GEN_STREAM`` Value, and its
    progress and generator state in that Value's host fields.

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
        self.context_tokens = config.stream_context_tokens
        self.frame_bucket = config.s3gen_frame_bucket
        self.max_rows = int(config.s3gen_max_batch_size) if config.s3gen_max_batch_size else self.MAX_BATCH_SIZE
        self.cache_frames = config.stream_mel_cache_frames
        self.cache_samples = self.cache_frames * self.samples_per_frame
        self.max_reference_tokens = max_reference_tokens(config)
        # crossfade of the re-synthesised tail: the second half of a Hamming
        # window fades the old tail out while the first half fades the new in
        self.register_buffer(
            "fade_window", torch.hamming_window(2 * self.cache_samples, periodic=False), persistent=False,
        )

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.stream = handles[S3GEN_STREAM]

    # -- reference conditioning ---------------------------------------------

    @torch.no_grad()
    def condition(self, wav24: torch.Tensor):
        from mstar.model.chatterbox.components.audio_frontend import resample, trim_reference

        wav24 = trim_reference(wav24, self.config.sample_rate, self.config.dec_cond_seconds)
        wav16 = resample(wav24, self.config.sample_rate, S3_SR)
        tokens, lens = self.s3_tokenizer([wav16])
        return self.s3gen.embed_reference(wav24[None], wav16[None], tokens[:, : int(lens[0])])

    def _reference(self, inputs: NameToTensorList):
        """The reference conditioning of the clip that rides with a stream's
        first chunk, computed or fetched from the per-voice cache; None for a
        chunk without the clip."""
        if not inputs.get(REF_AUDIO):
            return None
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
        # Only the engine's final-chunk flag ends the stream: a stop token does
        # not, or ``ignore_eos`` decoding past it would never be vocoded
        is_final = bool(kwargs.get("is_final_stream_chunk", False))
        # BOS/EOS and any other control id are not speech (reference
        # ``drop_invalid_tokens`` + ``< 6561`` filter)
        tokens = raw[raw < self.config.s3gen.vocab_size]
        meta = fwd_info.step_metadata
        rid = _rid(fwd_info)
        # what the last committed step left, so a row prepared again after a refused admit draws the same noise
        held = host(self.stream, rid, "main", 0)
        clip = ref = self._reference(inputs)
        if clip is not None and clip.num_prompt_tokens > self.max_reference_tokens:
            raise ValueError(
                f"the reference has {clip.num_prompt_tokens} S3 tokens, past the {self.max_reference_tokens} its "
                f"stream holds (dec_cond_seconds={self.config.dec_cond_seconds})"
            )
        if ref is None and not held["ref_len"]:
            if self.builtin_voice is None:
                raise ValueError("No reference voice given and the checkpoint ships no built-in voice")
            ref = self.builtin_voice
        prompt_tokens = held["ref_len"] if ref is None else ref.num_prompt_tokens
        plan = self._plan_chunk(held, tokens, is_final, prompt_tokens, int(fwd_info.random_seed))
        if clip is not None:
            plan.host["ref_len"] = prompt_tokens
        return NodeInputs(kwargs={
            "rid": rid,
            "plan": plan,
            # None: the stream's own, which its Value holds
            "ref": ref,
            "n_timesteps": int(meta.get("n_cfm_timesteps", self.config.generation.n_cfm_timesteps)),
            "watermark": bool(meta.get("watermark", self.config.generation.watermark)),
        })

    # -- synthesis ----------------------------------------------------------------

    MAX_BATCH_SIZE = 8  # default for ``s3gen_max_batch_size``: requests per padded flow solve

    def _generator(self, seed: int, device) -> torch.Generator:
        return torch.Generator(device=device).manual_seed(seed)

    def _with_trailing_silence(self, tokens: torch.Tensor) -> torch.Tensor:
        n = self.config.trailing_silence_tokens
        if n <= 0 or tokens.numel() == 0:
            return tokens
        silence = torch.full((n,), S3GEN_SILENCE_TOKEN, dtype=torch.long, device=tokens.device)
        return torch.cat([tokens, silence])

    def _finish(self, wav: torch.Tensor, watermark: bool) -> torch.Tensor:
        if watermark and self.watermarker is not None and wav.numel() > 0:
            wav = self.watermarker.apply(wav, self.config.sample_rate)
        return (wav.float().clamp(-1, 1) * 32767).to(torch.int16)

    def _noise_field(self, prompt_frames: int, generator: torch.Generator) -> torch.Tensor:
        max_tokens = self.config.t3.max_speech_tokens + self.config.trailing_silence_tokens
        frames = prompt_frames + max_tokens * self.frames_per_token
        return torch.randn(
            (1, self.config.s3gen.output_size, frames),
            dtype=self.s3gen.dtype, device=self.get_device(), generator=generator,
        )

    def _plan_chunk(
        self, held: Mapping[str, Any], tokens: torch.Tensor, is_final: bool, prompt_tokens: int, seed: int,
    ) -> _ChunkPlan:
        """Take a request's token history and decide what this step owes it: a
        flow row to solve, or audio that can go out right away (nothing new to
        decode yet, or the held-back tail once the stream has ended). ``held``
        is the stream's host fields, and the plan draws everything the chunk
        takes from the generator they hold."""
        plan = _ChunkPlan(
            host=dict(held), is_final=is_final, token_offset=held["token_offset"], first=not held["chunks"],
        )
        if held["done"]:
            return plan
        if is_final:
            tokens = self._with_trailing_silence(tokens)
        n = tokens.numel()
        usable = n if is_final else max(n - self.lookahead_tokens, 0)
        if usable <= plan.token_offset:
            # not enough new tokens yet, or nothing new to decode at the end:
            # release the held-back tail as it is
            plan.release = plan.host["done"] = is_final
            return plan
        device = self.get_device()
        generator = self._generator(seed, device)
        if held["rng"]:
            generator.set_state(torch.frombuffer(bytearray(held["rng"]), dtype=torch.uint8))
        prompt_frames = prompt_tokens * self.frames_per_token
        # A whole utterance arriving at once is the reference's offline path:
        # noise drawn on the spot from the request's generator (bit-exact with
        # the package). A stream instead denoises one fixed field every chunk.
        if is_final and not plan.token_offset:
            plan.tokens = tokens
            plan.noise = self.s3gen.draw_noise(1, prompt_frames, n * self.frames_per_token, generator)
        else:
            # The field is the first draw of the request's generator, so a
            # chunk after the stream's first solve draws it again from the seed.
            noise = self._noise_field(prompt_frames, self._generator(seed, device) if held["rng"] else generator)
            # A bounded window keeps only the last ``context_tokens`` settled
            # tokens (plus the prompt) in the solve; the noise field is sliced by
            # absolute frame so every frame keeps the draw it was started with.
            start = 0
            if self.context_tokens > 0:
                start = max(plan.token_offset - self.context_tokens, 0)
            if start > 0:
                ratio = self.frames_per_token
                gen = noise[:, :, prompt_frames + start * ratio : prompt_frames + n * ratio]
                noise = torch.cat([noise[:, :, :prompt_frames], gen], dim=2)
            plan.window_start = start
            plan.tokens = tokens[start:]
            plan.noise = noise
        # the vocoder runs over the new frames behind the held-back ones
        frames = (usable - plan.token_offset) * self.frames_per_token + (0 if plan.first else self.cache_frames)
        plan.vocoder_noise = self.s3gen.vocoder.m_source.draw_noise(
            1, frames * self.samples_per_frame, self.s3gen.dtype, device, generator,
        )
        plan.host.update(
            token_offset=usable, chunks=held["chunks"] + 1, done=is_final,
            rng=generator.get_state().numpy().tobytes(),
        )
        return plan

    def _emit_chunk(self, plan: _ChunkPlan, mel: torch.Tensor, held: dict[str, torch.Tensor], i: int) -> torch.Tensor:
        """Vocode the frames the flow solve added, continuing the vocoder's
        held-back tail in row ``i`` of ``held``, and return the waveform to emit now."""
        new_mel = mel[:, :, (plan.token_offset - plan.window_start) * self.frames_per_token:]
        mel_in = new_mel if plan.first else torch.cat([held["hift_mel"][i : i + 1], new_mel], dim=2)
        wav, source = self.s3gen.vocode(
            mel_in, noise=plan.vocoder_noise, cache_source=None if plan.first else held["hift_source"][i : i + 1],
            fade_in=plan.first,
        )
        if not plan.first:
            n_fade = self.cache_samples
            wav[:, :n_fade] = (
                wav[:, :n_fade] * self.fade_window[:n_fade]
                + held["hift_speech"][i : i + 1, -n_fade:] * self.fade_window[n_fade:]
            )
        if plan.is_final:
            return wav[0]
        held["hift_mel"][i] = mel_in[0, :, -self.cache_frames:]
        held["hift_source"][i] = source[0, :, -self.cache_samples:]
        held["hift_speech"][i] = wav[0, -self.cache_samples:]
        return wav[0, : -self.cache_samples]

    @torch.no_grad()
    def _solve(
        self, plans: list[_ChunkPlan], refs: list, n_timesteps: list[int], held: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        """Run every plan that needs the flow decoder, batched by step count,
        and return each plan's waveform (float). ``held`` is the step's rows of
        the streams' Value, which their references and vocoder tails are read
        from and written to."""
        outputs: list[torch.Tensor] = []
        rows: list[FlowRow | None] = []
        for i, (plan, ref) in enumerate(zip(plans, refs, strict=True)):
            n = plan.host["ref_len"]
            if ref is None:
                ref = ReferenceConditioning(
                    prompt_tokens=held["ref_tokens"][i : i + 1, :n],
                    prompt_feat=held["ref_feat"][i : i + 1, : n * self.frames_per_token],
                    embedding=held["ref_embedding"][i : i + 1],
                )
            elif n:
                held["ref_tokens"][i, :n] = ref.prompt_tokens[0]
                held["ref_feat"][i, : n * self.frames_per_token] = ref.prompt_feat[0]
                held["ref_embedding"][i] = ref.embedding[0]
            released = plan.release and not plan.first
            outputs.append(held["hift_speech"][i] if released else torch.zeros(0, device=self.get_device()))
            rows.append(None if plan.tokens is None else FlowRow(
                tokens=plan.tokens, ref=ref, finalize=plan.is_final, noise=plan.noise,
            ))
        pending = [i for i, row in enumerate(rows) if row is not None]
        for steps in sorted({n_timesteps[i] for i in pending}):
            group = [i for i in pending if n_timesteps[i] == steps]
            mels = self.s3gen.tokens_to_mel_rows(
                [rows[i] for i in group], n_timesteps=steps, frame_bucket=self.frame_bucket,
            )
            for i, mel in zip(group, mels, strict=True):
                outputs[i] = self._emit_chunk(plans[i], mel, held, i)
        return outputs

    # -- engine entry points ---------------------------------------------------

    def can_batch(self, batch, model_inputs: list[NodeInputs]) -> bool:
        del batch
        return 1 < len(model_inputs) <= self.max_rows

    def max_batch_size(self, graph_walk: str) -> int:
        # the scheduler sizes the node's batches from this (the base default is
        # one request), so without it the padded multi-request solve never runs
        del graph_walk
        return self.max_rows

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        del graph_walk, slot_lease, piecewise_leases, kwargs
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
            state={S3GEN_STREAM: ArrayStep(host={
                (rid, "main", 0): inp.kwargs["plan"].host for rid, inp in zip(request_ids, inputs, strict=True)
            })},
        )

    def preprocess(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs]):
        del graph_walk, engine_inputs
        if len(inputs) == 1:
            return dict(inputs[0].kwargs)
        return {key: [inp.kwargs[key] for inp in inputs] for key in inputs[0].kwargs}

    def forward(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rid: int, plan: _ChunkPlan, ref,
        n_timesteps: int, watermark: bool, **kwargs,
    ) -> NameToTensorList:
        del kwargs
        return self.forward_batched(graph_walk, engine_inputs, [rid], [plan], [ref], [n_timesteps], [watermark])[rid]

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rid: list[int], plan: list[_ChunkPlan],
        ref: list, n_timesteps: list[int], watermark: list[bool], **kwargs,
    ) -> dict[int, NameToTensorList]:
        """Several requests' chunks in one step: their flow solves share a
        padded batch, the vocoder runs per request behind its own cache.
        Keyed by rid handle, which is how the engine reads the outputs back."""
        del graph_walk, engine_inputs, kwargs
        held = dict(self.stream.read(0).fields)
        wavs = self._solve(plan, ref, list(n_timesteps), held)
        self.stream.write(0, **held)
        return {
            handle: {AUDIO_CHUNK: [self._finish(wav, wm)]}
            for handle, wav, wm in zip(rid, wavs, watermark, strict=True)
        }


def audio_seconds(num_samples: int, sample_rate: int) -> float:
    return num_samples / float(sample_rate) if sample_rate else math.nan
