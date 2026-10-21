# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Fish Audio S2-Pro
# ---------------------------------------------------------------------------
#
#   1. ARSubmodule (KV_CACHE engine)
#      - Builds the prefill embeddings: prompt tokens, with a voice clone's
#        reference frames embedded as token + codebook embeddings.
#      - Per step: the Slow AR samples one semantic code (or <|im_end|>),
#        the Fast AR fills codebooks 1-9 by argmax, and the frame's codes and
#        its embedding for the next step leave the same forward. The whole
#        decode step (both loops and sampling) is one CUDA graph.
#   2. CodecSubmodule (STATELESS engine)
#      - Decodes each stream window with the DAC; the window's context covers
#        the decoder's whole receptive field, so the streamed audio equals one
#        decode of the stream.
#   3. RefEncoderSubmodule (voice clone only)
#      - DAC-encodes the reference clip on the GPU in full fp32.
#
# Streaming topology:
#   AR --[codec_tokens, ScheduledLeftContextChunkPolicy((4, 8, 16), 32, 1026)]--> Codec
# ---------------------------------------------------------------------------

from __future__ import annotations

import contextlib
import itertools
import logging
import math
from collections import OrderedDict
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, BatchedAcceleratorGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.fish_s2_pro.components.dac import FishDACDecoder, FishDACEncoder
from mstar.model.fish_s2_pro.components.fast_ar import FishFastAR
from mstar.model.fish_s2_pro.components.slow_ar import FishSlowAR
from mstar.model.fish_s2_pro.config import (
    AR_ATTN,
    AR_KV,
    AR_POS,
    AR_SAMPLER,
    PROMPT_LAYOUT_KEY,
    REF_CLIP_KEY,
    FishS2ProConfig,
)
from mstar.model.submodule_base import (
    EMPTY_INPUT_METADATA,
    ARNodeInputs,
    ARNodeSubmodule,
    InputMetadata,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.streaming.stream_buffer import StreamChunkInfo
from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def full_fp32():
    """Disable TF32 for convolutions and matmuls while the codec runs.

    The reference encodes references in full fp32 (TF32 convolutions flip
    about 3% of the codes), and the codec decodes in fp32 too.
    """
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


# ===========================================================================
# 1. ARSubmodule - one 10-codebook frame per step
# ===========================================================================


class ARSubmodule(ARNodeSubmodule):
    """Slow AR + Fast AR: prompt prefill, then one frame per step.

    The semantic code is sampled from the 4,097-row semantic head (4,096
    codes and ``<|im_end|>``); codebooks 1-9 are the Fast AR's argmax. The
    next step's input is ``(token + sum of the 10 codebook embeddings) /
    sqrt(11)``, computed here and carried on the recurrent edge.
    """

    disable_torch_compile = True
    MAX_BATCH_SIZE = 32
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def __init__(self, slow_ar: FishSlowAR, fast_ar: FishFastAR, config: FishS2ProConfig) -> None:
        super().__init__()
        self.slow_ar = slow_ar
        self.fast_ar = fast_ar
        self.config = config
        self.eos_code = config.eos_code
        self._fast_kv: torch.Tensor | None = None
        self._decode_last_token_indices: torch.Tensor | None = None
        self._flag_stager = PinnedStager(torch.bool, numel=self.MAX_BATCH_SIZE)
        self._index_stager = PinnedStager(torch.long, numel=self.MAX_BATCH_SIZE)

    def _get_fast_kv(self, batch_size: int) -> torch.Tensor:
        """Frame-local Fast AR KV scratch, overwritten every step (stable graph addresses)."""
        if self._fast_kv is None:
            self._fast_kv = torch.zeros(
                self.fast_ar.kv_cache_shape(self.MAX_BATCH_SIZE),
                dtype=self.fast_ar.embeddings.weight.dtype,
                device=self.get_device(),
            )
        return self._fast_kv[:, :batch_size]

    def _get_decode_last_token_indices(self, batch_size: int) -> torch.Tensor:
        if self._decode_last_token_indices is None:
            self._decode_last_token_indices = torch.arange(
                self.MAX_BATCH_SIZE,
                dtype=torch.long,
                device=self.get_device(),
            )
        return self._decode_last_token_indices[:batch_size]

    def reference_embeds(self, codes: torch.Tensor) -> torch.Tensor:
        """``[F, 10]`` reference frames -> their prompt embeddings ``[F, D]``.

        Division by ``sqrt(11)`` here and multiplication by its inverse in
        ``next_input_embeds`` are each the reference's form; in bfloat16 the
        two can differ in the last bit.
        """
        tokens = self.slow_ar.semantic_head[codes[:, 0]]
        vq = self.fast_ar.codebook_embedding_sum(codes).to(tokens.dtype)
        return (tokens + vq) / math.sqrt(self.config.num_codebooks + 1)

    def next_input_embeds(self, head_index: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """The embedding the next step reads for a frame (its token + its codebooks)."""
        tokens = self.slow_ar.semantic_head[head_index]
        vq = self.fast_ar.codebook_embedding_sum(codes).to(tokens.dtype)
        return (tokens + vq) * self.config.vq_scale

    def _build_prefill(
        self, prompt_ids: torch.Tensor, prefix_len: int, ref_frames: int, ref_codes: torch.Tensor | None
    ) -> torch.Tensor:
        """Prompt embeddings; a clone's reference frames sit after ``prefix_len`` tokens."""
        ids = prompt_ids.to(device=self.get_device(), dtype=torch.long).view(-1)
        embeds = self.slow_ar.embed(ids)
        if ref_frames <= 0:
            return embeds
        if ref_codes is None or ref_codes.shape[0] < ref_frames:
            raise ValueError(
                f"Fish voice clone needs {ref_frames} reference frames, got "
                f"{0 if ref_codes is None else ref_codes.shape[0]}"
            )
        reference = self.reference_embeds(ref_codes[:ref_frames].to(device=ids.device, dtype=torch.long))
        return torch.cat([embeds[:prefix_len], reference.to(embeds.dtype), embeds[prefix_len:]])

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> ARNodeInputs:
        del kwargs
        if graph_walk in ("ar_prefill", "ar_prefill_clone"):
            prefix_len, ref_frames = fwd_info.step_metadata[PROMPT_LAYOUT_KEY]
            input_embeds = self._build_prefill(
                inputs["prompt_ids"][0],
                int(prefix_len),
                int(ref_frames),
                inputs["ref_codes"][0] if "ref_codes" in inputs else None,
            )
            self.request_state(fwd_info.rid_handle).add("generated_frames", 0)
        elif graph_walk == "ar_decode":
            input_embeds = inputs["ar_input_embeds"][0].to(device=self.get_device()).reshape(1, -1)
        else:
            raise ValueError(f"Unknown Fish S2-Pro AR walk: {graph_walk!r}")
        # A stop requested from a prefill walk is dropped (``ar_decode_loop`` is
        # not in it), and the codec stream ends at the first ``<|im_end|>``, so
        # the prefill frame and every ``ignore_eos`` frame must not sample it.
        sampling = fwd_info.resource_configs.get(AR_SAMPLER)
        suppress_eos = graph_walk != "ar_decode" or bool(getattr(sampling, "ignore_eos", False))
        return ARNodeInputs(
            input_embeds=input_embeds,
            input_seq_len=input_embeds.shape[0],
            kwargs={"suppress_eos": suppress_eos},
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        del engine_inputs
        device = self.get_device()
        seq_lens = [item.input_seq_len for item in inputs]
        embeds = [item.input_embeds for item in inputs]
        if graph_walk == "ar_decode":
            if any(seq_len != 1 for seq_len in seq_lens):
                raise ValueError("Fish S2-Pro decode expects one token per request")
            packed = embeds[0] if len(embeds) == 1 else torch.cat(embeds, dim=0)
            last_token_indices = self._get_decode_last_token_indices(len(inputs))
        else:
            packed = torch.cat(embeds, dim=0)
            last_token_indices = torch.empty(len(inputs), dtype=torch.long, device=device)
            self._index_stager.copy_(last_token_indices, [end - 1 for end in itertools.accumulate(seq_lens)])
        # Carried per request from ``prepare_inputs``: CUDA-graph replay runs
        # this with padding-slot ids, so request state cannot be read here.
        suppress_eos = torch.empty(len(inputs), dtype=torch.bool, device=device)
        self._flag_stager.copy_(suppress_eos, [item.kwargs["suppress_eos"] for item in inputs])
        return {"input_embeds": packed, "last_token_indices": last_token_indices, "suppress_eos": suppress_eos}

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        del graph_walk, slot_lease, kwargs
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                AR_KV: KVStep(),
                AR_ATTN: AttentionStep(causal=True),
                AR_POS: PositionStep(),
                AR_SAMPLER: SamplerStep(apply_penalty=True),
            },
        )

    def _run_frame(
        self,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_eos: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """One frame for every request: semantic sample, Fast AR codebooks, next input."""
        hidden = self.slow_ar(input_embeds, label="main")
        last_hidden = hidden.index_select(0, last_token_indices)
        logits = self.slow_ar.semantic_logits(last_hidden)
        logits[:, self.eos_code].masked_fill_(suppress_eos, float("-inf"))
        sampler: SamplerResource = engine_inputs.resources[AR_SAMPLER]
        head_index = sampler.sample(logits).reshape(-1)
        is_eos = head_index >= self.eos_code
        semantic = torch.where(is_eos, torch.zeros_like(head_index), head_index)
        codes = self.fast_ar(last_hidden, semantic, self._get_fast_kv(semantic.shape[0]))
        next_embeds = self.next_input_embeds(head_index, codes)
        # The codec stream carries the head index in codebook 0, so the frame
        # that ends the stream (<|im_end|>) is recognizable and dropped.
        stream = codes.clone()
        stream[:, 0] = head_index
        return {"ar_input_embeds": next_embeds, "codec_tokens": stream, "new_token": head_index}

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_eos: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, kwargs
        output = self._run_frame(engine_inputs, input_embeds, last_token_indices, suppress_eos)
        return {
            "ar_input_embeds": [output["ar_input_embeds"]],
            "codec_tokens": [output["codec_tokens"][0]],
            "new_token": [output["new_token"]],
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_eos: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        output = self._run_frame(engine_inputs, input_embeds, last_token_indices, suppress_eos)
        return {
            request_id: {
                "ar_input_embeds": [output["ar_input_embeds"][i : i + 1]],
                "codec_tokens": [output["codec_tokens"][i]],
                "new_token": [output["new_token"][i]],
            }
            for i, request_id in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs: Any,
    ) -> None:
        """Expose the sampled head index for ``check_stop`` without a GPU sync."""
        del request_info, kwargs
        if "new_token" in outputs:
            outputs["head_index"] = outputs.pop("new_token")
        elif "head_index" not in outputs and "codec_tokens" in outputs:
            outputs["head_index"] = [outputs["codec_tokens"][-1].reshape(-1)[0]]

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        """Stop on ``<|im_end|>`` or the request's frame limit.

        Frames are counted here, not in ``postprocess``: a speculative step's
        ``postprocess`` can run before the previous step's stop check, and a
        discarded speculative step never reaches this.
        """
        if "head_index" not in outputs:
            return set()
        head_index = int(outputs["head_index"][0].item())
        state = self.request_state(request_id)
        generated = int(state.get("generated_frames", 0)) + 1
        state.add("generated_frames", generated)
        max_tokens = request_info.max_tokens
        sampling = request_info.resource_configs.get(AR_SAMPLER)
        reached_eos = head_index == self.eos_code and not bool(getattr(sampling, "ignore_eos", False))
        if reached_eos or generated >= max_tokens:
            logger.debug("fish ar %s: stop after %d frames (eos=%s)", request_id, generated, reached_eos)
            return {"ar_decode_loop"}
        return set()

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return (
            batch.graph_walk in {"ar_prefill", "ar_prefill_clone", "ar_decode"}
            and bool(model_inputs)
            and len(model_inputs) <= self.MAX_BATCH_SIZE
        )

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[AcceleratorGraphConfig]:
        """Capture the whole one-token decode step (both ARs and sampling) per batch bucket."""
        del tp_world_size
        dtype = self.slow_ar.semantic_head.dtype
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="ar_decode",
                single_request_inputs=ARNodeInputs(
                    input_embeds=torch.zeros(1, self.config.slow_ar.dim, dtype=dtype, device=device),
                    input_seq_len=1,
                    kwargs={"suppress_eos": False},
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                compile=False,
            )
        ]

    def can_use_accelerator_graphs(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        if batch.graph_walk != "ar_decode" or not self.can_batch(batch, model_inputs):
            return False
        return super().can_use_accelerator_graphs(batch, model_inputs)


# ===========================================================================
# 2. CodecSubmodule - exact streaming DAC decode
# ===========================================================================


class CodecSubmodule(ARNodeSubmodule):
    """Turn streamed frames into PCM chunks that stitch into one whole decode.

    Each window is ``context`` already-decoded frames plus its new frames,
    starting ``start_offset`` frames into the stream. The quantizer's post
    transformer runs over the whole window at the frames' stream positions;
    the conv stack, which costs most, runs over the new frames and its
    ``conv_context`` frames of receptive field only. With the default
    context (the transformer's 1,016 frames plus the conv stack's 10), every
    emitted sample is the sample one decode of the whole stream produces.
    """

    disable_torch_compile = True
    disable_autocast = True
    MAX_BATCH_SIZE = 32

    def __init__(self, decoder: FishDACDecoder, config: FishS2ProConfig, *, exact_fp32: bool = True):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self.codec = config.codec
        self.hop = config.codec.hop_length
        self.exact_fp32 = exact_fp32
        receptive_field = decoder.conv_receptive_field()
        if self.codec.conv_context_frames < receptive_field:
            raise ValueError(
                f"Fish codec conv_context_frames={self.codec.conv_context_frames} is below the conv "
                f"stack's {receptive_field}-frame receptive field"
            )
        self.conv_context = self.codec.conv_context_frames
        self.max_new_frames = max([*self.codec.chunk_schedule, self.codec.chunk_frames])
        self._geometry_stager = PinnedStager(torch.long, numel=2 * self.MAX_BATCH_SIZE)
        if not self.codec.exact_streaming:
            logger.warning(
                "Fish codec left_context_frames=%d is below the decoder's %d-frame receptive field; "
                "streamed audio will differ from a whole decode",
                self.codec.left_context_frames,
                self.codec.transformer_context_frames + self.conv_context,
            )

    def conv_frames(self, width: int) -> int:
        """Frames the conv stack decodes for a batch padded to ``width``."""
        return min(width, self.conv_context + self.max_new_frames)

    @staticmethod
    def _codec_chunk(input_metadata: InputMetadata | None) -> StreamChunkInfo | None:
        return input_metadata.stream_chunks.get("codec_tokens") if input_metadata else None

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        input_metadata: InputMetadata = EMPTY_INPUT_METADATA,
        **kwargs: Any,
    ) -> ARNodeInputs:
        """Drop the end-of-stream frame and lay the window out as ``[10, frames]``."""
        del graph_walk, kwargs
        chunk = self._codec_chunk(input_metadata)
        codes = inputs["codec_tokens"][0].to(device=self.get_device(), dtype=torch.long)
        codes = codes.view(-1, self.config.num_codebooks)
        # A sync: which window holds the end-of-stream frame is not known on the host.
        codes = codes[codes[:, 0] < self.config.eos_code]
        frames = codes.shape[0]
        context = min(chunk.context_items if chunk is not None else 0, frames)
        start_offset = chunk.start_offset if chunk is not None else 0
        conv_start = context - min(context, self.conv_context)
        logger.debug(
            "codec %s: window offset=%d context=%d frames=%d final=%s",
            fwd_info.request_id,
            start_offset,
            context,
            frames,
            chunk is not None and chunk.is_final,
        )
        return ARNodeInputs(
            tensor_inputs={"codec_tokens": codes.t().contiguous()},
            kwargs={"frames": frames, "context": context, "conv_start": conv_start, "start_offset": start_offset},
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        """Stack the batch's windows, zero-padded on the right to the widest.

        A batch whose windows carry no new frames (a stream's context-only
        terminal flush) gets width 0 and decodes nothing.
        """
        del graph_walk, engine_inputs
        windows = [item.tensor_inputs["codec_tokens"] for item in inputs]
        has_new = any(item.kwargs["frames"] > item.kwargs["context"] for item in inputs)
        width = max(window.shape[-1] for window in windows) if has_new else 0
        device = windows[0].device
        codec_tokens = torch.stack(
            [
                torch.nn.functional.pad(window[:, :width], (0, width - min(width, window.shape[-1])))
                for window in windows
            ]
        )

        geometry = torch.empty(2, len(inputs), dtype=torch.long, device=device)
        self._geometry_stager.copy_(
            geometry,
            [item.kwargs["start_offset"] for item in inputs] + [item.kwargs["conv_start"] for item in inputs],
        )
        return {"codec_tokens": codec_tokens, "start_offset": geometry[0], "conv_start": geometry[1]}

    def _decode(self, codec_tokens: torch.Tensor, start_offset: torch.Tensor, conv_start: torch.Tensor) -> torch.Tensor:
        """``[B, 10, W]`` windows -> int16 PCM ``[B, conv_frames * hop]``."""
        width = codec_tokens.shape[-1]
        if width == 0:
            return codec_tokens.new_zeros((codec_tokens.shape[0], 0), dtype=torch.int16)
        guard = full_fp32() if self.exact_fp32 else contextlib.nullcontext()
        with guard:
            wav = self.decoder.decode_window(codec_tokens, start_offset, conv_start, self.conv_frames(width))
        return (wav.clamp(-1, 1) * 32767).to(torch.int16)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        start_offset: torch.Tensor,
        conv_start: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        return {"audio_chunk": [self._decode(codec_tokens, start_offset, conv_start)[0]]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        start_offset: torch.Tensor,
        conv_start: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        wavs = self._decode(codec_tokens, start_offset, conv_start)
        return {request_id: {"audio_chunk": [wavs[i]]} for i, request_id in enumerate(engine_inputs.request_ids)}

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: ARNodeInputs | None = None,
        **kwargs: Any,
    ) -> None:
        """Keep the new frames' samples: ``(context - conv_start) .. (frames - conv_start)`` frames in."""
        del request_info, kwargs
        if "audio_chunk" not in outputs:
            return
        if inputs is None:
            raise ValueError("codec postprocess needs the pass's inputs for its window geometry")
        frames, context = int(inputs.kwargs["frames"]), int(inputs.kwargs["context"])
        conv_start = int(inputs.kwargs["conv_start"])
        audio = outputs["audio_chunk"][0].reshape(-1)
        if frames <= context:
            outputs["audio_chunk"][0] = audio[:0]
            return
        start, end = (context - conv_start) * self.hop, (frames - conv_start) * self.hop
        if audio.numel() < end:
            raise ValueError(
                f"codec produced {audio.numel()} samples for a window of {frames} frames "
                f"({end} expected after frame {conv_start})"
            )
        outputs["audio_chunk"][0] = audio[start:end]
        logger.debug("codec %s: emit frames %d..%d", request_id, context, frames)

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        del batch
        return 0 < len(model_inputs) <= self.MAX_BATCH_SIZE

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[AcceleratorGraphConfig]:
        del device, tp_world_size
        return []

    def can_use_accelerator_graphs(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        del batch, model_inputs
        return False


# ===========================================================================
# 3. RefEncoderSubmodule - reference clip -> 10-codebook frames
# ===========================================================================


class RefEncoderSubmodule(NodeSubmodule):
    """DAC-encode one 44.1 kHz reference clip into ``[frames, 10]`` codes.

    Runs on the GPU under ``full_fp32``: TF32 convolutions would flip some
    of the codes. Codes
    are memoised by clip content: a voice is reused across requests and
    across the sentence chunks of one long request.
    """

    disable_torch_compile = True
    disable_autocast = True
    CACHE_SIZE = 64

    def __init__(self, encoder: FishDACEncoder, config: FishS2ProConfig) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        self._codes: OrderedDict[str, torch.Tensor] = OrderedDict()

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        del graph_walk, kwargs
        waveform = inputs["audio_inputs"][0].to(device=self.get_device(), dtype=torch.float32).reshape(-1)
        clip_key = fwd_info.step_metadata[REF_CLIP_KEY]
        return NodeInputs(tensor_inputs={"waveform": waveform}, kwargs={"clip_key": clip_key})

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), full_fp32():
            codes = self.encoder(waveform.view(1, 1, -1).float())
        return codes[0].t().contiguous().long()

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        waveform: torch.Tensor,
        clip_key: str | None = None,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        if clip_key is not None and clip_key in self._codes:
            self._codes.move_to_end(clip_key)
            return {"ref_codes": [self._codes[clip_key]]}
        codes = self.encode(waveform)
        if clip_key is not None:
            self._codes[clip_key] = codes
            while len(self._codes) > self.CACHE_SIZE:
                self._codes.popitem(last=False)
        return {"ref_codes": [codes]}
