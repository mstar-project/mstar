# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Qwen3-Omni
# ---------------------------------------------------------------------------
#
# Five submodules covering the full Thinker-Talker dual-AR pipeline:
#   1. AudioEncoderSubmodule   (enc_dec engine)
#   2. VisionEncoderSubmodule  (enc_dec engine)
#   3. ThinkerSubmodule        (ar engine -- 3D MRoPE, MoE, layer captures)
#   4. TalkerSubmodule         (ar engine -- streaming decode, Code Predictor)
#   5. Code2WavSubmodule       (audio_codec engine)
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.engine.cuda_graph_config import FlashInferPackedCudaGraphConfig
from mstar.engine.cuda_graph_runner import BasicBatchedCudaGraphConfig
from mstar.engine.kv_store import PositionInfo
from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav
from mstar.model.qwen3_omni.components.rope import (
    compute_3d_cos_sin,
    compute_rope_freqs,
    get_rope_index_audio,
    get_rope_index_text,
    get_rope_index_vision,
)
from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor, Qwen3OmniTalkerModel
from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.utils.sampling import CudaGraphableSampler, SeenTokenMask

logger = logging.getLogger(__name__)

# MSTAR_PREP_DEVICE_POS (default OFF): build the per-step thinker_decode MRoPE
# pos_ids without a pageable H2D + cudaStreamSynchronize. The prior
# `torch.tensor([[start_pos]*3], device=cuda)` was a measurable fraction of
# bs=1 wall (same class as the sampler cfg cache). Boot-static.
_PREP_DEVICE_POS = os.environ.get(
    "MSTAR_PREP_DEVICE_POS", "0"
).strip().lower() in ("1", "true", "yes", "on")
# MSTAR_PREP_DEVICE_POS_BATCHED (default OFF): same fix, but for the B>1
# thinker_decode path (`prepare_inputs_batched`). The B1 flag above only patches
# the per-request `prepare_inputs`; `prepare_inputs_batched` — which serves the
# ENTIRE throughput regime (B2..B32, incl. the cells we lose to vLLM 0.22) — still
# built pos_ids via `torch.tensor(starts).to(cuda, non_blocking=True)` from a
# PAGEABLE source, so non_blocking is silently ignored -> a blocking H2D +
# implicit sync on the decode thread every step. This flag replaces it with a
# reusable PINNED host buffer + a genuinely async H2D. Input-build only,
# capture-safe (the value copied into the captured static input is identical).
_PREP_DEVICE_POS_BATCHED = os.environ.get(
    "MSTAR_PREP_DEVICE_POS_BATCHED", "0"
).strip().lower() in ("1", "true", "yes", "on")
_PREP_POS_HITS = [0]  # prep_pos_device_hits (mechanism-alive counter)


def _refresh_prep_flags() -> None:
    """Re-read the flags at runtime. Safe to flip mid-run —
    they only change how the pos_ids INPUT tensor is built (pinned async vs
    pageable torch.tensor), nothing baked into the CUDA-graph capture; the value
    fed to the graph is byte-identical either way. Enables a clean single-boot
    A/B of the two paths without the warm-in/ordering noise a cross-boot A/B
    would carry."""
    global _PREP_DEVICE_POS, _PREP_DEVICE_POS_BATCHED
    _PREP_DEVICE_POS = os.environ.get(
        "MSTAR_PREP_DEVICE_POS", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    _PREP_DEVICE_POS_BATCHED = os.environ.get(
        "MSTAR_PREP_DEVICE_POS_BATCHED", "0"
    ).strip().lower() in ("1", "true", "yes", "on")


def _prep_pos_count() -> None:
    _PREP_POS_HITS[0] += 1
    if _PREP_POS_HITS[0] % 2000 == 0:
        logger.warning("MSTAR_PREP_DEVICE_POS prep_pos_device_hits=%d", _PREP_POS_HITS[0])


@dataclass
class _VisionPrefillStage:
    """Staged full-span vision Thinker prefill, computed once on chunk 0 and
    sliced per chunk (MSTAR_CHUNKED_PREFILL_V2_VISION).

    ``wrapped_embeds`` (total_len, hidden) = vision_bos + vision tokens +
    vision_eos. ``pos_ids`` (3, total_len) are the absolute 3D MRoPE positions
    computed from the ORIGINAL ``start_pos`` (never recomputed from an advanced
    start, so intra-block positions stay exact under chunking). ``deepstack`` is
    the per-layer (total_len, hidden) scatter (sentinels zeroed). ``mm_mask``
    (total_len,) marks vision (True) vs sentinel (False). ``total_len`` = span,
    ``mrope_pos_advance`` = full 3D-grid span jump (> total_len), ``start_pos`` =
    the request's position_id_start at walk entry (for the assert)."""
    wrapped_embeds: torch.Tensor
    pos_ids: torch.Tensor
    deepstack: list[torch.Tensor]
    mm_mask: torch.Tensor
    total_len: int
    mrope_pos_advance: int
    start_pos: float


@dataclass
class _AudioPrefillStage:
    """Staged full-span audio Thinker prefill, computed once by
    ``_build_audio_full`` (MSTAR_MERGED_PREFILL_AUDIO).

    ``wrapped_embeds`` (total_len, hidden) = audio_bos + audio tokens +
    audio_eos. ``pos_ids`` (3, total_len) are the absolute 3D MRoPE positions
    from ``start_pos`` (start/end sentinels text-like, audio tokens temporal
    +1/frame with h/w pinned). ``mm_mask`` (total_len,) marks audio (True) vs
    sentinel (False). ``total_len`` = span = audio_len + 2.

    Unlike vision there is NO deepstack and NO custom MRoPE advance: audio
    positions increment by exactly one per token (see get_rope_index_audio), so
    the post-span position lands at ``start_pos + total_len`` — i.e. the walk's
    MRoPE advance equals its ``seq_len``, the default ``advance_seq_lens`` step.
    That is why the merged text+audio walk carries the SAME post-preprocess
    signature as ``prefill_text`` (input_embeds + cos_3d + sin_3d +
    masks_for_talker) and replays on the ``prefill_text`` capture."""
    wrapped_embeds: torch.Tensor
    pos_ids: torch.Tensor
    mm_mask: torch.Tensor
    total_len: int


# ===================================================================
# 1. AudioEncoderSubmodule (enc_dec engine)
# ===================================================================


class AudioEncoderSubmodule(NodeSubmodule):
    """Thin wrapper around the HF Whisper-style audio encoder.

    Extracts mel spectrograms from raw audio inputs, pads for batching,
    and runs the encoder to produce audio embeddings that the Thinker
    will splice into its input sequence.

    Runs once per request (not batched across requests).
    """

    def __init__(self, audio_encoder: nn.Module, config: Qwen3OmniModelConfig):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        # Edge name from graph walk is "audio_features"
        audio_features = inputs["audio_features"][0]
        audio_seqlens = inputs.get("audio_seqlens", [None])[0]

        return NodeInputs(
            tensor_inputs={
                "audio_features": audio_features,
                "audio_seqlens": audio_seqlens,
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        audio_seqlens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> NameToTensorList:
        """Run the audio encoder and return embeddings.

        Returns:
            {"audio_embeds": [tensor of shape (audio_tokens, hidden_size)]}
        """
        logger.debug(
            "Running AudioEncoder with audio_features shape=%s",
            audio_features.shape,
        )
        audio_embeds = self.audio_encoder(
            audio_features,
            feature_lens=audio_seqlens,
            return_dict=True,
        ).last_hidden_state

        # Flatten to (num_audio_tokens, hidden_size) if needed
        if audio_embeds.dim() == 3:
            audio_embeds = audio_embeds.squeeze(0)

        return {"audio_embeds": [audio_embeds]}


class NativeAudioEncoderSubmodule(NodeSubmodule):
    """Native AuT submodule with cross-request batching.

    The audio encoder is varlen-packed (per-request windows via ``cu_seqlens``),
    so batching N requests needs no padding: concatenate their mel features along
    time, pass a multi-entry ``feature_lens``, run one forward, and slice the
    packed output back per request. Mirrors the Code2Wav preprocess/forward_batched
    contract. Batched-no-graph already beats the HF baseline, so no torch.compile /
    CUDA graphs are declared (the issue warns they don't uniformly help encoders).

    Batching scope (see ``can_batch``): cross-request batching only engages when
    every request carries exactly ONE ``feature_lens`` segment, so the per-request
    output split is unambiguous. A request with multiple audio clips
    (multi-segment ``feature_lens``) falls back to the sequential ``forward`` —
    it is still correct, just not batched with its peers.
    """

    def __init__(self, audio_encoder: nn.Module, config: Qwen3OmniModelConfig):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        return NodeInputs(tensor_inputs={
            "audio_features": inputs["audio_features"][0],
            "audio_seqlens": inputs.get("audio_seqlens", [None])[0],
        })

    @staticmethod
    def _req_token_count(seqlens: torch.Tensor) -> int:
        from mstar.model.qwen3_omni.components.audio_encoder import _feat_extract_output_lengths
        return int(_feat_extract_output_lengths(seqlens.reshape(-1)).sum())

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]):
        feats = [i.tensor_inputs["audio_features"] for i in inputs]
        lens = [i.tensor_inputs["audio_seqlens"].reshape(-1) for i in inputs]
        counts = [self._req_token_count(l) for l in lens]
        return {
            "audio_features": torch.cat(feats, dim=1),   # (mel, sum_T)
            "audio_seqlens": torch.cat(lens),            # (sum_segments,)
            "req_token_counts": counts,
        }

    def forward_batched(self, graph_walk, engine_inputs, audio_features,
                        audio_seqlens, req_token_counts=None, **kwargs):
        embeds = self.audio_encoder(audio_features, audio_seqlens).last_hidden_state
        if embeds.dim() == 3:
            embeds = embeds.squeeze(0)
        request_ids = engine_inputs.request_ids
        if req_token_counts is None:  # single-segment-per-request fallback
            req_token_counts = [self._req_token_count(audio_seqlens[i:i + 1])
                                for i in range(len(request_ids))]
        results: dict[str, NameToTensorList] = {}
        off = 0
        for rid, c in zip(request_ids, req_token_counts, strict=False):
            results[rid] = {"audio_embeds": [embeds[off:off + c]]}
            off += c
        return results

    def forward(self, graph_walk, engine_inputs, audio_features, audio_seqlens, **kwargs):
        embeds = self.audio_encoder(audio_features, audio_seqlens).last_hidden_state
        if embeds.dim() == 3:
            embeds = embeds.squeeze(0)
        return {"audio_embeds": [embeds]}

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        # Safe pad-free batching needs one feature_lens entry per request so the
        # output split is unambiguous; otherwise defer to sequential forward.
        for mi in model_inputs:
            sl = mi.tensor_inputs.get("audio_seqlens")
            if sl is None or sl.reshape(-1).numel() != 1:
                return False
        return True


# ===================================================================
# 2. VisionEncoderSubmodule (enc_dec engine)
# ===================================================================


class VisionEncoderSubmodule(NodeSubmodule):
    """Thin wrapper around the HF vision encoder (ViT + spatial merge).

    Extracts pixel_values and grid_thw from inputs, computes cu_seqlens
    for FlashAttention, runs the encoder, and returns vision embeddings
    plus DeepStack intermediate features for the Thinker.

    Runs once per request (not batched across requests).
    """

    def __init__(self, vision_encoder: nn.Module, config: Qwen3OmniModelConfig):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        """Extract pixel_values, grid_thw, and compute cu_seqlens.

        ``pixel_values`` and ``image_grid_thw`` are produced by
        ``Qwen3OmniModel.process_prompt`` from the raw ``image_inputs``
        loaded by the data worker.
        """
        # Edge name from graph walk is "pixel_values"
        pixel_values = inputs["pixel_values"][0]       # (N_patches, C, patch_H, patch_W)
        grid_thw = inputs.get("image_grid_thw", inputs.get("grid_thw", [None]))[0]

        # Normalize grid_thw to shape (num_images, 3).  Single-image requests
        # store grid_thw as a 1-D tensor [T, H, W] (after process_prompt
        # indexes proc_out["image_grid_thw"][0] to strip the batch dim);
        # the per-image iteration logic below requires 2-D.
        if grid_thw is None:
            raise ValueError(
                "VisionEncoder: 'image_grid_thw' input is None. "
                "Make sure process_prompt is producing image_grid_thw via the "
                "HF AutoImageProcessor."
            )
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)  # (1, 3)

        return NodeInputs(
            tensor_inputs={
                "pixel_values": pixel_values,
                "grid_thw": grid_thw,
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        """Run vision encoder, return embeddings and DeepStack features.

        Returns:
            {
                "vision_embeds": [tensor of shape (vision_tokens, hidden_size)],
                "deepstack": [list of intermediate layer features],
            }
        """
        logger.debug(
            "Running VisionEncoder with pixel_values shape=%s, grid_thw shape=%s",
            pixel_values.shape, grid_thw.shape,
        )
        # HF vision encoder returns (hidden_states, deepstack_features)
        # depending on the model variant; handle both cases
        encoder_output = self.vision_encoder(
            pixel_values,
            grid_thw=grid_thw,
        )

        if isinstance(encoder_output, tuple):
            vision_embeds, deepstack = encoder_output
        else:
            vision_embeds = encoder_output.pooler_output
            deepstack = encoder_output.deepstack_features

        if isinstance(deepstack, torch.Tensor):
            deepstack = [deepstack]

        return {
            "vision_embeds": [vision_embeds],
            "deepstack": deepstack if deepstack is not None else [torch.tensor([])],
        }


class NativeVisionEncoderSubmodule(NodeSubmodule):
    """Native ViT submodule with cross-request batching + DeepStack.

    Multiple images batch with no padding: concatenate their patch rows and
    ``grid_thw`` rows, run one forward (per-image attention isolated by the
    encoder's ``cu_seqlens``), then slice the merged tokens AND each DeepStack
    level back per request. Same output contract as ``VisionEncoderSubmodule``
    (``vision_embeds`` + positionally-spliced ``deepstack``) so nothing upstream
    of ``vision_encoder.forward`` changes.
    """

    def __init__(self, vision_encoder: nn.Module, config: Qwen3OmniModelConfig):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.config = config
        # Source the merge factor from the encoder it was built with (not the
        # mstar config) so the per-request token split can never diverge from
        # what the encoder actually produces.
        self.merge_sq = vision_encoder.spatial_merge_size ** 2

    def _merged_tokens(self, grid_thw: torch.Tensor) -> int:
        g = grid_thw if grid_thw.dim() == 2 else grid_thw.unsqueeze(0)
        return int((g[:, 0] * g[:, 1] * g[:, 2]).sum() // self.merge_sq)

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        pixel_values = inputs["pixel_values"][0]
        grid_thw = inputs.get("image_grid_thw", inputs.get("grid_thw", [None]))[0]
        if grid_thw is None:
            raise ValueError("NativeVisionEncoder: 'image_grid_thw' input is None.")
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)
        return NodeInputs(tensor_inputs={"pixel_values": pixel_values, "grid_thw": grid_thw})

    def preprocess(self, graph_walk, engine_inputs, inputs: list[NodeInputs]):
        pvs = [i.tensor_inputs["pixel_values"] for i in inputs]
        grids = [i.tensor_inputs["grid_thw"] for i in inputs]
        counts = [self._merged_tokens(g) for g in grids]
        return {
            "pixel_values": torch.cat(pvs, dim=0),
            "grid_thw": torch.cat(grids, dim=0),
            "req_token_counts": counts,
        }

    def _run(self, pixel_values, grid_thw):
        out = self.vision_encoder(pixel_values, grid_thw=grid_thw)
        if isinstance(out, tuple):
            embeds, deepstack = out
        else:
            embeds, deepstack = out.pooler_output, out.deepstack_features
        if isinstance(deepstack, torch.Tensor):
            deepstack = [deepstack]
        return embeds, deepstack

    def forward_batched(self, graph_walk, engine_inputs, pixel_values, grid_thw,
                        req_token_counts=None, **kwargs):
        embeds, deepstack = self._run(pixel_values, grid_thw)
        request_ids = engine_inputs.request_ids
        if req_token_counts is None:  # one-image-per-request fallback
            g = grid_thw if grid_thw.dim() == 2 else grid_thw.unsqueeze(0)
            req_token_counts = [self._merged_tokens(g[i:i + 1]) for i in range(len(request_ids))]
        results: dict[str, NameToTensorList] = {}
        off = 0
        for rid, c in zip(request_ids, req_token_counts, strict=False):
            results[rid] = {
                "vision_embeds": [embeds[off:off + c]],
                "deepstack": [d[off:off + c] for d in deepstack],
            }
            off += c
        return results

    def forward(self, graph_walk, engine_inputs, pixel_values, grid_thw, **kwargs):
        embeds, deepstack = self._run(pixel_values, grid_thw)
        return {
            "vision_embeds": [embeds],
            "deepstack": deepstack if deepstack else [torch.tensor([])],
        }

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        return True


# ===================================================================
# 3. ThinkerSubmodule (ar engine) -- MOST COMPLEX
# ===================================================================


class ThinkerSubmodule(ARNodeSubmodule):
    """Wraps the FlashInfer-based Thinker MoE transformer.

    Dispatches on graph_walk:
      - prefill_text:   embed text tokens, compute 3D MRoPE, fill KV cache
      - prefill_audio:  splice audio embeddings, extend KV cache
      - prefill_vision: splice vision embeddings, extend KV cache
      - thinker_decode: embed previous token, single-step decode

    All walks produce ``thinker_states`` (layer-0 + layer-N concat) that
    stream to the Talker partition.  ``thinker_decode`` additionally
    produces ``logits`` for text token sampling.
    """

    # Default MRoPE section for head_dim=128: [24, 20, 20]
    MROPE_SECTION = [24, 20, 20]

    def __init__(
        self,
        thinker_model: nn.Module,
        config: Qwen3OmniModelConfig,
    ):
        super().__init__()
        self.model = thinker_model  # Qwen3OmniThinkerModel
        self.config = config

        # Pre-compute inverse frequencies for 3D MRoPE
        self._inv_freq: torch.Tensor | None = None

        # Lazily-cached constant mask used by ``_preprocess_decode`` for the
        # Talker partition.  Every decode-step mask is the same constant
        # ``[[0], [1]]`` so we allocate it once per device instead of per
        # request per step.  Helps keep the captured graph's output-dict
        # contents self-evidently constant, too.
        self._decode_thinker_mask: torch.Tensor | None = None

        self._audio_bos_embed: torch.Tensor | None = None
        self._audio_eos_embed: torch.Tensor | None = None

        self._vision_bos_embed: torch.Tensor | None = None
        self._vision_eos_embed: torch.Tensor | None = None

        # Chunked-vision prefill staging (MSTAR_CHUNKED_PREFILL_V2_VISION):
        # rid -> _VisionPrefillStage. The full wrapped embeds / 3D pos_ids /
        # per-layer deepstack are computed ONCE on the first chunk of a
        # prefill_vision walk and sliced per chunk. Purged by cleanup_request.
        self._vision_stage: dict[str, "_VisionPrefillStage"] = {}

    def cleanup_request(self, request_id: str) -> None:
        """Free any per-request chunked-vision staging (large GPU tensors) when
        the request finishes or is aborted."""
        self._vision_stage.pop(request_id, None)

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        """Lazy-initialize and cache inverse frequencies."""
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = compute_rope_freqs(
                self.config.thinker_head_dim,
                rope_theta=self.config.thinker_text.rope_theta,
                device=device,
            )
        return self._inv_freq

    def _get_decode_thinker_mask(self, device: torch.device) -> torch.Tensor:
        """Return the constant ``[[0], [1]]`` decode mask (multimodal row =
        0, text-inclusion row = 1), lazily allocated per device."""
        if (
            self._decode_thinker_mask is None
            or self._decode_thinker_mask.device != device
        ):
            self._decode_thinker_mask = torch.tensor(
                [[0], [1]], dtype=torch.bool, device=device,
            )
        return self._decode_thinker_mask

    def _get_talker_text_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Cut system prompt and previous assistant parts out of the talker input
        """
        im_start_indexes = (
            input_ids == self.config.im_start_token_id
        ).nonzero(as_tuple=True)[0]
        mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i]
            segment_end_index = im_start_indexes[i + 1]
            role_token = input_ids[im_start_index + 1]
            # Talker should ignore thinker system prompt
            if role_token == self.config.system_token_id:
                mask[im_start_index:segment_end_index] = 0
            elif role_token == self.config.assistant_token_id:
                mask[im_start_index:segment_end_index] = 0
        return mask

    def _wrap_audio_input(self, audio_embeds: torch.Tensor):
        # Wrap the audio span in ``<|audio_bos|>`` / ``<|audio_eos|>``
        # sentinel token embeddings so the Thinker sees the same
        # prompt layout the HF processor produces.
        #
        # When MSTAR_VLLM_AUDIO_SENTINELS=1, use the real Qwen3-Omni audio
        # marker IDs (151669/151670, what vLLM uses) instead of the legacy
        # 151647/151648 (mislabeled <|audio_bos|>/<|audio_eos|> in config.py).
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            vllm_audio_sentinels_enabled,
        )

        device = self.get_device()
        if self._audio_bos_embed is None or self._audio_eos_embed is None:
            if vllm_audio_sentinels_enabled():
                audio_start_id = 151669
                audio_end_id = 151670
            else:
                audio_start_id = self.config.thinker.audio_start_token_id
                audio_end_id = self.config.thinker.audio_end_token_id
            start_tok = torch.tensor(
                [audio_start_id], dtype=torch.long, device=device
            )
            end_tok = torch.tensor(
                [audio_end_id], dtype=torch.long, device=device
            )
            self._audio_bos_embed = self.model.model.embed_tokens(start_tok)
            self._audio_eos_embed = self.model.model.embed_tokens(end_tok)

        return torch.cat([
            self._audio_bos_embed,
            audio_embeds,
            self._audio_eos_embed
        ], dim=0)

    def _wrap_vision_input(self, vision_embeds: torch.Tensor):
        # Wrap the vision span in ``<|vision_bos|>`` / ``<|vision_eos|>``
        # sentinel token embeddings.
        if self._vision_bos_embed is None or self._vision_eos_embed is None:
            device = vision_embeds.device
            vision_start_id = self.config.thinker.vision_start_token_id
            vision_end_id = self.config.thinker.vision_end_token_id
            start_tok = torch.tensor(
                [vision_start_id], dtype=torch.long, device=device
            )
            end_tok = torch.tensor(
                [vision_end_id], dtype=torch.long, device=device
            )
            self._vision_bos_embed = self.model.model.embed_tokens(start_tok)
            self._vision_eos_embed = self.model.model.embed_tokens(end_tok)

        return torch.cat([
            self._vision_bos_embed,
            vision_embeds,
            self._vision_eos_embed
        ], dim=0)

    def prepare_inputs_batched(
        self,
        graph_walk: str,
        inputs_list: list[NameToTensorList],
        pos_infos: list[dict[str, PositionInfo]],
        **kwargs,
    ) -> list[ARNodeInputs] | None:
        """Batched thinker_decode input prep: one token cat + ONE
        embed_tokens launch + one pos-ids H2D for the whole batch, instead
        of per-request embed_tokens (+ per-request tiny H2D) — at B32 that
        is 32 embedding launches -> 1. Returns per-request zero-copy views
        so everything downstream (preprocess, capture padding) is
        unchanged. Returns None for other walks (engine falls back to the
        per-request path).
        """
        if graph_walk != "thinker_decode":
            return None
        device = self.get_device()
        toks = []
        for inputs in inputs_list:
            t = inputs["text_inputs"][0]
            toks.append(t.reshape(-1)[:1])
        token_ids = torch.cat(toks).to(device)          # (bs,)
        embeds = self.model.model.embed_tokens(token_ids)  # (bs, hidden)
        starts = [
            pi.get("main", PositionInfo()).position_id_start for pi in pos_infos
        ]
        if _PREP_DEVICE_POS_BATCHED:
            # Sync-free: write starts into a PINNED host buffer, then a genuinely
            # async H2D (replaces the pageable torch.tensor(starts) whose
            # non_blocking=True was a no-op). Cycle a small ring of buffers so a
            # step's host write never overwrites the source of a prior step's
            # still-in-flight async copy when the CPU runs ahead of the GPU. The
            # ring depth exceeds the copy-in-flight depth, so a reused buffer's
            # prior H2D has always completed. start_pos values are read FRESH from
            # pos_infos each step, so positions can't drift.
            n = len(starts)
            ring = getattr(self, "_pos_host_ring", None)
            if ring is None:
                ring = [None] * 4
                self._pos_host_ring = ring
                self._pos_host_idx = 0
            self._pos_host_idx = (self._pos_host_idx + 1) % len(ring)
            host = ring[self._pos_host_idx]
            if host is None or host.numel() < n:
                host = torch.empty(max(n, 32), dtype=torch.float, pin_memory=True)
                ring[self._pos_host_idx] = host
            host.numpy()[:n] = starts                    # host-side write, no sync
            pos = host[:n].to(device, non_blocking=True)  # async H2D from pinned
            _prep_pos_count()
        else:
            pos = torch.tensor(starts, dtype=torch.float).to(device, non_blocking=True)
        pos3 = pos.unsqueeze(0).expand(3, -1)            # (3, bs)
        mask = self._get_decode_thinker_mask(device)
        return [
            ARNodeInputs(
                input_seq_len=1,
                input_embeds=embeds[i:i + 1],
                custom_pos_ids=pos3[:, i:i + 1],
                tensor_inputs={"masks_for_talker": mask},
            )
            for i in range(len(inputs_list))
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        seen_token_mask: SeenTokenMask,
        pos_info: dict[str, PositionInfo] = {},
        **kwargs
    ) -> ARNodeInputs:
        device = self.get_device()
        start_pos = pos_info.get("main", PositionInfo()).position_id_start

        # Mixed batch: the batch-level ``graph_walk`` is "thinker_mixed",
        # but each request in the batch carries its OWN walk (a decode row or
        # the single prefill-chunk row). ``prepare_inputs`` runs once per
        # request, so dispatch on the request's real walk from ``fwd_info``,
        # not the batch walk. The resulting per-request ARNodeInputs
        # (input_seq_len=1 for decode rows, =C for the chunk row) are what
        # ``preprocess`` concatenates into the mixed [1]*n+[C] layout.
        if graph_walk == "thinker_mixed":
            graph_walk = fwd_info.graph_walk

        if graph_walk == "thinker_decode":
            # Get previous token ID from text_inputs
            token_id = inputs["text_inputs"][0].to(device)  # (1,) or scalar
            if token_id.dim() == 0:
                token_id = token_id.unsqueeze(0)
            embeds = self.model.model.embed_tokens(token_id)

            # Next MRoPE position for all 3 components: read from the
            # per-request cache-manager state (kept in sync by the
            # post-forward ``advance_seq_lens`` call in ``thinker.py``).
            if _PREP_DEVICE_POS:
                # Sync-free: empty() (caching allocator, no copy) + fill_()
                # (scalar is a kernel arg, no H2D) replaces the pageable
                # torch.tensor(..., device=cuda). All 3 MRoPE components share
                # start_pos for a text token, so the value is byte-identical to
                # the torch.tensor path. start_pos is read FRESH from pos_info
                # (the advance_seq_lens source of truth) each step, so positions
                # can't drift — no self-maintained counter (unlike a slot
                # scheme, which risks desyncing from advance_seq_lens → MRoPE
                # drift). Fresh buffer per call → no reuse/sharing hazard at any
                # batch size (the runner copies it into the captured static
                # input before replay).
                pos_ids = torch.empty((3, 1), dtype=torch.float, device=device)
                pos_ids.fill_(float(start_pos))
                _prep_pos_count()
            else:
                pos_ids = torch.tensor(
                    [[start_pos], [start_pos], [start_pos]],
                    dtype=torch.float,
                    device=device,
                )  # (3, 1)

            return ARNodeInputs(
                input_seq_len=1,
                input_embeds=embeds,
                custom_pos_ids=pos_ids,
                tensor_inputs={
                    "masks_for_talker": self._get_decode_thinker_mask(device)
                }  # no additional tensors for decode step
            )

        if graph_walk == "prefill_text":
            text_ids = inputs["text_inputs"][0].to(device)  # (full_span,)

            # Resumable chunked prefill (MSTAR_CHUNKED_PREFILL_V2): the conductor
            # re-emits this walk with a per-chunk ``prefill_chunk_offset`` /
            # ``prefill_chunk_len`` window. Slice the full text span to this
            # chunk. Absent metadata (flag off, or a walk short enough not to
            # chunk) => full span, byte-identical to the unchunked path.
            chunk_off = int(fwd_info.step_metadata.get("prefill_chunk_offset", 0))
            chunk_len = fwd_info.step_metadata.get("prefill_chunk_len")
            if chunk_len is not None:
                text_ids = text_ids[chunk_off:chunk_off + int(chunk_len)]

            embeds = self.model.model.embed_tokens(text_ids)
            seq_len = text_ids.shape[0]

            # NOTE: newly-sampled tokens automatically added sto the seen token mask in decode
            seen_token_mask.add_tokens(text_ids)

            # Compute 3D MRoPE position IDs for a pure-text span.  Each
            # prefill graph walk is single-modality so we use the simple
            # per-modality helper instead of the full HF parser.
            #
            # ``start_pos`` is the next MRoPE position for this request,
            # carried forward across walks (and across chunks) by
            # ``state.position_id_start`` (advanced post-forward by
            # ``advance_seq_lens``). For a chunk it already reflects the
            # tokens consumed by prior chunks, so positions stay contiguous.
            pos_ids = get_rope_index_text(seq_len, start_pos, device)
            masks_for_talker = torch.stack([
                torch.zeros(text_ids.shape, dtype=torch.bool, device=device), # multimodal
                self._get_talker_text_mask(text_ids) # text inclusion
            ])
            return ARNodeInputs(
                input_seq_len=seq_len,
                input_embeds=embeds,
                custom_pos_ids=pos_ids,
                tensor_inputs={
                    "masks_for_talker": masks_for_talker
                }
            )

        if graph_walk == "prefill_audio":
            stage = self._build_audio_full(inputs, start_pos, device)
            masks_for_talker = torch.stack([stage.mm_mask, ~stage.mm_mask])
            return ARNodeInputs(
                input_seq_len=stage.total_len,
                input_embeds=stage.wrapped_embeds,
                custom_pos_ids=stage.pos_ids,
                tensor_inputs={
                    "masks_for_talker": masks_for_talker
                }
            )

        if graph_walk == "prefill_multimodal_audio":
            return self._build_merged_audio_inputs(
                fwd_info, inputs, start_pos, device, seen_token_mask,
            )

        if graph_walk == "prefill_vision":
            # Chunked-vision (MSTAR_CHUNKED_PREFILL_V2_VISION): stage the full
            # wrapped embeds / pos_ids / deepstack on the first chunk (offset 0),
            # then slice per chunk. Absent chunk metadata (flag off) => single
            # full-span step, byte-identical to the unchunked path below.
            chunk_len = fwd_info.step_metadata.get("prefill_chunk_len")
            if chunk_len is not None:
                return self._vision_chunk_inputs(
                    fwd_info, inputs, start_pos, device,
                )
            full = self._build_vision_full(inputs, start_pos, device)
            mm_mask = full.mm_mask
            masks_for_talker = torch.stack([mm_mask, ~mm_mask])
            tensor_inputs: dict[str, torch.Tensor] = {
                "masks_for_talker": masks_for_talker,
            }
            for i, ds in enumerate(full.deepstack):
                tensor_inputs[f"deepstack_{i}"] = ds
            return ARNodeInputs(
                input_seq_len=full.total_len,
                input_embeds=full.wrapped_embeds,
                custom_pos_ids=full.pos_ids,
                tensor_inputs=tensor_inputs,
                kwargs={"mrope_pos_advance": full.mrope_pos_advance},
            )

        if graph_walk == "prefill_multimodal":
            return self._build_merged_multimodal_inputs(
                fwd_info, inputs, start_pos, device, seen_token_mask,
            )

    def _build_merged_multimodal_inputs(
        self,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        start_pos: float,
        device,
        seen_token_mask: "SeenTokenMask",
    ) -> ARNodeInputs:
        """Merged text+vision Thinker prefill (MSTAR_MERGED_PREFILL).

        Build ONE ARNodeInputs spanning the whole prompt by concatenating the
        text and vision spans in modality order (``merged_vision_first`` from the
        conductor), threading the MRoPE start position across them EXACTLY as the
        separate ``prefill_text`` / ``prefill_vision`` walks would after
        ``advance_seq_lens`` (text advances ``position_id_start`` by ``seq_len``;
        vision by its 3D-grid ``mrope_pos_advance``). The per-span embeds /
        pos_ids / deepstack are computed by the SAME helpers with the SAME
        threaded ``start_pos`` as the standalone walks, so they are bit-identical
        — only the concatenation into one forward differs (causal attention over
        ``[A][B]`` == B attending to A's already-resident KV, so the KV cache is
        identical modulo kernel-tiling ULP drift). The resulting signature
        (``input_embeds`` + pos_ids + per-layer ``deepstack_<i>`` +
        ``mrope_pos_advance``) matches ``prefill_vision``, so this replays on the
        ``prefill_vision`` capture (text rows zero-fill deepstack, the same
        zero-rows pattern the mixed-batch vision path uses).
        """
        vision_first = bool(fwd_info.step_metadata.get("merged_vision_first"))
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)
        hidden = self.config.thinker_hidden_size

        # --- text span (verbatim from the prefill_text branch) ---
        text_ids = inputs["text_inputs"][0].to(device)
        text_embeds = self.model.model.embed_tokens(text_ids)
        text_len = text_ids.shape[0]
        seen_token_mask.add_tokens(text_ids)
        text_talker_mask = torch.stack([
            torch.zeros(text_ids.shape, dtype=torch.bool, device=device),
            self._get_talker_text_mask(text_ids),
        ])

        if vision_first:
            # vision occupies [start_pos, start_pos + mrope_pos_advance); text
            # continues from there (== prefill_vision then prefill_text).
            stage = self._build_vision_full(inputs, start_pos, device)
            text_pos = get_rope_index_text(
                text_len, start_pos + stage.mrope_pos_advance, device,
            )
            embeds = torch.cat([stage.wrapped_embeds, text_embeds], dim=0)
            pos_ids = torch.cat([stage.pos_ids, text_pos], dim=1)
            vision_talker = torch.stack([stage.mm_mask, ~stage.mm_mask])
            masks_for_talker = torch.cat([vision_talker, text_talker_mask], dim=1)
            text_zeros = torch.zeros(
                (text_len, hidden), dtype=stage.wrapped_embeds.dtype, device=device,
            )
            deepstack = [
                torch.cat([stage.deepstack[i], text_zeros], dim=0)
                for i in range(num_deepstack)
            ]
            total_advance = stage.mrope_pos_advance + text_len
        else:
            # text occupies [start_pos, start_pos + text_len); vision continues
            # from there (== prefill_text then prefill_vision).
            text_pos = get_rope_index_text(text_len, start_pos, device)
            stage = self._build_vision_full(inputs, start_pos + text_len, device)
            embeds = torch.cat([text_embeds, stage.wrapped_embeds], dim=0)
            pos_ids = torch.cat([text_pos, stage.pos_ids], dim=1)
            vision_talker = torch.stack([stage.mm_mask, ~stage.mm_mask])
            masks_for_talker = torch.cat([text_talker_mask, vision_talker], dim=1)
            text_zeros = torch.zeros(
                (text_len, hidden), dtype=stage.wrapped_embeds.dtype, device=device,
            )
            deepstack = [
                torch.cat([text_zeros, stage.deepstack[i]], dim=0)
                for i in range(num_deepstack)
            ]
            total_advance = text_len + stage.mrope_pos_advance

        tensor_inputs: dict[str, torch.Tensor] = {
            "masks_for_talker": masks_for_talker,
        }
        for i, ds in enumerate(deepstack):
            tensor_inputs[f"deepstack_{i}"] = ds
        return ARNodeInputs(
            input_seq_len=embeds.shape[0],
            input_embeds=embeds,
            custom_pos_ids=pos_ids,
            tensor_inputs=tensor_inputs,
            kwargs={"mrope_pos_advance": total_advance},
        )

    def _build_merged_audio_inputs(
        self,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        start_pos: float,
        device,
        seen_token_mask: "SeenTokenMask",
    ) -> ARNodeInputs:
        """Merged text+audio Thinker prefill (MSTAR_MERGED_PREFILL_AUDIO).

        Build ONE ARNodeInputs spanning the whole prompt by concatenating the
        text and audio spans in modality order (``merged_audio_first`` from the
        conductor), threading the MRoPE start position across them EXACTLY as the
        separate ``prefill_text`` / ``prefill_audio`` walks would after
        ``advance_seq_lens`` (each span advances ``position_id_start`` by its own
        ``seq_len``: text by ``text_len``, audio by ``audio_len + 2`` — audio has
        NO 3D-grid jump, its positions increment one per token). The per-span
        embeds / pos_ids are computed by the SAME helpers with the SAME threaded
        ``start_pos`` as the standalone walks, so they are bit-identical — only
        the concatenation into one forward differs (causal attention over
        ``[A][B]`` == B attending to A's already-resident KV).

        Because audio carries neither deepstack nor a custom MRoPE advance, the
        resulting signature (``input_embeds`` + pos_ids + masks_for_talker) is
        identical to ``prefill_text`` / ``prefill_audio``, so this replays on the
        ``prefill_text`` capture (NOT the vision capture) and the default
        ``advance_seq_lens`` (by ``seq_len``) lands the running position exactly
        where the standalone walks would.

        ``merged_audio_order`` selects the span layout:
          * ``"audio_first"`` / ``"text_first"`` — 2-entry legacy layout.
          * ``"interleaved"`` — vLLM-layout s2t ([prefix-text, audio,
            suffix-text]); the two text spans arrive as ``text_inputs`` (prefix)
            and ``text_inputs_suffix`` (suffix).
        """
        order = fwd_info.step_metadata.get("merged_audio_order")

        def _text_span(key: str, span_start: float):
            ids = inputs[key][0].to(device)
            embeds = self.model.model.embed_tokens(ids)
            seen_token_mask.add_tokens(ids)
            talker = torch.stack([
                torch.zeros(ids.shape, dtype=torch.bool, device=device),
                self._get_talker_text_mask(ids),
            ])
            pos = get_rope_index_text(ids.shape[0], span_start, device)
            return embeds, pos, talker, ids.shape[0]

        def _audio_span(span_start: float):
            stage = self._build_audio_full(inputs, span_start, device)
            talker = torch.stack([stage.mm_mask, ~stage.mm_mask])
            return stage.wrapped_embeds, stage.pos_ids, talker, stage.total_len

        # Build the ordered list of spans, threading start_pos across them EXACTLY
        # as the standalone walks would (each advances by its own seq_len — text
        # by token count, audio by audio_len+2; all linear, no side-channel).
        pos = start_pos
        spans = []  # (embeds, pos_ids, talker_mask)
        def _emit(span):
            e, p, t, n = span
            spans.append((e, p, t))
            return n

        if order == "interleaved":
            pos += _emit(_text_span("text_inputs", pos))
            pos += _emit(_audio_span(pos))
            pos += _emit(_text_span("text_inputs_suffix", pos))
        elif order == "audio_first":
            pos += _emit(_audio_span(pos))
            pos += _emit(_text_span("text_inputs", pos))
        else:  # "text_first"
            pos += _emit(_text_span("text_inputs", pos))
            pos += _emit(_audio_span(pos))

        embeds = torch.cat([s[0] for s in spans], dim=0)
        pos_ids = torch.cat([s[1] for s in spans], dim=1)
        masks_for_talker = torch.cat([s[2] for s in spans], dim=1)

        return ARNodeInputs(
            input_seq_len=embeds.shape[0],
            input_embeds=embeds,
            custom_pos_ids=pos_ids,
            tensor_inputs={
                "masks_for_talker": masks_for_talker,
            },
        )

    def _build_audio_full(
        self, inputs: NameToTensorList, start_pos: float, device,
    ) -> "_AudioPrefillStage":
        """Compute the full-span audio Thinker prefill tensors once: wrapped
        embeds (audio_bos + audio + audio_eos), 3D MRoPE pos_ids, and mm_mask.
        Extracted verbatim from the single-shot prefill_audio path so flag-off
        stays byte-identical."""
        audio_embeds = inputs["audio_embeds"][0].to(device)  # (audio_tokens, hidden)
        audio_len = audio_embeds.shape[0]

        mm_mask = torch.ones(audio_len + 2, dtype=torch.bool, device=device)
        mm_mask[[0, -1]] = 0

        wrapped_embeds = self._wrap_audio_input(audio_embeds)
        total_len = audio_len + 2
        # Position IDs:
        #   - audio_start_token: text-like position at start_pos
        #   - audio tokens:      temporal increments per frame,
        #                        h/w = start_pos (handled by helper)
        #   - audio_end_token:   text-like position right after
        start_pos_ids = get_rope_index_text(1, start_pos, device)
        audio_pos_ids = get_rope_index_audio(
            audio_len,
            start_pos + 1,
            device,
            self.config.thinker.position_id_per_seconds,
        )
        # M-RoPE parity: M* pins audio h/w to a constant, HF ramps them with
        # temporal. Under MSTAR_VLLM_PROMPT_LAYOUT, set h/w == temporal so the
        # 3D position_ids are byte-identical to HF get_rope_index.
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            vllm_prompt_layout_enabled,
        )
        if vllm_prompt_layout_enabled():
            audio_pos_ids = audio_pos_ids.clone()
            audio_pos_ids[1] = audio_pos_ids[0]
            audio_pos_ids[2] = audio_pos_ids[0]
        end_pos_ids = get_rope_index_text(
            1, start_pos + 1 + audio_len, device
        )
        pos_ids = torch.cat(
            [start_pos_ids, audio_pos_ids, end_pos_ids], dim=1
        )
        return _AudioPrefillStage(
            wrapped_embeds=wrapped_embeds,
            pos_ids=pos_ids,
            mm_mask=mm_mask,
            total_len=total_len,
        )

    def _build_vision_full(
        self, inputs: NameToTensorList, start_pos: float, device,
    ) -> "_VisionPrefillStage":
        """Compute the full-span vision Thinker prefill tensors once: wrapped
        embeds, 3D MRoPE pos_ids, per-layer deepstack scatter, mm_mask, and the
        full 3D-grid MRoPE advance. Extracted verbatim from the single-shot
        prefill_vision path so flag-off stays byte-identical."""
        vision_embeds = inputs["vision_embeds"][0].to(device)
        vision_len = vision_embeds.shape[0]

        mm_mask = torch.ones(vision_len + 2, dtype=torch.bool, device=device)
        mm_mask[[0, -1]] = 0

        wrapped_embeds = self._wrap_vision_input(vision_embeds)
        total_len = vision_len + 2
        grid_thw = inputs.get("image_grid_thw", [None])[0]
        seconds_per_grid = inputs.get("video_second_per_grid", [])
        seconds_per_grid = seconds_per_grid[0].item() if seconds_per_grid else None
        vision_pos_ids = get_rope_index_vision(
            grid_thw,
            start_pos + 1,  # leave room for the BOS token
            position_id_per_seconds=self.config.thinker.position_id_per_seconds,
            device=device,
            spatial_merge_size=self.config.vision.spatial_merge_size,
            seconds_per_grid=seconds_per_grid,
        )

        start_pos_ids = get_rope_index_text(1, start_pos, device)
        vstart = start_pos + 1
        sms = self.config.vision.spatial_merge_size
        _grid = grid_thw if grid_thw.dim() == 2 else grid_thw.unsqueeze(0)
        _max_pos = float("-inf")
        for _row in _grid:
            gt, gh, gw = int(_row[0]), int(_row[1]), int(_row[2])
            spatial_max = max(gh // sms, gw // sms) - 1 + vstart
            if seconds_per_grid is None:
                temporal_max = vstart
            else:
                temporal_max = (
                    (gt - 1) * seconds_per_grid
                    * self.config.thinker.position_id_per_seconds
                )
            _max_pos = max(_max_pos, spatial_max, temporal_max)
        end_pos_base = float(_max_pos) + 1
        end_pos_ids = get_rope_index_text(1, end_pos_base, device)

        pos_ids = torch.cat(
            [start_pos_ids, vision_pos_ids, end_pos_ids], dim=1
        )
        mrope_pos_advance = int(end_pos_base + 1 - start_pos)

        deepstack: list[torch.Tensor] = []
        for deepstack_inp in inputs["deepstack"]:
            full_deepstack = torch.zeros_like(wrapped_embeds)
            full_deepstack[mm_mask, :] = deepstack_inp
            deepstack.append(full_deepstack)

        return _VisionPrefillStage(
            wrapped_embeds=wrapped_embeds,
            pos_ids=pos_ids,
            deepstack=deepstack,
            mm_mask=mm_mask,
            total_len=total_len,
            mrope_pos_advance=mrope_pos_advance,
            start_pos=start_pos,
        )

    def _vision_chunk_inputs(
        self,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        start_pos: float,
        device,
    ) -> ARNodeInputs:
        """One chunk of a chunked vision Thinker prefill.

        On the first chunk (offset 0) build + stash the full staged tensors;
        every chunk slices [off:off+C] from the stage. The forward uses the
        STAGED pos_ids (absolute positions from the walk's original start_pos),
        so intra-block 3D positions are exact regardless of how position_id_start
        has advanced across chunks.

        Per-chunk MRoPE advance: KV seq_len advances by C automatically
        (plan_attention). position_id_start must end at start_pos +
        mrope_pos_advance (the full 3D-grid span). So non-last chunks advance by
        C; the last chunk advances by the remaining span
        (mrope_pos_advance - committed_C) so the running position lands exactly
        on the unchunked post-vision value.
        """
        rid = fwd_info.request_id
        off = int(fwd_info.step_metadata.get("prefill_chunk_offset", 0))
        clen = int(fwd_info.step_metadata["prefill_chunk_len"])

        if off == 0:
            stage = self._build_vision_full(inputs, start_pos, device)
            self._vision_stage[rid] = stage
            logger.debug(
                "chunked_prefill_v2 vision: rid=%s span=%d C=%d start_pos=%s",
                rid, stage.total_len, clen, start_pos,
            )
        else:
            stage = self._vision_stage.get(rid)
            if stage is None:
                # Should not happen (chunk 0 always stages first); fail loud so a
                # dropped stage can't silently corrupt positions.
                raise RuntimeError(
                    f"chunked vision prefill: missing stage for rid={rid} at "
                    f"offset={off}"
                )

        end = off + clen
        walk_done = end >= stage.total_len

        embeds = stage.wrapped_embeds[off:end]
        pos_ids = stage.pos_ids[:, off:end]
        mm_mask_chunk = stage.mm_mask[off:end]
        masks_for_talker = torch.stack([mm_mask_chunk, ~mm_mask_chunk])
        tensor_inputs: dict[str, torch.Tensor] = {
            "masks_for_talker": masks_for_talker,
        }
        for i, ds in enumerate(stage.deepstack):
            tensor_inputs[f"deepstack_{i}"] = ds[off:end]

        if walk_done:
            # Land position_id_start exactly on the unchunked post-vision value.
            pos_advance = stage.mrope_pos_advance - off
            if self._vision_prefill_assert_enabled():
                # After this chunk: seq_len += (prior off) + clen == total_len;
                # position_id_start += off + pos_advance == mrope_pos_advance.
                assert off + clen == stage.total_len, (
                    f"vision chunk span mismatch rid={rid}: off={off} C={clen} "
                    f"total={stage.total_len}"
                )
                assert off + pos_advance == stage.mrope_pos_advance, (
                    f"vision MRoPE advance mismatch rid={rid}: off={off} "
                    f"pos_advance={pos_advance} full={stage.mrope_pos_advance}"
                )
            self._vision_stage.pop(rid, None)
        else:
            # Non-last chunk: advance position by the chunk length (keeps
            # position_id_start and seq_len in step; the forward uses staged
            # absolute pos_ids so this value never feeds a chunk's positions).
            pos_advance = clen

        return ARNodeInputs(
            input_seq_len=clen,
            input_embeds=embeds,
            custom_pos_ids=pos_ids,
            tensor_inputs=tensor_inputs,
            kwargs={"mrope_pos_advance": pos_advance},
        )

    @staticmethod
    def _vision_prefill_assert_enabled() -> bool:
        return os.environ.get("MSTAR_CHUNKED_PREFILL_V2_ASSERT", "").strip().lower() \
            in ("1", "true", "yes", "on")

    @staticmethod
    def _mixed_batch_assert_enabled() -> bool:
        return os.environ.get("MSTAR_MIXED_BATCH_ASSERT", "").strip().lower() \
            in ("1", "true", "yes", "on")

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]: # input name to tensor
        device = self.get_device()
        # Concatenate across requests
        input_embeds = torch.cat([
            inp.input_embeds for inp in inputs
        ], dim=0)
        position_ids_3d = torch.cat([
            inp.custom_pos_ids for inp in inputs
        ], dim=1)  # (3, total_tokens)
        seq_lens = [
            inp.input_seq_len for inp in inputs
        ]

        # Compute cos/sin for 3D MRoPE.  Returned as separate tensor keys
        # (not a tuple) so the CUDA graph runner can detect them as static
        # inputs and copy them into the captured buffers at replay.
        inv_freq = self._get_inv_freq(device)
        cos_3d, sin_3d = compute_3d_cos_sin(
            position_ids_3d, inv_freq,
            mrope_section=self.MROPE_SECTION,
            target_dtype=input_embeds.dtype,
        )

        # Plan FlashInfer attention and rope for the main cache label
        cache_manager = engine_inputs.cache_manager
        cache_manager.set_active_label("main")
        assert cache_manager is not None
        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main"
        )
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        extra_inputs = {}
        # Mixed batch: the concatenation above already produces the mixed
        # [decode embeds (n,H); chunk embeds (C,H)] layout and the
        # ``seq_lens=[1]*n+[C]`` that plan_attention needs. For a TEXT chunk row
        # nothing further is required — the packed tensor signature is
        # input_embeds + cos_3d + sin_3d only, so decode + prefill_text rows
        # concatenate byte-identically to a plain prefill step.
        #
        # MSTAR_MIXED_BATCH_VISION: a VISION chunk row additionally
        # carries per-layer ``deepstack_<i>`` tensor_inputs and a
        # ``mrope_pos_advance`` kwarg (built by ``_vision_chunk_inputs``). Two
        # things then differ from the text case:
        #   * Deepstack: assemble packed (total_tokens, hidden) per-layer
        #     tensors exactly like ``prefill_vision`` below — the chunk row's
        #     (C, hidden) slice occupies its row span [n:n+C); decode rows have
        #     no deepstack key and are zero-filled (a no-op additive splice).
        #   * MRoPE advance: ``advance_seq_lens`` consumes ``custom_pos_advance``
        #     all-or-nothing per plan-state, so once the vision chunk needs a
        #     custom advance EVERY row must supply one. Decode/text rows fall
        #     back to their ``input_seq_len`` (=1 per decode token), the vision
        #     chunk supplies its 3D-grid span, padding rows supply 0.
        if graph_walk == "thinker_mixed":
            from mstar.model.qwen3_omni.qwen3_omni_model import (
                mixed_vision_capture_provisioned,
            )
            # Structure of a mixed batch: n decode rows (seq_len 1) followed by
            # exactly one chunk row (seq_len C > 1). Padding rows (seq_len 0)
            # appended by the runner sit AFTER the real rows; count only real
            # (>0) rows here. n_decode / C / total feed the DEBUG log + the
            # optional assert.
            real_lens = [sl for sl in seq_lens if sl > 0]
            n_decode = sum(1 for sl in real_lens if sl == 1)
            chunk_lens = [sl for sl in real_lens if sl > 1]
            total = sum(real_lens)
            # A vision chunk row is one carrying deepstack tensor_inputs.
            has_vision_chunk = any(
                any(k.startswith("deepstack_") for k in inp.tensor_inputs)
                for inp in inputs
            )
            # Gate the packed deepstack emission on what the capture ACTUALLY
            # provisioned, not the live flag: emitting deepstack for a step whose
            # captured graph lacks the per-layer static buffers is an illegal
            # memory access. The scheduler routes on the same latch.
            vision_capture = mixed_vision_capture_provisioned()
            logger.debug(
                "thinker_mixed step: n_decode=%d C=%s total_tokens=%d bucket=%d "
                "vision_chunk=%s vision_capture=%s",
                n_decode,
                chunk_lens[0] if chunk_lens else None,
                total,
                len(input_embeds),
                has_vision_chunk,
                vision_capture,
            )
            if self._mixed_batch_assert_enabled():
                assert len(chunk_lens) == 1, (
                    f"mixed batch expected exactly 1 chunk row (seq_len>1); "
                    f"got {len(chunk_lens)} in seq_lens={seq_lens}"
                )
                assert n_decode == len(real_lens) - 1, (
                    f"mixed batch: non decode/chunk row in seq_lens={seq_lens}"
                )
                if not vision_capture:
                    for inp in inputs:
                        assert not any(
                            k.startswith("deepstack_") for k in inp.tensor_inputs
                        ), (
                            "mixed batch got a vision chunk (deepstack tensors "
                            "present) but MSTAR_MIXED_BATCH_VISION is off — the "
                            "text-signature capture cannot splice deepstack; "
                            "assembly must gate the chunk row to prefill_text."
                        )
            # When the vision-capture flag is on, the ONE thinker_mixed capture
            # carries per-layer ``deepstack_<i>`` statics for every replay (see
            # get_cuda_graph_configs). ``static_input_keys`` therefore includes
            # them, and the runner's replay copy skips any key preprocess does
            # not re-emit — leaving a STALE deepstack buffer from a prior vision
            # step. So under the flag we must emit deepstack on EVERY mixed step,
            # zero-filled when the chunk row is text (a no-op additive splice),
            # so the copy overwrites the static buffer to zeros. With the flag
            # off there are no deepstack statics and this block never runs.
            if vision_capture:
                # Packed per-layer deepstack, identical shape to prefill_vision:
                # concat each row's (input_seq_len, hidden) contribution. Decode
                # rows (and a text chunk row) contribute zeros; a vision chunk
                # row contributes its (C, hidden) slice at its row span. Only
                # real rows (seq_len>0) produce nonzero deepstack; zero-length
                # padding rows add empty (0, hidden) slices, so the packed
                # length == total real tokens.
                num_deepstack = len(self.config.vision.deepstack_visual_indexes)
                for i in range(num_deepstack):
                    layer_tensors: list[torch.Tensor] = []
                    for inp in inputs:
                        t = inp.tensor_inputs.get(f"deepstack_{i}")
                        if t is None:
                            t = torch.zeros(
                                (inp.input_seq_len, self.config.thinker_hidden_size),
                                dtype=input_embeds.dtype, device=device,
                            )
                        layer_tensors.append(t)
                    extra_inputs[f"deepstack_{i}"] = torch.cat(layer_tensors, dim=0)
                # Per-request position advance. Decode/text rows advance by their
                # token count (input_seq_len), which equals the default seq_len
                # advance; a vision chunk supplies its explicit 3D-grid
                # ``mrope_pos_advance``; padding rows -> 0. ``advance_seq_lens``
                # consumes ``custom_pos_advance`` all-or-nothing per plan-state,
                # so once ANY row needs a custom advance every row must supply
                # one — hence the full list even on a text-chunk mixed step.
                mixed_pos_advance = [
                    inp.kwargs.get("mrope_pos_advance", inp.input_seq_len)
                    for inp in inputs
                ]
                if self._mixed_batch_assert_enabled():
                    for inp in inputs:
                        clen = inp.input_seq_len
                        for i in range(num_deepstack):
                            t = inp.tensor_inputs.get(f"deepstack_{i}")
                            if t is not None:
                                assert t.shape[0] == clen, (
                                    f"mixed vision chunk deepstack_{i} slice "
                                    f"len {t.shape[0]} != chunk C {clen}"
                                )
                extra_inputs["mrope_pos_advance"] = mixed_pos_advance
                cache_manager.set_custom_pos_advance(
                    mixed_pos_advance, label="main",
                )
        # prefill_multimodal (MSTAR_MERGED_PREFILL) carries the same
        # post-preprocess signature as prefill_vision — per-layer deepstack_<i>
        # statics + the mrope_pos_advance side-channel — so it runs the identical
        # assembly here (single request per merged prefill, so the single-request
        # invariant holds regardless of the vision-batch flag).
        if graph_walk in ("prefill_vision", "prefill_multimodal"):
            from mstar.model.qwen3_omni.qwen3_omni_model import (
                batch_vision_prefill_enabled,
            )
            if graph_walk == "prefill_vision" and not batch_vision_prefill_enabled():
                assert len(inputs) == 1, \
                    "Batching not implemented for Thinker vision prefill"
            if graph_walk == "prefill_multimodal":
                # Merged prefill is one request per step (reuses the bs=1
                # prefill_vision capture); the scheduler never packs two.
                assert len(inputs) == 1, \
                    "Batching not implemented for merged multimodal prefill"
            num_deepstack = len(self.config.vision.deepstack_visual_indexes)
            for i in range(num_deepstack):
                layer_tensors: list[torch.Tensor] = []
                for inp in inputs:
                    t = inp.tensor_inputs.get(f"deepstack_{i}")
                    if t is None:
                        t = torch.zeros(
                            (inp.input_seq_len, self.config.thinker_hidden_size),
                            dtype=input_embeds.dtype, device=device,
                        )
                    layer_tensors.append(t)
                extra_inputs[f"deepstack_{i}"] = torch.cat(layer_tensors, dim=0)
            mrope_pos_advance = [
                inp.kwargs.get("mrope_pos_advance", 0) for inp in inputs
            ]
            extra_inputs["mrope_pos_advance"] = mrope_pos_advance
            # Side-channel: stash on the cache_manager's plan state via the
            # public setter so the CUDA-graph runner's post-replay
            # ``advance_seq_lens()`` (which is called with no args) advances
            # ``position_id_start`` by the MRoPE 3D-grid span instead of by
            # ``seq_len``. The eager path consumes ``mrope_pos_advance`` from
            # the dict via model.forward → cache_handle.advance_seq_lens(
            # pos_id_ns=...); both paths converge on the same per-request
            # advance.
            cache_manager.set_custom_pos_advance(mrope_pos_advance, label="main")

        return {
            "input_embeds": input_embeds,
            "cos_3d": cos_3d,
            "sin_3d": sin_3d,
            "mrope_section": self.MROPE_SECTION,
            "seq_lens": seq_lens,
            "masks_for_talker": {
                rid: inp.tensor_inputs.get("masks_for_talker") \
                    for (rid, inp) in zip(engine_inputs.request_ids, inputs, strict=True)
            },
            **extra_inputs
        }

    # ---- forward ----

    def _collect_deepstack_kwargs(
        self, kwargs: dict[str, Any]
    ) -> list[torch.Tensor] | None:
        """Reassemble the deepstack list from per-layer ``deepstack_<i>`` keys.

        ``preprocess`` emits one key per ``vision.deepstack_visual_indexes``
        entry so each layer's visual feature gets its own static buffer in
        the captured prefill_vision config. Both the eager ``forward`` and
        the captured ``forward_batched`` call this helper to put the list
        back together before invoking ``Qwen3OmniThinkerModel.forward``.

        Returns None when no ``deepstack_*`` keys are present (e.g. for
        prefill_text / prefill_audio / thinker_decode), so the inner model
        receives ``deepstack_visual_embeds=None`` and skips the deepstack
        splice entirely.
        """
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)
        if not any(f"deepstack_{i}" in kwargs for i in range(num_deepstack)):
            return None
        out: list[torch.Tensor] = []
        for i in range(num_deepstack):
            t = kwargs.get(f"deepstack_{i}")
            if t is None:
                return None
            out.append(t)
        return out

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        mrope_section: list[int] | None = None,
        mrope_pos_advance: list[int] | None = None,
        masks_for_talker: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Run Thinker transformer, produce logits (decode) and thinker_states.

        ``thinker_states`` is only emitted when audio output is requested
        (checked via ``request_info.step_metadata["audio_output"]``). This
        saves cross-partition bandwidth for text-only requests. Defaults to
        ``True`` for backwards compatibility with callers that do not set
        the flag (e.g. unit tests).

        ``deepstack`` (used by prefill_vision) is reassembled from per-layer
        ``deepstack_<i>`` kwargs — see ``_collect_deepstack_kwargs``.
        """
        deepstack = self._collect_deepstack_kwargs(kwargs)
        request_info = engine_inputs.single_request_info
        audio_output = request_info.step_metadata.get(
            "audio_output", True,
        )

        cos_sin_3d = (cos_3d, sin_3d) if cos_3d is not None else None

        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cache_handle=engine_inputs.cache_manager,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
            mrope_pos_advance=mrope_pos_advance,
            deepstack_visual_embeds=deepstack,
        )

        result: NameToTensorList = {}

        # Decode: produce logits for text token sampling
        if graph_walk == "thinker_decode" or request_info.step_metadata.get("is_last_prefill", False):
            logits = self.model.lm_head(hidden[-1:, :])
            result["logits"] = [logits]

        # Pack thinker_states for Talker conditioning ONLY when audio output
        # is requested.  For text-only requests we skip this entirely to
        # avoid sending hidden states the Talker will never consume.
        if audio_output:
            # Concatenate layer-0 embeddings and layer-N hidden states along
            # last dim -> (tokens, 2 * hidden_size)
            if layer_n_hidden is not None:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_n_hidden], dim=-1,
                )
            else:
                # Fallback: use layer_0_embed doubled (shouldn't happen in
                # practice)
                thinker_states = torch.cat(
                    [layer_0_embed, layer_0_embed], dim=-1,
                )
            result["thinker_states"] = [thinker_states]
            result["thinker_mask"] = [next(iter(masks_for_talker.values()))] \
                if masks_for_talker else []
        return result

    # ---- batching ----
    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        return len(model_inputs) > 1

    PREFILL_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]

    @staticmethod
    def _env_buckets(name: str, default: list) -> list:
        """Triage-server capture trimming (task: warmup acceleration).
        MSTAR_DECODE_BUCKETS / MSTAR_PREFILL_BUCKETS = comma ints override
        the capture grids. Uncaptured sizes pad UP to the next captured
        bucket (existing lookup semantics), so a trimmed server is correct
        for any cell — just wasteful outside its intended sizes. NEVER use
        for committed sweeps; startup measured 204s -> ~150s with
        24,28,32 / 256,512 for an i2t-triage server (34 Thinker captures
        was the 70s startup tail)."""
        import os as _os
        raw = _os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            vals = sorted({int(x) for x in raw.split(",") if x.strip()})
        except ValueError:
            return default
        if not vals or max(vals) < max(default):
            # the largest bucket is load-bearing (everything pads up to it) —
            # never let an override SHRINK the ceiling. RAISING it is allowed
            # (packed-prefill scaling: a 4096/8192 top bucket lets bs=16/32
            # image prefills fit one captured forward); lookups still pad up,
            # so a larger ceiling is correctness-neutral, just more capture
            # memory.
            return default
        return vals

    @staticmethod
    def _env_batch_sizes(name: str, default: list) -> list:
        """Override a prefill capture_batch_sizes grid via env (comma ints).
        Experimental: push i2t prefill batching past bs=4 to batch more images'
        prefills per forward (vLLM-style). Each added size captures more graphs
        = more memory (OOM risk on the 30B Thinker); expand incrementally.
        Empty/unset -> default."""
        import os as _os
        raw = _os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            vals = sorted({int(x) for x in raw.split(",") if x.strip()})
        except ValueError:
            return default
        return vals or default

    # Mixed prefill+decode CAPTURED batch (MSTAR_MIXED_BATCH). Each bucket
    # is (padded_bs, total_tokens) where total_tokens = padded_bs decode-row
    # tokens (1 each) + one prefill-chunk row of C tokens minus the one row the
    # chunk occupies. Concretely: a mixed step has N decode rows + 1 chunk row,
    # padded to bs=32, so the row count is fixed at 32 and total_tokens =
    # (N decode tokens) + C. With N up to 31 and the chunk row replacing the
    # 32nd, the captured token bucket is 32 + C for the default chunk sizes:
    #   C=256 -> 288,  C=512 -> 544.
    # Padding to bs=32 contributes zero-length rows (input_seq_len=0), so a
    # step with fewer decode rows still lands on the same bucket; the packed
    # path walks only real_num_tokens (see _run_flashinfer_packed). Growing the
    # (bs, C) grid further is a separate lever from the fixed set below.
    MIXED_BATCH_BS = 32
    # 288 covers tail-merged chunks (C=256+tail<=32, e.g. the ubiquitous
    # 258-token vision span): 31 decodes + 258 = 289 tokens would otherwise
    # overflow the 288 bucket by ONE token and pad all the way to 544.
    _MIXED_BATCH_CHUNK_SIZES_DEFAULT = [256, 288, 512]

    @property
    def MIXED_BATCH_CHUNK_SIZES(self) -> list[int]:
        """Boot-time capture grid for the ``thinker_mixed`` chunk row C
        (``MSTAR_MIXED_CHUNK_SIZES``, comma ints, e.g. "256,288,512,1024,2048").

        The mixed step is a captured CUDA graph whose only chunk buckets are
        this list, so a chunk larger than ``max(MIXED_BATCH_CHUNK_SIZES)`` can
        never fold into a decode step no matter how large the admission budget
        is — routing it into an uncaptured shape is an illegal memory access
        on an uncaptured bucket. Raising the ceiling here is the only way past
        it.

        Parsed values are UNIONed with the default set, never replacing it —
        an override can only GROW the grid. This protects the 288 bucket
        (the fix for the 258-token vision-span tail-merge case)
        from being silently dropped by a caller who only meant to add a
        bigger bucket on top. Each added size is one more capture in
        ``get_cuda_graph_configs`` below (one more (bs=32, num_tokens) shape
        for the CUDA-graph runner to warm up), so grow it deliberately: more
        buckets means more GPU memory and longer boot time.

        Unset/empty/unparseable -> the untouched default (byte-identical).
        Boot-time only: capture happens once at process start, so this is
        not a dynflag — changing it requires a reboot.
        """
        raw = os.environ.get("MSTAR_MIXED_CHUNK_SIZES", "").strip()
        if not raw:
            return self._MIXED_BATCH_CHUNK_SIZES_DEFAULT
        try:
            vals = {int(x) for x in raw.split(",") if x.strip()}
        except ValueError:
            return self._MIXED_BATCH_CHUNK_SIZES_DEFAULT
        vals = {v for v in vals if v > 0}
        if not vals:
            return self._MIXED_BATCH_CHUNK_SIZES_DEFAULT
        return sorted(vals | set(self._MIXED_BATCH_CHUNK_SIZES_DEFAULT))

    # prefill_vision buckets are larger than text/audio because video
    # produces many vision tokens per request (UCF101 ≈ 1k–4k tokens; 8192
    # gives headroom for VideoMME-style longer clips). Capture only bs=1
    # because mstar's eager prefill_vision asserts a single request per step
    # (``preprocess`` line in this file), and V2T runs at concurrency 1
    # today. Costs ~4 captures × persistent FlashInfer wrappers + static
    # buffers for the 30B Thinker; revisit if memory becomes a constraint.
    _PREFILL_VISION_TOKEN_BUCKETS_BASE = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # Vision-token counts pad up to the smallest captured bucket. MSTAR_VISION_GRAPH_ALIGN=1
    # adds intermediate low-range buckets so a 258-token image pads to 320 (~24% slack)
    # instead of 512 (~99%), at the cost of a few extra bs=1 captures.
    _PREFILL_VISION_TOKEN_BUCKETS_ALIGNED = [
        128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384,
    ]
    PREFILL_VISION_CAPTURE_BATCH_SIZES = [1]
    PREFILL_VISION_BATCH_CAPTURE_BATCH_SIZES = [1, 2, 4]

    @property
    def PREFILL_VISION_TOKEN_BUCKETS(self) -> list[int]:
        if os.environ.get("MSTAR_VISION_GRAPH_ALIGN", "0").strip().lower() in ("1", "true", "yes", "on"):
            return self._PREFILL_VISION_TOKEN_BUCKETS_ALIGNED
        return self._PREFILL_VISION_TOKEN_BUCKETS_BASE

    def _build_prefill_text_packed(
        self, num_tokens: int, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Synthesize a tensor-only post-preprocess packed dict for capture.

        Produced inputs match ``preprocess(graph_walk="prefill_text")`` for the
        tensor entries the model forward actually reads (``input_embeds``,
        ``cos_3d``, ``sin_3d``). Non-tensor entries (``mrope_section``,
        ``seq_lens``, ``masks_for_talker``) are intentionally absent — the
        runner's static-buffer interning is tensor-only by design (non-tensor
        entries are model-static and don't need a per-bucket buffer), so
        ``forward_batched`` recovers ``mrope_section`` from a class constant
        and reads token boundaries from ``cache_manager.get_qo_indptr_buf``
        instead. Per-token cos/sin values come from running the real RoPE
        math on a sequential dummy position (3 components × num_tokens) so
        the captured kernels see non-degenerate inputs at trace time.
        """
        hidden_size = self.config.thinker_hidden_size
        # 3-row position grid (temporal, height, width) — same shape the eager
        # path passes to compute_3d_cos_sin via prepare_inputs/preprocess.
        pos_ids = torch.arange(
            num_tokens, dtype=torch.float, device=device,
        ).unsqueeze(0).expand(3, -1).contiguous()
        inv_freq = self._get_inv_freq(device)
        cos_3d, sin_3d = compute_3d_cos_sin(
            pos_ids, inv_freq,
            mrope_section=self.MROPE_SECTION,
            target_dtype=torch.bfloat16,
        )
        return {
            "input_embeds": torch.zeros(
                (num_tokens, hidden_size),
                dtype=torch.bfloat16, device=device,
            ),
            "cos_3d": cos_3d,
            "sin_3d": sin_3d,
        }

    def _build_prefill_vision_packed(
        self, num_tokens: int, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Synthesize a tensor-only post-preprocess packed dict for prefill_vision.

        Mirrors ``_build_prefill_text_packed`` and additionally provides
        ``deepstack_<i>`` (one tensor per ``vision.deepstack_visual_indexes``
        entry). Non-tensor extras (``mrope_section``, ``seq_lens``,
        ``mrope_pos_advance``, ``masks_for_talker``) are intentionally absent:
        the runner's static-buffer interning is tensor-only, so non-tensors
        come back from ``submodule.preprocess`` at replay time. ``mrope_pos_advance``
        flows through the ``_PlanState`` side-channel (see ``cache_manager._PlanState``).

        Visual_pos_masks is a length-``num_tokens`` bool tensor; at capture
        time we set it to all-False so the inner ``_deepstack_process``
        becomes a no-op on the captured-bucket trailing slack tokens. At
        replay, ``preprocess`` writes the real per-request mask (which has
        True at the per-frame token positions and False at the sentinel /
        padding positions).
        """
        hidden_size = self.config.thinker_hidden_size
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)

        pos_ids = torch.arange(
            num_tokens, dtype=torch.float, device=device,
        ).unsqueeze(0).expand(3, -1).contiguous()
        inv_freq = self._get_inv_freq(device)
        cos_3d, sin_3d = compute_3d_cos_sin(
            pos_ids, inv_freq,
            mrope_section=self.MROPE_SECTION,
            target_dtype=torch.bfloat16,
        )

        packed: dict[str, torch.Tensor] = {
            "input_embeds": torch.zeros(
                (num_tokens, hidden_size),
                dtype=torch.bfloat16, device=device,
            ),
            "cos_3d": cos_3d,
            "sin_3d": sin_3d,
        }
        # Per-layer deepstack tensors. Each has shape (num_tokens, hidden) and
        # is summed into the post-attention hidden state at the matching
        # ``deepstack_visual_indexes`` layer.
        for i in range(num_deepstack):
            packed[f"deepstack_{i}"] = torch.zeros(
                (num_tokens, hidden_size),
                dtype=torch.bfloat16, device=device,
            )
        return packed

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1):
        """Declare CUDA graph captures for ``thinker_decode`` and the prefill walks.

        Decode uses ``BasicBatchedCudaGraphConfig`` (one capture per bs;
        runner clones single_request_inputs and runs preprocess itself).
        Prefill uses ``FlashInferPackedCudaGraphConfig`` (one capture per
        (bs, num_tokens) bucket; the dict here IS the post-preprocess
        packed input — runner does not call preprocess at capture).

        ``prefill_text`` and ``prefill_audio`` share an identical post-preprocess
        tensor shape and ``forward_batched`` dispatch, so each walk gets its own
        bucketed capture (separate ``capture_graph_walk`` so the runner re-plans
        attention/RoPE on the right walk at replay).

        ``capture_batch_sizes`` is kept small for both because each capture
        allocates persistent FlashInfer wrappers + static buffers for the
        full 30B Thinker; revisit after profiling real deployments.
        """
        prefill_text_packed = {
            num_tokens: self._build_prefill_text_packed(num_tokens, device)
            for num_tokens in self._env_buckets(
                "MSTAR_PREFILL_BUCKETS", self.PREFILL_TOKEN_BUCKETS
            )
        }
        prefill_vision_packed = {
            num_tokens: self._build_prefill_vision_packed(num_tokens, device)
            for num_tokens in self.PREFILL_VISION_TOKEN_BUCKETS
        }
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            batch_vision_prefill_enabled,
            mark_mixed_vision_provisioned,
            mixed_batch_enabled,
            mixed_batch_vision_enabled,
        )
        prefill_vision_capture_bs = (
            self._env_batch_sizes(
                "MSTAR_VIS_BATCH_SIZES", self.PREFILL_VISION_BATCH_CAPTURE_BATCH_SIZES
            )
            if batch_vision_prefill_enabled()
            else self.PREFILL_VISION_CAPTURE_BATCH_SIZES
        )
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)

        prefill_vision_zero_padding_tensor_inputs = {
            "masks_for_talker": torch.zeros(
                (2, 0), dtype=torch.float, device=device,
            ),
        }
        for i in range(num_deepstack):
            prefill_vision_zero_padding_tensor_inputs[f"deepstack_{i}"] = (
                torch.zeros(
                    (0, self.config.thinker_hidden_size),
                    dtype=torch.bfloat16, device=device,
                )
            )
        configs = [
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="thinker_decode",
                requires_cfg=False,
                labels=["main"],
                single_request_inputs=ARNodeInputs(
                    input_seq_len=1,
                    input_embeds=torch.zeros(
                        (1, self.config.thinker_hidden_size),
                        device=device, dtype=torch.bfloat16
                    ),
                    custom_pos_ids=torch.tensor(
                        [[0], [0], [0]],
                        dtype=torch.float,
                        device=device,
                    ),
                    tensor_inputs={
                        "masks_for_talker": self._get_decode_thinker_mask(device)
                    }
                ),
                compile=True,
                # Denser high-end buckets: at B32 closed loop the live batch
                # churns through 17-31 as requests finish/join; without 24/28
                # every such step pads to 32 (up to ~2x wasted decode compute
                # on the padded rows at the low end of the bucket).
                capture_batch_sizes=self._env_buckets(
                    "MSTAR_DECODE_BUCKETS", [1, 2, 4, 8, 16, 24, 28, 32]
                ),
            ),
            FlashInferPackedCudaGraphConfig(
                capture_graph_walk="prefill_text",
                # prefill_multimodal_audio (MSTAR_MERGED_PREFILL_AUDIO) merges a
                # text + audio span into one Thinker forward. Audio carries no
                # deepstack and no custom MRoPE advance (positions +1/token), so
                # the merged span's post-preprocess signature is identical to
                # prefill_text/audio (input_embeds + cos_3d + sin_3d +
                # masks_for_talker) — it replays on THIS capture, not the vision
                # one. Harmless to list unconditionally: the walk is only ever
                # scheduled when the flag registers it. The merged span pads up
                # into the same text/audio token buckets.
                replay_graph_walks=[
                    "prefill_text", "prefill_audio", "prefill_multimodal_audio",
                ],
                packed_seq_len_to_inputs=prefill_text_packed,
                requires_cfg=False,
                labels=["main"],
                compile=True,
                causal_attention=True,
                capture_batch_sizes=self._env_batch_sizes(
                    "MSTAR_PREFILL_BATCH_SIZES", self.PREFILL_CAPTURE_BATCH_SIZES
                ),
                zero_padding_input=ARNodeInputs(
                    input_seq_len=0,
                    input_embeds=torch.zeros(
                        (0, self.config.thinker_hidden_size),
                        device=device, dtype=torch.bfloat16
                    ),
                    custom_pos_ids=torch.zeros(
                        (3, 0),
                        dtype=torch.float,
                        device=device,
                    ),
                    tensor_inputs={
                        "masks_for_talker": torch.zeros(
                            (2, 0),
                            dtype=torch.float,
                            device=device,
                        )
                    }
                ),
            ),
            # prefill_vision: separate capture because its post-preprocess
            # tensor signature has extras (deepstack_<i>)
            # that prefill_text/audio don't. mrope_pos_advance flows
            # out-of-band via ``BatchedCacheManager.set_custom_pos_advance``
            # — see ``cache_manager._PlanState.custom_pos_advance``.
            FlashInferPackedCudaGraphConfig(
                capture_graph_walk="prefill_vision",
                # prefill_multimodal (MSTAR_MERGED_PREFILL) has the identical
                # post-preprocess signature (input_embeds + cos_3d + sin_3d +
                # per-layer deepstack_<i>; mrope_pos_advance side-channel), so it
                # replays on this capture — no separate capture, no extra warmup.
                # Harmless to list unconditionally: the walk is only ever
                # scheduled when the flag registers it. The merged span (text +
                # vision) pads up into the same vision token buckets.
                replay_graph_walks=["prefill_vision", "prefill_multimodal"],
                packed_seq_len_to_inputs=prefill_vision_packed,
                requires_cfg=False,
                labels=["main"],
                compile=True,
                causal_attention=True,
                capture_batch_sizes=prefill_vision_capture_bs,
                zero_padding_input=ARNodeInputs(
                    input_seq_len=0,
                    input_embeds=torch.zeros(
                        (0, self.config.thinker_hidden_size),
                        device=device, dtype=torch.bfloat16,
                    ),
                    custom_pos_ids=torch.zeros(
                        (3, 0),
                        dtype=torch.float,
                        device=device,
                    ),
                    tensor_inputs=prefill_vision_zero_padding_tensor_inputs,
                    kwargs={"mrope_pos_advance": 0},
                ),
            ),
        ]

        # Mixed prefill+decode CAPTURED batch (MSTAR_MIXED_BATCH). Only
        # register the capture when the flag is ON so flag-off is byte-identical
        # (no extra captures, no ``thinker_mixed`` graph in the runner's table,
        # so the scheduler could never route to it even if it tried).
        #
        # The post-preprocess tensor signature of a text-only mixed batch is
        # IDENTICAL to prefill_text/audio (``input_embeds`` + ``cos_3d`` +
        # ``sin_3d``): the decode rows and a ``prefill_text`` chunk row both
        # contribute only these three packed tensors after ``preprocess``
        # concatenates them, so we synthesize the capture inputs with the same
        # ``_build_prefill_text_packed`` helper. The token buckets are
        # ``MIXED_BATCH_BS + C`` (see MIXED_BATCH_* above).
        #
        # MSTAR_MIXED_BATCH_VISION: to let a ``prefill_vision``
        # chunk ride the mixed step, the SAME captured bucket must additionally
        # carry per-layer ``deepstack_<i>`` statics (shape (num_tokens, hidden))
        # so ``preprocess``'s replay-time deepstack assembly has a static buffer
        # to copy into. Because ``static_input_keys`` is fixed at capture time,
        # the deepstack statics must be present on the ONE thinker_mixed capture
        # regardless of whether a given replay's chunk row is text or vision; a
        # text-chunk mixed step simply fills those buffers with zeros (a no-op
        # additive splice, see thinker._deepstack_process). So when the vision
        # flag is on we build the mixed buckets with the vision packed builder
        # (adds deepstack_<i>) and the vision-shaped zero_padding_input.
        # Illegal-memory-access safety: record the ONE capture's ground truth so the
        # scheduler routes a prefill_vision chunk into thinker_mixed ONLY when the
        # deepstack statics were actually baked in here. False whenever no
        # thinker_mixed graph exists (mixed_batch off) or it is the text-only
        # signature (mixed_batch_vision off) — either way a live-flag flip after
        # boot can never route to an unprovisioned graph. See
        # qwen3_omni_model.mixed_vision_capture_provisioned.
        mixed_vision_provisioned = (
            mixed_batch_enabled() and mixed_batch_vision_enabled()
        )
        mark_mixed_vision_provisioned(mixed_vision_provisioned)
        if mixed_batch_enabled():
            mixed_vision = mixed_vision_provisioned
            build_mixed_packed = (
                self._build_prefill_vision_packed
                if mixed_vision
                else self._build_prefill_text_packed
            )
            mixed_packed = {
                self.MIXED_BATCH_BS + c: build_mixed_packed(
                    self.MIXED_BATCH_BS + c, device,
                )
                for c in self.MIXED_BATCH_CHUNK_SIZES
            }
            # Zero-length padding row. Its tensor_inputs must declare the same
            # keys the real chunk row can carry so ``preprocess``'s per-row
            # concat sees consistent shapes; deepstack rows are zero-filled by
            # ``preprocess`` when absent, but padding rows still get explicit
            # (0, hidden) deepstack slices + the mrope_pos_advance kwarg under
            # the vision flag to mirror the prefill_vision padding contract.
            mixed_zero_padding_tensor_inputs: dict[str, torch.Tensor] = {
                "masks_for_talker": torch.zeros(
                    (2, 0), dtype=torch.float, device=device,
                ),
            }
            mixed_zero_padding_kwargs: dict[str, Any] = {}
            if mixed_vision:
                for i in range(num_deepstack):
                    mixed_zero_padding_tensor_inputs[f"deepstack_{i}"] = (
                        torch.zeros(
                            (0, self.config.thinker_hidden_size),
                            dtype=torch.bfloat16, device=device,
                        )
                    )
                mixed_zero_padding_kwargs["mrope_pos_advance"] = 0
            configs.append(
                FlashInferPackedCudaGraphConfig(
                    capture_graph_walk="thinker_mixed",
                    replay_graph_walks=["thinker_mixed"],
                    packed_seq_len_to_inputs=mixed_packed,
                    requires_cfg=False,
                    labels=["main"],
                    compile=True,
                    causal_attention=True,
                    capture_batch_sizes=[self.MIXED_BATCH_BS],
                    zero_padding_input=ARNodeInputs(
                        input_seq_len=0,
                        input_embeds=torch.zeros(
                            (0, self.config.thinker_hidden_size),
                            device=device, dtype=torch.bfloat16,
                        ),
                        custom_pos_ids=torch.zeros(
                            (3, 0),
                            dtype=torch.float,
                            device=device,
                        ),
                        tensor_inputs=mixed_zero_padding_tensor_inputs,
                        kwargs=mixed_zero_padding_kwargs,
                    ),
                )
            )
        return configs

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        mrope_section: list[int] | None = None,
        mrope_pos_advance: list[int] | None = None,
        masks_for_talker: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Batched Thinker forward shared between ``thinker_decode`` and the prefill walks.

        Decode path (1 token per request, ``hidden`` shape ``(bs, hidden)``):
          Always packs ``thinker_states`` + ``thinker_mask`` in every per-rid
          output dict so the captured CUDA graph has a static output shape
          regardless of request metadata. Per-rid filtering (dropping
          ``thinker_states`` / ``thinker_mask`` for ``audio_output=False``
          requests) happens OUTSIDE the captured region via
          ``filter_batched_output``.

        Prefill paths (``prefill_text``, ``prefill_audio``, ``prefill_vision``;
        multi-token-per-request, ``hidden`` shape ``(total_tokens, hidden)``):
          Last-token-per-request indices come from the persistent
          ``qo_indptr_buf`` on the FlashInfer prefill wrapper — the buffer is
          updated via ``.copy_()`` by ``plan_attention`` outside the captured
          graph, so its address stays stable across replay and the captured
          indexing op picks up real values. Emits packed sentinels only:
          ``__batched_logits__`` (last-token-per-request, ``(padded_bs, V)``)
          and ``__batched_thinker_states__`` (full packed
          ``(total_tokens, 2*hidden)`` for downstream Talker conditioning).
          Per-rid slicing of thinker_states + reattaching real per-token
          masks happens post-replay in ``unpack_packed_outputs`` because the
          slice ends depend on real per-request seq_lens, which the
          captured region cannot honor with fixed shapes.

          ``prefill_text`` / ``prefill_audio`` share one capture (their
          post-preprocess tensor signature is identical: ``input_embeds`` +
          ``cos_3d`` + ``sin_3d``). ``prefill_vision`` has its own capture
          because it adds  per-layer ``deepstack_<i>``
          tensors. ``mrope_pos_advance`` flows out-of-band via
          ``BatchedCacheManager.set_custom_pos_advance``, which
          ``preprocess`` populates — see
          ``cache_manager._PlanState.custom_pos_advance``. The model's inner
          ``cache_handle.advance_seq_lens(pos_id_ns=mrope_pos_advance)`` call
          executes only at capture time (it's a ``@torch.compiler.disable``'d
          Python op so it's not replayed); the runner's post-replay
          ``advance_seq_lens()`` is what advances the real state, and that
          path reads the side-channel.
        """

        # Packed dict from FlashInferPackedCudaGraphConfig is tensor-only by
        # design (the runner's static-buffer interning skips non-tensor
        # entries), so for prefill walks we recover mrope_section from the
        # class constant when the kwarg is missing. Decode goes through
        # preprocess which does pass it explicitly.
        # ``thinker_mixed`` is packed exactly like a prefill walk: the
        # forward runs over ``total_tokens = n + C`` packed rows and the
        # per-request last-token logits are gathered via ``qo_indptr[1:]-1``
        # (decode rows: their single token; chunk row: its last token). So it
        # takes the ``is_prefill`` packed-output branch below, emitting
        # ``__batched_logits__`` (n+1 rows) + ``__batched_thinker_states__``.
        is_prefill = graph_walk in (
            "prefill_text", "prefill_audio", "prefill_vision", "thinker_mixed",
            "prefill_multimodal", "prefill_multimodal_audio",
        )
        if mrope_section is None and is_prefill:
            mrope_section = self.MROPE_SECTION

        # prefill_vision: reassemble per-layer deepstack tensors from
        # ``deepstack_<i>`` kwargs into the list shape ``model.forward``
        # expects. None for non-vision walks → model skips the splice.
        deepstack = self._collect_deepstack_kwargs(kwargs)

        cos_sin_3d = (cos_3d, sin_3d) if cos_3d is not None else None
        cache_manager = engine_inputs.cache_manager
        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cache_handle=cache_manager,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
            mrope_pos_advance=mrope_pos_advance,
            deepstack_visual_embeds=deepstack,
        )

        if is_prefill:
            qo_indptr_buf = cache_manager.get_qo_indptr_buf("main")
            assert qo_indptr_buf is not None, (
                f"{graph_walk} forward_batched requires a properly initialized "
                "FlashInferPrefillWrapper (qo_indptr static buffer); got None."
            )
            last_token_indices = (qo_indptr_buf[1:] - 1).long()  # (padded_bs,)
            last_hidden = hidden.index_select(0, last_token_indices)
            logits = self.model.lm_head(last_hidden)  # (padded_bs, vocab)
            if layer_n_hidden is not None:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_n_hidden], dim=-1,
                )  # (total_tokens, 2*hidden)
            else:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_0_embed], dim=-1,
                )

            return {
                "__batched_logits__": logits,
                "__batched_thinker_states__": thinker_states,
            }

        # thinker_decode (existing behavior)
        logits = self.model.lm_head(hidden)  # (batch, vocab)

        # Always pack thinker_states once for the whole batch.  The
        # per-rid ``audio_output`` gating happens outside this function
        # via ``filter_batched_output`` so the captured graph's output
        # shape stays static.  The extra cat is O(tokens * hidden) and
        # negligible next to the transformer cost.
        if layer_n_hidden is not None:
            thinker_states = torch.cat(
                [layer_0_embed, layer_n_hidden], dim=-1,
            )
        else:
            thinker_states = torch.cat(
                [layer_0_embed, layer_0_embed], dim=-1,
            )

        request_ids = cache_manager.request_ids
        outputs: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(request_ids):
            req_out: NameToTensorList = {
                "logits": [logits[i : i + 1]],
                "thinker_states": [thinker_states[i : i + 1]],
            }
            if masks_for_talker is not None and rid in masks_for_talker:
                req_out["thinker_mask"] = [masks_for_talker[rid]]
            outputs[rid] = req_out
        # Expose the stacked [B, V] tensor under a sentinel key so the CUDA
        # graph runner can sample directly without concatenating per-rid slices.
        outputs["__batched_logits__"] = logits
        return outputs

    def unpack_packed_outputs(
        self,
        static_output: dict,
        request_ids: list[str],
        real_seq_lens: list[int],
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, "CurrentForwardPassInfo"],
    ) -> dict[str, dict[str, list[torch.Tensor]]]:
        """Slice the packed ``__batched_thinker_states__`` per real seq_len.

        Captured forward emits the full ``(total_tokens, 2*hidden)`` packed
        tensor; here we cut it at the real per-request token boundaries and
        reattach the per-request talker masks, which live on the original
        ARNodeInputs (the captured graph never saw them — masks vary in
        shape with text content). Drops per-rid emission for requests with
        ``audio_output=False``, mirroring ``filter_batched_output``'s gating
        for the decode path.
        """
        packed_states = static_output.get("__batched_thinker_states__")
        if packed_states is None:
            return {}

        out: dict[str, dict[str, list[torch.Tensor]]] = {}
        cum = 0
        for i, rid in enumerate(request_ids):
            sl = real_seq_lens[i]
            slice_start, slice_end = cum, cum + sl
            cum = slice_end

            info = per_request_info.get(rid) if per_request_info else None
            if info is not None and not info.step_metadata.get("audio_output", True):
                continue

            ts_slice = packed_states[slice_start:slice_end].clone()
            rid_out: dict[str, list[torch.Tensor]] = {
                "thinker_states": [ts_slice],
            }
            mask = inputs[i].tensor_inputs.get("masks_for_talker")
            if mask is not None:
                rid_out["thinker_mask"] = [mask]
            out[rid] = rid_out
        return out

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str]:
        return ["main"]

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        return_token = request_info.graph_walk == "thinker_decode" or \
            request_info.step_metadata.get("is_last_prefill", False)
        if not return_token:
            outputs.pop("new_token", None)

        if not request_info.step_metadata.get("audio_output", True):
            # drop thinker_states and thinker_mask
            outputs.pop("thinker_states", None)
            outputs.pop("thinker_mask", None)
        else:
            # Pick layer-0 for text positions and layer-N for multimodal,
            # drop positions the Talker won't consume (system prompt /
            # previous-assistant), and emit a 1D multimodal mask aligned to
            # the kept tokens. Doing this here (not in the Talker) avoids
            # shipping both hidden states for every token.
            ts_list = outputs.get("thinker_states")
            mask_list = outputs.get("thinker_mask")
            if ts_list and mask_list:
                ts = ts_list[0]                # (seq_len, 2*hidden)
                mask = mask_list[0]            # (2, seq_len)
                thinker_hidden = self.config.thinker_hidden_size
                layer_0 = ts[..., :thinker_hidden]
                layer_n = ts[..., thinker_hidden:]
                multimodal_mask = mask[0].bool()
                text_inclusion_mask = mask[1].bool()
                text_mask = text_inclusion_mask & ~multimodal_mask
                inclusion_mask = text_mask | multimodal_mask

                selected = torch.where(
                    multimodal_mask.unsqueeze(-1), layer_n, layer_0,
                )                              # (seq_len, hidden)
                outputs["thinker_states"] = [selected[inclusion_mask]]
                outputs["thinker_mask"] = [multimodal_mask[inclusion_mask]]

        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs or request_info.graph_walk != "thinker_decode":
            return set()
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.sampling_config["Thinker"].ignore_eos
        eos_token_id = self.config.im_end_token_id
        if (not ignore_eos and eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("thinker_decode_loop", 0) + 1 >= request_info.max_tokens):
            return {"thinker_decode_loop"}
        return set()

    def filter_batched_output(
        self,
        request_info: CurrentForwardPassInfo,
        rid_output: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        """Drop ``thinker_states`` + ``thinker_mask`` for text-only requests.

        ``forward_batched`` always emits these keys so the captured CUDA
        graph's output shape is static.  Here, outside the captured
        region, we gate them on the real request's ``audio_output`` flag
        so the Talker edge stays unrouted for text-only requests (matches
        the pre-capture eager-mode behaviour).
        """
        if request_info is None:
            return rid_output
        if request_info.step_metadata.get("audio_output", True):
            return rid_output
        return {
            k: v for k, v in rid_output.items()
            if k not in ("thinker_states", "thinker_mask")
        }


# ===================================================================
# 4. TalkerSubmodule (ar engine) -- SECOND MOST COMPLEX
# ===================================================================

class TalkerSubmodule(ARNodeSubmodule):
    MAX_BATCH_SIZE = 32

    def __init__(
        self,
        talker_model: Qwen3OmniTalkerModel,
        code_predictor: Qwen3OmniCodePredictor,
        config: Qwen3OmniModelConfig
    ):
        super().__init__()
        self.model = talker_model
        self.code_predictor = code_predictor
        self.talker_code_emb = self.model.model.codec_embedding
        self.config = config
        self.cp_cfg = config.code_predictor
        self.num_codes = self.cp_cfg.num_code_groups

        # Pre-computed TTS special inputs.
        self._tts_pad_embed_cached: torch.Tensor | None = None
        self._tts_bos_embed_cached: torch.Tensor | None = None
        self._tts_eos_embed_cached: torch.Tensor | None = None

        # Lazy-built suppress mask for layer-0 logits.  Shape (vocab_size,)
        # with True at positions to suppress.  Cached on first forward.
        self._suppress_mask: torch.Tensor | None = None

        # Per-request flag: whether we've already sent tts_eos_embed as
        # the text conditioning for this request. We use this flag to
        # inject tts_eos_embed for ONE step before falling back to pad.
        self._eos_embed_sent: set[str] = set()

        # TODO: this is hacky; when we have time, refactor it to make this
        # come from the engine
        self._cp_kv_cache: torch.Tensor | None = None

    def _get_cp_kv_cache(self):
        if self._cp_kv_cache is None:
            self._cp_kv_cache = torch.zeros((
                    self.cp_cfg.num_hidden_layers,
                    self.MAX_BATCH_SIZE, 2, self.num_codes,
                    self.cp_cfg.num_key_value_heads,
                    self.cp_cfg.head_dim
                ), dtype=self.talker_code_emb.weight.dtype,
                device=self.get_device(),
            )
        return self._cp_kv_cache

    def init_tts_embeds(self, thinker_embed_tokens: nn.Embedding) -> None:
        """Pre-compute TTS pad/bos/eos hidden states using the Thinker's
        embedding table + Talker text_projection

        Must be called after both the Thinker and Talker weights are loaded
        (only applicable when both reside on the same worker).  When they
        are on different workers, these embeddings should be transferred as
        constant tensors during model init.
        """
        device = next(self.model.parameters()).device
        # The Thinker embedding table may be fp32 while the Talker runs in
        # autocast dtype; match text_projection's weight dtype before projecting.
        proj_dtype = self.model.text_projection.linear_fc1.weight.dtype
        with torch.no_grad():
            pad_raw = thinker_embed_tokens(
                torch.tensor([self.config.tts_pad_token_id], device=device)
            ).to(proj_dtype)
            bos_raw = thinker_embed_tokens(
                torch.tensor([self.config.tts_bos_token_id], device=device)
            ).to(proj_dtype)
            eos_raw = thinker_embed_tokens(
                torch.tensor([self.config.tts_eos_token_id], device=device)
            ).to(proj_dtype)
            self._tts_pad_embed_cached = self.model.text_projection(pad_raw).squeeze(0)
            self._tts_bos_embed_cached = self.model.text_projection(bos_raw).squeeze(0)
            self._tts_eos_embed_cached = self.model.text_projection(eos_raw).squeeze(0)

        logger.info(
            "TalkerSubmodule: pre-computed TTS special embeddings via "
            "Thinker embed_tokens + Talker text_projection"
        )

    def _get_suppress_mask(self) -> torch.Tensor:
        """Return the bool mask of layer-0 logits to set to -inf.

        Matches HF's ``talker_supppressed_tokens`` list: suppress the top
        1024 IDs of the Talker vocab (the "special token" region) EXCEPT
        ``codec_eos_token_id``, which is the valid stop signal.
        """
        if  self._suppress_mask is None:
            device = next(self.model.parameters()).device
            vocab_size = self.config.talker_text.vocab_size
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            start = vocab_size - 1024
            start = max(start, 0)
            mask[start:vocab_size] = True
            # Do not suppress codec_eos (the valid stop signal).
            eos = self.config.talker.codec_eos_token_id
            if 0 <= eos < vocab_size:
                mask[eos] = False
            self._suppress_mask = mask
        return self._suppress_mask

    def _get_talker_embeds(
        self, selected_hidden: torch.Tensor, multimodal_mask: torch.Tensor,
    ):
        # ``selected_hidden`` is layer-0 embed for text positions and
        # layer-N hidden for multimodal positions (the Thinker's
        # postprocess already did that selection and dropped positions
        # the Talker won't consume). ``multimodal_mask`` is 1D, aligned
        # to the kept tokens, and tells us which projection to apply.
        text_mask = ~multimodal_mask
        projected = torch.empty(
            selected_hidden.shape[0], self.config.talker_hidden_size,
            device=selected_hidden.device, dtype=selected_hidden.dtype,
        )
        if text_mask.any():
            projected[text_mask] = self.model.text_projection(
                selected_hidden[text_mask]
            )
        if multimodal_mask.any():
            projected[multimodal_mask] = self.model.hidden_projection(
                selected_hidden[multimodal_mask]
            )
        return projected

    def _get_last_prefill_talker_hidden(
        self, thinker_hidden: torch.Tensor
    ):
        # Build assistant prefix (matching HF/sglang-omni/vllm-omni pattern):
        # Text hidden: [pad*4, bos, proj[3]] (6 tokens)
        # (note that the assistant prefix was handled in the previous prefill stage)

        # Text part of assistant prefix
        # W3: pad and bos embeddings use Thinker embed -> text_projection
        # (via pre-computed cached values from init_tts_embeds)
        pad = self._tts_pad_embed_cached.expand(4, -1)  # 4 pad tokens
        bos_text = self._tts_bos_embed_cached.unsqueeze(0) # 1 bos token

        return torch.cat([
            pad,          # pad * 4   (4 tokens)
            bos_text,     # bos       (1 token)
            self.model.text_projection(thinker_hidden), #  (1 token)
        ], dim=0)  # (9, talker_hidden)

    def _get_last_prefill_codec_hidden(
        self, speaker: str
    ):
        # Build assistant prefix (matching HF/sglang-omni/vllm-omni pattern):
        # Codec hidden: [codec_embed(nothink, think_bos, think_eos,
        #                speaker, pad, bos)] (6 tokens)
        tc = self.config.talker
        speaker_id = tc.speaker_id.get(speaker.lower())
        if speaker_id is None:
            logger.warning(f"Speaker {speaker} not implemented")
            speaker_id = tc.codec_pad_id

        # Codec part of assistant prefix
        return self.talker_code_emb(torch.tensor([
            tc.codec_nothink_id,
            tc.codec_think_bos_id,
            tc.codec_think_eos_id,
            speaker_id,
            tc.codec_pad_id,
            tc.codec_bos_id,
        ], device=self.get_device(), dtype=torch.long))


    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {},
        **kwargs
    ) -> ARNodeInputs:
        device = self.get_device()

        if graph_walk == "talker_prefill":
            selected_hidden = inputs["thinker_states"][0].to(device)
            multimodal_mask = inputs["thinker_mask"][0]
            input_embeds = self._get_talker_embeds(
                selected_hidden=selected_hidden,
                multimodal_mask=multimodal_mask,
            )
            seq_len = input_embeds.shape[0]

        if graph_walk == "talker_last_prefill":
            rid = fwd_info.request_id
            # Last-prefill is the assistant's first token, which is
            # text-only — Thinker postprocess pre-selected layer-0 for it.
            last_hidden = inputs["thinker_states"][0].to(device)

            input_embeds = self._get_last_prefill_codec_hidden(
                fwd_info.step_metadata.get("voice", "Ethan")
            ) + self._get_last_prefill_talker_hidden(last_hidden)
            seq_len = input_embeds.shape[0]

        elif graph_walk == "talker_decode":
            dtype = self.model.text_projection.linear_fc1.weight.dtype
            input_embeds = inputs["talker_input_embeds"][0].to(dtype)

            thinker_states = inputs.get("thinker_states", [])
            rid = fwd_info.request_id
            if thinker_states:
                # Decode is always text → text_projection.
                text_hidden = self.model.text_projection(
                    thinker_states[0].to(dtype)
                )
            elif rid not in self._eos_embed_sent:
                text_hidden = self._tts_eos_embed_cached
                self._eos_embed_sent.add(rid)
            else:
                text_hidden = self._tts_pad_embed_cached
            input_embeds += text_hidden
            seq_len = 1

        return ARNodeInputs(
            input_embeds=input_embeds,
            input_seq_len=seq_len,
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        cache_manager = engine_inputs.cache_manager
        assert cache_manager is not None
        cache_manager.set_active_label("main")

        seq_lens = [
            inp.input_seq_len for inp in inputs
        ]
        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main"
        )
        cache_manager.plan_rope(
            seq_lens=seq_lens, pos_ids=None, label="main"
        )

        input_embeds = torch.cat([
            inp.input_embeds for inp in inputs
        ], dim=0)
        device = self.get_device()

        extra_args = {}
        if graph_walk != "talker_prefill":
            extra_args["suppress_mask"] = self._get_suppress_mask()
        if graph_walk == "talker_last_prefill":
            extra_args["last_token_indices"] = (
                torch.tensor(seq_lens, device=device, dtype=torch.long).cumsum(0) - 1
            )
        return {
            "input_embeds": input_embeds,
            "all_codes": torch.zeros(
                (len(inputs), self.num_codes),
                device=device, dtype=torch.long
            ),
            "codec_emb_sum": torch.zeros(
                (len(inputs), self.cp_cfg.hidden_size), device=device
            ),
            "pos_buf": torch.zeros((len(inputs), 1), device=device, dtype=torch.long),
            **extra_args
        }

    def _forward_prefill(
        self, cache_handle: BatchedCacheManager,
        input_embeds: torch.Tensor,
    ):
        self.model(input_embeds=input_embeds, cache_handle=cache_handle)
        return {}

    def _forward_decode_like(
        self, request_ids: list[str],
        cache_handle: BatchedCacheManager,
        injected_sampler: CudaGraphableSampler,
        suppress_mask: torch.Tensor,
        all_codes: torch.Tensor,
        codec_emb_sum: torch.Tensor,
        pos_buf: torch.Tensor,
        is_batched_decode: bool,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor | None = None,
        **kwargs
    ):
        """
        Runs the Talker LLM for stages that graph walks that sample a token
        and feed into the code predictor.

        ``last_token_indices`` (when provided) is used to ``index_select`` the
        per-request last hidden out of a packed multi-token-per-request hidden
        — the batched ``talker_last_prefill`` path computes it in ``preprocess``
        as ``cumsum(seq_lens) - 1`` and passes it through here so the
        codec_head only sees one hidden per request. Mutually exclusive with
        the batched-decode (``hidden`` is already ``(bs, hidden)``) and
        non-batched (``hidden[-1:, :]``) branches.
        """
        hidden = self.model(
            input_embeds=input_embeds, cache_handle=cache_handle
        )
        if last_token_indices is not None:
            last_hidden = hidden.index_select(0, last_token_indices)
        elif not is_batched_decode:
            last_hidden = hidden[-1:, :]
        else:
            last_hidden = hidden
        logits = self.model.codec_head(last_hidden)
        logits = logits.masked_fill(suppress_mask.unsqueeze(0), float("-inf"))
        # Apply the repetition penalty to the Talker's layer-0 codes (the code
        # predictor depth loop below uses sample_with_config and stays penalty-free
        # for now). The penalty is baked into the captured graph.
        layer0_codes = injected_sampler.sample(
            request_ids, logits, apply_penalty=True
        )

        # code predictor section
        embed = self.talker_code_emb(layer0_codes)  # [bs, hidden]
        codec_emb_sum.add_(embed)
        all_codes[:, 0] = layer0_codes
        bs = all_codes.shape[0]

        kv_cache = self._get_cp_kv_cache()[:, :bs]

        # forward over [last_hidden] to update kv cache with the Talker's final hidden
        # state as context for the code prediction. This returns nothing because the
        # layer 0 code is already provided by the talker LLM
        cp = self.code_predictor
        codec_embedding = cp.model.codec_embedding
        cp.forward_depth_unrolled(
            last_hidden.unsqueeze(1), pos_buf, kv_cache, cache_pos=0,
        )
        pos_buf += 1

        for group_idx in range(1, self.num_codes):
            hidden = cp.forward_depth_unrolled(
                embed.unsqueeze(1), pos_buf, kv_cache, cache_pos=group_idx,
            ).squeeze(1)
            pos_buf += 1

            logits = torch.matmul(
                hidden, cp.lm_head_weight[group_idx - 1].t()
            )

            # TODO: allow setting the code predictor temperature
            tokens = injected_sampler.sample_with_config(
                logits=logits, temperature=1.0,
                top_k=50, top_p=0.8
            )
            all_codes[:, group_idx] = tokens
            embed = codec_embedding[group_idx - 1](tokens)
            codec_emb_sum.add_(embed)

        return {
            "talker_input_embeds": [codec_emb_sum],
            "codec_tokens": [all_codes],
            "new_token": [layer0_codes]
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if graph_walk == "talker_prefill":
            return self._forward_prefill(
                cache_handle=engine_inputs.cache_manager,
                input_embeds=input_embeds
            )
        return self._forward_decode_like(
            request_ids=engine_inputs.request_ids,
            cache_handle=engine_inputs.cache_manager,
            injected_sampler=engine_inputs.sampler,
            input_embeds=input_embeds,
            is_batched_decode=(graph_walk == "talker_decode"),
            **kwargs
        )

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """Batched Talker forward shared between ``talker_decode``, ``talker_prefill``, and ``talker_last_prefill``.

        Decode path (1 token per request, ``hidden`` shape ``(bs, hidden)``):
          Runs the full LLM + codec_head + suppress_mask via _forward_decode_like
          and emits per-rid {last_hidden, logits} entries plus a ``__batched_logits__``
          sentinel for the runner's sample-once fast path.

        Prefill path (``talker_prefill``; multi-token-per-request, ``hidden``
        shape ``(total_tokens, hidden)``):
          Runs only the LLM backbone — no codec_head, no sampling. Production
          ``talker_prefill`` exists solely to populate the KV cache for the
          subsequent ``talker_last_prefill`` + ``talker_decode_loop``, so
          ``_forward_prefill`` returns ``{}`` in eager. We expose the post-LLM
          hidden state under the ``__batched_talker_prefill_hidden__`` sentinel
          purely so the parity test can compare graph vs eager hidden activations;
          the runner's _sample_and_remap drops this key (no per-rid dict, no
          __batched_logits__) and returns ``{rid: {} for rid in request_ids}``,
          matching eager.

        Last-prefill path (``talker_last_prefill``; fixed 6 tokens per request,
        ``hidden`` shape ``(bs * 6, hidden)``):
          Same _forward_decode_like as decode but uses the ``last_token_indices``
          tensor produced by ``preprocess`` (``cumsum(seq_lens) - 1``) to
          ``index_select`` the per-request last hidden before codec_head.
          Captured under ``BasicBatchedCudaGraphConfig`` (single bucket per bs:
          total_tokens = bs * 6), which routes ``_create_persistent_wrappers``
          through ``FlashInferPrefillWrapper`` (since total_tokens != bs);
          per-rid output construction matches the decode branch.
        """
        assert graph_walk in (
            "talker_decode", "talker_prefill", "talker_last_prefill",
        )
        cache_handle = engine_inputs.cache_manager

        if graph_walk == "talker_prefill":
            hidden = self.model(
                input_embeds=input_embeds,
                cache_handle=cache_handle,
            )
            return {
                "__batched_talker_prefill_hidden__": hidden,
            }

        last_token_indices = kwargs.pop("last_token_indices", None)
        if graph_walk == "talker_last_prefill":
            assert last_token_indices is not None, (
                "talker_last_prefill forward_batched requires "
                "last_token_indices from preprocess; got None."
            )

        fwd_out = self._forward_decode_like(
            request_ids=engine_inputs.request_ids,
            cache_handle=cache_handle,
            injected_sampler=engine_inputs.sampler,
            is_batched_decode=True,
            last_token_indices=last_token_indices,
            input_embeds=input_embeds,
            **kwargs
        )

        outputs = {
            rid: {
                "talker_input_embeds": [fwd_out["talker_input_embeds"][0][i:i+1]],
                "codec_tokens": [fwd_out["codec_tokens"][0][i]],
                "new_token": [fwd_out["new_token"][0][i]],
            } for i, rid in enumerate(engine_inputs.request_ids)
        }
        return outputs

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        if "new_token" not in outputs:
            return
        # Rename new_token → layer0_codes for downstream graph routing.
        # check_stop reads layer0_codes (same tensor) on the worker's slow path.
        codes = outputs.pop("new_token")[0]
        outputs["layer0_codes"] = [codes]

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "layer0_codes" not in outputs:
            return set()
        token = outputs["layer0_codes"][0].item()
        eos_token_id = self.config.talker.codec_eos_token_id
        max_tokens = request_info.step_metadata.get(
            "talker_max_tokens", request_info.max_tokens
        )
        if (eos_token_id is not None and eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("talker_decode_loop", 0) + 1 >= max_tokens):
            return {"talker_decode_loop"}
        return set()

    def cleanup_request(self, request_id: str) -> None:
        """Remove per-request state when a request completes."""
        self._eos_embed_sent.discard(request_id)

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        return batch.graph_walk in [
            "talker_decode", "talker_last_prefill"
        ] and len(model_inputs) <= self.MAX_BATCH_SIZE

    def max_batch_size(self, graph_walk):
        return self.MAX_BATCH_SIZE

    TALKER_PREFILL_TOKEN_BUCKETS = [128, 256, 512, 1024]
    TALKER_PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]

    # Fixed assistant prefix per request: pad*4 + bos + projected_thinker = 6.
    TALKER_LAST_PREFILL_TOKENS_PER_REQ = 6
    TALKER_LAST_PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def _build_talker_prefill_packed(
        self, num_tokens: int, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Synthesize a tensor-only post-preprocess packed dict for talker_prefill capture.

        Talker uses standard 1D RoPE applied inside ``Qwen3OmniAttention`` via
        ``cache_handle.apply_rope()`` (the cache manager owns the position state,
        set up by ``plan_rope`` outside the captured region), so unlike Thinker
        prefill_text we don't need to provide cos/sin tensors here. The captured
        forward only reads ``input_embeds``; everything else flows through the
        cache_handle that the runner re-plans on each replay.
        """
        talker_hidden_size = self.config.talker_hidden_size
        return {
            "input_embeds": torch.zeros(
                (num_tokens, talker_hidden_size),
                dtype=torch.bfloat16, device=device,
            ),
        }

    def get_cuda_graph_configs(self, device: torch.device, tp_world_size: int = 1):
        """Declare CUDA graph captures for ``talker_decode``, ``talker_prefill``, and ``talker_last_prefill``.

        ``talker_decode``: ``BasicBatchedCudaGraphConfig`` (one capture per bs;
        runner clones single_request_inputs with input_seq_len=1 and runs
        preprocess itself).

        ``talker_prefill``: ``FlashInferPackedCudaGraphConfig`` (one capture per
        (bs, num_tokens) bucket; the dict here IS the post-preprocess packed
        input — runner does not call preprocess at capture).

        ``talker_last_prefill``: ``BasicBatchedCudaGraphConfig`` (one capture per
        bs; single_request_inputs has input_seq_len=6 so total_tokens = bs * 6).
        ``total_tokens != bs`` forces ``_create_persistent_wrappers`` to use a
        ``FlashInferPrefillWrapper`` instead of the decode wrapper, which means
        ``cache_handle.get_qo_indptr_buf("main")`` is non-None at replay so
        ``forward_batched`` can ``index_select`` per-request last hidden out of
        the packed ``(bs * 6, talker_hidden)`` LLM output before codec_head.
        """
        talker_prefill_packed = {
            num_tokens: self._build_talker_prefill_packed(num_tokens, device)
            for num_tokens in self.TALKER_PREFILL_TOKEN_BUCKETS
        }
        return [
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="talker_decode", requires_cfg=False, labels=["main"],
                single_request_inputs=ARNodeInputs(
                    input_embeds=torch.zeros(
                        (1, self.config.talker_hidden_size),
                        device=device, dtype=torch.bfloat16
                    ),
                    input_seq_len=1,
                ),
                capture_batch_sizes=[1, 2, 4, 8, 16, 32],
                compile=True
            ),
            FlashInferPackedCudaGraphConfig(
                capture_graph_walk="talker_prefill",
                replay_graph_walks=["talker_prefill"],
                packed_seq_len_to_inputs=talker_prefill_packed,
                requires_cfg=False,
                labels=["main"],
                causal_attention=True,
                capture_batch_sizes=self.TALKER_PREFILL_CAPTURE_BATCH_SIZES,
                compile=True
            ),
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="talker_last_prefill",
                requires_cfg=False,
                labels=["main"],
                single_request_inputs=ARNodeInputs(
                    input_embeds=torch.zeros(
                        (self.TALKER_LAST_PREFILL_TOKENS_PER_REQ, self.config.talker_hidden_size),
                        device=device, dtype=torch.bfloat16
                    ),
                    input_seq_len=self.TALKER_LAST_PREFILL_TOKENS_PER_REQ,
                ),
                capture_batch_sizes=self.TALKER_LAST_PREFILL_CAPTURE_BATCH_SIZES,
                compile=True
            ),
        ]

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str]:
        return ["main"]


# ===================================================================
# 5. Code2WavSubmodule (audio_codec engine)
# ===================================================================


class Code2WavSubmodule(NodeSubmodule):
    """Wraps the HF Code2Wav vocoder for streaming chunk decode.

    Receives codec_tokens from the Talker (via StreamBuffer), selects
    the first ``num_quantizers`` codebook layers, runs the ConvNet
    vocoder, trims overlap context, and returns the PCM audio chunk.
    """

    def __init__(self, code2wav_model: Qwen3OmniMoeCode2Wav, config: Qwen3OmniModelConfig):
        super().__init__()
        self.code2wav = code2wav_model
        self.config = config
        # Per-request set of request_ids that have already emitted their first
        # audio chunk. The first chunk has no prior audio to overlap with, so
        # its output must NOT be trimmed — the left-context trim only applies
        # to subsequent chunks. Matches HF chunked_decode's ``context_size =
        # left_context_size if start_index - left_context_size > 0 else start_index``
        # logic, where the first iteration has context_size=0.
        self._first_chunk_emitted: set[str] = set()
        self._latest_seq_len: dict[str, int] = {}

        # Pre-compute the total upsample factor. HF defines this as
        # ``np.prod(upsample_rates + upsampling_ratios)`` — both tuples
        # contribute (upsample_rates via the decoder blocks, upsampling_ratios
        # via the upsample stack). For Qwen3-Omni this is 8*5*4*3*2*2 = 1920.
        total_upsample = 1
        for r in self.config.code2wav.upsample_rates:
            total_upsample *= r
        for r in self.config.code2wav.upsampling_ratios:
            total_upsample *= r
        self.total_upsample = total_upsample

        self.full_seqlen = self.config.code2wav.codec_left_context_frames + \
            self.config.code2wav.codec_chunk_frames

    def get_stateless_flavor(self) -> str:
        # Code2Wav vocoder runs in fp32 with no autocast and no torch.compile.
        return "audio_codec"

    def cleanup_request(self, request_id):
        self._first_chunk_emitted.discard(request_id)
        self._latest_seq_len.pop(request_id, None)

    def get_cuda_graph_configs(self, device, tp_world_size: int = 1):
        num_quantizers = self.config.code2wav.num_quantizers
        return [
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="code2wav_chunk",
                single_request_inputs=ARNodeInputs(
                    tensor_inputs={
                        "codec_tokens": torch.zeros((
                            num_quantizers, self.full_seqlen,
                        ), dtype=torch.long, device=device),
                        "position_ids": torch.arange(self.full_seqlen, device=device)
                    },
                ),
                capture_batch_sizes=[1, 2, 4, 8, 16, 32],
                compile=False
            ),
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        num_quantizers = self.config.code2wav.num_quantizers  # 16
        codec_eos = self.config.talker.codec_eos_token_id
        codec_tokens = inputs["codec_tokens"][0]

        # Reshape to (num_frames, num_code_groups) if flat
        if codec_tokens.dim() == 1:
            num_groups = self.config.num_code_groups  # 16 (Qwen3-Omni)
            if codec_tokens.shape[0] % num_groups == 0:
                codec_tokens = codec_tokens.view(-1, num_groups)
            else:
                codec_tokens = codec_tokens.unsqueeze(0)

        # Filter out codec_eos frames
        if codec_tokens.dim() == 2 and codec_tokens.shape[0] > 0:
            eos_mask = codec_tokens[:, 0] == codec_eos
            if eos_mask.any():
                codec_tokens = codec_tokens[~eos_mask]

        # Select first num_quantizers codebook layers
        if codec_tokens.shape[-1] > num_quantizers:
            codec_tokens = codec_tokens[..., :num_quantizers]

        # pad sequence to full_seqlen for batching
        orig_seq_len = codec_tokens.shape[0]
        pad_len = self.full_seqlen - codec_tokens.shape[0]
        if pad_len > 0:
            pad = torch.zeros((
                pad_len, codec_tokens.shape[1]
            ),dtype=codec_tokens.dtype,
            device=codec_tokens.device)
            codec_tokens = torch.cat([codec_tokens, pad], dim=0)
        self._latest_seq_len[fwd_info.request_id] = orig_seq_len

        # Transpose to (Q, T)
        codec_tokens = codec_tokens.T  # (Q, T)
        return NodeInputs(tensor_inputs={
            "codec_tokens": codec_tokens,
            "position_ids": torch.arange(codec_tokens.shape[1], device=codec_tokens.device)
        })

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        all_codec_tokens = [
            inp.tensor_inputs["codec_tokens"] for inp in inputs
        ]
        # Assert all requests have the same numel so they can be batched
        assert all(t.numel() == all_codec_tokens[0].numel() for t in all_codec_tokens), (
            f"All codec token inputs must have the same numel for batching, "
            f"got: {[t.numel() for t in all_codec_tokens]}"
        )
        # Stack into (bs, Q, T)
        batched_codec_tokens = torch.stack(all_codec_tokens, dim=0)
        position_ids = torch.stack([
            inp.tensor_inputs["position_ids"] for inp in inputs
        ], dim=0)
        return {"codec_tokens": batched_codec_tokens, "position_ids": position_ids}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs
    ) -> dict[str, NameToTensorList]:
        """Run the streaming vocoder with per-request left-context trim.

        The Talker→Code2Wav StreamBuffer uses ``LeftContextChunkPolicy``:
        the first popped chunk for a request contains ``codec_chunk_frames``
        fresh frames with no overlap; every subsequent chunk contains
        ``codec_chunk_frames + codec_left_context_frames`` frames where the
        leading ``codec_left_context_frames`` are overlap from the previous
        chunk's tail. The overlap lets the causal ConvNet warm up its state
        at chunk boundaries; the corresponding waveform samples must be
        trimmed from the emitted audio (they were already emitted by the
        previous chunk).

        We delegate to ``Qwen3OmniCode2Wav.chunked_decode_streaming`` with a
        per-request context list derived from ``_first_chunk_emitted`` --
        ``0`` for any request that has not yet emitted,
        ``config.codec_left_context_frames`` otherwise. After each request's
        chunk is converted to int16 PCM, ``_first_chunk_emitted`` is updated
        inline so the next chunk for the same request trims correctly.
        """
        request_ids = engine_inputs.request_ids
        if codec_tokens is None or codec_tokens.numel() == 0:
            return {rid: {} for rid in request_ids}

        wavs = self.code2wav(
            codec_tokens, position_ids
        )

        results: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(request_ids):
            wav = wavs[i]
            audio_int16 = (wav.clamp(-1, 1) * 32767).to(torch.int16).squeeze()
            results[rid] = {"audio_chunk": [audio_int16]}
        return results

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs
    ) -> NameToTensorList:
        """Raw vocoder forward -- returns int16 PCM without any trim.

        Prefer ``forward_batched`` for the streaming path; this method exists
        for callers that need a non-streaming, single-shot decode (e.g.
        debugging or offline batch use).
        """
        if codec_tokens is None or codec_tokens.numel() == 0:
            return {}

        wav = self.code2wav(codec_tokens, position_ids)
        audio_int16 = (wav.clamp(-1, 1) * 32767).to(torch.int16)
        return {"audio_chunk": [audio_int16]}

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        return len({
            inputs.tensor_inputs["codec_tokens"].numel() \
                for inputs in model_inputs
        }) == 1

    def can_use_cuda_graphs(self, batch, model_inputs: list[NodeInputs]):
        res = super().can_use_cuda_graphs(batch, model_inputs) \
            and self.can_batch(batch, model_inputs) \
                and model_inputs[0].tensor_inputs["codec_tokens"].shape[1] == self.full_seqlen
        return res

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        if "audio_chunk" not in outputs:
            return

        orig_seq_len = self._latest_seq_len[request_id]
        cfg_ctx = self.config.code2wav.codec_left_context_frames
        left_context_size = 0 if request_id not in self._first_chunk_emitted else cfg_ctx
        trim = left_context_size * self.total_upsample
        self._first_chunk_emitted.add(request_id)
        outputs["audio_chunk"][0] = outputs["audio_chunk"][0][trim:orig_seq_len*self.total_upsample]

