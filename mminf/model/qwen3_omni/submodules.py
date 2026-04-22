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

from dataclasses import asdict
import logging
from typing import Optional

import torch
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import NodeBatch
from mminf.engine.cache_manager import BatchedCacheManager
from mminf.engine.code_predictor_engine import CodePredictorCudaGraphRunner, CodePredictorSubmodule
from mminf.engine.cuda_graph_runner import CudaGraphConfig
from mminf.model.base import NodeSubmodule
from mminf.model.qwen3_omni.components.rope import (
    compute_3d_cos_sin,
    compute_rope_freqs,
    get_rope_index_audio,
    get_rope_index_text,
    get_rope_index_vision,
)
from mminf.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor, Qwen3OmniTalkerModel
from mminf.model.qwen3_omni.config import Qwen3OmniModelConfig
from mminf.utils.sampling import Sampler

logger = logging.getLogger(__name__)


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

    def preprocess(
        self,
        graph_walk: str,
        cache_manager: BatchedCacheManager | None = None,
        per_request_inputs: list[NameToTensorList] | None = None,
        request_ids: list[str] | None = None,
        per_request_info: dict[str, CurrentForwardPassInfo] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract mel spectrograms from inputs and pad for the encoder."""
        assert len(per_request_inputs) == 1, (
            "AudioEncoder processes one request at a time"
        )
        inputs = per_request_inputs[0]

        # Edge name from graph walk is "audio_features"
        audio_features = inputs["audio_features"][0]
        audio_seqlens = inputs.get("audio_seqlens", [None])[0]

        return {
            "audio_features": audio_features,
            "audio_seqlens": audio_seqlens,
        }

    def forward(
        self,
        request_info: CurrentForwardPassInfo,
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

    def can_batch(self, batch: NodeBatch) -> bool:
        return False


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

    def preprocess(
        self,
        graph_walk: str,
        cache_manager: BatchedCacheManager | None = None,
        per_request_inputs: list[NameToTensorList] | None = None,
        request_ids: list[str] | None = None,
        per_request_info: dict[str, CurrentForwardPassInfo] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract pixel_values, grid_thw, and compute cu_seqlens.

        ``pixel_values`` and ``image_grid_thw`` are produced by
        ``Qwen3OmniModel.process_prompt`` from the raw ``image_inputs``
        loaded by the data worker.
        """
        assert len(per_request_inputs) == 1, (
            "VisionEncoder processes one request at a time"
        )
        inputs = per_request_inputs[0]

        # Edge name from graph walk is "pixel_values"
        pixel_values = inputs["pixel_values"][0]       # (N_patches, C, patch_H, patch_W)
        grid_thw = inputs.get("image_grid_thw", inputs.get("grid_thw", [None]))[0]

        device = pixel_values.device
        spatial_merge_size = self.config.vision.spatial_merge_size

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

        # Compute number of tokens per image after spatial merge
        # Each image: (t * h * w) / (spatial_merge_size^2)
        tokens_per_image = (
            grid_thw.prod(dim=-1) // (spatial_merge_size ** 2)
        )

        # cu_seqlens for FlashAttention within the ViT
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(tokens_per_image, dim=0), (1, 0)
        ).to(torch.int32).to(device)

        return {
            "pixel_values": pixel_values,
            "grid_thw": grid_thw,
            "cu_seqlens": cu_seqlens,
        }

    def forward(
        self,
        request_info: CurrentForwardPassInfo,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        cu_seqlens: torch.Tensor,
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

    def can_batch(self, batch: NodeBatch) -> bool:
        return False


# ===================================================================
# 3. ThinkerSubmodule (ar engine) -- MOST COMPLEX
# ===================================================================


class ThinkerSubmodule(NodeSubmodule):
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

    def preprocess(
        self,
        graph_walk: str,
        cache_manager: BatchedCacheManager,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, torch.Tensor]:
        if graph_walk == "prefill_text":
            return self._preprocess_prefill_text(
                cache_manager, per_request_inputs, request_ids, per_request_info
            )
        elif graph_walk == "prefill_audio":
            return self._preprocess_prefill_audio(
                cache_manager, per_request_inputs, request_ids, per_request_info
            )
        elif graph_walk == "prefill_vision":
            return self._preprocess_prefill_vision(
                cache_manager, per_request_inputs, request_ids, per_request_info
            )
        elif graph_walk == "thinker_decode":
            return self._preprocess_decode(
                cache_manager, per_request_inputs, request_ids, per_request_info
            )
        else:
            raise ValueError(f"Unknown Thinker graph walk: {graph_walk!r}")

    # ---- prefill_text ----

    def _preprocess_prefill_text(
        self,
        cache_manager: BatchedCacheManager,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, torch.Tensor]:
        """Embed text token IDs, compute 3D position IDs, plan attention."""
        device = next(self.model.parameters()).device

        all_embeds = []
        all_pos_ids_3d = []
        seq_lens = []

        # first row: multimodal mask (all zero for text prefill)
        # second row: text inclusion mask (cuts out system prompt)
        masks_for_talker = {}

        for inp, rid in zip(per_request_inputs, request_ids, strict=True):
            text_ids = inp["text_inputs"][0].to(device)  # (seq_len,)
            embeds = self.model.model.embed_tokens(text_ids)

            all_embeds.append(embeds)
            seq_len = text_ids.shape[0]
            seq_lens.append(seq_len)

            # Compute 3D MRoPE position IDs for a pure-text span.  Each
            # prefill graph walk is single-modality so we use the simple
            # per-modality helper instead of the full HF parser.
            #
            # ``start_pos`` is the next MRoPE position for this request,
            # carried forward across walks by ``state.position_id_start``
            # (advanced post-forward by ``advance_seq_lens``).
            state = cache_manager._get_state(rid, "main")
            start_pos = float(state.position_id_start)

            pos_ids = get_rope_index_text(seq_len, start_pos, device)
            all_pos_ids_3d.append(pos_ids)

            masks_for_talker[rid] = torch.stack([
                torch.zeros(text_ids.shape, dtype=torch.bool, device=device), # multimodal
                self._get_talker_text_mask(text_ids) # text inclusion
            ])

        # Concatenate across requests
        input_embeds = torch.cat(all_embeds, dim=0)
        position_ids_3d = torch.cat(all_pos_ids_3d, dim=1)  # (3, total_tokens)

        # Compute cos/sin for 3D MRoPE.  Returned as separate tensor keys
        # (not a tuple) so the CUDA graph runner can detect them as static
        # inputs and copy them into the captured buffers at replay.
        inv_freq = self._get_inv_freq(device)
        cos_3d, sin_3d = compute_3d_cos_sin(
            position_ids_3d, inv_freq, mrope_section=self.MROPE_SECTION,
            target_dtype=input_embeds.dtype,
        )

        # Plan FlashInfer attention and rope for the main cache label
        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main"
        )
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        return {
            "input_embeds": input_embeds,
            "cos_3d": cos_3d,
            "sin_3d": sin_3d,
            "mrope_section": self.MROPE_SECTION,
            "seq_lens": seq_lens,
            "masks_for_talker": masks_for_talker
        }

    # ---- prefill_audio ----

    def _preprocess_prefill_audio(
        self,
        cache_manager: BatchedCacheManager,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, torch.Tensor]:
        """Splice audio embeddings into the Thinker, extending KV cache."""
        device = next(self.model.parameters()).device

        all_embeds = []
        all_pos_ids_3d = []
        seq_lens = []

        # first row: multimodal mask (ones except bos and eos)
        # second row: text inclusion mask (bos and eos)
        masks_for_talker = {}

        audio_start_id = self.config.thinker.audio_start_token_id
        audio_end_id = self.config.thinker.audio_end_token_id

        for inp, rid in zip(per_request_inputs, request_ids, strict=True):
            audio_embeds = inp["audio_embeds"][0].to(device)  # (audio_tokens, hidden)
            audio_len = audio_embeds.shape[0]

            mm_mask = torch.ones(audio_len + 2, dtype=torch.bool, device=device)
            mm_mask[[0, -1]] = 0
            masks_for_talker[rid] = torch.stack([
                mm_mask,
                ~mm_mask
            ])

            # Wrap the audio span in ``<|audio_bos|>`` / ``<|audio_eos|>``
            # sentinel token embeddings so the Thinker sees the same
            # prompt layout the HF processor produces.
            start_tok = torch.tensor(
                [audio_start_id], dtype=torch.long, device=device
            )
            end_tok = torch.tensor(
                [audio_end_id], dtype=torch.long, device=device
            )
            start_embed = self.model.model.embed_tokens(start_tok)
            end_embed = self.model.model.embed_tokens(end_tok)

            wrapped_embeds = torch.cat(
                [start_embed, audio_embeds, end_embed], dim=0
            )
            all_embeds.append(wrapped_embeds)
            total_len = audio_len + 2
            seq_lens.append(total_len)

            # Use the position carried over from the previous walk as
            # the starting offset (tracked via ``state.position_id_start``;
            # advanced post-forward by ``advance_seq_lens``).
            state = cache_manager._get_state(rid, "main")
            start_pos = float(state.position_id_start)

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
            end_pos_ids = get_rope_index_text(
                1, start_pos + 1 + audio_len, device
            )
            pos_ids = torch.cat(
                [start_pos_ids, audio_pos_ids, end_pos_ids], dim=1
            )
            all_pos_ids_3d.append(pos_ids)

        input_embeds = torch.cat(all_embeds, dim=0)
        position_ids_3d = torch.cat(all_pos_ids_3d, dim=1)

        inv_freq = self._get_inv_freq(device)
        cos_3d, sin_3d = compute_3d_cos_sin(
            position_ids_3d, inv_freq, mrope_section=self.MROPE_SECTION,
            target_dtype=input_embeds.dtype,
        )

        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main"
        )
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        return {
            "input_embeds": input_embeds,
            "cos_3d": cos_3d,
            "sin_3d": sin_3d,
            "mrope_section": self.MROPE_SECTION,
            "seq_lens": seq_lens,
            "masks_for_talker": masks_for_talker
        }

    # ---- prefill_vision ----

    def _preprocess_prefill_vision(
        self,
        cache_manager: BatchedCacheManager,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, torch.Tensor]:
        """Splice vision embeddings into the Thinker, extending KV cache.

        Computes 3D position IDs for vision: temporal = constant per image,
        h/w = spatial grid positions (via the vision encoder's grid_thw).
        """
        device = next(self.model.parameters()).device

        all_embeds = []
        all_pos_ids_3d = []
        seq_lens = []
        # Per-rid MRoPE-position advance for ``advance_seq_lens``.  Vision
        # prefills span a non-contiguous 3D position range, so the next
        # walk's MRoPE position is ``max(pos_ids) + 1``, which is generally
        # larger than ``seq_len``.  Thread this through ``forward_batched``
        # -> ``thinker.py`` so the post-forward advance lands on the right
        # ``state.position_id_start``.
        mrope_pos_advance: list[int] = []
        # first row: multimodal mask (ones except bos and eos)
        # second row: text inclusion mask (bos and eos)
        masks_for_talker = {}

        deepstack = []
        visual_pos_masks = []

        vision_start_id = self.config.thinker.vision_start_token_id
        vision_end_id = self.config.thinker.vision_end_token_id
        spatial_merge = self.config.vision.spatial_merge_size

        for inp, rid in zip(per_request_inputs, request_ids, strict=True):
            vision_embeds = inp["vision_embeds"][0].to(device)
            vision_len = vision_embeds.shape[0]

            mm_mask = torch.ones(vision_len + 2, dtype=torch.bool, device=device)
            mm_mask[[0, -1]] = 0
            masks_for_talker[rid] = torch.stack([
                mm_mask,
                ~mm_mask
            ])
            visual_pos_masks.append(mm_mask)

            # Wrap the vision span in ``<|vision_bos|>`` / ``<|vision_eos|>``
            # sentinel token embeddings.
            start_tok = torch.tensor(
                [vision_start_id], dtype=torch.long, device=device
            )
            end_tok = torch.tensor(
                [vision_end_id], dtype=torch.long, device=device
            )
            start_embed = self.model.model.embed_tokens(start_tok)
            end_embed = self.model.model.embed_tokens(end_tok)

            wrapped_embeds = torch.cat(
                [start_embed, vision_embeds, end_embed], dim=0
            )
            all_embeds.append(wrapped_embeds)
            total_len = vision_len + 2
            seq_lens.append(total_len)

            state = cache_manager._get_state(rid, "main")
            start_pos = float(state.position_id_start)

            # Vision tokens use spatial 3D positions (temporal constant,
            # h/w from the spatial grid after merging).  If a proper
            # ``image_grid_thw`` is available, use ``get_rope_index_vision``;
            # otherwise fall back to a 1-D sequence (test path without
            # AutoImageProcessor).
            grid_thw = inp.get("image_grid_thw", [None])[0]
            seconds_per_grid = inp.get("video_second_per_grid", [])
            seconds_per_grid = seconds_per_grid[0].item() if seconds_per_grid else None
            vision_pos_ids = get_rope_index_vision(
                grid_thw.to(device),
                start_pos + 1,  # leave room for the BOS token
                position_id_per_seconds=self.config.thinker.position_id_per_seconds,
                device=device,
                spatial_merge_size=spatial_merge,
                seconds_per_grid=seconds_per_grid
            )

            # Sentinel token positions (text-like).
            start_pos_ids = get_rope_index_text(1, start_pos, device)
            end_pos_base = float(vision_pos_ids.max().item()) + 1
            end_pos_ids = get_rope_index_text(1, end_pos_base, device)

            pos_ids = torch.cat(
                [start_pos_ids, vision_pos_ids, end_pos_ids], dim=1
            )
            all_pos_ids_3d.append(pos_ids)

            # Next MRoPE position after this vision block is ``end_pos_base
            # + 1`` (one past the EOS token).  ``advance_seq_lens`` by
            # default advances ``position_id_start`` by ``seq_len``, which
            # for vision (= vision_len + 2) is typically smaller than the
            # 3D-grid span.  Emit the correct per-request advance so the
            # Thinker forward can pass ``pos_id_ns`` through.
            mrope_pos_advance.append(int(end_pos_base + 1 - start_pos))

            deepstack.append(inp["deepstack"])

        input_embeds = torch.cat(all_embeds, dim=0)
        position_ids_3d = torch.cat(all_pos_ids_3d, dim=1)
        visual_pos_masks = torch.cat(visual_pos_masks, dim=0)

        inv_freq = self._get_inv_freq(device)
        cos_3d, sin_3d = compute_3d_cos_sin(
            position_ids_3d, inv_freq, mrope_section=self.MROPE_SECTION,
            target_dtype=input_embeds.dtype,
        )

        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main"
        )
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        result = {
            "input_embeds": input_embeds,
            "cos_3d": cos_3d,
            "sin_3d": sin_3d,
            "mrope_section": self.MROPE_SECTION,
            "mrope_pos_advance": mrope_pos_advance,
            "seq_lens": seq_lens,
            "masks_for_talker": masks_for_talker,
            "visual_pos_masks": visual_pos_masks,
            "deepstack": deepstack
        }

        return result

    # ---- thinker_decode ----

    def _preprocess_decode(
        self,
        cache_manager: BatchedCacheManager,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, torch.Tensor]:
        """Embed previous token, compute 3D position for next position, plan decode."""
        device = next(self.model.parameters()).device

        all_embeds = []
        all_pos_ids_3d = []
        seq_lens = []
        # first row: multimodal mask (zero for decode)
        # second row: text inclusion mask (one for decode)
        masks_for_talker = {}

        for inp, rid in zip(per_request_inputs, request_ids, strict=True):
            # Get previous token ID from text_inputs
            token_id = inp["text_inputs"][0].to(device)  # (1,) or scalar
            if token_id.dim() == 0:
                token_id = token_id.unsqueeze(0)
            embeds = self.model.model.embed_tokens(token_id)
            all_embeds.append(embeds)
            seq_lens.append(1)

            # Next MRoPE position for all 3 components: read from the
            # per-request cache-manager state (kept in sync by the
            # post-forward ``advance_seq_lens`` call in ``thinker.py``).
            state = cache_manager._get_state(rid, "main")
            next_pos = float(state.position_id_start)

            pos_ids = torch.tensor(
                [[next_pos], [next_pos], [next_pos]],
                dtype=torch.float,
                device=device,
            )  # (3, 1)
            all_pos_ids_3d.append(pos_ids)

            # Constant across all decode rids and all steps — use the
            # lazily-cached per-device tensor.
            masks_for_talker[rid] = self._get_decode_thinker_mask(device)

        input_embeds = torch.cat(all_embeds, dim=0)
        position_ids_3d = torch.cat(all_pos_ids_3d, dim=1)

        inv_freq = self._get_inv_freq(device)
        cos_3d, sin_3d = compute_3d_cos_sin(
            position_ids_3d, inv_freq, mrope_section=self.MROPE_SECTION,
            target_dtype=input_embeds.dtype,
        )

        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main"
        )
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        return {
            "input_embeds": input_embeds,
            "cos_3d": cos_3d,
            "sin_3d": sin_3d,
            "mrope_section": self.MROPE_SECTION,
            "seq_lens": seq_lens,
            "masks_for_talker": masks_for_talker
        }

    # ---- forward ----

    def forward(
        self,
        request_info: CurrentForwardPassInfo,
        graph_walk: str = "",
        cache_handle: BatchedCacheManager | None = None,
        input_embeds: torch.Tensor | None = None,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        mrope_section: list[int] | None = None,
        mrope_pos_advance: list[int] | None = None,
        masks_for_talker: dict[str, torch.Tensor] | None = None,
        deepstack: list[torch.Tensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Run Thinker transformer, produce logits (decode) and thinker_states.

        ``thinker_states`` is only emitted when audio output is requested
        (checked via ``request_info.step_metadata["audio_output"]``). This
        saves cross-partition bandwidth for text-only requests. Defaults to
        ``True`` for backwards compatibility with callers that do not set
        the flag (e.g. unit tests).
        """
        cache_handle.set_active_label("main")

        if deepstack is not None:
            # deepstack was a list of lists...
            deepstack = deepstack[0]

        # Default True for backwards-compat (tests, text-only callers that
        # forgot to set the flag still get the old behaviour).
        audio_output = True
        if request_info is not None:
            audio_output = request_info.step_metadata.get(
                "audio_output", True,
            )

        cos_sin_3d = (cos_3d, sin_3d) if cos_3d is not None else None

        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cache_handle=cache_handle,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
            mrope_pos_advance=mrope_pos_advance,
            deepstack_visual_embeds=deepstack,
            visual_pos_masks=visual_pos_masks
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

    def can_batch(self, batch: NodeBatch) -> bool:
        return batch.graph_walk == "thinker_decode"

    def get_cuda_graph_configs(self, device: torch.device) -> list[CudaGraphConfig]:
        """Declare a CUDA graph capture for ``thinker_decode``.

        ``dummy_capture_inputs`` is the PRE-preprocess input (a single
        dummy token id per rid); the runner calls ``preprocess`` itself to
        produce the static input buffers (``input_embeds``, ``cos_3d``,
        ``sin_3d``, etc.).

        ``compile=False`` on first land — mirrors what Orpheus does until
        we confirm interaction with the Triton fused-MoE autotune cache.
        ``capture_batch_sizes`` is limited to small buckets since each
        capture allocates persistent FlashInfer wrappers + static buffers
        for the full 30B Thinker; revisit after profiling real deployments.
        """
        return [
            CudaGraphConfig(
                graph_walk="thinker_decode",
                requires_cfg=False,
                labels=["main"],
                dummy_capture_inputs=[{
                    "text_inputs": [
                        torch.zeros(1, dtype=torch.long, device=device),
                    ],
                }],
                compile=False,
                capture_batch_sizes=[1, 2, 4, 8, 16],
            ),
        ]

    def forward_batched(
        self,
        graph_walk: str,
        request_ids: list[str],
        cache_manager: BatchedCacheManager,
        packed_inputs: dict[str, torch.Tensor],
        per_request_info: dict | None = None,
        per_request_metadata: dict | None = None,
    ) -> dict[str, NameToTensorList]:
        """Batched decode: multiple requests each contribute 1 token.

        Always packs ``thinker_states`` + ``thinker_mask`` in every per-rid
        output dict so the captured CUDA graph has a static output shape
        regardless of request metadata.  Per-rid filtering (dropping
        ``thinker_states`` / ``thinker_mask`` for requests with
        ``audio_output=False``) happens OUTSIDE the captured region via
        ``filter_batched_output``, applied by both the AR engine's eager
        path and the CUDA graph runner.
        """
        assert graph_walk == "thinker_decode"

        input_embeds = packed_inputs["input_embeds"]  # (batch, hidden)
        cos_3d = packed_inputs.get("cos_3d")
        sin_3d = packed_inputs.get("sin_3d")
        cos_sin_3d = (cos_3d, sin_3d) if cos_3d is not None else None
        mrope_section = packed_inputs.get("mrope_section")
        mrope_pos_advance = packed_inputs.get("mrope_pos_advance")
        masks_for_talker = packed_inputs.get("masks_for_talker")

        cache_manager.set_active_label("main")

        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cache_handle=cache_manager,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
            mrope_pos_advance=mrope_pos_advance,
        )

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

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str]:
        return ["main"]

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

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]
        token = outputs["new_token"][0].item()
        eos_token_id = self.config.im_end_token_id
        if (eos_token_id is not None and eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("thinker_decode_loop", 0) + 1 >= request_info.max_tokens):
            request_info.register_loop_stop("thinker_decode_loop")


# ===================================================================
# 4. TalkerSubmodule (ar engine) -- SECOND MOST COMPLEX
# ===================================================================

class TalkerLLMSubmodule(NodeSubmodule):
    def __init__(
        self,
        talker_model: Qwen3OmniTalkerModel,
        config: Qwen3OmniModelConfig
    ):
        super().__init__()
        self.model = talker_model   
        self.config = config

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
    
    def init_tts_embeds(self, thinker_embed_tokens: nn.Embedding) -> None:
        """Pre-compute TTS pad/bos/eos hidden states using the Thinker's
        embedding table + Talker text_projection

        Must be called after both the Thinker and Talker weights are loaded
        (only applicable when both reside on the same worker).  When they
        are on different workers, these embeddings should be transferred as
        constant tensors during model init.
        """
        device = next(self.model.parameters()).device
        with torch.no_grad():
            pad_raw = thinker_embed_tokens(
                torch.tensor([self.config.tts_pad_token_id], device=device)
            )
            bos_raw = thinker_embed_tokens(
                torch.tensor([self.config.tts_bos_token_id], device=device)
            )
            eos_raw = thinker_embed_tokens(
                torch.tensor([self.config.tts_eos_token_id], device=device)
            )
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
        self, layer_0_embed: torch.Tensor, layer_n_hidden: torch.Tensor,
        multimodal_mask: torch.Tensor, text_inclusion_mask: torch.Tensor
    ):
        text_mask = text_inclusion_mask & (~multimodal_mask)
        inclusion_mask = text_mask | multimodal_mask
        device = layer_0_embed.device

        projected = torch.zeros(
            inclusion_mask.sum(), self.config.talker_hidden_size,
            device=device, dtype=layer_0_embed.dtype,
        )

        # Compute sub-masks relative to the included positions
        included_text_mask = text_mask[inclusion_mask]
        included_multimodal_mask = multimodal_mask[inclusion_mask]

        if included_text_mask.any():
            projected[included_text_mask] = self.model.text_projection(
                layer_0_embed[text_mask]
            )
        if included_multimodal_mask.any():
            projected[included_multimodal_mask] = self.model.hidden_projection(
                layer_n_hidden[multimodal_mask]
            )

        return projected
    
    def preprocess(
        self,
        graph_walk: str,
        cache_manager: BatchedCacheManager,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, torch.Tensor]:
        if graph_walk == "talker_prefill":
            return self._preprocess_prefill(
                cache_manager, per_request_inputs, request_ids, per_request_info
            )
        if graph_walk == "talker_last_prefill":
            return self._preprocess_last_prefill(
                cache_manager, per_request_inputs, request_ids, per_request_info
            )
        # talker_decode
        return self._preprocess_decode(
            cache_manager, per_request_inputs, request_ids, per_request_info
        )

    def _preprocess_prefill(
        self,
        cache_manager: BatchedCacheManager,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ):
        """Build Talker prefill from Thinker states chunk.
        Non-last chunks: project states, plan prefill, forward fills KV cache only.
        """
        assert len(per_request_inputs) == 1, (
            "Talker prefill processes one request at a time"
        )
        device = next(self.model.parameters()).device
        inputs = per_request_inputs[0]

        # 1. Unpack thinker_states -> split into layer_0 and layer_n
        thinker_states = inputs["thinker_states"][0].to(device)
        thinker_hidden = self.config.thinker_hidden_size
        layer_0_embed = thinker_states[..., :thinker_hidden]
        layer_n_hidden = thinker_states[..., thinker_hidden:]

        mask = inputs["thinker_mask"][0]
        projected = self._get_talker_embeds(
            layer_0_embed=layer_0_embed, layer_n_hidden=layer_n_hidden,
            multimodal_mask=mask[0, :],
            text_inclusion_mask=mask[1, :]
        )

        seq_len = projected.shape[0]
        cache_manager.plan_attention(
            seq_lens=[seq_len], is_causal=True, label="main"
        )
        cache_manager.plan_rope(
            seq_lens=[seq_len], pos_ids=None, label="main"
        )

        return {
            "input_embeds": projected,
        }
    
    def _preprocess_last_prefill(
        self,
        cache_manager: BatchedCacheManager,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ):
        """
        Last chunk: build assistant prefix, sample first token.

        The thinker_states come from the first token sampled by the talker so will
        be text.
        """

        assert len(per_request_inputs) == 1, (
            "Talker prefill processes one request at a time"
        )
        device = next(self.model.parameters()).device
        rid = request_ids[0]
        inputs = per_request_inputs[0]
        info = per_request_info[rid]

        thinker_states = inputs["thinker_states"][0].to(device)
        thinker_hidden = self.config.thinker_hidden_size
        projected = self.model.text_projection(
            thinker_states[..., :thinker_hidden]
        )

        tc = self.config.talker

        # Build assistant prefix (matching HF/sglang-omni/vllm-omni pattern):
        # Text hidden: [pad*4, bos, proj[3]] (9 tokens)
        # Codec hidden: [codec_embed(nothink, think_bos, think_eos,
        #                speaker, pad, bos)] (9 tokens)
        # (note that the assistant prefix was handled in the previous prefill stage)

        # Text part of assistant prefix
        # W3: pad and bos embeddings use Thinker embed -> text_projection
        # (via pre-computed cached values from init_tts_embeds)
        pad_embed = self._tts_pad_embed_cached.expand(4, -1)  # 4 pad tokens
        bos_text_embed = self._tts_bos_embed_cached.unsqueeze(0) # 1 bos token

        speaker = info.step_metadata.get("voice", "Ethan")
        speaker_id = tc.speaker_id.get(speaker.lower())
        if speaker_id is None:
            logger.warning(f"Speaker {speaker} not implemented")
            speaker_id = tc.codec_pad_id

        text_hidden = torch.cat([
            pad_embed,          # pad * 4   (4 tokens)
            bos_text_embed,     # bos       (1 token)
            projected,          #  (1 token)
        ], dim=0)  # (9, talker_hidden)

        # Codec part of assistant prefix
        codec_special_ids = torch.tensor([
            tc.codec_nothink_id,
            tc.codec_think_bos_id,
            tc.codec_think_eos_id,
            speaker_id,
            tc.codec_pad_id,
            tc.codec_bos_id,
        ], device=device, dtype=torch.long)
        codec_hidden = self.model.model.codec_embedding(codec_special_ids)

        # Combine text and codec parts
        input_embeds = text_hidden + codec_hidden  # (9, talker_hidden)

        seq_len = input_embeds.shape[0]
        cache_manager.plan_attention(
            seq_lens=[seq_len], is_causal=True, label="main"
        )
        cache_manager.plan_rope(
            seq_lens=[seq_len], pos_ids=None, label="main"
        )

        return {
            "input_embeds": input_embeds,
            "suppress_mask": self._get_suppress_mask()
        }
    
    def _preprocess_decode(
        self,
        cache_manager: BatchedCacheManager,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ):
        all_embeds = []
        dtype = self.model.text_projection.linear_fc1.weight.dtype
        for inp, rid in zip(per_request_inputs, request_ids, strict=True):
            emb = inp["talker_input_embeds"][0].to(dtype)
            
            thinker_states = inp.get("thinker_states", [])
            if thinker_states:
                thinker_hidden = self.config.thinker_hidden_size
                emb += self.model.text_projection(
                    thinker_states[0][..., :thinker_hidden].to(dtype)
                )
            elif rid not in self._eos_embed_sent:
                emb += self._tts_eos_embed_cached
                self._eos_embed_sent.add(rid)
            else:
                emb += self._tts_pad_embed_cached
            all_embeds.append(emb)


        seq_lens = [1] * len(per_request_inputs)
        cache_manager.plan_attention(
            seq_lens=seq_lens, label="main"
        )
        cache_manager.plan_rope(
            seq_lens=seq_lens, label="main"
        )

        input_embeds = torch.cat(all_embeds, dim=0)
        return {
            "input_embeds": input_embeds,
            "suppress_mask": self._get_suppress_mask()
        }
    
    def _forward_prefill(
        self, cache_handle: BatchedCacheManager,
        input_embeds: torch.Tensor,
    ):
        self.model(input_embeds=input_embeds, cache_handle=cache_handle)
        return {}

    def _forward_decode_like(
        self, cache_handle: BatchedCacheManager,
        input_embeds: torch.Tensor,
        suppress_mask: torch.Tensor,
        is_batched_decode: bool
    ):
        """
        Runs the Talker LLM for stages that graoh walks that sample a token
        and feed into the code predictor.
        """

        hidden = self.model(
            input_embeds=input_embeds, cache_handle=cache_handle
        )
        if not is_batched_decode:
            last_hidden = hidden[-1:, :]
        else:
            last_hidden = hidden
        logits = self.model.codec_head(last_hidden)
        logits = logits.masked_fill(suppress_mask.unsqueeze(0), float("-inf"))

        return {
            "last_hidden": [last_hidden],
            "logits": [logits]
        }
    
    def forward(
        self,
        request_info: CurrentForwardPassInfo,
        graph_walk: str,
        cache_handle: BatchedCacheManager,
        input_embeds: torch.Tensor | None = None,
        suppress_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if graph_walk == "talker_prefill":
            return self._forward_prefill(
                cache_handle=cache_handle, input_embeds=input_embeds
            )
        return self._forward_decode_like(
            cache_handle=cache_handle,
            input_embeds=input_embeds,
            suppress_mask=suppress_mask,
            is_batched_decode=(graph_walk == "talker_decode"),
        )
    
    def forward_batched(
        self,
        graph_walk: str,
        request_ids: list[str],
        cache_manager: BatchedCacheManager,
        packed_inputs: dict[str, torch.Tensor],
        per_request_info: dict[str, CurrentForwardPassInfo]
    ):
        assert graph_walk == "talker_decode"
        fwd_out = self._forward_decode_like(
            cache_handle=cache_manager,
            input_embeds=packed_inputs["input_embeds"],
            suppress_mask=packed_inputs["suppress_mask"],
            is_batched_decode=True
        )

        outputs = {
            rid: {
                "last_hidden": fwd_out["last_hidden"][0][i:i+1],
                "logits": fwd_out["logits"][0][i:i+1],
            } for i, rid in enumerate(request_ids)
        }
        outputs["__batched_logits__"] = fwd_out["logits"][0]
        return outputs
    
    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        if "new_token" not in outputs:
            return
        codes = outputs.pop("new_token")[0]
        token = codes.item()
        eos_token_id = self.config.talker.codec_eos_token_id
        if (eos_token_id is not None and eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("talker_decode_loop", 0) + 1 >= request_info.max_tokens):
            request_info.register_loop_stop("talker_decode_loop")
        
        outputs["layer0_codes"] = [codes]
    
    def cleanup_request(self, request_id: str) -> None:
        """Remove per-request state when a request completes."""
        self._eos_embed_sent.remove(request_id)
    
    def can_batch(self, batch: NodeBatch) -> bool:
        return batch.graph_walk == "talker_decode"
    
    def _get_dummy_capture_inputs(self, device):
        return [{
            "talker_input_embeds": [torch.zeros((1, self.config.talker_hidden_size), device=device)],
            "thinker_states": [
                torch.zeros((1, self.config.thinker_hidden_size), device=device)
            ],
        }]

    def get_cuda_graph_configs(self, device: torch.device) -> list[CudaGraphConfig]:
        return [
            CudaGraphConfig(
                graph_walk="talker_decode", requires_cfg=False, labels=["main"],
                dummy_capture_inputs=self._get_dummy_capture_inputs(device),
                capture_batch_sizes=[1, 2, 4, 8, 16]
            )
        ]
    
    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str]:
        return ["main"]


class Qwen3OmniCodePredictorSubmodule(CodePredictorSubmodule):
    """Runs the Qwen3-Omni residual-codebook AR loop.

    Fast path (Phase 2): delegates the entire 15-iteration depth loop to
    ``CodePredictorCudaGraphRunner`` for a single unrolled CUDA-graph replay.

    Eager fallback (used only when the runner could not be captured, e.g. on
    CPU or if capture failed): Python-level loop with paged-FlashInfer
    attention via ``cache_manager`` -- functionally equivalent but without
    the graph speedup.
    """

    def __init__(
        self, code_predictor: Qwen3OmniCodePredictor,
        talker_code_emb: nn.Embedding,
        config: Qwen3OmniModelConfig
    ):
        super().__init__()
        self.code_predictor = code_predictor
        self.config = config
        self.cp_cfg = self.config.code_predictor
        self.num_codes = self.cp_cfg.num_code_groups
        self.talker_code_emb = talker_code_emb

    def preprocess(
        self,
        graph_walk: str,
        cache_manager: BatchedCacheManager,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ):
        return {
            "last_hidden": torch.cat([
                inp["last_hidden"][0] for inp in per_request_inputs
            ], dim=0),
            "layer0_codes": torch.cat([
                inp["layer0_codes"][0] for inp in per_request_inputs
            ]),
        }

    def forward_batched(
        self,
        graph_walk: str,
        request_ids: list[str],
        cache_manager: BatchedCacheManager,
        sampler: Sampler,
        cuda_graph_runner: CodePredictorCudaGraphRunner | None,
        packed_inputs: dict[str, torch.Tensor],
        per_request_info: dict[str, CurrentForwardPassInfo]
    ) -> dict[str, NameToTensorList]:
        last_hidden = packed_inputs["last_hidden"]
        layer0_codes = packed_inputs["layer0_codes"]

        if cuda_graph_runner is not None:
            # Fast path: one unrolled CUDA-graph replay covers the full
            # 15-iter MTP loop (attention + LM heads + sampling + embedders).
            outputs = cuda_graph_runner.run(
                graph_walk=graph_walk,
                request_ids=request_ids,
                last_hidden=last_hidden,
                layer0_codes=layer0_codes,
            )
            all_codes = outputs["all_codes"]
            codec_emb_sum = outputs["codec_emb_sum"]
        else:
            all_codes, codec_emb_sum = self._forward_batched_eager(
                request_ids=request_ids,
                cache_manager=cache_manager,
                sampler=sampler,
                last_hidden=last_hidden,
                layer0_codes=layer0_codes,
            )

        return {
            req_id: {
                "talker_input_embeds": [codec_emb_sum[i:i+1]],
                "codec_tokens": [all_codes[i]],
            } for i, req_id in enumerate(request_ids)
        }

    def _forward_batched_eager(
        self,
        request_ids: list[str],
        cache_manager: BatchedCacheManager,
        sampler: Sampler,
        last_hidden: torch.Tensor,
        layer0_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback path: Python AR loop over the paged-FlashInfer attention.

        Kept for environments where CUDA-graph capture is unavailable
        (CPU testing, capture failure on exotic hardware). Functionally
        equivalent to the unrolled graph but without the kernel-launch
        savings.
        """
        bs = len(request_ids)
        codec_emb_sum = torch.zeros_like(last_hidden)
        all_codes = torch.zeros(
            (bs, self.num_codes),
            device=layer0_codes.device, dtype=torch.long,
        )
        all_codes[:, 0] = layer0_codes

        # "Prefill" over last_hidden.
        cache_manager.plan_attention([1] * bs, label="main")
        cache_manager.plan_rope([1] * bs, label="main")
        self.code_predictor(last_hidden, cache_manager)

        # Seed codec_emb_sum with the layer-0 codec embedding.
        embed = self.talker_code_emb(layer0_codes)
        codec_emb_sum = codec_emb_sum + embed

        for group_idx in range(1, self.num_codes):
            cache_manager.plan_attention([1] * bs, label="main")
            cache_manager.plan_rope([1] * bs, label="main")
            hidden = self.code_predictor(embed, cache_manager)

            orig_dtype = hidden.dtype
            lm_head = self.code_predictor.get_lm_head(group_idx)
            logits = lm_head(hidden.to(lm_head.weight.dtype)).to(orig_dtype)
            codes = sampler.sample(request_ids, logits)
            all_codes[:, group_idx] = codes

            embed = self.code_predictor.get_embedding(group_idx)(codes)
            codec_emb_sum = codec_emb_sum + embed

        return all_codes, codec_emb_sum

    def can_batch(self, batch: NodeBatch) -> bool:
        return True  # we can always batch

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

    def __init__(self, code2wav_model: nn.Module, config: Qwen3OmniModelConfig):
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

        # Pre-compute the total upsample factor. HF defines this as
        # ``np.prod(upsample_rates + upsampling_ratios)`` — both tuples
        # contribute (upsample_rates via the decoder blocks, upsampling_ratios
        # via the upsample stack). For Qwen3-Omni this is 8*5*4*3*2*2 = 1920.
        total_upsample = 1
        for r in self.config.code2wav.upsample_rates:
            total_upsample *= r
        for r in self.config.code2wav.upsampling_ratios:
            total_upsample *= r
        self._total_upsample = total_upsample
    
    def preprocess(
        self,
        per_request_inputs: list[NameToTensorList] | None = None,
        request_ids: list[str] | None = None,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """Unpack codec_tokens from StreamBuffer chunks for a batch of requests.

        For each request, selects the first ``num_quantizers`` (16) of the 32
        code groups and transposes to [num_quantizers, num_frames].  All
        requests in the batch must have the same num_frames so the results can
        be stacked into a single (bs, Q, T) tensor.
        """
        num_quantizers = self.config.code2wav.num_quantizers  # 16
        codec_eos = self.config.talker.codec_eos_token_id

        per_request_codec: list[torch.Tensor] = []

        for inputs in per_request_inputs:
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

            # Transpose to (Q, T)
            codec_tokens = codec_tokens.T  # (Q, T)
            per_request_codec.append(codec_tokens)

        # Assert all requests have the same numel so they can be batched
        assert all(t.numel() == per_request_codec[0].numel() for t in per_request_codec), (
            f"All codec token inputs must have the same numel for batching, "
            f"got: {[t.numel() for t in per_request_codec]}"
        )

        # Stack into (bs, Q, T)
        batched_codec_tokens = torch.stack(per_request_codec, dim=0)

        return {"codec_tokens": batched_codec_tokens}
    
    def forward_batched(
        self,
        request_ids: list[str],
        packed_inputs: dict[str, torch.Tensor],
        per_request_info: dict[str, CurrentForwardPassInfo],
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
        codec_tokens = packed_inputs.get("codec_tokens")
        if codec_tokens is None or codec_tokens.numel() == 0:
            return {rid: {} for rid in request_ids}

        cfg_ctx = self.config.code2wav.codec_left_context_frames
        left_context_size = [
            0 if rid not in self._first_chunk_emitted else cfg_ctx
            for rid in request_ids
        ]

        wavs = self.code2wav.chunked_decode_streaming(
            codec_tokens, left_context_size=left_context_size
        )

        results: dict[str, NameToTensorList] = {}
        for rid, wav in zip(request_ids, wavs, strict=True):
            audio_int16 = (wav.clamp(-1, 1) * 32767).to(torch.int16).squeeze()
            results[rid] = {"audio_chunk": [audio_int16]}
            self._first_chunk_emitted.add(rid)
        return results

    def forward(
        self,
        codec_tokens: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Raw vocoder forward -- returns int16 PCM without any trim.

        Prefer ``forward_batched`` for the streaming path; this method exists
        for callers that need a non-streaming, single-shot decode (e.g.
        debugging or offline batch use).
        """
        if codec_tokens is None or codec_tokens.numel() == 0:
            return {}

        wav = self.code2wav(codec_tokens)
        audio_int16 = (wav.clamp(-1, 1) * 32767).to(torch.int16)
        return {"audio_chunk": [audio_int16]}

    def can_batch(self, batch: NodeBatch) -> bool:
        return len({
            inputs["codec_tokens"][0].numel() for inputs in batch.per_request_input_tensors.values()
        }) == 1
    
    # Cuda graph memory is blowing up; possibly due to us just being a wrapper around the
    # transformers module. Until code2wav becomes a bottleneck, making this eager mode for now
    # def can_use_cuda_graphs(self, batch):
    #     total_numel = self.config.num_code_groups * (
    #         self.config.code2wav.codec_left_context_frames + self.config.code2wav.codec_chunk_frames
    #     )
    #     return all([
    #         inputs["codec_tokens"][0].numel() == total_numel for inputs in batch.per_request_input_tensors.values()
    #     ])
    
    # def get_cuda_graph_configs(self, device: torch.device) -> list[CudaGraphConfig]:
    #     return [
    #         CudaGraphConfig(
    #             graph_walk="code2wav_chunk",
    #             requires_cfg=False,
    #             dummy_capture_inputs=[{
    #                 "codec_tokens": [
    #                     torch.zeros(
    #                         (self.config.code2wav.codec_left_context_frames + self.config.code2wav.codec_chunk_frames, self.config.num_code_groups),
    #                         dtype=torch.long, device=device
    #                     ),
    #                 ],
    #             }],
    #             compile=True,
    #             capture_batch_sizes=[1, 2],
    #         ),
    #     ]
