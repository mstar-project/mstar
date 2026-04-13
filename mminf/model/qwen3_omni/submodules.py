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
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import NodeBatch
from mminf.engine.cache_manager import BatchedCacheManager
from mminf.engine.cuda_graph_runner import CudaGraphConfig
from mminf.model.base import NodeSubmodule
from mminf.model.qwen3_omni.components.rope import (
    compute_3d_cos_sin,
    compute_rope_freqs,
    get_rope_index_audio,
    get_rope_index_text,
    get_rope_index_vision,
)
from mminf.model.qwen3_omni.config import Qwen3OmniModelConfig

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

        # Per-request MRoPE position delta tracking
        self._mrope_position_deltas: dict[str, torch.Tensor] = {}

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        """Lazy-initialize and cache inverse frequencies."""
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = compute_rope_freqs(
                self.config.thinker_head_dim,
                rope_theta=self.config.thinker_text.rope_theta,
                device=device,
            )
        return self._inv_freq
    
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

        for inp, rid in zip(per_request_inputs, request_ids):
            text_ids = inp["text_inputs"][0].to(device)  # (seq_len,)
            embeds = self.model.model.embed_tokens(text_ids)

            all_embeds.append(embeds)
            seq_len = text_ids.shape[0]
            seq_lens.append(seq_len)

            # Compute 3D MRoPE position IDs for a pure-text span.  Each
            # prefill graph walk is single-modality so we use the simple
            # per-modality helper instead of the full HF parser.
            #
            # ``start_pos`` for the text prefill is picked up from the
            # running per-request delta (0 on the very first walk).
            delta = self._mrope_position_deltas.get(rid, torch.tensor(0.0))
            start_pos = float(delta.item()) if delta.numel() > 0 else 0.0

            pos_ids = get_rope_index_text(seq_len, start_pos, device)
            all_pos_ids_3d.append(pos_ids)

            # Advance the per-request position delta so the next walk
            # (audio / vision / decode) starts at the correct offset.
            self._mrope_position_deltas[rid] = torch.tensor(
                start_pos + seq_len, device=device
            )

            masks_for_talker[rid] = torch.stack([
                torch.zeros(text_ids.shape, dtype=torch.bool, device=device), # multimodal
                self._get_talker_text_mask(text_ids) # text inclusion
            ])

        # Concatenate across requests
        input_embeds = torch.cat(all_embeds, dim=0)
        position_ids_3d = torch.cat(all_pos_ids_3d, dim=1)  # (3, total_tokens)

        # Compute cos/sin for 3D MRoPE
        inv_freq = self._get_inv_freq(device)
        cos_sin_3d = compute_3d_cos_sin(
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
            "cos_sin_3d": cos_sin_3d,
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

        for inp, rid in zip(per_request_inputs, request_ids):
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

            # Use the position delta carried over from the text prefill
            # as the starting offset.
            delta = self._mrope_position_deltas.get(rid, torch.tensor(0.0))
            start_pos = float(delta.item()) if delta.numel() > 0 else 0.0

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

            # Update position delta for subsequent walks.  We advanced
            # by ``2 + audio_len`` positions (BOS + frames + EOS).
            self._mrope_position_deltas[rid] = torch.tensor(
                start_pos + total_len, device=device
            )

        input_embeds = torch.cat(all_embeds, dim=0)
        position_ids_3d = torch.cat(all_pos_ids_3d, dim=1)

        inv_freq = self._get_inv_freq(device)
        cos_sin_3d = compute_3d_cos_sin(
            position_ids_3d, inv_freq, mrope_section=self.MROPE_SECTION,
            target_dtype=input_embeds.dtype,
        )

        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main"
        )
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        return {
            "input_embeds": input_embeds,
            "cos_sin_3d": cos_sin_3d,
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
        # first row: multimodal mask (ones except bos and eos)
        # second row: text inclusion mask (bos and eos)
        masks_for_talker = {}

        deepstack = []
        visual_pos_masks = []

        vision_start_id = self.config.thinker.vision_start_token_id
        vision_end_id = self.config.thinker.vision_end_token_id
        spatial_merge = self.config.vision.spatial_merge_size

        for inp, rid in zip(per_request_inputs, request_ids):
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

            delta = self._mrope_position_deltas.get(rid, torch.tensor(0.0))
            start_pos = float(delta.item()) if delta.numel() > 0 else 0.0

            # Vision tokens use spatial 3D positions (temporal constant,
            # h/w from the spatial grid after merging).  If a proper
            # ``image_grid_thw`` is available, use ``get_rope_index_vision``;
            # otherwise fall back to a 1-D sequence (test path without
            # AutoImageProcessor).
            grid_thw = inp.get("image_grid_thw", [None])[0]
            if grid_thw is not None and grid_thw.numel() > 0:
                vision_pos_ids = get_rope_index_vision(
                    grid_thw.to(device),
                    start_pos + 1,  # leave room for the BOS token
                    device=device,
                    spatial_merge_size=spatial_merge,
                )
            else:
                # Testing/fallback path: no grid_thw available, so treat
                # vision tokens as a flat 1-D span with text-like positions.
                vision_pos_ids = get_rope_index_text(
                    vision_len, start_pos + 1, device
                )

            # Sentinel token positions (text-like).
            start_pos_ids = get_rope_index_text(1, start_pos, device)
            end_pos_base = float(vision_pos_ids.max().item()) + 1
            end_pos_ids = get_rope_index_text(1, end_pos_base, device)

            pos_ids = torch.cat(
                [start_pos_ids, vision_pos_ids, end_pos_ids], dim=1
            )
            all_pos_ids_3d.append(pos_ids)

            # Advance the per-request position delta by one past the
            # EOS token.
            self._mrope_position_deltas[rid] = torch.tensor(
                end_pos_base + 1, device=device
            )

            deepstack.append(inp["deepstack"])

        input_embeds = torch.cat(all_embeds, dim=0)
        position_ids_3d = torch.cat(all_pos_ids_3d, dim=1)
        visual_pos_masks = torch.cat(visual_pos_masks, dim=0)

        inv_freq = self._get_inv_freq(device)
        cos_sin_3d = compute_3d_cos_sin(
            position_ids_3d, inv_freq, mrope_section=self.MROPE_SECTION,
            target_dtype=input_embeds.dtype,
        )

        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main"
        )
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        deepstack 

        result = {
            "input_embeds": input_embeds,
            "cos_sin_3d": cos_sin_3d,
            "mrope_section": self.MROPE_SECTION,
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

        for inp, rid in zip(per_request_inputs, request_ids):
            # Get previous token ID from text_inputs
            token_id = inp["text_inputs"][0].to(device)  # (1,) or scalar
            if token_id.dim() == 0:
                token_id = token_id.unsqueeze(0)
            embeds = self.model.model.embed_tokens(token_id)
            all_embeds.append(embeds)
            seq_lens.append(1)

            # Next position for all 3 components: use current sequence length
            # from the cache manager state
            delta = self._mrope_position_deltas.get(rid, torch.tensor(0.0))
            next_pos = float(delta.item()) if delta.numel() > 0 else 0.0

            pos_ids = torch.tensor(
                [[next_pos], [next_pos], [next_pos]],
                dtype=torch.float,
                device=device,
            )  # (3, 1)
            all_pos_ids_3d.append(pos_ids)

            # Advance position delta
            self._mrope_position_deltas[rid] = torch.tensor(
                next_pos + 1, device=device
            )

            masks_for_talker[rid] = torch.tensor(
                [[0], [1]], dtype=torch.bool, device=device
            )

        input_embeds = torch.cat(all_embeds, dim=0)
        position_ids_3d = torch.cat(all_pos_ids_3d, dim=1)

        inv_freq = self._get_inv_freq(device)
        cos_sin_3d = compute_3d_cos_sin(
            position_ids_3d, inv_freq, mrope_section=self.MROPE_SECTION,
            target_dtype=input_embeds.dtype,
        )

        cache_manager.plan_attention(
            seq_lens=seq_lens, is_causal=True, label="main"
        )
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        return {
            "input_embeds": input_embeds,
            "cos_sin_3d": cos_sin_3d,
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
        cos_sin_3d: tuple[torch.Tensor, torch.Tensor] | None = None,
        mrope_section: list[int] | None = None,
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

        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cache_handle=cache_handle,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
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
        """Return dummy inputs for CUDA graph capture, or None if this walk
        doesn't support CUDA graphs.

        Default: returns text_inputs for "decode" walks. Override in subclasses
        for walks with different input names (e.g., Qwen3-Omni Thinker uses
        "input_embeds" and "cos_sin_3d"; Talker uses "input_embeds").
        """
        return [
            # CudaGraphConfig(
            #     graph_walk="thinker_decode", requires_cfg=False, labels=["main"],
            #     dummy_capture_inputs=[{"text_inputs": [torch.zeros(1, dtype=torch.long, device=device)]}],
            #     compile=False
            # ),
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

        ``thinker_states`` is only included in a request's outputs when
        that request has ``audio_output=True`` in its step_metadata. Text
        only requests skip it to save cross-partition bandwidth.
        """
        assert graph_walk == "thinker_decode"

        input_embeds = packed_inputs["input_embeds"]  # (batch, hidden)
        cos_sin_3d = packed_inputs.get("cos_sin_3d")
        mrope_section = packed_inputs.get("mrope_section")
        masks_for_talker = packed_inputs.get("masks_for_talker")

        cache_manager.set_active_label("main")

        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cache_handle=cache_manager,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
        )

        logits = self.model.lm_head(hidden)  # (batch, vocab)

        # Determine per-request audio_output flags (default True for
        # backwards compat).  If ANY request in the batch wants audio
        # output we still need to compute the packed thinker_states tensor;
        # we then only include it in the outputs for requests that asked
        # for it.
        request_ids = cache_manager.request_ids
        per_request_info = per_request_info or {}
        audio_output_flags: dict[str, bool] = {}
        for rid in request_ids:
            info = per_request_info.get(rid)
            if info is not None:
                audio_output_flags[rid] = info.step_metadata.get(
                    "audio_output", True,
                )
            else:
                audio_output_flags[rid] = True

        any_audio = any(audio_output_flags.values())

        if any_audio:
            # Pack thinker_states once for the whole batch
            if layer_n_hidden is not None:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_n_hidden], dim=-1,
                )
            else:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_0_embed], dim=-1,
                )
        else:
            thinker_states = None

        outputs: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(request_ids):
            req_out: NameToTensorList = {"logits": [logits[i : i + 1]]}
            if audio_output_flags[rid] and thinker_states is not None:
                req_out["thinker_states"] = [thinker_states[i : i + 1]]
                if rid in masks_for_talker:
                    req_out["thinker_mask"] = [masks_for_talker[rid]]
            outputs[rid] = req_out
        return outputs

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str]:
        return ["main"]


# ===================================================================
# 4. TalkerSubmodule (ar engine) -- SECOND MOST COMPLEX
# ===================================================================


class TalkerSubmodule(NodeSubmodule):
    """Wraps the Talker MoE transformer + inline Code Predictor.

    Dispatches on graph_walk:
      - talker_prefill: extend KV cache with projected Thinker states
        (multiple chunks), then on the LAST chunk build the assistant
        prefix and sample the first codec token.
      - talker_decode: re-embed previous all_codes, receive thinker_states
        as normal graph input, produce next codec token + 31
        residual codebook tokens via Code Predictor.

    The TalkerSubmodule manages per-request state:
      - _tts_pad_embed: lazy-initialized fallback embedding when Thinker
        hasn't generated enough tokens
    """

    def __init__(
        self,
        talker_model: nn.Module,
        code_predictor: nn.Module,
        config: Qwen3OmniModelConfig,
    ):
        super().__init__()
        self.model = talker_model    # Qwen3OmniTalkerModel
        self.code_predictor = code_predictor  # HF Code Predictor (float32)
        self.config = config

        # W3: Pre-computed TTS special embeddings.  These are produced by
        # running the THINKER's embed_tokens through the Talker's
        # text_projection.  Initialized via init_tts_embeds() after both
        # the Thinker and Talker weights are loaded.  Until then the
        # fallback is zeros (same as old behaviour).
        self._tts_pad_embed_cached: torch.Tensor | None = None
        self._tts_bos_embed_cached: torch.Tensor | None = None
        self._tts_eos_embed_cached: torch.Tensor | None = None

        # ---- Layer-0 codec sampling ------------------------------------
        # HF's reference Talker.generate call uses:
        #   do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
        #   repetition_penalty=1.05, suppress_tokens=[...]
        # where ``suppress_tokens`` masks out the "special token" region of
        # the Talker's codec vocab — namely [vocab_size - 1024, vocab_size)
        # (i.e. IDs 2048..3071 for vocab_size=3072) EXCEPT codec_eos_token_id.
        # Those IDs live in the Talker vocab but are NOT valid acoustic codes
        # for Code2Wav (Code2Wav's codebook is only 2048 per layer), so if
        # they're ever sampled as layer-0 they land in the wrong region of
        # Code2Wav's code_embedding table and produce garbled audio.
        # codec_eos is kept because it's the valid stop signal.
        self._talker_temperature: float = 0.9
        self._talker_top_k: int = 50
        self._talker_top_p: float = 1.0
        # HF uses ``repetition_penalty=1.05`` for the Talker.  Without it the
        # codec LM tends to loop in a subspace of recently-sampled tokens and
        # never picks ``codec_eos``, so the utterance "goes off the rails"
        # past the intended speech and keeps generating random words until
        # the outer ``max_output_tokens`` cap fires.  The penalty divides the
        # logit of any previously-seen token (positive logits) or multiplies
        # it (negative logits), biasing the distribution AWAY from repeats.
        self._talker_repetition_penalty: float = 1.05

        # Code Predictor residual sampling — HF uses top_k=50, top_p=0.8,
        # temperature=1.0 (the HF generate() default).  The CP does not
        # use repetition_penalty (HF's CP.generate call omits it).
        self._cp_temperature: float = 1.0
        self._cp_top_k: int = 50
        self._cp_top_p: float = 0.8

        # Lazy-built suppress mask for layer-0 logits.  Shape (vocab_size,)
        # with True at positions to suppress.  Cached on first forward.
        self._suppress_mask: torch.Tensor | None = None

        # Per-request seen-token mask for repetition penalty.  Indexed by
        # request_id -> bool tensor of shape (vocab_size,) where True means
        # "this layer-0 codec token has been sampled at least once for this
        # request".  Initialized lazily on the first layer-0 sample.  The
        # mask is cleared when the request completes (via ``cleanup_request``).
        self._seen_layer0_mask: dict[str, torch.Tensor] = {}

        # Per-request flag: whether we've already sent tts_eos_embed as
        # the text conditioning for this request.  In HF/vllm/sglang,
        # trailing_text_hidden ends with tts_eos_embed appended:
        #   trailing_text_hidden = cat(assistant_hidden[4:], tts_eos_embed)
        # This gives the Talker a "text is done" signal before switching
        # to tts_pad_embed.  In our streaming model, when the Thinker
        # finishes the stream returns empty chunks.  We use this flag to
        # inject tts_eos_embed for ONE step before falling back to pad.
        self._eos_embed_sent: set[str] = set()

        # No delay buffer needed for thinker_states alignment — verified
        # against vllm-omni.  trailing_text_hidden = assistant_hidden[4:]
        # starts at the second generated token.  Our stream naturally
        # delivers thinker_decode_1 for decode step 0 (matching vllm's
        # trailing_text_hidden[0]) because thinker_decode_0 was consumed
        # by the last prefill for prefix position 8.

    # ---- Stochastic sampling helpers -------------------------------------

    def _get_seen_mask(
        self, request_id: str, vocab_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return the per-request layer-0 seen-token mask, initializing if needed."""
        mask = self._seen_layer0_mask.get(request_id)
        if (
            mask is None
            or mask.shape[0] != vocab_size
            or mask.device != device
        ):
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            self._seen_layer0_mask[request_id] = mask
        return mask

    def _mark_seen(self, request_id: str, token_id: int) -> None:
        """Record that ``token_id`` has been sampled for this request.

        The token stays marked for the lifetime of the request so the
        repetition penalty persistently biases the distribution against it.
        HF's ``RepetitionPenaltyLogitsProcessor`` works the same way -- it
        penalizes ALL previously-seen tokens, not just recent ones.
        """
        mask = self._seen_layer0_mask.get(request_id)
        if mask is not None and 0 <= token_id < mask.shape[0]:
            mask[token_id] = True

    def _get_suppress_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        """Return the bool mask of layer-0 logits to set to -inf.

        Matches HF's ``talker_supppressed_tokens`` list: suppress the top
        1024 IDs of the Talker vocab (the "special token" region) EXCEPT
        ``codec_eos_token_id``, which is the valid stop signal.
        """
        if (
            self._suppress_mask is None
            or self._suppress_mask.shape[0] != vocab_size
            or self._suppress_mask.device != device
        ):
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            start = vocab_size - 1024
            if start < 0:
                start = 0
            mask[start:vocab_size] = True
            # Do not suppress codec_eos (the valid stop signal).
            eos = self.config.talker.codec_eos_token_id
            if 0 <= eos < vocab_size:
                mask[eos] = False
            self._suppress_mask = mask
        return self._suppress_mask

    @staticmethod
    def _top_k_top_p_sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float = 1.0,
        seen_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pure-PyTorch stochastic sampler with top-k + top-p filtering.

        Args:
            logits: shape ``(1, vocab_size)`` — logits for a single token.
            temperature: scaling factor applied before softmax.
            top_k: keep only the top-k highest-logit tokens (0 = disabled).
            top_p: keep the smallest set whose cumulative probability is
                >= top_p (1.0 = disabled).
            repetition_penalty: HuggingFace-style sign-aware penalty applied
                BEFORE temperature scaling.  For each token flagged in
                ``seen_token_mask``, the logit is divided by ``penalty`` if
                positive or multiplied by ``penalty`` if negative, matching
                the ``transformers`` ``RepetitionPenaltyLogitsProcessor``
                convention.  Ignored when ``seen_token_mask`` is None or
                when ``repetition_penalty == 1.0``.
            seen_token_mask: bool tensor of shape ``(vocab_size,)`` marking
                tokens the current request has already sampled.

        Returns:
            ``torch.LongTensor`` of shape ``(1,)`` — the sampled token ID.
        """
        # Work in float32 for numerical stability (logits from a bf16/fp16
        # codec_head would otherwise produce unstable softmax/topk results).
        logits = logits.float()

        # Repetition penalty (applied BEFORE temperature to match HF order).
        if (
            repetition_penalty != 1.0
            and seen_token_mask is not None
            and seen_token_mask.any()
        ):
            mask = seen_token_mask.to(logits.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # broadcast over batch dim
            penalized = torch.where(
                logits > 0, logits / repetition_penalty, logits * repetition_penalty
            )
            logits = torch.where(mask, penalized, logits)

        if temperature != 1.0:
            logits = logits / max(temperature, 1e-5)

        # top-k: mask everything outside the top-k logits
        if top_k > 0 and top_k < logits.shape[-1]:
            topk_vals, _ = torch.topk(logits, k=top_k, dim=-1)
            min_topk = topk_vals[..., -1, None]
            logits = torch.where(
                logits < min_topk, torch.full_like(logits, float("-inf")), logits
            )

        # top-p (nucleus): mask tokens past cumulative probability threshold
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = sorted_probs.cumsum(dim=-1)
            # Mark tokens to remove: those whose cumulative prob > top_p.
            # Shift right so the first token whose cumprob exceeds top_p
            # is still kept (the smallest set reaching >= top_p).
            remove = cumprobs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            # Scatter back to original vocab order
            remove_mask = torch.zeros_like(remove)
            remove_mask.scatter_(-1, sorted_idx, remove)
            logits = logits.masked_fill(remove_mask, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        # Guard against rare all-(-inf) rows (shouldn't happen post-topk).
        probs = torch.nan_to_num(probs, nan=0.0)
        if probs.sum(dim=-1).min() <= 0:
            # Fall back to argmax on (pre-filter) logits
            return logits.argmax(dim=-1).to(torch.long)
        return torch.multinomial(probs, num_samples=1).squeeze(-1).to(torch.long)

    # ---- W3: TTS special-token embeddings --------------------------------

    def init_tts_embeds(self, thinker_embed_tokens: nn.Embedding) -> None:
        """Pre-compute TTS pad/bos/eos embeddings using the Thinker's
        embedding table + the Talker's text_projection.

        The HF reference implementation does:
            tts_pad_embed = text_projection(thinker.embed_tokens(tts_pad_token_id))
            tts_bos_embed = text_projection(thinker.embed_tokens(tts_bos_token_id))
            tts_eos_embed = text_projection(thinker.embed_tokens(tts_eos_token_id))

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

    def _get_tts_pad_embed(self, device: torch.device) -> torch.Tensor:
        """Return the TTS pad embedding (Thinker embed -> text_projection).

        Falls back to zeros if init_tts_embeds() has not been called.
        """
        if self._tts_pad_embed_cached is not None:
            return self._tts_pad_embed_cached.to(device).unsqueeze(0)
        # Fallback: zeros (matches old behaviour before W3 fix)
        return torch.zeros(1, self.config.talker_hidden_size, device=device)

    def _get_tts_eos_embed(self, device: torch.device) -> torch.Tensor:
        """Return the TTS eos embedding (Thinker embed -> text_projection).

        Falls back to tts_pad_embed if init_tts_embeds() has not been called.
        """
        if self._tts_eos_embed_cached is not None:
            return self._tts_eos_embed_cached.to(device).unsqueeze(0)
        return self._get_tts_pad_embed(device)

    def _get_tts_bos_embed(self, device: torch.device) -> torch.Tensor:
        """Return the TTS bos embedding (Thinker embed -> text_projection).

        Falls back to Talker's own embed_tokens (old incorrect behaviour)
        if init_tts_embeds() has not been called.
        """
        if self._tts_bos_embed_cached is not None:
            return self._tts_bos_embed_cached.to(device).unsqueeze(0)
        # Fallback: zero vector (init_tts_embeds should be called during setup)
        return torch.zeros(
            1, self.config.talker_hidden_size, device=device,
            dtype=next(self.model.parameters()).dtype,
        )
    
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
        else:  # talker_decode
            return self._preprocess_decode(
                cache_manager, per_request_inputs, request_ids, per_request_info
            )

    # ---- talker_prefill ----
    def _preprocess_prefill(
        self,
        cache_manager: BatchedCacheManager,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, torch.Tensor]:
        """Build Talker prefill from Thinker states chunk.

        Non-last chunks: project states, plan prefill, forward fills KV cache only.
        Last chunk (is_last_prefill=True): build assistant prefix, sample first token.
        """
        assert len(per_request_inputs) == 1, (
            "Talker prefill processes one request at a time"
        )
        device = next(self.model.parameters()).device
        rid = request_ids[0]
        inputs = per_request_inputs[0]
        info = per_request_info[rid]

        is_last_prefill = info.step_metadata.get("is_last_prefill", False)
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

        if not is_last_prefill:
            seq_len = projected.shape[0]
            cache_manager.plan_attention(
                seq_lens=[seq_len], is_causal=True, label="main"
            )
            cache_manager.plan_rope(
                seq_lens=[seq_len], pos_ids=None, label="main"
            )

            return {
                "input_embeds": projected,
                "is_last_prefill": False,
                "seq_lens": [seq_len],
            }

        # ---- Last prefill: build assistant prefix ----
        tc = self.config.talker

        # Build assistant prefix (matching HF/sglang-omni/vllm-omni pattern):
        # Text hidden: [pad*4, bos, proj[3]] (9 tokens)
        # Codec hidden: [codec_embed(nothink, think_bos, think_eos,
        #                speaker, pad, bos)] (9 tokens)
        # (note that the assistant prefix was handled in the previous prefill stage)

        # Text part of assistant prefix
        # W3: pad and bos embeddings use Thinker embed -> text_projection
        # (via pre-computed cached values from init_tts_embeds)
        pad_embed = self._get_tts_pad_embed(device).expand(4, -1)  # 4 pad tokens
        bos_text_embed = self._get_tts_bos_embed(device)           # 1 bos token

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
            "is_last_prefill": True,
            "seq_lens": [seq_len],
        }

    # ---- talker_decode ----

    def _preprocess_decode(self, cache_manager, per_request_inputs, request_ids, per_request_info):
        """Build next decode step: re-embed all_codes + thinker_states.

        Matches HF's ``Qwen3OmniMoeTalkerForConditionalGeneration.prepare_inputs_for_generation``
        autoregressive path (see modeling_qwen3_omni_moe.py ~line 3270):

            codec_hiddens = [last_id_hidden] + mid_residual_hiddens + [last_residual_hidden]
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)
            inputs_embeds = inputs_embeds + trailing_text_hidden[generation_step]

        i.e. the next Talker step's input is the SUM of all ``num_code_groups``
        codec-layer embeddings (layer-0 via the Talker's own codec_embedding;
        layers 1..num_code_groups-1 via the Code Predictor's per-layer
        codec_embedding ModuleList), with the thinker's projected text hidden
        added on top.  Both sglang-omni and vllm-omni follow the same pattern.

        Steps:
        1. Re-embed all_codes into codec_embed_sum (layer-0 + 15 residuals)
        2. Get thinker_states from normal graph input (may be empty after Thinker EOS)
        3. Project thinker_states via text_projection, or use tts_pad_embed if empty
        4. input_embed = codec_embed_sum + text_hidden
        """
        device = next(self.model.parameters()).device
        all_embeds = []
        seq_lens = []

        # Code Predictor's residual embedding ModuleList is at
        # ``code_predictor.model.codec_embedding`` -- NOT
        # ``code_predictor.codec_embedding`` (which does not exist).  The
        # previous code used the wrong path guarded by ``hasattr``, which
        # silently skipped the residual summing entirely and caused the
        # Talker to drift into garbled output after a few decode steps.
        cp_residual_embeddings = self.code_predictor.model.codec_embedding

        for inp, rid in zip(per_request_inputs, request_ids):
            # 1. Re-embed all_codes
            all_codes = inp["all_codes"][0].to(device)
            if all_codes.dim() == 2:
                all_codes = all_codes.squeeze(0)

            # Layer-0 via the Talker's own codec_embedding (vocab=3072).
            layer0_code = all_codes[0:1]
            codec_embed_sum = self.model.model.codec_embedding(layer0_code)

            # Layers 1..num_code_groups-1 via the Code Predictor's per-layer
            # embedding ModuleList (each vocab=2048).  Residual layer ``i``
            # uses ``cp_residual_embeddings[i - 1]`` (layer 1 -> index 0,
            # layer 2 -> index 1, ..., layer 15 -> index 14).
            num_groups = min(all_codes.shape[0], self.config.num_code_groups)
            for i in range(1, num_groups):
                code_i = all_codes[i:i+1]
                emb_i = cp_residual_embeddings[i - 1](code_i).to(codec_embed_sum.dtype)
                codec_embed_sum = codec_embed_sum + emb_i

            # 2. Get thinker_states from normal graph input (stream).
            #
            # Alignment with vllm-omni: the last prefill consumed
            # thinker_decode_0 (first generated token) for prefix position 8.
            # trailing_text_hidden in vllm starts at assistant_hidden[4:] =
            # second generated token.  Our stream delivers thinker_decode_1
            # (second generated token) here — matching vllm's
            # trailing_text_hidden[0].  No delay buffer needed.
            thinker_states_list = inp.get("thinker_states", [])
            if thinker_states_list and thinker_states_list[0] is not None:
                thinker_state = thinker_states_list[0].to(device)
            else:
                thinker_state = None

            # Project thinker_state → text_hidden via text_projection.
            if thinker_state is not None:
                thinker_hidden = self.config.thinker_hidden_size
                if thinker_state.dim() >= 1 and thinker_state.shape[-1] >= thinker_hidden:
                    layer_0 = thinker_state[..., :thinker_hidden]
                    if layer_0.dim() == 1:
                        layer_0 = layer_0.unsqueeze(0)
                    text_hidden = self.model.text_projection(layer_0)
                    if text_hidden.shape[0] > 1:
                        text_hidden = text_hidden[-1:]
                else:
                    text_hidden = self._get_tts_pad_embed(device)
            else:
                # Empty thinker_states — Thinker has finished.
                # HF/vllm/sglang append tts_eos_embed at the end of
                # trailing_text_hidden so the Talker gets a "text done"
                # signal before switching to tts_pad_embed.  We replicate
                # that by using tts_eos_embed for the FIRST empty step,
                # then tts_pad_embed for all subsequent steps.
                if rid not in self._eos_embed_sent:
                    text_hidden = self._get_tts_eos_embed(device)
                    self._eos_embed_sent.add(rid)
                else:
                    text_hidden = self._get_tts_pad_embed(device)

            # Ensure text_hidden is (1, hidden)
            if text_hidden.dim() == 1:
                text_hidden = text_hidden.unsqueeze(0)

            # 3. input_embed = codec_embed_sum + text_hidden
            input_embed = codec_embed_sum + text_hidden
            all_embeds.append(input_embed)
            seq_lens.append(1)

        input_embeds = torch.cat(all_embeds, dim=0)
        cache_manager.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        return {"input_embeds": input_embeds, "seq_lens": seq_lens}

    # ---- forward ----

    def forward(
        self,
        request_info: CurrentForwardPassInfo,
        graph_walk: str = "",
        cache_handle: BatchedCacheManager | None = None,
        input_embeds: torch.Tensor | None = None,
        is_last_prefill: bool = False,
        **kwargs,
    ) -> NameToTensorList:
        """Run Talker forward, optionally sample codec token and run Code Predictor.

        Non-last prefill: fill KV cache only, return empty dict.
        Last prefill / decode: run transformer, sample layer-0 codec token,
        run Code Predictor for 31 residual codes, return logits + all_codes.
        """
        cache_handle.set_active_label("main")

        # Check for non-last prefill (KV-cache-only step)
        if graph_walk == "talker_prefill" and not is_last_prefill:
            self.model(input_embeds=input_embeds, cache_handle=cache_handle)
            return {}

        # Normal forward (last prefill or decode)
        hidden = self.model(
            input_embeds=input_embeds, cache_handle=cache_handle
        )
        last_hidden = hidden[-1:, :]  # (1, hidden_size)

        # Layer-0 codec logits: (1, codec_vocab=3072)
        logits = self.model.codec_head(last_hidden)

        # Suppress the top-1024 ID region of the Talker vocab (reserved for
        # special tokens; invalid as Code2Wav inputs) except codec_eos.
        suppress_mask = self._get_suppress_mask(logits.shape[-1], logits.device)
        logits = logits.masked_fill(suppress_mask.unsqueeze(0), float("-inf"))

        # Per-request seen-token mask for HF-style repetition penalty.
        # Sequential forward path has exactly one request in the batch.
        rid = cache_handle.request_ids[0] if cache_handle.request_ids else ""
        seen_mask = self._get_seen_mask(rid, logits.shape[-1], logits.device)

        # Stochastic layer-0 sampling (HF defaults: temperature=0.9, top_k=50,
        # top_p=1.0, repetition_penalty=1.05).  Argmax produces flat/garbled
        # audio — speech codec LMs require stochastic decoding for natural
        # output.  We sample INSIDE the submodule and return ``new_token``
        # directly so the AR engine's generic sampler does NOT re-sample from
        # logits (which would give a different token than the one the Code
        # Predictor was conditioned on).  Without repetition_penalty the
        # Talker drifts into loops and never emits codec_eos.
        layer0_code = self._top_k_top_p_sample(
            logits,
            temperature=self._talker_temperature,
            top_k=self._talker_top_k,
            top_p=self._talker_top_p,
            repetition_penalty=self._talker_repetition_penalty,
            seen_token_mask=seen_mask,
        )  # (1,)

        # Record the sampled token so the next step's repetition penalty
        # biases against it.
        self._mark_seen(rid, layer0_code)

        # Run Code Predictor for residual codebook layers (float32 precision)
        all_codes = self._run_code_predictor(last_hidden, layer0_code)

        return {
            # Return ``new_token`` directly — the AR engine routes this to the
            # next Talker step's input_ids. No ``logits`` key so ar_engine's
            # _sample_decode_outputs skips re-sampling.
            "new_token": [layer0_code],
            "all_codes": [all_codes],          # 16 code IDs, persisted for next step
            "codec_tokens": [all_codes],       # Streamed to Code2Wav
        }

    def _run_code_predictor(
        self,
        last_hidden: torch.Tensor,
        layer0_code: torch.Tensor,
    ) -> torch.Tensor:
        """Run Code Predictor for residual codebook layers 1..(num_code_groups-1).

        For Qwen3-Omni that's layers 1..15 (15 residual layers) since
        ``num_code_groups = 16``.

        Uses float32 precision for numerical correctness (the Code Predictor
        is a small 5-layer transformer that is sensitive to precision).
        No persistent KV cache -- each step is independent.

        Args:
            last_hidden: Talker's last hidden state, shape (1, hidden_size).
            layer0_code: Sampled layer-0 codec token ID, shape (1,).

        Returns:
            all_codes: tensor of shape (num_code_groups,) with all codec IDs
            (layer-0 + residual layers).
        """
        num_groups = self.config.num_code_groups
        device = last_hidden.device
        all_codes = torch.zeros(num_groups, dtype=torch.long, device=device)
        all_codes[0] = layer0_code

        if num_groups <= 1:
            return all_codes

        # Disable autocast for float32 Code Predictor inference.  HF and
        # vllm-omni found that fused/autocast kernels degrade audio quality
        # for the small (5-layer) Code Predictor.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            cp = self.code_predictor

            # IMPORTANT: Two DIFFERENT embedding tables are involved here.
            #
            #   1. The TALKER's ``codec_embedding`` is an ``nn.Embedding``
            #      with ``vocab_size = talker_text.vocab_size = 3072``.
            #      It's used to embed the LAYER-0 codec token that the
            #      Talker's ``codec_head`` sampled (in [0, 3072)).
            #
            #   2. The CODE PREDICTOR's ``codec_embedding`` is an
            #      ``nn.ModuleList`` of (num_code_groups - 1) = 15
            #      ``nn.Embedding`` instances, each with
            #      ``vocab_size = code_predictor.vocab_size = 2048``.
            #      These embed the RESIDUAL codes for layers 1..15, which
            #      the Code Predictor AR-samples from its per-layer
            #      ``lm_head[k]`` (each with vocab=2048).
            #
            # We previously used ``code_predictor.codec_embedding[0]`` to
            # embed the layer-0 code, but the layer-0 code is from the
            # Talker's 3072-vocab and can be >= 2048, which triggers an
            # out-of-range embedding lookup and a CUDA device-side assert.
            # The fix: use the TALKER's codec_embedding for layer-0, and
            # the Code Predictor's codec_embedding[k] only for residual
            # layers.  Both hidden_sizes are 1024 so the tensors are
            # compatible for concatenation.
            talker_codec_embedding = self.model.model.codec_embedding
            cp_residual_embeddings = cp.model.codec_embedding
            lm_heads = cp.lm_head

            # Build initial input: [last_hidden, layer0_embed], shape (1, 2, H).
            cp_dtype = next(cp.parameters()).dtype
            last_hidden_cp = last_hidden.to(cp_dtype).unsqueeze(0)  # (1, 1, H)
            # Layer-0 is embedded via the Talker's codec_embedding (vocab=3072).
            layer0_embed = talker_codec_embedding(
                layer0_code.unsqueeze(0)
            ).to(cp_dtype)  # (1, 1, H)
            cp_input = torch.cat(
                [last_hidden_cp, layer0_embed], dim=1,
            )  # (1, 2, H)

            # AR loop for residual layers 1 through (num_groups - 1).  At
            # step ``group_idx``, ``lm_heads[group_idx - 1]`` predicts the
            # layer-``group_idx`` residual code, and
            # ``cp_residual_embeddings[group_idx - 1]`` embeds it for the
            # next iteration.  Note: residual layer k uses index (k - 1)
            # into the ModuleList (layer 1 -> index 0, layer 2 -> index 1,
            # ..., layer 31 -> index 30).
            #
            # We re-prefill the entire growing sequence each step (no
            # persistent KV cache).  This is O(N^2) but the predictor is
            # tiny (5 layers, ~80M params, max 31 steps), and matches
            # vllm-omni's reference implementation for numerical fidelity.
            for group_idx in range(1, num_groups):
                # Forward through the Code Predictor's inner model.
                # ``cp.model`` is a Qwen3OmniMoeTalkerCodePredictorModel
                # which accepts ``inputs_embeds`` and returns a
                # BaseModelOutputWithPast.
                outputs = cp.model(
                    inputs_embeds=cp_input,
                    use_cache=False,
                )
                hidden_states = outputs.last_hidden_state  # (1, seq, H)

                # Logits for this residual layer (only the last position)
                cp_logits = lm_heads[group_idx - 1](
                    hidden_states[:, -1:, :]
                )  # (1, 1, vocab=2048)

                # Stochastic sampling — HF uses do_sample=True, top_k=50,
                # top_p=0.8.  Argmax residuals cause mode collapse (each
                # residual layer picks the "most probable" code, which biases
                # toward silence/DC and produces garbled/buzzing output).
                cp_logits_2d = cp_logits.squeeze(1)  # (1, vocab)
                code_i = self._top_k_top_p_sample(
                    cp_logits_2d.float(),
                    temperature=self._cp_temperature,
                    top_k=self._cp_top_k,
                    top_p=self._cp_top_p,
                ).squeeze()  # scalar in [0, 2048)
                all_codes[group_idx] = code_i

                # Embed the sampled residual code for the next iteration.
                # Residual layer ``group_idx`` uses index ``group_idx - 1``
                # in the Code Predictor's ModuleList.
                if group_idx < num_groups - 1:
                    next_embed = cp_residual_embeddings[group_idx - 1](
                        code_i.view(1, 1)
                    ).to(cp_dtype)  # (1, 1, H)
                    cp_input = torch.cat([cp_input, next_embed], dim=1)

        return all_codes

    # ---- batching ----

    def can_batch(self, batch: NodeBatch) -> bool:
        return batch.graph_walk == "talker_decode"
    
    def get_cuda_graph_configs(self, device: torch.device) -> list[CudaGraphConfig]:
        """Return dummy inputs for CUDA graph capture, or None if this walk
        doesn't support CUDA graphs.

        Default: returns text_inputs for "decode" walks. Override in subclasses
        for walks with different input names (e.g., Qwen3-Omni Thinker uses
        "input_embeds" and "cos_sin_3d"; Talker uses "input_embeds").
        """
        num_groups = self.config.num_code_groups
        return [
            # CudaGraphConfig(
            #     graph_walk="talker_decode", requires_cfg=False, labels=["main"],
            #     dummy_capture_inputs=[{
            #         "all_codes": [torch.zeros(num_groups, dtype=torch.long, device=device)],
            #         "thinker_states": [],
            #     }],
            #     compile=False
            # ),
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
        """Batched talker_decode: batch the transformer, per-request Code Predictor.

        The Talker transformer runs once on all requests (each seq_len=1).
        The Code Predictor then runs per-request (31 sequential AR steps
        can't be batched across different code histories).
        """
        assert graph_walk == "talker_decode"

        input_embeds = packed_inputs["input_embeds"]  # (batch, hidden)

        cache_manager.set_active_label("main")

        # Batched Talker transformer forward
        hidden = self.model(
            input_embeds=input_embeds, cache_handle=cache_manager
        )
        # hidden: (batch, hidden_size) — one token per request

        # Batched layer-0 codec logits
        logits = self.model.codec_head(hidden)  # (batch, codec_vocab)

        # Per-request: suppress special-token region, stochastic sample,
        # then run Code Predictor with the sampled layer-0 token.
        suppress_mask = self._get_suppress_mask(logits.shape[-1], logits.device)
        request_ids = cache_manager.request_ids
        result: dict[str, NameToTensorList] = {}

        for i, rid in enumerate(request_ids):
            last_hidden_i = hidden[i : i + 1]  # (1, hidden)
            logits_i = logits[i : i + 1]        # (1, codec_vocab)

            logits_i = logits_i.masked_fill(
                suppress_mask.unsqueeze(0), float("-inf")
            )
            seen_mask = self._get_seen_mask(rid, logits_i.shape[-1], logits_i.device)
            layer0_code = self._top_k_top_p_sample(
                logits_i,
                temperature=self._talker_temperature,
                top_k=self._talker_top_k,
                top_p=self._talker_top_p,
                repetition_penalty=self._talker_repetition_penalty,
                seen_token_mask=seen_mask,
            )  # (1,)
            self._mark_seen(rid, layer0_code)
            
            all_codes = self._run_code_predictor(last_hidden_i, layer0_code)

            result[rid] = {
                "new_token": [layer0_code],
                "all_codes": [all_codes],
                "codec_tokens": [all_codes],
            }

        return result

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str]:
        return ["main"]

    # ---- cleanup ----

    def cleanup_request(self, request_id: str) -> None:
        """Remove per-request state when a request completes."""
        self._seen_layer0_mask.pop(request_id, None)
        self._prefill_conv_tail.pop(request_id, None)
        self._eos_embed_sent.discard(request_id)


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
        graph_walk: str,
        cache_manager: BatchedCacheManager | None = None,
        per_request_inputs: list[NameToTensorList] | None = None,
        request_ids: list[str] | None = None,
        per_request_info: dict[str, CurrentForwardPassInfo] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Unpack codec_tokens from StreamBuffer chunk.

        Selects the first ``num_quantizers`` (16) of the 32 code groups,
        transposes to [1, num_quantizers, num_frames]
        """
        assert len(per_request_inputs) == 1, (
            "Code2Wav processes one request at a time"
        )
        rid = request_ids[0]
        inputs = per_request_inputs[0]

        # codec_tokens: accumulated from StreamBuffer
        # Shape varies: could be (num_frames, num_code_groups) or (num_frames,)
        codec_tokens = inputs["codec_tokens"][0]
        if isinstance(codec_tokens, dict):
            codec_tokens = codec_tokens.get("data", codec_tokens)

        num_quantizers = self.config.code2wav.num_quantizers  # 16

        # Reshape to (num_frames, num_code_groups) if flat
        if codec_tokens.dim() == 1:
            num_groups = self.config.num_code_groups  # 16 (Qwen3-Omni)
            if codec_tokens.shape[0] % num_groups == 0:
                codec_tokens = codec_tokens.view(-1, num_groups)
            else:
                # Single frame
                codec_tokens = codec_tokens.unsqueeze(0)

        # Filter out codec_eos frames — the vocoder should not decode EOS tokens.
        # EOS is identified by the layer-0 code (first column).
        codec_eos = self.config.talker.codec_eos_token_id
        if codec_tokens.dim() == 2 and codec_tokens.shape[0] > 0:
            eos_mask = codec_tokens[:, 0] == codec_eos
            if eos_mask.any():
                codec_tokens = codec_tokens[~eos_mask]
                if codec_tokens.shape[0] == 0:
                    return {"request_id": rid, "codec_tokens": torch.empty(0)}

        # Select first num_quantizers codebook layers
        if codec_tokens.shape[-1] > num_quantizers:
            codec_tokens = codec_tokens[..., :num_quantizers]

        # Transpose to [1, num_quantizers, num_frames] for Code2Wav
        codec_tokens = codec_tokens.T.unsqueeze(0)  # (1, Q, T)

        return {
            "request_id": rid,
            "codec_tokens": codec_tokens,
        }

    def forward(
        self,
        request_id: str | None = None,
        codec_tokens: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Run Code2Wav vocoder, trim left-context overlap, return audio chunk.

        The Talker→Code2Wav StreamBuffer uses a sliding-window policy with
        ``window=chunk_size + left_context_size`` (325) and ``stride=chunk_size``
        (300), so every popped chunk contains ``left_context_size`` (25) frames
        of overlap from the previous chunk. This overlap acts as the
        convolutional vocoder's "warmup" region and must be trimmed from the
        output of every chunk EXCEPT the first (which has no prior audio to
        overlap with).

        Mirrors HF's ``Qwen3OmniMoeCode2Wav.chunked_decode``:
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])

        Returns:
            {"audio_chunk": [int16 PCM tensor]} or {} if input empty.
        """
        if codec_tokens is None or codec_tokens.numel() == 0:
            return {}

        # Run the ConvNet vocoder
        wav = self.code2wav(codec_tokens)

        is_first_chunk = (
            request_id is None or request_id not in self._first_chunk_emitted
        )
        if request_id is not None:
            self._first_chunk_emitted.add(request_id)

        if is_first_chunk:
            # First chunk: no left context to discard — emit the full waveform.
            trimmed_wav = wav
        else:
            # Subsequent chunk: trim the ``left_context_size`` warmup frames
            # from the front of the output (they were already emitted by the
            # previous chunk).
            left_context_size = self.config.code2wav.left_context_size  # 25
            context_samples = left_context_size * self._total_upsample  # 25 * 1920 = 48000
            if wav.shape[-1] > context_samples:
                trimmed_wav = wav[:, :, context_samples:]
            else:
                trimmed_wav = wav

        # Convert to int16 PCM
        audio_int16 = (
            trimmed_wav.clamp(-1, 1) * 32767
        ).to(torch.int16).squeeze().detach()

        return {"audio_chunk": [audio_int16]}

    def can_batch(self, batch: NodeBatch) -> bool:
        return False
