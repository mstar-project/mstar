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
        audio_embeds = self.audio_encoder(audio_features)

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
            pixel_values=pixel_values,
            grid_thw=grid_thw,
        )

        if isinstance(encoder_output, tuple):
            vision_embeds, deepstack = encoder_output
        else:
            vision_embeds = encoder_output
            deepstack = None

        return {
            "vision_embeds": [vision_embeds],
            "deepstack": [deepstack] if deepstack is not None else [torch.tensor([])],
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

        audio_start_id = self.config.thinker.audio_start_token_id
        audio_end_id = self.config.thinker.audio_end_token_id

        for inp, rid in zip(per_request_inputs, request_ids):
            audio_embeds = inp["audio_embeds"][0].to(device)  # (audio_tokens, hidden)
            audio_len = audio_embeds.shape[0]

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

        vision_start_id = self.config.thinker.vision_start_token_id
        vision_end_id = self.config.thinker.vision_end_token_id
        spatial_merge = self.config.vision.spatial_merge_size

        for inp, rid in zip(per_request_inputs, request_ids):
            vision_embeds = inp["vision_embeds"][0].to(device)
            vision_len = vision_embeds.shape[0]

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

        # Pass deepstack through if available (for Thinker layers that need it)
        deepstack = None
        if per_request_inputs and "deepstack" in per_request_inputs[0]:
            deepstack = per_request_inputs[0]["deepstack"][0]

        result = {
            "input_embeds": input_embeds,
            "cos_sin_3d": cos_sin_3d,
            "mrope_section": self.MROPE_SECTION,
            "seq_lens": seq_lens,
        }
        if deepstack is not None:
            result["deepstack"] = deepstack

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
            CudaGraphConfig(
                graph_walk="thinker_decode", requires_cfg=False, labels=["main"],
                dummy_capture_inputs=[{"text_inputs": [torch.zeros(1, dtype=torch.long, device=device)]}]
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

        ``thinker_states`` is only included in a request's outputs when
        that request has ``audio_output=True`` in its step_metadata. Text
        only requests skip it to save cross-partition bandwidth.
        """
        assert graph_walk == "thinker_decode"

        input_embeds = packed_inputs["input_embeds"]  # (batch, hidden)
        cos_sin_3d = packed_inputs.get("cos_sin_3d")
        mrope_section = packed_inputs.get("mrope_section")

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
        sample_token = info.step_metadata.get("sample_token", False)

        # 1. Unpack thinker_states -> split into layer_0 and layer_n
        thinker_states = inputs["thinker_states"][0].to(device)
        thinker_hidden = self.config.thinker_hidden_size
        layer_0_embed = thinker_states[..., :thinker_hidden]
        layer_n_hidden = thinker_states[..., thinker_hidden:]

        # 2. Determine projection based on walk_name (W2).
        # The Thinker walk that produced these states determines whether
        # the tokens are text or multimodal:
        #   prefill_text   / thinker_decode -> text_projection(layer_0)
        #   prefill_audio  / prefill_vision -> hidden_projection(layer_n)
        # This replaces the old multimodal_mask approach which required
        # passing the full mask across partitions.
        walk_name = info.step_metadata.get("walk_name", "")
        multimodal_mask = info.step_metadata.get("multimodal_mask", None)

        # 3. Project Thinker states into Talker space
        if multimodal_mask is not None:
            # Legacy path: explicit mask (kept for backward compatibility)
            multimodal_mask = multimodal_mask.to(device)
            text_mask = ~multimodal_mask
            projected = torch.zeros(
                thinker_states.shape[0], self.config.talker_hidden_size,
                device=device, dtype=thinker_states.dtype,
            )
            if text_mask.any():
                projected[text_mask] = self.model.text_projection(
                    layer_0_embed[text_mask]
                )
            if multimodal_mask.any():
                projected[multimodal_mask] = self.model.hidden_projection(
                    layer_n_hidden[multimodal_mask]
                )
        elif walk_name in ("prefill_audio", "prefill_vision"):
            # Multimodal walk: all tokens are encoder embeddings
            projected = self.model.hidden_projection(layer_n_hidden)
        else:
            # Text walk (prefill_text, thinker_decode) or unknown:
            # all tokens are text -> text_projection(layer_0)
            projected = self.model.text_projection(layer_0_embed)

        if not is_last_prefill:
            # ---- Non-last prefill: KV-cache-only step ----
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
        talker_hidden = self.config.talker_hidden_size

        # Split projected into conversation body vs first decode state
        # The last token in thinker_states is from thinker_decode step 1
        conv_projected = projected[:-1]  # user conversation states
        first_decode_projected = projected[-1:]  # first Thinker decode token

        # Build assistant prefix (matching HF/sglang-omni/vllm-omni pattern):
        # Text hidden: [proj[0], proj[1], proj[2], pad*4, bos, proj[3]] (9 tokens)
        # Codec hidden: [zeros*3, codec_embed(nothink, think_bos, think_eos,
        #                speaker, pad, bos)] (9 tokens)

        # Text part of assistant prefix
        # W3: pad and bos embeddings use Thinker embed -> text_projection
        # (via pre-computed cached values from init_tts_embeds)
        pad_embed = self._get_tts_pad_embed(device).expand(4, -1)  # 4 pad tokens
        bos_text_embed = self._get_tts_bos_embed(device)           # 1 bos token

        # W4 (known limitation): The assistant prefix positions use the
        # last 4 projected Thinker states (conv_projected[-4:]).  This
        # heuristic is correct for standard ChatML templates where the
        # assistant role header ``<|im_start|>assistant\n`` occupies the
        # last 3-4 text tokens before decode starts.  A proper fix would
        # parse the ChatML structure and pass ``assistant_start_idx`` in
        # step_metadata, but that requires forwarding input_ids to the
        # Talker partition.
        n_proj = min(4, conv_projected.shape[0])
        if n_proj >= 4:
            prefix_proj = conv_projected[-4:]  # last 4 projected states
            prefix_proj_start = prefix_proj[:3]  # proj[0:3]
            prefix_proj_end = prefix_proj[3:]    # proj[3:4]
        else:
            # Fallback: repeat last state
            prefix_proj_start = conv_projected[-1:].expand(3, -1)
            prefix_proj_end = conv_projected[-1:]

        text_hidden = torch.cat([
            prefix_proj_start,  # proj[0:3] (3 tokens)
            pad_embed,          # pad * 4   (4 tokens)
            bos_text_embed,     # bos       (1 token)
            prefix_proj_end,    # proj[3:4] (1 token)
        ], dim=0)  # (9, talker_hidden)

        # Codec part of assistant prefix
        codec_zeros = torch.zeros(
            3, talker_hidden, device=device, dtype=text_hidden.dtype
        )
        codec_special_ids = torch.tensor([
            tc.codec_nothink_id,
            tc.codec_think_bos_id,
            tc.codec_think_eos_id,
            tc.codec_pad_id,   # speaker slot
            tc.codec_pad_id,
            tc.codec_bos_id,
        ], device=device, dtype=torch.long)
        codec_special_embeds = self.model.model.codec_embedding(codec_special_ids)

        codec_hidden = torch.cat([
            codec_zeros,          # 3 zero tokens
            codec_special_embeds, # 6 special tokens
        ], dim=0)  # (9, talker_hidden)

        # Combine text and codec parts
        assistant_prefix = text_hidden + codec_hidden  # (9, talker_hidden)

        # Full input: conversation body + assistant prefix
        # (conv_projected was already projected; assistant_prefix includes both)
        input_embeds = torch.cat([
            conv_projected,     # user conversation states
            assistant_prefix,   # 9-token assistant prefix
        ], dim=0)

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

        1. Re-embed all_codes (32 code IDs) -> codec_embed_sum
        2. Get thinker_states from normal graph input (may be empty after Thinker EOS)
        3. Project thinker_states via text_projection, or use tts_pad_embed if empty
        4. input_embed = codec_embed_sum + text_hidden
        """
        device = next(self.model.parameters()).device
        all_embeds = []
        seq_lens = []

        for inp, rid in zip(per_request_inputs, request_ids):
            # 1. Re-embed all_codes
            all_codes = inp["all_codes"][0].to(device)
            if all_codes.dim() == 2:
                all_codes = all_codes.squeeze(0)
            layer0_code = all_codes[0:1]
            codec_embed_sum = self.model.model.codec_embedding(layer0_code)
            # Sum layers 1-31 from Code Predictor embeddings
            if hasattr(self.code_predictor, 'codec_embedding') and all_codes.shape[0] > 1:
                for i in range(1, min(all_codes.shape[0], self.config.num_code_groups)):
                    code_i = all_codes[i:i+1]
                    emb_i = self.code_predictor.codec_embedding[i - 1](code_i)
                    codec_embed_sum = codec_embed_sum + emb_i

            # 2. Get thinker_states from normal graph input
            thinker_states_list = inp.get("thinker_states", [])
            if thinker_states_list and thinker_states_list[0] is not None:
                thinker_state = thinker_states_list[0].to(device)
                # Split into layer_0 and layer_n, project layer_0
                thinker_hidden = self.config.thinker_hidden_size
                if thinker_state.dim() >= 1 and thinker_state.shape[-1] >= thinker_hidden:
                    layer_0 = thinker_state[..., :thinker_hidden]
                    if layer_0.dim() == 1:
                        layer_0 = layer_0.unsqueeze(0)
                    text_hidden = self.model.text_projection(layer_0)
                    # Take last token if multiple
                    if text_hidden.shape[0] > 1:
                        text_hidden = text_hidden[-1:]
                else:
                    text_hidden = self._get_tts_pad_embed(device)
            else:
                # Empty thinker_states (Thinker has finished, or no data yet)
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

        # Layer-0 codec logits
        logits = self.model.codec_head(last_hidden)  # (1, codec_vocab)

        # NOTE: Using argmax as approximation. The AR engine samples separately
        # for new_token routing. For exact correctness, the Code Predictor should
        # use the sampled token, but that requires post-sampling execution.
        layer0_code = logits.argmax(dim=-1)  # (1,)

        # Run Code Predictor for residual codebook layers (float32 precision)
        all_codes = self._run_code_predictor(last_hidden, layer0_code)

        return {
            "logits": [logits.squeeze(0)],    # Sampled by AR engine -> "new_token"
            "all_codes": [all_codes],          # 32 code IDs, persisted for next step
            "codec_tokens": [all_codes],       # Streamed to Code2Wav
        }

    def _run_code_predictor(
        self,
        last_hidden: torch.Tensor,
        layer0_code: torch.Tensor,
    ) -> torch.Tensor:
        """Run Code Predictor for residual codebook layers 1-31.

        Uses float32 precision for numerical correctness (the Code Predictor
        is a small 5-layer transformer that is sensitive to precision).
        No persistent KV cache -- each step is independent.

        Args:
            last_hidden: Talker's last hidden state, shape (1, hidden_size).
            layer0_code: Sampled layer-0 codec token ID, shape (1,).

        Returns:
            all_codes: tensor of shape (num_code_groups,) with all 32 codec IDs.
        """
        num_groups = self.config.num_code_groups
        device = last_hidden.device
        all_codes = torch.zeros(num_groups, dtype=torch.long, device=device)
        all_codes[0] = layer0_code.item()

        if num_groups <= 1:
            return all_codes

        # Disable autocast for float32 Code Predictor inference
        with torch.amp.autocast(device_type="cuda", enabled=False):
            last_hidden_f32 = last_hidden.float()

            # Build initial Code Predictor input: [last_hidden, codec_embed(layer0)]
            if hasattr(self.code_predictor, 'codec_embedding'):
                code_embed = self.code_predictor.codec_embedding(layer0_code).float()
            elif hasattr(self.code_predictor, 'codec_embeddings'):
                code_embed = self.code_predictor.codec_embeddings[0](layer0_code).float()
            else:
                # Fallback: use Talker codec_embedding
                code_embed = self.model.model.codec_embedding(layer0_code).float()

            # Project Talker hidden into Code Predictor space if needed
            if hasattr(self.code_predictor, 'input_projection'):
                cp_hidden = self.code_predictor.input_projection(last_hidden_f32)
            else:
                cp_hidden = last_hidden_f32

            cp_input = torch.cat([cp_hidden, code_embed], dim=0)  # (2, hidden)

            # AR loop for layers 1 through (num_groups - 1)
            for group_idx in range(1, num_groups):
                # Run Code Predictor forward (no persistent KV cache)
                if hasattr(self.code_predictor, 'forward'):
                    cp_output = self.code_predictor(cp_input)
                else:
                    cp_output = cp_input

                # Get logits for this codebook layer
                if hasattr(self.code_predictor, 'codec_heads'):
                    cp_logits = self.code_predictor.codec_heads[group_idx - 1](
                        cp_output[-1:]
                    )
                elif hasattr(self.code_predictor, 'codec_head'):
                    cp_logits = self.code_predictor.codec_head(cp_output[-1:])
                else:
                    # Fallback: use a linear head if available
                    cp_logits = cp_output[-1:]

                code_i = cp_logits.argmax(dim=-1).squeeze()
                all_codes[group_idx] = code_i

                # Embed the newly sampled code for next iteration
                if group_idx < num_groups - 1:
                    if hasattr(self.code_predictor, 'codec_embeddings'):
                        next_embed = self.code_predictor.codec_embeddings[group_idx](
                            code_i.unsqueeze(0)
                        ).float()
                    elif hasattr(self.code_predictor, 'codec_embedding'):
                        next_embed = self.code_predictor.codec_embedding(
                            code_i.unsqueeze(0)
                        ).float()
                    else:
                        next_embed = self.model.model.codec_embedding(
                            code_i.unsqueeze(0)
                        ).float()

                    cp_input = torch.cat([cp_input, next_embed], dim=0)

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
            CudaGraphConfig(
                graph_walk="talker_decode", requires_cfg=False, labels=["main"],
                dummy_capture_inputs=[{
                    "all_codes": [torch.zeros(num_groups, dtype=torch.long, device=device)],
                    "thinker_states": [],
                }]
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

        # Per-request: Code Predictor + pack outputs
        request_ids = cache_manager.request_ids
        result: dict[str, NameToTensorList] = {}

        for i, rid in enumerate(request_ids):
            last_hidden_i = hidden[i : i + 1]  # (1, hidden)
            logits_i = logits[i : i + 1]        # (1, codec_vocab)

            layer0_code = logits_i.argmax(dim=-1)  # (1,)
            all_codes = self._run_code_predictor(last_hidden_i, layer0_code)

            result[rid] = {
                "logits": [logits_i.squeeze(0)],
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
        pass


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
            num_groups = self.config.num_code_groups  # 32
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
        request_info: CurrentForwardPassInfo,
        request_id: str = "",
        codec_tokens: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Run Code2Wav vocoder, trim context overlap, return audio chunk.

        Returns:
            {"audio_chunk": [int16 PCM tensor]} or {} if input too short.
        """
        if codec_tokens is None or codec_tokens.numel() == 0:
            logger.warning(
                "Code2Wav: empty codec_tokens for request %s", request_id
            )
            return {}

        logger.debug(
            "Running Code2Wav with codec_tokens shape=%s for request %s",
            codec_tokens.shape, request_id,
        )

        # Run the ConvNet vocoder
        wav = self.code2wav(codec_tokens)

        sliding_window = self.config.code2wav.sliding_window
        if codec_tokens.shape[-1] > sliding_window:
            # Trim the overlap region from the output waveform
            # The overlap in waveform samples corresponds to the context frames
            # processed by the upsampling ConvNet
            upsample_rate = 1
            for r in self.config.code2wav.upsample_rates:
                upsample_rate *= r
            context_samples = sliding_window * upsample_rate

            if wav.shape[-1] > context_samples:
                # Trim the full overlap region (matching sglang-omni pattern)
                trimmed_wav = wav[:, :, context_samples:]
            else:
                trimmed_wav = wav
        else:
            # First chunk or short chunk: no trimming needed, store full context
            trimmed_wav = wav

        # Convert to int16 PCM
        audio_int16 = (
            trimmed_wav.clamp(-1, 1) * 32767
        ).to(torch.int16).squeeze().detach()

        return {"audio_chunk": [audio_int16]}

    def can_batch(self, batch: NodeBatch) -> bool:
        return False
