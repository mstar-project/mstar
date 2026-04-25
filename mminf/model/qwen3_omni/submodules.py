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
from typing import Any, Optional

import torch
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import NodeBatch
from mminf.engine.cache_manager import BatchedCacheManager
from mminf.engine.code_predictor_engine import CodePredictorEngineInputs, CodePredictorSubmodule, MTPSampler
from mminf.engine.cuda_graph_config import FlashInferPackedCudaGraphConfig
from mminf.engine.cuda_graph_runner import BasicBatchedCudaGraphConfig
from mminf.engine.kv_store import PositionInfo
from mminf.model.qwen3_omni.components.rope import (
    compute_3d_cos_sin,
    compute_rope_freqs,
    get_rope_index_audio,
    get_rope_index_text,
    get_rope_index_vision,
)
from mminf.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor, Qwen3OmniTalkerModel
from mminf.model.qwen3_omni.config import Qwen3OmniModelConfig
from mminf.model.submodule_base import NodeSubmodule, ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeInputs

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
        device = self.get_device()
        if self._audio_bos_embed is None or self._audio_eos_embed is None:
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

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {}
    ) -> ARNodeInputs:
        device = self.get_device()
        start_pos = pos_info.get("main", PositionInfo()).position_id_start
        if graph_walk == "thinker_decode":
            # Get previous token ID from text_inputs
            token_id = inputs["text_inputs"][0].to(device)  # (1,) or scalar
            if token_id.dim() == 0:
                token_id = token_id.unsqueeze(0)
            embeds = self.model.model.embed_tokens(token_id)

            # Next MRoPE position for all 3 components: read from the
            # per-request cache-manager state (kept in sync by the
            # post-forward ``advance_seq_lens`` call in ``thinker.py``).
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
            text_ids = inputs["text_inputs"][0].to(device)  # (seq_len,)
            embeds = self.model.model.embed_tokens(text_ids)
            seq_len = text_ids.shape[0]

            # Compute 3D MRoPE position IDs for a pure-text span.  Each
            # prefill graph walk is single-modality so we use the simple
            # per-modality helper instead of the full HF parser.
            #
            # ``start_pos`` is the next MRoPE position for this request,
            # carried forward across walks by ``state.position_id_start``
            # (advanced post-forward by ``advance_seq_lens``).
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
            audio_embeds = inputs["audio_embeds"][0].to(device)  # (audio_tokens, hidden)
            audio_len = audio_embeds.shape[0]

            mm_mask = torch.ones(audio_len + 2, dtype=torch.bool, device=device)
            mm_mask[[0, -1]] = 0
            masks_for_talker = torch.stack([
                mm_mask,
                ~mm_mask
            ])

            wrapped_embeds = self._wrap_audio_input(audio_embeds)
            seq_len = audio_len + 2
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
            return ARNodeInputs(
                input_seq_len=seq_len,
                input_embeds=wrapped_embeds,
                custom_pos_ids=pos_ids,
                tensor_inputs={
                    "masks_for_talker": masks_for_talker
                }
            )

        if graph_walk == "prefill_vision":
            vision_embeds = inputs["vision_embeds"][0].to(device)
            vision_len = vision_embeds.shape[0]

            mm_mask = torch.ones(vision_len + 2, dtype=torch.bool, device=device)
            mm_mask[[0, -1]] = 0
            masks_for_talker = torch.stack([
                mm_mask,
                ~mm_mask
            ])

            wrapped_embeds = self._wrap_vision_input(vision_embeds)            
            total_len = vision_len + 2
            # Vision tokens use spatial 3D positions (temporal constant,
            # h/w from the spatial grid after merging).  If a proper
            # ``image_grid_thw`` is available, use ``get_rope_index_vision``;
            # otherwise fall back to a 1-D sequence (test path without
            # AutoImageProcessor).
            grid_thw = inputs.get("image_grid_thw", [None])[0]
            seconds_per_grid = inputs.get("video_second_per_grid", [])
            seconds_per_grid = seconds_per_grid[0].item() if seconds_per_grid else None
            vision_pos_ids = get_rope_index_vision(
                grid_thw.to(device),
                start_pos + 1,  # leave room for the BOS token
                position_id_per_seconds=self.config.thinker.position_id_per_seconds,
                device=device,
                spatial_merge_size=self.config.vision.spatial_merge_size,
                seconds_per_grid=seconds_per_grid
            )

            # Sentinel token positions (text-like).
            start_pos_ids = get_rope_index_text(1, start_pos, device)
            end_pos_base = float(vision_pos_ids.max().item()) + 1
            end_pos_ids = get_rope_index_text(1, end_pos_base, device)

            pos_ids = torch.cat(
                [start_pos_ids, vision_pos_ids, end_pos_ids], dim=1
            )

            # Next MRoPE position after this vision block is ``end_pos_base
            # + 1`` (one past the EOS token).  ``advance_seq_lens`` by
            # default advances ``position_id_start`` by ``seq_len``, which
            # for vision (= vision_len + 2) is typically smaller than the
            # 3D-grid span.  Emit the correct per-request advance so the
            # Thinker forward can pass ``pos_id_ns`` through.
            mrope_pos_advance = int(end_pos_base + 1 - start_pos)
            deepstack = inputs["deepstack"]

            return ARNodeInputs(
                input_seq_len=total_len,
                input_embeds=wrapped_embeds,
                custom_pos_ids=pos_ids,
                tensor_inputs={
                    "masks_for_talker": masks_for_talker,
                    "mrope_pos_advance": mrope_pos_advance,
                    "deepstack": deepstack,
                    "visual_pos_masks": mm_mask
                }
            )

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
        if graph_walk == "prefill_vision":
            assert len(inputs) == 1, \
                "Batching not implemented for Thinker vision prefill"
            inp = inputs[0]
            extra_inputs["deepstack"] = inp.tensor_inputs.get("deepstack", torch.tensor([]))
            extra_inputs["visual_pos_masks"] = inp.tensor_inputs.get(
                "visual_pos_masks", torch.tensor([]))
            extra_inputs["mrope_pos_advance"] = [
                inp.tensor_inputs.get("mrope_pos_advance", 0)
            ]

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
    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        return batch.graph_walk == "thinker_decode"

    PREFILL_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048]
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]

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

    def get_cuda_graph_configs(self, device: torch.device):
        """Declare CUDA graph captures for ``thinker_decode`` and ``prefill_text``.

        Decode uses ``BasicBatchedCudaGraphConfig`` (one capture per bs;
        runner clones single_request_inputs and runs preprocess itself).
        Prefill uses ``FlashInferPackedCudaGraphConfig`` (one capture per
        (bs, num_tokens) bucket; the dict here IS the post-preprocess
        packed input — runner does not call preprocess at capture).

        ``capture_batch_sizes`` is kept small for both because each capture
        allocates persistent FlashInfer wrappers + static buffers for the
        full 30B Thinker; revisit after profiling real deployments.
        """
        prefill_packed = {
            num_tokens: self._build_prefill_text_packed(num_tokens, device)
            for num_tokens in self.PREFILL_TOKEN_BUCKETS
        }
        return [
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
                capture_batch_sizes=[1, 2, 4, 8, 16],
            ),
            FlashInferPackedCudaGraphConfig(
                capture_graph_walk="prefill_text",
                replay_graph_walks=["prefill_text"],
                packed_seq_len_to_inputs=prefill_packed,
                requires_cfg=False,
                labels=["main"],
                compile=True,
                causal_attention=True,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
        ]

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
        """Batched Thinker forward shared between ``thinker_decode`` and ``prefill_text``.

        Decode path (1 token per request, ``hidden`` shape ``(bs, hidden)``):
          Always packs ``thinker_states`` + ``thinker_mask`` in every per-rid
          output dict so the captured CUDA graph has a static output shape
          regardless of request metadata. Per-rid filtering (dropping
          ``thinker_states`` / ``thinker_mask`` for ``audio_output=False``
          requests) happens OUTSIDE the captured region via
          ``filter_batched_output``.

        Prefill_text path (multi-token-per-request, ``hidden`` shape
        ``(total_tokens, hidden)``):
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
        """
        assert graph_walk in ("thinker_decode", "prefill_text")

        # Packed dict from FlashInferPackedCudaGraphConfig is tensor-only by
        # design (the runner's static-buffer interning skips non-tensor
        # entries), so for prefill_text we recover mrope_section from the
        # class constant when the kwarg is missing. Decode goes through
        # preprocess which does pass it explicitly.
        if mrope_section is None and graph_walk == "prefill_text":
            mrope_section = self.MROPE_SECTION

        cos_sin_3d = (cos_3d, sin_3d) if cos_3d is not None else None
        cache_manager = engine_inputs.cache_manager
        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cache_handle=cache_manager,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
            mrope_pos_advance=mrope_pos_advance,
        )

        if graph_walk == "prefill_text":
            qo_indptr_buf = cache_manager.get_qo_indptr_buf("main")
            assert qo_indptr_buf is not None, (
                "prefill_text forward_batched requires a CUDA-graph "
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
        if not request_info.step_metadata.get("audio_output", True):
            # drop thinker_states and thinker_match
            outputs.pop("thinker_states", None)
            outputs.pop("thinker_mask", None)

        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]
        token = outputs["new_token"][0].item()
        eos_token_id = self.config.im_end_token_id
        if (eos_token_id is not None and eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("thinker_decode_loop", 0) + 1 >= request_info.max_tokens):
            request_info.register_loop_stop("thinker_decode_loop")
    
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

class TalkerLLMSubmodule(ARNodeSubmodule):
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
    
    def _split_thinker_states(
        self, thinker_states: torch.Tensor
    ):
        thinker_hidden = self.config.thinker_hidden_size
        layer_0_embed = thinker_states[..., :thinker_hidden]
        layer_n_hidden = thinker_states[..., thinker_hidden:]
        return layer_0_embed, layer_n_hidden
    
    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {},
    ) -> ARNodeInputs:
        device = self.get_device()
    
        thinker_hidden = self.config.thinker_hidden_size    
        if graph_walk == "talker_prefill":
            thinker_states = inputs["thinker_states"][0].to(device)
            layer_0_embed, layer_n_hidden = self._split_thinker_states(thinker_states)
            mask = inputs["thinker_mask"][0]
            input_embeds = self._get_talker_embeds(
                layer_0_embed=layer_0_embed, layer_n_hidden=layer_n_hidden,
                multimodal_mask=mask[0, :],
                text_inclusion_mask=mask[1, :]
            )
            seq_len = input_embeds.shape[0]

        elif graph_walk == "talker_last_prefill":
            rid = fwd_info.request_id
            thinker_states = inputs["thinker_states"][0].to(device)
            layer_0_embed, _ = self._split_thinker_states(thinker_states)
            projected = self.model.text_projection(layer_0_embed)
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

            speaker = fwd_info.step_metadata.get("voice", "Ethan")
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

        elif graph_walk == "talker_decode":
            dtype = self.model.text_projection.linear_fc1.weight.dtype
            input_embeds = inputs["talker_input_embeds"][0].to(dtype)
            
            thinker_states = inputs.get("thinker_states", [])
            rid = fwd_info.request_id
            if thinker_states:
                thinker_hidden = self.config.thinker_hidden_size
                input_embeds += self.model.text_projection(
                    thinker_states[0][..., :thinker_hidden].to(dtype)
                )
            elif rid not in self._eos_embed_sent:
                input_embeds += self._tts_eos_embed_cached
                self._eos_embed_sent.add(rid)
            else:
                input_embeds += self._tts_pad_embed_cached
            seq_len = 1

        return ARNodeInputs(
            input_embeds=input_embeds,
            input_seq_len=seq_len
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

        extra_args = {}
        if graph_walk != "talker_prefill":
            extra_args["suppress_mask"] = self._get_suppress_mask()
        return {
            "input_embeds": input_embeds,
            **extra_args
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
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        suppress_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        if graph_walk == "talker_prefill":
            return self._forward_prefill(
                cache_handle=engine_inputs.cache_manager,
                input_embeds=input_embeds
            )
        return self._forward_decode_like(
            cache_handle=engine_inputs.cache_manager,
            input_embeds=input_embeds,
            suppress_mask=suppress_mask,
            is_batched_decode=(graph_walk == "talker_decode"),
        )
    
    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        suppress_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        """Batched Talker forward shared between ``talker_decode`` and ``talker_prefill``.

        Decode path (1 token per request, ``hidden`` shape ``(bs, hidden)``):
          Runs the full LLM + codec_head + suppress_mask via _forward_decode_like
          and emits per-rid {last_hidden, logits} entries plus a ``__batched_logits__``
          sentinel for the runner's sample-once fast path.

        Prefill path (multi-token-per-request, ``hidden`` shape ``(total_tokens, hidden)``):
          Runs only the LLM backbone — no codec_head, no sampling. Production
          ``talker_prefill`` exists solely to populate the KV cache for the
          subsequent ``talker_last_prefill`` + ``talker_decode_loop``, so
          ``_forward_prefill`` returns ``{}`` in eager. We expose the post-LLM
          hidden state under the ``__batched_talker_prefill_hidden__`` sentinel
          purely so the parity test can compare graph vs eager hidden activations;
          the runner's _sample_and_remap drops this key (no per-rid dict, no
          __batched_logits__) and returns ``{rid: {} for rid in request_ids}``,
          matching eager.
        """
        assert graph_walk in ("talker_decode", "talker_prefill")
        cache_handle = engine_inputs.cache_manager

        if graph_walk == "talker_prefill":
            hidden = self.model(
                input_embeds=input_embeds,
                cache_handle=cache_handle,
            )
            return {
                "__batched_talker_prefill_hidden__": hidden,
            }

        fwd_out = self._forward_decode_like(
            cache_handle=cache_handle,
            input_embeds=input_embeds,
            suppress_mask=suppress_mask,
            is_batched_decode=True
        )

        outputs = {
            rid: {
                "last_hidden": fwd_out["last_hidden"][0][i:i+1],
                "logits": fwd_out["logits"][0][i:i+1],
            } for i, rid in enumerate(engine_inputs.request_ids)
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
        self._eos_embed_sent.discard(request_id)
    
    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        return batch.graph_walk == "talker_decode"

    TALKER_PREFILL_TOKEN_BUCKETS = [128, 256, 512, 1024]
    TALKER_PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4]

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

    def get_cuda_graph_configs(self, device: torch.device):
        """Declare CUDA graph captures for ``talker_decode`` and ``talker_prefill``.

        Decode uses ``BasicBatchedCudaGraphConfig`` (one capture per bs; runner
        clones single_request_inputs and runs preprocess itself). Prefill uses
        ``FlashInferPackedCudaGraphConfig`` (one capture per (bs, num_tokens)
        bucket; the dict here IS the post-preprocess packed input — runner does
        not call preprocess at capture).
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
                    input_seq_len=1
                ),
                capture_batch_sizes=[1, 2, 4, 8, 16]
            ),
            FlashInferPackedCudaGraphConfig(
                capture_graph_walk="talker_prefill",
                replay_graph_walks=["talker_prefill"],
                packed_seq_len_to_inputs=talker_prefill_packed,
                requires_cfg=False,
                labels=["main"],
                compile=True,
                causal_attention=True,
                capture_batch_sizes=self.TALKER_PREFILL_CAPTURE_BATCH_SIZES,
            ),
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

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        pos_info: dict[str, PositionInfo] = {},
    ) -> ARNodeInputs:
        return ARNodeInputs(
            input_seq_len=1,
            input_ids=inputs["layer0_codes"][0],
            input_embeds=inputs["last_hidden"][0]
        )
    
    def get_num_code_groups(self):
        return self.num_codes
    
    def get_kv_cache_dtype(self):
        return self.talker_code_emb.weight.dtype
    
    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: CodePredictorEngineInputs,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        """"
        last_hidden: ``[bs, hidden]`` final Talker hidden state.
        layer0_codes: ``[bs]`` int64 sampled codebook-0 tokens.

        initialize to zero:
        "all_codes": [bs, num_codes] int64
        "codec_emb_sum": [bs, hidden] fp32
        """
        last_hidden = torch.cat([
                inp.input_embeds for inp in inputs
            ], dim=0)
        return {
            "last_hidden": last_hidden,
            "layer0_codes": torch.cat([
                inp.input_ids for inp in inputs
            ], dim=0),
            "all_codes": torch.zeros(
                (len(inputs), self.num_codes),
                device=inputs[0].input_ids.device, dtype=torch.long
            ),
            "codec_emb_sum": torch.zeros_like(last_hidden),
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: CodePredictorEngineInputs,
        last_hidden: torch.Tensor | None = None,
        layer0_codes: torch.Tensor | None = None,
        all_codes: torch.Tensor | None = None,
        codec_emb_sum: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        kv_cache = engine_inputs.kv_cache
        sampler: MTPSampler = engine_inputs.sampler

        cp = self.code_predictor
        codec_embedding = cp.model.codec_embedding
        lm_head_weight = cp.lm_head_weight

        pos = engine_inputs.init_pos_ids

        embed = self.talker_code_emb(layer0_codes)  # [bs, hidden]
        codec_emb_sum.add_(embed)
        all_codes[:, 0] = layer0_codes

        # forward over [last_hidden] to update kv cache with the Talker's final hidden
        # state as context for the code prediction. This returns nothing because the
        # layer 0 code is already provided by the talker LLM
        cp.forward_depth_unrolled(
            last_hidden.unsqueeze(1), pos, kv_cache, cache_pos=0,
        )
        pos += 1

        for group_idx in range(1, self.num_codes):
            hidden = cp.forward_depth_unrolled(
                embed.unsqueeze(1), pos, kv_cache, cache_pos=group_idx,
            ).squeeze(1)
            pos += 1

            logits = torch.matmul(
                hidden, lm_head_weight[group_idx - 1].t()
            )
            tokens = sampler.sample(logits)
            all_codes[:, group_idx] = tokens
            embed = codec_embedding[group_idx - 1](tokens)
            codec_emb_sum.add_(embed)

        return {
            req_id: {
                "talker_input_embeds": [codec_emb_sum[i:i+1]],
                "codec_tokens": [all_codes[i]],
            } for i, req_id in enumerate(engine_inputs.request_ids)
        }
    
    def _get_dummy_inputs(self, device):
        return ARNodeInputs(
            input_embeds=torch.zeros(
                (1, self.config.talker_hidden_size),
                device=device, dtype=torch.bfloat16
            ),
            input_ids=torch.zeros((1,), device=device, dtype=torch.long),
            input_seq_len=1
        )
    
    def get_cuda_graph_configs(self, device):
        return [
            BasicBatchedCudaGraphConfig(
                capture_graph_walk="talker_decode",
                replay_graph_walks=["talker_last_prefill", "talker_decode"],
                single_request_inputs=self._get_dummy_inputs(device=device),
            )
        ]

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
    
    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        num_quantizers = self.config.code2wav.num_quantizers  # 16
        codec_eos = self.config.talker.codec_eos_token_id

        per_request_codec: list[torch.Tensor] = []

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
        return NodeInputs(tensor_inputs={
            "codec_tokens": codec_tokens
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
        return {"codec_tokens": batched_codec_tokens}
    
    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
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
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        **kwargs
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

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        return len({
            inputs.tensor_inputs["codec_tokens"].numel() \
                for inputs in model_inputs
        }) == 1
