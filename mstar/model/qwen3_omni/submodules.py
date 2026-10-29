# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Qwen3-Omni
# ---------------------------------------------------------------------------
#
# Five submodules covering the full Thinker-Talker dual-AR pipeline:
#   1. AudioEncoderSubmodule
#   2. VisionEncoderSubmodule
#   3. ThinkerSubmodule        (3D MRoPE, MoE, layer captures)
#   4. TalkerSubmodule         (streaming decode, Code Predictor)
#   5. Code2WavSubmodule
# ---------------------------------------------------------------------------

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
from torch import nn
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.code_predictor_kernels import cp_embed_accumulate
from mstar.model.components.mrope import mrope_position_ids
from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav
from mstar.model.qwen3_omni.components.rope import (
    compute_3d_cos_sin,
    compute_rope_freqs,
    get_rope_index_audio,
    get_rope_index_text,
    get_rope_index_vision,
)
from mstar.model.qwen3_omni.components.talker import Qwen3OmniCodePredictor, Qwen3OmniTalkerModel
from mstar.model.qwen3_omni.config import (
    CODE_PRED_SAMPLER,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    TALKER_SAMPLER,
    THINKER_ATTN,
    THINKER_KV,
    THINKER_MIXED,
    THINKER_POS,
    THINKER_SAMPLER,
    Qwen3OmniModelConfig,
)
from mstar.model.qwen3_omni.qwen3_omni_model import image_span
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    InputMetadata,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    StopCheck,
    TokenStop,
)

logger = logging.getLogger(__name__)


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """True if fn takes `name` as a keyword (explicitly or via **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    param = params.get(name)
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def talker_text_mask(input_ids: torch.Tensor, config: Qwen3OmniModelConfig) -> torch.Tensor:
    """
    Cut system prompt and previous assistant parts out of the talker input
    """
    im_start_indexes = (
        input_ids == config.im_start_token_id
    ).nonzero(as_tuple=True)[0]
    mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)

    for i in range(len(im_start_indexes) - 1):
        im_start_index = im_start_indexes[i]
        segment_end_index = im_start_indexes[i + 1]
        role_token = input_ids[im_start_index + 1]
        # Talker should ignore thinker system prompt
        if role_token == config.system_token_id:
            mask[im_start_index:segment_end_index] = 0
        elif role_token == config.assistant_token_id:
            mask[im_start_index:segment_end_index] = 0
    return mask


# ===================================================================
# 1. AudioEncoderSubmodule (enc_dec engine)
# ===================================================================


class AudioEncoderSubmodule(NodeSubmodule):
    """Thin wrapper around the HF Whisper-style audio encoder.

    Runs the encoder over a step's mel spectrograms to produce the audio
    embeddings the Thinker splices into its input sequence. A step's clips
    are packed along time in request order, starting a new call when the next
    clip would pass the frame budget (``forward_batched``).
    """

    # The HF encoder splits its input by host-side chunk lengths (``.tolist()``),
    # so compiled frames guard on the chunk count and recompile for nearly
    # every new clip length, seconds each, on the request's critical path.
    disable_torch_compile = True
    # Clips per encoder step, and mel frames (100 per second) per encoder
    # call. A clip may run 30 min (``media_io.MAX_AUDIO_SECONDS``), and a
    # call's activations grow with its total frames, so the budget bounds one
    # call's memory; a step over it packs its clips into several calls. This
    # sits in the forward until the scheduler can budget a step's tokens.
    MAX_BATCH_SIZE = 16
    MAX_BATCH_FRAMES = 30_000

    def __init__(self, audio_encoder: nn.Module, config: Qwen3OmniModelConfig):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.config = config
        # Older transformers take return_dict; newer ones dropped it.
        self._encoder_takes_return_dict = _accepts_kwarg(audio_encoder.forward, "return_dict")

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        # Edge name from graph walk is "audio_features"
        audio_features = inputs["audio_features"][0]
        audio_seqlens = inputs["audio_seqlens"][0]
        # one clip, its valid mel frames packed along time: its length is the
        # tensor's width, known on the host
        frames = audio_features.shape[-1]
        return NodeInputs(
            tensor_inputs={
                "audio_features": audio_features,
                "audio_seqlens": audio_seqlens,
            },
            kwargs={"embed_len": int(_get_feat_extract_output_lengths(frames, self.audio_encoder.n_window))},
            input_seq_len=frames,
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        audio_seqlens: torch.Tensor,
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
        extra_kwargs = {"return_dict": True} if self._encoder_takes_return_dict else {}
        audio_out = self.audio_encoder(
            audio_features,
            feature_lens=audio_seqlens,
            **extra_kwargs,
        )
        audio_embeds = getattr(audio_out, "last_hidden_state", audio_out)

        # Flatten to (num_audio_tokens, hidden_size) if needed
        if audio_embeds.dim() == 3:
            audio_embeds = audio_embeds.squeeze(0)

        return {"audio_embeds": [audio_embeds]}

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int:
        return self.MAX_BATCH_SIZE

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        return {
            "audio_features": [inp.tensor_inputs["audio_features"] for inp in inputs],
            "audio_seqlens": [inp.tensor_inputs["audio_seqlens"].reshape(-1) for inp in inputs],
            "embed_lens": [inp.kwargs["embed_len"] for inp in inputs],
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: list[torch.Tensor],
        audio_seqlens: list[torch.Tensor],
        embed_lens: list[int],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Encode the step's clips packed along time in request order, starting
        a new call (each ~10 ms on an H200, mostly launches, whatever the clip
        length) when the next clip would pass ``MAX_BATCH_FRAMES``, and split
        each call's embeddings back by clip.
        The encoder windows its attention within each clip (``feature_lens``),
        so packing does not mix clips."""
        groups: list[list[int]] = []
        frames = 0
        for row, features in enumerate(audio_features):
            n = features.shape[-1]
            if groups and frames + n <= self.MAX_BATCH_FRAMES:
                groups[-1].append(row)
                frames += n
            else:
                groups.append([row])
                frames = n
        parts: list[torch.Tensor] = []
        for group in groups:
            embeds = self.forward(
                graph_walk, engine_inputs,
                torch.cat([audio_features[row] for row in group], dim=-1),
                torch.cat([audio_seqlens[row] for row in group]),
            )["audio_embeds"][0]
            parts.extend(embeds.split([embed_lens[row] for row in group]))
        return {
            rid: {"audio_embeds": [part]}
            for rid, part in zip(engine_inputs.request_ids, parts, strict=True)
        }


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

    marks_dynamic_dims = True

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

        # Built outside the compiled forward, since these helpers call `.tolist()` on the
        # grid, which would compile a graph for every image size.
        encoder = self.vision_encoder
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw, num_grid_per_side=encoder.num_grid_per_side, mode=encoder.interpolation_mode,
            align_corners=encoder.interpolation_align_corners, spatial_merge_size=encoder.spatial_merge_size,
        )
        tensor_inputs = {
            "pixel_values": pixel_values,
            "grid_thw": grid_thw,
            "interp_indices": interp_indices,
            "interp_weights": interp_weights,
            "position_ids": get_vision_position_ids(grid_thw, encoder.spatial_merge_size),
            "cu_seqlens": get_vision_cu_seqlens(grid_thw),
        }
        for name in ("pixel_values", "interp_indices", "interp_weights", "position_ids", "cu_seqlens"):
            torch._dynamo.mark_dynamic(tensor_inputs[name], 0)
        return NodeInputs(tensor_inputs=tensor_inputs)

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
            # each video frame is its own segment, and a call encodes one item, so this is one frame's patches
            max_seqlen=pixel_values.shape[0] // (kwargs["cu_seqlens"].shape[0] - 1),
            **kwargs,
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

    All walks produce ``thinker_states`` that stream to the Talker partition:
    the layer-0 + layer-N concat at prefill (postprocess picks per token), the
    layer-0 row at decode, whose token is text.  ``thinker_decode``
    additionally produces ``logits`` for text token sampling.
    """

    cuts_tensor_inputs = True

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
        # only a decode row's token ends the loop; a prefill is not in it yet
        self.token_stop = TokenStop(
            loop="thinker_decode_loop", buffer="new_token", stop_ids=frozenset({config.im_end_token_id}),
            sampler=THINKER_SAMPLER, walk="thinker_decode",
        )

        # Pre-compute inverse frequencies for 3D MRoPE
        self._inv_freq: torch.Tensor | None = None

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
        **kwargs
    ) -> ARNodeInputs:
        device = self.get_device()
        start_pos = self.node_resources[THINKER_POS].position(
            rid=fwd_info.rid_handle, label="main"
        )
        if graph_walk == "thinker_decode":
            # Get previous token ID from text_inputs
            token_id = inputs["text_inputs"][0].to(device)  # (1,) or scalar
            if token_id.dim() == 0:
                token_id = token_id.unsqueeze(0)
            embeds = self.model.model.embed_tokens(token_id)

            # Next MRoPE position for all 3 components: read from the
            # stream's position counter (advanced by the runner's commit
            # of each declared step).
            # Fill kernel, not ``torch.tensor(..., device=cuda)``: a pageable H2D
            # syncs the stream, stalling prepare(N+1) until step N drains.
            pos_ids = torch.full(
                (3, 1), float(start_pos), dtype=torch.float, device=device,
            )  # (3, 1)

            return ARNodeInputs(
                input_seq_len=1,
                input_embeds=embeds,
                custom_pos_ids=pos_ids,
            )

        if graph_walk == "prefill_text":
            text_ids = inputs["text_inputs"][0].to(device)  # (seq_len,)
            embeds = self.model.model.embed_tokens(text_ids)
            seq_len = text_ids.shape[0]
            masks_for_talker = torch.stack([
                torch.zeros(text_ids.shape, dtype=torch.bool, device=device), # multimodal
                talker_text_mask(text_ids, self.config) # text inclusion
            ])
            # A text row leaves its positions to the plan, which starts from the
            # counter a cache hit has already moved.
            return ARNodeInputs(
                input_seq_len=seq_len,
                input_embeds=embeds,
                input_ids=text_ids,
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
            # otherwise fall back to a 1-D sequence (test path)
            grid_thw = inputs.get("image_grid_thw", [None])[0]
            assert vision_len == int(grid_thw.prod()) // self.config.vision.spatial_merge_size ** 2, (
                f"the vision encoder gave {vision_len} rows, where the layout counts the grid's merged patches"
            )
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
            if seconds_per_grid is None:
                # Use the prompt layout's helper, so the cache and this walk move the position by the same amount.
                end_pos_base = start_pos + image_span(grid_thw, self.config.vision.spatial_merge_size).advance - 1
            else:
                end_pos_base = float(vision_pos_ids.max().item()) + 1
            end_pos_ids = get_rope_index_text(1, end_pos_base, device)

            pos_ids = torch.cat(
                [start_pos_ids, vision_pos_ids, end_pos_ids], dim=1
            )

            # Next MRoPE position after this vision block is ``end_pos_base
            # + 1`` (one past the EOS token). A commit advances positions by
            # the span by default, which for vision (= vision_len + 2) is
            # typically smaller than the 3D-grid span. Emit the correct
            # per-request advance so ``declare_step`` can declare it.
            mrope_pos_advance = int(end_pos_base + 1 - start_pos)
            deepstack = []
            for deepstack_inp in inputs["deepstack"]:
                full_deepstack = torch.zeros_like(wrapped_embeds)
                full_deepstack[mm_mask, :] = deepstack_inp
                deepstack.append(full_deepstack)

            return ARNodeInputs(
                input_seq_len=total_len,
                input_embeds=wrapped_embeds,
                custom_pos_ids=pos_ids,
                tensor_inputs={
                    "masks_for_talker": masks_for_talker,
                    "mrope_pos_advance": mrope_pos_advance,
                    "deepstack": deepstack,
                }
            )

    def split_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: ARNodeInputs,
        start: int,
        end: int,
    ) -> ARNodeInputs:
        tensors = inputs.tensor_inputs
        cut = super().split_inputs(graph_walk, fwd_info, replace(inputs, tensor_inputs={}), start, end)
        cut.tensor_inputs = {"masks_for_talker": tensors["masks_for_talker"][:, start:end]}
        if graph_walk == "prefill_vision":
            cut.tensor_inputs["deepstack"] = [layer[start:end] for layer in tensors["deepstack"]]
            cut.tensor_inputs["mrope_pos_advance"] = tensors["mrope_pos_advance"]
        elif graph_walk == "prefill_audio":
            cut.tensor_inputs["mrope_pos_advance"] = inputs.input_seq_len
        return cut

    def declare_step(
        self, graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        per_request_info: Mapping[int, CurrentForwardPassInfo] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        row_walks = [inp.graph_walk or graph_walk for inp in inputs]
        kept_rids = None
        if graph_walk != "thinker_decode":
            # decode rows, and the last prefill, keep their token through `postprocess`
            info = per_request_info or {}
            kept_rids = frozenset(
                rid for rid, walk in zip(request_ids, row_walks, strict=True)
                if walk == "thinker_decode" or (
                    rid in info and info[rid].step_metadata.get("is_last_prefill", False)
                )
            )
        prefill_tokens = {
            rid: inp.input_ids
            for rid, inp, walk in zip(request_ids, inputs, row_walks, strict=True)
            if walk == "prefill_text"
        }

        pos_advance = None
        if graph_walk != "thinker_decode" and any("mrope_pos_advance" in inp.tensor_inputs for inp in inputs):
            # A vision row's 3D-grid MRoPE span differs from its token count, and
            # a cut audio row moves past its whole clip, so each advances the
            # counter by its own `mrope_pos_advance`.
            pos_advance = tuple(
                int(inp.tensor_inputs.get("mrope_pos_advance", inp.input_seq_len))
                for inp in inputs
            )
        return SubmoduleStep(
            # must match `cg_key_info`, which picked the leased capture
            cg_key_info=self.DEEPSTACK_KEY if "prefill_vision" in row_walks else None,
            segments=[
                Segment(
                    request_id=rid,
                    label="main",
                    span=inp.input_seq_len,
                ) for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                THINKER_KV: KVStep(),
                THINKER_ATTN: AttentionStep(causal=True),
                THINKER_SAMPLER: SamplerStep(
                    apply_penalty=True,
                    prefill_tracked_tokens=prefill_tokens,
                    kept_rids=kept_rids,
                ),
                THINKER_POS: PositionStep(
                    advance=pos_advance
                )
            },

        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]: # input name to tensor
        self.model.maybe_log_expert_load(synthetic=engine_inputs.synthetic)
        device = self.get_device()
        # Concatenate across requests
        input_embeds = torch.cat([
            inp.input_embeds for inp in inputs
        ], dim=0)
        position_ids_3d = self._position_ids_3d(inputs)  # (3, total_tokens)
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

        extra_inputs = {}
        if graph_walk == "prefill_vision" or any("deepstack" in inp.tensor_inputs for inp in inputs):
            # Q3(b): emit deepstack as separate keys ``deepstack_<i>`` so each
            # tensor gets its own static buffer in the captured config (the
            # runner's static-buffer interning is per-key and tensor-typed;
            # passing a list-of-tensors under one key would not get interned
            # and addresses captured into the graph would be stale at replay).
            # ``forward_batched``/``forward`` reassemble the list from kwargs.
            extra_inputs = self._deepstack_inputs(inputs, input_embeds.dtype, device)

        if self.model.mask_padding:
            # Real rows lead the bucket; the captured MoE skips the rest.
            # A fill kernel, so no H2D copy.
            num_valid = engine_inputs.real_seq_len
            if num_valid is None:
                num_valid = input_embeds.shape[0]
            extra_inputs["num_valid_tokens"] = torch.full(
                (1,), num_valid, dtype=torch.int64, device=device,
            )

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

    def _position_ids_3d(self, inputs: list[ARNodeInputs]) -> torch.Tensor:
        return mrope_position_ids(self.node_resources[THINKER_POS].pos_ids("main"), inputs)

    def _deepstack_inputs(
        self, inputs: list[ARNodeInputs], dtype: torch.dtype, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Each deepstack layer packed over the step's rows; a row with no
        vision features (a text, audio or decode row beside a vision one)
        contributes zeros, which the splice adds as nothing."""
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)
        layers: list[list[torch.Tensor]] = [[] for _ in range(num_deepstack)]
        for inp in inputs:
            rows = inp.tensor_inputs.get("deepstack")
            if rows is None or (isinstance(rows, torch.Tensor) and rows.numel() == 0):
                zeros = torch.zeros(
                    (inp.input_seq_len, self.config.thinker_hidden_size), dtype=dtype, device=device,
                )
                rows = [zeros] * num_deepstack
            assert len(rows) == num_deepstack, (
                f"deepstack list length ({len(rows)}) does not match "
                f"vision.deepstack_visual_indexes ({num_deepstack})."
            )
            for layer, t in zip(layers, rows, strict=True):
                layer.append(t)
        return {
            f"deepstack_{i}": layer[0] if len(layer) == 1 else torch.cat(layer)
            for i, layer in enumerate(layers)
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
        request_info: CurrentForwardPassInfo = engine_inputs.single_request_info
        audio_output = request_info.step_metadata.get(
            "audio_output", True,
        )

        cos_sin_3d = (cos_3d, sin_3d) if cos_3d is not None else None

        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
            deepstack_visual_embeds=deepstack,
            label="main",
            num_valid_tokens=kwargs.get("num_valid_tokens"),
        )

        result: NameToTensorList = {}

        # Decode: produce logits for text token sampling
        if graph_walk == "thinker_decode" or request_info.step_metadata.get("is_last_prefill", False):
            logits = self.model.lm_head(hidden[-1:, :])
            result["new_token"] = [engine_inputs.resources[THINKER_SAMPLER].sample(
                engine_inputs.request_ids, logits=logits
            )]

        # Pack thinker_states for Talker conditioning ONLY when audio output
        # is requested.  For text-only requests we skip this entirely to
        # avoid sending hidden states the Talker will never consume.
        if audio_output and graph_walk == "thinker_decode":
            # a decode token is text: its layer-0 row and ``[False]``, fresh
            # each step, as ``forward_batched`` emits them
            result["thinker_states"] = [layer_0_embed]
            result["thinker_mask"] = [torch.zeros(1, dtype=torch.bool, device=hidden.device)]
        elif audio_output:
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
    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return len(model_inputs) > 1

    PREFILL_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048]
    # past 4 for thinker_mixed, whose decode rows ride in these captures
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    # prefill_vision buckets are larger than text/audio because video
    # produces many vision tokens per request
    PREFILL_VISION_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # coarse batch buckets, padded host-side; past 1 for mixed steps
    PREFILL_VISION_CAPTURE_BATCH_SIZES = [1, 4, 16, 32]
    DEEPSTACK_KEY = "deepstack"

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        vision_input = functools.partial(self._vision_capture_input, device)
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="thinker_decode",
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
                ),
                capture_batch_sizes=[1, 2, 4, 8, 16, 32],
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_text",
                replay_graph_walks=["prefill_text", "prefill_audio", THINKER_MIXED],
                make_node_input=lambda n: ARNodeInputs(
                    input_seq_len=n,
                    input_ids=torch.zeros((n,), dtype=torch.long, device=device),
                    input_embeds=torch.zeros(
                        (n, self.config.thinker_hidden_size),
                        device=device, dtype=torch.bfloat16
                    ),
                    custom_pos_ids=torch.zeros(
                        (3, n),
                        dtype=torch.float,
                        device=device,
                    ),
                    tensor_inputs={
                        "masks_for_talker": torch.zeros(
                            (2, n),
                            dtype=torch.float,
                            device=device,
                        ),
                    }
                ),
                capture_token_lengths=self.PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_CAPTURE_BATCH_SIZES,
            ),
            # any step holding a vision row: the packed forward plus deepstack
            PackedAcceleratorGraphConfig(
                capture_graph_walk="prefill_vision",
                replay_graph_walks=["prefill_vision", THINKER_MIXED],
                additional_key_info=self.DEEPSTACK_KEY,
                capture_token_lengths=self.PREFILL_VISION_TOKEN_BUCKETS,
                capture_batch_sizes=self.PREFILL_VISION_CAPTURE_BATCH_SIZES,
                make_node_input=vision_input,
            ),
        ]

    def cg_key_info(
        self, graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
        per_request_input_metadata=None,
        **kwargs,
    ):
        """A step holding a vision row replays the deepstack captures."""
        del per_request_input_metadata, kwargs
        if graph_walk == "prefill_vision" or graph_walk == THINKER_MIXED and any(
            info.graph_walk == "prefill_vision" for info in per_request_info.values()
        ):
            return self.DEEPSTACK_KEY
        return None

    def _vision_capture_input(self, device: torch.device, n: int) -> ARNodeInputs:
        num_deepstack = len(self.config.vision.deepstack_visual_indexes)
        return ARNodeInputs(
            input_seq_len=n,
            input_embeds=torch.zeros(
                (n, self.config.thinker_hidden_size),
                device=device, dtype=torch.bfloat16,
            ),
            custom_pos_ids=torch.zeros(
                (3, n),
                dtype=torch.float,
                device=device,
            ),
            tensor_inputs={
                "masks_for_talker": torch.zeros(
                    (2, n), dtype=torch.float, device=device,
                ),
                # Use a list (matches eager prepare_inputs) — preprocess turns it
                # into per-layer ``deepstack_<i>`` keys.
                "deepstack": [
                    torch.zeros(
                        (n, self.config.thinker_hidden_size),
                        dtype=torch.bfloat16, device=device,
                    )
                    for _ in range(num_deepstack)
                ],
                "mrope_pos_advance": n,
            },
        )

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        cos_3d: torch.Tensor | None = None,
        sin_3d: torch.Tensor | None = None,
        mrope_section: list[int] | None = None,
        masks_for_talker: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Batched Thinker forward shared between ``thinker_decode`` and the prefill walks.

        Decode path (1 token per request, ``hidden`` shape ``(bs, hidden)``):
        always emits ``thinker_states`` and ``thinker_mask`` as row outputs so
        the captured CUDA graph's outputs are fixed whatever the request
        metadata; ``postprocess`` drops them for a text-only request.
        Prefill packs ``thinker_states`` for ``unpack_packed_outputs``.
        """

        is_prefill = graph_walk in (
            "prefill_text", "prefill_audio", "prefill_vision", THINKER_MIXED,
        )

        # prefill_vision: reassemble per-layer deepstack tensors from
        # ``deepstack_<i>`` kwargs into the list shape ``model.forward``
        # expects. None for non-vision walks → model skips the splice.
        deepstack = self._collect_deepstack_kwargs(kwargs)

        cos_sin_3d = (cos_3d, sin_3d) if cos_3d is not None else None
        attn_manager: FlashInferManager = engine_inputs.resources[THINKER_ATTN]
        hidden, layer_0_embed, layer_n_hidden = self.model(
            input_embeds=input_embeds,
            cos_sin_3d=cos_sin_3d,
            mrope_section=mrope_section,
            deepstack_visual_embeds=deepstack,
            label="main",
            num_valid_tokens=kwargs.get("num_valid_tokens"),
        )

        request_ids = engine_inputs.request_ids
        sampler: SamplerResource = engine_inputs.resources[THINKER_SAMPLER]

        if is_prefill:
            last_hidden = attn_manager.select_last_hidden(hidden)
            logits = self.model.lm_head(last_hidden)  # (padded_bs, vocab)
            new_tokens = sampler.sample(request_ids, logits)
            if layer_n_hidden is not None:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_n_hidden], dim=-1,
                )  # (total_tokens, 2*hidden)
            else:
                thinker_states = torch.cat(
                    [layer_0_embed, layer_0_embed], dim=-1,
                )

            return {
                "__batched_thinker_states__": thinker_states,
                **{
                    rid: {
                        "new_token": [new_tokens[i:i + 1]]
                    } for i, rid in enumerate(request_ids)
                }
            }

        # thinker_decode (existing behavior)
        logits = self.model.lm_head(hidden)  # (batch, vocab)
        new_tokens = sampler.sample(request_ids, logits)

        # Row i is request i: one clone per output and one host copy of the
        # tokens per step. A decode token is text, so the Talker takes its
        # layer-0 row as is and its Talker mask is ``[False]``; a row view of
        # the step's clone, never one tensor shared by every row, which the
        # RDMA/TCP store would register once per row at the same address.
        # postprocess leaves both rows as they are, for the row-output store.
        return BatchedModelOutput(
            row_outputs={
                "thinker_states": layer_0_embed,
                "new_token": new_tokens,
                "thinker_mask": torch.zeros_like(new_tokens, dtype=torch.bool),
            },
            check_stop_buffers={"new_token": new_tokens},
        )

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
        ``audio_output=False``, as postprocess does for the decode path.
        """
        packed_states = static_output.get("__batched_thinker_states__")
        if packed_states is None:
            return {}

        out: dict[str, dict[str, list[torch.Tensor]]] = {}
        # a decode row's ``[False]``, as in ``forward_batched``: a row view of
        # one step tensor, not the cached mask every decode row's inputs share
        decode_masks = torch.zeros(len(request_ids), dtype=torch.bool, device=packed_states.device)
        cum = 0
        for i, rid in enumerate(request_ids):
            sl = real_seq_lens[i]
            slice_start, slice_end = cum, cum + sl
            cum = slice_end

            info = per_request_info.get(rid) if per_request_info else None
            if info is not None and not info.step_metadata.get("audio_output", True):
                continue

            ts_slice = packed_states[slice_start:slice_end]
            mask = inputs[i].tensor_inputs.get("masks_for_talker")
            if inputs[i].graph_walk == "thinker_decode":
                # a decode token is text: its layer-0 half, as forward_batched emits it
                ts_slice = ts_slice[..., :self.config.thinker_hidden_size]
            rid_out: dict[str, list[torch.Tensor]] = {
                "thinker_states": [ts_slice.clone()],
            }
            if inputs[i].graph_walk == "thinker_decode":
                rid_out["thinker_mask"] = [decode_masks[i:i + 1]]
            elif mask is not None:
                rid_out["thinker_mask"] = [mask]
            out[rid] = rid_out
        return out

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
            # drop thinker_states and thinker_match
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
            # A decode token is text the Talker keeps: its row is layer 0
            # already and its 1D mask a row of its own. The boolean indexing
            # below would sync the host on every row of every decode step.
            if ts_list and mask_list and request_info.graph_walk != "thinker_decode":
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
        # The same pad/eos before ``text_projection``: a decode step projects
        # its rows' text inputs in the captured forward, so these stand in
        # for the Thinker state once the Thinker has finished.
        self._tts_pad_raw: torch.Tensor | None = None
        self._tts_eos_raw: torch.Tensor | None = None

        # Lazy-built suppress mask for layer-0 logits.  Shape (vocab_size,)
        # with True at positions to suppress.  Cached on first forward.
        self._suppress_mask: torch.Tensor | None = None

        # Per-request flag: whether we've already sent tts_eos_embed as
        # the text conditioning for this request. We use this flag to
        # inject tts_eos_embed for ONE step before falling back to pad.
        self._eos_embed_sent: set[str] = set()

        # Lazy-built CodePredictor scratch KV cache, overwritten every step.
        self._cp_kv_cache: torch.Tensor | None = None
        self._cp_positions: torch.Tensor | None = None
        self._cp_pair_positions: torch.Tensor | None = None

    def _get_cp_kv_cache(self):
        # TODO: make this a resource
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

    def _get_cp_positions(self) -> torch.Tensor:
        """``[g, b, 0] = g``: the code predictor's position for depth step g,
        the same for every row. Constant, so a step slices it instead of
        counting up a buffer with a launch per depth step."""
        if self._cp_positions is None:
            self._cp_positions = torch.arange(
                self.num_codes, device=self.get_device(),
            ).view(-1, 1, 1).expand(-1, self.MAX_BATCH_SIZE, 1).contiguous()
        return self._cp_positions

    def _get_cp_pair_positions(self) -> torch.Tensor:
        """``[b] = [0, 1]``: the positions of the depth sequence's opening pair."""
        if self._cp_pair_positions is None:
            self._cp_pair_positions = torch.arange(
                2, device=self.get_device(),
            ).expand(self.MAX_BATCH_SIZE, 2).contiguous()
        return self._cp_pair_positions

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
            self._tts_pad_raw, self._tts_eos_raw = pad_raw, eos_raw
            self._tts_pad_embed_cached = self.model.text_projection(pad_raw).squeeze(0)
            self._tts_bos_embed_cached = self.model.text_projection(bos_raw).squeeze(0)

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
        # Build assistant prefix (matching the HF pattern):
        # Text hidden: [pad*4, bos, proj[3]] (9 tokens)
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
        # Build assistant prefix (matching the HF pattern):
        # Codec hidden: [codec_embed(nothink, think_bos, think_eos,
        #                speaker, pad, bos)] (9 tokens)
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
            rid = fwd_info.rid_handle
            # Last-prefill is the assistant's first token, which is
            # text-only — Thinker postprocess pre-selected layer-0 for it.
            last_hidden = inputs["thinker_states"][0].to(device)

            input_embeds = self._get_last_prefill_codec_hidden(
                fwd_info.step_metadata.get("voice", "Ethan")
            ) + self._get_last_prefill_talker_hidden(last_hidden)
            seq_len = input_embeds.shape[0]

        elif graph_walk == "talker_decode":
            # The previous step's codec embedding sum plus this step's text
            # conditioning, which the forward projects for the whole batch:
            # the Thinker's next token state (decode is always text), then
            # tts_eos for one step, then tts_pad.
            thinker_states = inputs.get("thinker_states", [])
            rid = fwd_info.rid_handle
            if thinker_states:
                text_raw = thinker_states[0]
            elif rid not in self._eos_embed_sent:
                text_raw = self._tts_eos_raw
                self._eos_embed_sent.add(rid)
            else:
                text_raw = self._tts_pad_raw
            return ARNodeInputs(
                input_embeds=inputs["talker_input_embeds"][0],
                input_seq_len=1,
                tensor_inputs={"text_raw": text_raw},
            )

        return ARNodeInputs(
            input_embeds=input_embeds,
            input_seq_len=seq_len,
        )

    def declare_step(
        self, graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) ->SubmoduleStep:
        steps = {
            TALKER_KV: KVStep(),
            TALKER_ATTN: AttentionStep(causal=True),
            TALKER_POS: PositionStep()
        }
        if graph_walk in ("talker_last_prefill", "talker_decode"):
            steps.update(
                **{
                    TALKER_SAMPLER: SamplerStep(
                        apply_penalty=True,
                    ),
                    CODE_PRED_SAMPLER: SamplerStep(
                        apply_penalty=False
                    )
                }
            )
        return SubmoduleStep(
            segments=[
                Segment(
                    request_id=rid,
                    label="main",
                    span=inp.input_seq_len,
                ) for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps=steps
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        seq_lens = [
            inp.input_seq_len for inp in inputs
        ]
        input_embeds = torch.cat([
            inp.input_embeds for inp in inputs
        ], dim=0)
        device = self.get_device()

        extra_args = {}
        if graph_walk == "talker_decode":
            dtype = self.model.text_projection.linear_fc1.weight.dtype
            input_embeds = input_embeds.to(dtype)
            extra_args["text_raw"] = torch.cat(
                [inp.tensor_inputs["text_raw"] for inp in inputs], dim=0,
            ).to(dtype)
        if graph_walk == "talker_last_prefill":
            extra_args["last_token_indices"] = (
                torch.tensor(seq_lens, device=device, dtype=torch.long).cumsum(0) - 1
            )
        return {
            "input_embeds": input_embeds,
            **extra_args
        }

    def _add_text_conditioning(
        self, input_embeds: torch.Tensor, text_raw: torch.Tensor | None,
    ) -> torch.Tensor:
        """A decode step's input: its codec embedding sum plus the projected
        text conditioning, one ``text_projection`` for the batch."""
        if text_raw is None:
            return input_embeds
        return input_embeds + self.model.text_projection(text_raw)

    def _forward_prefill(
        self, input_embeds: torch.Tensor,
    ):
        self.model(
            input_embeds=input_embeds, label="main",
        )
        return {}

    def _forward_decode_like(
        self, request_ids: list[str],
        talker_sampler: SamplerResource,
        code_pred_sampler: SamplerResource,
        is_batched_decode: bool,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor | None = None,
        **kwargs
    ):
        """
        Runs the Talker LLM for stages that grpoh walks that sample a token
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
            input_embeds=input_embeds, label="main",
        )
        if last_token_indices is not None:
            last_hidden = hidden.index_select(0, last_token_indices)
        elif not is_batched_decode:
            last_hidden = hidden[-1:, :]
        else:
            last_hidden = hidden
        logits = self.model.codec_head(last_hidden)
        logits = logits.masked_fill(self._get_suppress_mask().unsqueeze(0), float("-inf"))
        # Apply the repetition penalty to the Talker's layer-0 codes (the code
        # predictor depth loop below samples through its own aux config and
        # stays penalty-free). The penalty is baked into the captured graph.
        layer0_codes = talker_sampler.sample(
            request_ids, logits
        )

        # code predictor section
        embed = self.talker_code_emb(layer0_codes)  # [bs, hidden]
        bs = embed.shape[0]
        # Scratch made here rather than passed in, so a captured step zeroes it
        # in the graph instead of copying zeros into static inputs.
        codec_emb_sum = embed.float()
        all_codes = torch.zeros((bs, self.num_codes), device=embed.device, dtype=torch.long)
        all_codes[:, 0] = layer0_codes
        positions = self._get_cp_positions()

        kv_cache = self._get_cp_kv_cache()[:, :bs]

        # The depth sequence opens with the Talker's hidden state and the layer-0
        # code's embedding (as in HF's code-predictor prefill): one two-token pass
        # whose second token predicts layer 1, then one token per layer.
        cp = self.code_predictor
        codec_embedding = cp.model.codec_embedding
        for group_idx in range(1, self.num_codes):
            if group_idx == 1:
                cp.depth.bind_step(kv_cache, self._get_cp_pair_positions()[:bs], cache_pos=0)
                hidden = cp.forward_depth_unrolled(torch.stack([last_hidden, embed], dim=1))[:, 1]
            else:
                cp.depth.bind_step(kv_cache, positions[group_idx, :bs], cache_pos=group_idx)
                hidden = cp.forward_depth_unrolled(embed.unsqueeze(1)).squeeze(1)

            logits = torch.matmul(
                hidden, cp.lm_head_weight[group_idx - 1].t()
            )

            # Params come from the "code_predictor" aux sampler's static
            # buffers, so per-request values survive graph replay.
            tokens = code_pred_sampler.sample(
                request_ids, logits,
            )
            embed = cp_embed_accumulate(
                tokens, codec_embedding[group_idx - 1].weight, codec_emb_sum, all_codes, group_idx,
            )

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
                input_embeds=input_embeds
            )
        input_embeds = self._add_text_conditioning(input_embeds, kwargs.pop("text_raw", None))
        return self._forward_decode_like(
            request_ids=engine_inputs.request_ids,
            talker_sampler=engine_inputs.resources[TALKER_SAMPLER],
            code_pred_sampler=engine_inputs.resources[CODE_PRED_SAMPLER],
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
          Runs the full LLM + codec_head + suppress_mask via _forward_decode_like,
          samples inline through the injected samplers, and emits
          ``talker_input_embeds``, ``codec_tokens`` and ``layer0_codes`` as row
          outputs.

        Prefill path (``talker_prefill``; multi-token-per-request, ``hidden``
        shape ``(total_tokens, hidden)``): Runs only the LLM backbone, and
        populates the KV cache.

        Last-prefill path (``talker_last_prefill``; fixed 9 tokens per request,
        ``hidden`` shape ``(bs * 9, hidden)``)
        """
        assert graph_walk in (
            "talker_decode", "talker_prefill", "talker_last_prefill",
        )

        if graph_walk == "talker_prefill":
            hidden = self.model(
                input_embeds=input_embeds,
                label="main",
            )
            return {
                "__batched_talker_prefill_hidden__": hidden,
            }

        last_token_indices = kwargs.pop("last_token_indices", None)
        input_embeds = self._add_text_conditioning(input_embeds, kwargs.pop("text_raw", None))
        if graph_walk == "talker_last_prefill":
            assert last_token_indices is not None, (
                "talker_last_prefill forward_batched requires "
                "last_token_indices from preprocess; got None."
            )

        fwd_out = self._forward_decode_like(
            request_ids=engine_inputs.request_ids,
            talker_sampler=engine_inputs.resources[TALKER_SAMPLER],
            code_pred_sampler=engine_inputs.resources[CODE_PRED_SAMPLER],
            is_batched_decode=True,
            last_token_indices=last_token_indices,
            input_embeds=input_embeds,
            **kwargs
        )

        # Row i is request i: the engine clones each output once for the batch
        # and the stop check reads one host copy, instead of three clones and
        # a device-to-host copy per request.
        layer0_codes = fwd_out["new_token"][0]
        return BatchedModelOutput(
            row_outputs={
                "talker_input_embeds": fwd_out["talker_input_embeds"][0],
                "codec_tokens": fwd_out["codec_tokens"][0],
                "layer0_codes": layer0_codes,
            },
            check_stop_buffers={"layer0_codes": layer0_codes},
        )

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
    ) -> set[str] | StopCheck:
        """Stop on codec_eos or the frame limit. The codec_eos frame stays out
        of the Code2Wav stream: it is not a codebook index, and only this host
        copy tells it from audio."""
        if "layer0_codes" not in outputs:
            return set()
        token = outputs["layer0_codes"][0].item()
        eos_token_id = self.config.talker.codec_eos_token_id
        max_tokens = request_info.step_metadata.get(
            "talker_max_tokens", request_info.max_tokens
        )
        is_eos = eos_token_id is not None and eos_token_id == token
        stops = set()
        if is_eos or \
                (request_info.dynamic_loop_iter_counts.get("talker_decode_loop", 0) + 1 >= max_tokens):
            stops = {"talker_decode_loop"}
        return StopCheck(stops, withheld=("codec_tokens",)) if is_eos else stops

    def cleanup_request(self, request_id: str) -> None:
        """Remove per-request state when a request completes."""
        super().cleanup_request(request_id)
        self._eos_embed_sent.discard(request_id)

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
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

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        return [
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="talker_decode",
                single_request_inputs=ARNodeInputs(
                    input_embeds=torch.zeros(
                        (1, self.config.talker_hidden_size),
                        device=device, dtype=torch.bfloat16
                    ),
                    input_seq_len=1,
                    tensor_inputs={"text_raw": torch.zeros(
                        (1, self.config.thinker_hidden_size),
                        device=device, dtype=torch.bfloat16,
                    )},
                ),
                capture_batch_sizes=[1, 2, 4, 8, 16, 32],
                compile=True
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk="talker_prefill",
                replay_graph_walks=["talker_prefill"],
                make_node_input=lambda n: ARNodeInputs(
                    input_embeds=torch.zeros(
                        (n, self.config.talker_hidden_size),
                        device=device, dtype=torch.bfloat16
                    ),
                    input_seq_len=n,
                ),
                capture_token_lengths=self.TALKER_PREFILL_TOKEN_BUCKETS,
                capture_batch_sizes=self.TALKER_PREFILL_CAPTURE_BATCH_SIZES,
                compile=True
            ),
            BatchedAcceleratorGraphConfig(
                capture_graph_walk="talker_last_prefill",
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


# ===================================================================
# 5. Code2WavSubmodule
# ===================================================================


class Code2WavSubmodule(NodeSubmodule):
    """Wraps the HF Code2Wav vocoder for streaming chunk decode.

    Receives codec_tokens from the Talker (via StreamBuffer), selects
    the first ``num_quantizers`` codebook layers, runs the ConvNet
    vocoder, trims overlap context, and returns the PCM audio chunk.
    """

    # Runs in the checkpoint's bf16 without autocast. ``consolidate()``
    # compiles the forward itself, so the engine must not compile it again.
    disable_torch_compile = True
    disable_autocast = True

    # No capture carries this key, so a batch of terminal flushes runs eager
    # and skips the decode when none of its windows has a new frame.
    _CONTEXT_ONLY_KEY = "context_only"

    def __init__(self, code2wav_model: Qwen3OmniMoeCode2Wav, config: Qwen3OmniModelConfig):
        super().__init__()
        self.code2wav = code2wav_model
        self.config = config

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

        # Every window, from the ramp's first few frames to a steady
        # context + chunk window, is zero-padded to this one shape. The
        # vocoder is causal, so the padding changes only the look-ahead
        # samples that ``postprocess`` drops, and the convs keep the input
        # lengths of a full window (bf16 cuDNN convs go wrong at some other
        # odd lengths on B200).
        self.full_seqlen = self.config.code2wav.codec_max_window

    @staticmethod
    def _chunk(rid, per_request_input_metadata: Mapping[str, InputMetadata] | None):
        meta = (per_request_input_metadata or {}).get(rid)
        return meta.stream_chunks.get("codec_tokens") if meta is not None else None

    @classmethod
    def _context_only(
        cls, request_ids, per_request_input_metadata: Mapping[str, InputMetadata] | None,
    ) -> bool:
        """Whether every window is a terminal flush with no new frames, which
        emits no audio: a reply whose frames end on a chunk boundary ends with
        one. Every stream item is a frame (the Talker withholds codec_eos), so
        the counts say so."""
        for rid in request_ids:
            chunk = cls._chunk(rid, per_request_input_metadata)
            if chunk is None or chunk.num_items > chunk.context_items:
                return False
        return bool(request_ids)

    def cg_key_info(
        self,
        graph_walk: str,
        per_request_info: Mapping[str, CurrentForwardPassInfo],
        per_request_input_metadata: Mapping[str, InputMetadata] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._context_only(list(per_request_info), per_request_input_metadata):
            return self._CONTEXT_ONLY_KEY
        return None

    def get_accelerator_graph_configs(self, device, tp_world_size: int = 1):
        num_quantizers = self.config.code2wav.num_quantizers
        return [
            BatchedAcceleratorGraphConfig(
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
        input_metadata: InputMetadata | None = None,
        **kwargs
    ) -> NodeInputs:
        num_quantizers = self.config.code2wav.num_quantizers  # 16
        num_groups = self.config.num_code_groups  # 16 (Qwen3-Omni)
        # Every item is a frame: the Talker withholds its codec_eos frame
        # (``TalkerSubmodule.check_stop``).
        received = inputs.get("codec_tokens")
        if received:
            codec_tokens = received[0]
        else:
            # a stream whose first frame was codec_eos ends with an empty chunk
            codec_tokens = torch.zeros((0, num_groups), dtype=torch.long, device=self.get_device())
        if codec_tokens.dim() == 3:
            # stacked (1, groups) frames, the Talker's batched rows
            codec_tokens = codec_tokens.flatten(0, 1)

        # Reshape to (num_frames, num_code_groups) if flat
        if codec_tokens.dim() == 1:
            if codec_tokens.shape[0] % num_groups == 0:
                codec_tokens = codec_tokens.view(-1, num_groups)
            else:
                codec_tokens = codec_tokens.unsqueeze(0)

        # Select first num_quantizers codebook layers
        if codec_tokens.shape[-1] > num_quantizers:
            codec_tokens = codec_tokens[..., :num_quantizers]

        # The window's leading frames that an earlier window already decoded.
        chunk = (input_metadata.stream_chunks.get("codec_tokens") if input_metadata is not None else None)
        orig_seq_len = codec_tokens.shape[0]
        context = chunk.context_items if chunk is not None else 0

        # pad sequence to full_seqlen for batching
        pad_len = self.full_seqlen - codec_tokens.shape[0]
        if pad_len > 0:
            pad = torch.zeros((
                pad_len, codec_tokens.shape[1]
            ),dtype=codec_tokens.dtype,
            device=codec_tokens.device)
            codec_tokens = torch.cat([codec_tokens, pad], dim=0)

        # Transpose to (Q, T)
        codec_tokens = codec_tokens.T  # (Q, T)
        # The window's geometry rides with this pass rather than in request
        # state: the engine may prepare a request's next window before it
        # postprocesses this one.
        return NodeInputs(
            tensor_inputs={
                "codec_tokens": codec_tokens,
                "position_ids": torch.arange(codec_tokens.shape[1], device=codec_tokens.device)
            },
            kwargs={"frames": orig_seq_len, "context": context},
        )

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
        metadata = engine_inputs.per_request_input_metadata
        if not engine_inputs.captured and self._context_only(engine_inputs.request_ids, metadata):
            return {"codec_tokens": batched_codec_tokens[..., :0], "position_ids": position_ids[..., :0]}
        return {"codec_tokens": batched_codec_tokens, "position_ids": position_ids}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs
    ) -> dict[str, NameToTensorList]:
        """Decode a batch of padded windows to untrimmed int16 PCM.

        The Talker→Code2Wav StreamBuffer uses ``ScheduledLeftContextChunkPolicy``:
        chunk k for a request carries ``codec_chunk_schedule[k]`` new frames
        (``codec_chunk_frames`` once the schedule runs out) behind up to
        ``codec_left_context_frames`` frames an earlier chunk delivered, so
        the first chunk has no overlap. The overlap lets the vocoder warm up
        at chunk boundaries; ``postprocess`` cuts each request's new audio
        out of the window.

        A zero-frame batch (terminal flushes with no new frames, see
        ``preprocess``) decodes nothing and emits empty chunks.
        """
        request_ids = engine_inputs.request_ids
        if codec_tokens is None:
            return {rid: {} for rid in request_ids}
        if codec_tokens.numel() == 0:
            return {
                rid: {"audio_chunk": [torch.empty(0, dtype=torch.int16, device=codec_tokens.device)]}
                for rid in request_ids
            }

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

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return len({
            inputs.tensor_inputs["codec_tokens"].numel() \
                for inputs in model_inputs
        }) == 1

    def can_use_accelerator_graphs(self, batch, model_inputs: list[NodeInputs]):
        res = super().can_use_accelerator_graphs(batch, model_inputs) \
            and self.can_batch(batch, model_inputs) \
                and model_inputs[0].tensor_inputs["codec_tokens"].shape[1] == self.full_seqlen
        return res

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        inputs: NodeInputs | None = None,
        **kwargs
    ):
        if "audio_chunk" not in outputs:
            return
        if inputs is None:
            raise ValueError("Code2Wav postprocess needs the pass's inputs for its window geometry")

        orig_seq_len = int(inputs.kwargs["frames"])
        left_context_size = int(inputs.kwargs["context"])
        audio = outputs["audio_chunk"][0].reshape(-1)
        if audio.numel() == 0:
            outputs["audio_chunk"][0] = audio
            return
        # Each decoder block's ConvTranspose1d(2r, stride r, padding r) needs
        # the next frame, so a window decodes to ``tail`` samples (555 for
        # Qwen3-Omni) short of window * total_upsample. The next window
        # re-decodes those frames as context and emits the missing samples, so
        # both ends of the slice move back by ``tail``. Moving the end also
        # drops the samples that looked ahead into the zero padding.
        window = max(orig_seq_len, self.full_seqlen)
        tail = max(window * self.total_upsample - audio.numel(), 0)
        start = max(left_context_size * self.total_upsample - tail, 0)
        end = max(orig_seq_len * self.total_upsample - tail, start)
        outputs["audio_chunk"][0] = audio[start:end]

