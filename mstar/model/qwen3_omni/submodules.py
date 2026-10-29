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
from typing import Any, NamedTuple

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
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.flashinfer import FlashInferManager
from mstar.engine.resources.sampler.resource import SamplerResource
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
    AUDIO_PART,
    CODE_PRED_SAMPLER,
    PREFILL_AUDIO,
    PREFILL_AUDIO_VISION,
    PREFILL_TEXT,
    PREFILL_VISION,
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    TALKER_SAMPLER,
    TEXT_PART,
    THINKER_ATTN,
    THINKER_KV,
    THINKER_MIXED,
    THINKER_POS,
    THINKER_SAMPLER,
    VIDEO_PART,
    VISION_PREFILL_WALKS,
    Qwen3OmniModelConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    InputMetadata,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.platforms import current_platform

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

    Encodes every audio clip of a prompt, one at a time, into the embeddings
    the Thinker splices into its prompt row in place of the clip's pad tokens.

    Runs once per request (not batched across requests).
    """

    def __init__(self, audio_encoder: nn.Module, config: Qwen3OmniModelConfig):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.config = config
        # Eager on AMD GPUs. Compiled, the HF encoder splits the clip by
        # chunk_lengths.tolist(), so dynamo guards on those values and
        # recompiles for most new clip lengths, 0.2-6 s each on an MI300X.
        # Eager costs about 8 ms a clip there.
        if current_platform.is_rocm():
            self.disable_torch_compile = True
        # Older transformers take return_dict; newer ones dropped it.
        self._encoder_takes_return_dict = _accepts_kwarg(audio_encoder.forward, "return_dict")

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        # one entry per clip, in prompt order
        features = inputs["audio_features"]
        seqlens = inputs.get("audio_seqlens") or [None] * len(features)
        return NodeInputs(
            tensor_inputs={
                "audio_features": list(features),
                "audio_seqlens": list(seqlens),
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: list[torch.Tensor],
        audio_seqlens: list[torch.Tensor | None],
        **kwargs,
    ) -> NameToTensorList:
        """Run the audio encoder on each clip.

        Returns:
            {"audio_embeds": [one (audio_tokens, hidden_size) tensor per clip]}
        """
        extra_kwargs = {"return_dict": True} if self._encoder_takes_return_dict else {}
        embeds = []
        for features, seqlens in zip(audio_features, audio_seqlens, strict=True):
            audio_out = self.audio_encoder(features, feature_lens=seqlens, **extra_kwargs)
            clip = getattr(audio_out, "last_hidden_state", audio_out)
            # Flatten to (num_audio_tokens, hidden_size) if needed
            embeds.append(clip.squeeze(0) if clip.dim() == 3 else clip)
        return {"audio_embeds": embeds}


# ===================================================================
# 2. VisionEncoderSubmodule (enc_dec engine)
# ===================================================================


class VisionEncoderSubmodule(NodeSubmodule):
    """Thin wrapper around the HF vision encoder (ViT + spatial merge).

    Encodes every image and video of a prompt, one at a time, into the
    embeddings the Thinker splices into its prompt row plus each one's
    DeepStack intermediate features.

    Runs once per request (not batched across requests).
    """

    def __init__(self, vision_encoder: nn.Module, config: Qwen3OmniModelConfig):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.config = config
        # Eager on AMD GPUs, like AudioEncoderSubmodule: compiled, each new
        # image grid seen recompiled it (grid_thw.tolist() values are guarded),
        # up to 9 s on an MI300X. Eager costs about 13 ms an image there.
        if current_platform.is_rocm():
            self.disable_torch_compile = True

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> NodeInputs:
        """One ``pixel_values`` and ``image_grid_thw`` per image or video, in
        prompt order, as ``Qwen3OmniModel.process_prompt`` produced them."""
        pixel_values = inputs["pixel_values"]
        grids = inputs.get("image_grid_thw", [])
        if len(grids) != len(pixel_values):
            raise ValueError(
                f"VisionEncoder: {len(pixel_values)} pixel_values but {len(grids)} image_grid_thw; "
                "process_prompt makes one grid per image or video."
            )
        # The encoder takes a (num_grids, 3) grid. A single-image request's
        # grid arrives 1-D, [T, H, W].
        return NodeInputs(
            tensor_inputs={
                "pixel_values": list(pixel_values),
                "grid_thw": [grid.unsqueeze(0) if grid.dim() == 1 else grid for grid in grids],
            }
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: list[torch.Tensor],
        grid_thw: list[torch.Tensor],
        **kwargs,
    ) -> NameToTensorList:
        """Run the vision encoder on each image or video.

        Returns:
            {
                "vision_embeds": [one (vision_tokens, hidden_size) tensor each],
                "deepstack": [each one's DeepStack layers, one after another],
            }
        """
        embeds: list[torch.Tensor] = []
        deepstack: list[torch.Tensor] = []
        for pixels, grid in zip(pixel_values, grid_thw, strict=True):
            # HF vision encoder returns (hidden_states, deepstack_features)
            # depending on the model variant; handle both cases
            encoder_output = self.vision_encoder(pixels, grid_thw=grid)
            if isinstance(encoder_output, tuple):
                item_embeds, item_deepstack = encoder_output
            else:
                item_embeds = encoder_output.pooler_output
                item_deepstack = encoder_output.deepstack_features
            if isinstance(item_deepstack, torch.Tensor):
                item_deepstack = [item_deepstack]
            embeds.append(item_embeds)
            deepstack.extend(item_deepstack or ())
        return {
            "vision_embeds": embeds,
            "deepstack": deepstack or [torch.tensor([])],
        }


# ===================================================================
# 3. ThinkerSubmodule (ar engine) -- MOST COMPLEX
# ===================================================================


class _PromptPart(NamedTuple):
    """One text span or attachment of a Thinker prefill row."""
    embeds: torch.Tensor
    pos_ids: torch.Tensor  # (3, rows)
    masks: torch.Tensor  # (2, rows): multimodal, then text inclusion
    advance: int  # how far the part moves the MRoPE position
    ids: torch.Tensor | None = None  # a text span's token ids
    deepstack: list[torch.Tensor] | None = None  # a vision part's, per layer over its rows


class ThinkerSubmodule(ARNodeSubmodule):
    """Wraps the FlashInfer-based Thinker MoE transformer.

    Dispatches on graph_walk:
      - the prefill walks: the whole prompt as one row, text spans embedded
        and encoder outputs spliced in prompt order, with 3D MRoPE positions
      - thinker_decode: embed previous token, single-step decode

    Every step samples a text token. Every step but the first decode step
    streams ``thinker_states`` (layer-0 + layer-N concat) to the Talker
    partition; the prefill sends the first sampled token's embedding along
    with the prompt's, so the Talker can start without waiting for decode.
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
                tensor_inputs={
                    "masks_for_talker": self._get_decode_thinker_mask(device)
                }  # no additional tensors for decode step
            )

        return self._prefill_inputs(fwd_info, inputs, start_pos)

    @staticmethod
    def _media_masks(rows: int, device: torch.device) -> torch.Tensor:
        """An attachment's talker masks: its encoder rows are multimodal, and
        its two sentinels are text the Talker keeps."""
        multimodal = torch.ones(rows + 2, dtype=torch.bool, device=device)
        multimodal[0] = False
        multimodal[-1] = False
        return torch.stack([multimodal, ~multimodal])

    def _text_part(self, ids: torch.Tensor, pos: int) -> _PromptPart:
        device = self.get_device()
        ids = ids.to(device)
        n = ids.shape[0]
        masks = torch.stack([
            torch.zeros(ids.shape, dtype=torch.bool, device=device),  # multimodal
            talker_text_mask(ids, self.config),  # text inclusion
        ])
        return _PromptPart(
            self.model.model.embed_tokens(ids), get_rope_index_text(n, pos, device), masks, n, ids=ids,
        )

    def _audio_part(self, audio_embeds: torch.Tensor, pos: int) -> _PromptPart:
        device = self.get_device()
        audio_embeds = audio_embeds.to(device)  # (audio_tokens, hidden)
        n = audio_embeds.shape[0]
        # Position IDs:
        #   - audio_start_token: text-like position at pos
        #   - audio tokens:      temporal increments per frame,
        #                        h/w = pos (handled by helper)
        #   - audio_end_token:   text-like position right after
        pos_ids = torch.cat([
            get_rope_index_text(1, pos, device),
            get_rope_index_audio(n, pos + 1, device, self.config.thinker.position_id_per_seconds),
            get_rope_index_text(1, pos + 1 + n, device),
        ], dim=1)
        return _PromptPart(self._wrap_audio_input(audio_embeds), pos_ids, self._media_masks(n, device), n + 2)

    def _vision_part(
        self, vision_embeds: torch.Tensor, deepstack: list[torch.Tensor], grid_thw: torch.Tensor,
        seconds_per_grid: torch.Tensor | None, pos: int,
    ) -> _PromptPart:
        device = self.get_device()
        vision_embeds = vision_embeds.to(device)
        n = vision_embeds.shape[0]
        assert n == int(grid_thw.prod()) // self.config.vision.spatial_merge_size ** 2, (
            f"the vision encoder gave {n} rows, where the layout counts the grid's merged patches"
        )
        wrapped = self._wrap_vision_input(vision_embeds)
        # Vision tokens use spatial 3D positions (temporal constant for an
        # image, h/w from the spatial grid after merging), between two
        # text-like sentinel positions.
        vision_pos_ids = get_rope_index_vision(
            grid_thw.to(device),
            pos + 1,  # leave room for the BOS token
            position_id_per_seconds=self.config.thinker.position_id_per_seconds,
            device=device,
            spatial_merge_size=self.config.vision.spatial_merge_size,
            seconds_per_grid=None if seconds_per_grid is None else seconds_per_grid.item(),
        )
        end_pos_base = float(vision_pos_ids.max().item()) + 1
        pos_ids = torch.cat([
            get_rope_index_text(1, pos, device),
            vision_pos_ids,
            get_rope_index_text(1, end_pos_base, device),
        ], dim=1)
        # The deepstack features land on the encoder rows, between the sentinels.
        full_deepstack = []
        for layer in deepstack:
            full = torch.zeros_like(wrapped)
            full[1:-1] = layer
            full_deepstack.append(full)
        # The next position is one past the EOS token, which follows the
        # grid's largest position: more than the row count for a 2D grid.
        advance = int(end_pos_base + 1 - pos)
        return _PromptPart(wrapped, pos_ids, self._media_masks(n, device), advance, deepstack=full_deepstack)

    def _prefill_inputs(
        self, fwd_info: CurrentForwardPassInfo, inputs: NameToTensorList, start_pos: int,
    ) -> ARNodeInputs:
        """A whole prompt as one row, spliced in prompt order.

        `prefill_order` tags each part text, audio, image or video; the nth tag
        of a kind takes the nth tensor of that kind, and images and videos
        share the vision encoder's outputs in order. ``start_pos`` is the
        stream's next MRoPE position, and each part moves the cursor by its
        own advance, which a vision part makes larger than its row count; the
        row then reports its total for ``declare_step``.
        """
        texts = inputs.get("text_inputs", [])
        audios = inputs.get("audio_embeds", [])
        visions = inputs.get("vision_embeds", [])
        grids = inputs.get("image_grid_thw", [])
        seconds = inputs.get("video_second_per_grid", [])
        deepstack = inputs.get("deepstack", [])
        layers = len(self.config.vision.deepstack_visual_indexes)
        # the order a text-only walk ran with before it carried one
        order = fwd_info.step_metadata.get("prefill_order") or (TEXT_PART,) * len(texts)

        parts: list[_PromptPart] = []
        pos = start_pos
        text_i = audio_i = vision_i = video_i = 0
        for kind in order:
            if kind == TEXT_PART:
                part = self._text_part(texts[text_i], pos)
                text_i += 1
            elif kind == AUDIO_PART:
                part = self._audio_part(audios[audio_i], pos)
                audio_i += 1
            else:
                video = kind == VIDEO_PART
                part = self._vision_part(
                    visions[vision_i], deepstack[vision_i * layers:(vision_i + 1) * layers],
                    grids[vision_i], seconds[video_i] if video else None, pos,
                )
                vision_i += 1
                video_i += video
            parts.append(part)
            pos += part.advance

        device = self.get_device()
        input_embeds = torch.cat([part.embeds for part in parts])
        ids = [part.ids for part in parts if part.ids is not None]
        tensor_inputs: dict[str, Any] = {
            "masks_for_talker": torch.cat([part.masks for part in parts], dim=1),
            # the prompt's text, for the repetition penalty
            "text_token_ids": torch.cat(ids) if ids else torch.zeros(0, dtype=torch.long, device=device),
        }
        if vision_i:
            tensor_inputs["mrope_pos_advance"] = pos - start_pos
            tensor_inputs["deepstack"] = [
                torch.cat([
                    part.deepstack[layer] if part.deepstack else part.embeds.new_zeros(part.embeds.shape)
                    for part in parts
                ])
                for layer in range(layers)
            ]
        return ARNodeInputs(
            input_seq_len=input_embeds.shape[0],
            input_embeds=input_embeds,
            custom_pos_ids=torch.cat([part.pos_ids for part in parts], dim=1),
            tensor_inputs=tensor_inputs,
        )

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
        # every row keeps its token: a prefill is one step, so it is the last
        prefill_tokens = {
            rid: ids for rid, inp in zip(request_ids, inputs, strict=True)
            if (ids := inp.tensor_inputs.get("text_token_ids")) is not None
        }

        pos_advance = None
        if any("mrope_pos_advance" in inp.tensor_inputs for inp in inputs):
            # A vision row's 3D-grid MRoPE span is larger than its token count,
            # so it advances the counter by its own; other rows by their span.
            pos_advance = tuple(
                int(inp.tensor_inputs.get("mrope_pos_advance", inp.input_seq_len)) for inp in inputs
            )
        deepstack = any(walk in VISION_PREFILL_WALKS for walk in row_walks)
        return SubmoduleStep(
            # must match `cg_key_info`, which picked the leased capture
            cg_key_info=self.DEEPSTACK_KEY if deepstack else None,
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

        extra_inputs = {}
        if graph_walk in VISION_PREFILL_WALKS or any("deepstack" in inp.tensor_inputs for inp in inputs):
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

        # every step samples: decode, and the prefill, which is its request's only one
        logits = self.model.lm_head(hidden[-1:, :])
        new_token = engine_inputs.resources[THINKER_SAMPLER].sample(logits)
        result: NameToTensorList = {"new_token": [new_token]}
        if graph_walk != "thinker_decode" and audio_output:
            result["first_token_embed"] = [self.model.model.embed_tokens(new_token)]

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
    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return len(model_inputs) > 1

    PREFILL_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048]
    # past 4 for thinker_mixed, whose decode rows ride in these captures
    PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    # vision prefill buckets are larger than text/audio because video
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
                    tensor_inputs={
                        "masks_for_talker": self._get_decode_thinker_mask(device)
                    }
                ),
                capture_batch_sizes=[1, 2, 4, 8, 16, 32],
            ),
            PackedAcceleratorGraphConfig(
                capture_graph_walk=PREFILL_TEXT,
                replay_graph_walks=[PREFILL_TEXT, PREFILL_AUDIO, THINKER_MIXED],
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
                capture_graph_walk=PREFILL_VISION,
                replay_graph_walks=[PREFILL_VISION, PREFILL_AUDIO_VISION, THINKER_MIXED],
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
        if graph_walk in VISION_PREFILL_WALKS or graph_walk == THINKER_MIXED and any(
            info.graph_walk in VISION_PREFILL_WALKS for info in per_request_info.values()
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
        Always packs ``thinker_states`` + ``thinker_mask`` in every per-rid
        output dict so the captured CUDA graph has a static output shape
        regardless of request metadata. Per-rid filtering happens in
        ``unpack_packed_output``.
        """

        is_prefill = graph_walk != "thinker_decode"

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
            new_tokens = sampler.sample(logits)
            # what the first decode step would stream for this token, sent
            # with the prompt instead; decode rows of a mixed step drop it
            first_embeds = self.model.model.embed_tokens(new_tokens)
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
                        "new_token": [new_tokens[i:i + 1]],
                        "first_token_embed": [first_embeds[i:i + 1]],
                    } for i, rid in enumerate(request_ids)
                }
            }

        # thinker_decode (existing behavior)
        logits = self.model.lm_head(hidden)  # (batch, vocab)
        new_tokens = sampler.sample(logits)

        if layer_n_hidden is not None:
            thinker_states = torch.cat(
                [layer_0_embed, layer_n_hidden], dim=-1,
            )
        else:
            thinker_states = torch.cat(
                [layer_0_embed, layer_0_embed], dim=-1,
            )

        outputs: dict[str, NameToTensorList] = {}
        for i, rid in enumerate(request_ids):
            req_out: NameToTensorList = {
                "thinker_states": [thinker_states[i : i + 1]],
                "new_token": [new_tokens[i:i+1]]
            }
            if masks_for_talker is not None and rid in masks_for_talker:
                req_out["thinker_mask"] = [masks_for_talker[rid]]
            outputs[rid] = req_out

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

    def postprocess(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
    ):
        first_embed = outputs.pop("first_token_embed", None)
        decode = request_info.graph_walk == "thinker_decode"
        if not request_info.step_metadata.get("audio_output", True) or (
            decode and not request_info.dynamic_loop_iter_counts.get("thinker_decode_loop", 0)
        ):
            # Text-only requests send the Talker nothing, and the first decode
            # step's token already went out with the prefill.
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
                states = selected[inclusion_mask]
                multimodal_mask = multimodal_mask[inclusion_mask]
                if not decode and first_embed:
                    # the sampled token's layer-0 row, as its decode step would send it
                    states = torch.cat([states, first_embed[0].to(states.dtype)])
                    multimodal_mask = torch.cat([multimodal_mask, multimodal_mask.new_zeros(1)])
                outputs["thinker_states"] = [states]
                outputs["thinker_mask"] = [multimodal_mask]

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
        ignore_eos= request_info.resource_configs[THINKER_SAMPLER].ignore_eos
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
            if k not in ("thinker_states", "thinker_mask", "first_token_embed")
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

        # Lazy-built CodePredictor scratch KV cache, overwritten every step.
        self._cp_kv_cache: torch.Tensor | None = None

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

    def _assistant_prefix(self, speaker: str, first_token_state: torch.Tensor) -> torch.Tensor:
        """The assistant prefix's rows, matching the HF pattern: codec embeds of
        [nothink, think_bos, think_eos, speaker, pad, bos] plus text rows of
        [pad * 4, bos, the first text token]. The prompt rows before it already
        hold the assistant turn's opening tokens."""
        tc = self.config.talker
        speaker_id = tc.speaker_id.get(speaker.lower())
        if speaker_id is None:
            logger.warning(f"Speaker {speaker} not implemented")
            speaker_id = tc.codec_pad_id
        codec = self.talker_code_emb(torch.tensor([
            tc.codec_nothink_id,
            tc.codec_think_bos_id,
            tc.codec_think_eos_id,
            speaker_id,
            tc.codec_pad_id,
            tc.codec_bos_id,
        ], device=self.get_device(), dtype=torch.long))
        # W3: pad and bos embeddings use Thinker embed -> text_projection
        # (via pre-computed cached values from init_tts_embeds)
        text = torch.cat([
            self._tts_pad_embed_cached.expand(4, -1),
            self._tts_bos_embed_cached.unsqueeze(0),
            self.model.text_projection(first_token_state),
        ], dim=0)
        return codec + text

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs
    ) -> ARNodeInputs:
        device = self.get_device()

        if graph_walk == "talker_prefill":
            # The Thinker's prefill states: the prompt rows the Talker keeps,
            # then the first text token's row, which is text (layer-0).
            states = inputs["thinker_states"][0].to(device)
            multimodal_mask = inputs["thinker_mask"][0]
            prompt = self._get_talker_embeds(
                selected_hidden=states[:-1],
                multimodal_mask=multimodal_mask[:-1],
            )
            prefix = self._assistant_prefix(fwd_info.step_metadata.get("voice", "Ethan"), states[-1:])
            input_embeds = torch.cat([prompt, prefix])
            seq_len = input_embeds.shape[0]

        elif graph_walk == "talker_decode":
            dtype = self.model.text_projection.linear_fc1.weight.dtype
            input_embeds = inputs["talker_input_embeds"][0].to(dtype)

            thinker_states = inputs.get("thinker_states", [])
            rid = fwd_info.rid_handle
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

    def declare_step(
        self, graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) ->SubmoduleStep:
        # both walks sample a frame
        steps = {
            TALKER_KV: KVStep(),
            TALKER_ATTN: AttentionStep(causal=True),
            TALKER_POS: PositionStep(),
            TALKER_SAMPLER: SamplerStep(apply_penalty=True),
            CODE_PRED_SAMPLER: SamplerStep(apply_penalty=False),
        }
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
        input_embeds = torch.cat([
            inp.input_embeds for inp in inputs
        ], dim=0)
        device = self.get_device()

        extra_args = {}
        # Decode's codec scratch, one row per request. talker_prefill makes its
        # own in the forward: its packed capture would intern these by the
        # token bucket (codec_emb_sum's hidden dim aliases one).
        if graph_walk == "talker_decode":
            extra_args = self._frame_scratch(len(inputs), device)
            extra_args["suppress_mask"] = self._get_suppress_mask()
        return {
            "input_embeds": input_embeds,
            **extra_args
        }

    def _frame_scratch(self, bs: int, device: torch.device) -> dict[str, torch.Tensor]:
        """The buffers one frame's code predictor loop fills, one row per request."""
        return {
            "all_codes": torch.zeros((bs, self.num_codes), device=device, dtype=torch.long),
            "codec_emb_sum": torch.zeros((bs, self.cp_cfg.hidden_size), device=device),
            "pos_buf": torch.zeros((bs, 1), device=device, dtype=torch.long),
        }

    def _sample_frame(
        self,
        last_hidden: torch.Tensor,
        talker_sampler: SamplerResource,
        code_pred_sampler: SamplerResource,
        suppress_mask: torch.Tensor,
        all_codes: torch.Tensor,
        codec_emb_sum: torch.Tensor,
        pos_buf: torch.Tensor,
        **kwargs
    ) -> NameToTensorList:
        """Sample one codec frame per row of ``last_hidden`` (bs, hidden): the
        layer-0 code from the codec head, then the code predictor's depth loop
        for the other groups."""
        logits = self.model.codec_head(last_hidden)
        logits = logits.masked_fill(suppress_mask.unsqueeze(0), float("-inf"))
        # Apply the repetition penalty to the Talker's layer-0 codes (the code
        # predictor depth loop below samples through its own aux config and
        # stays penalty-free). The penalty is baked into the captured graph.
        layer0_codes = talker_sampler.sample(logits)

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

            # Params come from the "code_predictor" aux sampler's static
            # buffers, so per-request values survive graph replay.
            tokens = code_pred_sampler.sample(logits)
            all_codes[:, group_idx] = tokens
            embed = codec_embedding[group_idx - 1](tokens)
            codec_emb_sum.add_(embed)

        return {
            "talker_input_embeds": [codec_emb_sum],
            "codec_tokens": [all_codes],
            "new_token": [layer0_codes]
        }

    def _frame_from_hidden(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, last_hidden: torch.Tensor, **kwargs,
    ) -> NameToTensorList:
        if graph_walk == "talker_prefill":
            # made here, not in preprocess: see `preprocess`
            kwargs = {
                **self._frame_scratch(last_hidden.shape[0], last_hidden.device),
                "suppress_mask": self._get_suppress_mask(),
            }
        return self._sample_frame(
            last_hidden,
            talker_sampler=engine_inputs.resources[TALKER_SAMPLER],
            code_pred_sampler=engine_inputs.resources[CODE_PRED_SAMPLER],
            **kwargs,
        )

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        hidden = self.model(input_embeds=input_embeds, label="main")
        # one request: a prefill samples from its last row
        last_hidden = hidden[-1:, :] if graph_walk == "talker_prefill" else hidden
        return self._frame_from_hidden(graph_walk, engine_inputs, last_hidden, **kwargs)

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """Batched Talker forward shared between ``talker_decode`` and ``talker_prefill``.

        Decode (1 token per request, ``hidden`` shape ``(bs, hidden)``) and
        prefill (each request's prompt rows and assistant prefix, packed) both
        sample a frame from each request's last row, inline through the
        injected samplers, and emit per-rid {talker_input_embeds, codec_tokens,
        new_token} entries.
        """
        assert graph_walk in ("talker_decode", "talker_prefill")
        hidden = self.model(input_embeds=input_embeds, label="main")
        if graph_walk == "talker_prefill":
            last_hidden = engine_inputs.resources[TALKER_ATTN].select_last_hidden(hidden)
        else:
            last_hidden = hidden
        fwd_out = self._frame_from_hidden(graph_walk, engine_inputs, last_hidden, **kwargs)

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
        super().cleanup_request(request_id)
        self._eos_embed_sent.discard(request_id)

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return batch.graph_walk in [
            "talker_decode", "talker_prefill"
        ] and len(model_inputs) <= self.MAX_BATCH_SIZE

    def max_batch_size(self, graph_walk):
        return self.MAX_BATCH_SIZE

    # A prefill row is the prompt rows the Talker keeps (an image's merged
    # patches among them) and the assistant prefix.
    TALKER_PREFILL_TOKEN_BUCKETS = [128, 256, 512, 1024, 2048]
    TALKER_PREFILL_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]

    # Assistant prefix per request: pad*4 + bos + the first text token.
    ASSISTANT_PREFIX_ROWS = 6

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
    def _maybe_context_only(
        cls, request_ids, per_request_input_metadata: Mapping[str, InputMetadata] | None,
    ) -> bool:
        """Whether every window may be a terminal flush that emits no audio:
        its new items are at most the Talker's codec_eos frame, which ends
        every stream the Talker stops itself, so a reply whose frames end on a
        chunk boundary ends with such a flush. The counts can't tell that
        frame from a real one, so ``preprocess`` checks the decoded frames."""
        for rid in request_ids:
            chunk = cls._chunk(rid, per_request_input_metadata)
            if chunk is None or chunk.num_items - chunk.context_items > int(chunk.is_final):
                return False
        return bool(request_ids)

    def cg_key_info(
        self,
        graph_walk: str,
        per_request_info: Mapping[str, CurrentForwardPassInfo],
        per_request_input_metadata: Mapping[str, InputMetadata] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._maybe_context_only(list(per_request_info), per_request_input_metadata):
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

        # The window's leading frames that an earlier window already decoded.
        # EOS is only ever a stream's last item, but it can be a context item
        # of the final flush (the previous window's last new item). The filter
        # above dropped it, so the clamp keeps the context within the frames.
        chunk = (input_metadata.stream_chunks.get("codec_tokens") if input_metadata is not None else None)
        orig_seq_len = codec_tokens.shape[0]
        context = min(chunk.context_items if chunk is not None else 0, orig_seq_len)

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
        if not engine_inputs.captured and self._maybe_context_only(engine_inputs.request_ids, metadata) and all(
            inp.kwargs["frames"] <= self._chunk(rid, metadata).context_items
            for rid, inp in zip(engine_inputs.request_ids, inputs, strict=True)
        ):
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

