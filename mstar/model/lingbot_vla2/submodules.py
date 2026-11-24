"""LingBot-VLA 2.0 node submodules for vision encoding, VLM prefix prefill, and action expert flow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.state import ArrayStep, Segment, SlotLease, SubmoduleStep
from mstar.model.components.lingbot_vla2_action_expert import LingBotVLAActionExpert
from mstar.model.components.lingbot_vla2_prefix import (
    LingBotVLA2PrefixQueries,
    compact_prefix,
)
from mstar.model.components.mrope import compute_inv_freq, layout_position_ids
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.components.qwen3_vit import (
    vision_interpolation,
)
from mstar.model.components.qwen3_vit import (
    vision_position_ids as vision_grid_position_ids,
)
from mstar.model.components.qwen_vision import QwenVisionModel
from mstar.model.lingbot_vla2.config import (
    LLM_KV,
    VISION_TOKENS,
    LingBotVLA2Config,
)
from mstar.model.pi05.kernels.image_normalize import normalize_float_images
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule
from mstar.utils.h2d import to_device_nonblocking

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch


def patchify_lingbot_images(
    images: Sequence[torch.Tensor] | torch.Tensor,
    *,
    image_size: int,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
    """Resize, normalize (mean=0.5, std=0.5), and patchify camera images for Qwen3-VL ViT.

    Args:
        images: Either a tensor of shape ``[N, 3, H, W]`` or a sequence of ``[3, H, W]``
            RGB images (``uint8`` in ``[0, 255]`` or floating-point in ``[0, 1]``).
        image_size: Target square height and width (divisible by ``patch_size * spatial_merge_size``).
        patch_size: ViT spatial patch size.
        temporal_patch_size: ViT temporal patch size (frames are repeated to fill it).
        spatial_merge_size: Spatial merger block factor.

    Returns:
        ``(pixel_values, grid_thw)`` where ``pixel_values`` has shape
        ``[N * gh * gw, 3 * temporal_patch_size * patch_size * patch_size]`` in ``float32``
        and ``grid_thw`` has one ``(1, gh, gw)`` tuple per image.
    """
    stride = patch_size * spatial_merge_size
    if image_size <= 0 or image_size % stride != 0:
        raise ValueError(
            f"image_size {image_size} must be positive and divisible by "
            f"patch_size * spatial_merge_size ({stride})"
        )

    image_list = list(images) if not isinstance(images, torch.Tensor) else [images[i] for i in range(images.shape[0])]
    if not image_list:
        raise ValueError("LingBot-VLA 2.0 requires at least one camera image")

    prepared: list[torch.Tensor] = []
    for img in image_list:
        if img.ndim != 3:
            raise ValueError(f"Expected camera image [C, H, W], got shape {tuple(img.shape)}")
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        elif img.shape[0] == 4:
            img = img[:3]
        elif img.shape[0] != 3:
            raise ValueError(f"Expected 1, 3, or 4 channels, got {img.shape[0]}")

        if img.dtype == torch.uint8:
            img = img.to(torch.float32) / 127.5 - 1.0
        else:
            img = normalize_float_images(img.to(torch.float32))

        if img.shape[-2] != image_size or img.shape[-1] != image_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        prepared.append(img)

    normalized = torch.stack(prepared, dim=0)  # [N, 3, H, W] in [-1, 1]
    n_imgs, c, h, w = normalized.shape
    gh, gw = h // patch_size, w // patch_size
    bh, bw = gh // spatial_merge_size, gw // spatial_merge_size

    frames = normalized.unsqueeze(1).expand(n_imgs, temporal_patch_size, c, h, w)
    patches = frames.reshape(
        n_imgs,
        temporal_patch_size,
        c,
        bh,
        spatial_merge_size,
        patch_size,
        bw,
        spatial_merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    patches = patches.reshape(n_imgs * gh * gw, c * temporal_patch_size * patch_size * patch_size)
    grid = [(1, gh, gw)] * n_imgs
    return patches, grid


def _image_layout_from_tokens(
    text_ids: torch.Tensor,
    *,
    image_token_id: int,
    gh: int,
    gw: int,
    tokens_per_image: int,
) -> dict[str, list[Any]]:
    """Build the ``mrope.layout_position_ids`` dict from contiguous ``image_token_id`` runs."""
    is_img = (text_ids == image_token_id).tolist()
    offsets: list[int] = []
    tokens: list[int] = []
    idx = 0
    length = len(is_img)
    while idx < length:
        if not is_img[idx]:
            idx += 1
            continue
        start = idx
        while idx < length and is_img[idx]:
            idx += 1
        run_len = idx - start
        if run_len != tokens_per_image:
            raise ValueError(
                f"Contiguous image_token_id run at offset {start} has length {run_len}, "
                f"expected tokens_per_image={tokens_per_image}"
            )
        offsets.append(start)
        tokens.append(run_len)
    return {
        "offset": offsets,
        "tokens": tokens,
        "grid": [(1, gh, gw)] * len(offsets),
    }


def _recompute_fp32_rotary_buffers(module: torch.nn.Module) -> None:
    """Keep non-persistent rotary inverse frequency tables in float32 across device/dtype casts."""
    if isinstance(module, LingBotVLA2ViTEncoderSubmodule):
        rotary = module.encoder.rotary_pos_emb
        target = rotary.inv_freq.device
        if target.type != "meta":
            rotary.register_buffer("inv_freq", rotary._inv_freq(target), persistent=False)
    elif isinstance(module, LingBotVLA2LLMSubmodule):
        param = next(module.language_model.parameters())
        if param.device.type != "meta":
            inv = compute_inv_freq(
                module.config.text.head_dim,
                module.config.text.rope_theta,
                device=param.device,
            )
            module.language_model.register_buffer("mrope_inv_freq", inv, persistent=False)
    elif isinstance(module, LingBotVLA2ActionSubmodule):
        param = next(module.action_expert.parameters())
        if param.device.type != "meta":
            module.action_expert.reset_non_persistent_buffers(device=param.device)


class LingBotVLA2ViTEncoderSubmodule(NodeSubmodule):
    """Qwen3-VL ViT encoder node emitting merged visual tokens and deepstack features."""

    def __init__(self, encoder: QwenVisionModel, config: LingBotVLA2Config):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.vision_cfg = config.vision

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.vision = handles[VISION_TOKENS]

    def _apply(self, fn, recurse=True):
        out = super()._apply(fn, recurse=recurse)
        # Keep VisionRotaryEmbedding's non-persistent inv_freq buffer in float32
        # across .to(dtype) and .bfloat16()/.float() casts.
        _recompute_fp32_rotary_buffers(self)
        return out

    def can_batch(
        self,
        batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        del batch
        return len(model_inputs) > 1

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        del graph_walk, fwd_info, kwargs
        images = inputs.get("image_inputs") or []
        if not images:
            raise ValueError("LingBot-VLA 2.0 vit_encoder requires at least one image in image_inputs")
        if len(images) > self.config.num_cameras:
            raise ValueError(
                f"Received {len(images)} camera images, max num_cameras={self.config.num_cameras}"
            )
        device = self.get_device()
        device_images = [to_device_nonblocking(img, device) for img in images]
        patches, grid = patchify_lingbot_images(
            device_images,
            image_size=self.config.image_size,
            patch_size=self.vision_cfg.patch_size,
            temporal_patch_size=self.vision_cfg.temporal_patch_size,
            spatial_merge_size=self.vision_cfg.spatial_merge_size,
        )
        return NodeInputs(
            input_seq_len=int(patches.shape[0]),
            tensor_inputs={"pixel_values": patches},
            resource_step_info={"main": [gh * gh for _, gh, _ in grid]},
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        per_request_info: Mapping[int, CurrentForwardPassInfo] | None = None,
        **kwargs: Any,
    ) -> SubmoduleStep:
        del graph_walk, slot_lease, piecewise_leases, per_request_info, kwargs
        gh = self.config.image_size // self.vision_cfg.patch_size
        patches_per_img = gh * gh
        segments = [
            Segment(request_id=rid, label="main", span=patches_per_img)
            for rid, inp in zip(request_ids, inputs, strict=True)
            for _ in range(inp.input_seq_len // patches_per_img)
        ]
        return SubmoduleStep(segments=tuple(segments), state={VISION_TOKENS: ArrayStep()})

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: Sequence[NodeInputs],
    ) -> dict[str, Any]:
        del graph_walk, engine_inputs
        device = self.get_device()
        gh = self.config.image_size // self.vision_cfg.patch_size
        patches_per_img = gh * gh
        merge = self.vision_cfg.spatial_merge_size
        num_images_list = [inp.input_seq_len // patches_per_img for inp in inputs]
        grid = [(1, gh, gh)] * sum(num_images_list)
        indices, weights = vision_interpolation(grid, self.vision_cfg.num_grid_per_side, merge)
        position_ids = vision_grid_position_ids(grid, merge)
        pixel_values = torch.cat(
            [inp.tensor_inputs["pixel_values"] for inp in inputs], dim=0
        ).to(device=device, dtype=self.encoder.patch_embed.proj.weight.dtype)
        return {
            "pixel_values": pixel_values,
            "indices": to_device_nonblocking(indices, device),
            "weights": to_device_nonblocking(weights, device),
            "position_ids": to_device_nonblocking(position_ids, device),
            "num_images_list": num_images_list,
        }

    def _run_encoder(
        self,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_embeds, deepstack = self.encoder(
            pixel_values, indices, weights, position_ids
        )
        if deepstack:
            deepstack_tensor = torch.stack(deepstack, dim=0)
        else:
            deepstack_tensor = torch.empty(
                (0, visual_embeds.shape[0], visual_embeds.shape[1]),
                dtype=visual_embeds.dtype,
                device=visual_embeds.device,
            )
        return visual_embeds, deepstack_tensor

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
        num_images_list: list[int],
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, num_images_list, kwargs
        visual_embeds, deepstack_tensor = self._run_encoder(
            pixel_values, indices, weights, position_ids
        )
        return {
            "img_emb": [visual_embeds],
            "deepstack_features": [deepstack_tensor],
        }

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
        num_images_list: list[int],
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        visual_embeds, deepstack_tensor = self._run_encoder(
            pixel_values, indices, weights, position_ids
        )
        merged_per_img = self.config.tokens_per_image
        outputs: dict[str, NameToTensorList] = {}
        offset = 0
        for rid, num_images in zip(engine_inputs.request_ids, num_images_list, strict=True):
            count = num_images * merged_per_img
            outputs[rid] = {
                "img_emb": [visual_embeds[offset : offset + count]],
                "deepstack_features": [deepstack_tensor[:, offset : offset + count, :]],
            }
            offset += count
        return outputs


class LingBotVLA2LLMSubmodule(NodeSubmodule):
    """Qwen3-VL language backbone prefix prefill + alignment queries writer to ``LLM_KV``."""

    def __init__(
        self,
        language_model: Qwen3DenseLM,
        prefix_queries: LingBotVLA2PrefixQueries,
        config: LingBotVLA2Config,
    ):
        super().__init__()
        self.language_model = language_model
        self.prefix_queries = prefix_queries
        self.config = config

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv = handles[LLM_KV]

    def _apply(self, fn, recurse=True):
        out = super()._apply(fn, recurse=recurse)
        _recompute_fp32_rotary_buffers(self)
        return out

    def can_batch(
        self,
        batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        del batch
        return len(model_inputs) > 1

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        del graph_walk, fwd_info, kwargs
        device = self.get_device()
        text_ids = inputs["text_inputs"][0].to(dtype=torch.long)
        img_emb = inputs["img_emb"][0]
        deepstack = inputs["deepstack_features"][0]

        gh = self.config.image_size // self.config.vision.patch_size
        merge = self.config.vision.spatial_merge_size
        layout = _image_layout_from_tokens(
            text_ids,
            image_token_id=self.config.image_token_id,
            gh=gh,
            gw=gh,
            tokens_per_image=self.config.tokens_per_image,
        )
        text_ids_dev = to_device_nonblocking(text_ids, device)
        pos_3d, pos_extent = layout_position_ids(
            layout, merge, int(text_ids.numel()), start=0, device=device
        )
        num_queries = self.prefix_queries.config.num_query_tokens
        total_tokens = int(text_ids.numel()) + num_queries
        return NodeInputs(
            input_seq_len=total_tokens,
            tensor_inputs={
                "text_ids": text_ids_dev,
                "img_emb": to_device_nonblocking(img_emb, device),
                "deepstack_features": to_device_nonblocking(deepstack, device),
                "position_ids_3d": pos_3d,
            },
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        per_request_info: Mapping[int, CurrentForwardPassInfo] | None = None,
        **kwargs: Any,
    ) -> SubmoduleStep:
        del graph_walk, slot_lease, piecewise_leases, per_request_info, kwargs
        segments = tuple(
            Segment(request_id=rid, label="main", span=inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        )
        return SubmoduleStep(segments=segments, state={LLM_KV: ArrayStep()})

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: Sequence[NodeInputs],
    ) -> dict[str, Any]:
        del graph_walk, engine_inputs
        extra_tokens = self.prefix_queries()
        packed_embeds: list[torch.Tensor] = []
        packed_pos_3d: list[torch.Tensor] = []
        per_req_ds: list[tuple[torch.Tensor, ...]] = []
        prefix_end_positions: list[torch.Tensor] = []

        for inp in inputs:
            text_ids = inp.tensor_inputs["text_ids"]
            img_emb = inp.tensor_inputs["img_emb"]
            deepstack_tensor = inp.tensor_inputs["deepstack_features"]
            pos_3d = inp.tensor_inputs["position_ids_3d"]

            token_embeds = self.language_model.embed(text_ids)
            image_mask = text_ids == self.config.image_token_id
            ds_list = [deepstack_tensor[k] for k in range(deepstack_tensor.shape[0])]
            packed = compact_prefix(
                token_embeds,
                pos_3d,
                image_mask,
                visual_embeds=img_emb,
                deepstack_features=ds_list,
                extra_prefix_tokens=extra_tokens,
            )
            packed_embeds.append(packed.input_embeds)
            packed_pos_3d.append(packed.position_ids_3d)
            per_req_ds.append(packed.deepstack_addends)
            pos_end = (
                int(packed.position_ids_3d.max().item()) + 1
                if packed.position_ids_3d.numel() > 0
                else 0
            )
            prefix_end_positions.append(
                torch.tensor([pos_end], dtype=torch.long, device=token_embeds.device)
            )

        num_ds = len(per_req_ds[0]) if per_req_ds else 0
        merged_ds = [
            torch.cat([req_ds[k] for req_ds in per_req_ds], dim=0)
            for k in range(num_ds)
        ]
        return {
            "input_embeds": torch.cat(packed_embeds, dim=0),
            "position_ids_3d": torch.cat(packed_pos_3d, dim=1),
            "deepstack_addends": merged_ds,
            "prefix_end_positions": prefix_end_positions,
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        position_ids_3d: torch.Tensor,
        deepstack_addends: list[torch.Tensor],
        prefix_end_positions: list[torch.Tensor],
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        self.language_model(
            input_embeds,
            position_ids_3d=position_ids_3d,
            deepstack_addends=deepstack_addends or None,
        )
        return {"prefix_done": [prefix_end_positions[0]]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        position_ids_3d: torch.Tensor,
        deepstack_addends: list[torch.Tensor],
        prefix_end_positions: list[torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        self.language_model(
            input_embeds,
            position_ids_3d=position_ids_3d,
            deepstack_addends=deepstack_addends or None,
        )
        return {
            rid: {"prefix_done": [pos]}
            for rid, pos in zip(engine_inputs.request_ids, prefix_end_positions, strict=True)
        }


class LingBotVLA2ActionSubmodule(NodeSubmodule):
    """Flow-matching continuous action expert reading cached VLM KV via ``PrefixSuffixGroups``."""

    def __init__(self, action_expert: LingBotVLAActionExpert, config: LingBotVLA2Config):
        super().__init__()
        self.action_expert = action_expert
        self.config = config

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv = handles[LLM_KV]

    def _apply(self, fn, recurse=True):
        out = super()._apply(fn, recurse=recurse)
        _recompute_fp32_rotary_buffers(self)
        return out

    def can_batch(
        self,
        batch: ExecutingBatch,
        model_inputs: list[NodeInputs],
    ) -> bool:
        del batch
        return len(model_inputs) > 1

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        del graph_walk, kwargs
        device = self.get_device()
        state = to_device_nonblocking(inputs["robot_state"][0], device).to(torch.float32)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
        if state.ndim != 1 or state.shape[0] != self.config.max_state_dim:
            raise ValueError(
                f"Expected robot_state shape ({self.config.max_state_dim},), got {tuple(state.shape)}"
            )
        prefix_end = to_device_nonblocking(inputs["prefix_done"][0], device).to(torch.long).reshape(1)

        generator = torch.Generator(device=device).manual_seed(fwd_info.random_seed)
        noisy_actions = torch.randn(
            self.config.chunk_size,
            self.config.max_action_dim,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        return NodeInputs(
            input_seq_len=1 + self.config.chunk_size,
            tensor_inputs={
                "state": state,
                "noisy_actions": noisy_actions,
                "prefix_end": prefix_end,
            },
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[NodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        per_request_info: Mapping[int, CurrentForwardPassInfo] | None = None,
        **kwargs: Any,
    ) -> SubmoduleStep:
        del graph_walk, slot_lease, piecewise_leases, per_request_info, kwargs
        segments = tuple(
            Segment(request_id=rid, label="main", span=inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        )
        return SubmoduleStep(segments=segments, state={LLM_KV: ArrayStep(keep=False)})

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: Sequence[NodeInputs],
    ) -> dict[str, Any]:
        del graph_walk, engine_inputs
        return {
            "state": torch.stack([inp.tensor_inputs["state"] for inp in inputs], dim=0),
            "noisy_actions": torch.stack(
                [inp.tensor_inputs["noisy_actions"] for inp in inputs], dim=0
            ),
            "prefix_end_positions": torch.cat(
                [inp.tensor_inputs["prefix_end"] for inp in inputs], dim=0
            ),
        }

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        prefix_end_positions: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        actions = self.action_expert.denoise_actions(
            state,
            noisy_actions,
            prefix_end_positions,
            num_steps=self.config.num_flow_steps,
        )
        return {"actions": [actions[0]]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        prefix_end_positions: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        actions = self.action_expert.denoise_actions(
            state,
            noisy_actions,
            prefix_end_positions,
            num_steps=self.config.num_flow_steps,
        )
        return {
            rid: {"actions": [actions[i]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }


__all__ = [
    "LingBotVLA2ActionSubmodule",
    "LingBotVLA2LLMSubmodule",
    "LingBotVLA2ViTEncoderSubmodule",
    "patchify_lingbot_images",
]
