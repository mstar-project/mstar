"""Phi-4-multimodal's images: the processor's dynamic-HD intake, and the
``vision_encoder`` node over the tower in ``components/vision.py``.

Intake (`image_intake`, the data worker) reproduces ``Phi4MMImageProcessor``: the
crop grid closest to the image's aspect within ``max_crops``, a bilinear resize
(PIL's, as the processor runs it) and a white pad to that grid, the crops, the
bicubic global view, and the padded patches. It ships the crops as bytes and the
global view, which the bicubic leaves off the byte grid, in fp32; the node
normalizes the bytes on the device, as the processor rounds. Each image's grid
``(crops down, crops across, valid patch rows, valid patch cols)`` rides the
walk's ``MEDIA_LAYOUT``, and from it the node derives everything else on the
host (`image_plan`).

The node packs the valid patches of every crop of every image of the step, each
image's global view first, one ragged attention segment per crop, and emits one
``vision_embeds`` row per image pad in prompt order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from mstar.communication.tensors import NameToTensorList
from mstar.engine.accelerator_graph_config import PiecewiseAcceleratorGraphConfig
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedAttnBackend,
)
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.model.components.block_loop_encoder import PackedBlockLoopEncoder
from mstar.model.keys import VISION_ATTN
from mstar.model.multimodal import MediaEncoder, as_uint8_chw
from mstar.model.multimodal_decoder import PreparedMedia
from mstar.model.phi4mm.components.vision import Phi4MMVisionConfig, Phi4MMVisionModel
from mstar.model.phi4mm.config import IMAGE_PAD_ID
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs
from mstar.utils.h2d import to_device_nonblocking

VISION_NODE = "vision_encoder"
VISION_OUTPUT = "vision_embeds"
# an image's crops, bytes ``[crops, 3, S, S]``, and its global view, fp32 ``[1, 3, S, S]``
PIXEL_VALUES = "pixel_values"
GLOBAL_PIXEL_VALUES = "global_pixel_values"

# `pool_index` codes for a row that is not a pooled cell of encoded patches,
# mapped past the step's patches once the step is packed (`Phi4MMVisionModel.project`)
MASKED, SUB_GN, GLB_GN = -1, -2, -3


def vision_media_encoder() -> MediaEncoder:
    """The ``vision_encoder`` node as the VLM scaffold's walks see it."""
    return MediaEncoder("vision", VISION_NODE, (PIXEL_VALUES, GLOBAL_PIXEL_VALUES), VISION_OUTPUT, (IMAGE_PAD_ID,))


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------


class ImageGrid(NamedTuple):
    """An image's layout, its ``MEDIA_LAYOUT`` grid: the crop grid, and how many
    patch rows and columns of the padded canvas hold image (the rest is the
    white pad, whose patches the processor masks)."""

    crops_down: int
    crops_across: int
    valid_rows: int
    valid_cols: int


def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    # `Phi4MMImageProcessor.find_closest_aspect_ratio`, tie-break included
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


class _HDLayout(NamedTuple):
    grid: ImageGrid
    # the resized image, before the pad
    width: int
    height: int


def hd_layout(width: int, height: int, config: Phi4MMVisionConfig) -> _HDLayout:
    """`Phi4MMImageProcessor.dynamic_preprocess`'s crop grid, resize and patch
    mask, for an image of ``width`` x ``height`` pixels. A ``ValueError`` (a
    400) where the processor refuses the image."""
    size, patch, max_num = config.crop_size, config.patch_size, config.max_crops
    w_crop_num, h_crop_num = math.ceil(width / float(size)), math.ceil(height / float(size))
    if w_crop_num * h_crop_num > max_num:
        target_ratios = sorted(
            {(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
             if 1 <= i * j <= max_num},
            key=lambda x: x[0] * x[1],
        )
        across, down = _closest_aspect_ratio(width / height, target_ratios, width, height, size)
    else:
        across, down = w_crop_num, h_crop_num
    target_width, target_height = size * across, size * down
    ratio_width, ratio_height = target_width / width, target_height / height
    if ratio_width < ratio_height:
        new_w, new_h = target_width, int(height * ratio_width)
        pad_w, pad_h = 0, target_height - int(height * ratio_width)
    else:
        new_w, new_h = int(width * ratio_height), target_height
        pad_w, pad_h = target_width - int(width * ratio_height), 0
    side = size // patch
    rows = side * down - (pad_h // patch if pad_h >= patch else 0)
    cols = side * across - (pad_w // patch if pad_w >= patch else 0)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"a {width}x{height} image leaves no unpadded patch after its resize")
    if min(new_h, target_height) < 10 or min(new_w, target_width) < 10:
        raise ValueError(f"a {width}x{height} image has too extreme an aspect ratio (resized to {new_w}x{new_h})")
    return _HDLayout(ImageGrid(down, across, rows, cols), new_w, new_h)


def image_tokens(grid: Sequence[int], config: Phi4MMVisionConfig) -> int:
    """The image's pads: its kept crop cells, a ``sub_GN`` closing each of their
    rows, ``glb_GN``, and the global view's cells with theirs."""
    _, _, rows, cols = grid
    half = config.patches_per_side // 2
    kept_rows, kept_cols = (rows + 1) // 2, (cols + 1) // 2
    return kept_rows * (kept_cols + 1) + 1 + half * (half + 1)


def image_intake(image: torch.Tensor, metadata: dict, config: Phi4MMVisionConfig) -> PreparedMedia:
    """One image (``Model.load_image``'s ``[3, H, W]``, float in [0, 1] or
    bytes) as the processor prepares it: ``pixel_values``, the crops in
    row-major order as bytes ``[crops, 3, S, S]``, ``global_pixel_values``,
    the normalized canvas resized bicubically to one crop, fp32 ``[1, 3, S, S]``,
    and its tokens and `ImageGrid`."""
    from PIL import Image

    pixels = as_uint8_chw(image)
    _, height, width = pixels.shape
    layout = hd_layout(width, height, config)
    down, across = layout.grid.crops_down, layout.grid.crops_across
    size = config.crop_size
    pil = Image.fromarray(np.ascontiguousarray(pixels.permute(1, 2, 0).numpy()))
    if (layout.width, layout.height) != (width, height):
        # torchvision's resize of a PIL image: PIL's bilinear, antialiased
        pil = pil.resize((layout.width, layout.height), Image.Resampling.BILINEAR)
    canvas = torch.full((3, size * down, size * across), 255, dtype=torch.uint8)
    canvas[:, :layout.height, :layout.width] = torch.from_numpy(np.array(pil)).permute(2, 0, 1)
    crops = (
        canvas.view(3, down, size, across, size).permute(1, 3, 0, 2, 4).reshape(down * across, 3, size, size)
        .contiguous()
    )
    # ToTensor + Normalize, then the processor's bicubic of the whole canvas
    half = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    normalized = canvas.float().div(255).sub_(half).div_(half)
    global_view = F.interpolate(normalized[None], size=(size, size), mode="bicubic")
    grid = layout.grid
    return PreparedMedia(
        {PIXEL_VALUES: crops, GLOBAL_PIXEL_VALUES: global_view},
        [image_tokens(grid, config)],
        [tuple(grid)],
    )


# ---------------------------------------------------------------------------
# The node's host maths over a grid
# ---------------------------------------------------------------------------


def navit_position_ids(rows: int, cols: int, side: int) -> torch.Tensor:
    """A crop's position ids over its ``rows`` x ``cols`` valid patches, in
    row-major order: ``SiglipVisionEmbeddings``' fractional bucketing, op for
    op (a full crop gets ``arange(side ** 2)``)."""
    boundaries = torch.arange(1 / side, 1.0, 1 / side)
    coords_h = torch.arange(0, 1 - 1e-6, 1 / torch.tensor(rows))
    coords_w = torch.arange(0, 1 - 1e-6, 1 / torch.tensor(cols))
    bucket_h = torch.bucketize(coords_h, boundaries, right=True)
    bucket_w = torch.bucketize(coords_w, boundaries, right=True)
    ids = (bucket_h[:, None] * side + bucket_w).flatten()
    assert ids.numel() == rows * cols
    return ids


class ImagePlan(NamedTuple):
    """One image's pack, relative to its own first patch and first crop."""

    crops: int
    # attention segments: the global view, then each crop with a valid patch
    segments: tuple[int, ...]
    patches: int
    tokens: int
    # per packed patch
    position_ids: torch.Tensor
    # each crop patch's row of the crops' patch embedding (``crop * side**2 + patch``)
    crop_rows: torch.Tensor
    # ``[tokens, 4]``: a pack index, or `MASKED` / `SUB_GN` / `GLB_GN`
    pool_index: torch.Tensor


def _pool_cells(lookup: torch.Tensor, cell_rows: int, cell_cols: int) -> torch.Tensor:
    """``[cell_rows * (cell_cols + 1), 4]``: each 2x2 cell of ``lookup`` in
    ``avg_pool2d``'s order, a ``sub_GN`` closing every row."""
    cells = torch.stack([
        lookup[0:2 * cell_rows:2, 0:2 * cell_cols:2], lookup[0:2 * cell_rows:2, 1:2 * cell_cols:2],
        lookup[1:2 * cell_rows:2, 0:2 * cell_cols:2], lookup[1:2 * cell_rows:2, 1:2 * cell_cols:2],
    ], dim=-1)
    sep = torch.full((cell_rows, 1, 4), SUB_GN, dtype=torch.long)
    return torch.cat([cells, sep], dim=1).reshape(-1, 4)


@lru_cache(maxsize=256)
def image_plan(grid: tuple[int, int, int, int], side: int) -> ImagePlan:
    """The pack of an image of `ImageGrid` ``grid``, ``side`` patches a crop
    side: the global view's patches, then each crop's valid ones (row-major
    within the crop), and the HD transform's rows over them in ``sub_glb``
    order. Cached: grids repeat, and the tensors are only read."""
    down, across, rows, cols = grid
    full = side * side
    canvas = torch.full((side * down, side * across), MASKED, dtype=torch.long)
    segments, position_ids, crop_rows = [full], [navit_position_ids(side, side, side)], []
    offset = full
    for k in range(down * across):
        r, c = divmod(k, across)
        vh, vw = min(max(rows - side * r, 0), side), min(max(cols - side * c, 0), side)
        if vh == 0 or vw == 0:
            # wholly padding: its cells lie past the kept region
            continue
        n = vh * vw
        canvas[side * r:side * r + vh, side * c:side * c + vw] = torch.arange(offset, offset + n).view(vh, vw)
        local = torch.arange(vh)[:, None] * side + torch.arange(vw)
        crop_rows.append((k * full + local).flatten())
        position_ids.append(navit_position_ids(vh, vw, side))
        segments.append(n)
        offset += n
    kept_rows, kept_cols, half = (rows + 1) // 2, (cols + 1) // 2, side // 2
    pool_index = torch.cat([
        _pool_cells(canvas, kept_rows, kept_cols),
        torch.full((1, 4), GLB_GN, dtype=torch.long),
        _pool_cells(torch.arange(full).view(side, side), half, half),
    ])
    return ImagePlan(
        crops=down * across, segments=tuple(segments), patches=offset, tokens=pool_index.shape[0],
        position_ids=torch.cat(position_ids), crop_rows=torch.cat(crop_rows), pool_index=pool_index,
    )


class StepIndices(NamedTuple):
    """A packed step's host-built indices (`Phi4MMVisionModel.embed` / ``project``)."""

    patch_src: torch.Tensor
    position_ids: torch.Tensor
    pool_index: torch.Tensor


def step_indices(plans: Sequence[ImagePlan], side: int) -> StepIndices:
    """Every image's plan of a step, packed in order: the patch embedding runs
    over every global view, then every crop."""
    full = side * side
    images = len(plans)
    total = sum(p.patches for p in plans)
    src, pool = [], []
    crop_base = patch_base = 0
    global_rows = torch.arange(full)
    for g, plan in enumerate(plans):
        src += [global_rows + g * full, plan.crop_rows + (images + crop_base) * full]
        pool.append(torch.where(plan.pool_index >= 0, plan.pool_index + patch_base, plan.pool_index))
        crop_base += plan.crops
        patch_base += plan.patches
    pool_index = torch.cat(pool)
    # MASKED, SUB_GN, GLB_GN -> the rows `project` appends after the patches
    pool_index = torch.where(pool_index >= 0, pool_index, total - 1 - pool_index)
    return StepIndices(torch.cat(src), torch.cat([p.position_ids for p in plans]), pool_index)


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


class _NodeConfig(NamedTuple):
    """What `PackedBlockLoopEncoder` reads of its config."""

    hidden_size: int
    head_dim: int


class Phi4MMVisionEncoderSubmodule(PackedBlockLoopEncoder):
    """Images -> one LLM-width row per image pad, every image of the step's
    requests packed into one call (STATELESS).

    Each crop attends to itself alone, a segment of the ragged attention
    resource under ``VISION_ATTN``; only valid patches are packed. The grid maths
    runs on the host from the walk's ``MEDIA_LAYOUT`` (`image_plan`, cached by
    grid), so nothing reads the device.
    """

    # The block loop's capture ladder (not captured yet): a 36-crop image, 37 * 1024 =
    # 37888 patches, fits the top bucket alone; a step past it will run eagerly.
    BLOCK_LOOP_PATCH_BUCKETS = [1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960]
    MAX_BATCH_SIZE = 8
    # Shapes vary per step; the block loop's capture is to be the accelerator.
    disable_torch_compile = True
    # The tower runs in its own dtype (bf16), as the reference does; its
    # LayerNorms and softmax accumulate in fp32 regardless.
    disable_autocast = True
    BLOCK_LOOP_DTYPE = torch.bfloat16

    def __init__(self, model: Phi4MMVisionModel, config: Phi4MMVisionConfig):
        super().__init__(model, _NodeConfig(config.hidden_size, config.head_dim), output=VISION_OUTPUT)
        self.vision_config = config

    def get_piecewise_accelerator_graph_configs(self, *args, **kwargs) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        # the block loop runs eagerly until its capture has a parity test
        return {}

    def prepare_inputs(self, graph_walk, fwd_info, inputs: NameToTensorList, **kwargs) -> NodeInputs:
        side = self.vision_config.patches_per_side
        plans = [image_plan(tuple(grid), side) for grid in self.grids(fwd_info)]
        crops, views = inputs.get(PIXEL_VALUES, []), inputs.get(GLOBAL_PIXEL_VALUES, [])
        if len(crops) != len(plans) or len(views) != len(plans):
            raise ValueError(
                f"{len(crops)} crop sets and {len(views)} global views for the layout's {len(plans)} images"
            )
        for i, (plan, image) in enumerate(zip(plans, crops, strict=True)):
            if image.shape[0] != plan.crops:
                raise ValueError(f"image {i} has {image.shape[0]} crops for its grid's {plan.crops}")

        def cat(parts: list[torch.Tensor]) -> torch.Tensor:
            return parts[0] if len(parts) == 1 else torch.cat(parts)

        return NodeInputs(
            # what the engine leases the captured block loop's bucket by
            input_seq_len=sum(p.patches for p in plans),
            tensor_inputs={PIXEL_VALUES: cat(crops), GLOBAL_PIXEL_VALUES: cat(views)},
            # one segment per crop and global view, in pack order
            resource_step_info={"main": [n for p in plans for n in p.segments]},
            kwargs={"plans": plans},
        )

    def preprocess(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs]):
        out = super().preprocess(graph_walk, engine_inputs, inputs)
        device = self.get_device()
        plans = [plan for inp in inputs for plan in inp.kwargs["plans"]]
        indices = step_indices(plans, self.vision_config.patches_per_side)
        out.update({name: to_device_nonblocking(t, device) for name, t in indices._asdict().items()})
        # rows per request, to cut the output back apart
        out["row_counts"] = tuple(sum(p.tokens for p in inp.kwargs["plans"]) for inp in inputs)
        return out

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        segment_lengths: Mapping[str, Sequence[int]],
        row_patches: tuple[int, ...] = (),
        row_counts: tuple[int, ...] = (),
        **tensors: torch.Tensor,
    ) -> dict[str, NameToTensorList]:
        hidden, cos, sin = self.model.embed(**tensors)
        replayed = self._replay_block_loop(engine_inputs, hidden, cos, sin, segment_lengths)
        encoded = self.model.encode(hidden, cos, sin) if replayed is None else replayed.get_view("hidden")
        embeds = self.model.project(encoded, **tensors)
        rows = embeds.split(list(row_counts))
        return {rid: {self.output: [r]} for rid, r in zip(engine_inputs.request_ids, rows, strict=True)}


def vision_resources(config: Phi4MMVisionConfig, node: str = VISION_NODE) -> list[NodeResourceSpec]:
    """The tower's attention, its only resource: one segment per crop and per
    global view, every request's in a step."""
    return [
        RaggedAttentionSpec(
            resource_key=VISION_ATTN, nodes={node},
            config=RaggedAttentionConfig(
                num_qo_heads=config.num_attention_heads,
                num_kv_heads=config.num_attention_heads,
                head_dim=config.head_dim,
                # a step's requests, an image each with its global view and crops; a
                # step with more runs its requests apart (`_fits_capture`)
                max_segments_per_step=Phi4MMVisionEncoderSubmodule.MAX_BATCH_SIZE * (1 + config.max_crops),
                # FA3 varlen takes head_dim 72 as it is; FlashInfer pads it to 128.
                # Falls back to FlashInfer where FA3 can't load.
                backend=RaggedAttnBackend.DENSE,
            ),
        ),
    ]


def create_vision_submodule(
    checkpoint_dir: str, device: str, dtype: torch.dtype, config: Phi4MMVisionConfig | None = None,
) -> Phi4MMVisionEncoderSubmodule:
    """The node with its weights loaded in ``dtype`` on ``device``."""
    from mstar.model.multimodal_decoder import MultimodalDecoderModel
    from mstar.model.phi4mm.components.vision import load_vision_weights

    config = config or Phi4MMVisionConfig.from_dir(checkpoint_dir)
    model = MultimodalDecoderModel._build(lambda: Phi4MMVisionModel(config), dtype, device)
    load_vision_weights(model, checkpoint_dir, device=device)
    return Phi4MMVisionEncoderSubmodule(model.eval(), config)
