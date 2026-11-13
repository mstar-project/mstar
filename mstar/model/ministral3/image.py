"""Host image preparation and Pixtral's row-delimited prompt layout.

The checkpoint's fast processor resizes byte RGB images with torchvision's
bicubic filter, then normalizes in float32. No upstream model runs here.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.model.ministral3.components.vision import PixtralConfig, pack_image_patches
from mstar.model.multimodal import MediaPads, as_uint8_chw
from mstar.model.multimodal_decoder import MediaPrompt, PreparedMedia

IMAGE_GRIDS = "pixtral_image_grids"


@dataclass(frozen=True)
class ImagePrompt:
    """Decoder pad runs and whole-image grids have different boundaries."""

    pads: MediaPads
    inputs: NameToTensorList
    image_grids: list[tuple[int, int, int]]


class Ministral3ImageProcessor:
    def __init__(self, directory: str | Path, config: PixtralConfig, tokenizer):
        self.config = config
        root = json.loads((Path(directory) / "config.json").read_text())
        processor = json.loads((Path(directory) / "processor_config.json").read_text())
        params = processor["image_processor"]
        if (processor["processor_class"] != "PixtralProcessor"
                or processor["patch_size"] != config.patch_size
                or params["patch_size"] != config.patch_size
                or processor["spatial_merge_size"] != config.spatial_merge_size
                or params["resample"] != 3 or params.get("do_center_crop")
                or not all(params[k] for k in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"))):
            raise ValueError("Unsupported Pixtral image processor configuration")
        self.longest_edge = params["size"]["longest_edge"]
        alignment = config.patch_size * config.spatial_merge_size
        if self.longest_edge < alignment or self.longest_edge % alignment:
            raise ValueError("Pixtral longest_edge must be a positive multiple of patch_size * spatial_merge_size")
        scale = params["rescale_factor"]
        if scale <= 0:
            raise ValueError("Pixtral rescale_factor must be positive")
        # Match the fast reference's fused rescale/normalize arithmetic.
        self.mean = torch.tensor(params["image_mean"], dtype=torch.float32) * (1.0 / scale)
        self.std = torch.tensor(params["image_std"], dtype=torch.float32) * (1.0 / scale)
        if self.mean.shape != (3,) or self.std.shape != (3,) or not (self.std > 0).all():
            raise ValueError("Pixtral requires three RGB normalization channels with positive standard deviations")
        tokens = tuple(processor[k] for k in ("image_token", "image_break_token", "image_end_token"))
        ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in tokens)
        if any(tokenizer.encode(token, add_special_tokens=False) != [i] for token, i in zip(tokens, ids, strict=True)):
            raise ValueError("Pixtral image markers must each encode as one special token")
        if len(set(ids)) != 3 or ids[0] != root["image_token_index"]:
            raise ValueError("Pixtral image token IDs disagree with the checkpoint")
        self.image_id, self.break_id, self.end_id = ids
        self.special_tokens = tokens

    def resized_size(self, height: int, width: int) -> tuple[int, int]:
        if min(height, width) <= 0:
            raise ValueError("Pixtral images must have nonempty height and width")
        ratio = max(height / self.longest_edge, width / self.longest_edge)
        if ratio > 1:
            height, width = math.floor(height / ratio), math.floor(width / ratio)
        if min(height, width) == 0:
            raise ValueError("Pixtral image aspect ratio produces an empty resized dimension")
        alignment = self.config.patch_size * self.config.spatial_merge_size
        return ((height + alignment - 1) // alignment * alignment,
                (width + alignment - 1) // alignment * alignment)

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms.v2 import functional as tvf

        if (not isinstance(image, torch.Tensor) or image.device.type != "cpu" or image.ndim != 3
                or image.shape[0] not in (1, 3, 4)):
            raise ValueError("Pixtral expects CPU CHW images with 1, 3 or 4 channels")
        if image.is_floating_point():
            if not torch.isfinite(image).all() or not ((image >= 0) & (image <= 1)).all():
                raise ValueError("Floating image inputs must contain finite pixels in [0, 1]")
        elif image.dtype != torch.uint8:
            raise ValueError("Pixtral images must be uint8 or floating pixels in [0, 1]")
        size = self.resized_size(*image.shape[-2:])
        image = as_uint8_chw(image)
        resized = tvf.resize(image, list(size), interpolation=tvf.InterpolationMode.BICUBIC, antialias=True)
        return tvf.normalize(resized.float(), self.mean, self.std)

    def prepare(self, image: torch.Tensor, metadata: dict) -> PreparedMedia:
        pixels, _, grids = pack_image_patches([self.normalize(image)], self.config)
        _, h, w = grids[0]
        return PreparedMedia({"pixel_values": pixels}, [h * w // self.config.spatial_merge_size ** 2], grids)

    def add_row_delimiters(self, prompt: MediaPrompt, max_tokens: int) -> ImagePrompt:
        """Split the scaffold's contiguous image pads into rows, preserving text.

        Each row ends in IMG_BREAK, except the last, which ends in IMG_END.
        Pad offsets/counts describe actual contiguous decoder runs; whole-image
        grids are carried separately so attention cannot split an image by row.
        """
        pads = prompt.pads
        grids = [tuple(grid) for grid in pads.grid]
        merge = self.config.spatial_merge_size
        length = pads.ids.numel() + sum(h // merge for _, h, _ in grids)
        if length > max_tokens:
            raise ValueError("Ministral 3 prompt exceeds max_position_embeddings after image expansion")
        expanded = MediaPads(pads.ids, [], [], [], [])
        pieces, cursor, size = [], 0, 0
        for offset, count, output, (t, h, w) in zip(pads.offset, pads.tokens, pads.output, grids, strict=True):
            rows, columns = h // merge, w // merge
            if t != 1 or count != rows * columns:
                raise ValueError("Pixtral image grid disagrees with its placeholder count")
            pieces.append(pads.ids[cursor:offset])
            size += offset - cursor
            for row in range(rows):
                expanded.output.append(output)
                expanded.grid.append(())
                expanded.offset.append(size)
                expanded.tokens.append(columns)
                pieces.append(pads.ids.new_full((columns,), self.image_id))
                pieces.append(pads.ids.new_tensor([self.end_id if row == rows - 1 else self.break_id]))
                size += columns + 1
            cursor = offset + count
        if pieces:
            expanded = expanded._replace(ids=torch.cat([*pieces, pads.ids[cursor:]]))
        return ImagePrompt(expanded, prompt.inputs, grids)
