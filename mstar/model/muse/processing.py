"""Muse Glimmer's tokenizer, chat template and image preprocessing.

Native rather than through ``AutoProcessor``: the HF processor classes need a
transformers newer than the serving venv pins. Ported from transformers'
``image_processing_muse_glimmer.py`` and ``processing_muse_glimmer.py``
(Apache-2.0).
"""
from __future__ import annotations

import itertools
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mstar.model.multimodal import as_uint8_chw

IMAGE_START = "<|image_start|>"
IMAGE_END = "<|image_end|>"
# the template writes one per image; the prompt expands it to the image's tokens
IMAGE_PAD = "<|patch|>"


def load_tokenizer(local_dir: str | Path):
    """The fast tokenizer with the repo's chat template and special tokens.

    Built from ``tokenizer.json`` directly: the repo's ``tokenizer_config.json``
    is in a format older transformers cannot read.
    """
    from transformers import PreTrainedTokenizerFast

    local_dir = Path(local_dir)
    cfg = json.loads((local_dir / "tokenizer_config.json").read_text())
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(local_dir / "tokenizer.json"),
        bos_token=cfg.get("bos_token"),
        eos_token=cfg.get("eos_token"),
        pad_token=cfg.get("pad_token"),
    )
    template = local_dir / "chat_template.jinja"
    if template.exists():
        tokenizer.chat_template = template.read_text()
    return tokenizer


def smart_resize(height: int, width: int, patch_size: int, max_tokens: int) -> tuple[int, int]:
    """The integer merged-patch grid closest to the aspect ratio under the
    token cap, in pixels. No floor: a small image is scaled up."""
    ideal_h = height / patch_size
    ideal_w = width / patch_size
    ratio = ideal_w / ideal_h if ideal_h > 0 else 1.0
    if ideal_h * ideal_w > max_tokens:
        ideal_h = (max_tokens / ratio) ** 0.5
        ideal_w = ideal_h * ratio
    candidates = [
        (gh, gw)
        for gh, gw in set(itertools.product(
            [math.floor(ideal_h), math.ceil(ideal_h)],
            [math.floor(ideal_w), math.ceil(ideal_w)],
        ))
        if gh >= 1 and gw >= 1 and gh * gw <= max_tokens
    ]
    if not candidates:
        # a side under one merged patch at the cap (a long, thin image): it
        # takes one, and the other as many as the cap leaves
        gh = min(max(1, round(ideal_h)), max_tokens)
        candidates = [(gh, min(max(1, round(ideal_w)), max_tokens // gh))]
    gh, gw = min(candidates, key=lambda g: abs(g[0] / g[1] - height / width))
    return gh * patch_size, gw * patch_size


def _lanczos(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The reference's resize: torchvision's antialiased Lanczos on the uint8
    tensor, which needs torchvision >= 0.27; PIL's Lanczos before that."""
    import torchvision
    from torchvision.transforms.v2 import functional as tvF

    major, minor = (int(v) for v in torchvision.__version__.split(".")[:2])
    if (major, minor) >= (0, 27):
        return tvF.resize(
            image, [height, width], interpolation=tvF.InterpolationMode.LANCZOS, antialias=True,
        )
    warnings.warn(
        "torchvision < 0.27 has no tensor Lanczos; Muse images resize with PIL's, "
        "whose pixels differ slightly from the reference's",
        stacklevel=2,
    )
    pil = Image.fromarray(image.permute(1, 2, 0).numpy())
    resized = np.asarray(pil.resize((width, height), Image.Resampling.LANCZOS))
    return torch.from_numpy(resized.copy()).permute(2, 0, 1)


def preprocess_image(
    image, patch_size: int = 14, merge_size: int = 2, temporal_patch_size: int = 2,
    max_tokens: int = 4096,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """``(pixel_values [h * w, 1176], (1, h, w))`` with the grid in patches.

    Lanczos resize, then ``(x / 255 - 0.5) / 0.5``. Each patch is laid out
    ``(temporal, channel, py, px)``, the one frame repeated across the
    temporal pair.
    """
    pixels = as_uint8_chw(image)
    height, width = pixels.shape[-2:]
    th, tw = smart_resize(height, width, patch_size * merge_size, max_tokens)
    x = _lanczos(pixels, th, tw).float() / 255.0
    x = (x - 0.5) / 0.5
    gh, gw = th // patch_size, tw // patch_size
    patches = x.view(3, gh, patch_size, gw, patch_size).permute(1, 3, 0, 2, 4)
    patches = patches.unsqueeze(2).expand(-1, -1, temporal_patch_size, -1, -1, -1)
    return patches.reshape(gh * gw, -1).contiguous(), (1, gh, gw)
