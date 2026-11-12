"""Building the Qwen-Image modules from a local checkpoint snapshot."""

from __future__ import annotations

from pathlib import Path

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.diffusion.qwen.encoder import QwenHiddenStateEncoder
from mstar.model.components.diffusion.qwen.weight_loading import qkv_rules, remap_text_encoder_key, text_encoder_skip
from mstar.model.components.diffusion.weight_loading import (
    iter_diffusers_component,
    iter_transformers_component,
    load_native,
    materialize,
)
from mstar.model.components.qwen_vision import QWEN25_VISION_PREFIXES, Qwen25VisionModel, qwen25_vision_key
from mstar.model.loader.base import load_weights_into
from mstar.model.qwen_image.components.transformer import QwenImageTransformer
from mstar.model.qwen_image.config import QwenImageConfig


def build_transformer(
    config: QwenImageConfig, snapshot: Path, device, comm_group: CommGroup | None = None, dtype=torch.bfloat16,
) -> QwenImageTransformer:
    with torch.device("meta"):
        dit = QwenImageTransformer(config.transformer, comm_group)
    materialize(dit, dtype, device)
    return load_native(
        dit, iter_diffusers_component(Path(snapshot) / "transformer", device), lambda name: name,
        "Qwen-Image transformer",
    )


def build_text_encoder(config: QwenImageConfig, snapshot: Path, device, dtype=torch.bfloat16) -> QwenHiddenStateEncoder:
    """Qwen2.5-VL's language model through its normed last hidden state; the tower and the LM head stay on disk."""
    with torch.device("meta"):
        encoder = QwenHiddenStateEncoder(config.text_encoder)
    materialize(encoder, dtype, device)
    skip = text_encoder_skip(config.text_encoder)
    return load_native(
        encoder, iter_transformers_component(Path(snapshot) / "text_encoder", device, skip=skip),
        remap_text_encoder_key, "Qwen2.5-VL language model", stacked_params=qkv_rules(config.text_encoder), skip=skip,
    )


def build_vision_tower(config: QwenImageConfig, snapshot: Path, device, dtype=torch.bfloat16) -> Qwen25VisionModel:
    """The text encoder checkpoint's vision tower. Its rotary ``inv_freq`` is derived (``materialize`` recomputes it,
    fp32), so every parameter, not every buffer, must come from the checkpoint."""
    with torch.device("meta"):
        tower = Qwen25VisionModel(config.vision)
    materialize(tower, dtype, device)

    def skip(name: str) -> bool:
        return not name.startswith(QWEN25_VISION_PREFIXES)

    load_weights_into(
        tower, iter_transformers_component(Path(snapshot) / "text_encoder", device, skip=skip),
        name_remapper=qwen25_vision_key, skip_predicate=skip, strict=True,
    )
    return tower


def build_vae(snapshot: Path, device, dtype=torch.bfloat16) -> torch.nn.Module:
    """``AutoencoderKLQwenImage``, whose state dict is ``AutoencoderKLWan``'s key for key, in diffusers' Wan class."""
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(str(snapshot), subfolder="vae", torch_dtype=dtype).to(device).eval()
