"""Streaming checkpoint -> native module loading for Qwen-Image-2.1.

The transformer keeps the diffusers names; q/k/v stack into ``to_qkv`` and the SwiGLU's
``gate_layer`` / ``proj`` into ``gate_proj`` (sharded under TP by the parallel linears'
loaders), and ``to_out.0`` drops its index. The text encoder and the vision tower come from
the Qwen3-VL checkpoint in ``text_encoder/``; the VAE is diffusers' own module.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.diffusion.qwen.encoder import QwenHiddenStateEncoder
from mstar.model.components.diffusion.qwen.weight_loading import (
    build_qwen3_vl_tower,
    qkv_rules,
    remap_text_encoder_key,
    text_encoder_skip,
)
from mstar.model.components.diffusion.weight_loading import (
    iter_diffusers_component,
    iter_transformers_component,
    load_native,
    materialize,
)
from mstar.model.components.qwen_vision import QwenVisionModel
from mstar.model.loader.base import StackedParamRule
from mstar.model.qwen_image21.components.transformer import QwenImage21Transformer
from mstar.model.qwen_image21.config import QwenImage21Config, QwenImage21TransformerConfig

TRANSFORMER_RULES = [
    StackedParamRule(".attn.to_qkv", ".attn.to_q", "q"),
    StackedParamRule(".attn.to_qkv", ".attn.to_k", "k"),
    StackedParamRule(".attn.to_qkv", ".attn.to_v", "v"),
    StackedParamRule(".img_mlp.gate_proj", ".img_mlp.gate_layer", 0),
    StackedParamRule(".img_mlp.gate_proj", ".img_mlp.proj", 1),
]


def remap_transformer_key(name: str) -> str:
    return name.replace(".attn.to_out.0.", ".attn.to_out.")


def load_transformer(
    module: QwenImage21Transformer, weights, what: str = "Qwen-Image-2.1 transformer",
) -> QwenImage21Transformer:
    load_native(module, weights, remap_transformer_key, what, stacked_params=TRANSFORMER_RULES)
    module.set_prefix_modulation()
    return module


def build_transformer(
    config: QwenImage21TransformerConfig, snapshot: Path, device, comm_group: CommGroup | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> QwenImage21Transformer:
    with torch.device("meta"):
        dit = QwenImage21Transformer(config, comm_group)
    materialize(dit, dtype, device)
    return load_transformer(dit, iter_diffusers_component(snapshot / "transformer", device))


def build_text_encoder(config: QwenImage21Config, snapshot: Path, device, dtype: torch.dtype = torch.bfloat16
                       ) -> QwenHiddenStateEncoder:
    """Qwen3-VL-8B's language model, every layer; the final norm, the LM head and the vision tower
    stay on disk."""
    with torch.device("meta"):
        encoder = QwenHiddenStateEncoder(config.text)
    materialize(encoder, dtype, device)
    skip = text_encoder_skip(config.text)
    return load_native(
        encoder, iter_transformers_component(Path(snapshot) / "text_encoder", device, skip=skip),
        remap_text_encoder_key, "Qwen-Image-2.1 text encoder", stacked_params=qkv_rules(config.text), skip=skip,
    )


def build_vision_tower(snapshot: Path, device, dtype: torch.dtype = torch.bfloat16) -> QwenVisionModel:
    """Qwen3-VL-8B's vision tower with its deepstack mergers, from the text encoder's shards."""
    return build_qwen3_vl_tower(Path(snapshot) / "text_encoder", device, "Qwen-Image-2.1 vision tower", dtype)


def build_vae(snapshot: Path, device, dtype: torch.dtype = torch.bfloat16):
    """diffusers' ``AutoencoderKLQwenImage21``, in the pipeline's dtype."""
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(str(Path(snapshot) / "vae"), torch_dtype=dtype)
    return vae.to(device).eval()
