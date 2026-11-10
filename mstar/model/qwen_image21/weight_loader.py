"""Streaming checkpoint -> native module loading for Qwen-Image-2.1.

The transformer keeps the diffusers names; q/k/v stack into ``to_qkv`` and the SwiGLU's
``gate_layer`` / ``proj`` into ``gate_proj`` (sharded under TP by the parallel linears'
loaders), and ``to_out.0`` drops its index.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.diffusion.weight_loading import iter_diffusers_component, load_native, materialize
from mstar.model.loader.base import StackedParamRule
from mstar.model.qwen_image21.components.transformer import QwenImage21Transformer
from mstar.model.qwen_image21.config import QwenImage21TransformerConfig

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
