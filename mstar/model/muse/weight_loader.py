"""Loading a Muse Glimmer checkpoint into this stack.

The text stack sits under ``model.language_model``; the attention's output
gate (``self_attn.gate_proj``) and the MLP's (``mlp.gate_proj``) share a
suffix, so the stacked rules name their module. The vision tower, adapter and
projection load into one module, as one node.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import torch
from torch import nn

from mstar.model.loader.base import StackedParamRule, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards

_STACKED_PARAMS: list[StackedParamRule] = [
    StackedParamRule(".self_attn.qkvg_proj", ".self_attn.q_proj", "q"),
    StackedParamRule(".self_attn.qkvg_proj", ".self_attn.gate_proj", "gate"),
    StackedParamRule(".self_attn.qkvg_proj", ".self_attn.k_proj", "k"),
    StackedParamRule(".self_attn.qkvg_proj", ".self_attn.v_proj", "v"),
    StackedParamRule(".mlp.gate_up_proj", ".mlp.gate_proj", 0),
    StackedParamRule(".mlp.gate_up_proj", ".mlp.up_proj", 1),
]

_TEXT_PREFIX = "model.language_model."
_LM_HEAD = "lm_head.weight"
_VISION = {
    "model.vision_tower.": "",
    "model.vision_adapter.fc1.": "adapter_fc1.",
    "model.vision_adapter.fc2.": "adapter_fc2.",
    "model.vision_projection.": "projection.",
}


def text_name_remapper(name: str) -> str | None:
    if name == _LM_HEAD:
        return name
    if not name.startswith(_TEXT_PREFIX):
        return None
    return "model." + name[len(_TEXT_PREFIX):]


def vision_name_remapper(name: str) -> str | None:
    for prefix, ours in _VISION.items():
        if name.startswith(prefix):
            return ours + name[len(prefix):]
    return None


def _load(model, path, device, remapper, selectors, stacked=None) -> set[str]:
    """Fill every parameter of ``model`` from tensors that all have one, or
    raise: a half-loaded tower, or a checkpoint tensor this port has no
    parameter for (a weighted norm, a bias), produces plausible garbage."""
    weights = itertools.chain.from_iterable(
        iter_safetensors_shards(Path(path), device=device, **selector) for selector in selectors
    )
    return load_weights_into(
        model, weights, stacked_params=stacked, name_remapper=remapper, strict=True,
    )


def load_text_weights(
    model: nn.Module, path: str | Path, device: torch.device | str = "cpu",
) -> set[str]:
    return _load(
        model, path, device, text_name_remapper,
        [{"prefix": _TEXT_PREFIX}, {"keys": {_LM_HEAD}}],
        stacked=_STACKED_PARAMS,
    )


def load_vision_weights(
    model: nn.Module, path: str | Path, device: torch.device | str = "cpu",
) -> set[str]:
    return _load(
        model, path, device, vision_name_remapper,
        [{"prefix": prefix} for prefix in _VISION],
    )
