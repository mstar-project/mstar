"""Loading a Qwen3.5 checkpoint into this stack.

Mostly a rename: the text stack sits under ``language_model`` and the delta
net mixer is ``linear_attn`` (ours: ``self_attn``); vision tensors lose their
``model.visual.`` prefix. ``_STACKED_PARAMS`` routes the separate q/k/v, gate/up
and delta net input projections by shard id into their fused parameters. The two
towers load separately, as separate nodes. The MTP head (``mtp.*``) is skipped.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mstar.model.loader.base import StackedParamRule, load_weights_into
from mstar.model.loader.iterators import iter_safetensors_shards

# ints index `MergedColumnParallelLinear.output_sizes`; names key
# `SPLIT_SHARD_BLOCKS`
_STACKED_PARAMS: list[StackedParamRule] = [
    StackedParamRule(".qkv_proj", ".q_proj", "q"),
    StackedParamRule(".qkv_proj", ".k_proj", "k"),
    StackedParamRule(".qkv_proj", ".v_proj", "v"),
    StackedParamRule(".gate_up_proj", ".gate_proj", 0),
    StackedParamRule(".gate_up_proj", ".up_proj", 1),
    StackedParamRule(".in_proj_fused", ".in_proj_qkv", "qkv"),
    StackedParamRule(".in_proj_fused", ".in_proj_z", "z"),
    StackedParamRule(".in_proj_fused", ".in_proj_a", "a"),
    StackedParamRule(".in_proj_fused", ".in_proj_b", "b"),
]

_TEXT_PREFIX = "model.language_model."
_VISION_PREFIX = "model.visual."
# at the checkpoint's root, and only when untied (9B and 27B)
_LM_HEAD = "lm_head.weight"


def qwen3_5_name_remapper(name: str) -> str | None:
    if name == _LM_HEAD:
        # top level in the checkpoint and in ours, so it passes through
        return name
    if not name.startswith(_TEXT_PREFIX):
        return None
    return name.replace(_TEXT_PREFIX, "model.").replace(
        ".linear_attn.", ".self_attn."
    )


def qwen3_5_vision_name_remapper(name: str) -> str | None:
    if not name.startswith(_VISION_PREFIX):
        return None
    return name[len(_VISION_PREFIX):]


def _load(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str,
    remapper,
    selectors: list[dict],
    expected: set[str],
    stacked: list[StackedParamRule] | None = None,
) -> set[str]:
    """Fill ``expected`` from the shards each selector picks out, or raise.

    A selector (``prefix=`` or ``keys=``) only narrows which shards open;
    ``remapper`` decides what loads. An unfilled parameter is fatal, since a
    half-loaded tower produces plausible garbage.
    """
    loaded: set[str] = set()
    for selector in selectors:
        loaded |= load_weights_into(
            model,
            iter_safetensors_shards(Path(path), device=device, **selector),
            name_remapper=remapper,
            stacked_params=stacked,
        )
    missing = sorted(expected - loaded)
    if missing:
        raise ValueError(
            f"{len(missing)} parameter(s) got no weight, e.g. {missing[:5]}"
        )
    return loaded


def load_qwen3_5_weights(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    """Load the text tower. Returns the parameter paths that were filled."""
    tied = model.config.tie_word_embeddings
    selectors = [{"prefix": _TEXT_PREFIX}]
    if not tied:
        # an untied head sits at the root, outside the text prefix
        selectors.append({"keys": {_LM_HEAD}})
    return _load(
        model, path, device, qwen3_5_name_remapper, selectors,
        expected={
            n for n, _ in model.named_parameters()
            # tied: the checkpoint carries no tensor for it
            if not (tied and n == _LM_HEAD)
        },
        stacked=_STACKED_PARAMS,
    )


def load_qwen3_5_vision_weights(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str = "cpu",
) -> set[str]:
    """Load the ViT. Returns the parameter paths that were filled."""
    return _load(
        model, path, device, qwen3_5_vision_name_remapper,
        [{"prefix": _VISION_PREFIX}],
        expected={n for n, _ in model.named_parameters()},
    )
