"""Loading a Qwen3.5 checkpoint into this stack.

Mostly a rename: the text stack sits under ``language_model`` and the delta
net mixer is ``linear_attn`` (ours: ``self_attn``); vision tensors lose their
``model.visual.`` prefix. ``_STACKED_PARAMS`` routes the separate q/k/v, gate/up
and delta net input projections by shard id into their fused parameters. The two
towers load separately, as separate nodes. The MTP head (``mtp.*``) is skipped.

A MoE checkpoint (``qwen3_5_moe``) stores each routed expert's projections
apart (``mlp.experts.<e>.gate_proj.weight``); the remap names them per expert and
``stacked_params`` routes them into the block's stacked ``experts.gate_up_proj``
/ ``experts.down_proj``, a quantized checkpoint's scales beside their weights.
Each TP rank reads only its slice of the experts (``expert_read_slices``).
"""
from __future__ import annotations

import json
import re
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
# one routed expert's tensor (any suffix a quantized checkpoint gives its weight)
_EXPERT = re.compile(r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight\w*)$")
# an FP8 KV cache's calibrated scales (nvidia's NVFP4 checkpoints); the cache here is bf16
_KV_SCALES = (".k_scale", ".v_scale")


def stacked_params(num_experts: int = 0) -> list[StackedParamRule]:
    """``_STACKED_PARAMS`` behind one rule per routed expert projection. The
    expert rules go first: the dense ``.gate_proj`` rule would also match an
    expert's remapped name."""
    rules = [
        StackedParamRule(f".experts.{target}", f".experts.{proj}.__expert{e}__.weight", f"{kind}:{e}")
        for e in range(num_experts)
        for proj, target, kind in (
            ("gate_proj", "gate_up_proj", "gate"), ("up_proj", "gate_up_proj", "up"),
            ("down_proj", "down_proj", "down"),
        )
    ]
    return rules + _STACKED_PARAMS


def qwen3_5_name_remapper(name: str) -> str | None:
    if name == _LM_HEAD:
        # top level in the checkpoint and in ours, so it passes through
        return name
    if not name.startswith(_TEXT_PREFIX) or name.endswith(_KV_SCALES):
        return None
    name = name.replace(_TEXT_PREFIX, "model.").replace(
        ".linear_attn.", ".self_attn."
    )
    expert = _EXPERT.match(name) if ".experts." in name else None
    if expert is not None:
        prefix, index, proj, suffix = expert.groups()
        return f"{prefix}.experts.{proj}.__expert{index}__.{suffix}"
    return name


def expert_read_slices(
    keys, moe_intermediate_size: int, tp_rank: int, tp_size: int,
) -> dict[str, tuple[int, int, int]]:
    """``key -> (dim, start, end)``: the part of each routed expert's tensor
    this rank loads, so a TP rank reads an eighth of the experts at TP 8, not
    all of them. Gate and up are sliced by rows, down by columns (the
    intermediate width each); NVFP4 packs two values a byte along K and keeps a
    scale per 16, block fp8 a scale per 128 x 128. Per-tensor scalars are read
    whole, and so is a format whose scale blocks would straddle the slice."""
    keys = list(keys)
    shard = moe_intermediate_size // tp_size
    if tp_size == 1 or shard * tp_size != moe_intermediate_size:
        return {}
    nvfp4 = {key.rpartition(".")[0] for key in keys if key.endswith(".weight_scale_2")}
    fp8_block = {key.rpartition(".")[0] for key in keys if key.endswith(".weight_scale_inv")}
    specs: dict[str, tuple[int, int, int]] = {}
    for key in keys:
        if ".experts." not in key:
            continue
        expert = _EXPERT.match(key)
        if expert is None:
            continue
        _, _, proj, suffix = expert.groups()
        base = key.rpartition(".")[0]
        down = proj == "down_proj"
        if base in nvfp4:
            unit = {"weight": 2 if down else 1, "weight_scale": 16 if down else 1}.get(suffix)
            if unit is not None and shard % 16:
                unit = None
        elif base in fp8_block:
            unit = {"weight": 1, "weight_scale_inv": 128}.get(suffix)
            if unit is not None and shard % 128:
                unit = None
        else:
            unit = 1 if suffix == "weight" else None
        if unit is None:
            continue
        span = shard // unit
        specs[key] = (1 if down else 0, tp_rank * span, (tp_rank + 1) * span)
    return specs


def _checkpoint_keys(path: Path) -> list[str]:
    """Every tensor name of the checkpoint, from its index or its one file's header (no data read)."""
    index = path / "model.safetensors.index.json"
    if index.is_file():
        return list(json.loads(index.read_text())["weight_map"])
    from mstar.quantization.checkpoint import read_headers

    return list(read_headers(path))


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
    expected,
    stacked: list[StackedParamRule] | None = None,
    slice_spec=None,
) -> set[str]:
    """Fill ``expected()`` from the shards each selector picks out, or raise.

    A selector (``prefix=`` or ``keys=``) only narrows which shards open;
    ``remapper`` decides what loads. An unfilled parameter is fatal, since a
    half-loaded tower produces plausible garbage. ``expected`` is asked after
    the load: a layer loaded as a quantized checkpoint stores it under other
    names (``mstar.quantization``).
    """
    loaded: set[str] = set()
    for selector in selectors:
        loaded |= load_weights_into(
            model,
            iter_safetensors_shards(Path(path), device=device, slice_spec=slice_spec, **selector),
            name_remapper=remapper,
            stacked_params=stacked,
        )
    missing = sorted(expected() - loaded)
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
    config = model.config
    tied = config.tie_word_embeddings
    selectors = [{"prefix": _TEXT_PREFIX}]
    if not tied:
        # an untied head sits at the root, outside the text prefix
        selectors.append({"keys": {_LM_HEAD}})
    slices = None
    group = model.model.comm_group
    if config.is_moe and group.world_size > 1:
        slices = expert_read_slices(
            _checkpoint_keys(Path(path)), config.moe_intermediate_size, group.rank, group.world_size,
        ).get
    return _load(
        model, path, device, qwen3_5_name_remapper, selectors,
        expected=lambda: {
            n for n, _ in model.named_parameters()
            # tied: the checkpoint carries no tensor for it
            if not (tied and n == _LM_HEAD)
        },
        stacked=stacked_params(config.num_experts),
        slice_spec=slices,
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
        expected=lambda: {n for n, _ in model.named_parameters()},
    )
