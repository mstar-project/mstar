"""Static LoRA merging for the DiT scaffold.

Serving-time LoRA here means folding the adapter into the base weights once at load
(``W += scale * alpha / r * B @ A``): zero cost per step, CUDA graphs and compiled kernels
see plain Linear weights, and the merged model behaves exactly like the reference pipeline
after ``load_lora_weights`` + ``fuse_lora``. Per-request adapter switching is out of scope.

The checkpoint layouts this accepts, and how a module path finds its parameter, are
``mstar.engine.resources.lora.checkpoint``'s; here ``<module>`` is a *diffusers* module path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import torch
from torch import nn

from mstar.engine.resources.lora.checkpoint import (
    KeyConverter,
    LoraAdapter,
    LoraFile,
    read_lora_file,
)
from mstar.model.components.linear import FusedColumnLinear
from mstar.model.loader.base import StackedParamRule, apply_stacked

logger = logging.getLogger(__name__)


def resolve_lora_module(
    module_path: str,
    remap: Callable[[str], str],
    stacked_params: Iterable[StackedParamRule] = (),
) -> tuple[str, str | int | None]:
    """``(native parameter, shard id)`` an adapter module's delta lands on.

    The same route the weight loader takes: the checkpoint remap, then the stacked-shard
    rules that send e.g. ``q_proj`` into a fused ``qkv_proj`` (``shard id`` None otherwise).
    """
    return apply_stacked(remap(module_path + ".weight"), list(stacked_params))


def _merge_delta(param: nn.Parameter, delta: torch.Tensor, owner: nn.Module, shard_id) -> None:
    """``param[rows] += delta`` in fp32 with one final rounding to the parameter dtype."""
    if shard_id is None:
        target = param.data
    else:
        if not isinstance(owner, FusedColumnLinear):
            raise TypeError(f"stacked LoRA target {type(owner).__name__} is not a FusedColumnLinear")
        offset, size = owner.shard_slice(shard_id)
        target = param.data.narrow(0, offset, size)
    if target.shape != delta.shape:
        raise ValueError(f"LoRA delta {tuple(delta.shape)} does not match weight rows {tuple(target.shape)}")
    target.copy_((target.float() + delta.to(target.device, torch.float32)).to(target.dtype))


def merge_lora(
    module: nn.Module,
    adapter: LoraAdapter,
    remap: Callable[[str], str],
    stacked_params: Iterable[StackedParamRule] = (),
    scale: float = 1.0,
) -> list[str]:
    """Fold ``adapter`` into ``module``'s weights; returns the native parameters it touched.

    Every adapter layer must land on a parameter: an unmatched module path is a mapping bug,
    not something to skip silently (the reference would have raised as well).
    """
    params = dict(module.named_parameters())
    rules = list(stacked_params)
    touched = []
    for module_path, layer in adapter.layers.items():
        target, shard_id = resolve_lora_module(module_path, remap, rules)
        if target not in params:
            raise KeyError(f"LoRA layer {module_path!r} maps to {target!r}, which is not a parameter of the module")
        factor = scale * layer.alpha / layer.a.shape[0]
        delta = (layer.b.float() @ layer.a.float()) * factor
        owner_path = target.rsplit(".", 1)[0]
        _merge_delta(params[target], delta, module.get_submodule(owner_path), shard_id)
        touched.append(target)
    logger.info("merged LoRA into %d parameters (scale %.3f)", len(touched), scale)
    return touched


def apply_loras(
    module: nn.Module,
    files: Iterable[LoraFile],
    remap: Callable[[str], str],
    stacked_params: Iterable[StackedParamRule] = (),
    convert_keys: KeyConverter | None = None,
) -> None:
    """Load and merge each adapter in order (later adapters see the earlier merges)."""
    for file in files:
        adapter = read_lora_file(file.path, convert_keys=convert_keys)
        merge_lora(module, adapter, remap, stacked_params, scale=file.scale)
        logger.info("LoRA %s: %d layers merged at scale %.3f", file.path, len(adapter), file.scale)
