"""Carrying out a ``QuantPlan`` on one rank's weights root, around its model's own weight loader.

- ``prepare`` (before the checkpoint streams in): each ``DIRECT`` layer swaps its bf16 weight for the format's
  empty storage, named as the checkpoint names it, so the model's loader (remap, stacked rules, the layer's
  own TP sharding) fills it unchanged.
- ``intercept`` (wrapping the stream): tensors of ``DEQUANT`` / ``REQUANT`` layers are paired per checkpoint
  linear (``weight`` with its scale) and come out as one bf16 ``weight``, which loads as a bf16 checkpoint's
  would.
- ``finalize`` (after the load): ``QUANTIZE`` / ``REQUANT`` layers quantize their loaded bf16 shard; every
  ``DIRECT`` layer's scales are checked to have arrived; the formats' kernels load before any capture.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator

import torch
from torch import nn

from mstar.model.components.quant_linear import QuantLinear, linear_format
from mstar.model.loader.base import StackedParamRule, _StackedMatcher
from mstar.quantization.checkpoint import split_name
from mstar.quantization.resolve import EXPERTS, LINEAR, Action, QuantPlan

logger = logging.getLogger(__name__)

# checkpoint tensors a dequantized linear no longer needs (static activation scales, kv scales)
_DROPPED = ("input_scale",)
_FLOATS = (torch.bfloat16, torch.float16, torch.float32)


def _block_of(root: nn.Module, path: str) -> nn.Module:
    """The MoE block that owns ``<block>.experts``."""
    return root.get_submodule(path.rpartition(".")[0])


def prepare(root: nn.Module, plan: QuantPlan) -> None:
    """Give every ``DIRECT`` layer the stored format's empty storage (its scales NaN until loaded)."""
    for layer in plan.by_action(Action.DIRECT):
        if layer.kind == EXPERTS:
            _block_of(root, layer.path).experts_prepare(layer.runtime, layer.block or (128, 128))
            experts = root.get_submodule(layer.path)
            for name in ("gate_up_proj_scale_inv", "down_proj_scale_inv"):
                getattr(experts, name).data.view(torch.float32).fill_(float("nan"))
            continue
        module = root.get_submodule(layer.path)
        fmt = linear_format(layer.runtime)
        module.quant_prepare(fmt)
        for name, dtype in module._quant_views.items():
            if dtype == torch.float32:
                getattr(module, name).data.view(torch.float32).fill_(float("nan"))
    if plan.by_action(Action.DIRECT):
        logger.info("%s: %d layers load as stored", plan.node, len(plan.by_action(Action.DIRECT)))


def intercept(
    weights: Iterable[tuple[str, torch.Tensor]],
    plan: QuantPlan,
    stacked_params: list[StackedParamRule] | None = None,
    name_remapper: Callable[[str], str | None] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Iterator[tuple[str, torch.Tensor]]:
    """``weights`` with the quantized tensors of every dequantized layer replaced by its bf16 ``weight``. A
    tensor's layer is found the way ``load_weights_into`` places it: ``name_remapper``, then the stacked rules."""
    dequant = {layer.path: layer for layer in plan.by_action(Action.DEQUANT, Action.REQUANT)}
    if not dequant:
        yield from weights
        return
    match = _StackedMatcher(list(stacked_params or []))
    located: dict[str, bool] = {}

    def is_dequantized(base: str) -> bool:
        if base not in located:
            mapped = name_remapper(base + ".weight") if name_remapper is not None else base + ".weight"
            located[base] = mapped is not None and match(mapped)[0].rpartition(".")[0] in dequant
        return located[base]

    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        base, suffix = split_name(name)
        if not suffix or not is_dequantized(base):
            yield name, tensor
            continue
        if suffix in _DROPPED:
            continue
        if suffix not in ("weight", "weight_scale_inv", "weight_scale") or (
            suffix == "weight" and tensor.dtype in _FLOATS
        ):
            # a bias, or a shard the checkpoint keeps bf16 (a mixed layer): as it is
            yield name, tensor
            continue
        slot = pending.setdefault(base, {})
        slot[suffix] = tensor
        if "weight" in slot and len(slot) == 2:
            del pending[base]
            (scale_name,) = (s for s in slot if s != "weight")
            stored = "fp8_block" if scale_name == "weight_scale_inv" else _per_tensor_or_channel(slot[scale_name])
            buffers = {"weight": slot["weight"], scale_name: slot[scale_name]}
            yield base + ".weight", linear_format(stored).dequantize(buffers).to(dtype)
    if pending:
        raise ValueError(
            f"{plan.node}: the checkpoint stream ended with quantized tensors missing their pair "
            f"(weight or scale): {sorted((b, sorted(s)) for b, s in pending.items())[:5]}"
        )


def _per_tensor_or_channel(scale: torch.Tensor) -> str:
    return "fp8_tensor" if scale.numel() == 1 else "fp8_channel"


def finalize(root: nn.Module, plan: QuantPlan, device: torch.device | str | None = None) -> None:
    """Quantize the layers loaded bf16, check the ``DIRECT`` layers' scales all arrived, load the kernels."""
    formats = set()
    for layer in plan.by_action(Action.DIRECT):
        tensors = []
        if layer.kind == EXPERTS:
            experts = root.get_submodule(layer.path)
            tensors = [getattr(experts, n) for n in ("gate_up_proj_scale_inv", "down_proj_scale_inv")]
        else:
            module = root.get_submodule(layer.path)
            tensors = [getattr(module, n) for n, d in module._quant_views.items() if d == torch.float32]
            formats.add(module.quant_format)
        for tensor in tensors:
            if torch.isnan(tensor.data.view(torch.float32)).any():
                raise ValueError(f"{plan.node}: {layer.path!r} is stored {layer.stored}, but some of its scales "
                                 "never arrived from the checkpoint")
    for layer in plan.by_action(Action.QUANTIZE, Action.REQUANT):
        if layer.kind == EXPERTS:
            _block_of(root, layer.path).experts_quantize(layer.runtime)
            continue
        fmt = linear_format(layer.runtime)
        formats.add(fmt)
        module = root.get_submodule(layer.path)
        if hasattr(module, "quant_quantize"):
            module.quant_quantize(fmt)
        else:
            parent_name, _, leaf = layer.path.rpartition(".")
            setattr(root.get_submodule(parent_name), leaf, QuantLinear.from_linear(module, fmt))
    if device is not None:
        for fmt in formats:
            fmt.prepare_device(torch.device(device))
    if plan.active:
        logger.info("%s: quantization %s", plan.node, plan.summary())


__all__ = ["finalize", "intercept", "prepare", "LINEAR", "EXPERTS"]
