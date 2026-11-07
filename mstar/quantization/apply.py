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

from mstar.model.components.quant_linear import LinearFormat, QuantLinear, linear_format
from mstar.model.loader.base import StackedParamRule, _StackedMatcher
from mstar.quantization.checkpoint import split_name
from mstar.quantization.resolve import EXPERTS, LINEAR, Action, QuantPlan

logger = logging.getLogger(__name__)

# checkpoint tensors a dequantized linear no longer needs (static activation scales, kv scales)
_DROPPED = ("input_scale",)
_FLOATS = (torch.bfloat16, torch.float16, torch.float32)


def format_of(layer) -> LinearFormat:
    """The format object a linear layer's plan runs: fp8_block at the checkpoint's block when it loads as stored."""
    if layer.runtime == "fp8_block" and layer.action is Action.DIRECT:
        from mstar.model.components.fp8_formats import Fp8BlockFormat

        return Fp8BlockFormat.for_block(layer.block)
    return linear_format(layer.runtime)


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
        module.quant_prepare(format_of(layer))
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

    def layer_of(base: str):
        if base not in located:
            mapped = name_remapper(base + ".weight") if name_remapper is not None else base + ".weight"
            located[base] = dequant.get(match(mapped)[0].rpartition(".")[0]) if mapped is not None else None
        return located[base]

    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        base, suffix = split_name(name)
        layer = layer_of(base) if suffix else None
        if layer is None:
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
            weight, scale = slot["weight"], slot[scale_name]
            if scale_name == "weight_scale_inv":
                fmt = _block_format(weight, scale, layer.block)
            else:
                fmt = linear_format(_per_tensor_or_channel(scale))
            yield base + ".weight", fmt.dequantize({"weight": weight, scale_name: scale}).to(dtype)
    if pending:
        raise ValueError(
            f"{plan.node}: the checkpoint stream ended with quantized tensors missing their pair "
            f"(weight or scale): {sorted((b, sorted(s)) for b, s in pending.items())[:5]}"
        )


def _block_format(weight: torch.Tensor, scale: torch.Tensor, hint: tuple[int, int] | None) -> LinearFormat:
    """The block-fp8 format of one stored pair: the layer's block, else what the shapes say (a sliced read
    hands over this rank's slice of both, which keeps the block)."""
    from mstar.model.components.fp8_formats import Fp8BlockFormat
    from mstar.quantization.checkpoint import TensorInfo, classify

    stored = classify({"weight": TensorInfo("F8_E4M3", tuple(weight.shape)),
                       "weight_scale_inv": TensorInfo("F32", tuple(scale.shape))}, "fp8", hint)
    if stored.format != "fp8_block":
        raise ValueError(f"a weight {tuple(weight.shape)} with scales {tuple(scale.shape)} is no block-fp8 pair")
    return Fp8BlockFormat.for_block(stored.block)


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
