"""Carrying out a ``QuantPlan`` on one rank's weights root, around its model's own weight loader.

- ``prepare`` (before the checkpoint streams in): each ``DIRECT`` layer swaps its bf16 weight for the format's
  empty storage, named as the checkpoint names it, so the model's loader (remap, stacked rules, the layer's
  own TP sharding) fills it unchanged.
- ``intercept`` (wrapping the stream): tensors of ``DIRECT`` layers pass through their format's
  ``adapt_checkpoint`` (AWQ / GPTQ repacked to the compressed-tensors layout under its names, ``weight_shape``
  dropped); tensors of ``DEQUANT`` / ``REQUANT`` layers are paired per checkpoint linear (every quantized tensor
  the checkpoint holds for it) and come out as one bf16 ``weight``, which loads as a bf16 checkpoint's would.
- ``finalize`` (after the load): every ``DIRECT`` layer's scales are checked to have arrived and handed to its
  format's ``after_load``; ``QUANTIZE`` / ``REQUANT`` layers quantize their loaded bf16 shard; the formats'
  kernels load before any capture.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator

import torch
from torch import nn

from mstar.model.components.quant_linear import LinearFormat, QuantLinear, linear_format
from mstar.model.loader.base import StackedParamRule, _StackedMatcher
from mstar.quantization.checkpoint import CheckpointQuant, split_name
from mstar.quantization.formats import BF16
from mstar.quantization.resolve import EXPERTS, LINEAR, Action, LayerPlan, QuantPlan

logger = logging.getLogger(__name__)

# checkpoint tensors a dequantized linear no longer needs (static activation scales, ct's unpadded shape)
_DROPPED = ("input_scale", "input_global_scale", "weight_shape")
# tensors of a checkpoint linear that load as they are whatever its format (its bias)
_PASSED = ("bias",)
_FLOATS = (torch.bfloat16, torch.float16, torch.float32)


def format_of(layer) -> LinearFormat:
    """The format object a linear layer's plan runs, as its checkpoint stores it when it loads as stored: fp8_block
    at the checkpoint's block, a 4-bit checkpoint's own variant (``LinearFormat.for_stored``: group, layout)."""
    if layer.action is not Action.DIRECT:
        return linear_format(layer.runtime)
    if layer.runtime == "fp8_block":
        from mstar.model.components.fp8_formats import Fp8BlockFormat

        return Fp8BlockFormat.for_block(layer.block)
    return linear_format(layer.runtime).for_stored(layer.source)


def stored_format(stored) -> LinearFormat:
    """The format object of one checkpoint linear (a ``checkpoint.StoredBase``), to dequantize it."""
    if stored.format == "fp8_block":
        from mstar.model.components.fp8_formats import Fp8BlockFormat

        return Fp8BlockFormat.for_block(stored.block)
    return linear_format(stored.format).for_stored(stored)


def _block_of(root: nn.Module, path: str) -> nn.Module:
    """The MoE block that owns ``<block>.experts``."""
    return root.get_submodule(path.rpartition(".")[0])


def _is_scale(name: str, stands_for: torch.dtype | None) -> bool:
    # data tensors can hold any bit pattern (two e2m1 codes of -6 are 0xFF); scales can't be NaN
    return "scale" in name and stands_for is not None and stands_for.is_floating_point


def _fill_unloaded(tensor: torch.Tensor, stands_for: torch.dtype | None) -> None:
    """Mark a scale not loaded yet: NaN, in whatever float it stands for (0xFF is NaN in e4m3 and e8m0)."""
    if stands_for.itemsize == 1:
        tensor.data.view(torch.uint8).fill_(0xFF)
    else:
        tensor.data.view(stands_for).fill_(float("nan"))


def _unloaded(tensor: torch.Tensor, stands_for: torch.dtype | None) -> bool:
    if stands_for.itemsize == 1:
        return bool((tensor.data.view(torch.uint8) == 0xFF).any())
    return bool(torch.isnan(tensor.data.view(stands_for)).any())


def prepare(root: nn.Module, plan: QuantPlan) -> None:
    """Give every ``DIRECT`` layer the stored format's empty storage (its scales NaN until loaded)."""
    for layer in plan.by_action(Action.DIRECT):
        if layer.kind == EXPERTS:
            block = _block_of(root, layer.path)
            block.experts_prepare(layer.runtime, layer.block or (128, 128), layer.source)
            experts = root.get_submodule(layer.path)
            for name, dtype in block.experts_scales().items():
                if _is_scale(name, dtype):
                    _fill_unloaded(getattr(experts, name), dtype)
            continue
        module = root.get_submodule(layer.path)
        module.quant_prepare(format_of(layer))
        for name, dtype in module._quant_views.items():
            if _is_scale(name, dtype):
                _fill_unloaded(getattr(module, name), dtype)
    if plan.by_action(Action.DIRECT):
        logger.info("%s: %d layers load as stored", plan.node, len(plan.by_action(Action.DIRECT)))


def intercept(
    weights: Iterable[tuple[str, torch.Tensor]],
    plan: QuantPlan,
    stacked_params: list[StackedParamRule] | None = None,
    name_remapper: Callable[[str], str | None] | None = None,
    dtype: torch.dtype = torch.bfloat16,
    ckpt: CheckpointQuant | None = None,
    prefix: str = "",
) -> Iterator[tuple[str, torch.Tensor]]:
    """``weights`` with every ``DIRECT`` layer's tensors adapted to its format's storage and the quantized tensors of
    every dequantized layer replaced by its bf16 ``weight``. A tensor's layer is found the way
    ``load_weights_into`` places it: ``name_remapper``, then the stacked rules. ``ckpt`` (the checkpoint's scan)
    says which tensors make up each checkpoint linear (its names carry ``prefix``, which the stream's don't);
    without it a linear is a ``weight`` and one fp8 scale."""
    dequant = {layer.path: layer for layer in plan.by_action(Action.DEQUANT, Action.REQUANT)}
    direct = {layer.path: layer for layer in plan.by_action(Action.DIRECT)}
    if not dequant and not direct:
        yield from weights
        return
    match = _StackedMatcher(list(stacked_params or []))
    located: dict[str, LayerPlan | None] = {}
    formats: dict[str, LinearFormat] = {}

    def layer_of(base: str) -> LayerPlan | None:
        if base not in located:
            mapped = name_remapper(base + ".weight") if name_remapper is not None else base + ".weight"
            path = None if mapped is None else match(mapped)[0].rpartition(".")[0]
            located[base] = dequant.get(path) or direct.get(path)
        return located[base]

    def cached_format(layer: LayerPlan) -> LinearFormat:
        if layer.path not in formats:
            formats[layer.path] = format_of(layer)
        return formats[layer.path]

    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        base, suffix = split_name(name)
        layer = layer_of(base) if suffix else None
        if layer is None:
            yield name, tensor
            continue
        if layer.action is Action.DIRECT:
            adapted = cached_format(layer).adapt_checkpoint(suffix, tensor)
            if adapted is None:
                continue
            suffix, tensor = adapted
            if layer.kind == EXPERTS and suffix == "input_scale":
                # an expert's activation scale follows its weight's name onto the stacked experts params
                suffix = "weight_input_scale"
            yield f"{base}.{suffix}", tensor
            continue
        if suffix in _DROPPED:
            continue
        stored = ckpt.bases.get(prefix + base) if ckpt is not None else None
        if stored is not None:
            # the checkpoint's own account of this linear: every quantized tensor it holds for it
            if stored.format == BF16 or suffix in _PASSED:
                yield name, tensor
                continue
            expected = set(ckpt.suffixes[prefix + base]) - set(_DROPPED) - set(_PASSED)
            slot = pending.setdefault(base, {})
            slot[suffix] = tensor
            if set(slot) == expected:
                del pending[base]
                fmt = stored_format(stored)
                buffers = {}
                for s, t in slot.items():
                    adapted = fmt.adapt_checkpoint(s, t)
                    if adapted is not None:
                        buffers[adapted[0]] = fmt.from_checkpoint(*adapted)
                yield base + ".weight", fmt.dequantize(buffers).to(dtype)
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
    """Check the ``DIRECT`` layers' scales all arrived and finish their layout, quantize the layers loaded bf16,
    load the kernels."""
    formats = set()
    for layer in plan.by_action(Action.DIRECT):
        if layer.kind == EXPERTS:
            block = _block_of(root, layer.path)
            experts = root.get_submodule(layer.path)
            missing = [n for n, d in block.experts_scales().items()
                       if _is_scale(n, d) and _unloaded(getattr(experts, n), d)]
        else:
            module = root.get_submodule(layer.path)
            missing = [n for n, d in module._quant_views.items()
                       if _is_scale(n, d) and _unloaded(getattr(module, n), d)]
        if missing:
            raise ValueError(f"{plan.node}: {layer.path!r} is stored {layer.stored}, but some of its scales never "
                             f"arrived from the checkpoint ({missing})")
        if layer.kind == EXPERTS:
            block.experts_after_load()
        else:
            module.quant_after_load()
            formats.add(module.quant_format)
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


__all__ = ["finalize", "format_of", "intercept", "prepare", "stored_format", "LINEAR", "EXPERTS"]
