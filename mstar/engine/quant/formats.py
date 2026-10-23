"""Quant formats: what a quantized weight stores, and the kernels that read it.

A format binds a layout to kernels, so adding one is a reviewed change. Online
scales come from the local TP shard, so they shard with it by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mstar.engine.quant.ops import FP8
from mstar.engine.quant.scale_rules import get_scale_rule, qmax


@dataclass(frozen=True)
class QuantFormat:
    name: str
    value_dtype: torch.dtype
    # "channel": one scale per output row; "tensor": one for the weight
    granularity: str
    # candidate kernels, in preference order
    kernels: tuple[str, ...]

    def blocks(self, weight: torch.Tensor) -> torch.Tensor:
        """``weight`` [N, K] as fp32 [G, B], one row per scale."""
        w = weight.float()
        return w if self.granularity == "channel" else w.reshape(1, -1)

    def quantize(self, weight: torch.Tensor, rule: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``weight`` [N, K] as values [N, K] and fp32 scales [G, 1]."""
        blocks = self.blocks(weight)
        scale = get_scale_rule(rule)(blocks, self.value_dtype)
        bound = qmax(self.value_dtype)
        values = (blocks / scale.unsqueeze(-1)).clamp(-bound, bound).to(self.value_dtype)
        return values.reshape(weight.shape), scale.unsqueeze(-1)

    def dequantize(self, values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """fp32 weight [N, K] from ``quantize``'s output."""
        blocks = values.float().reshape(scale.shape[0], -1)
        return (blocks * scale).reshape(values.shape)


FORMATS: dict[str, QuantFormat] = {}


def register_format(fmt: QuantFormat) -> QuantFormat:
    if fmt.name in FORMATS:
        raise ValueError(f"quant format {fmt.name!r} is already registered")
    FORMATS[fmt.name] = fmt
    return fmt


_FP8_KERNELS = ("scaled_mm_rowwise", "dequant", "fp8_emulated")
register_format(QuantFormat("fp8_channel", FP8, "channel", _FP8_KERNELS))
register_format(QuantFormat("fp8_tensor", FP8, "tensor", _FP8_KERNELS))
