"""fp8 linears: e4m3 weights with one scale per output channel, activations
quantised per token on the fly, the product on the platform's fp8 GEMM
(``mstar::fp8_rowwise_mm``).

A model opts in per deployment config (``linear_dtype: fp8``) for the linears
it declares eligible, as fnmatch patterns over module names: the GEMMs over
the token stream, never the small modulation, embedding or output
projections. ``apply_linear_dtype`` swaps those modules once at load (after
any LoRA merge, before compile), so the compiled forward sees the per-token
quantisation as plain torch and fuses it into the producer of the
activations, usually the modulated norm, with no extra pass over the hidden
state.

The output moves from the bf16 model's within the error of e4m3 rounding
(3 mantissa bits). Whether that is acceptable is a per-model quality call,
which is why it is opt-in.
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Sequence

import torch
from torch import nn

from mstar.kernels.fp8_gemm import FP8_ROWWISE_MM, quantize_rows
from mstar.model.components.linear import FusedColumnLinear
from mstar.platforms import current_platform

logger = logging.getLogger(__name__)

LINEAR_DTYPES = ("bf16", "fp8")
# hipBLASLt's fp8 kernels need both GEMM dimensions of the weight in multiples of 16
_ALIGN = 16

# An uncompiled forward (a Wan2.2 size outside compile_dit_sizes, say) would otherwise quantise in
# five eager kernels per linear, which cost more than the fp8 GEMM saves; this runs it as one.
_quantize_rows_fused = torch.compile(quantize_rows, dynamic=True)


def _quantize_tokens(x: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.compiler.is_compiling() or x.device.type == "cpu":
        # traced into the caller's graph, where it fuses with whatever produced x
        return quantize_rows(x, fp8_dtype)
    return _quantize_rows_fused(x, fp8_dtype)


class Fp8Linear(nn.Module):
    """``x @ W.T (+ b)`` with ``W`` stored as e4m3 rows times fp32 per-row scales.

    The output has the input's dtype; the bias stays in it too."""

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight)
        # [1, out_features]: the layout the GEMM takes for the weight's scales
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("bias", bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear | FusedColumnLinear, fp8_dtype: torch.dtype) -> Fp8Linear:
        weight, scale = quantize_rows(linear.weight.detach(), fp8_dtype)
        bias = None if linear.bias is None else linear.bias.detach().clone()
        return cls(weight, scale.T.contiguous(), bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq, x_scale = _quantize_tokens(x.reshape(-1, self.in_features), self.weight.dtype)
        out = FP8_ROWWISE_MM(xq, x_scale, self.weight, self.weight_scale, self.bias, x.dtype)
        return out.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, dtype={self.weight.dtype}"


def _fits(linear: nn.Module) -> bool:
    out_features, in_features = linear.weight.shape
    return out_features % _ALIGN == 0 and in_features % _ALIGN == 0


def quantize_linears(module: nn.Module, patterns: Sequence[str], fp8_dtype: torch.dtype) -> list[str]:
    """Replace every ``nn.Linear`` / ``FusedColumnLinear`` under ``module`` whose name matches
    one of ``patterns`` with an ``Fp8Linear``; returns the names replaced. A match whose
    dimensions the GEMM cannot take keeps its weights."""
    replaced = []
    for name, child in list(module.named_modules()):
        if not isinstance(child, (nn.Linear, FusedColumnLinear)):
            continue
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            continue
        if not _fits(child):
            logger.info("%s stays %s: %s is not a multiple of %d", name, child.weight.dtype,
                        tuple(child.weight.shape), _ALIGN)
            continue
        parent, _, attr = name.rpartition(".")
        setattr(module.get_submodule(parent), attr, Fp8Linear.from_linear(child, fp8_dtype))
        replaced.append(name)
    return replaced


def fp8_unavailable_reason() -> str | None:
    """None when this process can run fp8 linears; otherwise why not. Resolves the
    GEMM's implementation, so call it at load, never inside a capture."""
    if current_platform.fp8_dtype() is None:
        arch = current_platform.arch()
        return f"{current_platform.name}{f' {arch}' if arch else ''} has no fp8 GEMM"
    return FP8_ROWWISE_MM.unavailable_reason()


def check_linear_dtype(linear_dtype: str) -> str:
    """A model config's ``linear_dtype``, checked when the model is constructed."""
    if linear_dtype not in LINEAR_DTYPES:
        raise ValueError(f"linear_dtype {linear_dtype!r} is not one of {LINEAR_DTYPES}")
    return linear_dtype


def apply_linear_dtype(module: nn.Module, linear_dtype: str, patterns: Sequence[str], what: str) -> list[str]:
    """Serve ``module``'s linears matching ``patterns`` in ``linear_dtype`` (a model config's
    ``linear_dtype``). ``bf16`` keeps the checkpoint's weights. ``fp8`` quantises them where
    the platform has an fp8 GEMM and keeps them otherwise, with a warning, so one config
    serves on every GPU. Returns the names quantised."""
    if check_linear_dtype(linear_dtype) == "bf16":
        return []
    reason = fp8_unavailable_reason()
    if reason is not None:
        logger.warning("%s: linear_dtype fp8 requested, but %s; its linears stay bf16", what, reason)
        return []
    replaced = quantize_linears(module, patterns, current_platform.fp8_dtype())
    if not replaced:
        raise ValueError(f"{what}: linear_dtype fp8 matched no linear (patterns {list(patterns)})")
    logger.info("%s: %d linears run fp8 (%s)", what, len(replaced), current_platform.fp8_dtype())
    return replaced
