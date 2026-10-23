"""Quant kernels: the GEMMs that read a format, with what each one needs."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mstar.utils.device import DeviceCapability


@dataclass(frozen=True)
class QuantKernel:
    name: str
    # activation schemes it runs (None: weight-only)
    activations: frozenset[str | None]
    # (device, output dtype) -> None if usable, else why not
    supports: Callable[[DeviceCapability, torch.dtype], str | None]
    # (N, K) -> whether it takes a weight of this shape
    shape_ok: Callable[[int, int], bool]
    # (module, x, bias) -> y in x's dtype
    run: Callable[[torch.nn.Module, torch.Tensor, torch.Tensor | None], torch.Tensor]
    # (module) -> None; lays the loaded weight out for run
    prepare: Callable[[torch.nn.Module], None] = lambda module: None
    capturable: bool = True
    # chosen without being pinned; False for reference kernels
    default: bool = True


KERNELS: dict[str, QuantKernel] = {}


def register_kernel(kernel: QuantKernel) -> QuantKernel:
    if kernel.name in KERNELS:
        raise ValueError(f"quant kernel {kernel.name!r} is already registered")
    KERNELS[kernel.name] = kernel
    return kernel


def _rows(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])


# --- scaled_mm_rowwise: fp8 x fp8 on sm90+, both scales in the epilogue; split-K Triton at small M ---

def _rowwise_supports(device: DeviceCapability, dtype: torch.dtype) -> str | None:
    cc = device.compute_capability
    if cc is None or cc < (9, 0):
        return f"needs CUDA sm90 or newer, device is {device.device} (capability {cc})"
    if dtype != torch.bfloat16:
        return f"row-wise scaled_mm writes bf16 only, the node runs {dtype}"
    return None


def _rowwise_prepare(module: torch.nn.Module) -> None:
    # scale_b must be [1, N] and contiguous, also for a per-tensor scale
    weight = module.weight
    n, k = weight.shape
    scale_b = module.weight_scale.reshape(1, -1).expand(1, n)
    pad = -n % 16
    if pad:
        # scaled_mm takes N in multiples of 16; zero rows, sliced off in run
        padded = weight.new_zeros(n + pad, k)
        padded[:n] = weight
        module.weight = torch.nn.Parameter(padded, requires_grad=False)
        scale_b = torch.cat([scale_b, scale_b.new_ones(1, pad)], dim=1)
    module.quant_out_features = n
    module.register_buffer("weight_scale_b", scale_b.contiguous(), persistent=False)


# at or below this many rows, a split-K Triton GEMM beats scaled_mm's tiles (H100, qwen3_5 shapes)
SMALL_M = 64


def _rowwise_run(module, x, bias):
    rows = _rows(x)
    n = module.quant_out_features
    padded = module.weight.shape[0] != n
    q, scale = torch.ops.mstar.fp8_quant_per_token(rows)
    if rows.shape[0] <= SMALL_M:
        out = torch.ops.mstar.fp8_mm_small_m(q, scale, module.weight, module.weight_scale_b, bias, n, x.dtype)
        return out.reshape(*x.shape[:-1], n)
    out = torch._scaled_mm(
        q, module.weight.t(), scale_a=scale, scale_b=module.weight_scale_b,
        bias=None if padded else bias, out_dtype=x.dtype,
    )
    if padded:
        # a column view, like a split of the output
        out = out[:, :n] if bias is None else out[:, :n] + bias
    return out.reshape(*x.shape[:-1], n)


register_kernel(QuantKernel(
    name="scaled_mm_rowwise",
    activations=frozenset({"dynamic_token"}),
    supports=_rowwise_supports,
    shape_ok=lambda n, k: k % 16 == 0,
    run=_rowwise_run,
    prepare=_rowwise_prepare,
))


# --- dequant: weight-only; dequantize per call, then a plain GEMM ---

def _dequant_weight(module, dtype: torch.dtype) -> torch.Tensor:
    scale = module.weight_scale
    w = module.weight.float().reshape(scale.shape[0], -1) * scale
    return w.reshape(module.weight.shape).to(dtype)


register_kernel(QuantKernel(
    name="dequant",
    activations=frozenset({None}),
    supports=lambda device, dtype: None,
    shape_ok=lambda n, k: True,
    run=lambda module, x, bias: F.linear(x, _dequant_weight(module, x.dtype), bias),
))


# --- fp8_emulated: W8A8 numerics in fp32 on any device; tests and reference only ---

def _emulated_run(module, x, bias):
    from mstar.engine.quant.ops import _quant_per_token_portable

    q, scale = _quant_per_token_portable(_rows(x))
    out = F.linear(q.float() * scale, _dequant_weight(module, torch.float32))
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype).reshape(*x.shape[:-1], out.shape[-1])


register_kernel(QuantKernel(
    name="fp8_emulated",
    activations=frozenset({"dynamic_token"}),
    supports=lambda device, dtype: None,
    shape_ok=lambda n, k: True,
    run=_emulated_run,
    default=False,
))
