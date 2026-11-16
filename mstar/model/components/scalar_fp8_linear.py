"""Scalar-scaled E4M3 weights with checkpoint-supplied static activation scales.

Unlike block FP8, each projection owns one weight scale and one activation
scale. Keeping projections separate preserves different Q/K/V or gate/up scales.
No checkpoint weight is expanded to BF16: the matrix multiplication consumes two
FP8 operands, accumulates with ``use_fast_accum=False``, and produces BF16.
"""
from __future__ import annotations

from functools import partial

import torch
from torch import nn

from mstar.platforms import current_platform

FP8_DTYPE = torch.float8_e4m3fn


def quantize_static_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Divide by the supplied scale and round to saturating E4M3.

    Division uses FP32 and overflow saturates to finite E4M3 values. Torch's
    FP8 cast alone does not saturate.
    """
    limit = torch.finfo(FP8_DTYPE).max
    return (x.float() / scale.float()).clamp(-limit, limit).to(FP8_DTYPE)


def _scaled_fp8_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    activation_scale: torch.Tensor,
) -> torch.Tensor:
    """Tensor arithmetic shared by the GPU op and operand-contract tests."""
    quantized = quantize_static_fp8(x, activation_scale)
    # Despite its checkpoint name, weight_scale_inv is a dequantization
    # multiplier, just like activation_scale. _scaled_mm requires FP32 scales.
    return torch._scaled_mm(
        quantized.reshape(-1, weight.shape[1]).contiguous(),
        weight.view(FP8_DTYPE).t(),
        scale_a=activation_scale.float(),
        scale_b=weight_scale_inv.float(),
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    ).reshape(*x.shape[:-1], weight.shape[0])


def require_scalar_fp8_device(device: str | torch.device) -> None:
    """Reject unsupported execution targets before allocating checkpoint storage."""
    device = torch.device(device)
    if device.type != "cuda" or not current_platform.has_cuda_capability(8, 9, device=device):
        raise RuntimeError("scalar FP8 linear requires NVIDIA SM89+ and FP8 _scaled_mm support")


@torch.library.custom_op("mstar::scalar_fp8_linear", mutates_args=())
def scalar_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    activation_scale: torch.Tensor,
) -> torch.Tensor:
    """Native FP8 matmul; unsupported backends fail instead of dequantizing."""
    _check_inputs(x, weight, weight_scale_inv, activation_scale)
    require_scalar_fp8_device(x.device)
    return _scaled_fp8_mm(x, weight, weight_scale_inv, activation_scale)


def _check_inputs(x, weight, weight_scale_inv, activation_scale):
    if x.dtype != torch.bfloat16:
        raise TypeError("scalar FP8 linear expects BF16 activations")
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise TypeError("scalar FP8 weight must be a 2D uint8 container of E4M3 bytes")
    if x.ndim < 1 or x.shape[-1] != weight.shape[1]:
        raise ValueError("activation width does not match scalar FP8 weight")
    if any(size % 16 for size in weight.shape):
        raise ValueError("scalar FP8 matrix dimensions must be multiples of 16")
    for scale in (weight_scale_inv, activation_scale):
        if scale.ndim != 0 or scale.dtype != torch.float32:
            raise TypeError("scalar FP8 scales must be FP32 scalar tensors")
    if any(t.device != x.device for t in (weight, weight_scale_inv, activation_scale)):
        raise ValueError("scalar FP8 inputs, weights and scales must share a device")


@scalar_fp8_linear.register_fake
def _scalar_fp8_linear_fake(x, weight, weight_scale_inv, activation_scale):
    _check_inputs(x, weight, weight_scale_inv, activation_scale)
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


class ScalarFp8Linear(nn.Module):
    """An unfused, unsharded, bias-free projection with two scalar scales.

    ``weight_loader`` hooks consume original E4M3 checkpoint tensors and positive
    scalar scales. Scales are promoted losslessly from BF16 to FP32. The engine's
    later ``module.to(dtype=...)`` preserves both FP8 bytes and FP32 scale values.
    As with other linear components, the enclosing loader must require all three
    parameters; this module does not synthesize missing checkpoint scales.
    """

    def __init__(self, in_features: int, out_features: int, *, device=None):
        super().__init__()
        if any(size <= 0 or size % 16 for size in (in_features, out_features)):
            raise ValueError("scalar FP8 matrix dimensions must be positive multiples of 16")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        for name in ("weight_scale_inv", "activation_scale"):
            self.register_parameter(name, nn.Parameter(
                torch.empty((), dtype=torch.float32, device=device), requires_grad=False,
            ))
        self._attach_weight_loaders()

    def _attach_weight_loaders(self):
        self.weight.weight_loader = self._load_weight
        for name in ("weight_scale_inv", "activation_scale"):
            getattr(self, name).weight_loader = partial(self._load_scale, name)

    def _apply(self, fn, recurse=True):
        # Integer views pass through Module.to without floating-point conversion.
        # Applying fn to them also supports meta -> to_empty materialization.
        scales = {
            name: getattr(self, name).detach().view(torch.int32)
            for name in ("weight_scale_inv", "activation_scale")
        }
        result = super()._apply(fn, recurse=recurse)
        for name, bits in scales.items():
            getattr(self, name).data = fn(bits).view(torch.float32)
        self._attach_weight_loaders()
        return result

    def _load_weight(self, param: nn.Parameter, loaded: torch.Tensor):
        if loaded.dtype != FP8_DTYPE:
            raise TypeError("scalar FP8 loader requires original E4M3 checkpoint weights")
        if loaded.shape != param.shape:
            raise ValueError(f"checkpoint weight {tuple(loaded.shape)} != {tuple(param.shape)}")
        param.data.copy_(loaded.contiguous().view(torch.uint8))

    def _load_scale(self, name: str, param: nn.Parameter, loaded: torch.Tensor):
        if loaded.ndim != 0 or loaded.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"{name} must be a BF16 or FP32 checkpoint scalar")
        # Loading is outside the tensor forward/capture path; validate once here.
        if not torch.isfinite(loaded).item() or loaded.item() <= 0:
            raise ValueError(f"{name} must be finite and positive")
        param.data.copy_(loaded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return scalar_fp8_linear(x, self.weight, self.weight_scale_inv, self.activation_scale)


class ScalarFp8MergedLinear(nn.Module):
    """Separate FP8 matmuls concatenated in declaration order.

    This has the output layout of a merged dense projection, but each constituent
    retains its own checkpoint weight and activation scale. It does not fuse
    quantization or GEMMs and does not shard across ranks.
    """

    def __init__(self, in_features: int, outputs: dict[str, int], *, device=None):
        super().__init__()
        if not outputs:
            raise ValueError("scalar FP8 merged projection requires at least one output")
        self.projections = nn.ModuleDict({
            name: ScalarFp8Linear(in_features, size, device=device)
            for name, size in outputs.items()
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([projection(x) for projection in self.projections.values()], dim=-1)
