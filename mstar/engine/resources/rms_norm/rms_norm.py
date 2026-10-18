"""Backend-neutral RMSNorm custom operators.

The operators keep fused accelerator implementations behind stable
``mstar::rms_norm`` and ``mstar::fused_add_rms_norm`` graph boundaries. NVIDIA
GPUs run FlashInfer, AMD GPUs FlashInfer's ROCm fork, XPU vllm-xpu-kernels, and
other devices the portable PyTorch implementation.
"""

import torch

from mstar.platforms.dispatch import PlatformDispatch


def _prepare_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    """Apply the dtype contract shared by every RMSNorm kernel."""
    orig_dtype = x.dtype
    device_type = x.device.type
    if norm_dtype is not None:
        x = x.to(norm_dtype)
    elif torch.is_autocast_enabled(device_type):
        x = x.to(torch.get_autocast_dtype(device_type))
    elif x.dtype == torch.float32 and device_type == "cuda":
        x = x.to(torch.bfloat16)
    if weight.dtype != x.dtype:
        weight = weight.to(x.dtype)
    return x, weight, orig_dtype


@torch.library.custom_op("mstar::rms_norm", mutates_args=())
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    norm_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Portable RMSNorm fallback, returned in ``x``'s original dtype."""
    x, weight, orig_dtype = _prepare_inputs(x, weight, norm_dtype)
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    output = x * torch.rsqrt(variance + eps).to(x.dtype)
    return (output * weight).to(orig_dtype)


def _flashinfer_rmsnorm():
    import flashinfer

    return flashinfer.norm.rmsnorm


def _rocm_rmsnorm():
    from mstar.engine.resources.rms_norm.rocm import rmsnorm

    return rmsnorm


# torch calls AMD GPUs "cuda" devices too, so the kernel picks by platform
_rmsnorm = PlatformDispatch("mstar::rms_norm", cuda=_flashinfer_rmsnorm, rocm=_rocm_rmsnorm)


@torch.library.register_kernel(rms_norm, "cuda")
def _rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    norm_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    x, weight, orig_dtype = _prepare_inputs(x, weight, norm_dtype)
    return _rmsnorm(x, weight, eps=eps).to(orig_dtype)


@torch.library.register_kernel(rms_norm, "xpu")
def _rms_norm_xpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    norm_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    import vllm_xpu_kernels._C  # noqa: F401

    x, weight, orig_dtype = _prepare_inputs(x, weight, norm_dtype)
    output = torch.empty_like(x)
    torch.ops._C.rms_norm(output, x, weight, eps)
    return output.to(orig_dtype)


@rms_norm.register_fake
def _rms_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    norm_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "mstar::fused_add_rms_norm", mutates_args=("hidden_states", "residual"),
)
def fused_add_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
) -> None:
    """In place: ``residual += hidden_states``, then ``hidden_states =
    norm(residual) * weight`` (``* (1 + weight)`` when ``gemma``).

    FlashInfer's numerics: the sum is normalized in fp32, before the new residual
    is rounded to its dtype. This body is the portable fallback.
    """
    total = hidden_states.float() + residual.float()
    residual.copy_(total)
    scale = weight.float() + 1.0 if gemma else weight.float()
    hidden_states.copy_(
        total * torch.rsqrt(total.pow(2).mean(dim=-1, keepdim=True) + eps) * scale
    )


def _flashinfer_fused_add_rms_norm():
    from flashinfer.norm import fused_add_rmsnorm, gemma_fused_add_rmsnorm

    def run(hidden_states, residual, weight, eps, gemma):
        fused = gemma_fused_add_rmsnorm if gemma else fused_add_rmsnorm
        fused(hidden_states, residual, weight, eps)

    return run


def _rocm_fused_add_rms_norm():
    from mstar.engine.resources.rms_norm.rocm import fused_add_rmsnorm

    return fused_add_rmsnorm


_fused_add_rmsnorm = PlatformDispatch(
    "mstar::fused_add_rms_norm",
    cuda=_flashinfer_fused_add_rms_norm,
    rocm=_rocm_fused_add_rms_norm,
)


@torch.library.register_kernel(fused_add_rms_norm, "cuda")
def _fused_add_rms_norm_cuda(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
) -> None:
    _fused_add_rmsnorm(hidden_states, residual, weight, eps, gemma)


@fused_add_rms_norm.register_fake
def _fused_add_rms_norm_fake(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
) -> None:
    return None


def _fused_add_rms_norm_op():
    return torch.ops.mstar.fused_add_rms_norm


# ``(hidden_states, residual, weight, eps, gemma)``, both 2-D, updated in place.
# NVIDIA GPUs keep calling FlashInfer directly (a graph break per layer under
# dynamo); they can move onto the op once an NVIDIA worker_phases A/B shows it
# neutral or better.
run_fused_add_rms_norm = PlatformDispatch(
    "fused_add_rms_norm",
    cuda=_flashinfer_fused_add_rms_norm,
    rocm=_fused_add_rms_norm_op,
)


def run_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    rms_norm_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply the backend-neutral ``mstar::rms_norm`` custom operator."""
    return torch.ops.mstar.rms_norm(
        input,
        weight,
        eps,
        rms_norm_dtype,
    )
