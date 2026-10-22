"""Backend-neutral RMSNorm custom operator.

The operator keeps fused accelerator implementations behind one stable
``mstar::rms_norm`` graph boundary. CUDA currently dispatches to FlashInfer,
XPU dispatches to vllm-xpu-kernels, and other devices use the portable
PyTorch implementation. FlashInfer takes fp16 and bf16 only, so an fp32 input
on CUDA runs the portable path too.
"""

import torch

_FUSED_DTYPES = (torch.float16, torch.bfloat16)


def _prepare_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    """Apply the dtype contract shared by every RMSNorm kernel: ``norm_dtype``
    if given, else ``x``'s own; never ambient autocast state."""
    orig_dtype = x.dtype
    if norm_dtype is not None:
        x = x.to(norm_dtype)
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
    return _rms_norm_portable(x, weight, eps).to(orig_dtype)


def _rms_norm_portable(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    output = x * torch.rsqrt(variance + eps).to(x.dtype)
    return output * weight


@torch.library.register_kernel(rms_norm, "cuda")
def _rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    norm_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    x, weight, orig_dtype = _prepare_inputs(x, weight, norm_dtype)
    if x.dtype not in _FUSED_DTYPES:
        return _rms_norm_portable(x, weight, eps).to(orig_dtype)
    import flashinfer

    return flashinfer.norm.rmsnorm(x, weight, eps=eps).to(orig_dtype)


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
