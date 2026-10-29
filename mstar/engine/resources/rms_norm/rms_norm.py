"""Backend-neutral RMSNorm operator.

``mstar::rms_norm`` keeps fused accelerator implementations behind one graph
boundary, declared once in the kernel registry (``mstar.kernels``). GPUs run
FlashInfer, XPU runs vllm-xpu-kernels, and CPU tensors the portable PyTorch
reference.
"""

import torch

from mstar.kernels import KernelOp, ShapePolicy, Tolerance


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


def _rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    norm_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Portable RMSNorm, returned in ``x``'s original dtype."""
    x, weight, orig_dtype = _prepare_inputs(x, weight, norm_dtype)
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    output = x * torch.rsqrt(variance + eps).to(x.dtype)
    return (output * weight).to(orig_dtype)


def _rms_norm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    norm_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.empty_like(x)


def _flashinfer():
    import flashinfer

    def rms_norm(x, weight, eps, norm_dtype=None):
        x, weight, orig_dtype = _prepare_inputs(x, weight, norm_dtype)
        return flashinfer.norm.rmsnorm(x, weight, eps=eps).to(orig_dtype)

    return rms_norm


def _vllm_xpu():
    import vllm_xpu_kernels._C  # noqa: F401

    def rms_norm(x, weight, eps, norm_dtype=None):
        x, weight, orig_dtype = _prepare_inputs(x, weight, norm_dtype)
        output = torch.empty_like(x)
        torch.ops._C.rms_norm(output, x, weight, eps)
        return output.to(orig_dtype)

    return rms_norm


def _rows_hidden(x, weight, eps, norm_dtype=None) -> dict[str, int]:
    hidden = x.shape[-1]
    return {"rows": x.numel() // hidden if hidden else 0, "hidden": hidden}


def _inputs(shape, dtype, device, generator=None):
    x = torch.randn(shape["rows"], shape["hidden"], dtype=dtype, device=device, generator=generator)
    weight = torch.randn(shape["hidden"], dtype=dtype, device=device, generator=generator)
    return x * 3, weight, 1e-6, None


RMS_NORM = KernelOp(
    "rms_norm",
    "(Tensor x, Tensor weight, float eps, ScalarType? norm_dtype=None) -> Tensor",
    fake=_rms_norm_fake,
    reference=_rms_norm_reference,
    # torch calls AMD GPUs "cuda" devices, so they run FlashInfer's API too
    impls={"cuda": _flashinfer, "rocm": _flashinfer, "xpu": _vllm_xpu},
    shapes=ShapePolicy(
        dims=("rows", "hidden"),
        key=_rows_hidden,
        inputs=_inputs,
        # decode steps up to the largest decode capture bucket; prompts above
        classes=(("decode", lambda s: s["rows"] <= 128), ("prefill", lambda s: s["rows"] > 128)),
        warm=(
            {"rows": 1, "hidden": 4096}, {"rows": 64, "hidden": 4096},
            {"rows": 512, "hidden": 4096}, {"rows": 8 * 16, "hidden": 128},
        ),
        dtypes=(torch.bfloat16, torch.float16),
    ),
    # FlashInfer's numerics: fp32 math, one round-to-nearest-even store
    tolerance=Tolerance.rounded_once(),
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
