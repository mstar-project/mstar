"""Backend-neutral RMSNorm operators.

``mstar::rms_norm`` and ``mstar::fused_add_rms_norm`` keep fused accelerator
implementations behind stable graph boundaries, declared once in the kernel
registry (``mstar.kernels``). NVIDIA GPUs run FlashInfer, AMD GPUs FlashInfer's
ROCm fork, XPU vllm-xpu-kernels, and CPU tensors the portable PyTorch
reference.
"""

import torch

from mstar.kernels import KernelOp, ShapePolicy, Tolerance
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


def _flashinfer_rocm():
    from mstar.engine.resources.rms_norm.rocm import rmsnorm

    def rms_norm(x, weight, eps, norm_dtype=None):
        x, weight, orig_dtype = _prepare_inputs(x, weight, norm_dtype)
        return rmsnorm(x, weight, eps).to(orig_dtype)

    return rms_norm


def _vllm_xpu():
    import vllm_xpu_kernels._C  # noqa: F401

    def rms_norm(x, weight, eps, norm_dtype=None):
        x, weight, orig_dtype = _prepare_inputs(x, weight, norm_dtype)
        output = torch.empty_like(x)
        torch.ops._C.rms_norm(output, x, weight, eps)
        return output.to(orig_dtype)

    return rms_norm


def _rows_hidden(x, *args) -> dict[str, int]:
    hidden = x.shape[-1]
    return {"rows": x.numel() // hidden if hidden else 0, "hidden": hidden}


def _inputs(shape, dtype, device, generator=None):
    x = torch.randn(shape["rows"], shape["hidden"], dtype=dtype, device=device, generator=generator)
    weight = torch.randn(shape["hidden"], dtype=dtype, device=device, generator=generator)
    return x * 3, weight, 1e-6, None


# decode steps up to the largest decode capture bucket; prompts above
_CLASSES = (("decode", lambda s: s["rows"] <= 128), ("prefill", lambda s: s["rows"] > 128))

RMS_NORM = KernelOp(
    "rms_norm",
    "(Tensor x, Tensor weight, float eps, ScalarType? norm_dtype=None) -> Tensor",
    fake=_rms_norm_fake,
    reference=_rms_norm_reference,
    impls={"cuda": _flashinfer, "rocm": _flashinfer_rocm, "xpu": _vllm_xpu},
    shapes=ShapePolicy(
        dims=("rows", "hidden"),
        key=_rows_hidden,
        inputs=_inputs,
        classes=_CLASSES,
        warm=(
            {"rows": 1, "hidden": 4096}, {"rows": 64, "hidden": 4096},
            {"rows": 512, "hidden": 4096}, {"rows": 8 * 16, "hidden": 128},
        ),
        dtypes=(torch.bfloat16, torch.float16),
    ),
    # FlashInfer's numerics: fp32 math, one round-to-nearest-even store
    tolerance=Tolerance.rounded_once(),
)


def _fused_add_rms_norm_reference(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
) -> None:
    """In place: ``residual += hidden_states``, then ``hidden_states =
    norm(residual) * weight`` (``* (1 + weight)`` when ``gemma``).

    FlashInfer's numerics: the sum is normalized in fp32, before the new residual
    is rounded to its dtype.
    """
    total = hidden_states.float() + residual.float()
    residual.copy_(total)
    scale = weight.float() + 1.0 if gemma else weight.float()
    hidden_states.copy_(
        total * torch.rsqrt(total.pow(2).mean(dim=-1, keepdim=True) + eps) * scale
    )


def _fused_add_rms_norm_fake(hidden_states, residual, weight, eps, gemma) -> None:
    return None


def _flashinfer_fused_add_rms_norm():
    from flashinfer.norm import fused_add_rmsnorm, gemma_fused_add_rmsnorm

    def run(hidden_states, residual, weight, eps, gemma):
        fused = gemma_fused_add_rmsnorm if gemma else fused_add_rmsnorm
        fused(hidden_states, residual, weight, eps)

    return run


def _rocm_fused_add_rms_norm():
    from mstar.engine.resources.rms_norm.rocm import fused_add_rmsnorm

    return fused_add_rmsnorm


def _tuned_fused_add_rms_norm():
    """The Triton kernel for the shape classes this device's config promotes
    it for, FlashInfer's ROCm fork for the rest and wherever no config exists,
    so a product nobody has measured keeps the fork."""
    from mstar.engine.resources.rms_norm.rocm import fused_add_rmsnorm as fork
    from mstar.kernels import configs
    from mstar.platforms import current_platform

    facts = current_platform.device_facts()
    if configs.load_config(FUSED_ADD_RMS_NORM.name, facts.arch, facts.product) is None:
        fork.kernel_selection = f"flashinfer-rocm (no config for {facts.arch}/{facts.product})"
        return fork
    from mstar.kernels.triton.norm import fused_add_rms_norm as triton_kernel

    chosen: dict[tuple[int, int], object] = {}

    def choose(rows: int, hidden: int):
        entry = FUSED_ADD_RMS_NORM.config({"rows": rows, "hidden": hidden})
        if entry is None or entry.get("impl") != "triton":
            return fork
        knobs = {k: v for k, v in entry.items() if k != "impl"}
        return lambda h, r, w, eps, gemma: triton_kernel(h, r, w, eps, gemma, knobs)

    def run(hidden_states, residual, weight, eps, gemma):
        key = (hidden_states.shape[0], hidden_states.shape[1])
        impl = chosen.get(key)
        if impl is None:
            impl = chosen[key] = choose(*key)
        impl(hidden_states, residual, weight, eps, gemma)

    run.kernel_selection = f"triton by shape class ({facts.arch}/{facts.product} config), else flashinfer-rocm"
    return run


def _fused_add_inputs(shape, dtype, device, generator=None):
    rows, hidden = shape["rows"], shape["hidden"]
    hidden_states = torch.randn(rows, hidden, dtype=dtype, device=device, generator=generator)
    residual = torch.randn(rows, hidden, dtype=dtype, device=device, generator=generator) * 4
    weight = torch.randn(hidden, dtype=dtype, device=device, generator=generator) * 0.5
    return hidden_states, residual, weight, 1e-6, False


# ``(hidden_states, residual, weight, eps, gemma)``, both 2-D, updated in place
FUSED_ADD_RMS_NORM = KernelOp(
    "fused_add_rms_norm",
    "(Tensor(a!) hidden_states, Tensor(b!) residual, Tensor weight, float eps, bool gemma) -> ()",
    fake=_fused_add_rms_norm_fake,
    reference=_fused_add_rms_norm_reference,
    # XPU tensors take the portable body, as they did before the registry
    impls={
        "cuda": _flashinfer_fused_add_rms_norm,
        "rocm": _rocm_fused_add_rms_norm,
        "rocm:gfx942": _tuned_fused_add_rms_norm,
        "rocm:gfx950": _tuned_fused_add_rms_norm,
        "xpu": lambda: _fused_add_rms_norm_reference,
    },
    shapes=ShapePolicy(
        dims=("rows", "hidden"),
        key=_rows_hidden,
        inputs=_fused_add_inputs,
        classes=_CLASSES,
        warm=({"rows": 1, "hidden": 4096}, {"rows": 64, "hidden": 4096}, {"rows": 513, "hidden": 2048}),
        dtypes=(torch.bfloat16,),
    ),
    # fp32 math, one round-to-nearest-even store per output
    tolerance=Tolerance.rounded_once(),
)


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
