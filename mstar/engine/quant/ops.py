"""Activation quantize ops, as custom ops so ``torch.compile`` keeps one node
per quantize (step 5b pattern-matches them) and CUDA graphs can capture them."""
import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
# floor on a scale, so an all-zero row quantizes to zeros, not NaN
MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


def _quant_per_token_portable(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    # times the reciprocal, as CUDA torch divides by a scalar; the Triton kernel does the same
    scale = (xf.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)).clamp_min(MIN_SCALE)
    return (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


@torch.library.custom_op("mstar::fp8_quant_per_token", mutates_args=())
def fp8_quant_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` [M, K] as e4m3 values and an fp32 scale per row [M, 1]."""
    return _quant_per_token_portable(x)


@torch.library.register_kernel(fp8_quant_per_token, "cuda")
def _fp8_quant_per_token_cuda(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from mstar.engine.quant._triton import quant_per_token

    if x.stride(-1) != 1:
        x = x.contiguous()
    return quant_per_token(x, FP8_MAX, 1.0 / FP8_MAX, MIN_SCALE)


@fp8_quant_per_token.register_fake
def _fp8_quant_per_token_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(x.shape, dtype=FP8, device=x.device),
        torch.empty((*x.shape[:-1], 1), dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op("mstar::fp8_mm_small_m", mutates_args=())
def fp8_mm_small_m(
    a: torch.Tensor, scale_a: torch.Tensor, b: torch.Tensor, scale_b: torch.Tensor,
    bias: torch.Tensor | None, n_out: int, out_dtype: torch.dtype,
) -> torch.Tensor:
    """``(a @ b[:n_out].T) * scale_a * scale_b + bias``: fp8 ``a`` [M, K], ``b`` [N, K],
    scales [M, 1] and [1, N]; for small M, where split-K beats scaled_mm."""
    out = (a.float() @ b[:n_out].float().t()) * scale_a * scale_b[:, :n_out]
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


@torch.library.register_kernel(fp8_mm_small_m, "cuda")
def _fp8_mm_small_m_cuda(a, scale_a, b, scale_b, bias, n_out, out_dtype):
    from mstar.engine.quant._triton import fp8_mm_small_m as run

    return run(a, scale_a, b, scale_b, bias, out_dtype, n_out)


@fp8_mm_small_m.register_fake
def _fp8_mm_small_m_fake(a, scale_a, b, scale_b, bias, n_out, out_dtype):
    return torch.empty((a.shape[0], n_out), dtype=out_dtype, device=a.device)
