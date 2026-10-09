"""Fused NoPE-MLA latent glue: both latent RMSNorms and the zero-kpe cache row in one kernel."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ModuleNotFoundError:  # no triton: Glm5NextMLAAttention keeps its torch path
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _mla_latent_norm_kernel(
        fused_ptr, fused_stride, q_w_ptr, kv_w_ptr, q_out_ptr, latent_ptr,
        q_eps, kv_eps,
        Q: "tl.constexpr", L: "tl.constexpr", KPE: "tl.constexpr",
        BLOCK_Q: "tl.constexpr", BLOCK_KPE: "tl.constexpr",
    ):
        """Program ``(t, 0)`` RMS-normalizes token ``t``'s q segment ``fused[t, :Q]`` into
        ``q_out[t]``. Program ``(t, 1)`` normalizes its kv segment ``fused[t, Q:Q+L]`` into
        ``latent[t, :L]`` and zeroes ``latent[t, L:]``. fp32 math, as FlashInfer's rmsnorm.
        """
        t = tl.program_id(0)
        row = fused_ptr + t * fused_stride
        # Triton needs a name bound in both branches to have one type, so each
        # branch keeps its own (q_* is [BLOCK_Q], kv_* is [L]).
        if tl.program_id(1) == 0:
            q_offs = tl.arange(0, BLOCK_Q)
            q_mask = q_offs < Q
            q = tl.load(row + q_offs, mask=q_mask, other=0.0).to(tl.float32)
            q_rrms = tl.rsqrt((tl.sum(q * q) / Q + q_eps).to(tl.float32))
            q_w = tl.load(q_w_ptr + q_offs, mask=q_mask, other=0.0).to(tl.float32)
            q_y = q * q_rrms * q_w
            tl.store(q_out_ptr + t * Q + q_offs, q_y.to(q_out_ptr.dtype.element_ty),
                     mask=q_mask)
        else:
            kv_offs = tl.arange(0, L)
            kv = tl.load(row + Q + kv_offs).to(tl.float32)
            kv_rrms = tl.rsqrt((tl.sum(kv * kv) / L + kv_eps).to(tl.float32))
            kv_w = tl.load(kv_w_ptr + kv_offs).to(tl.float32)
            kv_y = kv * kv_rrms * kv_w
            out = latent_ptr + t * (L + KPE)
            tl.store(out + kv_offs, kv_y.to(latent_ptr.dtype.element_ty))
            pe = tl.arange(0, BLOCK_KPE)
            zeros = tl.zeros([BLOCK_KPE], dtype=latent_ptr.dtype.element_ty)
            tl.store(out + L + pe, zeros, mask=pe < KPE)

    def mla_latent_norm(
        fused: torch.Tensor,
        q_norm_weight: torch.Tensor, q_eps: float,
        kv_norm_weight: torch.Tensor, kv_eps: float,
        kpe: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``fused`` is ``[T, Q + L]`` (``q_a || kv_a`` projections). Returns
        ``q_c = rmsnorm(fused[:, :Q])`` as ``[T, Q]`` and the cache row
        ``rmsnorm(fused[:, Q:]) || zeros(kpe)`` as ``[T, L + kpe]``, both contiguous."""
        T = fused.shape[0]
        Q, L = q_norm_weight.shape[0], kv_norm_weight.shape[0]
        q_c = fused.new_empty(T, Q)
        latent = fused.new_empty(T, L + kpe)
        if T:
            _mla_latent_norm_kernel[(T, 2)](
                fused, fused.stride(0), q_norm_weight, kv_norm_weight, q_c, latent,
                q_eps, kv_eps,
                Q=Q, L=L, KPE=kpe,
                BLOCK_Q=triton.next_power_of_2(Q), BLOCK_KPE=triton.next_power_of_2(max(kpe, 1)),
            )
        return q_c, latent

else:

    def mla_latent_norm(fused, q_norm_weight, q_eps, kv_norm_weight, kv_eps, kpe):  # pragma: no cover
        raise RuntimeError("mla_latent_norm needs triton (CUDA path only)")


def mla_latent_norm_available(
    fused: torch.Tensor, q_norm_weight: torch.Tensor, kv_norm_weight: torch.Tensor,
) -> bool:
    """Fused path is usable: triton present, bf16 CUDA activations and norm weights (the
    serving dtype; the torch path casts anything else to it first), a unit-stride row, and a
    power-of-two latent width."""
    L = kv_norm_weight.shape[0]
    return (
        _HAS_TRITON and fused.is_cuda and fused.dim() == 2 and fused.stride(1) == 1
        and fused.dtype == q_norm_weight.dtype == kv_norm_weight.dtype == torch.bfloat16
        and L > 0 and (L & (L - 1)) == 0
    )
