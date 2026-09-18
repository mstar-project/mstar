"""Fused Sinkhorn-Knopp kernel for the mHC comb matrix (M3 Phase 2)."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ModuleNotFoundError:  # pure-math files import on CPU-only boxes (ground rule 4)
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _sinkhorn_kernel(
        m_ptr, num_iters: "tl.constexpr", H: "tl.constexpr", eps,
    ):
        """One program per ``[H, H]`` matrix; the whole matrix stays in registers for all
        iterations.
        """
        pid = tl.program_id(0)
        rows = tl.arange(0, H)
        cols = tl.arange(0, H)
        offs = pid * H * H + rows[:, None] * H + cols[None, :]
        m = tl.load(m_ptr + offs).to(tl.float32)  # [H, H]

        # Keep every intermediate fp32. ``eps`` is a python float (fp64 in
        # triton), so ``+ eps`` promotes the divide to fp64; ``.to(tl.float32)``
        # after each step both matches the reference's fp32 arithmetic and keeps
        # the loop-carried reduction fp32 (triton rejects an fp32->fp64 loop var
        # — the bug the serve surfaced that the isolated launch did not).
        csum = tl.sum(m, axis=0)               # [H], per-column sum over rows
        m = (m / (csum[None, :] + eps)).to(tl.float32)
        for _ in range(num_iters - 1):
            rsum = tl.sum(m, axis=1)           # [H], per-row sum over cols
            m = (m / (rsum[:, None] + eps)).to(tl.float32)
            csum = tl.sum(m, axis=0)
            m = (m / (csum[None, :] + eps)).to(tl.float32)

        tl.store(m_ptr + offs, m)

    def sinkhorn_normalize_fused(
        matrix: torch.Tensor, num_iters: int, eps: float,
    ) -> torch.Tensor:
        """CUDA fused Sinkhorn: one kernel launch replaces the ~78 of the reference loop.
        ``matrix`` is ``[..., H, H]`` fp32 on CUDA; returns a new tensor (functional,
        like the reference).
        """
        H = matrix.shape[-1]
        out = matrix.reshape(-1, H, H).contiguous().clone()
        n = out.shape[0]
        if n:
            _sinkhorn_kernel[(n,)](out, num_iters, H, eps)
        return out.reshape(matrix.shape)

else:

    def sinkhorn_normalize_fused(matrix, num_iters, eps):  # pragma: no cover
        raise RuntimeError("sinkhorn_normalize_fused needs triton (CUDA path only)")


def fused_sinkhorn_available(matrix: torch.Tensor) -> bool:
    """Fused path is usable: triton present, CUDA tensor, square power-of-two H."""
    if not (_HAS_TRITON and matrix.is_cuda):
        return False
    H = matrix.shape[-1]
    return matrix.shape[-2] == H and H > 0 and (H & (H - 1)) == 0
