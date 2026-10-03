"""Per-token-group FP8 activation quantization.

The e4m3 activation quant that feeds block-scale W8A8 GEMMs (the fused MoE
experts today).  Adapted from sglang's ``fused_moe_triton_kernels.py``
(Apache-2.0).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# e4m3; declared here so utils/ keeps no dependency on model/.
FP8_DTYPE = torch.float8_e4m3fn

# Groups quantized per program.  One program per group (sglang's shape) is
# 393k programs for an 8k x 6144 prefill; 16 matches vLLM's CUDA quantizer.
GROUPS_PER_PROGRAM = 16


@triton.jit
def per_token_group_quant_fp8_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    num_groups,
    eps,
    fp8_min,
    fp8_max,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """Quantize ``GROUPS`` contiguous ``group_size`` slices to e4m3, one fp32 scale each.

    Groups tile the rows of a contiguous 2-D tensor: group ``g`` is
    ``y.view(-1)[g*group_size:(g+1)*group_size]`` and writes scale slot ``g``
    of the row-major ``(M, K // group_size)`` scales.  Program ``p`` takes
    groups ``[p*GROUPS, (p+1)*GROUPS)``; the tail past ``num_groups`` is masked.
    """
    p = tl.program_id(0).to(tl.int64)
    groups = p * GROUPS + tl.arange(0, GROUPS)
    cols = tl.arange(0, BLOCK)
    g_mask = groups < num_groups
    mask = g_mask[:, None] & (cols < group_size)[None, :]
    offs = groups[:, None] * group_size + cols[None, :]

    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # amax / e4m3-max with an eps floor so all-zero groups get a finite scale.
    y_s = tl.maximum(tl.max(tl.abs(y), axis=1), eps) / fp8_max
    y_q = tl.minimum(tl.maximum(y / y_s[:, None], fp8_min), fp8_max).to(y_q_ptr.dtype.element_ty)
    tl.store(y_q_ptr + offs, y_q, mask=mask)
    tl.store(y_s_ptr + groups, y_s, mask=g_mask)


@torch.compiler.disable
def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to e4m3, one fp32 scale per ``group_size`` K-slice per row.

    Returns ``(x_q, x_s)`` with ``x_q`` e4m3 of ``x.shape`` and ``x_s`` fp32 of
    shape ``(M, K // group_size)``; dequant is ``x_q * x_s`` (the same
    multiply-back convention as the checkpoint's ``weight_scale_inv``).

    ``torch.compiler.disable``: Inductor's (re)compile of this kernel fails in
    Triton's make_llir ("PassManager::run failed") while the kernel's own JIT
    path is fine.  The graph break keeps Inductor out of the launch while
    stream capture still records it.  Remove when the toolchain bug is fixed.
    """
    assert x.dim() == 2 and x.is_contiguous()
    assert x.shape[-1] % group_size == 0, (
        f"last dim {x.shape[-1]} must be a multiple of group_size {group_size}"
    )

    M, K = x.shape
    finfo = torch.finfo(FP8_DTYPE)
    x_q = torch.empty_like(x, dtype=FP8_DTYPE)
    x_s = torch.empty((M, K // group_size), dtype=torch.float32, device=x.device)

    num_groups = M * (K // group_size)
    grid = (triton.cdiv(num_groups, GROUPS_PER_PROGRAM),)
    per_token_group_quant_fp8_kernel[grid](
        x,
        x_q,
        x_s,
        group_size,
        num_groups,
        eps,
        finfo.min,
        finfo.max,
        BLOCK=triton.next_power_of_2(group_size),
        GROUPS=GROUPS_PER_PROGRAM,
    )
    return x_q, x_s
