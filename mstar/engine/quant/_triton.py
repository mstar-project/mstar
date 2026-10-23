"""Triton kernels behind the quant ops; imported only on CUDA."""
import torch
import triton
import triton.language as tl


def _row_warps(block: int) -> int:
    # one program per row: wide rows want more warps (H100, M <= 64 decode rows)
    return 16 if block >= 4096 else 8 if block >= 2048 else 4


@triton.jit
def _quant_per_token_kernel(
    x_ptr, q_ptr, s_ptr, K, stride_x, stride_q,
    fp8_max: tl.constexpr, inv_max: tl.constexpr, min_scale: tl.constexpr, BLOCK: tl.constexpr,
    ONE_PASS: tl.constexpr,
):
    # one program per row: amax, then scale and cast; a row that fits one block is read once
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride_x
    q_row = q_ptr + row * stride_q
    if ONE_PASS:
        cols = tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < K, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x), axis=0) * inv_max, min_scale)
        tl.store(s_ptr + row, scale)
        # div_rn: IEEE division, so values match the portable path bit for bit
        q = tl.clamp(tl.math.div_rn(x, scale), -fp8_max, fp8_max)
        tl.store(q_row + cols, q.to(q_ptr.dtype.element_ty), mask=cols < K)
    else:
        amax = tl.zeros((BLOCK,), dtype=tl.float32)
        for start in range(0, K, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            x = tl.load(x_row + cols, mask=cols < K, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.abs(x))
        scale = tl.maximum(tl.max(amax, axis=0) * inv_max, min_scale)
        tl.store(s_ptr + row, scale)
        for start in range(0, K, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            x = tl.load(x_row + cols, mask=cols < K, other=0.0).to(tl.float32)
            q = tl.clamp(tl.math.div_rn(x, scale), -fp8_max, fp8_max)
            tl.store(q_row + cols, q.to(q_ptr.dtype.element_ty), mask=cols < K)


def quant_per_token(x: torch.Tensor, fp8_max: float, inv_max: float, min_scale: float):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    m, k = x2.shape
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    if m:
        block = triton.next_power_of_2(k)
        one_pass = block <= 8192
        block = block if one_pass else 4096
        _quant_per_token_kernel[(m,)](
            x2, q, s, k, x2.stride(0), q.stride(0),
            fp8_max=fp8_max, inv_max=inv_max, min_scale=min_scale, BLOCK=block, ONE_PASS=one_pass,
            num_warps=_row_warps(block),
        )
    return q.reshape(shape), s.reshape(*shape[:-1], 1)


@triton.jit
def _fp8_mm_splitk_kernel(
    a_ptr, b_ptr, part_ptr, M, N, K, stride_a, stride_b, stride_pk, k_per_split,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # partial A @ B^T over one K split, fp32; a reduce pass applies the scales
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_start = pid_k * k_per_split
    for kk in range(0, k_per_split, BLOCK_K):
        rk = k_start + kk + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + rm[:, None] * stride_a + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rn[None, :] * stride_b + rk[:, None], mask=(rn[None, :] < N) & (rk[:, None] < K), other=0.0)
        acc = tl.dot(a, b, acc)
    out = part_ptr + pid_k * stride_pk + rm[:, None] * N + rn[None, :]
    tl.store(out, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _fp8_mm_reduce_kernel(
    part_ptr, sa_ptr, sb_ptr, bias_ptr, out_ptr, M, N, stride_pk, stride_out,
    SPLIT_K: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(SPLIT_K):
        acc += tl.load(part_ptr + k * stride_pk + row * N + cols, mask=mask, other=0.0)
    acc = acc * tl.load(sa_ptr + row) * tl.load(sb_ptr + cols, mask=mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_out + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)


_SMS: dict[int, int] = {}


def fp8_mm_small_m(a, sa, b, sb, bias, out_dtype, n_out, block_n=64, block_k=128):
    """``(a @ b.T) * sa * sb + bias`` for few rows of ``a``; split-K fills the GPU
    where scaled_mm's tiles leave most SMs idle."""
    m, k = a.shape
    dev = a.device.index or 0
    if dev not in _SMS:
        _SMS[dev] = torch.cuda.get_device_properties(a.device).multi_processor_count
    tiles_n = triton.cdiv(n_out, block_n)
    split = max(1, min(triton.cdiv(k, block_k), (2 * _SMS[dev]) // tiles_n))
    k_per_split = triton.cdiv(triton.cdiv(k, split), block_k) * block_k
    split = triton.cdiv(k, k_per_split)
    part = torch.empty((split, m, n_out), dtype=torch.float32, device=a.device)
    out = torch.empty((m, n_out), dtype=out_dtype, device=a.device)
    if m == 0:
        return out
    _fp8_mm_splitk_kernel[(tiles_n, split)](
        a, b, part, m, n_out, k, a.stride(0), b.stride(0), part.stride(0), k_per_split,
        BLOCK_M=max(16, triton.next_power_of_2(m)), BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4, num_stages=4,
    )
    _fp8_mm_reduce_kernel[(m, triton.cdiv(n_out, 1024))](
        part, sa, sb, bias if bias is not None else part, out, m, n_out, part.stride(0), out.stride(0),
        SPLIT_K=split, HAS_BIAS=bias is not None, BLOCK=1024, num_warps=4,
    )
    return out


@triton.jit
def _add_rms_norm_quant_kernel(
    h_ptr, r_ptr, w_ptr, q_ptr, s_ptr, K, stride_h, stride_r, stride_q, eps,
    weight_bias: tl.constexpr, fp8_max: tl.constexpr, inv_max: tl.constexpr, min_scale: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # FlashInfer's fused add + RMSNorm in place (sum of squares on the fp32 sum),
    # then the stored norm quantized per token
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    h_row = h_ptr + row * stride_h + cols
    r_row = r_ptr + row * stride_r + cols
    x = tl.load(h_row, mask=mask, other=0.0).to(tl.float32) + tl.load(r_row, mask=mask, other=0.0).to(tl.float32)
    tl.store(r_row, x.to(r_ptr.dtype.element_ty), mask=mask)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / K + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32) + weight_bias
    y = (x * rstd * w).to(h_ptr.dtype.element_ty)
    tl.store(h_row, y, mask=mask)
    y = y.to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(y), axis=0) * inv_max, min_scale)
    tl.store(s_ptr + row, scale)
    q = tl.clamp(tl.math.div_rn(y, scale), -fp8_max, fp8_max)
    tl.store(q_ptr + row * stride_q + cols, q.to(q_ptr.dtype.element_ty), mask=mask)


def add_rms_norm_quant(h, r, weight, eps, gemma, fp8_max, inv_max, min_scale):
    """In place ``r += h; h = norm(r)``; returns ``h`` quantized per token. 2D, rows of at most 8192."""
    m, k = h.shape
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=h.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=h.device)
    if m:
        block = triton.next_power_of_2(k)
        _add_rms_norm_quant_kernel[(m,)](
            h, r, weight, q, s, k, h.stride(0), r.stride(0), q.stride(0), eps,
            weight_bias=1.0 if gemma else 0.0, fp8_max=fp8_max, inv_max=inv_max, min_scale=min_scale,
            BLOCK=block, num_warps=_row_warps(block),
        )
    return q, s
