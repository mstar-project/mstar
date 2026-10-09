"""Mamba-2 (SSD) recurrence over a recurrent state pool.

Two entry points, mirroring ``causal_conv1d``: ``mamba2_state_update`` is the
single-token step, one Triton program per (row, head), reading and writing the
pool's ``[max_slots, H, P, N]`` block in place through per-row slot indices;
``mamba2_scan`` is the sequential reference over a whole sequence, used for
prefill (a system prompt is short and runs eagerly) and as the oracle the
kernel is tested against.

The recurrence, per head ``h`` with ``P`` channels and an ``N``-wide state,
grouped ``B``/``C`` shared by ``H / G`` consecutive heads:

    dt  = clamp(softplus(dt + dt_bias), time_step_limit)
    S  <- exp(dt * A) * S + (dt * x) B^T          S: [P, N]
    y   = S C + D * x

which is HF's ``NemotronHMamba2Mixer`` / ``mamba_ssm.selective_state_update``
with ``dt_softplus=True`` and the kernel's contiguous group-to-head mapping.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1


@triton.jit
def _softplus(x):
    # torch.nn.functional.softplus with its default threshold of 20
    return tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.jit
def _mamba2_state_update_kernel(
    x_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, d_ptr, dt_bias_ptr, state_ptr, slots_ptr, out_ptr,
    stride_x_row, stride_x_head, stride_x_p,
    stride_dt_row, stride_dt_head,
    stride_b_row, stride_b_group, stride_b_n,
    stride_c_row, stride_c_group, stride_c_n,
    stride_s_slot, stride_s_head, stride_s_p, stride_s_n,
    stride_o_row, stride_o_head, stride_o_p,
    heads_per_group, dt_min, dt_max, pad_slot_id,
    HEAD_DIM: tl.constexpr, STATE: tl.constexpr,
    BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_D: tl.constexpr, HAS_DT_BIAS: tl.constexpr, DT_SOFTPLUS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    offs_p = tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, BLOCK_N)
    mask_p = offs_p < HEAD_DIM
    mask_n = offs_n < STATE
    out_ptrs = out_ptr + row * stride_o_row + head * stride_o_head + offs_p * stride_o_p

    slot = tl.load(slots_ptr + row)
    if slot == pad_slot_id:
        # NO_SLOT: no state to read or write for this row; its output is
        # defined (zeros) rather than whatever the buffer held. (The sink
        # slot, when the pool has one, is a real index and absorbs the write.)
        tl.store(out_ptrs, tl.zeros([BLOCK_P], dtype=tl.float32).to(out_ptr.dtype.element_ty), mask=mask_p)
        return
    group = head // heads_per_group

    x = tl.load(
        x_ptr + row * stride_x_row + head * stride_x_head + offs_p * stride_x_p,
        mask=mask_p, other=0.0,
    ).to(tl.float32)
    dt = tl.load(dt_ptr + row * stride_dt_row + head * stride_dt_head).to(tl.float32)
    if HAS_DT_BIAS:
        dt = dt + tl.load(dt_bias_ptr + head).to(tl.float32)
    if DT_SOFTPLUS:
        dt = _softplus(dt)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    a = tl.load(a_ptr + head).to(tl.float32)
    da = tl.exp(dt * a)

    bv = tl.load(
        b_ptr + row * stride_b_row + group * stride_b_group + offs_n * stride_b_n,
        mask=mask_n, other=0.0,
    ).to(tl.float32)
    cv = tl.load(
        c_ptr + row * stride_c_row + group * stride_c_group + offs_n * stride_c_n,
        mask=mask_n, other=0.0,
    ).to(tl.float32)

    s_ptrs = (
        state_ptr + slot.to(tl.int64) * stride_s_slot + head * stride_s_head
        + offs_p[:, None] * stride_s_p + offs_n[None, :] * stride_s_n
    )
    mask_pn = mask_p[:, None] & mask_n[None, :]
    s = tl.load(s_ptrs, mask=mask_pn, other=0.0).to(tl.float32)
    s = s * da + (dt * x)[:, None] * bv[None, :]
    tl.store(s_ptrs, s.to(s_ptrs.dtype.element_ty), mask=mask_pn)

    y = tl.sum(s * cv[None, :], axis=1)
    if HAS_D:
        y = y + tl.load(d_ptr + head).to(tl.float32) * x
    tl.store(out_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask_p)


def _effective_dt(
    dt: torch.Tensor, dt_bias: torch.Tensor | None, dt_softplus: bool,
    time_step_limit: tuple[float, float],
) -> torch.Tensor:
    dt = dt.float()
    if dt_bias is not None:
        dt = dt + dt_bias.float()
    if dt_softplus:
        dt = torch.nn.functional.softplus(dt)
    lo, hi = time_step_limit
    if lo > 0.0 or math.isfinite(hi):
        dt = dt.clamp(min=lo, max=hi)
    return dt


def _expand_groups(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``[..., G, N] -> [..., H, N]``: head ``h`` reads group ``h // (H / G)``."""
    return t.repeat_interleave(num_heads // t.shape[-2], dim=-2)


def mamba2_state_update_torch(
    x: torch.Tensor, dt: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    state: torch.Tensor, slot_indices: torch.Tensor,
    d: torch.Tensor | None = None, dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = True,
    time_step_limit: tuple[float, float] = (0.0, float("inf")),
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Reference for ``mamba2_state_update`` in plain torch (any device)."""
    rows, num_heads, _ = x.shape
    live = slot_indices != pad_slot_id
    idx = slot_indices[live].long()
    dt = _effective_dt(dt, dt_bias, dt_softplus, time_step_limit)[live]      # [n, H]
    da = torch.exp(dt * a.float())                                            # [n, H]
    xb = x[live].float()                                                      # [n, H, P]
    bx = _expand_groups(b[live].float(), num_heads)                           # [n, H, N]
    cx = _expand_groups(c[live].float(), num_heads)
    s = state.index_select(0, idx).float()                                    # [n, H, P, N]
    s = da[:, :, None, None] * s + (dt[:, :, None] * xb)[..., None] * bx[:, :, None, :]
    state.index_copy_(0, idx, s.to(state.dtype))
    y = torch.einsum("bhpn,bhn->bhp", s, cx)
    if d is not None:
        y = y + d.float().view(1, -1, 1) * xb
    out = x.new_zeros(x.shape)
    out[live] = y.to(x.dtype)
    return out


def mamba2_state_update(
    x: torch.Tensor, dt: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    state: torch.Tensor, slot_indices: torch.Tensor,
    d: torch.Tensor | None = None, dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = True,
    time_step_limit: tuple[float, float] = (0.0, float("inf")),
    pad_slot_id: int = PAD_SLOT_ID,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """One Mamba-2 step per row, state read and written in the pool in place.

    ``x`` ``[rows, H, P]``, ``dt`` ``[rows, H]`` (raw: bias, softplus and the
    clamp are applied here), ``a`` = ``-exp(A_log)`` ``[H]``, ``b``/``c``
    ``[rows, G, N]``, ``state`` ``[max_slots, H, P, N]`` (fp32 or bf16),
    ``slot_indices`` ``[rows]`` int32 (``pad_slot_id`` rows are skipped). Returns
    ``y`` ``[rows, H, P]`` in ``x``'s dtype. Every row must address a distinct
    slot except rows sharing the pool's sink slot, whose writes nobody reads.
    """
    if x.device.type != "cuda":
        return mamba2_state_update_torch(
            x, dt, a, b, c, state, slot_indices, d=d, dt_bias=dt_bias,
            dt_softplus=dt_softplus, time_step_limit=time_step_limit, pad_slot_id=pad_slot_id,
        )
    rows, num_heads, head_dim = x.shape
    num_groups, state_size = b.shape[1], b.shape[2]
    assert c.shape == b.shape and state.shape[1:] == (num_heads, head_dim, state_size), (
        x.shape, b.shape, c.shape, state.shape,
    )
    assert num_heads % num_groups == 0
    if out is None:
        out = torch.empty_like(x)
    if rows == 0:
        return out
    lo, hi = time_step_limit
    hi = float(hi) if math.isfinite(hi) else float(torch.finfo(torch.float32).max)
    dummy = a  # stands in for absent optional pointers; never read
    grid = (rows, num_heads)
    _mamba2_state_update_kernel[grid](
        x, dt, a, b, c,
        d if d is not None else dummy,
        dt_bias if dt_bias is not None else dummy,
        state, slot_indices.to(torch.int32), out,
        x.stride(0), x.stride(1), x.stride(2),
        dt.stride(0), dt.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        num_heads // num_groups, float(lo), hi, pad_slot_id,
        HEAD_DIM=head_dim, STATE=state_size,
        BLOCK_P=triton.next_power_of_2(head_dim), BLOCK_N=triton.next_power_of_2(state_size),
        HAS_D=d is not None, HAS_DT_BIAS=dt_bias is not None, DT_SOFTPLUS=dt_softplus,
    )
    return out


def mamba2_scan(
    x: torch.Tensor, dt: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    h0: torch.Tensor | None,
    d: torch.Tensor | None = None, dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = True,
    time_step_limit: tuple[float, float] = (0.0, float("inf")),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential SSD scan over one sequence: ``x`` ``[L, H, P]``, ``dt``
    ``[L, H]``, ``b``/``c`` ``[L, G, N]``, ``h0`` ``[H, P, N]`` or None (zeros).
    Returns ``(y [L, H, P], final state [H, P, N] fp32)``. The reference the
    step kernel is checked against; also the prefill path, which is short."""
    seq_len, num_heads, head_dim = x.shape
    state_size = b.shape[-1]
    dt = _effective_dt(dt, dt_bias, dt_softplus, time_step_limit)             # [L, H]
    da = torch.exp(dt * a.float())
    bx = _expand_groups(b.float(), num_heads)                                 # [L, H, N]
    cx = _expand_groups(c.float(), num_heads)
    xf = x.float()
    s = h0.float() if h0 is not None else xf.new_zeros(num_heads, head_dim, state_size)
    ys = []
    for t in range(seq_len):
        s = da[t].view(-1, 1, 1) * s + (dt[t].unsqueeze(-1) * xf[t]).unsqueeze(-1) * bx[t].unsqueeze(1)
        ys.append(torch.einsum("hpn,hn->hp", s, cx[t]))
    y = torch.stack(ys, dim=0) if ys else xf.new_zeros(0, num_heads, head_dim)
    if d is not None:
        y = y + d.float().view(1, -1, 1) * xf
    return y.to(x.dtype), s
