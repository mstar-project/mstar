"""One decoder layer's MLP block as a single persistent kernel, for decode batches of 1-4 rows.

In place, as the graph's two fused add+norm kernels leave things:
    r1 = residual + h;  x = norm(r1) * (1 + w_post)
    a = silu(x @ Wg^T) * (x @ Wu^T)
    r2 = r1 + a @ Wd^T;  residual = r2;  h = norm(r2) * (1 + w_next)

Grid = SM count. CTA p owns rows [start, start + count) of the intermediate dim (I / G each, the
remainder spread one apiece) and walks them RB at a time in one pipelined loop: per group it loads
the group's gate, up and Wd^T rows (Wd^T is the down weight stored transposed, so a group's rows are
contiguous), finishes those rows of `a` (each row covers all of K), and adds their contribution to
every output column into a register accumulator. Every weight byte is read once and the loop keeps
NS groups in flight. The accumulator goes to an fp32 buffer with relaxed atomics; no CTA ever
waits: the last one to bump the arrival counter finishes the residual and the next norm, then
zeroes the buffer and the counter for the next launch.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.kernels.megakernel.sync import row as _row


@triton.jit
def _mlp_kernel(h_ptr, res_ptr, wpost_ptr, wgu_ptr, wdt_ptr, wnext_ptr, acc_ptr, cnt_ptr, eps,
                B: tl.constexpr, BB: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
                RB: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    G = tl.num_programs(0)
    per = I // G
    extra = I % G
    start = pid * per + tl.minimum(pid, extra)
    count = per + (pid < extra).to(tl.int32)
    bi = tl.arange(0, BB)
    bmask = bi < B
    hk = tl.arange(0, H)
    rb = tl.arange(0, RB)

    hv = tl.load(h_ptr + bi[:, None] * H + hk[None, :], mask=bmask[:, None], other=0.0).to(tl.float32)
    rv = tl.load(res_ptr + bi[:, None] * H + hk[None, :], mask=bmask[:, None], other=0.0).to(tl.float32)
    r1 = hv + rv
    rstd = tl.rsqrt(tl.sum(r1 * r1, axis=1) / H + eps)
    wp = tl.load(wpost_ptr + hk).to(tl.float32)
    x = (r1 * rstd[:, None] * (1.0 + wp)[None, :]).to(tl.bfloat16).to(tl.float32)

    out = tl.zeros([BB, H], dtype=tl.float32)
    for r0 in tl.range(0, count, RB, num_stages=NS):
        live = (r0 + rb) < count
        rows = start + r0 + rb
        wg = tl.load(wgu_ptr + rows[:, None] * H + hk[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        wu = tl.load(wgu_ptr + (I + rows)[:, None] * H + hk[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        wd = tl.load(wdt_ptr + rows[:, None] * H + hk[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        for b in tl.static_range(BB):
            if b < B:
                xb = _row(x, bi, b)
                g = tl.sum(wg * xb[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
                u = tl.sum(wu * xb[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
                a = (g * tl.sigmoid(g) * u).to(tl.bfloat16).to(tl.float32)
                contrib = tl.sum(a[:, None] * wd, axis=0)
                out += tl.where(bi[:, None] == b, contrib[None, :], 0.0)
    tl.atomic_add(acc_ptr + bi[:, None] * H + hk[None, :], out, mask=bmask[:, None], sem="relaxed")

    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
    if done == G - 1:
        accv = tl.load(acc_ptr + bi[:, None] * H + hk[None, :], mask=bmask[:, None], other=0.0,
                       cache_modifier=".cg")
        r1b = r1.to(tl.bfloat16).to(tl.float32)
        r2 = r1b + accv.to(tl.bfloat16).to(tl.float32)
        rstd2 = tl.rsqrt(tl.sum(r2 * r2, axis=1) / H + eps)
        wn = tl.load(wnext_ptr + hk).to(tl.float32)
        hn = r2 * rstd2[:, None] * (1.0 + wn)[None, :]
        tl.store(res_ptr + bi[:, None] * H + hk[None, :], r2.to(tl.bfloat16), mask=bmask[:, None])
        tl.store(h_ptr + bi[:, None] * H + hk[None, :], hn.to(tl.bfloat16), mask=bmask[:, None])
        tl.store(acc_ptr + bi[:, None] * H + hk[None, :], tl.zeros([BB, H], dtype=tl.float32),
                 mask=bmask[:, None])
        tl.atomic_xchg(cnt_ptr, 0, sem="release", scope="gpu")
class MlpMegakernel:
    """Scratch (fp32 accumulator, arrival counter) and launch config. One instance serves any number of
    layers whose launches are ordered on one stream: each launch leaves both zeroed."""

    def __init__(self, hidden: int, inter: int, max_rows: int = 4, device="cuda", grid: int | None = None,
                 rb: int = 4, ns: int = 3, num_warps: int = 4):
        self.H, self.I = hidden, inter
        self.grid = grid or torch.cuda.get_device_properties(device).multi_processor_count
        self.acc = torch.zeros(max_rows, hidden, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(1, dtype=torch.int32, device=device)
        self.rb, self.ns, self.num_warps = rb, ns, num_warps

    def __call__(self, h, residual, w_post, w_gu, w_dt, w_next, eps: float):
        """``h`` (the mixer's output) and ``residual`` are [rows, hidden] bf16, updated in place; ``w_gu`` is
        the fused [2 * inter, hidden] gate/up weight and ``w_dt`` the down weight transposed, [inter, hidden]."""
        B = h.shape[0]
        _mlp_kernel[(self.grid,)](
            h, residual, w_post, w_gu, w_dt, w_next, self.acc, self.cnt, eps,
            B=B, BB=triton.next_power_of_2(B), H=self.H, I=self.I, RB=self.rb, NS=self.ns,
            num_warps=self.num_warps,
        )
