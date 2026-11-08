"""`_fused_sample_kernel`'s draw of one row as a Gluon device function, for a kernel that draws in place
(`SamplerResource.fused_draw_args`).

Same code and the same 4-warp layout of a 2048-wide row as that kernel's ([8] per thread, 32 lanes, 4 warps), so
every reduction and the scan run in the same order and the drawn token is the same, bit for bit: the cuts compare
the probabilities' full float32 bits, as that kernel's do. No repetition penalty: a step with one gets no fused draw
arguments.
"""
from __future__ import annotations

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from mstar.engine.resources.sampler.utils import _kth_largest_bits


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _argmax_tie_break_left(value1, index1, value2, index2):
    """`triton.language.standard._argmax_combine` with tie_break_left."""
    tie = (value1 == value2) & (index1 < index2)
    gt = (value1 > value2) | tie
    return gl.where(gt, value1, value2), gl.where(gt, index1, index2)


def layout():
    return gl.BlockedLayout([8], [32], [4], [0])


@gluon.jit
def draw_row(lptr, temp, k, top_p, seed, offset, V, BLOCK: gl.constexpr, LAY: gl.constexpr):
    offs = gl.arange(0, BLOCK, layout=LAY)
    valid = offs < V
    x = gl.load(lptr + offs, mask=valid, other=-float("inf")).to(gl.float32)
    greedy = temp == 0
    x_max, arg = gl.reduce((x, offs), 0, _argmax_tie_break_left)
    inv_temp = gl.where(greedy, 1.0, 1.0 / gl.maximum(temp, 1e-30))
    e = gl.where(valid, gl.minimum(gl.exp((x - x_max) * inv_temp), 1.0), 0.0)
    key = e.to(gl.int32, bitcast=True)
    t = 0
    if k > 0:
        if k < V:
            t = _kth_largest_bits(key, k)
    if top_p < 1.0:
        target = top_p * gl.sum(gl.where(key >= t, e, 0.0), axis=0)
        lo = t
        hi = 0x3F800000
        for _ in range(30):
            mid = lo + (hi - lo) // 2
            ok = gl.sum(gl.where(key > mid, e, 0.0), axis=0) < target
            hi = gl.where(ok, mid, hi)
            lo = gl.where(ok, lo, mid + 1)
        t = lo
    q = gl.where(key >= t, e, 0.0)
    u = (1.0 - tl.rand(seed, offset)) * gl.sum(q, axis=0)
    csum = gl.associative_scan(q, 0, _add)
    drawn = gl.sum((csum < u).to(gl.int32), axis=0)
    drawn = gl.minimum(drawn, gl.max(gl.where(q > 0, offs, 0), axis=0))
    return gl.where(greedy, arg, drawn).to(gl.int64)
