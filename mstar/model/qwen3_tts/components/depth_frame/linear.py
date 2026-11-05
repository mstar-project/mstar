"""The frame kernel's projection tasks: `_cp_linear_kernel`'s tiles with the K extent, the input and the epilogue
chosen at runtime.

One kernel holds every task type's code, and an SM's instruction cache does not hold it all: the frame ran each task
type about 2x slower than the same task alone until its K loops were rolled up. Each task type still calls these with
constant arguments, so their branches fold. `lin_task` uses `_cp_linear_kernel`'s tiles, MMA and reduction layouts, so
its outputs match that kernel bit for bit; `split_task` does not.
"""
from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma, warpgroup_mma, warpgroup_mma_wait

from mstar.model.qwen3_tts.components.depth_frame import cluster as C


@gluon.jit
def _stream_src(src, xd0, xd1, xd2, xd3, x_s, xbar, xc, s, k0, XSTAGES: gl.constexpr, BLOCK_K: gl.constexpr):
    """The task's x chunk ``s`` (chunk k0 + s of input ``src``: 0 the residual stream,
    1 attention out, 2 SwiGLU out, 3 the Talker hidden state) into its slot."""
    sl = (xc + s) % XSTAGES
    bar = xbar.index(sl)
    dst = x_s.index(sl)
    mbarrier.expect(bar, xd0.block_type.nbytes)
    col = (k0 + s) * BLOCK_K
    if src == 0:
        tma.async_copy_global_to_shared(xd0, [0, col], bar, dst)
    elif src == 1:
        tma.async_copy_global_to_shared(xd1, [0, col], bar, dst)
    elif src == 2:
        tma.async_copy_global_to_shared(xd2, [0, col], bar, dst)
    else:
        tma.async_copy_global_to_shared(xd3, [0, col], bar, dst)


@gluon.jit
def lin_task(
    pid, src, xd0, xd1, xd2, xd3, x_s, xbar, xc, p16, p32, p64, pg, wbar, wph,
    O, B, P, M, K, KC, eps, n_part, s_o, norm, epi,
    NP: gl.constexpr, MAX_ROWS: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr, TWO: gl.constexpr, XSTAGES: gl.constexpr, NPAGES: gl.constexpr,
    MMA: gl.constexpr, BLK_P: gl.constexpr, BLK_O: gl.constexpr, BLK_1D: gl.constexpr,
):
    """One tile: ``norm`` 0 or 2 (row scale from the ``n_part`` partials), ``epi`` 0 store, 2 SwiGLU (``TWO``: the
    tile's gate and up rows stacked in one 64-row weight tile, one N = 64 MMA per K step, so each x chunk is read from
    shared memory once), 3 bias add with partials. Returns the advanced ``xc``."""
    for s in range(gl.minimum(KC, XSTAGES)):
        _stream_src(src, xd0, xd1, xd2, xd3, x_s, xbar, xc, s, 0, XSTAGES, BLOCK_K)
    # the epilogue's global reads go out with the x prologue, off the path after the K loop
    rm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, BLK_O))
    rn = pid * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, BLK_O))
    rows = (rm < M)[:, None]
    o_ptrs = O + rm[:, None] * s_o + rn[None, :]
    ss = gl.zeros([BLOCK_M], gl.float32, gl.SliceLayout(0, BLK_P))
    if norm == 2:
        t = gl.arange(0, NP, layout=gl.SliceLayout(1, BLK_P))
        rmp = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, BLK_P))
        pp = gl.load(P + t[:, None] * MAX_ROWS + rmp[None, :], mask=(t < n_part)[:, None], other=0.0)
        ss = gl.sum(pp, axis=0)
    mbarrier.wait(wbar, wph)
    WN: gl.constexpr = 2 * BLOCK_N if TWO else BLOCK_N
    acc = gl.zeros([BLOCK_M, WN], gl.float32, MMA)
    for kc in range(KC):
        g = xc + kc
        st = g % XSTAGES
        xs = x_s.index(st)
        mbarrier.wait(xbar.index(st), (g // XSTAGES) & 1)
        if TWO:
            w = p64.index(((pg + 4 * kc) % NPAGES) // 4)
        elif BLOCK_N == 32:
            w = p32.index(((pg + 2 * kc) % NPAGES) // 2)
        else:
            w = p16.index((pg + kc) % NPAGES)
        acc = warpgroup_mma(xs, w.permute((1, 0)), acc, is_async=True)
        done = warpgroup_mma_wait(num_outstanding=1, deps=(acc, xs))
        acc = done[0]
        if (kc >= 1) & (kc - 1 + XSTAGES < KC):
            _stream_src(src, xd0, xd1, xd2, xd3, x_s, xbar, xc, kc - 1 + XSTAGES, 0, XSTAGES, BLOCK_K)
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    if norm == 2:
        inv = 1.0 / gl.sqrt(ss / K + eps)
        inv = gl.convert_layout(inv, gl.SliceLayout(1, MMA))
        acc = acc * inv[:, None]
    if TWO:
        # columns 0-31 are the gate, 32-63 the up projection: a thread holds column c and c + 32
        g, u = gl.split(gl.permute(gl.reshape(acc, [BLOCK_M, 2, BLOCK_N]), (0, 2, 1)))
        acc = g / (1.0 + gl.exp(-g)) * u
    if epi == 3:
        rnb = pid * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, MMA))
        acc = acc + gl.load(B + rnb).to(gl.float32)[None, :]
    y_mma = acc.to(gl.bfloat16)
    y = gl.convert_layout(y_mma, BLK_O)
    gl.store(o_ptrs, y, mask=rows)
    if epi == 3:
        # reduced in the MMA layout, as Triton does for a bias tile
        rm1 = gl.arange(0, BLOCK_M, layout=BLK_1D)
        yf = y_mma.to(gl.float32)
        gl.store(P + pid * MAX_ROWS + rm1, gl.convert_layout(gl.sum(yf * yf, axis=1), BLK_1D))
    return xc + KC


@gluon.jit
def split_task(
    tile2, src, xd0, xd1, xd2, xd3, x_s, xbar, xc, p16, pg, wbar, wph, O, P, M, KH, OFF, parity, s_o,
    MAX_ROWS: gl.constexpr, XSTAGES: gl.constexpr, NPAGES: gl.constexpr, MMA: gl.constexpr,
    BLK_O: gl.constexpr, BLK_1D: gl.constexpr,
):
    """Rank ``tile2 % 2`` of a 64 x 16 residual-add tile (O, DOWN) split across a CTA pair: it
    streams K chunks rank * KH .. rank * KH + KH - 1. Rank 1 sends its fp32 partial to rank 0
    (see `cluster`); rank 0 adds it to its own, always in that order, then adds the residual,
    stores the tile and its row partials. OFF: the combine buffer's offset in shared memory.
    Returns the advanced ``xc``."""
    BLOCK_K: gl.constexpr = 128
    tile = tile2 // 2
    rank = tile2 % 2
    k0 = rank * KH
    bar = C.smem_base(gl.zeros([128], gl.int32, BLK_1D)) + (OFF + 4096)
    if rank == 0:
        C.expect_tx(bar)
    for s in range(gl.minimum(KH, XSTAGES)):
        _stream_src(src, xd0, xd1, xd2, xd3, x_s, xbar, xc, s, k0, XSTAGES, BLOCK_K)
    rm = gl.arange(0, 64, layout=gl.SliceLayout(1, BLK_O))
    rl = gl.arange(0, 16, layout=gl.SliceLayout(0, BLK_O))
    rows = (rm < M)[:, None]
    o_ptrs = O + rm[:, None] * s_o + (tile * 16 + rl)[None, :]
    res = gl.zeros([64, 16], gl.bfloat16, BLK_O)
    if rank == 0:
        res = gl.load(o_ptrs, mask=rows, other=0.0)
    mbarrier.wait(wbar, wph)
    acc = gl.zeros([64, 16], gl.float32, MMA)
    for kc in range(KH):
        g = xc + kc
        st = g % XSTAGES
        xs = x_s.index(st)
        mbarrier.wait(xbar.index(st), (g // XSTAGES) & 1)
        acc = warpgroup_mma(xs, p16.index((pg + kc) % NPAGES).permute((1, 0)), acc, is_async=True)
        done = warpgroup_mma_wait(num_outstanding=1, deps=(acc, xs))
        acc = done[0]
        if (kc >= 1) & (kc - 1 + XSTAGES < KH):
            _stream_src(src, xd0, xd1, xd2, xd3, x_s, xbar, xc, kc - 1 + XSTAGES, k0, XSTAGES, BLOCK_K)
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
    accb = gl.convert_layout(acc, BLK_O)
    idx = rm[:, None] * 16 + rl[None, :]
    base = C.smem_base(idx) + OFF
    addr = base + idx * 4
    if rank == 1:
        C.st_async4(addr, accb, base + 4096)
    else:
        C.wait_cluster(bar, gl.zeros_like(bar) + parity)
        accb = accb + C.ld_local4(addr)
        y = (accb + res.to(gl.float32)).to(gl.bfloat16)
        gl.store(o_ptrs, y, mask=rows)
        yf = y.to(gl.float32)
        rm1 = gl.arange(0, 64, layout=BLK_1D)
        gl.store(P + tile * MAX_ROWS + rm1, gl.convert_layout(gl.sum(yf * yf, axis=1), BLK_1D))
    return xc + KH


def layouts(block_n: int):
    """The MMA, partials, store and row layouts `_cp_linear_kernel` gets at 64 rows and 4 warps."""
    mma = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, block_n, 16])
    blk_p = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    blk_o = gl.BlockedLayout([1, 8], [16, 2] if block_n == 16 else [8, 4], [4, 1], [1, 0])
    return mma, blk_p, blk_o, gl.BlockedLayout([1], [32], [4], [0])


def shared_layout(rank: int = 2):
    return gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=rank)
