"""One CodePredictor frame (16 depth positions) in one persistent launch.

Each CTA walks a static task list (`schedule.py`), one packed 32-byte row per
task, loaded while the previous task runs. A task waits on a counter, runs one
tile of a `code_predictor_kernels` kernel and signals a counter. Weight tiles
live in a ring of 4 KB shared-memory pages; the host decides where each tile
goes and before which task its TMA copies are issued, so a tile streams in
while the CTA waits on earlier phases.

Task types: projections (`linear.lin_task`), attention for one q head and 8
rows (`attend.attend4`, 4 rows a call), and the per-row draw
(`fused_draw.draw_row`) and embedding step. The kernel runs in clusters of two
CTAs, and an O or DOWN tile's K loop is split across a cluster
(`linear.split_task`), so the frame's outputs differ from `_depth_loop`'s in
the last bits.

Shapes are the 1.7B CodePredictor's: hidden 1024, 16 q / 8 kv heads of 128,
intermediate 3072, vocab 2048, 5 layers, 16 positions; Talker width 2048.
"""
from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma

from mstar.engine.resources.sampler.fused_draw import draw_row
from mstar.model.qwen3_tts.components.depth_frame import cluster as C
from mstar.model.qwen3_tts.components.depth_frame import schedule
from mstar.model.qwen3_tts.components.depth_frame.attend import attend4
from mstar.model.qwen3_tts.components.depth_frame.linear import lin_task, split_task

# `schedule`'s task encoding, as constants a Gluon function can read
T_QKV, T_O, T_GATEUP, T_DOWN, T_ATTEND, T_INPROJ, T_HEAD = (
    gl.constexpr(t) for t in (schedule.T_QKV, schedule.T_O, schedule.T_GATEUP, schedule.T_DOWN, schedule.T_ATTEND,
                              schedule.T_INPROJ, schedule.T_HEAD)
)
ROW = gl.constexpr(schedule.ROW)


@gluon.jit
def _acquire(ptr, NT: gl.constexpr):
    """Every thread loads the counter with acquire semantics. A 1-element tensor
    keeps the min a no-op: no shuffles, no shared memory, no CTA barrier, so each
    warp leaves its spin as soon as it sees the target."""
    lay: gl.constexpr = gl.BlockedLayout([1], [32], [NT // 32], [0])
    p = ptr + gl.zeros([1], gl.int32, lay)
    v = gl.inline_asm_elementwise("ld.acquire.gpu.global.u32 $0, [$1];", "=r,l", [p],
                                  dtype=gl.int32, is_pure=False, pack=1)
    return gl.min(v, 0)


@gluon.jit
def _wait(ptr, target, NT: gl.constexpr):
    """Spin until the counter reaches ``target``. Every thread acquires, so its
    later reads are ordered after the producers' stores; the proxy fence then
    orders this CTA's TMA (async-proxy) reads after them too."""
    c = _acquire(ptr, NT)
    while c < target:
        c = _acquire(ptr, NT)
    lay: gl.constexpr = gl.BlockedLayout([1], [32], [NT // 32], [0])
    z = gl.zeros([1], gl.int32, lay)
    gl.inline_asm_elementwise("fence.proxy.async.global; mov.u32 $0, $1;", "=r,r", [z],
                              dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _signal(ptr):
    gl.barrier()
    gl.atomic_add(ptr, 1, sem="release", scope="gpu")


@gluon.jit
def _issue_tile(desc, bar, p16, pg, row, NB: gl.constexpr, CB: gl.constexpr, BN: gl.constexpr,
                NPAGES: gl.constexpr, c0):
    """A weight tile as NB copies of CB chunks each, from chunk c0: the 4-D descriptors
    view a weight as [chunk, 64-column half, row, 64 columns], so one copy writes CB
    chunks' pages, each chunk as the [BN, 128] tile the MMA reads."""
    PPB: gl.constexpr = CB * (BN // 16)  # pages per copy
    mbarrier.expect(bar, NB * desc.block_type.nbytes)
    for j in gl.static_range(NB):
        tma.async_copy_global_to_shared(desc, [c0 + CB * j, 0, row, 0], bar,
                                        p16.index((pg + PPB * j) % NPAGES)._reinterpret(gl.bfloat16, [CB, 2, BN, 64],
                                                                                        desc.layout))


@gluon.jit
def _issue(e, bars, p16, wqkv_desc, wo_desc, wgu_desc, wd_desc, win_desc, whead_desc, NPAGES: gl.constexpr):
    """Weight tile from a packed issue entry: type | first page << 4 | barrier << 10 | rank << 13 | first row << 16.
    An O or DOWN tile is the rank's half of K."""
    ty = e & 15
    pg = (e >> 4) & 63
    bar = bars.index((e >> 10) & 7)
    rank = (e >> 13) & 1
    row = e >> 16
    if ty == T_QKV:
        _issue_tile(wqkv_desc, bar, p16, pg, row, 1, 8, 32, NPAGES, 0)
    elif ty == T_O:
        _issue_tile(wo_desc, bar, p16, pg, row, 1, 8, 16, NPAGES, rank * 8)
    elif ty == T_GATEUP:
        _issue_tile(wgu_desc, bar, p16, pg, row, 2, 4, 64, NPAGES, 0)
    elif ty == T_DOWN:
        _issue_tile(wd_desc, bar, p16, pg, row, 3, 4, 16, NPAGES, rank * 12)
    elif ty == T_INPROJ:
        _issue_tile(win_desc, bar, p16, pg, row, 2, 8, 16, NPAGES, 0)
    else:  # T_HEAD
        _issue_tile(whead_desc, bar, p16, pg, row, 1, 8, 16, NPAGES, 0)


@gluon.jit
def _prefetch_kv(KV, s_kv_b, s_kv_c, s_kv_n, s_kv_h, rb, hk, cache_pos):
    """L2 prefetch of the cached keys and values an attention task reads (rows
    rb .. rb + 7, kv head hk, positions before cache_pos): earlier positions
    wrote them, so this goes out before the task's wait."""
    lay: gl.constexpr = gl.BlockedLayout([1, 1, 4], [2, 2, 8], [4, 1, 1], [2, 1, 0])
    r = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, lay)))
    c = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, lay)))
    q = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, lay)))  # position * 2 + 128-byte line
    n = gl.minimum(q // 2, gl.maximum(cache_pos - 1, 0))
    p = (KV + (rb + r)[:, None, None] * s_kv_b + c[None, :, None] * s_kv_c + hk * s_kv_h
         + n[None, None, :] * s_kv_n + (q % 2)[None, None, :] * 64)
    gl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [p],
                              dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def _embed_row(b, g, tok, X, P, CODES, s_codes, SUM, s_sum, W, s_w, TP, s_tp,
               INIT: gl.constexpr, PROJ: gl.constexpr, MAX_ROWS: gl.constexpr, E1: gl.constexpr, E2: gl.constexpr):
    """`_cp_embed_accumulate_kernel` for row b: record the code, add (or start)
    the Talker-width embedding sum, and with PROJ write the next input from the
    projected table with its four 256-wide partial sums of squares. A 256 chunk
    is reduced across 2 warps as in today's 2-warp programs: two chunks at once,
    one per warp pair."""
    # the running sum: 2048 wide, 8 chunks of 256
    c8 = gl.arange(0, 2048, layout=E1)
    e = gl.load(W + tok * s_w + c8)
    if INIT:
        gl.store(SUM + b * s_sum + c8, e.to(SUM.dtype.element_ty))
    else:
        acc = gl.load(SUM + b * s_sum + c8)
        gl.store(SUM + b * s_sum + c8, acc + e.to(acc.dtype))
    gl.store(CODES + b * s_codes + g, tok.to(CODES.dtype.element_ty))
    if PROJ:
        for half in gl.static_range(2):
            rc = half * 2 + gl.arange(0, 2, layout=gl.SliceLayout(1, E2))
            cc = gl.arange(0, 256, layout=gl.SliceLayout(0, E2))
            offs = rc[:, None] * 256 + cc[None, :]
            e2 = gl.load(TP + tok * s_tp + offs)
            gl.store(X + b * 1024 + offs, e2)
            ef = e2.to(gl.float32)
            part = gl.sum(ef * ef, axis=1)
            gl.store(P + rc * MAX_ROWS + b, part)


@gluon.jit
def frame_kernel(
    TASKS, NTASKS, CTRS,
    x_desc, att_desc, act_desc, hid_desc,
    wqkv_desc, wo_desc, wgu_desc, wd_desc, win_desc, whead_desc,
    X, QKV, ATT, ACT, LOGITS, P, BIAS_IN,
    QN, KN, POS, s_pos, ROPE, KV, s_kv_l, s_kv_b, s_kv_c, s_kv_n, s_kv_h,
    CODES_IN, s_codes_in, CODES, s_codes, SUM, s_sum, TAB0, s_tab0, CEMB, s_cemb_g, s_cemb, TP0, TPROJ, s_tproj_g, s_tp,
    TEMP, TOPK, TOPP, SEED, OFFSET,
    M, eps, eps_q, eps_k, sm_scale, OFF,
    MAXT: gl.constexpr, NPAGES: gl.constexpr, TB: gl.constexpr, XSTAGES: gl.constexpr,
    MMA16: gl.constexpr, MMA32: gl.constexpr, MMA64: gl.constexpr, BLK_P: gl.constexpr,
    BLK_O16: gl.constexpr, BLK_O32: gl.constexpr, BLK_1D: gl.constexpr,
    L1: gl.constexpr, L3: gl.constexpr, K3: gl.constexpr, V3: gl.constexpr,
    E1: gl.constexpr, E2: gl.constexpr, SAMPLE: gl.constexpr, SLAY: gl.constexpr,
):
    cta = gl.program_id(0)
    p16 = gl.allocate_shared_memory(gl.bfloat16, [NPAGES, 16, 128], x_desc.layout)
    p32 = p16._reinterpret(gl.bfloat16, [NPAGES // 2, 32, 128], x_desc.layout)
    p64 = p16._reinterpret(gl.bfloat16, [NPAGES // 4, 64, 128], x_desc.layout)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [XSTAGES, 64, 128], x_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [TB, 1], mbarrier.MBarrierLayout())
    xbar = gl.allocate_shared_memory(gl.int64, [XSTAGES, 1], mbarrier.MBarrierLayout())
    for b in gl.static_range(TB):
        mbarrier.init(bars.index(b), count=1)
    for s in gl.static_range(XSTAGES):
        mbarrier.init(xbar.index(s), count=1)
    z1 = gl.zeros([128], gl.int32, BLK_1D)
    C.cluster_init(C.smem_base(z1) + (OFF + 4096))

    n = gl.load(NTASKS + cta)
    base = TASKS + cta * (MAXT * ROW)
    # task i's row (see `schedule.pack`) is loaded while task i - 1 runs
    c0 = gl.load(base)
    c1 = gl.load(base + 1)
    c2 = gl.load(base + 2)
    c3 = gl.load(base + 3)
    c4 = gl.load(base + 4)
    xc = 0
    ncomb = 0  # split tiles this CTA has combined as rank 0: the combine mbarrier's phase
    for i in range(n):
        nb = base + gl.minimum(i + 1, n - 1) * ROW
        n0 = gl.load(nb)
        n1 = gl.load(nb + 1)
        n2 = gl.load(nb + 2)
        n3 = gl.load(nb + 3)
        n4 = gl.load(nb + 4)
        nis = (c0 >> 20) & 15
        if nis > 0:
            _issue(c3, bars, p16, wqkv_desc, wo_desc, wgu_desc, wd_desc, win_desc, whead_desc, NPAGES)
        if nis > 1:
            _issue(c4, bars, p16, wqkv_desc, wo_desc, wgu_desc, wd_desc, win_desc, whead_desc, NPAGES)
        ty = c0 & 15
        layer = (c0 >> 4) & 15
        tile = (c0 >> 8) & 255
        pos = (c0 >> 16) & 15
        n_part = c0 >> 24
        if ty == T_ATTEND:
            if pos > 0:
                _prefetch_kv(KV + layer * s_kv_l, s_kv_b, s_kv_c, s_kv_n, s_kv_h, (tile % 8) * 8, (tile // 8) // 2, pos)
        _wait(CTRS + (c1 & 65535), c1 >> 16, 128)
        pg = (c2 >> 16) & 255
        wk = c2 >> 24
        wbar = bars.index(wk % TB)
        wph = (wk // TB) & 1
        if ty == T_QKV:
            xc = lin_task(tile, 0, x_desc, att_desc, act_desc, hid_desc, x_s, xbar, xc, p16, p32, p64, pg, wbar, wph,
                          QKV, X, P, M, 1024, 8, eps, n_part, 4096, 2, 0,
                          64, 64, 64, 32, 128, False, XSTAGES, NPAGES, MMA32, BLK_P, BLK_O32, BLK_1D)
        elif ty == T_GATEUP:
            xc = lin_task(tile, 0, x_desc, att_desc, act_desc, hid_desc, x_s, xbar, xc, p16, p32, p64, pg, wbar, wph,
                          ACT, X, P, M, 1024, 8, eps, 64, 3072, 2, 2,
                          64, 64, 64, 32, 128, True, XSTAGES, NPAGES, MMA64, BLK_P, BLK_O32, BLK_1D)
        elif ty == T_ATTEND:
            # q head tile // 8, rows 8 * (tile % 8) ..
            h = tile // 8
            rb = (tile % 8) * 8
            for r in range(2):
                attend4(QKV, 4096, QN + layer * 128, KN + layer * 128, POS + pos * s_pos, ROPE, KV + layer * s_kv_l,
                        s_kv_b, s_kv_c, s_kv_n, s_kv_h, ATT, 2048,
                        rb + 4 * r, h, M, pos, eps_q, eps_k, sm_scale,
                        16, 8, 128, 64, 16, L1, L3, K3, V3)
        elif ty == T_O:
            xc = split_task(tile, 1, x_desc, att_desc, act_desc, hid_desc, x_s, xbar, xc, p16, pg, wbar, wph,
                            X, P, M, 8, OFF, ncomb & 1, 1024, 64, XSTAGES, NPAGES, MMA16, BLK_O16, BLK_1D)
            ncomb += 1 - tile % 2
        elif ty == T_DOWN:
            xc = split_task(tile, 2, x_desc, att_desc, act_desc, hid_desc, x_s, xbar, xc, p16, pg, wbar, wph,
                            X, P, M, 12, OFF, ncomb & 1, 1024, 64, XSTAGES, NPAGES, MMA16, BLK_O16, BLK_1D)
            ncomb += 1 - tile % 2
        elif ty == T_INPROJ:
            xc = lin_task(tile, 3, x_desc, att_desc, act_desc, hid_desc, x_s, xbar, xc, p16, p32, p64, pg, wbar, wph,
                          X, BIAS_IN, P, M, 2048, 16, eps, 0, 1024, 0, 3,
                          64, 64, 64, 16, 128, False, XSTAGES, NPAGES, MMA16, BLK_P, BLK_O16, BLK_1D)
        elif ty == T_HEAD:
            xc = lin_task(tile, 0, x_desc, att_desc, act_desc, hid_desc, x_s, xbar, xc, p16, p32, p64, pg, wbar, wph,
                          LOGITS, BIAS_IN, P, M, 1024, 8, eps, 64, 2048, 2, 0,
                          64, 64, 64, 16, 128, False, XSTAGES, NPAGES, MMA16, BLK_P, BLK_O16, BLK_1D)
        elif tile < M:
            # T_EMBED: row `tile`'s code for group `pos`: drawn here (SAMPLE, groups 1-15) or given, then its embedding
            if SAMPLE:
                if pos == 0:
                    tok = gl.load(CODES_IN + tile * s_codes_in)
                else:
                    off0 = gl.load(OFFSET + tile)
                    tok = draw_row(LOGITS + tile * 2048, gl.load(TEMP + tile), gl.load(TOPK + tile),
                                     gl.load(TOPP + tile), gl.load(SEED + tile), off0 + (pos - 1), 2048, 2048, SLAY)
            else:
                tok = gl.load(CODES_IN + tile * s_codes_in + pos)
            if pos == 0:
                _embed_row(tile, pos, tok, X, P, CODES, s_codes, SUM, s_sum, TAB0, s_tab0, TP0, s_tp,
                           True, True, 64, E1, E2)
            elif pos < 15:
                _embed_row(tile, pos, tok, X, P, CODES, s_codes, SUM, s_sum,
                           CEMB + (pos - 1) * s_cemb_g, s_cemb, TPROJ + (pos - 1) * s_tproj_g, s_tp,
                           False, True, 64, E1, E2)
            else:
                _embed_row(tile, pos, tok, X, P, CODES, s_codes, SUM, s_sum,
                           CEMB + (pos - 1) * s_cemb_g, s_cemb, TPROJ, s_tp,
                           False, False, 64, E1, E2)
        _signal(CTRS + (c2 & 65535))
        c0 = n0
        c1 = n1
        c2 = n2
        c3 = n3
        c4 = n4
    C.cluster_sync(z1)  # no CTA exits while its partner may still address its shared memory
