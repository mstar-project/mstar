"""AITER attention managers against a torch reference on an AMD GPU.

Drives the managers like the engine does (KV plan, ``plan``, then ``run`` per
layer, captured and replayed where a bucket would be) over a cache in mstar's
layout, ``[pages, 2, page, Hkv, D]`` with page 0 as the sink. The reference is
fp32 SDPA per request with the causal mask aligned bottom-right, since new
queries are the last positions.

    pytest test/rocm/test_attention_aiter.py
"""

import random

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedCrossAttentionStep,
    cross_label,
)
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import (
    SINK_PAGE,
    KVPlanOutput,
    KVPlanOutputs,
    SequenceView,
    build_paged_indptrs,
)
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AITER kernels run on AMD GPUs"
)

DEV = torch.device("cuda")
# bf16 output vs an fp32 reference. A row attending one key returns that V
# row, so the error scales with the output (bf16 spacing is 2**-7 relative).
ATOL, RTOL = 1e-2, 1.6e-2
NUM_PAGES = 160

MIXES = {
    "prefill": [(0, 37), (0, 300), (0, 129)],
    "extend": [(250, 17), (128, 128), (1000, 5), (63, 66)],
    "decode": [(c, 1) for c in (0, 63, 64, 127, 128, 129, 500, 2047)],
    "mixed": [(0, 50), (300, 1), (77, 1), (500, 40), (0, 1)],
}


def cdiv(a, b):
    return -(-a // b)


class Cache:
    """One layer of a paged cache, and the K/V each request wrote into it."""

    def __init__(self, page, hkv, d, fill=0.0, dtype=torch.bfloat16, seed=0):
        self.page, self.hkv, self.d, self.dtype = page, hkv, d, dtype
        self.layer = torch.full((NUM_PAGES, 2, page, hkv, d), fill, dtype=dtype, device=DEV)
        self.free = list(range(1, NUM_PAGES))
        random.Random(seed).shuffle(self.free)
        self.gen = torch.Generator(device=DEV).manual_seed(seed)

    def request(self, rid, length):
        pages = [self.free.pop() for _ in range(cdiv(length, self.page))]
        k = torch.randn(length, self.hkv, self.d, generator=self.gen, device=DEV).to(self.dtype)
        v = torch.randn(length, self.hkv, self.d, generator=self.gen, device=DEV).to(self.dtype)
        if length:
            pos = torch.arange(length, device=DEV)
            pg = torch.tensor(pages, device=DEV)[pos // self.page]
            self.layer[pg, 0, pos % self.page] = k
            self.layer[pg, 1, pos % self.page] = v
        return pages, k, v


def reference(q, rows, scale, causal=True):
    """``rows``: (q_len, k, v) per request, in packed order."""
    outs, at = [], 0
    for q_len, k, v in rows:
        if q_len == 0:
            continue
        g = q.shape[1] // k.shape[1]
        qi = q[at:at + q_len].float().transpose(0, 1)
        ki = k.float().repeat_interleave(g, dim=1).transpose(0, 1)
        vi = v.float().repeat_interleave(g, dim=1).transpose(0, 1)
        mask = None
        if causal:
            mask = torch.ones(q_len, k.shape[0], dtype=torch.bool, device=DEV).tril(k.shape[0] - q_len)
        outs.append(F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=scale).transpose(0, 1))
        at += q_len
    return torch.cat(outs)


def plan_context(views, lease=None, preplan=False, slot=0, page=128, label="main", kv="kv"):
    ctx = StepContext(
        request_ids=[v.request_id for v in views], graph_walk="w", slot=slot, capture=False,
        is_preplan=preplan, slot_lease=lease,
    )
    ctx.plan_results = {kv: KVPlanOutputs(
        {label: KVPlanOutput(cpu_indptrs=build_paged_indptrs(views, page), views=views)},
    )}
    return ctx


def paged_manager(page, hq, hkv, d, dtype=torch.bfloat16):
    from mstar.engine.resources.attn.aiter import AiterPagedAttentionManager

    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=hkv, head_dim=d, num_qo_heads=hq,
        max_seq_len=4096, max_num_pages=NUM_PAGES, page_size=page,
    )
    return AiterPagedAttentionManager("kv", DEV, dtype, cfg)


def layout(cache, mix):
    """Write each request's K/V; return its views, reference rows and q tokens."""
    views, rows = [], []
    for i, (ctx_len, q_len) in enumerate(mix):
        pages, k, v = cache.request(f"r{i}", ctx_len + q_len)
        views.append(SequenceView(f"r{i}", "main", pages, ctx_len + q_len, q_len))
        rows.append((q_len, k, v))
    return views, rows


def max_err(out, ref):
    """Largest error relative to the tolerance, and the count of non-finite
    outputs. Below 1 means every element is within ``ATOL + RTOL * |ref|``."""
    o = out[: ref.shape[0]].float()
    scaled = ((o - ref).abs() / (ATOL + RTOL * ref.abs())).nan_to_num(nan=float("inf"))
    return scaled.max().item(), int((~torch.isfinite(o)).sum())


# --- parity -------------------------------------------------------------------


@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("group", [1, 2, 4, 8])
@pytest.mark.parametrize("mix", list(MIXES))
def test_paged_parity(page, d, group, mix):
    hkv = 4
    cache = Cache(page, hkv, d)
    views, rows = layout(cache, MIXES[mix])
    m = paged_manager(page, hkv * group, hkv, d)
    m.plan(AttentionStep(), plan_context(views, page=page))
    q = torch.randn(sum(r[0] for r in rows), hkv * group, d, device=DEV).to(torch.bfloat16)
    err, nonfinite = max_err(m.run(q, "main", cache.layer), reference(q, rows, d ** -0.5))
    assert nonfinite == 0 and err < 1, (err, nonfinite)


def test_paged_parity_fp16_and_odd_geometry():
    for d, hq, hkv in ((128, 24, 8), (80, 16, 4), (96, 16, 4)):
        cache = Cache(128, hkv, d, dtype=torch.float16)
        views, rows = layout(cache, MIXES["mixed"])
        m = paged_manager(128, hq, hkv, d, dtype=torch.float16)
        m.plan(AttentionStep(), plan_context(views))
        q = torch.randn(sum(r[0] for r in rows), hq, d, device=DEV).to(torch.float16)
        err, nonfinite = max_err(m.run(q, "main", cache.layer), reference(q, rows, d ** -0.5))
        assert nonfinite == 0 and err < 1, (d, err, nonfinite)


def test_dense_aiter_varlen_aligns_causal_bottom_right():
    """``attn/dense.py`` passes causal walks to AITER's varlen kernel with new
    queries after a frozen prefix, which needs the bottom-right causal mask."""
    from aiter import flash_attn_varlen_func

    gen = torch.Generator(device=DEV).manual_seed(0)
    lens = [(4, 260), (1, 517), (128, 128)]
    q = torch.randn(sum(a for a, _ in lens), 32, 128, generator=gen, device=DEV).to(torch.bfloat16)
    kv = [(torch.randn(b, 8, 128, generator=gen, device=DEV).to(torch.bfloat16),
           torch.randn(b, 8, 128, generator=gen, device=DEV).to(torch.bfloat16)) for _, b in lens]
    cu_q = torch.tensor([0, 4, 5, 133], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, 260, 777, 905], dtype=torch.int32, device=DEV)
    out = flash_attn_varlen_func(
        q, torch.cat([k for k, _ in kv]), torch.cat([v for _, v in kv]), cu_q, cu_k, 128, 517, causal=True,
    )
    out = out[0] if isinstance(out, tuple) else out
    ref = reference(q, [(a, k, v) for (a, _), (k, v) in zip(lens, kv, strict=True)], 128 ** -0.5)
    err, nonfinite = max_err(out, ref)
    assert nonfinite == 0 and err < 1, (err, nonfinite)


# --- page tails left non-finite by an earlier owner --------------------------


@pytest.mark.parametrize("mix", list(MIXES))
def test_page_128_masks_non_finite_page_tails(mix):
    cache = Cache(128, 8, 128, fill=float("nan"))
    views, rows = layout(cache, MIXES[mix])
    m = paged_manager(128, 32, 8, 128)
    m.plan(AttentionStep(), plan_context(views))
    q = torch.randn(sum(r[0] for r in rows), 32, 128, device=DEV).to(torch.bfloat16)
    err, nonfinite = max_err(m.run(q, "main", cache.layer), reference(q, rows, 128 ** -0.5))
    assert nonfinite == 0 and err < 1, (err, nonfinite)


def test_page_64_reads_page_tails_unmasked_so_auto_avoids_it():
    """At page 64 the kernel's tile equals the page and its V load is unmasked.
    ``nan_safe_page_size`` steers AUTO away from this case. If the assertion
    below starts failing, AITER masks the tail now and the rule can go."""
    from mstar.engine.resources.attn.aiter import nan_safe_page_size

    assert not nan_safe_page_size(64) and nan_safe_page_size(128)
    cache = Cache(64, 8, 128, fill=float("nan"))
    views, rows = layout(cache, MIXES["decode"])
    m = paged_manager(64, 32, 8, 128)
    m.plan(AttentionStep(), plan_context(views, page=64))
    q = torch.randn(len(rows), 32, 128, device=DEV).to(torch.bfloat16)
    _, nonfinite = max_err(m.run(q, "main", cache.layer), reference(q, rows, 128 ** -0.5))
    assert nonfinite > 0


# --- capture and replay -------------------------------------------------------


def _capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("kind", ["decode", "prefill"])
def test_capture_once_replay_new_lengths(page, kind):
    hq, hkv, d = 32, 8, 128
    bs, tokens = (8, 8) if kind == "decode" else (4, 512)
    m = paged_manager(page, hq, hkv, d)
    lease = SlotLease(slot=0, bucket=BucketKey("w", bs, tokens))
    q_buf = torch.zeros(tokens, hq, d, dtype=torch.bfloat16, device=DEV)

    def pad(views):
        span = 1 if kind == "decode" else 0
        return views + [
            SequenceView(f"pad{i}", "main", [SINK_PAGE] * cdiv(span, page), span, span)
            for i in range(bs - len(views))
        ]

    capture_views = pad([SequenceView("c0", "main", [1], 1, 1)] if kind == "decode"
                        else [SequenceView("c0", "main", [1, 2, 3, 4, 5, 6, 7, 8], 512, 512)])
    m.plan(AttentionStep(), plan_context(capture_views, lease, page=page))
    cache = Cache(page, hkv, d)
    graph, out = _capture(lambda: m.run(q_buf, "main", cache.layer))

    if kind == "decode":
        replays = [[(c, 1) for c in (5, 100, 1000, 3000, 64, 127, 128, 4000)],
                   [(c, 1) for c in (2000, 0, 63, 257, 33)], [(1500, 1)]]
    else:
        replays = [[(0, 200), (100, 50), (0, 200), (1000, 62)], [(0, 100), (40, 7)], [(2000, 300)]]
    for i, mix in enumerate(replays):
        cache.free = list(range(1, NUM_PAGES))
        random.Random(i).shuffle(cache.free)
        views, rows = layout(cache, mix)
        m.plan(AttentionStep(), plan_context(pad(views), lease, page=page))
        n = sum(r[0] for r in rows)
        q = torch.randn(n, hq, d, device=DEV).to(torch.bfloat16)
        q_buf[:n].copy_(q)
        graph.replay()
        torch.cuda.synchronize()
        err, nonfinite = max_err(out, reference(q, rows, d ** -0.5))
        assert nonfinite == 0 and err < 1, (i, err, nonfinite)


def test_preplanned_slot_replays_while_the_live_one_is_untouched():
    """Preplan stages step N+1 on the other slot, and promotion hands it to ``run``."""
    hq, hkv, d, page = 16, 4, 128, 128
    m = paged_manager(page, hq, hkv, d)
    cache = Cache(page, hkv, d)
    leases = [SlotLease(slot=s, bucket=BucketKey("w", 4, 4)) for s in (0, 1)]
    q_buf = torch.zeros(4, hq, d, dtype=torch.bfloat16, device=DEV)
    graphs = []
    for lease in leases:
        views, _ = layout(cache, [(10, 1)] * 4)
        m.plan(AttentionStep(), plan_context(views, lease))
        graphs.append(_capture(lambda: m.run(q_buf, "main", cache.layer)))
    views_a, rows_a = layout(cache, [(c, 1) for c in (7, 300, 129, 1)])
    m.plan(AttentionStep(), plan_context(views_a, leases[0]))
    views_b, rows_b = layout(cache, [(c, 1) for c in (900, 2, 64, 513)])
    m.plan(AttentionStep(), plan_context(views_b, leases[1], preplan=True))
    q = torch.randn(4, hq, d, device=DEV).to(torch.bfloat16)
    q_buf.copy_(q)
    graphs[0][0].replay()
    torch.cuda.synchronize()
    assert max_err(graphs[0][1], reference(q, rows_a, d ** -0.5))[0] < 1
    m.plan(AttentionStep(), plan_context(views_b, leases[1]))  # promote
    graphs[1][0].replay()
    torch.cuda.synchronize()
    assert max_err(graphs[1][1], reference(q, rows_b, d ** -0.5))[0] < 1


# --- non-causal ----------------------------------------------------------------


def test_non_causal_single_query_rows_run_on_aiter():
    cache = Cache(128, 8, 128)
    views, rows = layout(cache, [(c, 1) for c in (0, 77, 500)])
    m = paged_manager(128, 32, 8, 128)
    m.plan(AttentionStep(causal=False), plan_context(views))
    assert m._fallback is None
    q = torch.randn(3, 32, 128, device=DEV).to(torch.bfloat16)
    err, nonfinite = max_err(m.run(q, "main", cache.layer), reference(q, rows, 128 ** -0.5, causal=False))
    assert nonfinite == 0 and err < 1


def test_non_causal_multi_query_rows_go_to_flashinfer():
    from mstar.engine.resources.attn.aiter import flashinfer_rocm_unavailable_reason

    if flashinfer_rocm_unavailable_reason() is not None:
        pytest.skip("AMD's FlashInfer build is not installed")
    cache = Cache(128, 8, 128)
    views, rows = layout(cache, [(100, 60), (0, 33), (300, 7)])
    m = paged_manager(128, 32, 8, 128)
    m.plan(AttentionStep(causal=False), plan_context(views))
    assert m._fallback is not None
    q = torch.randn(100, 32, 128, device=DEV).to(torch.bfloat16)
    err, nonfinite = max_err(m.run(q, "main", cache.layer), reference(q, rows, 128 ** -0.5, causal=False))
    assert nonfinite == 0 and err < 1, (err, nonfinite)


def _cross_manager(hq, d):
    from mstar.engine.resources.attn.aiter import AiterCrossManager

    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=hq, head_dim=d, num_qo_heads=hq,
        max_seq_len=1500, max_num_pages=NUM_PAGES, page_size=128,
    )
    return AiterCrossManager("ctx", "kv", "context", DEV, torch.bfloat16, cfg)


def _cross_context(q_spans, ctx_pages, ctx_lens, lease=None):
    views = [SequenceView(f"r{i}", "main", [SINK_PAGE], n, n) for i, n in enumerate(q_spans)]
    ctx = plan_context(views, lease)
    ctx_views = [
        SequenceView(f"r{i}", "context", pages, length, 0)
        for i, (pages, length) in enumerate(zip(ctx_pages, ctx_lens, strict=True))
    ]
    ctx.plan_results["ctx"] = KVPlanOutputs(
        {"context": KVPlanOutput(cpu_indptrs=build_paged_indptrs(ctx_views, 128), views=ctx_views)},
    )
    return ctx


@pytest.mark.parametrize("q_spans", [[1, 1, 1], [4, 1, 3]])
def test_cross_attention_parity(q_spans):
    hq, d = 20, 64  # whisper-large's decoder
    cache = Cache(128, hq, d)
    lens = [1500, 700, 64]
    written = [cache.request(f"r{i}", n) for i, n in enumerate(lens)]
    m = _cross_manager(hq, d)
    m.plan(AttentionStep(causal=False), _cross_context(q_spans, [w[0] for w in written], lens))
    q = torch.randn(sum(q_spans), hq, d, device=DEV).to(torch.bfloat16)
    rows = [(n, k, v) for n, (_, k, v) in zip(q_spans, written, strict=True)]
    err, nonfinite = max_err(m.run(q, "main", cache.layer), reference(q, rows, d ** -0.5, causal=False))
    assert nonfinite == 0 and err < 1, (err, nonfinite)


def test_cross_attention_captured_decode():
    hq, d, bs = 20, 64, 4
    cache = Cache(128, hq, d)
    m = _cross_manager(hq, d)
    lease = SlotLease(slot=0, bucket=BucketKey("decode", bs, bs))
    first = [cache.request(f"c{i}", 10) for i in range(bs)]
    m.plan(AttentionStep(causal=False), _cross_context([1] * bs, [w[0] for w in first], [10] * bs, lease))
    q_buf = torch.zeros(bs, hq, d, dtype=torch.bfloat16, device=DEV)
    graph, out = _capture(lambda: m.run(q_buf, "main", cache.layer))
    lens = [1500, 3, 700, 129]
    written = [cache.request(f"r{i}", n) for i, n in enumerate(lens)]
    m.plan(AttentionStep(causal=False), _cross_context([1] * bs, [w[0] for w in written], lens, lease))
    q = torch.randn(bs, hq, d, device=DEV).to(torch.bfloat16)
    q_buf.copy_(q)
    graph.replay()
    torch.cuda.synchronize()
    rows = [(1, k, v) for _, k, v in written]
    err, nonfinite = max_err(out, reference(q, rows, d ** -0.5, causal=False))
    assert nonfinite == 0 and err < 1, (err, nonfinite)


# --- ragged (cacheless) ---------------------------------------------------------


def _ragged(cls, d, hq=16, hkv=16, max_segments=2):
    cfg = RaggedAttentionConfig(num_qo_heads=hq, num_kv_heads=hkv, head_dim=d, max_segments_per_request=max_segments)
    return cls(device=DEV, dtype=torch.bfloat16, config=cfg)


def _segments(label, spans):
    return tuple(Segment(f"r{i}", label, n) for i, n in enumerate(spans))


@pytest.mark.parametrize("d", [64, 72, 80, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_ragged_parity(d, causal):
    from mstar.engine.resources.attn.ragged.aiter import AiterRaggedManager

    spans = [300, 17, 64]
    m = _ragged(AiterRaggedManager, d)
    m.plan(AttentionStep(segments=_segments("img", spans), causal=causal),
           StepContext(request_ids=["r0", "r1", "r2"], graph_walk="w", slot=0, capture=False))
    n = sum(spans)
    q, k, v = (torch.randn(n, 16, d, device=DEV).to(torch.bfloat16) for _ in range(3))
    cu = [0, 300, 317, 381]
    rows = [(spans[i], k[cu[i]:cu[i + 1]], v[cu[i]:cu[i + 1]]) for i in range(3)]
    err, nonfinite = max_err(m.run(q, k, v, "img"), reference(q, rows, d ** -0.5, causal=causal))
    assert nonfinite == 0 and err < 1, (err, nonfinite)


@pytest.mark.parametrize("d", [72, 128])
def test_ragged_captured_with_empty_padding_segments(d):
    from mstar.engine.resources.attn.ragged.aiter import AiterRaggedManager

    m = _ragged(AiterRaggedManager, d)
    lease = SlotLease(slot=0, bucket=BucketKey("w", 2, 512))
    m.plan(AttentionStep(segments=_segments("img", [256, 256]), causal=False),
           StepContext(request_ids=["r0", "r1"], graph_walk="w", slot=0, capture=False, slot_lease=lease))
    bufs = [torch.zeros(512, 16, d, dtype=torch.bfloat16, device=DEV) for _ in range(3)]
    graph, out = _capture(lambda: m.run(*bufs, "img"))
    for spans in ([100, 37], [500, 12], [7]):
        m.plan(AttentionStep(segments=_segments("img", spans), causal=False),
               StepContext(request_ids=[f"r{i}" for i in range(len(spans))], graph_walk="w",
                           slot=0, capture=False, slot_lease=lease))
        n = sum(spans)
        q, k, v = (torch.randn(n, 16, d, device=DEV).to(torch.bfloat16) for _ in range(3))
        for buf, t in zip(bufs, (q, k, v), strict=True):
            buf[:n].copy_(t)
        graph.replay()
        torch.cuda.synchronize()
        cu = [0]
        for s in spans:
            cu.append(cu[-1] + s)
        rows = [(spans[i], k[cu[i]:cu[i + 1]], v[cu[i]:cu[i + 1]]) for i in range(len(spans))]
        err, nonfinite = max_err(out, reference(q, rows, d ** -0.5, causal=False))
        assert nonfinite == 0 and err < 1, (spans, err, nonfinite)


@pytest.mark.parametrize("d", [64, 72])
def test_ragged_one_request_with_several_segments_replays(d):
    """One request a replay with a segment per image or frame group, like
    Qwen3.5's vision block loop at batch size 1. Captured over one span, as the
    piecewise runner does, then replayed with up to ``max_segments_per_request``."""
    from mstar.engine.resources.attn.ragged.aiter import AiterRaggedManager

    m = _ragged(AiterRaggedManager, d, hq=12, hkv=12, max_segments=4)
    lease = SlotLease(slot=0, bucket=BucketKey("w", 1, 1024))

    def plan(spans):
        m.plan(AttentionStep(segments=tuple(Segment("r0", "main", n) for n in spans), causal=False),
               StepContext(request_ids=["r0"], graph_walk="w", slot=0, capture=False, slot_lease=lease))

    plan([1024])
    bufs = [torch.zeros(1024, 12, d, dtype=torch.bfloat16, device=DEV) for _ in range(3)]
    graph, out = _capture(lambda: m.run(*bufs))
    for spans in ([256, 64, 16], [256, 256, 256, 256], [100, 0, 400, 7], [1000]):
        plan(spans)
        n = sum(spans)
        q, k, v = (torch.randn(n, 12, d, device=DEV).to(torch.bfloat16) for _ in range(3))
        for buf, t in zip(bufs, (q, k, v), strict=True):
            buf[:n].copy_(t)
        graph.replay()
        torch.cuda.synchronize()
        cu = [0]
        for s in spans:
            cu.append(cu[-1] + s)
        rows = [(spans[i], k[cu[i]:cu[i + 1]], v[cu[i]:cu[i + 1]]) for i in range(len(spans))]
        err, nonfinite = max_err(out, reference(q, rows, d ** -0.5, causal=False))
        assert nonfinite == 0 and err < 1, (spans, err, nonfinite)
    with pytest.raises(ValueError, match="5 segments"):
        plan([16] * 5)


def test_ragged_cross_parity():
    from mstar.engine.resources.attn.ragged.aiter import AiterRaggedCrossManager

    m = _ragged(AiterRaggedCrossManager, 128, hq=16, hkv=8)
    q_spans, kv_spans = [40, 9], [77, 300]
    segments = _segments("vid", q_spans) + _segments("txt", kv_spans)
    m.plan(RaggedCrossAttentionStep(segments=segments, pairs=(("vid", "txt"),)),
           StepContext(request_ids=["r0", "r1"], graph_walk="w", slot=0, capture=False))
    q = torch.randn(sum(q_spans), 16, 128, device=DEV).to(torch.bfloat16)
    k = torch.randn(sum(kv_spans), 8, 128, device=DEV).to(torch.bfloat16)
    v = torch.randn(sum(kv_spans), 8, 128, device=DEV).to(torch.bfloat16)
    rows = [(40, k[:77], v[:77]), (9, k[77:], v[77:])]
    out = m.run(q, k, v, cross_label("vid", "txt"))
    err, nonfinite = max_err(out, reference(q, rows, 128 ** -0.5, causal=False))
    assert nonfinite == 0 and err < 1, (err, nonfinite)
