"""Ragged FA3 varlen backend: eager plan/run, causal, CUDA-graph replay, cross.

Skipped when CUDA or ``fa3_fwd_interface.flash_attn_varlen_func`` (AITER on
ROCm) is missing. The FlashInfer ragged suite lives in
``test_ragged_attention.py``; this file only covers ``RaggedAttnBackend.DENSE``.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources import (
    AttentionStep,
    BucketKey,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedAttnBackend,
    RaggedCrossAttentionSpec,
    RaggedCrossAttentionStep,
    Segment,
    SlotLease,
    StepContext,
    cross_label,
)
from mstar.engine.resources.attn.dense import _fa3_unavailable_reason
from mstar.engine.resources.attn.ragged.dense import DenseRaggedCrossManager, DenseRaggedManager
from mstar.engine.resources.attn.ragged.dense_wrappers import (
    DensePrefixedPrefillWrapper,
    DenseRaggedCrossPrefillWrapper,
    DenseRaggedPrefillWrapper,
)
from mstar.engine.resources.attn.ragged.wrappers import zero_unplanned_rows
from mstar.engine.resources.base import EngineResourceInfo, build_resource

_FA3_REASON = _fa3_unavailable_reason()

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="ragged dense attention requires CUDA"),
    pytest.mark.skipif(_FA3_REASON is not None, reason=f"no FA3 varlen kernel: {_FA3_REASON}"),
]

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
HEADS, HEAD_DIM = 4, 72
MAX_SEGMENTS, MAX_TOKENS = 4, 512
TOL = 3e-2
ATTN = "ragged"


def cu(seg_lens: list[int]) -> torch.Tensor:
    out = [0]
    for n in seg_lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32)


def qkv(total: int, seed: int = 0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return tuple(
        torch.randn(total, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE, generator=gen)
        for _ in range(3)
    )


def ref(q, k, v, cu_seqlens, causal=False):
    """Per-segment SDPA at the true head dim, accumulated in fp32."""
    out = torch.zeros_like(q)
    offsets = cu_seqlens.tolist()
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        if end <= start:
            continue
        qs, ks, vs = (t[start:end].transpose(0, 1).unsqueeze(0).float() for t in (q, k, v))
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=HEAD_DIM ** -0.5, is_causal=causal)
        out[start:end] = o.squeeze(0).transpose(0, 1).to(out.dtype)
    return out


def close(got, want, tol=TOL):
    err = (got.float() - want.float()).abs().max().item()
    assert err < tol, f"max_abs_err={err} exceeds {tol}"


def wrapper(**overrides) -> DenseRaggedPrefillWrapper:
    kwargs = dict(
        num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM,
        device=DEVICE, q_data_type=DTYPE,
    )
    kwargs.update(overrides)
    return DenseRaggedPrefillWrapper(**kwargs)


def graph_wrapper() -> DenseRaggedPrefillWrapper:
    return wrapper(
        max_num_segments=MAX_SEGMENTS, max_total_tokens=MAX_TOKENS, accelerator_graph=True,
    )


@pytest.mark.parametrize("seg_lens", [[128, 96, 40], [7, 1, 300]], ids=["mixed", "ragged"])
def test_eager_matches_reference(seg_lens):
    w, layout = wrapper(), cu(seg_lens)
    q, k, v = qkv(sum(seg_lens))
    w.plan(layout)
    close(w.run(q, k, v), ref(q, k, v, layout))
    assert w.num_segments == len(seg_lens)


def test_eager_causal_matches_reference():
    """FA causal is bottom-right; equal q/k lengths match SDPA's top-left."""
    w, layout = wrapper(), cu([128, 96, 40])
    q, k, v = qkv(264)
    w.plan(layout, causal=True)
    close(w.run(q, k, v), ref(q, k, v, layout, causal=True))


def _count_fixed_length_calls(monkeypatch):
    import mstar.engine.resources.attn.ragged.dense_wrappers as dw

    if dw.flash_attn_func is None:
        pytest.skip("no FA3 fixed-length kernel (AITER)")
    calls, fixed = [], dw.flash_attn_func
    monkeypatch.setattr(dw, "flash_attn_func", lambda *a, **kw: calls.append(1) or fixed(*a, **kw))
    return dw, calls


def _varlen(dw, q, k, v, q_len, kv_len, causal, scale):
    out = dw.flash_attn_varlen_func(
        q, k, v, cu([q_len]).to(DEVICE), cu([kv_len]).to(DEVICE), q_len, kv_len, causal=causal, softmax_scale=scale,
    )
    return out[0] if isinstance(out, tuple) else out


# the test file's 4 heads at 72, and Cosmos3's GQA geometry at an even and an odd count of 128-row blocks
@pytest.mark.parametrize("heads,kv_heads,head_dim,q_len,kv_len", [
    (HEADS, HEADS, HEAD_DIM, 300, 300),
    (32, 8, 128, 256, 256),
    (32, 8, 128, 384, 475),
])
def test_one_segment_runs_the_fixed_length_kernel_bit_exactly(monkeypatch, heads, kv_heads, head_dim, q_len, kv_len):
    """A lone non-causal eager segment runs on FA3's fixed-length kernel, with the
    varlen kernel's output bit for bit; rows past the plan keep it on varlen."""
    dw, calls = _count_fixed_length_calls(monkeypatch)
    geometry = dict(num_qo_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim)
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    q = torch.randn(q_len, heads, head_dim, dtype=DTYPE, device=DEVICE, generator=gen)
    k, v = (torch.randn(kv_len, kv_heads, head_dim, dtype=DTYPE, device=DEVICE, generator=gen) for _ in range(2))
    if q_len == kv_len:
        w = wrapper(**geometry)
        w.plan(cu([q_len]))
    else:
        w = cross_wrapper(**geometry)
        w.plan(cu([q_len]), cu([kv_len]))
    got = w.run(q, k, v)
    assert calls, "a one-segment eager plan should take the fixed-length kernel"
    assert torch.equal(got, _varlen(dw, q, k, v, q_len, kv_len, False, w.sm_scale))
    calls.clear()
    pad = torch.zeros(10, heads, head_dim, dtype=DTYPE, device=DEVICE)
    w.run(torch.cat([q, pad]), k, v)  # rows the plan doesn't cover
    assert not calls


def test_causal_and_captured_single_segments_stay_on_varlen(monkeypatch):
    """The fixed-length kernel is slower on short causal segments, and a graph
    replays its bucket's kernel, so neither takes it."""
    _, calls = _count_fixed_length_calls(monkeypatch)
    w, layout = wrapper(), cu([300])
    q, k, v = qkv(300)
    w.plan(layout, causal=True)
    close(w.run(q, k, v), ref(q, k, v, layout, causal=True))
    g = graph_wrapper()
    g.plan(cu([MAX_TOKENS]))
    q, k, v = qkv(MAX_TOKENS)
    g.run(q, k, v)
    assert not calls


def _count_fixed_segment_calls(monkeypatch):
    import mstar.engine.resources.attn.ragged.dense_wrappers as dw

    if dw.fa3_fixed_into is None:
        pytest.skip("no FA3 fixed-length entry point that writes into an output (fa3-fwd 0.0.2)")
    calls, into = [], dw.fa3_fixed_into
    monkeypatch.setattr(dw, "fa3_fixed_into", lambda *a: calls.append(a[0].shape[1]) or into(*a))
    return dw, calls


def _varlen_segments(dw, q, k, v, q_lens, kv_lens, scale):
    out = dw.flash_attn_varlen_func(
        q, k, v, cu(q_lens).to(DEVICE), cu(kv_lens).to(DEVICE), max(q_lens), max(kv_lens),
        causal=False, softmax_scale=scale,
    )
    return out[0] if isinstance(out, tuple) else out


# Cosmos3's guidance pair (one context longer than the other) and a three-request pack,
# through the cross wrapper and the prefixed one Cosmos3's DiT plans, with q / k / v
# as separate tensors or as strided slices of one fused projection
@pytest.mark.parametrize("q_lens,kv_lens", [
    ([1536, 1536], [2436, 1596]),
    ([1600, 2000, 1536], [1700, 2900, 1536]),
])
@pytest.mark.parametrize("cls", [DenseRaggedCrossPrefillWrapper, DensePrefixedPrefillWrapper])
@pytest.mark.parametrize("strided", [False, True])
def test_long_segments_each_run_the_fixed_length_kernel_bit_exactly(monkeypatch, q_lens, kv_lens, cls, strided):
    """Every segment of an eager non-causal plan at least the threshold long runs
    on its own fixed-length call into one output, with the varlen kernel's output
    bit for bit."""
    dw, calls = _count_fixed_segment_calls(monkeypatch)
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    if strided:
        # one [N, 32 + 8 + 8, 128] projection: the rows of q, k and v are 48 heads apart
        rows = max(sum(q_lens), sum(kv_lens))
        fused = torch.randn(rows, 48, 128, dtype=DTYPE, device=DEVICE, generator=gen)
        q, k, v = fused[:sum(q_lens), :32], fused[:sum(kv_lens), 32:40], fused[:sum(kv_lens), 40:]
    else:
        q = torch.randn(sum(q_lens), 32, 128, dtype=DTYPE, device=DEVICE, generator=gen)
        k, v = (torch.randn(sum(kv_lens), 8, 128, dtype=DTYPE, device=DEVICE, generator=gen) for _ in range(2))
    w = cls(num_qo_heads=32, num_kv_heads=8, head_dim=128, device=DEVICE, q_data_type=DTYPE)
    w.plan(cu(q_lens), cu(kv_lens))
    got = w.run(q, k, v)
    assert calls == q_lens
    assert torch.equal(got, _varlen_segments(dw, q, k, v, q_lens, kv_lens, w.sm_scale))


def test_short_or_causal_segments_stay_on_varlen(monkeypatch):
    """One segment under the threshold keeps the whole plan on varlen, where the
    launches would cost more than they save; so does a causal plan."""
    dw, calls = _count_fixed_segment_calls(monkeypatch)
    short = dw._FIXED_SEGMENT_MIN_ROWS - 1
    for q_lens, causal in (([2048, short], False), ([2048, 2048], True)):
        q, k, v = (t.repeat(1, 8, 1) for t in qkv(sum(q_lens)))
        w = wrapper(num_qo_heads=HEADS * 8, num_kv_heads=HEADS * 8)
        w.plan(cu(q_lens), causal=causal)
        w.run(q, k, v)
    assert not calls


def capture_wrapper(w):
    static = tuple(
        torch.zeros(MAX_TOKENS, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) for _ in range(3)
    )
    w.plan(cu([MAX_TOKENS // MAX_SEGMENTS] * MAX_SEGMENTS))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            w.run(*static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = w.run(*static)
    torch.cuda.synchronize()
    return graph, static, out


def replay(w, graph, static, out, seg_lens, seed=0):
    total = sum(seg_lens)
    q, k, v = qkv(total, seed=seed)
    layout = cu(seg_lens)
    w.plan(layout)
    for buf, real in zip(static, (q, k, v), strict=True):
        buf.zero_()
        buf[:total].copy_(real)
    graph.replay()
    torch.cuda.synchronize()
    return out[:total], ref(q, k, v, layout)


@pytest.mark.parametrize(
    "seg_lens",
    [[128] * 4, [100, 60, 30, 20], [100, 60, 30, 0], [190, 20, 0, 0], [7, 0, 0, 0]],
    ids=["as_captured", "shorter", "one_pad", "two_pad", "mostly_pad"],
)
def test_graph_replay_matches_reference(seg_lens):
    w = graph_wrapper()
    close(*replay(w, *capture_wrapper(w), seg_lens))


def test_graph_replay_zeros_rows_past_the_layout():
    """Rows past the planned total read back zero after a replay that left real
    values there. The next layer reads them as its padding rows, and the kernel
    reads padding keys next to the last segment, so they must stay finite."""
    w = graph_wrapper()
    graph, static, out = capture_wrapper(w)
    replay(w, graph, static, out, [128] * 4)
    assert (out != 0).any(dim=(1, 2)).all()
    seg_lens = [100, 60, 30, 0]
    total = sum(seg_lens)
    close(*replay(w, graph, static, out, seg_lens, seed=1))
    assert (out[total:] == 0).all()


@pytest.mark.parametrize("first_zero", [0, 190, 191, MAX_TOKENS], ids=["all", "mid", "unaligned", "none"])
def test_tail_zeroing_overwrites_nan_and_keeps_real_rows(first_zero):
    out = torch.randn(MAX_TOKENS, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    out[first_zero:] = float("nan")
    real = out[:first_zero].clone()
    zero_unplanned_rows(out, torch.tensor([0, first_zero], dtype=torch.int32, device=DEVICE))
    assert torch.equal(out[:first_zero], real)
    assert (out[first_zero:] == 0).all()


def test_graph_mode_pads_fewer_segments():
    w = graph_wrapper()
    w.plan(cu([10, 20]))
    assert w.num_segments == 2
    assert w._qo_indptr_buf.tolist() == [0, 10, 30, 30, 30]


def manager(**config_overrides) -> DenseRaggedManager:
    kwargs = dict(
        num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM,
        backend=RaggedAttnBackend.DENSE,
    )
    kwargs.update(config_overrides)
    spec = RaggedAttentionSpec(
        resource_key=ATTN, nodes={"encoder"}, config=RaggedAttentionConfig(**kwargs),
    )
    mgr = build_resource(spec, EngineResourceInfo(device=DEVICE, kv_dtype=DTYPE))
    assert type(mgr) is DenseRaggedManager
    return mgr


def ctx(rids, lease=None, is_preplan=False) -> StepContext:
    return StepContext(
        request_ids=list(rids), graph_walk="encode", slot=0, capture=False,
        is_preplan=is_preplan, slot_lease=lease,
    )


def segments(*spans, label="main"):
    return tuple(Segment(request_id=f"r{i}", label=label, span=span) for i, span in enumerate(spans))


def step(*spans, causal=False, label="main"):
    return AttentionStep(segments=segments(*spans, label=label), causal=causal)


def test_plan_then_run_matches_reference():
    mgr = manager()
    spans = [128, 96]
    mgr.plan(step(*spans), ctx(["r0", "r1"]))
    q, k, v = qkv(sum(spans))
    close(mgr.run(q, k, v), ref(q, k, v, cu(spans)))
    assert mgr.num_segments() == 2


def test_step_causal_flag_reaches_the_kernel():
    mgr = manager()
    spans = [128, 96]
    q, k, v = qkv(sum(spans))
    mgr.plan(step(*spans, causal=True), ctx(["r0", "r1"]))
    close(mgr.run(q, k, v), ref(q, k, v, cu(spans), causal=True))


def test_manager_builds_captured_wrappers_without_a_workspace():
    """A leased plan builds a graph-mode wrapper; FA3 has no workspace to allocate."""
    mgr = manager()
    before = torch.cuda.memory_allocated(DEVICE)
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="__piecewise__", bs=2, num_tokens=256))
    mgr.plan(step(64, 32), ctx(["r0", "r1"], lease=lease))
    assert mgr._current_plan_states["main"].accelerator_graph
    mgr.plan(step(64, 32), ctx(["r0", "r1"]))
    # a captured and an eager wrapper: layout buffers and one bucket-sized output
    assert torch.cuda.memory_allocated(DEVICE) - before < 1 << 20


def test_preplan_promotes_on_the_next_plan():
    mgr = manager()
    two = step(64, 32)
    lease = SlotLease(slot=1, bucket=BucketKey(graph_walk="__piecewise__", bs=2, num_tokens=256))
    mgr.plan(two, ctx(["r0", "r1"], lease=lease, is_preplan=True))
    parked = mgr._preplan_states["main"]
    mgr.plan(two, ctx(["r0", "r1"], lease=lease))
    assert mgr._current_plan_states["main"] is parked
    q, k, v = qkv(96)
    close(mgr.run(q, k, v), ref(q, k, v, cu([64, 32])))


def cross_ref(q, k, v, q_cu, kv_cu):
    out = torch.zeros_like(q)
    q_off, kv_off = q_cu.tolist(), kv_cu.tolist()
    for qs, qe, ks, ke in zip(q_off[:-1], q_off[1:], kv_off[:-1], kv_off[1:], strict=True):
        if qe <= qs:
            continue
        qq, kk, vv = (t.transpose(0, 1).unsqueeze(0).float() for t in (q[qs:qe], k[ks:ke], v[ks:ke]))
        o = F.scaled_dot_product_attention(qq, kk, vv, scale=HEAD_DIM ** -0.5)
        out[qs:qe] = o.squeeze(0).transpose(0, 1).to(out.dtype)
    return out


def cross_wrapper(**overrides) -> DenseRaggedCrossPrefillWrapper:
    kwargs = dict(
        num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM,
        device=DEVICE, q_data_type=DTYPE,
    )
    kwargs.update(overrides)
    return DenseRaggedCrossPrefillWrapper(**kwargs)


@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [([128, 96], [32, 300]), ([7, 200], [64, 64]), ([50], [1])],
    ids=["longer_keys", "shorter_keys", "single_key"],
)
def test_eager_cross_matches_reference(q_lens, kv_lens):
    w = cross_wrapper()
    q = qkv(sum(q_lens), seed=1)[0]
    _, k, v = qkv(sum(kv_lens), seed=2)
    w.plan(cu(q_lens), cu(kv_lens))
    close(w.run(q, k, v), cross_ref(q, k, v, cu(q_lens), cu(kv_lens)))


@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [
        ([128, 128, 128, 128], [64, 64, 64, 64]),
        ([100, 60, 30, 0], [200, 10, 30, 0]),
        ([7, 0, 0, 0], [300, 0, 0, 0]),
    ],
    ids=["as_captured", "ragged", "mostly_pad"],
)
def test_graph_cross_replay_matches_reference(q_lens, kv_lens):
    w = cross_wrapper(
        max_num_segments=MAX_SEGMENTS, max_total_tokens=MAX_TOKENS, accelerator_graph=True,
    )
    sq, sk, sv = (torch.zeros(MAX_TOKENS, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) for _ in range(3))
    w.plan(cu([MAX_TOKENS // MAX_SEGMENTS] * MAX_SEGMENTS), cu([64] * MAX_SEGMENTS))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            w.run(sq, sk, sv)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = w.run(sq, sk, sv)
    torch.cuda.synchronize()

    q = qkv(sum(q_lens), seed=3)[0]
    _, k, v = qkv(sum(kv_lens), seed=4)
    w.plan(cu(q_lens), cu(kv_lens))
    for buf, real in ((sq, q), (sk, k), (sv, v)):
        buf.zero_()
        buf[: real.shape[0]].copy_(real)
    graph.replay()
    torch.cuda.synchronize()
    close(out[: sum(q_lens)], cross_ref(q, k, v, cu(q_lens), cu(kv_lens)))


def cross_manager() -> DenseRaggedCrossManager:
    spec = RaggedCrossAttentionSpec(
        resource_key="cross", nodes={"dit"},
        config=RaggedAttentionConfig(
            num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM,
            backend=RaggedAttnBackend.DENSE,
        ),
    )
    mgr = build_resource(spec, EngineResourceInfo(device=DEVICE, kv_dtype=DTYPE))
    assert type(mgr) is DenseRaggedCrossManager
    return mgr


def av_segments(video, text, audio):
    segs = []
    for label, spans in (("video", video), ("text", text), ("audio", audio)):
        segs += [Segment(f"r{i}", label, span) for i, span in enumerate(spans)]
    return tuple(segs)


def test_cross_step_plans_each_pair_and_nothing_else():
    mgr = cross_manager()
    step_ = RaggedCrossAttentionStep(
        segments=av_segments([96, 64], [32, 32], [16, 8]),
        pairs=(("video", "text"), ("audio", "video")),
    )
    mgr.plan(step_, ctx(["r0", "r1"]))
    assert sorted(mgr._current_plan_states) == sorted(
        [cross_label("video", "text"), cross_label("audio", "video")]
    )
    with pytest.raises(KeyError, match="no plan for label 'video'"):
        mgr.run(*qkv(160), label="video")

    v, t, a = qkv(160, seed=5), qkv(64, seed=6), qkv(24, seed=7)
    close(
        mgr.run(v[0], t[1], t[2], label=cross_label("video", "text")),
        cross_ref(v[0], t[1], t[2], cu([96, 64]), cu([32, 32])),
    )
    close(
        mgr.run(a[0], v[1], v[2], label=cross_label("audio", "video")),
        cross_ref(a[0], v[1], v[2], cu([16, 8]), cu([96, 64])),
    )
