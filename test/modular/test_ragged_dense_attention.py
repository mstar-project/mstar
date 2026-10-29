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
import triton

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
from mstar.engine.resources.attn.dense import _varlen_unavailable_reason
from mstar.engine.resources.attn.ragged.dense import DenseRaggedCrossManager, DenseRaggedManager
from mstar.engine.resources.attn.ragged.dense_wrappers import (
    _ZERO_BLOCK,
    DenseRaggedCrossPrefillWrapper,
    DenseRaggedPrefillWrapper,
    _zero_rows_from,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource

_FA3_REASON = _varlen_unavailable_reason()

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
    n_valid = torch.tensor(first_zero, dtype=torch.int32, device=DEVICE)
    _zero_rows_from[(triton.cdiv(out.numel(), _ZERO_BLOCK),)](
        out, n_valid, out.numel(), out[0].numel(), BLOCK=_ZERO_BLOCK,
    )
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
