"""Ragged (cacheless) attention: the FlashInfer wrapper, the resource that
plans it, and both under CUDA-graph capture.

The load-bearing property throughout: a graph captured once per bucket can be
re-planned before each replay against a different varlen layout, with the
segment count padded out by zero-length segments. Everything runs at
head_dim=72, which FlashInfer has no kernel for, so the pad-to-128 path is
always exercised.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.cuda_graph_config import (
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewisePackedConfig,
)
from mstar.engine.cuda_graph_runner import PiecewiseCudaGraphRunner
from mstar.engine.resources import (
    AttentionStep,
    BucketKey,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    Segment,
    SlotLease,
    StepContext,
    StepRunner,
    SubmoduleStep,
)
from mstar.engine.resources.attn.ragged.flashinfer import FlashInferRaggedManager
from mstar.engine.resources.attn.ragged.wrappers import (
    SUPPORTED_HEAD_DIMS,
    RaggedPrefillWrapper,
    padded_head_dim,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ragged attention requires CUDA"
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
HEADS, HEAD_DIM = 4, 72
WIDTH = HEADS * HEAD_DIM
MAX_SEGMENTS, MAX_TOKENS = 4, 512
TOL = 3e-2                      # bf16 vs an fp32 reference; observed ~8e-3
ATTN = "ragged"


@pytest.fixture(autouse=True)
def _small_workspaces(monkeypatch):
    # WorkspacePool allocates this many MB per (label, slot); the default 512
    # would be ~half a gigabyte per wrapper for tests that build several.
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "64")


# --- helpers ---------------------------------------------------------------

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


def workspace():
    return torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)


def wrapper(**overrides) -> RaggedPrefillWrapper:
    kwargs = dict(
        workspace_buffer=workspace(), num_qo_heads=HEADS, num_kv_heads=HEADS,
        head_dim=HEAD_DIM, device=DEVICE, q_data_type=DTYPE,
    )
    kwargs.update(overrides)
    return RaggedPrefillWrapper(**kwargs)


def graph_wrapper() -> RaggedPrefillWrapper:
    return wrapper(
        max_num_segments=MAX_SEGMENTS, max_total_tokens=MAX_TOKENS, use_cuda_graph=True
    )


# --- head-dim padding ------------------------------------------------------

@pytest.mark.parametrize(
    ("head_dim", "expected"), [(64, 64), (72, 128), (128, 128), (129, 256), (256, 256)]
)
def test_padded_head_dim_rounds_up(head_dim, expected):
    assert padded_head_dim(head_dim) == expected


def test_padded_head_dim_rejects_oversized():
    with pytest.raises(ValueError, match="exceeds the largest supported"):
        padded_head_dim(SUPPORTED_HEAD_DIMS[-1] + 1)


def test_sm_scale_uses_true_head_dim_not_padded():
    """FlashInfer's own default would derive it from the padded dim."""
    w = wrapper()
    assert w.padded_head_dim == 128
    assert w.sm_scale == pytest.approx(HEAD_DIM ** -0.5)


# --- the wrapper, eager ----------------------------------------------------

@pytest.mark.parametrize("seg_lens", [[128, 96, 40], [7, 1, 300]], ids=["mixed", "ragged"])
def test_eager_matches_reference(seg_lens):
    w, layout = wrapper(), cu(seg_lens)
    q, k, v = qkv(sum(seg_lens))
    w.plan(layout)
    close(w.run(q, k, v), ref(q, k, v, layout))
    assert w.num_segments == len(seg_lens)


def test_eager_causal_matches_reference():
    w, layout = wrapper(), cu([128, 96, 40])
    q, k, v = qkv(264)
    w.plan(layout, causal=True)
    close(w.run(q, k, v), ref(q, k, v, layout, causal=True))


def test_eager_accepts_device_cu_seqlens():
    """Works, just costs a sync — encoders build cu_seqlens on device today."""
    w, layout = wrapper(), cu([128, 96, 40])
    q, k, v = qkv(264)
    w.plan(layout.to(DEVICE))
    close(w.run(q, k, v), ref(q, k, v, layout))


# --- the wrapper, under CUDA graph -----------------------------------------

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


def test_graph_replay_stable_across_changing_layouts():
    """Re-planning one captured wrapper repeatedly must not leak state."""
    w = graph_wrapper()
    graph, static, out = capture_wrapper(w)
    layouts = [[128] * 4, [100, 60, 30, 0], [7, 0, 0, 0], [64] * 4]
    for i, seg_lens in enumerate(layouts * 2):
        close(*replay(w, graph, static, out, seg_lens, seed=i))


def test_graph_mode_pads_fewer_segments():
    w = graph_wrapper()
    w.plan(cu([10, 20]))
    assert w.num_segments == 2
    assert w._cu_host.tolist() == [0, 10, 30, 30, 30]


def test_graph_mode_rejects_too_many_segments():
    with pytest.raises(ValueError, match="segments exceeds"):
        graph_wrapper().plan(cu([10] * (MAX_SEGMENTS + 1)))


def test_graph_mode_rejects_too_many_tokens():
    with pytest.raises(ValueError, match="tokens exceeds"):
        graph_wrapper().plan(cu([MAX_TOKENS, 1]))


def test_graph_mode_priming_survives_a_small_first_plan():
    """FlashInfer latches max rows on the first plan; the ctor primes at the
    ceiling so planning small first can't cap the bucket below its own size."""
    w = graph_wrapper()
    w.plan(cu([8] * 4))
    w.plan(cu([MAX_TOKENS // MAX_SEGMENTS] * MAX_SEGMENTS))
    assert w.num_segments == MAX_SEGMENTS


def test_graph_mode_requires_bucket_bounds():
    with pytest.raises(AssertionError, match="max_num_segments required"):
        wrapper(use_cuda_graph=True)


# --- the resource ----------------------------------------------------------

def manager(**config_overrides) -> FlashInferRaggedManager:
    kwargs = dict(num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM)
    kwargs.update(config_overrides)
    spec = RaggedAttentionSpec(
        resource_key=ATTN, nodes={"encoder"}, config=RaggedAttentionConfig(**kwargs),
    )
    return build_resource(
        spec, EngineResourceInfo(device=DEVICE, kv_dtype=DTYPE),
    )


def ctx(rids, lease=None, is_preplan=False) -> StepContext:
    return StepContext(
        request_ids=list(rids), graph_walk="encode", slot=0, capture=False,
        is_preplan=is_preplan, slot_lease=lease,
    )


def bucket(bs: int, num_tokens: int) -> BucketKey:
    return BucketKey(graph_walk="__piecewise__", bs=bs, num_tokens=num_tokens)


def segments(*spans, label="main"):
    return tuple(
        Segment(request_id=f"r{i}", label=label, span=span)
        for i, span in enumerate(spans)
    )


def step(*spans, causal=False, label="main"):
    """``AttentionStep.causal`` defaults to True (the KV-backed common case);
    an encoder tower declares it off, so spell it out here."""
    return AttentionStep(segments=segments(*spans, label=label), causal=causal)


def test_spec_declares_no_dependencies():
    """Nothing is cached, so nothing has to plan before it — unlike the paged
    backend, which reads its layout off a KV plan."""
    assert manager().depends_on() == set()


def test_shards_head_counts_at_build():
    """Head counts are declared pre-sharding; the engine narrows them."""

    class _Group:
        world_size = 2

    spec = RaggedAttentionSpec(
        resource_key=ATTN, nodes={"encoder"},
        config=RaggedAttentionConfig(
            num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM,
        ),
    )
    mgr = build_resource(
        spec,
        EngineResourceInfo(device=DEVICE, kv_dtype=DTYPE, joint_comm_group=_Group()),
    )
    assert mgr._kwargs["num_qo_heads"] == HEADS // 2
    # head_dim never shards
    assert mgr._kwargs["head_dim"] == HEAD_DIM


def test_plan_then_run_matches_reference():
    mgr = manager()
    spans = [128, 96]
    mgr.plan(step(*spans), ctx(["r0", "r1"]))
    q, k, v = qkv(sum(spans))
    close(mgr.run(q, k, v), ref(q, k, v, cu(spans)))
    assert mgr.num_segments() == 2


def test_step_causal_flag_reaches_the_kernel():
    """``AttentionStep.causal`` defaults True for the KV-backed case; a ragged
    step that declares it must actually mask."""
    mgr = manager()
    spans = [128, 96]
    q, k, v = qkv(sum(spans))
    mgr.plan(step(*spans, causal=True), ctx(["r0", "r1"]))
    close(mgr.run(q, k, v), ref(q, k, v, cu(spans), causal=True))


def test_plan_groups_segments_by_label():
    """Two labels in one step are two independent layouts, each with its own
    persistent wrapper."""
    mgr = manager()
    two_labels = AttentionStep(causal=False, segments=(
        Segment("r0", "main", 64), Segment("r1", "main", 32),
        Segment("r0", "other", 16),
    ))
    mgr.plan(two_labels, ctx(["r0", "r1"]))
    assert sorted(mgr._current_plan_states) == ["main", "other"]
    assert mgr.num_segments("main") == 2
    assert mgr.num_segments("other") == 1
    assert mgr._current_plan_states["main"] is not mgr._current_plan_states["other"]


def test_plan_clears_labels_the_step_did_not_declare():
    """A stale wrapper is a stale layout: attending it would silently read the
    previous step's segmentation."""
    mgr = manager()
    mgr.plan(
        AttentionStep(causal=False, segments=(
            Segment("r0", "main", 8), Segment("r0", "other", 8),
        )),
        ctx(["r0"]),
    )
    mgr.plan(step(8), ctx(["r0"]))
    assert list(mgr._current_plan_states) == ["main"]
    with pytest.raises(KeyError, match="no plan for label 'other'"):
        mgr.run(*qkv(8), label="other")


def test_eager_wrappers_are_reused_across_steps():
    """Constructing one allocates FlashInfer's own buffers; a step must not."""
    mgr = manager()
    mgr.plan(step(64), ctx(["r0"]))
    first = mgr._current_plan_states["main"]
    mgr.plan(step(32, 16), ctx(["r0", "r1"]))
    assert mgr._current_plan_states["main"] is first


def test_cg_wrapper_is_sized_by_the_config_not_the_first_plan():
    """The static buffers are fixed for the wrapper's life, so a small first
    plan must not cap the bucket: every later layout would overrun them."""
    mgr = manager(max_segments_per_request=3)
    lease = SlotLease(slot=0, bucket=bucket(bs=2, num_tokens=MAX_TOKENS))
    mgr.plan(step(8), ctx(["r0"], lease=lease))
    w = mgr._current_plan_states["main"]
    assert w.max_num_segments == 2 * 3
    assert w.max_total_tokens == MAX_TOKENS
    assert w.use_cuda_graph


def test_cg_wrapper_is_per_bucket_slot_and_label():
    mgr = manager()
    small = step(8, 8)
    seen = []
    for lease in (
        SlotLease(slot=0, bucket=bucket(2, 256)),
        SlotLease(slot=1, bucket=bucket(2, 256)),
        SlotLease(slot=0, bucket=bucket(2, 512)),
        SlotLease(slot=0, bucket=bucket(2, 256)),  # back to the first
    ):
        mgr.plan(small, ctx(["r0", "r1"], lease=lease))
        seen.append(mgr._current_plan_states["main"])
    assert len({id(w) for w in seen}) == 3
    assert seen[0] is seen[3]


def test_eager_and_captured_wrappers_do_not_share():
    """A captured wrapper's buffers are baked into a graph; an eager re-plan
    through the same object would corrupt them."""
    mgr = manager()
    small = step(8, 8)
    mgr.plan(small, ctx(["r0", "r1"], lease=SlotLease(slot=0, bucket=bucket(2, 256))))
    captured = mgr._current_plan_states["main"]
    mgr.plan(small, ctx(["r0", "r1"]))
    assert mgr._current_plan_states["main"] is not captured


def test_zero_span_padding_rows_attend_nothing():
    """A captured replay pads its batch with zero-length rows; the real rows'
    output must be identical to the same layout without them."""
    mgr = manager()
    spans = [96, 40]
    lease = SlotLease(slot=0, bucket=bucket(bs=4, num_tokens=256))
    q, k, v = qkv(sum(spans))

    mgr.plan(step(*spans, 0, 0), ctx(["r0", "r1", "r2", "r3"], lease=lease))
    assert mgr.num_segments() == 4
    close(mgr.run(q, k, v), ref(q, k, v, cu(spans)))


def test_preplan_promotes_on_the_next_plan():
    """Pre-planning a step ahead parks the wrappers; the real plan promotes
    them rather than re-planning."""
    mgr = manager()
    two = step(64, 32)
    lease = SlotLease(slot=1, bucket=bucket(2, 256))
    mgr.plan(two, ctx(["r0", "r1"], lease=lease, is_preplan=True))
    assert mgr._current_plan_states == {}
    parked = mgr._preplan_states["main"]

    mgr.plan(two, ctx(["r0", "r1"], lease=lease))
    assert mgr._current_plan_states["main"] is parked
    assert mgr._preplan_states == {}
    q, k, v = qkv(96)
    close(mgr.run(q, k, v), ref(q, k, v, cu([64, 32])))


def test_preplan_requires_a_capture_slot():
    """Eager wrappers share one workspace per label with the in-flight forward."""
    mgr = manager()
    with pytest.raises(AssertionError, match="preplan requires a cuda graph step"):
        mgr.plan(step(8), ctx(["r0"], is_preplan=True))


def test_clear_preplan_drops_a_pending_one():
    mgr = manager()
    lease = SlotLease(slot=1, bucket=bucket(1, 256))
    one = step(8)
    mgr.plan(one, ctx(["r0"], lease=lease, is_preplan=True))
    mgr.clear_preplan()
    # not a promotion: the next plan re-plans from scratch
    mgr.plan(one, ctx(["r0"], lease=lease, is_preplan=True))


def test_run_without_a_plan_names_the_label():
    mgr = manager()
    with pytest.raises(KeyError, match="no plan for label 'main'"):
        mgr.run(*qkv(8))


def test_default_label_cursor_resets_each_plan():
    mgr = manager()
    mgr.set_default_label("other")
    mgr.plan(step(8), ctx(["r0"]))
    assert mgr.default_label == "main"


# --- through the piecewise runner ------------------------------------------

PW_TOKENS, PW_BS = 256, 2


def piecewise_runner(mgr, capture_fn):
    resources = {ATTN: mgr}
    step_runner = StepRunner(resources, node_resources={"encoder": [ATTN]})

    def declare_step(request_ids, seq_lens):
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=n)
                for rid, n in zip(request_ids, seq_lens, strict=True)
            ],
            steps={ATTN: AttentionStep(causal=False)},
        )

    runner = PiecewiseCudaGraphRunner(
        label="encoder_block_loop",
        config=PiecewisePackedConfig(
            capture_fn=capture_fn,
            make_static_inputs=lambda shape: {
                "x": torch.zeros(shape.total_tokens, WIDTH, dtype=DTYPE, device=DEVICE)
            },
            declare_step=declare_step,
            total_tokens=[PW_TOKENS],
            capture_batch_sizes=[PW_BS],
        ),
        resources=resources,
        step_runner=step_runner,
        device=DEVICE,
        autocast_dtype=DTYPE,
        node_name="encoder",
    )
    runner.warmup_and_capture()
    assert runner.any_graphs, "capture produced no graphs"
    return runner


def _attend_block(inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
    attn = inp.resources[ATTN]
    x = inp.static_inputs["x"].reshape(-1, HEADS, HEAD_DIM)
    return {"x": attn.run(x, x, x).reshape(-1, WIDTH)}


def test_piecewise_bucket_sizes_its_own_wrapper():
    mgr = manager()
    piecewise_runner(mgr, _attend_block)
    (w,) = mgr._cg_plan_states.values()
    # segment ceiling = bs * max_segments_per_request; token ceiling = bucket
    assert w.max_num_segments == PW_BS
    assert w.max_total_tokens == PW_TOKENS


@pytest.mark.parametrize(
    "seg_lens",
    [[128, 128], [100, 60], [200, 0], [64, 64]],
    ids=["as_captured", "shorter", "one_empty", "small"],
)
def test_piecewise_replay_matches_reference(seg_lens):
    mgr = manager()
    runner = piecewise_runner(mgr, _attend_block)
    total = sum(seg_lens)
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    x = torch.randn(total, WIDTH, dtype=DTYPE, device=DEVICE, generator=gen)

    out = runner.run(
        static_inputs={"x": x},
        request_ids=[f"r{i}" for i in range(len(seg_lens))],
        seq_lens=seg_lens,
    )
    flat = x.reshape(-1, HEADS, HEAD_DIM)
    want = ref(flat, flat, flat, cu(seg_lens)).reshape(-1, WIDTH)
    close(out.get_view("x"), want)


def test_piecewise_replay_stable_across_changing_layouts():
    """One captured bucket, re-planned per replay, must not drift."""
    mgr = manager()
    runner = piecewise_runner(mgr, _attend_block)
    for i, seg_lens in enumerate([[128, 128], [64, 32], [200, 56], [128, 128]]):
        total = sum(seg_lens)
        gen = torch.Generator(device=DEVICE).manual_seed(i)
        x = torch.randn(total, WIDTH, dtype=DTYPE, device=DEVICE, generator=gen)
        out = runner.run(
            static_inputs={"x": x},
            request_ids=[f"r{j}" for j in range(len(seg_lens))],
            seq_lens=seg_lens,
        )
        flat = x.reshape(-1, HEADS, HEAD_DIM)
        want = ref(flat, flat, flat, cu(seg_lens)).reshape(-1, WIDTH)
        close(out.get_view("x"), want)


def test_piecewise_pads_a_short_batch_with_zero_length_rows():
    """One real request into a bs=2 bucket: the padding row contributes a
    zero-length segment and the real output is unaffected."""
    mgr = manager()
    runner = piecewise_runner(mgr, _attend_block)
    gen = torch.Generator(device=DEVICE).manual_seed(7)
    x = torch.randn(150, WIDTH, dtype=DTYPE, device=DEVICE, generator=gen)

    out = runner.run(static_inputs={"x": x}, request_ids=["r0"], seq_lens=[150])
    flat = x.reshape(-1, HEADS, HEAD_DIM)
    want = ref(flat, flat, flat, cu([150])).reshape(-1, WIDTH)
    close(out.get_view("x"), want)
    assert mgr.num_segments() == PW_BS  # the padding row still plans a row


def test_capture_shape_partition_fits_the_bucket():
    """The capture-time plan partitions the token bucket across the batch; the
    wrapper's guards must accept it."""
    shapes = PiecewisePackedConfig(
        capture_fn=_attend_block,
        make_static_inputs=lambda shape: {},
        total_tokens=[PW_TOKENS],
    ).get_capture_shapes([PW_BS])
    (shape,) = shapes
    assert isinstance(shape, PiecewiseCaptureShape)
    assert sum(shape.seq_lens) == PW_TOKENS == shape.total_tokens
    assert len(shape.seq_lens) == PW_BS
