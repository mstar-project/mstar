"""The KDA linear-attention resource: built off the pool, planned against it.

What the manager owns is the plan: which rows run, which walk they take, and
the token layout its kernel bundle asked for, staged where a captured graph
will read it. The math is the bundle's, so a recording bundle stands in here;
``KDACallable`` hands the bundle each layer's blocks. The torch reference bundle
runs at the end, and once on CUDA to show it never waits on the host.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.convenience import KDACallable
from mstar.engine.resources.linear_attn.config import (
    LinearAttnBackend,
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.linear_attn.kda import KDAManager, KDAParams, SpecBlocks
from mstar.engine.resources.linear_attn.kda_torch import TorchKDAKernels
from mstar.engine.resources.linear_attn.kda_triton import (
    TritonKDAKernels,
    _merged_row,
    varlen_layout,
)
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import SINK_SLOT, RecurrentStatePool
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

POOL, KDA = "kda_state", "kda"
HEADS, DIM = 2, 16
CPU = torch.device("cpu")


class _Recorder:
    """A bundle that asks for its spans back as the layout and records runs."""

    def __init__(self):
        self.layouts: list[tuple] = []
        self.runs: list[tuple] = []
        self.verifies: list[tuple] = []

    def layout(self, spans, num_tokens, fixed):
        self.layouts.append((spans, num_tokens, fixed))
        return [num_tokens, *spans]

    def run_paged(self, qkv, g, beta, plan, conv_state, rec_state, p, gate=None):
        self.runs.append((plan, conv_state, rec_state, gate))
        return qkv

    def run_verify(self, qkv, g, beta, plan, conv_state, rec_state, spec, p, gate=None):
        self.verifies.append((plan, spec, gate))
        return qkv


def specs(heads=HEADS, v_heads=HEADS, backend=LinearAttnBackend.TRITON, speculative_tokens=0,
          sink=True):
    geometry = DeltaNetGeometry(
        num_k_heads=heads, num_v_heads=v_heads, head_k_dim=DIM, head_v_dim=DIM,
        conv_kernel_size=4,
    )
    pool = RecurrentStateSpec(POOL, {"llm"}, RecurrentStateConfig(
        num_layers=3, blocks=geometry.to_blocks(speculative_tokens=speculative_tokens),
        max_slots=6, disable_sink_slot=not sink,
    ))
    kda = LinearAttnSpec(KDA, {"llm"}, LinearAttnConfig(
        recurrent_state=POOL, variant=LinearAttnVariant.KDA, backend=backend,
        gate_lower_bound=-5.0,
    ))
    return pool, kda


def build(device=CPU, **kwargs) -> tuple[RecurrentStatePool, KDAManager]:
    pool_spec, kda_spec = specs(**kwargs)
    pool = build_resource(pool_spec, EngineResourceInfo(device=device))
    kda = build_resource(kda_spec, EngineResourceInfo(
        device=device, dependencies={POOL: pool_spec},
    ))
    return pool, kda


def plan(pool, kda, rids, spans, ctx, speculative=False):
    segments = tuple(Segment(rid, "main", n) for rid, n in zip(rids, spans, strict=True))
    for rid in rids:
        pool.ingest_request(rid)
    pool_step = RecurrentStep(segments=segments)
    assert pool.admit(pool_step, ctx).ok
    ctx.plan_results[POOL] = pool.plan(pool_step, ctx)
    return kda.plan(LinearAttnStep(segments=segments, speculative=speculative), ctx)["main"]


def commit(pool, rids, spans, ctx):
    """The pool's commit of the step just planned: its rows' slots now have state."""
    segments = tuple(Segment(rid, "main", n) for rid, n in zip(rids, spans, strict=True))
    pool.commit(RecurrentStep(segments=segments), ctx)


def params(h=HEADS, d=DIM):
    return KDAParams(
        conv_weight=torch.zeros(3 * h * d, 4), A_log=torch.zeros(h), dt_bias=torch.zeros(h * d),
        lower_bound=-5.0, num_heads=h, head_dim=d, scale=d ** -0.5,
    )


def eager(rids, walk="prefill"):
    return StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)


def leased(real, padded, bs, tokens, walk="prefill", slot=0):
    ctx = StepContext(
        request_ids=tuple(real), graph_walk=walk, slot=slot, capture=False,
        slot_lease=SlotLease(slot=slot, bucket=BucketKey(walk, bs, tokens)),
    )
    ctx.set_padded_rids(tuple(padded))
    return ctx


def test_builds_off_the_pool():
    pool, kda = build()
    assert isinstance(kda, KDAManager) and kda.depends_on() == {POOL}
    assert kda.geometry.num_v_heads == HEADS and kda.geometry.head_k_dim == DIM
    # no CUDA: the torch reference
    assert isinstance(kda.kernels, TorchKDAKernels)
    # the config's GDN default is taken as the one KDA backend there is
    _, kda = build(backend=LinearAttnBackend.FLASHINFER)
    assert kda.config.backend is LinearAttnBackend.TRITON


def test_refuses_a_pool_it_cannot_run():
    with pytest.raises(ValueError, match="one head count"):
        build(heads=1, v_heads=2)


def test_refuses_a_pool_without_a_sink():
    # padding rows address the sink; without one they would land on a live slot
    with pytest.raises(ValueError, match="sink slot"):
        build(sink=False)


def test_the_torch_reference_at_an_unvalidated_head_dim(monkeypatch):
    from mstar.engine.resources.linear_attn import kda as kda_mod

    monkeypatch.setattr(kda_mod.kda_triton, "_HAS_TRITON", True)
    monkeypatch.setattr(kda_mod.kda_triton, "TritonKDAKernels", lambda: "bundle")
    cuda = torch.device("cuda")
    assert kda_mod._default_kernels(cuda, 128) == "bundle"
    assert isinstance(kda_mod._default_kernels(cuda, 16), TorchKDAKernels)


def test_prefill_plan_stages_the_bundles_layout():
    pool, kda = build()
    kda.set_kernels(rec := _Recorder())
    p = plan(pool, kda, ["a", "b"], [5, 3], eager(["a", "b"]))
    assert not p.is_decode and p.num_tokens == 8 and p.num_rows == 2
    assert rec.layouts == [((5, 3), 8, False)]
    assert p.layout.tolist() == [8, 5, 3]
    slots = [pool._slots[r]["main"].index for r in "ab"]
    assert p.slot_ids.tolist() == slots and SINK_SLOT not in slots


def test_decode_plan_needs_no_layout():
    pool, kda = build()
    kda.set_kernels(rec := _Recorder())
    p = plan(pool, kda, ["a", "b"], [1, 1], eager(["a", "b"], walk="decode"))
    assert p.is_decode and p.layout is None and rec.layouts == []


def test_a_captured_prefill_keeps_one_layout_buffer():
    """Replays of one bucket fill the same buffer, sized by the bucket; the
    padding row is on the sink; one-token rows in a prefill bucket are still
    a prefill, since the graph replays the prefill kernels."""
    pool, kda = build()
    kda.set_kernels(rec := _Recorder())
    first = plan(pool, kda, ["a", "pad"], [5, 0], leased(["a"], ["a", "pad"], bs=2, tokens=64))
    assert first.slot_ids.tolist()[1] == SINK_SLOT
    second = plan(pool, kda, ["b", "c"], [1, 1], leased(["b", "c"], ["b", "c"], bs=2, tokens=64))
    assert not first.is_decode and not second.is_decode
    assert first.num_tokens == second.num_tokens == 64
    assert rec.layouts == [((5, 0), 64, True), ((1, 1), 64, True)]
    assert first.layout.data_ptr() == second.layout.data_ptr()
    assert second.layout.tolist() == [64, 1, 1]
    # another capture slot of the same bucket has its own buffer
    third = plan(pool, kda, ["d"], [2], leased(["d"], ["d"], bs=1, tokens=64, slot=1))
    assert third.layout.data_ptr() != first.layout.data_ptr()


def test_a_prefill_bucket_of_one_token_rows_is_still_a_prefill():
    """A prefill bucket with as many tokens as rows captured one-token spans, planned as a
    decode: the graph then replayed the decode kernels over a step's uneven spans, each
    token on another request's slot."""
    pool, kda = build()
    kda.set_kernels(_Recorder())
    capture = plan(pool, kda, ["a", "b"], [1, 1], leased(["a", "b"], ["a", "b"], bs=2, tokens=2))
    replay = plan(pool, kda, ["c", "pad"], [2, 0], leased(["c"], ["c", "pad"], bs=2, tokens=2))
    assert not capture.is_decode and not replay.is_decode
    # a decode bucket stays a decode
    decode = plan(pool, kda, ["d", "e"], [1, 1],
                  leased(["d", "e"], ["d", "e"], bs=2, tokens=2, walk="decode", slot=1))
    assert decode.is_decode


def test_preplan_promotes_the_staged_plan():
    pool, kda = build()
    kda.set_kernels(_Recorder())
    ctx = eager(["a"], walk="decode")
    ctx.is_preplan = True
    staged = plan(pool, kda, ["a"], [1], ctx)
    ctx.is_preplan = False
    promoted = kda.plan(LinearAttnStep(segments=(Segment("a", "main", 1),)), ctx)["main"]
    assert promoted is staged
    with pytest.raises(KeyError, match="no plan for label 'draft'"):
        kda.current_plan("draft")


def test_run_hands_the_bundle_this_steps_plan():
    pool, kda = build()
    kda.set_kernels(rec := _Recorder())
    p = plan(pool, kda, ["a"], [4], eager(["a"]))
    params = KDAParams(
        conv_weight=torch.zeros(3 * HEADS * DIM, 4), A_log=torch.zeros(HEADS),
        dt_bias=torch.zeros(HEADS * DIM), lower_bound=-5.0, num_heads=HEADS,
        head_dim=DIM, scale=DIM ** -0.5,
    )
    conv, state = pool.block("conv", 1), pool.block("state", 1)
    gate = torch.zeros(4, DIM)
    kda.run(torch.zeros(4, 3 * HEADS * DIM), torch.zeros(4, DIM), torch.zeros(4, HEADS),
            conv, state, params, gate=gate)
    ((got, got_conv, got_state, got_gate),) = rec.runs
    assert got is p and got_conv is conv and got_state is state and got_gate is gate


def _ptr(t):
    return t.data_ptr(), tuple(t.shape)


def test_the_callable_hands_the_bundle_each_layers_blocks():
    """The layer names no block: the callable resolves its conv and state, by the cursor or
    an explicit index, and a pool without speculative blocks has none to hand over."""
    pool, kda = build()
    kda.set_kernels(rec := _Recorder())
    p = plan(pool, kda, ["a"], [4], eager(["a"]))
    mix = KDACallable(pool, kda)
    assert mix.conv_dtype == pool.block("conv", 0).dtype
    assert mix.current_plan() is p
    gate = torch.zeros(4, DIM)
    args = (torch.zeros(4, 3 * HEADS * DIM), torch.zeros(4, DIM), torch.zeros(4, HEADS), params())
    mix(*args, gate=gate, layer_idx=2)
    mix.set_layer_idx(1)
    mix(*args, gate=gate)
    for (got, conv, state, got_gate), layer in zip(rec.runs, (2, 1), strict=True):
        assert got is p and got_gate is gate
        assert _ptr(conv) == _ptr(pool.block("conv", layer))
        assert _ptr(state) == _ptr(pool.block("state", layer))
    assert mix._blocks[1].spec is None


def test_a_speculative_step_runs_verify_blocks():
    pool, kda = build(speculative_tokens=2)
    assert kda.speculative_tokens == 2
    kda.set_kernels(rec := _Recorder())
    mix = KDACallable(pool, kda)
    # each request's prefill commits before its first verify step
    ctx = eager(["a", "b"])
    plan(pool, kda, ["a", "b"], [4, 2], ctx)
    commit(pool, ["a", "b"], [4, 2], ctx)
    layouts = len(rec.layouts)
    p = plan(pool, kda, ["a", "b"], [3, 3], eager(["a", "b"], walk="decode"), speculative=True)
    assert p.is_verify and p.block == 3 and not p.is_decode
    assert p.layout is None and len(rec.layouts) == layouts
    gate = torch.zeros(6, DIM)
    mix(torch.zeros(6, 3 * HEADS * DIM), torch.zeros(6, DIM), torch.zeros(6, HEADS), params(),
        gate=gate, layer_idx=1)
    ((got, spec, got_gate),) = rec.verifies
    assert got is p and got_gate is gate and rec.runs == []
    want = SpecBlocks.of(pool, 1)
    assert spec.prefix.shape == (6, 2, 3, 3 * HEADS * DIM) and spec.prefix_len.shape == (6, 1)
    assert [_ptr(t) for t in spec] == [_ptr(t) for t in want]

    # the verdicts: a keeps one of its two drafts, b none; the next step reads the other side
    slots = [pool._slots[r]["main"].index for r in "ab"]
    mix.set_prefix_len(torch.tensor([1, 0]))
    assert want.prefix_len[slots, 0].tolist() == [2, 1]
    assert want.side[slots, 0].tolist() == [1, 1]
    commit(pool, ["a", "b"], [3, 3], ctx)
    plan(pool, kda, ["a", "b"], [3, 3], eager(["a", "b"], walk="decode"), speculative=True)
    mix.set_prefix_len(torch.tensor([2, 2]))
    assert want.prefix_len[slots, 0].tolist() == [3, 3]
    assert want.side[slots, 0].tolist() == [0, 0]
    # every layer shares the count
    assert SpecBlocks.of(pool, 0).prefix_len.data_ptr() == want.prefix_len.data_ptr()


def test_set_prefix_len_leaves_the_slots_outside_the_step():
    """A captured verify bucket's padding row writes the sink; another request's slot,
    not in the step, keeps its count and side."""
    pool, kda = build(speculative_tokens=2)
    kda.set_kernels(_Recorder())
    mix = KDACallable(pool, kda)
    ctx = eager(["a", "c"])
    plan(pool, kda, ["a", "c"], [2, 2], ctx)
    commit(pool, ["a", "c"], [2, 2], ctx)
    spec = SpecBlocks.of(pool, 0)
    a, c = (pool._slots[r]["main"].index for r in "ac")
    spec.prefix_len[c], spec.side[c] = 2, 1
    plan(pool, kda, ["a", "pad"], [3, 3],
         leased(["a"], ["a", "pad"], bs=2, tokens=6, walk="decode"), speculative=True)
    mix.set_prefix_len(torch.tensor([1, 2]))
    assert (spec.prefix_len[a, 0].item(), spec.side[a, 0].item()) == (2, 1)
    assert (spec.prefix_len[c, 0].item(), spec.side[c, 0].item()) == (2, 1)
    assert spec.prefix_len[SINK_SLOT, 0].item() == 3


def test_a_verify_step_fits_the_pools_blocks():
    pool, kda = build()
    with pytest.raises(ValueError, match="one block of 1..1"):
        plan(pool, kda, ["a"], [2], eager(["a"], walk="decode"), speculative=True)
    pool, kda = build(speculative_tokens=2)
    with pytest.raises(ValueError, match="got spans"):
        plan(pool, kda, ["a", "b"], [3, 2], eager(["a", "b"], walk="decode"), speculative=True)
    kda.set_kernels(_Recorder())
    plan(pool, kda, ["c"], [3], eager(["c"], walk="decode"), speculative=True)
    with pytest.raises(ValueError, match="SpecBlocks"):
        kda.run(torch.zeros(3, 3 * HEADS * DIM), torch.zeros(3, DIM), torch.zeros(3, HEADS),
                pool.block("conv", 0), pool.block("state", 0), params())


# --- the torch reference bundle -------------------------------------------------


def _torch_params(device, seed=0):
    gen = torch.Generator().manual_seed(seed)
    h, d = HEADS, DIM

    def rand(*shape, scale=0.3):
        return (torch.randn(*shape, generator=gen) * scale).to(device)

    return KDAParams(
        conv_weight=rand(3 * h * d, 4), A_log=rand(h), dt_bias=rand(h * d), lower_bound=-5.0,
        num_heads=h, head_dim=d, scale=d ** -0.5,
    )


def _torch_inputs(tokens, device, seed=1):
    gen = torch.Generator().manual_seed(seed)
    h, d = HEADS, DIM
    return (
        torch.randn(tokens, 3 * h * d, generator=gen).to(device, torch.bfloat16),
        torch.randn(tokens, h * d, generator=gen).to(device),
        torch.randn(tokens, h, generator=gen).to(device),
    )


def _prefill_then_decode(pool, kda, rids, spans, p, device, guard=contextlib.nullcontext):
    """A prefill step over ``spans``, then one decode token per row; the outputs. Only the
    layer calls run under ``guard()``."""
    mix = KDACallable(pool, kda)
    prompt = _torch_inputs(sum(spans), device)
    token = _torch_inputs(len(rids), device, seed=2)
    ctx = eager(rids)
    plan(pool, kda, rids, spans, ctx)
    with guard():
        first = mix(*prompt, p, layer_idx=1)
    commit(pool, rids, spans, ctx)
    ctx = eager(rids, walk="decode")
    plan(pool, kda, rids, [1] * len(rids), ctx)
    with guard():
        second = mix(*token, p, layer_idx=1)
    commit(pool, rids, [1] * len(rids), ctx)
    return first, second


def test_the_torch_bundle_keeps_each_row_in_its_own_slot():
    """Two prompts in one step compute and leave behind what each would alone, in another
    slot: the prefill reads and writes every row's slot by its device id."""
    p = _torch_params(CPU)
    qkv, g, beta = _torch_inputs(8, CPU)
    pool, kda = build()
    plan(pool, kda, ["x", "y"], [5, 3], eager(["x", "y"]))
    together = KDACallable(pool, kda)(qkv, g, beta, p, layer_idx=1)
    for rid, rows in (("x", slice(0, 5)), ("y", slice(5, 8))):
        alone, akda = build()
        plan(alone, akda, [rid], [rows.stop - rows.start], eager([rid]))
        out = KDACallable(alone, akda)(qkv[rows], g[rows], beta[rows], p, layer_idx=1)
        torch.testing.assert_close(out, together[rows], rtol=0, atol=0)
        mine, theirs = pool._slots[rid]["main"].index, alone._slots[rid]["main"].index
        for name in ("conv", "state"):
            torch.testing.assert_close(pool.block(name, 1)[mine], alone.block(name, 1)[theirs],
                                       rtol=0, atol=0)
    assert pool.block("state", 0).abs().sum() == 0  # another layer's blocks are untouched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_the_torch_bundle_never_waits_on_the_host():
    """The reference prefill and decode on CUDA under sync debugging: no step reads a
    device value back (the prefill once read its slot ids with ``.tolist()``)."""
    cuda = torch.device("cuda")
    pool, kda = build(device=cuda)
    assert isinstance(kda.kernels, TorchKDAKernels)  # head dim 16: no fused kernels

    @contextlib.contextmanager
    def no_sync():
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            yield
        finally:
            torch.cuda.set_sync_debug_mode(mode)

    got = _prefill_then_decode(pool, kda, ["x", "y"], [70, 3], _torch_params(cuda), cuda, no_sync)
    ref_pool, ref_kda = build()
    want = _prefill_then_decode(ref_pool, ref_kda, ["x", "y"], [70, 3], _torch_params(CPU), CPU)
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a.cpu(), b, rtol=2e-2, atol=2e-2)


# --- the Triton bundle's host side ---------------------------------------------


def test_triton_layout_eager_is_exact():
    # starts, lens, first chunks, then (span, chunk) pairs
    assert varlen_layout([70, 9]) == [0, 70, 70, 9, 0, 2, 0, 0, 0, 1, 1, 0]
    assert TritonKDAKernels().layout((70, 9), 79, fixed=False) == varlen_layout([70, 9])


def test_triton_layout_under_capture_fills_the_table():
    """Two real chunks, then rows that start at span 0's end, up to the rows
    any split of the bucket's tokens can need."""
    layout = TritonKDAKernels().layout((70, 0), 256, fixed=True)
    n, rows = 2, 256 // 64 + 2
    assert layout[:3 * n] == [0, 70, 70, 0, 0, 2]
    table = layout[3 * n:]
    assert len(table) == 2 * rows
    assert table[:4] == [0, 0, 0, 1]
    assert set(zip(table[4::2], table[5::2], strict=True)) == {(0, 2)}


def test_triton_bundle_reads_one_projection_row():
    h, d = HEADS, DIM
    params = KDAParams(
        conv_weight=torch.zeros(3 * h * d, 4), A_log=torch.zeros(h), dt_bias=torch.zeros(h * d),
        lower_bound=-5.0, num_heads=h, head_dim=d, scale=d ** -0.5,
        f_b=torch.zeros(h * d, d), g_b=torch.zeros(h * d, d), norm_weight=torch.ones(d),
    )
    proj = torch.zeros(4, 3 * h * d + h + 2 * d + 6)  # padded rows, as the model's GEMM
    p3 = 3 * h * d
    qkv, beta = proj[:, :p3], proj[:, p3:p3 + h]
    g, gate = proj[:, p3 + h:p3 + h + d], proj[:, p3 + h + d:p3 + h + 2 * d]
    assert _merged_row(qkv, g, beta, gate, params) is qkv
    with pytest.raises(ValueError, match="column slice"):
        _merged_row(qkv, g, beta.clone(), gate, params)
    with pytest.raises(ValueError, match="column slice"):
        _merged_row(qkv, gate, beta, g, params)
    params.g_b = None
    with pytest.raises(ValueError, match="up-projections"):
        _merged_row(qkv, g, beta, gate, params)
