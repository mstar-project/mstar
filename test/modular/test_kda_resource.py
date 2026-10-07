"""The KDA linear-attention resource: built off the pool, planned against it.

What the manager owns is the plan: which rows run, which walk they take, and
the token layout its kernel bundle asked for, staged where a captured graph
will read it. The math is the bundle's, so a recording bundle stands in here.

CPU-only: nothing here launches a kernel.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.linear_attn.config import (
    LinearAttnBackend,
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.linear_attn.kda import KDAManager, KDAParams, SpecBlocks
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


def build(**kwargs) -> tuple[RecurrentStatePool, KDAManager]:
    pool_spec, kda_spec = specs(**kwargs)
    pool = build_resource(pool_spec, EngineResourceInfo(device=CPU))
    kda = build_resource(kda_spec, EngineResourceInfo(
        device=CPU, dependencies={POOL: pool_spec},
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
    # no CUDA: no default bundle, the model installs its reference
    assert kda.kernels is None
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


def test_no_default_bundle_at_an_unvalidated_head_dim(monkeypatch):
    from mstar.engine.resources.linear_attn import kda as kda_mod

    monkeypatch.setattr(kda_mod.kda_triton, "_HAS_TRITON", True)
    monkeypatch.setattr(kda_mod.kda_triton, "TritonKDAKernels", lambda: "bundle")
    cuda = torch.device("cuda")
    assert kda_mod._default_kernels(cuda, 128) == "bundle"
    assert kda_mod._default_kernels(cuda, 16) is None


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


def test_a_speculative_step_runs_verify_blocks():
    pool, kda = build(speculative_tokens=2)
    assert kda.speculative_tokens == 2
    kda.set_kernels(rec := _Recorder())
    p = plan(pool, kda, ["a", "b"], [3, 3], eager(["a", "b"], walk="decode"), speculative=True)
    assert p.is_verify and p.block == 3 and not p.is_decode
    assert p.layout is None and rec.layouts == []
    spec = SpecBlocks.of(pool, 1)
    assert spec.prefix.shape == (6, 2, 3, 3 * HEADS * DIM) and spec.prefix_len.shape == (6, 1)
    gate = torch.zeros(6, DIM)
    kda.run(torch.zeros(6, 3 * HEADS * DIM), torch.zeros(6, DIM), torch.zeros(6, HEADS),
            pool.block("conv", 1), pool.block("state", 1), params(), gate=gate, spec=spec)
    assert rec.verifies == [(p, spec, gate)] and rec.runs == []

    # the verdicts: a keeps one of its two drafts, b none; the next step reads the other side
    slots = [pool._slots[r]["main"].index for r in "ab"]
    kda.set_prefix_len(spec, torch.tensor([1, 0]))
    assert spec.prefix_len[slots, 0].tolist() == [2, 1]
    assert spec.side[slots, 0].tolist() == [1, 1]
    kda.set_prefix_len(spec, torch.tensor([2, 2]))
    assert spec.prefix_len[slots, 0].tolist() == [3, 3]
    assert spec.side[slots, 0].tolist() == [0, 0]
    # every layer shares the count
    assert SpecBlocks.of(pool, 0).prefix_len is spec.prefix_len


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
