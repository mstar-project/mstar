"""A read that runs the dense kernel on its eager steps and the paged kernel on the steps a graph replays.

Cosmos3 denoises on FlashInfer's paged kernel when a step holds a slot lease, since only that kernel captures, and on
FA3's dense kernel otherwise, which is faster eager. A step planned on one kernel and run on the other attends a plan
it never built. A step that keeps its adds runs the paged kernel too, as main's window commit did: every later step
attends what it keeps, so a commit on the dense kernel changed a kv request's second window from its first step. Each
rule is checked over CPU stand-ins, and each refusal names the array and the cause.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_state_calls import (
    DIM,
    HEADS,
    LAYERS,
    SLOTS,
    TOKENS,
    _array,
    _ctx,
    _refused,
    _StubAttn,
    _StubDense,
    _StubKV,
    _tensors,
)
from test_state_kept_reads import _CpuVarlenSdpa

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources import BucketKey, SlotLease
from mstar.engine.resources.attn import flashinfer
from mstar.engine.resources.attn.base import EagerSlotKey
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep, AttnBackend
from mstar.engine.resources.attn.dense import DenseAttentionManager
from mstar.engine.resources.kv.config import (
    KVReqConfig,
    KVSpec,
    KVStep,
    PagedKVConfig,
    RetentionPolicy,
    Span,
    SpanKind,
)
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Attend, Bidirectional, Block, Causal, KeptWindow, Layout, Read
from mstar.engine.state.adapters import (
    KeptWindowEagerDense,
    build_specs,
    kv_layouts,
    node_handles,
    state_setup,
    state_steps,
)
from mstar.engine.state.analysis import deployment_problems
from mstar.model.base import LabelLayout

EAGER = {"attn_gen": {"eager_backend": "dense"}}


def _eager(**reads) -> Array:
    """A kept array read as Cosmos3's is: causally on its prefill walk and as a whole on its generation walk."""
    attend = Attend(q_heads=HEADS)
    return _array(reads=reads or {
        "attn": Read(mask=Causal(), op=attend, walks=frozenset({"prefill"})),
        "attn_gen": Read(mask=Bidirectional(), op=attend, walks=frozenset({"gen"})),
    })


def _handle(kv=None, paged=None, dense=None, array=None):
    """Built as a node builds it, with the setup the model's kernel settings give."""
    array = array or _eager()
    setup, problems = state_setup({}, [array], EAGER)
    assert not problems
    model = SimpleNamespace(declare_state=lambda: [array], get_combined_graph_walks=lambda: {})
    resources = {"kv": kv or _StubKV(), "attn": paged or _StubAttn(), "attn_gen": dense or _StubDense()}
    return node_handles(model, "LLM", resources, setup=setup)["kv"]


def _expanded(h, leased: bool, keep: bool = False) -> SubmoduleStep:
    """``h``'s step on the generation walk. A leased one is its capture, since a replay makes no call."""
    step = SubmoduleStep(segments=[Segment(1, "main", TOKENS)], state={"kv": ArrayStep(keep=keep)})
    expanded = state_steps(step, {"kv": h}, graph_walk="gen", leased=leased)
    lease = _ctx(leased=leased).slot_lease
    expanded.set_ctx(StepContext(request_ids=(1,), graph_walk="gen", slot=0, capture=leased, slot_lease=lease))
    return expanded


def test_an_eager_read_builds_the_dense_kernel_under_its_own_key():
    setup, _ = state_setup({}, [_eager()], EAGER)
    specs = {spec.resource_key: spec for spec in build_specs([_eager()], kernel_settings=EAGER, setup=setup)
             if isinstance(spec, AttentionSpec)}
    assert specs.keys() == {"attn", "attn_gen"}, "the two reads would not share the paged spec"
    assert specs["attn_gen"].config.backend is AttnBackend.DENSE, "the eager steps would run the paged kernel"


def test_the_split_checks_take_an_eager_read_for_a_paged_one():
    setup, _ = state_setup({}, [_eager()], EAGER)
    sdpa_reads = [read for read, binding in setup.by_read.items() if not binding.plans]
    assert sdpa_reads == [], "the engine and validate_deployment would raise reading plans on the eager binding"


def test_a_leased_step_plans_the_paged_kernel_and_others_the_dense():
    h = _handle()
    assert {leased: sorted(_expanded(h, leased).steps.keys() - {"kv"}) for leased in (False, True)} == {
        False: ["attn_gen"], True: ["attn"],
    }, "a captured step would replay a plan on the dense kernel, which has no graph"


@pytest.mark.parametrize("leased", [False, True], ids=["eager", "leased"])
def test_the_handle_runs_the_kernel_its_step_planned(leased):
    paged, dense = _StubAttn(), _StubDense()
    h = _handle(paged=paged, dense=dense)
    step = _expanded(h, leased)
    h.begin_step(step)
    q, k, v = _tensors()
    h.add(0, k=k, v=v)
    h.read(0, q)
    ran, idle = (paged, dense) if leased else (dense, paged)
    assert ran.runs and not idle.runs, "the read would attend through a plan its step didn't build"


def test_a_step_that_keeps_its_adds_runs_the_paged_kernel():
    paged, dense = _StubAttn(), _StubDense()
    h = _handle(paged=paged, dense=dense)
    step = _expanded(h, leased=False, keep=True)
    h.begin_step(step)
    q, k, v = _tensors()
    h.add(0, k=k, v=v)
    h.read(0, q)
    assert paged.runs and not dense.runs, "later steps would attend entries the dense kernel computed"


def test_an_eager_read_runs_the_paged_leg_before_its_first_step():
    paged = _StubAttn()
    h = _handle(paged=paged)
    assert h.reads["attn_gen"] is paged and all(h._attn[layer] is paged for layer in range(LAYERS)), (
        "a capture's preparing pass reads the handle before any step opens it"
    )


def test_a_kept_add_reaches_the_pages():
    kv = _StubKV()
    h = _handle(kv=kv)
    step = _expanded(h, leased=False, keep=True)
    h.begin_step(step)
    q, k, v = _tensors()
    for layer in range(LAYERS):
        h.add(layer, k=k, v=v)
        h.read(layer, q)
    h.check_step(step)
    assert kv.writes == [(layer, "main") for layer in range(LAYERS)], "the commit would keep entries no page holds"


@pytest.mark.parametrize("value", ["dense", "paged"])
def test_an_eager_kernel_pairs_only_with_a_paged_read(value):
    within = _array(max_len=0, reads={"vision": Read(mask=Block(), op=Attend(q_heads=HEADS))})
    _refused(r"kernel_settings\['vision'\] sets \['eager_backend'\]",
             lambda: build_specs([within], kernel_settings={"vision": {"eager_backend": value}}))
    if value != "dense":
        setup, _ = state_setup({}, [_eager()], {"attn_gen": {"eager_backend": value}})
        _refused(r"kernel_settings\['attn_gen'\] sets \['eager_backend'\]",
                 lambda: build_specs([_eager()], kernel_settings={"attn_gen": {"eager_backend": value}}, setup=setup))


def test_an_eager_read_does_not_hold_its_group_s_key():
    attend = Attend(q_heads=HEADS)
    first = _eager(
        attn_gen=Read(mask=Bidirectional(), op=attend, walks=frozenset({"gen"})),
        attn=Read(mask=Causal(), op=attend, walks=frozenset({"prefill"})),
    )
    _refused(r"Array 'kv': no layout yet: reads\['attn_gen'\] builds a second spec under its own key",
             lambda: state_setup({}, [first], EAGER))


def test_an_eager_kernel_runs_only_a_bidirectional_read():
    attend = Attend(q_heads=HEADS)
    causal = _eager(
        attn=Read(mask=Causal(), op=attend, walks=frozenset({"prefill"})),
        attn_gen=Read(mask=Causal(), op=attend, walks=frozenset({"gen"})),
    )
    setup, problems = state_setup({}, [causal], EAGER)
    assert problems == [
        "kernel_settings['attn_gen']: eager_backend dense: Array 'kv': no layout yet: reads['attn_gen'].mask must be "
        "Bidirectional to take eager_backend dense, since the dense kernel has no causal plan and no window"
    ]
    assert "attn_gen" not in setup.by_read, "a causal read would attend past its mask on the dense kernel"


def _windowed_pool(page_size: int, sink: int, window: int) -> KVManager:
    """A CPU pool of one layer of one head of one value, whose entries hold their log positions, with request ``a``
    under an exact retention of ``sink`` and ``window``."""
    cfg = PagedKVConfig(
        num_layers=1, num_qo_heads=1, num_kv_heads=1, head_dim=1, max_seq_len=64 * page_size, max_num_pages=64,
        page_size=page_size,
    )
    pool = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")), device=torch.device("cpu"),
        dtype=torch.float32,
    )
    layout = LabelLayout(growth=(("rollout", 1),), retention=RetentionPolicy(window, sink, exact=True))
    request = KVReqConfig(needed_labels=["main"])
    request.apply_conductor_config(kv_layout={"main": layout.row()}, loop_caps={"rollout": 10_000})
    pool.ingest_request("a", overrides=request)
    return pool


@pytest.mark.parametrize("page_size", [16, 128])
def test_the_dense_prefix_is_the_kept_reach(page_size):
    """Against the dense reference: after each of 30 commits of uneven spans, a denoise step on the dense kernel
    attends the first ``sink`` log positions and the last ``window`` the stream kept, while the pool releases pages
    behind the sink. Gathering whole pages from the first would attend the released ones' successors in their place."""
    sink, window = 2 * page_size + 5, 3 * page_size + 7
    pool = _windowed_pool(page_size, sink, window)
    dense = DenseAttentionManager(
        kv_cache="kv", device=torch.device("cpu"), dtype=torch.float32, kv_config=pool.config,
        wrapper_class=_CpuVarlenSdpa,
    )
    ctx = StepContext(request_ids=("a",), graph_walk="rollout", slot=0, capture=False)
    start = 0
    for commit in range(30):
        span = 1 + (7 * commit) % (page_size // 2)
        step = KVStep(segments=(Segment("a", "main", span),))
        assert pool.admit(step, ctx).ok
        pool.plan(step, ctx)
        pool.write_kv(*([torch.arange(start, start + span, dtype=torch.float32)[:, None, None]] * 2), layer_idx=0,
                      label="main")
        pool.commit(step, ctx)
        start += span
        denoise = KVStep(segments=(Segment("a", "main", 4),), commit=False)
        assert pool.admit(denoise, ctx).ok
        plan = dense._build_plan(pool.plan(denoise, ctx)["main"], False, EagerSlotKey(label="main", slot=0))
        k, _ = dense._prefix_kv(plan.segments[0], 0, pool.layer_view(0))
        pool.commit(denoise, ctx)
        expected = sorted({*range(min(sink, start)), *range(max(sink, start - window), start)})
        assert plan.cu_k == (0, len(expected) + 4), f"the denoise after commit {commit} plans another key length"
        assert k.flatten().long().tolist() == expected, f"the denoise after commit {commit} attends other entries"


# Cosmos3's windowed generation read beside its prefill read


def _windowed() -> Array:
    """Cosmos3's windowed array: read causally on its prefill walk and through a kept window on its generation walk."""
    attend = Attend(q_heads=HEADS)
    return _array(reads={
        "attn": Read(mask=Causal(), op=attend, walks=frozenset({"prefill"})),
        "attn_gen": Read(mask=Bidirectional(kept_window=True), op=attend, walks=frozenset({"gen"})),
    })


def _span(walk: str, length: int) -> Span:
    return Span(SpanKind.ROWS, walk, length, length, "rows")


def _laid_out(layout: Layout):
    return kv_layouts([_windowed()], {"kv": {"main": layout}}, {})["kv"]["main"]


def test_a_windowed_read_on_its_own_walk_takes_its_eager_pair():
    setup, problems = state_setup({}, [_windowed()], EAGER)
    assert not problems and type(setup.by_read["attn_gen"]) is KeptWindowEagerDense, (
        "the windowed generation read would have no kernel beside the prefill read"
    )


def test_a_windowed_deployment_without_its_eager_kernel_has_no_layout():
    """The gathered kept-window kernel is a spec of its own, over the layers the prefill read's paged spec covers."""
    _refused(r"Array 'kv': no layout yet: reads\['attn'\] and reads\['attn_gen'\] build different specs and both "
             "cover layer 0", lambda: build_specs([_windowed()]))


def test_a_neighbour_read_stays_inside_the_sink():
    layout = Layout(spans=(_span("prefill", 10), _span("gen", 20)), kept_windows={"attn_gen": KeptWindow(4, window=8)})
    with pytest.raises(RuntimeError, match=r"a span of walk 'prefill' ends at entry 10, past the 4 entries "
                                           r"reads\['attn_gen'\]'s kept window never releases"):
        _laid_out(layout)


def test_growth_beside_a_released_window_has_no_layout():
    layout = Layout(spans=(_span("prefill", 4),), growth=(("rollout", 8),),
                    kept_windows={"attn_gen": KeptWindow(4, window=8)})
    with pytest.raises(RuntimeError, match=r"\['main'\]\.growth: a loop's entries name no walk"):
        _laid_out(layout)


def test_a_window_of_max_len_never_releases():
    """Cosmos3 gives a request that isn't kv mode a window of the array's whole length, so the prefill read may reach
    past the empty sink."""
    layout = Layout(spans=(_span("prefill", 10), _span("gen", 20)),
                    kept_windows={"attn_gen": KeptWindow(0, window=SLOTS)})
    retention = _laid_out(layout).retention
    assert retention == RetentionPolicy(SLOTS, 0, exact=True) and not retention.can_release(SLOTS), (
        "a window of the array's length would release entries the prefill read reaches"
    )


def test_a_window_that_never_releases_keeps_the_intake_checks():
    """A request past the pool's length is refused, and its loop's room counted, under a window that never releases as
    with no window; only a window that releases keeps fewer entries than its positions."""
    spec = KVSpec(resource_key="kv", nodes={"dit"}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, page_size=16, max_num_pages=64,
    ))
    resolve_spec_dependencies([spec, AttentionSpec(resource_key="attn", nodes={"dit"}, config=AttentionConfig("kv"))])
    never = LabelLayout(growth=(("rollout", 1),), retention=RetentionPolicy(64, 0, exact=True))
    request = KVReqConfig(needed_labels=["main"])
    request.apply_conductor_config(kv_layout={"main": never.row()}, loop_caps={"rollout": 100})
    assert request.loop_room(spec) == {"rollout": 64}, "the loop would run past the pool's length"
    with pytest.raises(ValueError, match="this request needs 100 positions of 'main', past the model's context of 64"):
        request.validate(spec)


def test_a_windowed_array_takes_no_max_seq_len_past_its_max_len():
    """The sink guard reads the array's max_len and the intake the pool's max_seq_len, so the two must agree."""
    model = SimpleNamespace(declare_state=lambda: [_windowed()])
    assert deployment_problems(model, {"kv": {"max_seq_len": 2 * SLOTS}}) == [
        f"Array 'kv': resources.kv.max_seq_len is {2 * SLOTS}, past its max_len of {SLOTS}, which its kept window's "
        "requests are checked against; raise max_len"
    ], "a request past max_len would release entries under a window the guard let through"


# The paged leg of a kept window


def _committed(entries: int):
    """A pool of pages of 16 whose request ``a`` keeps a sink of 8 and a window of 16, after ``entries`` kept entries:
    its config, and the plan of a step of 4 after them."""
    pool = _windowed_pool(16, sink=8, window=16)
    ctx = StepContext(request_ids=("a",), graph_walk="rollout", slot=0, capture=False)
    for _ in range(0, entries, 8):
        step = KVStep(segments=(Segment("a", "main", 8),))
        assert pool.admit(step, ctx).ok
        pool.plan(step, ctx)
        pool.write_kv(*([torch.zeros(8, 1, 1)] * 2), layer_idx=0, label="main")
        pool.commit(step, ctx)
    denoise = KVStep(segments=(Segment("a", "main", 4),), commit=False)
    assert pool.admit(denoise, ctx).ok
    return pool.config, pool.plan(denoise, ctx)


def test_a_released_stream_never_plans_under_a_lease():
    """The paged kernel attends every page a row holds, which is the window's reach only while the stream keeps no
    entry past it. A captured step replays the paged kernel, so it refuses such a row."""
    config, planned = _committed(48)
    paged = flashinfer.FlashInferManager("kv", torch.device("cpu"), torch.float32, config)
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="rollout", cg_key_info=None, bs=1, num_tokens=4))
    ctx = SimpleNamespace(resident=None, slot_lease=lease, is_preplan=False, force_prefill=False, slot=0,
                          plan_phase="res.plan", plan_results={"kv": planned})
    with pytest.raises(RuntimeError, match="holds entries its kept window no longer reaches"):
        paged.plan(AttentionStep(causal=False), ctx)


def test_a_stream_its_window_reaches_whole_is_not_marked():
    _, planned = _committed(16)
    assert not planned["main"].beyond_reach, "a captured step would refuse a stream that released nothing"


# A capture of the paged leg


class _StubPaged(_StubAttn):
    """Records its runs instead of reading the pool's pages."""

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        self.runs.append((layer_idx, label))
        return q


def test_a_capture_warms_a_kept_window_read_up_on_its_padding_row():
    """A capture runs its step eagerly on padding rows before it records, and a padding row opens with no layout, so
    with no kept window. Refused as a request with no window is, no windowed Cosmos3 config that captures could boot."""
    pool = KVManager(
        cfg=PagedKVConfig(num_layers=LAYERS, num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=DIM, max_seq_len=SLOTS,
                          max_num_pages=8, page_size=16),
        name="kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")), device=torch.device("cpu"),
        dtype=torch.float32,
    )
    pool.build_accelerator_graph_buffers([], 1, TOKENS)
    pool.ingest_capture_row(-2)
    paged = _StubPaged()
    h = _handle(kv=pool, paged=paged, array=_windowed())
    step = SubmoduleStep(segments=[Segment(-2, "main", TOKENS)], state={"kv": ArrayStep(keep=False)})
    step = state_steps(step, {"kv": h}, graph_walk="gen", leased=True)
    lease = _ctx(leased=True).slot_lease
    ctx = StepContext(request_ids=(-2,), graph_walk="gen", slot=0, capture=True, slot_lease=lease)
    step.set_ctx(ctx)
    assert pool.admit(step.steps["kv"], ctx).ok
    pool.plan(step.steps["kv"], ctx)
    q, k, v = _tensors()
    step.begin_calls()
    for layer in range(LAYERS):
        h.add(layer, k=k, v=v)
        h.read(layer, q)
    step.check_calls()
    step.end_calls()
    assert paged.runs == [(layer, "main") for layer in range(LAYERS)], "the capture would record no paged read"

