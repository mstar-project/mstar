"""Per-request LoRA's custom ops, the async worker, the step's compute scope and the
base-only captures, on CPU."""

from __future__ import annotations

import itertools
import threading
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("safetensors")

import test_lora_pool as P  # noqa: E402

from mstar.engine.accelerator_graph_config import PackedAcceleratorGraphConfig  # noqa: E402
from mstar.engine.resources.base import EngineResourceInfo, Resource  # noqa: E402
from mstar.engine.resources.lora import LoraBaseOnly, LoraStep  # noqa: E402
from mstar.engine.resources.lora.ops import LoraOpsBackend, lora_apply_, lora_expand_, lora_shrink  # noqa: E402
from mstar.engine.resources.lora.pool import LoraAdapterPool  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.engine.resources.step import (  # noqa: E402
    ADMIT_OK,
    PIECEWISE_WALK,
    BucketKey,
    Segment,
    SlotLease,
    StepContext,
    SubmoduleStep,
)
from mstar.model.submodule_base import ARNodeInputs  # noqa: E402

F64 = torch.float64
CPU = torch.device("cpu")
# as the engine builds a pool: scoped from the start, so a delta outside a step's compute raises
ENGINE_INFO = EngineResourceInfo(device=CPU, compute_dtype=F64)
adapters = P.adapters  # a1 on q/k/v/o, a2 on q and v, a3 on o, through two layers


def test_custom_ops_pass_opcheck():
    """Schemas (``lora_expand_`` declares the ``y`` it writes), fake shapes and the
    functionalized path, on addressing with a capture bucket's tail."""
    widths, ranks, scalings = [6, 3, 5], torch.tensor([0, 1, 4, 3], dtype=torch.int32), torch.tensor([0, 0.5, 2, -1.0])
    token_slots = torch.tensor([2, 0, 1, 3, 2, 2, 1], dtype=torch.int32)
    order = torch.cat([torch.argsort(token_slots, stable=True).to(torch.int32), torch.zeros(3, dtype=torch.int32)])
    indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.bincount(token_slots, minlength=4).cumsum(0)])
    common = (order, indptr.to(torch.int32), ranks)
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(4, 3 * 4, 12, generator=g, dtype=F64), torch.randn(4, 14, 4, generator=g, dtype=F64)
    x, y = torch.randn(10, 12, generator=g, dtype=F64), torch.randn(10, 14, generator=g, dtype=F64)
    torch.library.opcheck(lora_shrink, (x, a, *common, len(widths)))
    h = lora_shrink(x, a, *common, len(widths))
    torch.library.opcheck(lora_expand_, (y, h, b, *common, scalings, widths))
    torch.library.opcheck(lora_apply_, (y, x, a, b, *common, scalings, widths))


# --- the async worker and the compute scope -------------------------------------------------

class _Plain(Resource):
    """Admits and records what the runner drives; does not track compute."""

    def __init__(self, preplan=False):
        self._preplan, self.calls = preplan, []

    build = None

    @property
    def supports_preplan(self):
        return self._preplan

    def admit(self, step, ctx):
        self.calls.append("admit")
        return ADMIT_OK

    def rollback_admit(self, step, ctx):
        self.calls.append("rollback")

    def plan(self, step, ctx):
        self.calls.append("plan")


def _runner_step(rids, keys):
    step = SubmoduleStep(segments=[Segment(r, "main", 1) for r in rids], steps={k: LoraStep() for k in keys})
    step.set_ctx(StepContext(request_ids=list(rids), graph_walk="decode", slot=0, capture=False))
    return step


def test_a_pre_plan_never_touches_the_pool(adapters):
    """A speculated step's pre-admit and pre-plan (plan thread) take no slot and stage
    nothing: the pool admits and plans on the GPU thread."""
    pool, kv = P.make_pool(adapters[1], max_slots=1), _Plain(preplan=True)
    runner = StepRunner({"kv": kv, "lora": pool})
    P.ingest(pool, "r", "a1")
    step = _runner_step(["r"], ["kv", "lora"])
    step.ctx.is_preplan = True
    assert runner.pre_admit(step).ok
    runner.pre_plan(step)
    assert kv.calls == ["admit", "plan"]
    assert pool.slot_of("r") is None and pool._current is None
    runner.clear_preplan()
    assert pool.slot_of("r") is None
    live = _runner_step(["r"], ["kv", "lora"])
    assert runner.admit(live).ok
    runner.plan(live)
    assert pool.slot_of("r") == 1 and P.token_slots_of(pool.addressing()) == [1]


def test_a_refusal_unwinds_the_pools_partial_takes_through_the_runner(adapters):
    """r1 takes the one slot (a1 resident) before r2's miss finds none: the pool gives r1's
    take back, the runner rolls back the resource before it, and a1 stays cached."""
    pool, kv = P.make_pool(adapters[1], max_slots=1), _Plain()
    P.ingest(pool, "r0", "a1")
    assert StepRunner({"lora": pool}).admit(_runner_step(["r0"], ["lora"])).ok
    pool.remove_request("r0")  # a1 stays resident, unheld
    P.ingest(pool, "r1", "a1")
    P.ingest(pool, "r2", "a2")
    outcome = StepRunner({"kv": kv, "lora": pool}).admit(_runner_step(["r1", "r2"], ["kv", "lora"]))
    assert not outcome.ok and outcome.failed_resource == "lora"
    assert outcome.reason.request_id == "r2" and kv.calls == ["admit", "rollback"]
    assert pool.slot_of("r1") is None and pool.slot_state(1).refcount == 0
    assert pool.resident_slot("a1") == 1


def test_a_delta_runs_only_between_its_steps_plan_and_commit(adapters):
    """A delta outside a step (prepare_inputs, a cached output) would read another step's
    addressing: the pool keeps a step's addressing from its plan to its commit or rollback,
    for the thread that planned it, and a delta called outside that refuses. The async
    worker runs prepare_inputs on the main thread while the GPU thread's step is planned."""
    pool = P.make_pool(adapters[1])
    P.ingest(pool, "r", "a1")

    def delta():
        x, y = torch.randn(1, P.HIDDEN, dtype=F64), torch.zeros(1, P.Q_OUT + 2 * P.KV_OUT, dtype=F64)
        pool.layer_delta("qkv", 0)(x, y)

    def on_another_thread():
        refused = []

        def run():
            try:
                delta()
            except RuntimeError as exc:
                refused.append(exc)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        return refused

    with pytest.raises(RuntimeError, match="no planned step addressing its rows"):
        delta()
    for end in (pool.commit, pool.rollback_admit):
        s, c = P.step(["r"]), P.ctx(["r"])
        assert pool.admit(s, c).ok
        pool.plan(s, c)
        assert on_another_thread(), "a delta on another thread ran against this step's rows"
        delta()
        end(s, c)
        with pytest.raises(RuntimeError, match="outside"):
            delta()


class _Warmed(torch.nn.Module):
    """A node whose ``fc`` the engine compiles and warms (``compile_regions``)."""

    disable_torch_compile = True

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4, dtype=F64)

    def compile_regions(self):
        return {"fc": [(torch.ones(2, 4, dtype=F64),)]}


@pytest.mark.parametrize("compiled", [False, True])
def test_a_warm_up_outside_any_step_runs_a_bound_module_as_a_base_only_step(compiled, monkeypatch):
    """The engine warms compile_regions (and warmup_compiles) outside any step: a bound
    module there adds no delta, eager or traced, instead of refusing for want of a plan,
    and the pool is left with none."""
    from lora_module_helpers import pool as lora_pool
    from lora_module_helpers import serve

    from mstar.engine.engine import Engine
    from mstar.model.components.lora.modules import bind_module_lora, module_lora_spec

    real = torch.compile
    monkeypatch.setattr(torch, "compile", (lambda fn, **kw: real(fn, backend="eager", **kw)) if compiled
                        else (lambda fn, **kw: fn))
    torch._dynamo.reset()
    with torch.device("meta"):
        spec = module_lora_spec(_Warmed(), resource_key="lora", nodes={"N"}, root_path="")
    node, pool = _Warmed(), lora_pool(serve(spec, {}))
    bind_module_lora(node, {"lora": pool})
    engine = object.__new__(Engine)
    engine._device, engine._device_capability = CPU, SimpleNamespace(is_available=True)
    engine._submodules = {"N": SimpleNamespace(submodule=node, fp32_precision=None, node_dtype=None,
                                               lora_pools=(pool,))}
    engine._compile_regions()
    assert pool._current is None and pool.active
    with pytest.raises(RuntimeError, match="no plan"):
        node.fc(torch.ones(2, 4, dtype=F64))


# --- base-only captures ------------------------------------------------------------------

class _CountingBackend(LoraOpsBackend):
    calls = 0

    def apply_(self, *args, **kwargs):
        self.calls += 1
        return super().apply_(*args, **kwargs)


def _delta_runs(pool, backend, tokens=2):
    """Whether the qkv delta at layer 0 launches anything, and that a skip leaves y as is."""
    before, y = backend.calls, torch.randn(tokens, P.Q_OUT + 2 * P.KV_OUT, dtype=F64)
    out = pool.layer_delta("qkv", 0)(torch.randn(tokens, P.HIDDEN, dtype=F64), y.clone())
    if backend.calls == before:
        assert torch.equal(out, y)
    return backend.calls > before


def test_every_bucket_is_captured_again_under_a_base_only_key_beside_the_nodes_own():
    """Each bucket under the node's own key (a model's cg_key_info) and under LoraBaseOnly
    of it, sharing the config; a node without LoRA captures exactly what it did."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    configs = [PackedAcceleratorGraphConfig(
        capture_graph_walk="prefill", capture_token_lengths=[16, 32], capture_batch_sizes=[1, 2],
        make_node_input=lambda n: ARNodeInputs(input_seq_len=n), additional_key_info=cfg,
    ) for cfg in (True, False)]

    def specs(lora_keys):
        runner = object.__new__(AcceleratorGraphRunner)
        runner._capture_configs, runner._num_slots = configs, 2
        runner._submodule = SimpleNamespace(lora_keys=lora_keys)
        return runner._get_slot_specs()

    plain, served = specs(()), specs(("lora",))
    assert {s.bucket.cg_key_info for s in plain} == {True, False} and len(served) == 2 * len(plain)
    own = {(s.bucket, s.slot, s.config_idx) for s in plain}
    twins = {(s.bucket, s.slot, s.config_idx) for s in served} - own
    assert twins == {
        (BucketKey(b.graph_walk, b.bs, b.num_tokens, LoraBaseOnly(b.cg_key_info)), slot, idx) for b, slot, idx in own
    }
    assert str(LoraBaseOnly(True)) == "lora-base-only:True"


def _engine_with(pool, captured_keys, model_key="cfg"):
    """An engine whose node keys captures ``model_key`` and has ``captured_keys``; and what it asks."""
    from mstar.engine.engine import Engine

    asked = []

    def lease_slot(graph_walk, bs, slot, num_tokens=None, cg_key_info=None):
        asked.append(cg_key_info)
        return SlotLease(slot, BucketKey(graph_walk, 8, 8, cg_key_info)) if cg_key_info in captured_keys else None

    engine = object.__new__(Engine)
    engine._submodules = {"LLM": SimpleNamespace(
        submodule=SimpleNamespace(cg_key_info=lambda walk, info, **kw: model_key), lora_pools=(pool,),
        accelerator_graph_runner=SimpleNamespace(lease_slot=lease_slot, is_packed=lambda lease: False),
        resident=None,
    )}
    return engine, asked


def _batch(rids):
    batch = SimpleNamespace(node_name="LLM", slot=0, request_ids=list(rids), inputs=None, cg_key_info=None,
                            per_request_info_wrapped={}, per_request_input_metadata={},
                            step_context=SimpleNamespace(graph_walk="decode", force_prefill=None))
    batch.lease_slot = lambda lease: setattr(batch, "lease", lease)
    return batch


def test_the_lease_takes_the_base_only_capture_only_when_no_row_names_an_adapter(adapters):
    pool = P.make_pool(adapters[1])
    for rid, name in (("b0", None), ("b1", None), ("r", "a1")):
        P.ingest(pool, rid, name)
    engine, asked = _engine_with(pool, {"cfg", LoraBaseOnly("cfg")})
    lease = engine.reserve_replay_slot(_batch(["b0", "b1"]))
    assert lease.bucket.cg_key_info == LoraBaseOnly("cfg") and asked == [LoraBaseOnly("cfg")]
    # one adapter row: the LoRA graph, never asked for the base-only one
    asked.clear()
    assert engine.reserve_replay_slot(_batch(["b0", "r"])).bucket.cg_key_info == "cfg"
    assert asked == ["cfg"]
    # no base-only bucket fits (not captured, or dropped): the LoRA graph serves it
    engine, asked = _engine_with(pool, {"cfg"})
    assert engine.reserve_replay_slot(_batch(["b0"])).bucket.cg_key_info == "cfg"
    assert asked == [LoraBaseOnly("cfg"), "cfg"]


def test_the_declaration_check_still_compares_the_nodes_own_key():
    """The engine's lease/declare_step check unwraps only the pool's part: a model whose
    declare_step disagrees with its cg_key_info still fails under a base-only lease."""
    from mstar.engine.engine import Engine

    def plan(step, phase):
        raise NotImplementedError("reached plan")

    engine = object.__new__(Engine)
    engine._enable_nvtx, engine._runner = False, SimpleNamespace(plan=plan)
    engine._pre_plan_done = {}
    lease = SlotLease(slot=0, bucket=BucketKey("decode", 1, 1, LoraBaseOnly("cfg")))
    ctx = StepContext(request_ids=[1], graph_walk="decode", slot=0, capture=False, slot_lease=lease)
    mgmt = SimpleNamespace(accelerator_graph_runner=None, submodule=None)

    def drive(key):
        engine._drive_step(SimpleNamespace(node_name="LLM", preplan_event=None), mgmt, [1], [], {},
                           ctx, lease, True, SubmoduleStep(steps={}, cg_key_info=key), False)

    with pytest.raises(RuntimeError, match="disagree"):
        drive("other")
    with pytest.raises(RuntimeError, match="disagree"):
        drive(LoraBaseOnly("cfg"))  # a model never stamps the pool's part itself
    with pytest.raises(NotImplementedError, match="reached plan"):
        drive("cfg")


def test_a_base_only_step_launches_no_delta_captured_or_eager(adapters):
    pool = LoraAdapterPool(P.make_spec(adapters[1]), "cpu", backend=(backend := _CountingBackend()))
    P.ingest(pool, "b", None)
    P.ingest(pool, "r", "a1")
    # captured under the base-only key: no addressing staged, no buffer, no kernel
    base_only, rids = SlotLease(slot=0, bucket=BucketKey("decode", 2, 2, LoraBaseOnly())), [-1, -2]
    s, c = P.step(rids), P.ctx(rids, capture=True, lease=base_only)
    assert pool.admit(s, c).ok and pool.plan(s, c) is None
    assert not pool.active and not pool._cg and not _delta_runs(pool, backend)
    pool.commit(s, c)
    # its own key's capture records the kernels
    s, c = P.step(rids), P.ctx(rids, capture=True, lease=SlotLease(slot=0, bucket=BucketKey("decode", 2, 2, None)))
    assert pool.admit(s, c).ok and pool.plan(s, c) is not None
    assert pool.active and _delta_runs(pool, backend)
    # eager: base rows alone skip; one adapter row runs every row's delta
    for rows, runs in ((["b"], False), (["b", "r"], True), (["b"], False)):
        s, c = P.step(rows), P.ctx(rows)
        assert pool.admit(s, c).ok
        pool.plan(s, c)
        assert _delta_runs(pool, backend, tokens=len(rows)) == runs, rows
        pool.commit(s, c)
    # a base-only replay of a row that names an adapter would drop its delta silently
    s, c = P.step(["r", -2]), P.ctx(["r"], lease=base_only, padded=["r", -2])
    assert pool.admit(s, c).ok
    with pytest.raises(RuntimeError, match="base-only"):
        pool.plan(s, c)


def test_a_piecewise_region_inside_a_base_only_step_hands_its_state_back(adapters):
    """A region whose capture records LoRA kernels plans its own rows (active), and its commit restores
    the enclosing base-only step's state for the eager code after it. Past the step, and
    past a region's capture (which plans outside any step), no plan is left: a delta
    refuses, eager or compiled, rather than add a base-only skip or a capture's rows."""
    pool = LoraAdapterPool(P.make_spec(adapters[1]), "cpu", backend=(backend := _CountingBackend()))
    P.ingest(pool, "b", None)
    outer_s, outer_c = P.step(["b"]), P.ctx(["b"])
    assert pool.admit(outer_s, outer_c).ok
    pool.plan(outer_s, outer_c)
    assert not pool.active
    region = SlotLease(slot=0, bucket=BucketKey(PIECEWISE_WALK, 1, 4, "backbone_layers"))
    rs = P.step(["b"], [4])
    rc = StepContext(request_ids=["b"], graph_walk=PIECEWISE_WALK, slot=0, capture=False, slot_lease=region)
    assert pool.admit(rs, rc).ok
    pool.plan(rs, rc)
    assert pool.active
    pool.commit(rs, rc)
    assert not pool.active and not _delta_runs(pool, backend)
    pool.commit(outer_s, outer_c)
    delta = pool.layer_delta("qkv", 0)
    x, y = torch.randn(4, P.HIDDEN, dtype=F64), torch.zeros(4, P.Q_OUT + 2 * P.KV_OUT, dtype=F64)

    def layer(x, y):
        return delta(x, y) * 2

    torch._dynamo.reset()
    compiled = torch.compile(layer, backend="eager")
    with pytest.raises(RuntimeError, match="no plan"):
        compiled(x, y)
    cs, cc = P.step([-1], [4]), StepContext(
        request_ids=[-1], graph_walk=PIECEWISE_WALK, slot=0, capture=True, slot_lease=region)
    assert pool.admit(cs, cc).ok
    pool.plan(cs, cc)
    pool.commit(cs, cc)
    for call in (delta, compiled):
        with pytest.raises(RuntimeError, match="no plan"):
            call(x, y)


class _Attn(Resource):
    build = None
    force_double_buffer = False


def _region_runner(pool, **config):
    from mstar.engine.accelerator_graph_config import PiecewisePackedConfig
    from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner

    config = PiecewisePackedConfig(
        capture_fn=lambda inp: inp.static_inputs["x"], make_static_inputs=lambda shape: {},
        attn_key="attn", total_tokens=[8], **config,
    )
    # records the buckets the region claims buffers for, which it captures under
    step_runner = SimpleNamespace(built=[])
    step_runner.build_accelerator_graph_buffers = lambda specs, **kwargs: step_runner.built.extend(specs)
    return PiecewiseAcceleratorGraphRunner(
        label="N_region", config=config, resources={"attn": _Attn(), "lora": pool},
        step_runner=step_runner,
        device=CPU, autocast_dtype=None, num_slots=2, node_name="N", lora_keys=("lora",),
    )


def test_a_piecewise_region_plans_its_own_lora_rows_and_one_row_replays_only_base_rows(adapters):
    """A region's step carries the node's LoRA rows, laid out from its segments as the
    node's are. A region replayed as one row per step, which addresses every row to the
    first request's adapter, is captured base-only, recording no LoRA kernel, and leased
    only for a step none of whose rows runs one; a step with one runs the region eagerly
    under the node's step."""
    pool = P.make_pool(adapters[1])
    runner = _region_runner(pool, capture_batch_sizes=[2])
    bucket = BucketKey(PIECEWISE_WALK, 2, 8, "N_region")
    step = runner._declare(["r0", -5], [5, 3], bucket, capture=False, slot=0, real_bs=1)
    assert step.steps["lora"].segments == (Segment("r0", "main", 5), Segment(-5, "main", 3))
    assert runner.prepare_for_capture() and not runner.dropped_shapes

    for rid, adapter in (("b0", None), ("b1", None), ("r1", "a1")):
        P.ingest(pool, rid, adapter)
    one_row = _region_runner(pool, capture_batch_sizes=[1], lease_before_step=True, one_row_per_step=True)
    assert one_row.prepare_for_capture() and not one_row.dropped_shapes
    assert {spec.bucket.cg_key_info for spec in runner._step_runner.built} == {"N_region"}
    assert {spec.bucket.cg_key_info for spec in one_row._step_runner.built} == {LoraBaseOnly("N_region")}
    assert one_row.can_lease([], ["b0", "b1"]) and not one_row.can_lease([], ["b0", "r1"])
    # every row, not only the first one the replay names
    with pytest.raises(RuntimeError, match="replays one row per step"):
        one_row.run({}, request_ids=["b0", "r1"])
    assert runner.can_lease([], ["b0", "r1"])


def test_the_base_only_choice_is_the_same_on_every_rank(adapters):
    """It reads only which adapter each request named (and which adapters the files say
    are empty, the same on every rank): two ranks whose ingests landed in another order,
    and whose slots differ, pick the same capture."""
    named = {"b0": None, "b1": None, "r1": "a1", "r2": "a2"}
    ranks = [P.make_pool(adapters[1], max_slots=2), P.make_pool(adapters[1], max_slots=2)]
    for pool, order in zip(ranks, (list(named), list(reversed(named))), strict=True):
        for rid in order:
            P.ingest(pool, rid, named[rid])
    assert P.run_step(ranks[0], ["r1", "r2"]).ok  # rank 0 has both resident
    for n in range(1, 5):
        for rows in itertools.permutations(named, n):
            want = any(named[r] is not None for r in rows)
            assert [pool.runs_adapters(rows) for pool in ranks] == [want, want], rows
    # once the adapter rows are removed, the shortcut agrees too
    for pool in ranks:
        pool.remove_request("r1")
        pool.remove_request("r2")
        assert not pool._live and not pool.runs_adapters(list(named))
