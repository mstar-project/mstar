"""Per-request LoRA under capture (idle captures included), torch.compile, the async worker
and the step's compute scope, on CPU. A CUDA graph cannot be captured on a CPU, so a replay
is emulated by running the kernels on the views the capture planned, after the replay's
plan wrote them."""

from __future__ import annotations

import itertools
import threading
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("safetensors")

import lora_module_helpers as Q  # noqa: E402
import test_lora_pool as P  # noqa: E402

from mstar.engine.accelerator_graph_config import (  # noqa: E402
    BatchedAcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
)
from mstar.engine.resources.base import EngineResourceInfo, Resource  # noqa: E402
from mstar.engine.resources.lora import LORA_RESOURCE_KEY, LoraReqConfig  # noqa: E402
from mstar.engine.resources.lora.ops import LoraOpsBackend, lora_apply_, lora_expand_, lora_shrink  # noqa: E402
from mstar.engine.resources.lora.pool import ZERO_SLOT, LoraAdapterPool, _Layout  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.engine.resources.step import (  # noqa: E402
    ADMIT_OK,
    PIECEWISE_WALK,
    BucketKey,
    ResourceStep,
    Segment,
    SlotLease,
    StepContext,
    SubmoduleStep,
)
from mstar.model.keys import MIXED  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, NodeInputs, engine_step  # noqa: E402

F64 = torch.float64
CPU = torch.device("cpu")
# as the engine builds a pool: scoped from the start, so a delta outside a step's compute raises
ENGINE_INFO = EngineResourceInfo(device=CPU, compute_dtype=F64)
adapters = P.adapters  # a1 on q/k/v/o, a2 on q and v, a3 on o, through two layers


@pytest.fixture
def qwen_adapters(tmp_path):
    a2 = ("self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj")
    return {"a1": Q.write_peft_adapter(tmp_path / "a1", 4, tuple(Q.PROJ_SHAPES), alpha=8.0, seed=11)[0],
            "a2": Q.write_peft_adapter(tmp_path / "a2", 2, a2, text_only=True, seed=12)[0]}


def _llm(paths, with_pool=True):
    rows = Q._Rows()
    pool = LoraAdapterPool(Q.lora_spec(paths), "cpu", backend=LoraOpsBackend()) if with_pool else None
    llm = Q.bind(Q.build_stack(Q.hf_text_weights()), rows, {LORA_RESOURCE_KEY: pool} if pool is not None else {})
    return llm, pool, rows


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


def test_qwen3_5_stack_with_adapters_compiles_and_matches_eager(qwen_adapters):
    """Adapters add no graph break to the layer stack (the plain stack breaks at its
    per-layer cursor moves, compiled out on purpose), and Inductor equals eager, whose
    parity with merged weights test_lora_models checks."""
    batch = [("r0", "a1", 5), ("r1", None, 3), ("r2", "a2", 4), ("r3", "a1", 2)]
    llm, pool, rows = _llm(qwen_adapters)
    inputs = [ARNodeInputs(input_seq_len=n, input_ids=torch.zeros(n, dtype=torch.long)) for *_, n in batch]
    Q.admit_declared(pool, engine_step(llm, "prefill_text", [r for r, *_ in batch], inputs), batch, "prefill_text")
    rows.lengths = [n for *_, n in batch]
    x = torch.randn(sum(rows.lengths), Q.HIDDEN, generator=torch.Generator().manual_seed(7), dtype=F64)

    plain, _, plain_rows = _llm(qwen_adapters, with_pool=False)
    plain_rows.lengths = rows.lengths

    def breaks(model):
        torch._dynamo.reset()
        return torch._dynamo.explain(lambda x: model.model.model(x, label="main"))(x).graph_break_count

    def fwd(x):
        return llm.model.model(x, label="main")

    with torch.no_grad():
        assert breaks(llm) == breaks(plain)
        eager = fwd(x)
        torch._dynamo.reset()
        compiled = torch.compile(fwd, backend="inductor")(x)
    # the stack without LoRA differs between eager and Inductor by ~1e-6 on its own
    torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-5)


def _capture_plan(llm, pool, config, bs, n):
    """The runner's warm-up for one bucket; returns the addressing the graph records."""
    walk, rids = config.capture_graph_walk, list(range(-1, -1 - bs, -1))
    for rid in rids:
        pool.ingest_capture_row(rid)
    lease = SlotLease(slot=0, bucket=BucketKey(walk, bs, n, config.additional_key_info))
    step = engine_step(llm, walk, rids, config.get_node_inputs(bs, n), slot_lease=lease).steps[LORA_RESOURCE_KEY]
    ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=True, slot_lease=lease,
                      force_prefill=isinstance(config, PackedAcceleratorGraphConfig))
    assert pool.admit(step, ctx).ok
    return pool.plan(step, ctx), lease


def test_lora_changes_no_capture_bucket_and_captures_on_the_zero_slot(qwen_adapters):
    """Buckets and keys are the plain node's; every capture, the mixed walk's included, spans
    its whole bucket on the zero adapter in 16-byte aligned arrays and takes no slot."""
    (plain, *_), (served, pool, _) = _llm(qwen_adapters, with_pool=False), _llm(qwen_adapters)
    assert served.segment_keys == (LORA_RESOURCE_KEY,) and not served.disable_torch_compile
    configs = served.get_accelerator_graph_configs(CPU)

    def buckets(configs):
        return {(c.capture_graph_walk, c.additional_key_info, bs, n, type(c), getattr(c, "token_layout", None))
                for c in configs for bs in (c.capture_batch_sizes or ()) for n in c.get_total_tokens(bs)}

    assert configs and buckets(configs) == buckets(plain.get_accelerator_graph_configs(CPU))
    pool.ingest_request("live", LoraReqConfig(adapter="a1"))  # resident, not in any capture
    num_slots, seen = pool.spec.num_slots, 0
    for config in configs:
        walk = config.capture_graph_walk
        for bs in config.capture_batch_sizes:
            inputs, rids = config.get_node_inputs(bs, config.get_total_tokens(bs)[0]), list(range(-1, -1 - bs, -1))
            with_lora, without = engine_step(served, walk, rids, inputs), engine_step(plain, walk, rids, inputs)
            assert with_lora.cg_key_info == without.cg_key_info
            assert set(with_lora.steps) - set(without.steps) == {LORA_RESOURCE_KEY}
            for n in config.get_total_tokens(bs):
                addr, lease = _capture_plan(served, pool, config, bs, n)
                assert addr.token_order.shape == (n,), (walk, bs, n)
                assert addr.slot_indptr.tolist() == [0] + [n] * num_slots
                assert P.token_slots_of(addr) == [ZERO_SLOT] * n
                assert addr.token_order.data_ptr() % 16 == 0 and addr.slot_indptr.data_ptr() % 16 == 0
                if walk == "decode":  # its own bucket's size, not the largest's
                    buf = pool._cg[next(k for k in pool._cg if k.bucket == lease.bucket)]
                    assert buf.data.numel() == _Layout.of(bs, n, num_slots, pool._groupings).size
                seen += 1
    assert seen > 50
    assert all(pool.slot_state(i).refcount == 0 for i in range(num_slots))


@pytest.mark.parametrize("walk,bs,n,rows", [
    ("decode", 8, 8, [("d0", "decode", 1), ("d1", "decode", 1), ("d2", "decode", 1)]),
    # decode rows, then a prompt, then an empty padding row
    (MIXED, 4, 32, [("d0", "decode", 1), ("d1", "decode", 1), ("p0", "prefill_text", 9)]),
])
def test_a_replay_reaches_the_graph_through_the_captured_views(qwen_adapters, walk, bs, n, rows):
    """Replayed on the captured views at the bucket's width, each real token gets its own
    adapter's delta exactly as an eager step of the same rows; the tail is untouched."""
    llm, pool, _ = _llm(qwen_adapters)
    named = {"d0": "a1", "d1": None, "d2": "a2", "p0": "a2"}
    for rid, _, _ in rows:
        pool.ingest_request(rid, None if named[rid] is None else LoraReqConfig(adapter=named[rid]))
    config = next(c for c in llm.get_accelerator_graph_configs(CPU) if c.capture_graph_walk == walk)
    captured, lease = _capture_plan(llm, pool, config, bs, n)
    # the replay's plan, padded as AcceleratorGraphRunner pads it, on the capture's handles
    rids = [rid for rid, _, _ in rows]
    inputs = [ARNodeInputs(input_seq_len=s, input_ids=torch.zeros(s, dtype=torch.long), graph_walk=w)
              for _, w, s in rows] + (config.get_node_inputs(bs - len(rows), 0) if bs > len(rows) else [])
    padded = [*rids, *list(range(-1, -1 - bs, -1))[len(rids):]]
    step = engine_step(llm, walk, padded, inputs, slot_lease=lease).steps[LORA_RESOURCE_KEY]
    ctx = StepContext(request_ids=rids, graph_walk=walk, slot=0, capture=False, slot_lease=lease)
    ctx.set_padded_rids(padded)
    assert pool.admit(step, ctx).ok
    pool.plan(step, ctx)
    assert pool.addressing().token_order.data_ptr() == captured.token_order.data_ptr()
    pool.commit(step, ctx)

    real, g = sum(span for _, _, span in rows), torch.Generator().manual_seed(11)
    for target, layer in [("qkv_proj", 1), ("o_proj", 3), ("gate_up_proj", 0), ("down_proj", 2)]:
        t, d = pool.targets[target], pool.layer_delta(target, layer)
        x = torch.randn(n, t.in_features, generator=g, dtype=F64)
        base = torch.randn(n, t.out_features, generator=g, dtype=F64)
        replayed = base.clone()
        LoraOpsBackend().apply_(x, replayed, d.a, d.b, captured, d.ranks, d.scalings, d.out_slices)
        eager_step = ResourceStep(segments=tuple(Segment(rid, "main", span) for rid, _, span in rows))
        eager_ctx = StepContext(request_ids=rids, graph_walk="eager", slot=0, capture=False)
        assert pool.admit(eager_step, eager_ctx).ok
        pool.plan(eager_step, eager_ctx)
        torch.testing.assert_close(replayed[:real], d(x[:real], base[:real].clone()), rtol=1e-12, atol=1e-12)
        assert torch.equal(replayed[real:], base[real:])
        at = 0
        for rid, _, span in rows:
            # a1 touches every target; a2 only q/v and down; the base model none
            touches = named[rid] == "a1" or (named[rid] == "a2" and target in ("qkv_proj", "down_proj"))
            assert (not torch.equal(replayed[at:at + span], base[at:at + span])) == touches, (rid, target)
            at += span


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
    step = SubmoduleStep(segments=[Segment(r, "main", 1) for r in rids], steps={k: ResourceStep() for k in keys})
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
def test_a_warm_up_outside_any_step_runs_a_bound_module_as_an_idle_step(compiled, monkeypatch):
    """The engine warms compile_regions (and warmup_compiles) outside any step: a bound
    module there adds no delta, eager or traced, instead of refusing for want of a plan,
    and the pool is left with none."""
    from lora_module_helpers import pool as lora_pool
    from lora_module_helpers import serve

    from mstar.model.components.lora.modules import bind_module_lora, module_lora_spec
    from mstar.testing_utils.engine import make_engine, make_mgmt

    real = torch.compile
    monkeypatch.setattr(torch, "compile", (lambda fn, **kw: real(fn, backend="eager", **kw)) if compiled
                        else (lambda fn, **kw: fn))
    torch._dynamo.reset()
    spec = module_lora_spec(lambda _: _Warmed(), nodes={"N"}, root_path="")
    node, pool = _Warmed(), lora_pool(serve(spec, {}))
    bind_module_lora(node, {"lora": pool})
    engine = make_engine(
        _device_capability=SimpleNamespace(is_available=True),
        _submodules={"N": make_mgmt(node, resources={"lora": pool})},
    )
    engine._compile_regions()
    assert pool._current is None and pool.active
    with pytest.raises(RuntimeError, match="no plan"):
        node.fc(torch.ones(2, 4, dtype=F64))


# --- idle captures ----------------------------------------------------------------------

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


def test_every_bucket_is_captured_again_idle_beside_its_plain_one():
    """A node with an idle-capable resource captures each bucket twice, sharing the config;
    one without captures exactly what it did. An idle bucket is kept only beside its plain
    one, and is not reported as running eagerly while that one replays its steps."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner, _with_plain_twin

    configs = [PackedAcceleratorGraphConfig(
        capture_graph_walk="prefill", capture_token_lengths=[16, 32], capture_batch_sizes=[1, 2],
        make_node_input=lambda n: ARNodeInputs(input_seq_len=n), additional_key_info=cfg,
    ) for cfg in (True, False)]

    def runner(idle_capture):
        runner = object.__new__(AcceleratorGraphRunner)
        runner._capture_configs, runner._num_slots = configs, 2
        runner._resources = {"res": SimpleNamespace(idle_capture=idle_capture)}
        runner._submodule_name, runner._blocked_buckets = "LLM", []
        return runner

    plain, served = runner(False)._get_slot_specs(), runner(True)._get_slot_specs()
    assert {s.bucket.cg_key_info for s in plain} == {True, False} and not any(s.bucket.idle for s in plain)
    own = {(s.bucket, s.slot, s.config_idx) for s in plain}
    assert {(s.bucket, s.slot, s.config_idx) for s in served} == own | {
        (BucketKey(b.graph_walk, b.bs, b.num_tokens, b.cg_key_info, idle=True), slot, idx) for b, slot, idx in own
    }
    big, small = BucketKey("prefill", 2, 32, True), BucketKey("prefill", 1, 16, True)
    big_idle, small_idle = BucketKey("prefill", 2, 32, True, idle=True), BucketKey("prefill", 1, 16, True, idle=True)
    assert _with_plain_twin({big, big_idle, small_idle}) == {big, big_idle}
    reporting = runner(True)
    reporting._report_dropped([big, big_idle, small, small_idle], {big, small})
    assert reporting.dropped_buckets == []
    reporting._report_dropped([big, big_idle, small, small_idle], {big, big_idle})
    assert reporting.dropped_buckets == [small, small_idle]


def test_only_a_config_whose_step_drives_the_idle_resource_captures_idle_twins():
    """A config whose declared step carries no step under the idle-capable resource's key
    would capture the same kernels twice."""
    from unittest.mock import Mock

    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    configs = [BatchedAcceleratorGraphConfig(
        capture_graph_walk="decode", capture_batch_sizes=[1, 2], additional_key_info=drives,
        single_request_inputs=NodeInputs(tensor_inputs={}, input_seq_len=1),
    ) for drives in (True, False)]

    def declare_step(slot_lease, **kwargs):
        return SimpleNamespace(steps={"lora": object()} if slot_lease.bucket.cg_key_info else {})

    resource = SimpleNamespace(can_capture=lambda: True, force_double_buffer=False, idle_capture=True)
    runner = AcceleratorGraphRunner(
        submodule_name="LLM", resources={"lora": resource}, step_runner=Mock(), device=CPU,
        submodule=SimpleNamespace(get_accelerator_graph_configs=lambda *args: configs, declare_step=declare_step),
        autocast_dtype=None, joint_comm_group=SimpleNamespace(world_size=1), num_slots=2,
    )

    buckets = {spec.bucket for spec in runner.prepare_for_capture()}
    assert {(b.cg_key_info, b.idle) for b in buckets} == {(True, False), (True, True), (False, False)}
    assert len(buckets) == 6 and not runner.dropped_buckets


def test_a_node_takes_at_most_one_idle_capable_resource():
    """A step is idle or not as a whole, so one idle twin cannot leave two resources out."""
    from mstar.testing_utils.engine import make_mgmt

    resource = SimpleNamespace(idle_capture=True, force_double_buffer=False, supports_preplan=False)
    with pytest.raises(ValueError, match="at most one idle-capable resource"):
        make_mgmt(resources={"lora": resource, "vision_lora": resource})


def _mgmt(idle_keys, runner=None, adapters=None):
    """A node whose idle-capable resources answer from ``adapters`` (rid -> the adapter
    it was ingested with), as a pool answers from its ingests."""
    from mstar.testing_utils.engine import make_mgmt

    named = adapters or {}
    resource = SimpleNamespace(
        idle_capture=True, force_double_buffer=False, supports_preplan=False,
        idle_for=lambda rids: all(named.get(rid) is None for rid in rids),
    )
    return make_mgmt(
        SimpleNamespace(cg_key_info=lambda walk, info, **kw: "cfg"),
        resources=dict.fromkeys(idle_keys, resource), accelerator_graph_runner=runner,
    )


def _engine_with(captured, idle_keys=(LORA_RESOURCE_KEY,), adapters=None):
    """An engine whose node captured the decode buckets ``captured``, (bs, idle) pairs."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphBucket, AcceleratorGraphRunner
    from mstar.testing_utils.engine import make_engine

    config = BatchedAcceleratorGraphConfig(
        capture_graph_walk="decode", single_request_inputs=NodeInputs(tensor_inputs={}, input_seq_len=1),
    )
    runner = object.__new__(AcceleratorGraphRunner)
    runner._buckets = {
        BucketKey("decode", bs, bs, "cfg", idle=idle): AcceleratorGraphBucket(config, 0, slots=[None])
        for bs, idle in captured
    }
    return make_engine(_submodules={"LLM": _mgmt(idle_keys, runner, adapters)})


def _batch(rids):
    infos = {rid: SimpleNamespace(resource_configs={}) for rid in rids}
    batch = SimpleNamespace(node_name="LLM", slot=0, request_ids=list(infos), inputs=None, cg_key_info=None,
                            per_request_info=infos, per_request_input_metadata={}, per_request_info_wrapped={},
                            step_context=StepContext(request_ids=list(infos), graph_walk="decode", slot=0,
                                                     capture=False))
    batch.lease_slot = lambda lease: setattr(batch.step_context, "slot_lease", lease)
    return batch


@pytest.mark.parametrize("rows,captured,leased,ctx_idle", [
    # base rows replay the idle capture
    ({"b0": None, "b1": None}, {(8, False), (8, True)}, (8, True), True),
    # one adapter row: the plain capture
    ({"b0": None, "r": "a1"}, {(8, False), (8, True)}, (8, False), False),
    # the tightest plain bucket has no idle twin (not captured, or dropped): it serves the
    # step rather than a looser idle bucket, and the resource plans for its kernels
    ({"b0": None}, {(1, False), (8, False), (8, True)}, (1, False), False),
    # nothing captured: eager, and idle
    ({"b0": None}, set(), None, True),
])
def test_the_lease_takes_the_idle_twin_only_when_no_row_uses_the_resource(rows, captured, leased, ctx_idle):
    engine = _engine_with(captured, adapters=rows)
    batch = _batch(rows)
    lease = engine.reserve_replay_slot(batch)
    assert (lease and (lease.bucket.bs, lease.bucket.idle)) == leased
    assert batch.step_context.idle is ctx_idle and batch.cg_key_info == "cfg"
    # a node without an idle-capable resource leases the plain capture, and is never idle
    engine = _engine_with(captured, idle_keys=(), adapters=rows)
    batch = _batch(rows)
    lease = engine.reserve_replay_slot(batch)
    assert (lease is None or not lease.bucket.idle) and not batch.step_context.idle


def test_each_nodes_pool_reads_the_config_the_conductor_opens_a_request_with():
    """One spec over two nodes is served as a pool per node (``serving_resources``),
    keyed ``lora@<node>`` in the conductor as on the workers: a request on an adapter is
    opened with a config under each pool's key. A base-model request carries none."""
    import dataclasses

    from mstar.conductor.conductor import Conductor
    from mstar.engine.resources.lora import LORA_ADAPTER_KWARG
    from mstar.testing_utils.model import make_model

    spec = P.make_spec({})
    spec.nodes = {"LLM", "vision"}
    spec.targets = {"qkv": dataclasses.replace(P.QKV, node="LLM"), "o": dataclasses.replace(P.O, node="vision")}
    conductor = object.__new__(Conductor)
    conductor.model = make_model(get_node_resources=lambda: [spec], get_request_resource_configs=lambda **kw: {})
    conductor.model_config = {"resources": {LORA_RESOURCE_KEY: {}}}
    conductor._kv_labels = {}
    configs = conductor._get_resource_configs({LORA_ADAPTER_KWARG: "a1"}, {})
    assert configs == {"lora@LLM": LoraReqConfig("a1"), "lora@vision": LoraReqConfig("a1")}
    assert conductor._get_resource_configs({}, {}) == {}


def test_an_idle_lease_checks_the_models_key():
    """The lease/declare_step check compares the model's own key, the same under an idle
    lease."""
    from mstar.testing_utils.engine import make_engine

    def plan(step, phase):
        raise NotImplementedError("reached plan")

    engine = make_engine(_runner=SimpleNamespace(plan=plan))
    lease = SlotLease(slot=0, bucket=BucketKey("decode", 1, 1, "cfg", idle=True))
    ctx = StepContext(request_ids=[1], graph_walk="decode", slot=0, capture=False, slot_lease=lease, idle=True)

    def drive(key, info):
        batch = SimpleNamespace(node_name="LLM", preplan_event=None, per_request_info={1: info})
        engine._drive_step(batch, _mgmt((LORA_RESOURCE_KEY,)), [1], [], {}, ctx, lease, True,
                           SubmoduleStep(steps={}, cg_key_info=key), False)

    with pytest.raises(RuntimeError, match="disagree"):
        drive("other", SimpleNamespace(resource_configs={}))
    with pytest.raises(NotImplementedError, match="reached plan"):
        drive("cfg", SimpleNamespace(resource_configs={}))


def test_an_idle_step_launches_no_delta_and_a_plain_lease_stages_addressing(adapters):
    """The pool reads only ``ctx.idle``. Idle, captured or eager, it stages nothing and
    launches nothing. A step of base rows replaying the plain bucket (its idle twin was not
    captured) stages its rows on the zero adapter, which the recorded kernels read."""
    pool = LoraAdapterPool(P.make_spec(adapters[1]), "cpu", backend=(backend := _CountingBackend()))
    P.ingest(pool, "b", None)
    P.ingest(pool, "r", "a1")
    rids = [-1, -2]
    idle = SlotLease(slot=0, bucket=BucketKey("decode", 2, 2, None, idle=True))
    s, c = P.step(rids), P.ctx(rids, capture=True, lease=idle, idle=True)
    assert pool.admit(s, c).ok and pool.plan(s, c) is None
    assert not pool.active and not pool._cg and not _delta_runs(pool, backend)
    pool.commit(s, c)
    plain = SlotLease(slot=0, bucket=BucketKey("decode", 2, 2, None))
    s, c = P.step(rids), P.ctx(rids, capture=True, lease=plain)
    assert pool.admit(s, c).ok and pool.plan(s, c) is not None
    assert pool.active and _delta_runs(pool, backend)
    pool.commit(s, c)
    # the plain-bucket fallback: a base row and a padding row, both on the zero adapter
    s, c = P.step(["b", -2]), P.ctx(["b"], lease=plain, padded=["b", -2])
    assert pool.admit(s, c).ok and pool.plan(s, c) is not None
    assert pool.active and P.token_slots_of(pool.addressing()) == [0, 0]
    pool.commit(s, c)
    for rows, step_idle in ((["b"], True), (["b", "r"], False), (["b"], True)):
        s, c = P.step(rows), P.ctx(rows, idle=step_idle)
        assert pool.admit(s, c).ok
        pool.plan(s, c)
        assert _delta_runs(pool, backend, tokens=len(rows)) is not step_idle, rows
        pool.commit(s, c)


def test_a_piecewise_region_inside_an_idle_step_hands_its_state_back(adapters):
    """A region whose capture records LoRA kernels plans its own rows (active), and its commit restores
    the enclosing idle step's state for the eager code after it. Past the step, and
    past a region's capture (which plans outside any step), no plan is left: a delta
    refuses, eager or compiled, rather than add an idle skip or a capture's rows."""
    pool = LoraAdapterPool(P.make_spec(adapters[1]), "cpu", backend=(backend := _CountingBackend()))
    P.ingest(pool, "b", None)
    outer_s, outer_c = P.step(["b"]), P.ctx(["b"], idle=True)
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
        device=CPU, autocast_dtype=None, num_slots=2, node_name="N", segment_keys=("lora",),
    )


def test_a_piecewise_region_plans_its_own_lora_rows_and_one_row_replays_only_idle_rows(adapters):
    """A region's step carries the node's LoRA rows, laid out from its segments as the
    node's are. A region replayed as one row per step, which addresses every row to the
    first request's adapter, is captured idle, recording no LoRA kernel, and leased only
    for a step none of whose rows runs one; a step with one runs the region eagerly under
    the node's step."""
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
    assert {(spec.bucket.cg_key_info, spec.bucket.idle) for spec in runner._step_runner.built} == {("N_region", False)}
    assert {(spec.bucket.cg_key_info, spec.bucket.idle) for spec in one_row._step_runner.built} == {("N_region", True)}
    assert one_row.can_lease([], ["b0", "b1"]) and not one_row.can_lease([], ["b0", "r1"])
    # every row, not only the first one the replay names
    with pytest.raises(RuntimeError, match="replays one row per step"):
        one_row.run({}, request_ids=["b0", "r1"])
    assert runner.can_lease([], ["b0", "r1"])


def test_the_idle_choice_is_the_same_on_every_rank(adapters):
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
            assert [not pool.idle_for(rows) for pool in ranks] == [want, want], rows
    # once the adapter rows are removed, the shortcut agrees too
    for pool in ranks:
        pool.remove_request("r1")
        pool.remove_request("r2")
        assert not pool._live and pool.idle_for(list(named))
