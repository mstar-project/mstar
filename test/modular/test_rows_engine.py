"""A real ``Engine`` driving a walk with persistent rows: every step runs in
row order, reads its requests' state from their rows, and leaves the rows
right across joins, leaves, steps of another walk, removals and handles that
come back as new requests.

The toy resource keeps a host count (mirrored in the rows) and a device
accumulator (checked out of a per-request store); each step adds one to
both, so a request's state is the number of steps it ran.
"""
import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.distributed.communication import WorkerParallelGroups  # noqa: E402
from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources import StepContext  # noqa: E402
from mstar.engine.resources.base import Resource  # noqa: E402
from mstar.engine.resources.spec import NodeResourceSpec  # noqa: E402
from mstar.engine.resources.step import ResourceStep, SubmoduleStep  # noqa: E402
from mstar.engine.rows import RowState, RowWalk  # noqa: E402
from mstar.model.submodule_base import NodeInputs, NodeSubmodule  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
NODE, ROWS_WALK, OTHER_WALK, MAX_ROWS, HANDLES = "toy", "decode", "other", 8, 32


class _CounterRows(RowState):
    def __init__(self, res: "_Counter", space):
        self.res = res
        self.count = space.buffer("count", (), torch.int32, step=1)
        self.acc = space.buffer("acc", (2,), torch.float32, mirrored=False)
        self.released_slots: list[int] = []

    def check_out(self, rows):
        d = rows.diff
        counts = np.array([self.res.count[rid] for rid in d.join_rids], np.int32)
        rows.write(self.count, d.join_slots, counts)
        rows.copy_in(self.acc, d.join_slots, self.res.store, np.array(d.join_rids))

    def check_in(self, rows):
        d = rows.diff
        keep = [i for i, rid in enumerate(d.leave_rids) if rid not in d.dropped]
        if not keep:
            return
        slots = d.leave_slots[keep]
        rids = [d.leave_rids[i] for i in keep]
        rows.copy_out(self.acc, slots, self.res.store, np.array(rids))

        def store_counts(released):
            gone = set(released.tolist())
            for slot, rid in zip(slots.tolist(), rids, strict=True):
                if slot not in gone:
                    self.res.count[rid] = int(self.count.host[slot])

        rows.on_commit(store_counts)

    def released(self, slots):
        self.released_slots.extend(slots.tolist())


class _Counter(Resource):
    keeps_rows = True

    @classmethod
    def build(cls, spec, info):
        return cls(info.device, spec.keeps_rows)

    def __init__(self, device, keeps_rows=True):
        self.device = device
        self.keeps_rows = keeps_rows
        self.count: dict[int, int] = {}
        self.store = torch.zeros(HANDLES, 2, device=device)
        self.rows_by_walk: dict[str, _CounterRows] = {}
        self.offloaded: dict[int, tuple[int, torch.Tensor]] = {}

    def ingest_request(self, rid, overrides):
        if rid not in self.count:
            self.count[rid] = 0
            self.store[rid] = 0

    def remove_request(self, rid):
        self.count.pop(rid, None)

    @property
    def supports_eviction(self):
        return True

    def is_offloaded(self, rid):
        return rid in self.offloaded

    def reclaimable(self, rid):
        return int(rid in self.count and rid not in self.offloaded)

    def offload(self, rid):
        # what the store holds for it, which a row that wasn't checked in
        # hasn't written back yet
        self.offloaded[rid] = (self.count[rid], self.store[rid].cpu().clone())
        self.store[rid] = -1
        self.count[rid] = -1
        return 1

    def reload(self, rid):
        count, acc = self.offloaded.pop(rid)
        self.count[rid] = count
        self.store[rid] = acc.to(self.device)
        return True

    @property
    def rows(self) -> _CounterRows:
        return self.rows_by_walk[ROWS_WALK]

    def row_state(self, space, node_name, graph_walk):
        if not self.keeps_rows:
            return None
        rows = self.rows_by_walk[graph_walk] = _CounterRows(self, space)
        return rows

    def plan(self, step, ctx):
        pass

    def commit(self, step, ctx):
        if ctx.rows is None:
            for rid in ctx.request_ids:
                self.count[rid] += 1


class _CounterSpec(NodeResourceSpec):
    keeps_rows: bool = True

    @property
    def resource_class(self):
        return _Counter


class _Toy(NodeSubmodule):
    def __init__(self, cap=MAX_ROWS, walks=(ROWS_WALK,)):
        super().__init__()
        self.cap = cap
        self.walks = frozenset(walks)

    def persistent_row_walks(self):
        return {walk: RowWalk() for walk in self.walks}

    def max_batch_size(self, graph_walk):
        return self.cap

    def can_batch(self, batch, model_inputs):
        return True

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        return NodeInputs(tensor_inputs={}, input_seq_len=1)

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        return SubmoduleStep(steps={"counter": ResourceStep()})

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {}

    def forward(self, graph_walk, engine_inputs, **kwargs):
        raise AssertionError("every step runs batched")

    def forward_batched(self, graph_walk, engine_inputs, **kwargs):
        res = engine_inputs.resources["counter"]
        rids = list(engine_inputs.request_ids)
        n = len(rids)
        if graph_walk in self.walks:
            rows = res.rows_by_walk[graph_walk]
            count = rows.count.rows(n).clone()
            acc = rows.acc.rows(n)[:, 0].clone()
            rows.acc.device[:n] += 1
        else:
            count = torch.tensor([res.count[rid] for rid in rids])
            index = torch.tensor(rids, device=res.store.device)
            acc = res.store[index, 0].clone()
            res.store[index] += 1
        return {
            rid: {"count": [count[i:i + 1]], "acc": [acc[i:i + 1]]}
            for i, rid in enumerate(rids)
        }


def _engine(device, sub=None, keeps_rows=True, max_live=None) -> Engine:
    spec = _CounterSpec(resource_key="counter", nodes={NODE})
    spec.keeps_rows = keeps_rows
    engine = Engine(None, autocast_dtype=None, max_concurrent_requests=max_live)
    engine.load_model(
        submodules={NODE: sub or _Toy()}, specs=[spec],
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device(device), transfer_engine_info=None,
    )
    engine.warmup()
    return engine


def _batch(walk, rids) -> ExecutingBatch:
    return ExecutingBatch(
        node_name=NODE,
        per_request_info={rid: CurrentForwardPassInfo(
            request_id=f"r{rid}", rid_handle=rid, graph_walk=walk,
            fwd_index=0, random_seed=0, max_tokens=1,
        ) for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
        per_request_input_tensors={rid: {} for rid in rids},
    )


def _step(engine, walk, rids):
    batch = _batch(walk, rids)
    engine.prepare_inputs(batch)
    out = engine.exec_and_postprocess(batch)
    return batch, {
        rid: (int(out.get(rid)["count"][0]), float(out.get(rid)["acc"][0]))
        for rid in rids
    }


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("max_live", [None, 16])
def test_steps_read_each_request_s_own_state_across_walks_and_removals(device, seed, max_live):
    rng = random.Random(seed)
    engine = _engine(device, max_live=max_live)
    space = engine._rows[NODE].space(ROWS_WALK)
    ran: dict[int, int] = {}  # rid -> steps it ran: its count and accumulator
    free = list(range(HANDLES))
    for _ in range(150):
        live = list(ran)
        if live and rng.random() < 0.15:
            rid = rng.choice(live)
            engine.remove_request(rid)
            del ran[rid]
            # a handle is recycled, often straight into the next step
            free.insert(0 if rng.random() < 0.5 else len(free), rid)
        while free and len(ran) < 12 and rng.random() < 0.3:
            rid = free.pop(0)
            engine.add_request(rid)
            ran[rid] = 0
        live = list(ran)
        if not live:
            continue
        walk = ROWS_WALK if rng.random() < 0.7 else OTHER_WALK
        rids = rng.sample(live, rng.randint(1, min(MAX_ROWS, len(live))))
        batch, seen = _step(engine, walk, rids)
        assert seen == {rid: (ran[rid], float(ran[rid])) for rid in rids}
        for rid in rids:
            ran[rid] += 1
        if walk == ROWS_WALK:
            # the step ran in row order, and the rows are its requests
            assert list(batch.request_ids) == list(space.batch.order)
            assert space.batch.n_held <= (max_live or MAX_ROWS)
            space.check_mirrors()
        else:
            assert not any(space.holds(rid) for rid in rids)


@pytest.mark.parametrize("device", DEVICES)
def test_a_removed_request_s_row_is_released_and_its_handle_rejoins_fresh(device):
    engine = _engine(device)
    rows = engine._rows[NODE]
    counter = engine._resources["counter"]
    for rid in (1, 2, 3):
        engine.add_request(rid)
    _step(engine, ROWS_WALK, [1, 2, 3])
    slot = rows.space(ROWS_WALK).batch.slot_of(2)
    engine.remove_request(2)
    assert not rows.holds_any([2])
    engine.add_request(2)
    _, seen = _step(engine, ROWS_WALK, [1, 2, 3])
    assert seen == {1: (1, 1.0), 2: (0, 0.0), 3: (1, 1.0)}
    assert counter.rows.released_slots == [slot]


@pytest.mark.parametrize("device", DEVICES)
def test_a_step_of_another_walk_waits_to_plan_until_its_rows_are_checked_in(device):
    engine = _engine(device)
    for rid in (1, 2):
        engine.add_request(rid)
    _step(engine, ROWS_WALK, [1, 2])
    assert not engine._enter_rows(_batch(OTHER_WALK, [2]), preplan=True)
    assert engine._enter_rows(_batch(OTHER_WALK, [3]), preplan=True)


@pytest.mark.parametrize("device", DEVICES)
def test_a_pre_plan_is_stale_once_its_rows_moved_on(device):
    engine = _engine(device)
    for rid in (1, 2):
        engine.add_request(rid)
    batch = _batch(ROWS_WALK, [1, 2])
    engine._enter_rows(batch)
    batch.preplanned_rids = tuple(batch.request_ids)
    assert not engine.preplan_is_stale(batch)
    _step(engine, ROWS_WALK, [2])
    assert engine.preplan_is_stale(batch)


@pytest.mark.parametrize("device", DEVICES)
def test_a_request_holding_a_row_is_evicted_with_the_state_its_row_held(device):
    engine = _engine(device)
    engine._graph_runtime = SimpleNamespace(get_rid_string=str)  # for the reload's log line
    space = engine._rows[NODE].space(ROWS_WALK)
    for rid in (1, 2, 3):
        engine.add_request(rid)
    for _ in range(3):
        _step(engine, ROWS_WALK, [1, 2, 3])
    assert engine.offload_request(NODE, 2) > 0
    assert not space.holds(2)
    space.check_mirrors()
    _step(engine, ROWS_WALK, [1, 3])
    assert engine.reload_request(NODE, 2)
    _, seen = _step(engine, ROWS_WALK, [1, 2, 3])
    assert seen == {1: (4, 4.0), 2: (3, 3.0), 3: (4, 4.0)}


@pytest.mark.parametrize("drop", [False, True])
def test_a_pre_plan_leaves_the_batch_to_the_main_thread(drop):
    """A pre-plan doesn't write the batch back: the main thread may be
    dropping rows from it meanwhile, which then makes the pre-plan stale."""
    engine = _engine("cpu")
    for rid in (1, 2, 3):
        engine.add_request(rid)
    _step(engine, ROWS_WALK, [1, 2])
    batch = _batch(ROWS_WALK, [3, 2, 1])
    node_rows = engine._rows[NODE]
    begin = node_rows.begin

    def begin_while_the_main_thread_drops(walk, rids):
        rows = begin(walk, rids)
        if drop:
            batch.request_ids = [r for r in batch.request_ids if r != 2]
        return rows

    node_rows.begin = begin_while_the_main_thread_drops
    order = engine._enter_rows(batch, preplan=True)
    node_rows.begin = begin
    assert list(order) == [3, 2, 1]
    batch.preplanned_rids = tuple(order)  # as _pre_plan_for_batch records it
    assert list(batch.request_ids) == ([3, 1] if drop else [3, 2, 1])
    assert engine.preplan_is_stale(batch) == drop


class _Captured(_Toy):
    """Capture sizes past the cap, which doesn't cap eager steps."""

    def get_accelerator_graph_configs(self, device, tp_world_size=1):
        return [BatchedAcceleratorGraphConfig(
            capture_graph_walk=ROWS_WALK,
            single_request_inputs=NodeInputs(tensor_inputs={}, input_seq_len=1),
            capture_batch_sizes=[8, 16, 32], caps_eager_batch_size=False, compile=False,
        )]


@pytest.mark.parametrize("cap,rows", [(24, 32), (16, 16), (40, 40)])
def test_rows_hold_the_capture_a_capped_step_pads_to(cap, rows):
    engine = _engine("cpu", sub=_Captured(cap=cap))
    space = engine._rows[NODE].space(ROWS_WALK)
    assert space.max_rows == rows
    with pytest.raises(ValueError, match="the space holds"):
        space.buffers[0].rows(rows + 1)


def _pre_planned_step(engine, rids):
    """What ``Engine._pre_plan_for_batch`` does to a batch, without a lease."""
    batch = _batch(ROWS_WALK, rids)
    order = engine._enter_rows(batch, preplan=True)
    step = engine._submodules[NODE].submodule.declare_step(
        graph_walk=ROWS_WALK, request_ids=list(order), inputs=None,
    )
    step.set_ctx(batch.step_context)
    batch.step_context.is_preplan = True
    engine._runner.pre_admit(step)
    engine._runner.pre_plan(step)
    batch.step_context.is_preplan = False
    batch.step = step
    batch.preplanned_rids = tuple(order)
    return batch


def test_a_pre_planned_step_runs_on_its_rows(monkeypatch):
    monkeypatch.setattr(_Counter, "supports_preplan", True, raising=False)
    engine = _engine("cpu")
    space = engine._rows[NODE].space(ROWS_WALK)
    for rid in (1, 2):
        engine.add_request(rid)
    _step(engine, ROWS_WALK, [1, 2])
    batch = _pre_planned_step(engine, [2, 1])
    engine.prepare_inputs(batch)
    assert not engine.preplan_is_stale(batch)
    out = engine.exec_and_postprocess(batch)
    assert {rid: int(out.get(rid)["count"][0]) for rid in (1, 2)} == {1: 1, 2: 1}
    space.check_mirrors()
    _, seen = _step(engine, ROWS_WALK, [1, 2])
    assert seen == {1: (2, 2.0), 2: (2, 2.0)}


@pytest.mark.parametrize("device", DEVICES)
def test_a_request_moving_between_two_persistent_walks_carries_its_state(device):
    engine = _engine(device, sub=_Toy(walks=(ROWS_WALK, OTHER_WALK)))
    node_rows = engine._rows[NODE]
    for rid in (1, 2):
        engine.add_request(rid)
    seen, last = [], None
    for walk in (ROWS_WALK, OTHER_WALK, OTHER_WALK, ROWS_WALK, OTHER_WALK):
        # a pre-plan waits for the check-in from the other walk
        waits = last not in (None, walk)
        assert (engine._enter_rows(_batch(walk, [1]), preplan=True) is None) == waits
        _, s = _step(engine, walk, [1, 2])
        last = walk
        seen.append(s[1])
        assert not node_rows.holds_any([1, 2], skip_walk=walk)
    assert seen == [(i, float(i)) for i in range(5)]


def test_a_resource_shared_with_another_node_refuses_rows_at_startup():
    spec = _CounterSpec(resource_key="counter", nodes={NODE, "peer"})
    engine = Engine(None, autocast_dtype=None)
    engine.load_model(
        submodules={NODE: _Toy(), "peer": _Toy(walks=())}, specs=[spec],
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cpu"), transfer_engine_info=None,
    )
    with pytest.raises(ValueError, match="shared with other nodes"):
        engine.warmup()


def test_a_resource_that_cannot_keep_rows_refuses_the_walk_at_startup():
    with pytest.raises(ValueError, match="resource 'counter'"):
        _engine("cpu", keeps_rows=False)


def test_a_walk_without_a_batch_cap_is_refused_at_startup():
    with pytest.raises(ValueError, match="no batch cap"):
        _engine("cpu", sub=_Toy(cap=None))
