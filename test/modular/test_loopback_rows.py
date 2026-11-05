"""Device loop-back rows (``mstar.engine.resources.loopback_rows``): a node's
loop-back values kept on the device by request, written at a step's tail and
read at the next step's head, and which walks may keep them (``Engine.rows_refusal``)."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.engine import Engine
from mstar.engine.resources.loopback_rows import TRASH_ROW, LoopbackRows
from mstar.engine.resources.step import BucketKey, SlotLease
from mstar.model.submodule_base import CarryPair, DeviceLoopback

CPU = torch.device("cpu")
WALK = "decode"


def test_rows_are_read_by_request_and_padding_reads_the_trash_row():
    rows = LoopbackRows(CPU, capacity=4)
    for rid in ("a", "b"):
        rows.add(rid)
    rows.write("t", ["b", "a", -1], torch.tensor([2, 1, 99]))
    assert rows.read("t", ["a", "b"]).tolist() == [1, 2]
    assert rows.read("t", [-2]).tolist() == [99], "a padding rid reads what padding wrote, the trash row"
    assert rows._row_of["a"] != TRASH_ROW and rows._row_of["b"] != TRASH_ROW


def test_a_table_keeps_its_address_and_a_dropped_row_is_handed_on():
    """Captures hold a table's address, so it never grows: a request past its rows waits."""
    rows = LoopbackRows(CPU, capacity=3)
    rids = [0, 1, 2]
    for rid in rids:
        rows.add(rid)
    rows.write("h", rids, torch.arange(6.0).reshape(3, 2))
    table = rows._tables["h"]
    with pytest.raises(RuntimeError, match="MSTAR_SAMPLER_SLOTS"):
        rows.add("late")
    row = rows._row_of[1]
    rows.drop(1)
    rows.add("new")
    assert rows._row_of["new"] == row and rows._tables["h"] is table
    rows.write("h", ["new"], torch.tensor([[8.0, 8.0]]))
    assert rows.read("h", [0, "new", 2]).tolist() == [[0.0, 1.0], [8.0, 8.0], [4.0, 5.0]]


def _mgmt(rows, loopback=None, row_signals=None, runner=None, world_size=1):
    submodule = SimpleNamespace(
        device_loopback=loopback,
        repeat_carry=lambda walk, signals: {s: CarryPair(f"{s}_out", s) for s in signals},
    )
    return SimpleNamespace(
        submodule=submodule, loopback_rows=rows, row_signals=row_signals or {}, accelerator_graph_runner=runner,
        joint_comm_group=SimpleNamespace(world_size=world_size),
    )


def test_an_eager_steps_tail_keeps_its_token_its_states_and_its_kept_signals():
    """The token from its stop buffer, a declared state and a kept loop-back
    from row outputs, all by the step's request ids. A capture does this itself."""
    rows = LoopbackRows(CPU)
    for rid in (1, 2):
        rows.add(rid)
    loopback = DeviceLoopback(walk=WALK, signal="text_inputs", buffer="new_token", states=("rules",))
    mgmt = _mgmt(rows, loopback, {WALK: {"h": CarryPair("h_out", "h")}}, runner=object())
    outputs = SimpleNamespace(
        check_stop_buffers={"new_token": torch.tensor([5, 6, 0])},
        row_outputs={"rules": torch.tensor([[1, 1], [2, 2], [0, 0]]), "h_out": torch.ones(3, 4)},
        packed_outputs={},
    )
    Engine._keep_rows(Engine.__new__(Engine), mgmt, WALK, [1, 2, -1], outputs)
    assert rows.read("text_inputs", [2, 1]).tolist() == [6, 5]
    assert rows.read("rules", [1, 2]).tolist() == [[1, 1], [2, 2]]
    assert rows.read("h", [1]).tolist() == [[1.0] * 4]
    # a mixed step keeps the token too (its decode rows read it next), not the walk's kept signals
    outputs.check_stop_buffers["new_token"] = torch.tensor([7, 8, 0])
    Engine._keep_rows(Engine.__new__(Engine), mgmt, "decode+prefill", [1, 2, -1], outputs)
    assert rows.read("text_inputs", [1, 2]).tolist() == [7, 8]


def test_a_capture_reads_its_rows_at_its_head_and_writes_them_at_its_tail_with_no_host_between():
    """What a replay does, in order, through the device index the plan staged:
    each replay's head reads what the last one's tail wrote, a padding row goes
    to the trash row, and a host write lands before the first head."""
    rows = LoopbackRows(CPU)
    for rid in (1, 2):
        rows.add(rid)
    bucket = BucketKey(WALK, 3, 3)
    ctx = SimpleNamespace(
        slot_lease=SlotLease(slot=0, bucket=bucket), resident=None, padded_request_ids=(2, 1, -1),
    )
    rows.plan(None, ctx)
    static = torch.zeros(3, dtype=torch.long)
    assert not rows.head(bucket, 0, "t", static), "no table yet: the first warm-up"
    rows.tail(bucket, 0, "t", torch.tensor([0, 0, 0]))
    rows.write_rows("t", [1, 2], torch.tensor([9, 20, 10]), [1, 0])  # two rows' routed first values
    for _ in range(3):
        assert rows.head(bucket, 0, "t", static)
        rows.tail(bucket, 0, "t", static + 1)
    assert rows.read("t", [2, 1]).tolist() == [12, 23]
    assert static.tolist()[:2] == [11, 22]


def test_only_a_routed_row_is_written_ahead_of_a_captures_head():
    rows = LoopbackRows(CPU)
    for rid in (1, 2):
        rows.add(rid)
    rows.write("t", [1, 2], torch.tensor([5, 6]))
    mgmt, slot = SimpleNamespace(loopback_rows=rows), SimpleNamespace(row_heads={"t": "input_ids"})
    batch = SimpleNamespace(request_ids=[1, 2], kept_rows=[True, False])
    inputs = SimpleNamespace(request_ids=[1, 2, -1])
    Engine._write_routed_rows(batch, mgmt, slot, inputs, {"input_ids": torch.tensor([0, 9, 0])})
    assert rows.read("t", [1, 2]).tolist() == [5, 9]
    batch.kept_rows = [True, True]
    Engine._write_routed_rows(batch, mgmt, slot, inputs, {"input_ids": torch.tensor([0, 0, 0])})
    assert rows.read("t", [1, 2]).tolist() == [5, 9]


def test_a_resident_step_gathers_its_rows_index_from_the_planners_slots():
    """No host index: the slot's cell names the row, and a padding row's zero is the trash row."""
    rows = LoopbackRows(CPU)
    rows.add(7)
    cells = torch.zeros(8, dtype=torch.long)
    planner = SimpleNamespace(store=cells, table=lambda: SimpleNamespace(stride=2, col=1))
    rows.bind_resident(planner, "rows")
    cell = [0, 0]
    rows.resident_row(7, 3, cell, None)
    cells[3 * 2 + 1] = cell[1]
    bucket = BucketKey(WALK, 2, 2)
    rows.resident_gather(SimpleNamespace(bucket=bucket, slot=0, slots=torch.tensor([3, 0])))
    assert rows.indices(bucket, 0).tolist() == [rows._row_of[7], TRASH_ROW]


def test_a_request_waits_for_a_row_and_the_ranks_of_a_lockstep_node_never_do():
    rows = LoopbackRows(CPU, capacity=1)
    assert rows.ask(1, "n", WALK) is not None and rows.reserve(1, "n", WALK, None) is None
    assert rows.ask(1, "n", WALK) is None, "it holds its row"
    mark = rows.reserve(2, "n", WALK, None)
    assert mark == 0 and not rows.grew_since(mark)
    rows.remove_request(1)
    assert rows.grew_since(mark) and rows.reserve(2, "n", WALK, None) is None
    lockstep = LoopbackRows(CPU, capacity=1, decides=False)
    assert lockstep.ask(1, "n", WALK) is None
    lockstep.ingest_request(1)
    assert lockstep.has_row(1)


def _runner(shapes):
    """A capture runner whose WALK captures leave ``h_out`` at each of ``shapes``."""
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._buckets = {
        i: SimpleNamespace(
            config=SimpleNamespace(capture_graph_walk=WALK),
            slots=[SimpleNamespace(static_outputs=SimpleNamespace(
                row_outputs={"h_out": torch.zeros(shape)}, check_stop_buffers={}, packed_outputs={},
            ))],
        )
        for i, shape in enumerate(shapes)
    }
    return runner


@pytest.mark.parametrize(("shapes", "world_size", "why"), [
    ([(1, 4), (8, 4)], 1, None),
    ([(1, 4), (8, 6)], 1, "runner: loop-back 'h' changes shape with the bucket"),
    ([], 1, "runner: it has no capture"),
    ([(1, 4)], 2, "deployment: it runs on more than one rank"),
], ids=["fixed", "variable_shape", "no_capture", "tp"])
def test_a_walk_keeps_a_loop_back_in_rows_only_at_a_fixed_shape_on_one_rank(shapes, world_size, why):
    engine = Engine.__new__(Engine)
    mgmt = _mgmt(LoopbackRows(CPU), runner=_runner(shapes), world_size=world_size)
    engine._submodules = {"n": mgmt}
    engine._ask_rows("n", WALK, {"h"})
    assert engine.rows_refusal("n", WALK) == why
    if why is None:
        assert mgmt.row_signals == {WALK: {"h": CarryPair("h_out", "h")}}
        assert engine.device_loopback_signals("n", WALK) == {"h"}
    else:
        assert mgmt.row_signals == {} and engine.device_loopback_signals("n", WALK) == frozenset()
