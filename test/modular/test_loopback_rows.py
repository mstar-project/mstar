"""Device loop-back rows (``mstar.engine.loopback_rows``): a node's loop-back
values kept on the device by request, written at a step's tail and read at the
next step's head, and which walks may keep them (``Engine.keep_loopback_rows``)."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.engine import Engine
from mstar.engine.loopback_rows import TRASH_ROW, LoopbackRows
from mstar.model.submodule_base import CarryPair, DeviceLoopback

CPU = torch.device("cpu")
WALK = "decode"


def test_rows_are_read_by_request_and_padding_reads_the_trash_row():
    rows = LoopbackRows(CPU)
    for rid in ("a", "b"):
        rows.add(rid)
    rows.write("t", ["b", "a", -1], torch.tensor([2, 1, 99]))
    assert rows.read("t", ["a", "b"]).tolist() == [1, 2]
    assert rows.read("t", [-2]).tolist() == [99], "a padding rid reads what padding wrote, the trash row"
    assert rows._row_of["a"] != TRASH_ROW and rows._row_of["b"] != TRASH_ROW


def test_growth_keeps_every_row_and_a_dropped_row_is_handed_on():
    rows = LoopbackRows(CPU)
    rids = list(range(20))  # past the first table's rows
    for rid in rids[:3]:
        rows.add(rid)
    rows.write("h", rids[:3], torch.arange(6.0).reshape(3, 2))
    for rid in rids[3:]:
        rows.add(rid)
    rows.write("h", rids[3:], torch.full((17, 2), 7.0))
    assert rows.read("h", rids[:3]).tolist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    row = rows._row_of[1]
    rows.drop(1)
    rows.add("new")
    assert rows._row_of["new"] == row
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


def test_a_steps_tail_keeps_its_token_its_states_and_its_kept_signals():
    """The token from its stop buffer, a declared state and a kept loop-back
    from row outputs, all by the step's (padded) request ids; the next step's
    head reads each as the last step left it."""
    rows = LoopbackRows(CPU)
    for rid in (1, 2):
        rows.add(rid)
    loopback = DeviceLoopback(walk=WALK, signal="text_inputs", buffer="new_token", states=("rules",))
    mgmt = _mgmt(rows, loopback, {WALK: {"h": "h_out"}}, runner=object())
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
    engine._device = CPU
    mgmt = _mgmt(None, runner=_runner(shapes), world_size=world_size)
    engine._submodules = {"n": mgmt}
    assert engine.keep_loopback_rows("n", WALK, {"h"}) == why
    if why is None:
        assert mgmt.row_signals == {WALK: {"h": "h_out"}} and isinstance(mgmt.loopback_rows, LoopbackRows)
        assert engine.device_loopback_signals("n", WALK) == {"h"}
    else:
        assert mgmt.row_signals == {} and engine.device_loopback_signals("n", WALK) == frozenset()
