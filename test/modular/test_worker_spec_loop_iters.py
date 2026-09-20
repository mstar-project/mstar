"""A step's loop-iter counts are fixed when its batch is built.

A speculated N+1 is in flight together with N for the same rid, at different
iterations, so each step carries its own counts (``Worker._step_info``). A
continuing rid's are predicted by its graph io (``speculative_loop_indices``);
a fresh rid's node was already routed, so the live counts are the truth.
"""

from types import SimpleNamespace

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    Loop,
    Sequential,
    SpeculativeNodeInfo,
)
from mstar.graph.graph_io import WorkerGraphIO
from mstar.worker.worker import Worker


def _info(rid="X"):
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk="decode", fwd_index=0, random_seed=0,
        max_tokens=8,
    )


def test_step_info_owns_its_counts_and_shares_the_rest():
    canonical = _info()
    canonical.dynamic_loop_iter_counts["decode_loop"] = 4
    view = Worker._step_info(canonical, {"decode_loop": 5})
    assert view.dynamic_loop_iter_counts == {"decode_loop": 5}
    assert canonical.dynamic_loop_iter_counts == {"decode_loop": 4}
    # publish / stop bookkeeping still lands on the canonical dicts
    assert view.resource_publish_info is canonical.resource_publish_info
    assert view.loop_stop_times is canonical.loop_stop_times
    assert view.step_metadata is canonical.step_metadata


def _ar_io(curr_iter):
    io = WorkerGraphIO(Sequential(sections=[
        Loop(
            name="decode_loop",
            section=GraphNode(
                name="decode", input_names={"token"},
                outputs=[GraphEdge(name="token", next_node="decode")],
            ),
            outputs=[], max_iters=100,
        ),
    ]))
    io.loops["decode_loop"].curr_iter = curr_iter
    return io


def _worker(live, wgio):
    w = Worker.__new__(Worker)
    w.worker_graphs_manager = SimpleNamespace(
        get_dynamic_loop_iters=lambda rid, partition: dict(live[rid]),
    )
    w._get_wgio_for_rid = lambda batch, rid: wgio
    return w


_NEW_ITER = SpeculativeNodeInfo(
    node_name="decode", is_new_loop_iter=True,
    loop_name="decode_loop", advancing_loop_name="decode_loop",
)
_PENDING = SimpleNamespace(partition="p0", batch=None)


def test_continuing_rid_gets_the_predicted_next_iteration():
    # io still reads 5 (N's routing lands after the spec submit): N+1 is 6
    w = _worker({"X": {"decode_loop": 5}}, _ar_io(5))
    assert w._spec_loop_iters(
        _PENDING, _NEW_ITER, "X", continuing=True,
    ) == {"decode_loop": 6}


def test_fresh_rid_keeps_the_live_counts():
    # Y was already routed to its ready decode node (first decode after
    # prefill): its io is the truth, no +1
    w = _worker({"Y": {"decode_loop": 0}}, _ar_io(5))
    assert w._spec_loop_iters(
        _PENDING, _NEW_ITER, "Y", continuing=False,
    ) == {"decode_loop": 0}


def test_same_iter_speculation_does_not_advance():
    same = SpeculativeNodeInfo(
        node_name="decode", is_new_loop_iter=False, loop_name="decode_loop",
    )
    w = _worker({"X": {"decode_loop": 5}}, _ar_io(5))
    assert w._spec_loop_iters(
        _PENDING, same, "X", continuing=True,
    ) == {"decode_loop": 5}
