"""A speculative batch runs under loop counters of its own.

A request's shared ``CurrentForwardPassInfo`` is refreshed as each batch is
built, so while iteration k of a loop is in flight it reads k. A speculative
next iteration built over it would run k again: for a denoise step the same
sigma twice, and no veto of the overshoot past the last step. So
``Worker._assemble_speculation`` hands the speculative batch a view with the
counters it will run at, and leaves the shared info to the in-flight batch,
whose stop check still has to read k.

``Worker._build_executing_batch`` refreshes the shared counters itself, so a
batch built off the ready queue on any path (the yield-away one included) runs
at the loop's current index.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.graph.base import GraphEdge, GraphNode, SpeculativeNodeInfo  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

LOOP = "denoise"
PARTITION = "p"


def _info(rid: str, k: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk="t2i", fwd_index=0, random_seed=0, max_tokens=0,
        step_metadata={"num_inference_steps": 4}, dynamic_loop_iter_counts={LOOP: k},
    )


def _dit(name: str = "dit") -> GraphNode:
    return GraphNode(
        name=name, input_names={"text_embeds", "latents"},
        outputs=[GraphEdge(name="latents", next_node=name)],
    )


def _worker(shared: dict[str, CurrentForwardPassInfo], loop_idx: dict[str, int]) -> SimpleNamespace:
    """A bare worker: the graphs manager answers with the shared info and the
    loops' current indices; the two batch builders are the real ones."""
    worker = SimpleNamespace(
        worker_graphs_manager=SimpleNamespace(
            get_fwd_info=lambda rid, partition: shared[rid],
            get_dynamic_loop_iters=lambda rid, partition: {LOOP: loop_idx[rid]},
            get_partition_for_node=lambda node_name: PARTITION,
        ),
        tensor_manager=SimpleNamespace(get_tensor=lambda request_id, uuid: None),
    )
    for name in ("_make_executing_batch", "_speculative_fwd_info", "_assemble_speculation",
                 "_build_executing_batch"):
        setattr(worker, name, getattr(Worker, name).__get__(worker))
    return worker


def _assemble(worker, spec_node_info, rids, continuing):
    node = _dit()
    return worker._assemble_speculation(
        SimpleNamespace(graph_walk="t2i", partition=PARTITION),
        node, spec_node_info,
        {rid: node for rid in rids}, {rid: "wg" for rid in rids}, {rid: {} for rid in rids},
        {}, continuing, is_same_node=spec_node_info.node_name == node.name,
    )


def test_a_continuing_rid_runs_one_iteration_past_the_in_flight_one():
    shared = {"a": _info("a", 2)}
    worker = _worker(shared, loop_idx={"a": 2})  # iteration 2 in flight, loop not advanced
    spec = _assemble(
        worker, SpeculativeNodeInfo("dit", is_new_loop_iter=True, loop_name=LOOP), ["a"], ["a"],
    )
    view = spec.node_batch.per_request_info["a"]
    assert view.dynamic_loop_iter_counts == {LOOP: 3}
    # the in-flight batch keeps the shared info: its stop check reads 2
    assert shared["a"].dynamic_loop_iter_counts == {LOOP: 2}
    assert view is not shared["a"]
    # what an engine records on the view is seen by later steps
    assert view.step_metadata is shared["a"].step_metadata
    assert view.resource_publish_info is shared["a"].resource_publish_info
    assert view.loop_stop_times is shared["a"].loop_stop_times


def test_a_fresh_rid_runs_at_its_loops_current_index():
    # b's last step was iteration 0; its loop advanced to 1 when that step routed,
    # but the shared counter still reads 0
    shared = {"a": _info("a", 2), "b": _info("b", 0)}
    worker = _worker(shared, loop_idx={"a": 2, "b": 1})
    spec = _assemble(
        worker, SpeculativeNodeInfo("dit", is_new_loop_iter=True, loop_name=LOOP),
        ["a", "b"], ["a"],
    )
    counts = {rid: info.dynamic_loop_iter_counts for rid, info in spec.node_batch.per_request_info.items()}
    assert counts == {"a": {LOOP: 3}, "b": {LOOP: 1}}


def test_a_transition_into_the_loop_starts_at_its_first_iteration():
    # speculating dit from the text encoder: not a new iteration of the loop
    shared = {"a": _info("a", 0)}
    worker = _worker(shared, loop_idx={"a": 0})
    spec = _assemble(
        worker, SpeculativeNodeInfo("dit", is_new_loop_iter=False, loop_name=LOOP), ["a"], ["a"],
    )
    assert spec.node_batch.per_request_info["a"].dynamic_loop_iter_counts == {LOOP: 0}
    assert spec.is_new_iter is False


def test_a_batch_built_off_the_ready_queue_reads_the_loops_current_index():
    shared = {"a": _info("a", 0)}
    worker = _worker(shared, loop_idx={"a": 1})
    batch = SimpleNamespace(node_name="dit", graph_walk="t2i", node_objects={"a": _dit()})
    node_batch = worker._build_executing_batch(batch)
    assert node_batch.per_request_info["a"] is shared["a"]
    assert shared["a"].dynamic_loop_iter_counts == {LOOP: 1}
