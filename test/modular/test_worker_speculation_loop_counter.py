"""A speculative batch runs under loop counters of its own.

A request's shared ``CurrentForwardPassInfo`` is refreshed as each batch is
built, so while iteration k of a loop is in flight it reads k. A speculative
next iteration built over it would run k again: for a denoise step the same
sigma twice, and no veto of the overshoot past the last step. So
``Worker._assemble_speculation`` gives the speculative batch per-request
``InputMetadata`` carrying the counters it will run at (the engine reads them
through ``ForwardPassInfoWrapper``), and leaves the shared info to the in-flight
batch, whose stop check still has to read k.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.graph.runtime.base import SpeculationOutput  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

LOOP = "denoise"
PARTITION = "p"


def _info(rid: int, k: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk="t2i", fwd_index=0, random_seed=0, max_tokens=0,
        step_metadata={"num_inference_steps": 4}, dynamic_loop_iter_counts={LOOP: k},
    )


def _worker(shared: dict[int, CurrentForwardPassInfo]):
    """A bare worker: the request state answers with the shared info (which the
    worker refreshes to the loop's index when a step's outputs are routed); the
    assembly path is the real one."""
    worker = SimpleNamespace(
        request_state=SimpleNamespace(get_fwd_info=lambda rid, partition: shared[rid]),
        _graph_runtime=SimpleNamespace(get_consumed_edges=lambda *a: []),
        _stream_chunks_for=lambda rid, node_name, inputs: None,
        _settle_final_streams=lambda node_name, final_edges: (set(), set()),
    )
    for name in ("_make_executing_batch", "_build_input_metadata_and_fwd_info", "_assemble_speculation"):
        setattr(worker, name, getattr(Worker, name).__get__(worker))
    return worker


def _assemble(worker, spec_target, rids, continuing):
    return worker._assemble_speculation(
        SimpleNamespace(
            node_name="dit", graph_walk="t2i", partition=PARTITION,
        ),
        spec_target,
        {rid: 0 for rid in rids},          # request_to_worker_graph
        {rid: {} for rid in rids},         # per_request_inputs
        {},                                # consumed_streaming_edges
        continuing,
        is_same_node=spec_target.node_name == "dit",
    )


def _target(**kwargs) -> SpeculationOutput:
    return SpeculationOutput(node_name="dit", graph_walk="t2i", loop_name=LOOP, **kwargs)


def test_a_continuing_rid_runs_one_iteration_past_the_in_flight_one():
    shared = {1: _info(1, 2)}
    worker = _worker(shared)  # iteration 2 in flight: the shared counter reads 2
    spec = _assemble(worker, _target(is_new_loop_iter=True), [1], [1])

    counters = spec.node_batch.per_request_input_metadata[1].dynamic_loop_iter_counts
    assert counters == {LOOP: 3}
    # the shared info is untouched: the in-flight batch's stop check reads 2
    assert shared[1].dynamic_loop_iter_counts == {LOOP: 2}
    # the speculative batch still carries the shared info itself, so what an
    # engine records on it is seen by later steps
    assert spec.node_batch.per_request_info[1] is shared[1]


def test_a_fresh_rid_runs_at_its_loops_current_index():
    # 2's last step was iteration 0; routing it advanced the loop to 1 and refreshed
    # the shared counter, so a fresh rid is not incremented: it is already there
    shared = {1: _info(1, 2), 2: _info(2, 1)}
    worker = _worker(shared)
    spec = _assemble(worker, _target(is_new_loop_iter=True), [1, 2], [1])

    counts = {
        rid: meta.dynamic_loop_iter_counts
        for rid, meta in spec.node_batch.per_request_input_metadata.items()
    }
    assert counts == {1: {LOOP: 3}, 2: {LOOP: 1}}


def test_a_transition_into_the_loop_starts_at_its_first_iteration():
    # speculating dit from the text encoder: not a new iteration of the loop
    shared = {1: _info(1, 0)}
    worker = _worker(shared)
    spec = _assemble(worker, _target(is_new_loop_iter=False), [1], [1])

    assert spec.node_batch.per_request_input_metadata[1].dynamic_loop_iter_counts == {LOOP: 0}
    assert spec.is_new_iter is False
