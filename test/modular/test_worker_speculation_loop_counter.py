"""A speculative batch runs under loop counters of its own.

A request's shared ``CurrentForwardPassInfo`` is refreshed as each batch is
built, so while iteration k of a loop is in flight it reads k. A speculative
next iteration built over it would run k again: for a denoise step the same
sigma twice, and no veto of the overshoot past the last step. So
``SpeculationBuilder.assemble`` gives the speculative batch per-request
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
from mstar.worker.micro_scheduler import MicroScheduler  # noqa: E402
from mstar.worker.speculation import SpeculationBuilder, SpeculationContext  # noqa: E402
from mstar.worker.stream_buffers import StreamBufferManager  # noqa: E402

LOOP = "denoise"
PARTITION = "p"


def _info(rid: int, k: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk="t2i", fwd_index=0, random_seed=0, max_tokens=0,
        step_metadata={"num_inference_steps": 4}, dynamic_loop_iter_counts={LOOP: k},
    )


def _scheduler():
    """The real ``label_for`` over an empty combined-walk map: these batches
    are uniform, so it labels each with its own walk and carries no
    per-row walks."""
    scheduler = SimpleNamespace(combined_walk_of={})
    for name in ("label_for", "_key"):
        setattr(scheduler, name, getattr(MicroScheduler, name).__get__(scheduler))
    return scheduler


def _context(shared: dict[int, CurrentForwardPassInfo]) -> SpeculationContext:
    """The request state answers with the shared info (which the worker
    refreshes to the loop's index when a step's outputs are routed); the
    assembly path is the real one."""
    request_state = SimpleNamespace(
        get_fwd_info=lambda rid, partition: shared[rid], per_request_info={},
    )
    return SpeculationContext(
        graph_runtime=SimpleNamespace(get_consumed_edges=lambda *a: []),
        tensor_manager=None, request_state=request_state, scheduler=_scheduler(),
        streams=StreamBufferManager(request_state, None, []),
    )


def _assemble(ctx, spec_target, rids, continuing):
    return SpeculationBuilder(ctx).assemble(
        SimpleNamespace(
            node_name="dit", graph_walk="t2i", partition=PARTITION,
        ),
        spec_target,
        {rid: 0 for rid in rids},          # request_to_worker_graph
        {rid: {} for rid in rids},         # per_request_inputs
        {},                                # consumed_streaming_edges
        continuing,
        {},                                # fresh_final_edges
        is_same_node=spec_target.node_name == "dit",
    )


def _target(**kwargs) -> SpeculationOutput:
    return SpeculationOutput(node_name="dit", graph_walk="t2i", loop_name=LOOP, **kwargs)


def test_a_continuing_rid_runs_one_iteration_past_the_in_flight_one():
    shared = {1: _info(1, 2)}
    ctx = _context(shared)  # iteration 2 in flight: the shared counter reads 2
    in_flight = _assemble(ctx, _target(is_new_loop_iter=False), [1], [])
    spec = _assemble(ctx, _target(is_new_loop_iter=True), [1], [1])

    view = spec.node_batch.per_request_info_wrapped[1]
    assert view.dynamic_loop_iter_counts == {LOOP: 3}
    assert spec.node_batch.per_request_input_metadata[1].dynamic_loop_iter_counts == {LOOP: 3}
    # the shared info is untouched: the in-flight batch's stop check reads 2
    assert shared[1].dynamic_loop_iter_counts == {LOOP: 2}
    # the speculative batch still carries the shared info itself, so what an
    # engine records on it is seen by later steps
    assert spec.node_batch.per_request_info[1] is shared[1]
    assert view is not shared[1]
    assert view.step_metadata is shared[1].step_metadata
    assert view.resource_publish_info is shared[1].resource_publish_info
    assert view.resource_configs is shared[1].resource_configs
    shared[1].dynamic_loop_iter_counts[LOOP] = 4
    assert in_flight.node_batch.per_request_info_wrapped[1].dynamic_loop_iter_counts == {LOOP: 2}
    assert view.dynamic_loop_iter_counts == {LOOP: 3}


def test_a_fresh_rid_runs_at_its_loops_current_index():
    # 2's last step was iteration 0; routing it advanced the loop to 1 and refreshed
    # the shared counter, so a fresh rid is not incremented: it is already there
    shared = {1: _info(1, 2), 2: _info(2, 1)}
    ctx = _context(shared)
    spec = _assemble(ctx, _target(is_new_loop_iter=True), [1, 2], [1])

    counts = {
        rid: info.dynamic_loop_iter_counts
        for rid, info in spec.node_batch.per_request_info_wrapped.items()
    }
    assert counts == {1: {LOOP: 3}, 2: {LOOP: 1}}


def test_a_transition_into_the_loop_starts_at_its_first_iteration():
    # speculating dit from the text encoder: not a new iteration of the loop
    shared = {1: _info(1, 0)}
    ctx = _context(shared)
    spec = _assemble(ctx, _target(is_new_loop_iter=False), [1], [1])

    assert spec.node_batch.per_request_info_wrapped[1].dynamic_loop_iter_counts == {LOOP: 0}
    assert spec.is_new_iter is False
