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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.graph.loop_indices import LoopLayout
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
        graph_runtime=SimpleNamespace(
            get_consumed_edges=lambda *a: [],
            loop_layout=lambda walk, name: LoopLayout(name),
        ),
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


@pytest.mark.parametrize("columnar", [False, True])
def test_an_outer_restart_resets_inner_inputs_in_both_speculation_paths(columnar):
    from mstar.worker.spec_rows import loop_counter_snapshots

    shared = {1: _info(1, 3), 2: _info(2, 0)}
    shared[1].dynamic_loop_iter_counts["outer"] = 0
    shared[2].dynamic_loop_iter_counts["outer"] = 1
    layout = LoopLayout("outer", descendants=(LOOP,))
    ctx = _context(shared)
    ctx.graph_runtime.loop_layout = lambda walk, name: layout
    ctx.graph_runtime.graph_passes = lambda selected: [7] * len(selected)
    ctx.request_state.fwd_infos = lambda rids, partition: [shared[rid] for rid in rids]
    ctx = replace(ctx, engine_manager=SimpleNamespace(get_engine=lambda node: SimpleNamespace(
        iteration_layouts=lambda node: {"t2i": LoopLayout(LOOP, ancestors=("outer",))},
    )))
    target = SpeculationOutput(
        node_name="dit", graph_walk="t2i", loop_name="outer", is_new_loop_iter=True,
    )
    spec = SpeculationBuilder(ctx).assemble_rows(
        SimpleNamespace(node_name="dit", graph_walk="t2i", partition=PARTITION), target, [1], 0,
        fresh_request_to_worker_graph={2: 0}, fresh_inputs={2: {}},
    ) if columnar else _assemble(ctx, target, [1, 2], [1])
    snapshots = loop_counter_snapshots(list(shared.values()), layout, 1)
    for rid, columns in zip(shared, snapshots, strict=True):
        expected = {"outer": 1, LOOP: 0}
        assert spec.node_batch.per_request_info_wrapped[rid].dynamic_loop_iter_counts == expected
        assert columns.dynamic_loop_iter_counts == expected
        assert spec.node_batch.per_request_info_wrapped[rid].metadata.graph_pass == 7
        assert spec.node_batch.per_request_input_metadata[rid].graph_pass == 7
        assert spec.node_batch.per_request_info_wrapped[rid].metadata.iteration_origin == 0
    assert shared[1].dynamic_loop_iter_counts == {"outer": 0, LOOP: 3}


def test_actual_graph_passes_are_snapshotted_only_for_iteration_input_rows():
    from mstar.engine.engine import Engine
    from mstar.model.submodule_base import InputMetadata
    from mstar.worker.executing_batch import stamp_iteration_context

    engine = Engine.__new__(Engine)
    engine._loops = {("dit", "t2i"): LoopLayout(LOOP)}
    engine._submodules = {"dit": SimpleNamespace(
        submodule=SimpleNamespace(ITERATION_INPUTS={"sigma": object()}), device_stops={"t2i": object()},
    )}
    calls = []
    def passes(selected):
        calls.append(selected)
        return [7]
    runtime = SimpleNamespace(graph_passes=passes)
    infos = {rid: _info(rid, 2) for rid in (1, 2)}
    metadata = {rid: InputMetadata(dynamic_loop_iter_counts={LOOP: 6}) for rid in (1, 2)}
    stamp_iteration_context(runtime, engine, "dit", "mixed", infos, metadata, {1: 10, 2: 20}, {1: "t2i", 2: "prefill"})
    assert calls == [{1: 10}]
    assert metadata[1].graph_pass == 7 and metadata[2].graph_pass is None
    assert metadata[1].dynamic_loop_iter_counts == {LOOP: 6}
    assert metadata[1].iteration_origin == 2 and metadata[2].iteration_origin is None
    infos[1].dynamic_loop_iter_counts[LOOP] = 3
    assert metadata[1].iteration_origin == 2, "landing the predecessor cannot mutate its submitted successor"
    engine._submodules["dit"].device_stops.clear()
    stamp_iteration_context(runtime, engine, "dit", "t2i", infos, metadata, {1: 10, 2: 20})
    assert calls == [{1: 10}], "repeats off must not query graph state"
