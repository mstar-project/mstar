"""Requests at different iterations of a ``Loop`` share its node's steps, up to the node's batch cap.

A diffusion model's denoise loop is one node looped per request. With a cap above one row the worker serves several
requests' loops in the same steps: a request that arrives while others are mid-loop joins the next step at iteration
0, each row leaves at its own cap, rows past the cap wait, and a cancelled row is cut from the next step built.

A real worker (``make_worker``) with the real scheduler and speculation over the real graph runtime is driven in the
run loop's order; the engine is a stand-in that admits every row.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources.step import FULL_ADMIT_OK
from mstar.graph.runtime.base import ColumnarEdgeSpecs
from mstar.model.submodule_base import BatchedModelOutput, StepOrder
from mstar.testing_utils.worker import deliver, make_worker
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import DrainRequest, WorkerMessage, WorkerMessageType
from mstar.worker.executing_batch import build_executing_batch
from mstar.worker.micro_scheduler import CHAIN_LANE
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.step_executor import PendingBatch
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import Worker

pytest.importorskip("mstar_rust", reason="mstar_rust not built")

import test_spec_rows as rows  # noqa: E402

NODE, PART, WALK, LOOP = rows.NODE, rows.PART, rows.WALK, rows.LOOP


def _worker(cap: int) -> Worker:
    runtime, tensors = rows._runtime(max_iters=64)
    tensors.has_inflight_reads = lambda rid: False  # a drain asks before it acknowledges
    state = RequestStateManager(node_to_partition={NODE: PART})
    engine = SimpleNamespace(
        get_max_batch_size=lambda node, walk: cap,
        batch_cap_varies=lambda node, walk: False,
        get_max_batch_tokens=lambda node, walk: None,
        get_max_rows_in_combined=lambda node, walk: None,
        get_step_order=lambda node, walk: StepOrder.ARRIVAL,
        splits_by_capture_key=lambda node, walk: False,
        check_ready=lambda node, rid, fwd_info, allow_reload=True: FULL_ADMIT_OK,
        order_split_rows=lambda node, walk, rids, walks: rids,
        release_cached_prefix=lambda batch: None,
    )
    return make_worker(
        graph_runtime=runtime, worker_id=rows.ME, tensor_manager=tensors, request_state=state,
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        streams=StreamBufferManager(state, tensors, []),
        communicator=SimpleNamespace(send=lambda *a, **k: None),
        parallel_leader_nodes={NODE},
    )


def _admit(w: Worker, wire: str, steps: int) -> int:
    """A request whose loop is capped at ``steps`` iterations, its first input at the node."""
    rid = w._graph_runtime.add_request(
        request_id=wire, partition=PART, graph_walk=WALK, partition_worker_graph_ids=[rows.WG],
        worker_graph_to_workers=ParallelList([rows.WG], [[rows.ME]]), loop_caps={LOOP: steps})
    w.request_state.add_request(rid, CurrentForwardPassInfo(
        request_id=wire, graph_walk=WALK, fwd_index=0, random_seed=0, max_tokens=1 << 30, rid_handle=rid,
        partition_name=PART,
    ))
    w.lifecycle.add(rid, wire)
    block = ColumnarEdgeSpecs.empty()
    block.add(rid, rows.SIG, [], False, next_node=NODE)
    w._graph_runtime.ingest_inputs_batch(block)
    return rid


def _hold(w: Worker, pending: PendingBatch) -> None:
    """What launching a step records: its rows in flight on the chain, and their turn."""
    w.pipeline._hold_rows(CHAIN_LANE, pending.batch.request_to_worker_graph)
    w.scheduler.note_ran(pending.batch)


def _first(w: Worker) -> PendingBatch:
    batch = w.scheduler.next_batch()
    w.executor._set_in_flight(batch, True)
    pending = PendingBatch(
        batch=batch, node_batch=build_executing_batch(w.speculator._services, batch), node_name=NODE,
        partition=PART, graph_walk=WALK, future=None,
    )
    _hold(w, pending)
    return pending


def _next(w: Worker, pending: PendingBatch) -> PendingBatch | None:
    """One pass of the run loop while ``pending`` is on the GPU: build the next step, settle it against
    ``pending``'s outputs, launch it, then postprocess ``pending``. None when nothing follows."""
    spec, _ = w.speculator.build(pending)
    w.executor.clear_in_flight(pending.batch)
    if spec is None:
        rows._route(w, pending)
        w.pipeline._apply_drains(set())
        return None
    if not spec.is_yield_away:
        w.speculator.thread_outputs(spec, BatchedModelOutput(
            rows_only=True, row_request_ids=tuple(pending.node_batch.request_ids),
        ))
    w.speculator.settle(spec, success=True, dropped_rids=spec.dropped)
    w.speculator.note_launched(spec)
    rows._route(w, pending)
    launched = PendingBatch(
        batch=spec.scheduled_batch, node_batch=spec.node_batch, node_name=NODE, partition=PART, graph_walk=WALK,
        future=None, speculative_new_iter=spec.is_new_iter, loop_name=spec.loop_name,
    )
    w.pipeline._apply_drains(set(launched.batch.request_to_worker_graph))
    _hold(w, launched)
    return launched


def _step_rows(pending: PendingBatch) -> list[tuple[int, int]]:
    """``(request, loop iteration)`` per row, as the node's ``prepare_inputs`` reads them."""
    batch = pending.node_batch
    infos = batch.per_request_info_wrapped
    return [(rid, infos[rid].dynamic_loop_iter_counts.get(LOOP, 0)) for rid in batch.request_ids]


def _serve(w: Worker, pending: PendingBatch, arrivals=None, cancels=None, limit: int = 32):
    """Every step's rows from ``pending`` until nothing follows. ``arrivals`` / ``cancels``: step index -> what
    happens while that step is on the GPU."""
    steps = [_step_rows(pending)]
    while pending is not None and len(steps) < limit:
        at = len(steps) - 1
        for arrive in (arrivals or {}).get(at, ()):
            arrive()
        for wire in (cancels or {}).get(at, ()):
            deliver(w, WorkerMessage(
                message_type=WorkerMessageType.DRAIN_REQUEST, body=DrainRequest(request_id=wire),
            ))
        pending = _next(w, pending)
        if pending is not None:
            steps.append(_step_rows(pending))
    return steps


def test_a_request_joins_a_running_loop_at_its_own_iteration_and_each_leaves_at_its_cap():
    w = _worker(cap=4)
    a = _admit(w, "a", steps=4)
    late = {}
    steps = _serve(w, _first(w), arrivals={
        1: [lambda: late.update(b=_admit(w, "b", steps=3))],
        2: [lambda: late.update(c=_admit(w, "c", steps=2))],
    })
    b, c = late["b"], late["c"]
    assert steps == [
        [(a, 0)],
        [(a, 1)],
        [(a, 2), (b, 0)],
        [(a, 3), (b, 1), (c, 0)],
        [(b, 2), (c, 1)],
    ]


def test_rows_past_the_cap_wait_for_a_row_to_leave():
    w = _worker(cap=2)
    a, b, c = (_admit(w, wire, steps=2) for wire in "abc")
    steps = _serve(w, _first(w))
    assert [sorted(step) for step in steps[:2]] == [[(a, 0), (b, 0)], [(a, 1), (b, 1)]]
    assert steps[2:] == [[(c, 0)], [(c, 1)]]


def test_a_cancelled_row_is_cut_from_the_next_step_built_and_the_rest_go_on():
    w = _worker(cap=4)
    a, b = _admit(w, "a", steps=5), _admit(w, "b", steps=5)
    steps = _serve(w, _first(w), cancels={1: ["b"]})
    # the cancel lands while step 1 runs: step 2 is built without b, and a finishes its loop
    assert [sorted(step) for step in steps[:2]] == [[(a, 0), (b, 0)], [(a, 1), (b, 1)]]
    assert steps[2:] == [[(a, 2)], [(a, 3)], [(a, 4)]]
    assert b in w.lifecycle.failed_rids
