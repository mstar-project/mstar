"""A row that leaves a step (its prepare_inputs, postprocess or check_stop
raised, or prepare_inputs declined it) must not end its streams on that step.

``final_stream_rids`` drives the stream-terminated loop stop in
``Worker._postprocess_batch``: every rid in it gets its innermost loop stopped,
and ``Engine.finalize_stopped_requests`` then reads that rid's forward-pass
info. A dropped row is no longer in the batch, so stopping it raised
``KeyError`` there and the main loop failed every request in the batch.

Drives the real ``Engine.prepare_inputs``, ``Worker._drop_failed_rids`` and
``Worker._postprocess_batch``; the graph runtime and tensor store are stubs
that stop at output routing.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mstar.communication.tensors import StoredOutputs  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.graph.runtime.base import PhasedPostprocess, RouteOutput  # noqa: E402
from mstar.model.submodule_base import BatchedModelOutput  # noqa: E402
from mstar.worker.micro_scheduler import ScheduledBatch  # noqa: E402
from mstar.worker.worker import PendingBatch, Worker  # noqa: E402

NODE, WALK, LOOP = "vocoder", "w", "stream_loop"


class _ReachedRoutingError(Exception):
    """Postprocess got past the stop stage into output routing."""


class _Sub:
    """A stream consumer; ``bad`` rows raise in the named stage, ``skip`` rows
    are declined by prepare_inputs."""

    def __init__(self, bad=(), stage="prepare", skip=()):
        self.bad, self.stage, self.skip = set(bad), stage, set(skip)

    def uniform_row_inputs(self, graph_walk, has_loopback=False):
        return None

    def prepare_inputs(self, **kw):
        rid = kw["fwd_info"].info.rid
        if self.stage == "prepare" and rid in self.bad:
            raise RuntimeError("bad chunk")
        if rid in self.skip:
            return None
        return SimpleNamespace(input_seq_len=1)

    def can_batch(self, batch, model_inputs):
        return True

    def check_stop(self, rid, info, outputs):
        if self.stage == "check_stop" and rid in self.bad:
            raise RuntimeError("bad stop check")
        return set()


class _Runtime(PhasedPostprocess):
    """Answers ``stop_loops_batched`` like the real runtime for a running
    dynamic loop: it does not know a rid failed until the REMOVE arrives."""

    supports_inline_emit = False

    def __init__(self, pending_stops=()):
        self.stopped: list[int] = []
        self.stream_done: list[int] = []
        self.pending_stops = set(pending_stops)

    def pending_loop_stop_rids(self, graph_walk, loop_name):
        return set(self.pending_stops)

    def cleanup_consumed_inputs(self, node, rids, wgs):
        return [], []

    def clear_pending_loop_stops(self):
        pass

    def mark_stream_partition_done(self, rid, partition):
        self.stream_done.append(rid)

    def complete_and_route_batch(self, route_input):
        return RouteOutput(
            completion_id=1, register_uuids=[], register_rids=[],
            new_token_output_idxs=[], local_streaming_by_signal={}, rids_needing_request_info=set(),
        )

    def send_outputs(self, send_input):
        return []

    def update_dynamic_loop_iters(self, per_request_info, partition):
        pass

    def stop_loops_batched(self, partition, graph_walk, last_node_run, loop_names):
        rids = [rid for rid, _ in loop_names]
        self.stopped.extend(rids)
        return rids


def _engine(sub):
    eng = Engine.__new__(Engine)
    eng._submodules = {NODE: SimpleNamespace(submodule=sub, resources={})}
    eng._enable_nvtx = False
    eng._loopback_steps = {}
    eng._runner = StepRunner({})
    return eng


def _batch(rids):
    info = {
        r: SimpleNamespace(rid=r, update_publish_info=lambda p: None) for r in rids
    }
    return ExecutingBatch(
        node_name=NODE, per_request_info=info,
        step_context=StepContext(
            request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False,
        ),
        # every row takes its stream's final chunk this step
        final_stream_rids=set(rids),
        stream_partition_done_rids=set(rids),
    )


def _worker(eng, runtime, through_send=False):
    def _store(rids, outputs, signals, **k):
        if not through_send:
            raise _ReachedRoutingError
        return StoredOutputs([], [], [], [0] * (len(rids) * len(signals)))

    w = Worker.__new__(Worker)
    w._phase_period = 0
    w._stop_token_signal = {}  # no declared stop policies: check_stop decides
    w.enable_nvtx = False
    w.enable_prof = False
    w.device = torch.device("cpu")
    w._graph_runtime = runtime
    w.scheduler = SimpleNamespace(failed_rids=set())
    w.tensor_manager = SimpleNamespace(
        cleanup_collectable=lambda *a: None,
        store_and_return_tensor_info_batch=_store,
        drop_host_copies=lambda: None,
        dereference_batch_uniform=lambda uuids: None,
    )
    w.streams = SimpleNamespace(consumption_batch=lambda rids: [])
    w.engine_manager = SimpleNamespace(get_engine=lambda n: eng)
    w._innermost_loop_by_node = {(WALK, NODE): LOOP}
    w.reported: dict[int, str] = {}
    w._fail_requests = w.reported.update
    w.request_state = SimpleNamespace(
        buffer_publish_info=lambda *a: None, flush_publish_info=lambda *a: None,
        get_fwd_info=lambda rid, partition: None,
        get_pending_publish_info=lambda rid, partition: {},
    )
    return w


def _postprocess(sub, rids=(1, 2), pending_stops=(), through_send=False):
    """What ``Worker.run`` does with a finished step: drop and report the rows
    a per-rid stage failed, then postprocess the rest. ``pending_stops`` makes
    the step a speculative new loop iteration whose rows in it already
    stopped a step earlier."""
    eng, runtime = _engine(sub), _Runtime(pending_stops)
    batch = _batch(rids)
    eng.prepare_inputs(batch)
    outputs = BatchedModelOutput(per_rid_outputs={
        r: {"audio": [torch.zeros(1)]} for r in batch.request_ids
    })
    w = _worker(eng, runtime, through_send)
    pending = PendingBatch(
        batch=ScheduledBatch(
            node_name=NODE, graph_walk=WALK,
            request_to_worker_graph=dict.fromkeys(rids, 0),
        ),
        node_batch=batch, node_name=NODE, partition="p", graph_walk=WALK,
        future=Future(), speculative_new_iter=bool(pending_stops),
        loop_name=LOOP,
    )
    if batch.failed_requests:
        failed = dict(batch.failed_requests)
        w._drop_failed_rids(pending, outputs, failed)
        w._fail_requests(failed)
    if through_send:
        w._postprocess_batch(pending, outputs)
    else:
        with pytest.raises(_ReachedRoutingError):
            w._postprocess_batch(pending, outputs)
    return batch, runtime, w


def test_a_row_dropped_by_prepare_inputs_does_not_end_its_streams():
    batch = _batch([1, 2])
    _engine(_Sub(bad={2})).prepare_inputs(batch)
    assert batch.request_ids == [1]
    assert batch.final_stream_rids == {1}
    assert batch.stream_partition_done_rids == {1}


def test_a_failed_prepare_inputs_row_does_not_fail_the_batch():
    batch, runtime, w = _postprocess(_Sub(bad={2}, stage="prepare"))
    assert runtime.stopped == [1]
    assert set(w.reported) == {2}


def test_a_declined_row_is_not_stopped():
    batch, runtime, _ = _postprocess(_Sub(skip={2}))
    assert runtime.stopped == [1]
    assert 2 not in batch.final_stream_rids


def test_a_failed_check_stop_row_does_not_fail_the_batch():
    batch, runtime, w = _postprocess(_Sub(bad={2}, stage="check_stop"))
    assert runtime.stopped == [1]
    assert set(w.reported) == {2}
    assert batch.final_stream_rids == {1}
    assert batch.stream_partition_done_rids == {1}


def test_a_row_that_overstayed_its_loop_stop_is_not_stopped_again():
    batch, runtime, w = _postprocess(_Sub(), pending_stops={2})
    assert runtime.stopped == [1]
    assert not w.reported
    assert batch.final_stream_rids == {1}
    assert batch.stream_partition_done_rids == {1}


def test_a_failed_check_stop_row_does_not_report_its_stream_partition_done():
    """The send reports the step's stream partitions done; not a row the stop
    check failed, which the STOPS break drops after the request was built."""
    _, runtime, w = _postprocess(_Sub(bad={2}, stage="check_stop"), through_send=True)
    assert set(w.reported) == {2}
    assert runtime.stream_done == [1]
