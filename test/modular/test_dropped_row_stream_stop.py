"""A row that leaves a step (its prepare_inputs, postprocess or check_stop
raised, or prepare_inputs declined it) must not end its streams on that step.

``final_stream_rids`` drives the stream-terminated loop stop in
``Worker._postprocess_batch``: every rid in it gets its innermost loop stopped,
and ``Engine.finalize_stopped_requests`` then reads that rid's forward-pass
info. A dropped row is no longer in the batch, so stopping it raised
``KeyError`` there and the main loop failed every request in the batch.

Also: an ``END_PARTITION`` stop (a prefill that sampled a stop token) reports
the partition done on the step and never reaches the runtime as a loop name;
from a row inside a loop it fails the request.

Drives the real ``Engine.prepare_inputs``, ``Worker._drop_failed_rids`` and
``Worker._postprocess_batch``; the graph runtime and tensor store are stubs
that stop at output routing.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.model.submodule_base import END_PARTITION, BatchedModelOutput  # noqa: E402
from mstar.worker.micro_scheduler import ScheduledBatch  # noqa: E402
from mstar.worker.worker import PendingBatch, Worker  # noqa: E402

NODE, WALK, LOOP = "vocoder", "w", "stream_loop"


class _ReachedRoutingError(Exception):
    """Postprocess got past the stop stage into output routing."""


class _Sub:
    """A stream consumer; ``bad`` rows raise in the named stage, ``skip`` rows
    are declined by prepare_inputs."""

    def __init__(self, bad=(), stage="prepare", skip=(), stops=None):
        self.bad, self.stage, self.skip = set(bad), stage, set(skip)
        self.stops = stops or {}

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
        return set(self.stops.get(rid, ()))


class _Runtime:
    """Answers ``stop_loops_batched`` like the real runtime for a running
    dynamic loop: it does not know a rid failed until the REMOVE arrives."""

    def __init__(self, pending_stops=()):
        self.stopped: list[int] = []
        self.loop_names: dict[int, set[str]] = {}
        self.pending_stops = set(pending_stops)

    def pending_loop_stop_rids(self, graph_walk, loop_name):
        return set(self.pending_stops)

    def cleanup_consumed_inputs(self, node, rids, wgs):
        return [], []

    def clear_pending_loop_stops(self):
        pass

    def stop_loops_batched(self, partition, graph_walk, last_node_run, loop_names):
        rids = [rid for rid, _ in loop_names]
        self.stopped.extend(rids)
        self.loop_names.update((rid, set(names)) for rid, names in loop_names)
        return rids


def _engine(sub):
    eng = Engine.__new__(Engine)
    eng._submodules = {NODE: SimpleNamespace(submodule=sub, resources={})}
    eng._enable_nvtx = False
    eng._loopback_steps = {}
    eng._runner = StepRunner({})
    return eng


def _batch(rids, streams=True):
    info = {
        r: SimpleNamespace(rid=r, update_publish_info=lambda p: None) for r in rids
    }
    return ExecutingBatch(
        node_name=NODE, per_request_info=info,
        step_context=StepContext(
            request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False,
        ),
        # every row takes its stream's final chunk this step
        final_stream_rids=set(rids) if streams else set(),
        partition_done_rids=set(rids) if streams else set(),
    )


def _worker(eng, runtime, in_loop=True):
    def _store(*a, **k):
        raise _ReachedRoutingError

    w = Worker.__new__(Worker)
    w._phase_period = 0
    w.enable_nvtx = False
    w.enable_prof = False
    w.device = torch.device("cpu")
    w._last_active = {}
    w._graph_runtime = runtime
    w.scheduler = SimpleNamespace(failed_rids=set())
    w.tensor_manager = SimpleNamespace(
        cleanup_collectable=lambda *a: None,
        store_and_return_tensor_info_batch=_store,
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda n: eng)
    w._innermost_loop_by_node = {(WALK, NODE): LOOP} if in_loop else {}
    w.reported: dict[int, str] = {}
    w._fail_requests = w.reported.update
    w.request_state = SimpleNamespace(buffer_publish_info=lambda *a: None)
    return w


def _postprocess(sub, rids=(1, 2), pending_stops=(), streams=True, in_loop=True):
    """What ``Worker.run`` does with a finished step: drop and report the rows
    a per-rid stage failed, then postprocess the rest. ``pending_stops`` makes
    the step a speculative new loop iteration whose rows in it already
    stopped a step earlier."""
    eng, runtime = _engine(sub), _Runtime(pending_stops)
    batch = _batch(rids, streams)
    eng.prepare_inputs(batch)
    outputs = BatchedModelOutput(per_rid_outputs={
        r: {"audio": [torch.zeros(1)]} for r in batch.request_ids
    })
    w = _worker(eng, runtime, in_loop)
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
    with pytest.raises(_ReachedRoutingError):
        w._postprocess_batch(pending, outputs)
    return batch, runtime, w


def test_a_row_dropped_by_prepare_inputs_does_not_end_its_streams():
    batch = _batch([1, 2])
    _engine(_Sub(bad={2})).prepare_inputs(batch)
    assert batch.request_ids == [1]
    assert batch.final_stream_rids == {1}
    assert batch.partition_done_rids == {1}


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
    assert batch.partition_done_rids == {1}


def test_a_row_that_overstayed_its_loop_stop_is_not_stopped_again():
    batch, runtime, w = _postprocess(_Sub(), pending_stops={2})
    assert runtime.stopped == [1]
    assert not w.reported
    assert batch.final_stream_rids == {1}
    assert batch.partition_done_rids == {1}


def test_an_end_partition_stop_reports_the_partition_done():
    """END_PARTITION from a row outside any loop is never a loop name for the runtime."""
    batch, runtime, w = _postprocess(_Sub(stops={1: {END_PARTITION}}), streams=False, in_loop=False)
    assert batch.partition_done_rids == {1}
    assert runtime.loop_names == {}
    assert not w.reported


def test_an_end_partition_stop_inside_a_loop_fails_the_request(caplog):
    """A model bug, not a bad request: the client gets an internal error and
    the log names the node."""
    with caplog.at_level("ERROR", logger="mstar.engine.engine"):
        batch, runtime, w = _postprocess(_Sub(stops={1: {END_PARTITION}}), streams=False)
    assert set(w.reported) == {1}
    assert w.reported[1].startswith("internal error")
    assert "END_PARTITION" in caplog.text and repr(NODE) in caplog.text
    assert batch.request_ids == [2] and batch.partition_done_rids == set()
    assert runtime.stopped == []
