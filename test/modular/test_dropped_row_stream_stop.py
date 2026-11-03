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

from mstar.communication.tensors import StoredOutputs  # noqa: E402
from mstar.engine.chunking import ChunkingManagement  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.graph.runtime.base import PhasedPostprocess, RouteOutput  # noqa: E402
from mstar.model.submodule_base import END_PARTITION, BatchedModelOutput, chunk_loop_name  # noqa: E402
from mstar.worker.chunk_loops import ChunkLoops  # noqa: E402
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


class _Runtime(PhasedPostprocess):
    """Answers ``stop_loops_batched`` like the real runtime for a running
    dynamic loop: it does not know a rid failed until the REMOVE arrives."""

    supports_inline_emit = False

    def __init__(self, pending_stops=()):
        self.stopped: list[int] = []
        self.loop_names: dict[int, set[str]] = {}
        self.partition_done: list[int] = []
        self.pending_stops = set(pending_stops)

    def pending_loop_stop_rids(self, graph_walk, loop_name):
        return set(self.pending_stops)

    def cleanup_consumed_inputs(self, node, rids, wgs):
        return [], []

    def clear_pending_loop_stops(self):
        pass

    def mark_partition_done(self, rid, partition):
        self.partition_done.append(rid)

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
        self.loop_names.update((rid, set(names)) for rid, names in loop_names)
        return rids


def _engine(sub):
    eng = Engine.__new__(Engine)
    eng._submodules = {NODE: SimpleNamespace(submodule=sub, resources={})}
    eng._enable_nvtx = False
    eng._loopback_steps = {}
    eng._chunking = ChunkingManagement(eng._submodules)
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


def _worker(eng, runtime, in_loop=True, through_send=False, loop=LOOP):
    def _store(rids, outputs, signals, **k):
        if not through_send:
            raise _ReachedRoutingError
        return StoredOutputs([], [], [], [0] * (len(rids) * len(signals)))

    w = Worker.__new__(Worker)
    w._phase_period = 0
    w._stop_token_signal = {}  # no declared stop policies: check_stop decides
    w._walk_drivers = {}  # no producer-driven walk transitions here
    w._walk_driven_edges = {}
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
    w._innermost_loop_by_node = {(WALK, NODE): loop} if in_loop else {}
    w.reported: dict[int, str] = {}
    w._fail_requests = w.reported.update
    w.request_state = SimpleNamespace(
        buffer_publish_info=lambda *a: None, flush_publish_info=lambda *a: None,
        get_fwd_info=lambda rid, partition: None,
        get_pending_publish_info=lambda rid, partition: {},
    )
    return w


def _postprocess(
    sub, rids=(1, 2), pending_stops=(), cached=(), chunk_loop=False,
    streams=True, in_loop=True, through_send=False, loop=LOOP,
):
    """What ``Worker.run`` does with a finished step: drop and report the rows
    a per-rid stage failed, then postprocess the rest. ``pending_stops`` makes
    the step a speculative new loop iteration whose rows in it already
    stopped a step earlier."""
    eng, runtime = _engine(sub), _Runtime(pending_stops)
    batch = _batch(rids, streams)
    eng.prepare_inputs(batch)
    # rows the prefix cache served whole: declined by prepare, kept by postprocess
    batch.cached_rids = set(cached)
    outputs = BatchedModelOutput(per_rid_outputs={
        r: {"audio": [torch.zeros(1)]} for r in batch.request_ids
    })
    w = _worker(eng, runtime, in_loop, through_send, loop)
    if chunk_loop:
        w.chunk_loops = ChunkLoops({(NODE, WALK)})
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


def test_a_cache_served_row_ends_its_chunk_loop_without_publishing():
    """A chunkable walk the prefix cache served whole never runs, but its chunk
    Loop still has to end, which its route stride does. Finalizing a stop for
    it once read the row's forward-pass info, which a row that did not run no
    longer has: ``KeyError``, and every request in the step failed with it."""
    from mstar.model.submodule_base import CHUNK_LOOP_END

    batch, runtime, w = _postprocess(_Sub(skip={2}), cached={2}, chunk_loop=True)
    served = ScheduledBatch(
        node_name=NODE, graph_walk=WALK, request_to_worker_graph={2: 0},
        chunk_loop_walks=w.chunk_loops.walks_by_node[NODE],
    )
    assert served.loop_strides([2]) == [CHUNK_LOOP_END], "the served row's loop still ends"
    assert runtime.stopped == [1], "by its stride, not a stop"
    assert not w.reported
    assert 2 not in batch.resource_publish_info


def test_an_end_partition_stop_reports_the_partition_done():
    """END_PARTITION from a row outside any loop is never a loop name for the runtime."""
    batch, runtime, w = _postprocess(_Sub(stops={1: {END_PARTITION}}), streams=False, in_loop=False)
    assert batch.partition_done_rids == {1}
    assert runtime.loop_names == {}
    assert not w.reported


def test_an_end_partition_stop_inside_a_loop_fails_the_request():
    batch, runtime, w = _postprocess(_Sub(stops={1: {END_PARTITION}}), streams=False)
    assert set(w.reported) == {1}
    assert batch.request_ids == [2] and batch.partition_done_rids == set()
    assert runtime.stopped == []


def test_a_failed_check_stop_row_does_not_report_its_partition_done():
    """The send reports the step's partitions done; not a row the stop check
    failed, which the STOPS break drops after the request was built."""
    _, runtime, w = _postprocess(_Sub(bad={2}, stage="check_stop"), through_send=True)
    assert set(w.reported) == {2}
    assert runtime.partition_done == [1]


def test_an_end_partition_stop_on_a_rows_last_chunk_ends_its_partition():
    """A row whose innermost loop is its own chunk Loop is on its last chunk
    when it decides a stop, and that Loop ends by its own count: an
    END_PARTITION there ends the partition rather than failing the request."""
    batch, runtime, w = _postprocess(
        _Sub(stops={1: {END_PARTITION}}), streams=False, loop=chunk_loop_name(NODE, WALK),
    )
    assert not w.reported
    assert batch.partition_done_rids == {1}
    assert runtime.loop_names == {}
