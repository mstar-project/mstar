"""Worker- and preprocess-worker-side teardown drain.

DRAIN_REQUEST stops scheduling/reading a request and ACKs READS_DONE once its
in-flight reads finish; the hard cleanup (force_cleanup_request) waits for the
conductor's REMOVE_REQUEST. No read may start for a draining request.
"""

from types import SimpleNamespace

from mstar.utils.ipc_format import (
    ConductorMessageType,
    DrainRequest,
    InputSignals,
    MessageSource,
    RemoveRequest,
)
from mstar.worker.worker import Worker

# ── worker ──────────────────────────────────────────────────────────────────

def _worker(
    known_rids=("X",), in_flight=(), draining=(), reads_done=(),
    pending_drains=(), inflight_reads=False, is_follower=False, tp_follow=(),
):
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w.is_tp_follower = is_follower
    w.sent = []
    w.cleared = []
    w.failed = set()
    w.forced = []
    w.communicator = SimpleNamespace(send=lambda e, m: w.sent.append((e, m)))
    w._in_flight_rids = set(in_flight)
    w._pending_drains = set(pending_drains)
    w._draining_rids = set(draining)
    w._reads_done_sent = set(reads_done)
    w._pending_removes = set()
    w._last_active = {}
    w.streaming_buffers = {}
    w.scheduler = SimpleNamespace(
        clear_rid=lambda rid: w.cleared.append(rid),  # noqa: PLW0108
        fail_rids=lambda rids: w.failed.update(rids),  # noqa: PLW0108
        pending_tp_follow_count=dict.fromkeys(tp_follow, 1),
    )
    w.worker_graphs_manager = SimpleNamespace(
        per_request_info={
            rid: SimpleNamespace(sharding_config=SimpleNamespace(groups=[]))
            for rid in known_rids
        },
        remove_request=lambda rid: None,
    )
    w.engine_manager = SimpleNamespace(
        remove_request=lambda rid: None, lru_tracked_nodes=lambda: [],
    )
    w.profile_info = SimpleNamespace(pop_request=lambda rid: None)
    w.tensor_manager = SimpleNamespace(
        has_inflight_reads=lambda rid: inflight_reads,
        force_cleanup_request=lambda rid: w.forced.append(rid), # noqa: PLW0108
    )
    return w


def _reads_done(w):
    return [
        m for e, m in w.sent
        if e == "conductor" and m.message_type == ConductorMessageType.READS_DONE
    ]


def test_drain_acks_reads_done_when_no_inflight_reads():
    w = _worker(inflight_reads=False)
    Worker._drain_request(w, DrainRequest(request_id="X"))

    assert "X" in w._draining_rids  # reads gated until REMOVE
    assert "X" in w._reads_done_sent
    assert "X" in w.failed  # scheduler stopped from starting new leader work
    assert "X" not in w.cleared  # but state kept until REMOVE
    acks = _reads_done(w)
    assert len(acks) == 1
    assert acks[0].body.request_id == "X"
    assert acks[0].body.entity_id == "w0"


def test_drain_waits_for_inflight_reads_then_acks():
    w = _worker(inflight_reads=True)
    Worker._drain_request(w, DrainRequest(request_id="X"))
    assert "X" in w._draining_rids and not _reads_done(w)

    # Async reads finish; the run-loop hook re-checks and now ACKs.
    w.tensor_manager.has_inflight_reads = lambda rid: False
    Worker._apply_pending_drains(w, set())
    assert len(_reads_done(w)) == 1


def test_drain_waits_for_committed_tp_follow_batches():
    """A queued TP-follow batch (leader already sent the ZMQ) must drain before
    READS_DONE, or REMOVE tears the worker-graph queues out from under it."""
    w = _worker(tp_follow=("X",))
    Worker._drain_request(w, DrainRequest(request_id="X"))
    assert "X" in w._draining_rids and not _reads_done(w)

    # The follow batch gets scheduled; count drops to 0.
    w.scheduler.pending_tp_follow_count.pop("X")
    Worker._apply_pending_drains(w, set())
    assert len(_reads_done(w)) == 1


def test_drain_deferred_behind_inflight_gpu_step():
    w = _worker(in_flight=("X",))
    Worker._drain_request(w, DrainRequest(request_id="X"))
    # Held until the GPU step referencing the rid is gone.
    assert "X" in w._pending_drains and "X" not in w._draining_rids
    assert not _reads_done(w)

    Worker._apply_pending_drains(w, set())  # step finished
    assert "X" in w._draining_rids and len(_reads_done(w)) == 1


def test_follower_ignores_conductor_drain():
    w = _worker(is_follower=True)
    Worker._drain_request(w, DrainRequest(request_id="X"))  # source=CONDUCTOR
    assert "X" not in w._draining_rids and not w.sent

    # ... but honors the leader's forwarded drain.
    Worker._drain_request(
        w, DrainRequest(request_id="X", source=MessageSource.TP_RANK_0)
    )
    assert "X" in w._draining_rids and len(_reads_done(w)) == 1


def test_drain_acks_only_once():
    w = _worker(inflight_reads=False)
    Worker._drain_request(w, DrainRequest(request_id="X"))
    Worker._drain_request(w, DrainRequest(request_id="X"))
    assert len(_reads_done(w)) == 1


def test_remove_force_cleans_and_clears_drain_state():
    w = _worker(draining=("X",), reads_done=("X",))
    Worker._remove_request(w, RemoveRequest(request_id="X"))
    assert w.forced == ["X"]
    assert "X" not in w._draining_rids
    assert "X" not in w._reads_done_sent


def test_process_new_inputs_skips_reads_for_draining_rid():
    w = _worker(draining=("X",))

    def _boom(*a, **k):
        raise AssertionError("must not start a read while draining")

    w.tensor_manager.start_read_tensors = _boom
    # Returns before touching the read path.
    Worker._process_new_inputs(
        w, InputSignals(request_id="X", inputs=[], request_info=None)
    )


def test_add_new_request_skips_draining_rid():
    w = _worker(draining=("X",))
    # Bails before touching engine/graph managers (out-of-order NEW after DRAIN).
    Worker._add_new_request(w, SimpleNamespace(request_id="X"))


# ── preprocess worker ───────────────────────────────────────────────────────

def _preprocess(inflight_reads=False):
    from mstar.api_server.data_worker import PreprocessWorkerThread

    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt._draining_rids = set()
    wt._reads_done_sent = set()
    wt.sent = []
    wt.forced = []
    wt.communicator = SimpleNamespace(send=lambda e, m: wt.sent.append((e, m)))
    wt.tensor_manager = SimpleNamespace(
        has_inflight_reads=lambda rid: inflight_reads,
        force_cleanup_request=lambda rid: wt.forced.append(rid), # noqa: PLW0108
    )
    wt.tensor_uuid_to_metadata_per_request = {}
    wt.request_model_kwargs = {}
    return wt


def _pw_reads_done(wt):
    return [
        m for e, m in wt.sent
        if e == "conductor" and m.message_type == ConductorMessageType.READS_DONE
    ]


def test_preprocess_drain_acks_when_reads_done():
    wt = _preprocess(inflight_reads=False)
    wt._begin_drain("X")
    assert "X" in wt._reads_done_sent
    acks = _pw_reads_done(wt)
    assert len(acks) == 1
    assert acks[0].body.entity_id == "api_server_preprocess_worker"


def test_preprocess_drain_waits_for_reads():
    wt = _preprocess(inflight_reads=True)
    wt._begin_drain("X")
    assert "X" in wt._draining_rids and not _pw_reads_done(wt)

    wt.tensor_manager.has_inflight_reads = lambda rid: False
    wt._complete_drain_if_ready("X")
    assert len(_pw_reads_done(wt)) == 1


def test_preprocess_reads_done_idempotent():
    wt = _preprocess()
    wt._send_reads_done("X")
    wt._send_reads_done("X")
    assert len(_pw_reads_done(wt)) == 1


def test_preprocess_hard_cleanup_force_drops_and_clears():
    wt = _preprocess()
    wt._draining_rids.add("X")
    wt._reads_done_sent.add("X")
    wt.tensor_uuid_to_metadata_per_request["X"] = {"u": {}}
    wt.request_model_kwargs["X"] = {}

    wt._hard_cleanup("X")
    assert wt.forced == ["X"]
    assert "X" not in wt._draining_rids
    assert "X" not in wt._reads_done_sent
    assert "X" not in wt.tensor_uuid_to_metadata_per_request
    assert "X" not in wt.request_model_kwargs
