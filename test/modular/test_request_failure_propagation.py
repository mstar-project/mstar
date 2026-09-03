"""Runtime inference errors must reach the client as a structured failure
instead of a blanket request timeout (issue #123).

Covers the whole chain: the engine attributing a per-rid stage failure, the
worker turning it into a FAIL_REQUESTS report (and never re-awaiting the
future that raised), the conductor tearing the request down and telling the
api server, and the api server releasing the waiting client with an error.
"""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from mstar.api_server.entrypoint import APIServer, PendingRequest
from mstar.api_server.request_types import (
    APIServerMessage,
    RequestFailed,
    ResultChunk,
    ResultTensors,
)
from mstar.conductor.conductor import Conductor
from mstar.engine.base import (
    BaseEngine,
    EngineType,
    NodeBatch,
    NodeOutput,
    PreparedBatch,
    StopCheckResult,
)
from mstar.profile.format import RequestProfile, RequestTiming
from mstar.utils.ipc_format import (
    ConductorMessageType,
    FailRequests,
    ReadsDone,
    WorkerMessageType,
)
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.worker import PendingBatch, Worker


def _node_batch(rids, node_name="node", walk="walk"):
    return NodeBatch(
        node_name=node_name,
        graph_walk=walk,
        request_ids=list(rids),
        per_request_input_tensors={rid: {} for rid in rids},
        per_request_info={rid: SimpleNamespace() for rid in rids},
    )


# ── engine: per-rid attribution ────────────────────────────────────────────


class _Engine(BaseEngine):
    """Minimal engine exercising the base template's error plumbing."""

    def __init__(self, failed):
        super().__init__()
        self._failed = failed

    def engine_type(self):
        return EngineType.STATELESS

    def load_model(self, *a, **k):
        return None

    def add_request(self, request_id, **kwargs):
        return None

    def remove_request(self, request_id):
        return None

    def prepare_batch(self, batch: NodeBatch) -> PreparedBatch:
        batch.request_ids = [r for r in batch.request_ids if r not in self._failed]
        return PreparedBatch(batch=batch, failed_requests=dict(self._failed))

    def execute_forward(self, planned) -> NodeOutput:
        return NodeOutput(
            per_request_output_tensors={
                rid: {} for rid in planned.batch.request_ids
            }
        )


def test_prepare_failure_only_kills_the_offending_rid():
    out = _Engine({"bad": "boom"}).execute_batch(_node_batch(["good", "bad"]))
    assert out.failed_requests == {"bad": "boom"}
    assert "good" in out.per_request_output_tensors


def test_prepare_failure_survives_the_empty_batch_early_return():
    """Every rid failing empties the batch; the template returns early there,
    and must still carry the errors out."""
    out = _Engine({"a": "boom", "b": "bang"}).execute_batch(_node_batch(["a", "b"]))
    assert out.failed_requests == {"a": "boom", "b": "bang"}


def test_minibatch_split_keeps_failures_from_every_chunk():
    engine = _Engine({"r0": "boom", "r3": "bang"})
    engine.get_max_batch_size = lambda node_name, graph_walk: 2
    out = engine.execute_with_max_batch_size(_node_batch(["r0", "r1", "r2", "r3"]))
    assert out.failed_requests == {"r0": "boom", "r3": "bang"}


def test_check_stop_failure_is_attributed_to_one_rid():
    """check_stop reads tensor values per rid, so a raise there names its
    culprit — the rest of the batch must still get its stop decisions."""
    from mstar.engine.stateless_engine import StatelessEngine

    class _Submodule:
        def check_stop(self, rid, req_info, outputs):
            if rid == "bad":
                raise RuntimeError("nan in logits")
            return {"decode"}

    engine = StatelessEngine.__new__(StatelessEngine)
    engine.submodules = {"node": _Submodule()}
    batch = _node_batch(["good", "bad"])
    output = NodeOutput(
        per_request_output_tensors={"good": {"o": []}, "bad": {"o": []}},
    )
    result = engine.check_stop_for_batch(batch, output)

    assert result.stops == {"good": {"decode"}}
    assert "nan in logits" in result.failed_requests["bad"]


def test_check_stop_default_reports_nothing():
    assert _Engine({}).check_stop_for_batch(_node_batch(["r1"]), NodeOutput(
        per_request_output_tensors={},
    )) == StopCheckResult()


def test_stateless_engine_attributes_prepare_inputs_raise():
    from mstar.engine.stateless_engine import StatelessEngine

    class _Submodule:
        def prepare_inputs(self, graph_walk, fwd_info, inputs):
            raise RuntimeError("bad shape")

    engine = StatelessEngine.__new__(StatelessEngine)
    engine.enable_nvtx = False
    engine.config = SimpleNamespace(name="test")
    batch = _node_batch(["r1"])
    node_inputs, skipped, failed = engine._prepare_inputs(batch, _Submodule())
    assert node_inputs == [] and skipped == set()
    assert "bad shape" in failed["r1"]


# ── worker ─────────────────────────────────────────────────────────────────


def _worker(known_rids=("r1", "r2")):
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w.sent = []
    w.communicator = SimpleNamespace(
        send=lambda entity_id, msg: w.sent.append((entity_id, msg))
    )
    w.worker_graphs_manager = SimpleNamespace(
        per_request_info={rid: object() for rid in known_rids}
    )
    w.scheduler = MicroScheduler.__new__(MicroScheduler)
    w.scheduler.failed_rids = set()
    w.scheduler.held_until = {}
    return w


def test_fail_requests_reports_per_rid_errors_once():
    w = _worker()
    w._fail_requests({"r1": "boom", "r2": "bang"})
    assert len(w.sent) == 1
    entity, msg = w.sent[0]
    assert entity == "conductor"
    assert msg.message_type == ConductorMessageType.FAIL_REQUESTS
    assert msg.body == FailRequests(errors={"r1": "boom", "r2": "bang"})
    assert w.scheduler.failed_rids == {"r1", "r2"}


def test_fail_requests_ignores_rids_the_worker_already_dropped():
    """The conductor answers a failure with REMOVE_REQUEST. It won't send one
    for a request it no longer tracks, so reporting an unknown rid would pin
    it in failed_rids forever."""
    w = _worker(known_rids=("r1",))
    w._fail_requests({"gone": "boom"})
    assert w.sent == []
    assert w.scheduler.failed_rids == set()


def test_drop_failed_rids_excises_them_from_the_finished_batch():
    """A rid that raised in postprocess is still in the batch; routing its
    (missing) outputs is how a failed request becomes a 200 with no body."""
    w = _worker()
    scheduled = ScheduledBatch(
        node_name="node", graph_walk="walk",
        node_objects={"r1": object(), "r2": object()},
        request_to_worker_graph={"r1": "wg", "r2": "wg"},
    )
    node_batch = _node_batch(["r1", "r2"])
    pending = PendingBatch(
        batch=scheduled, node_batch=node_batch, node_name="node",
        partition="default", graph_walk="walk", future=None,
    )
    output = NodeOutput(
        per_request_output_tensors={"r1": {"o": []}, "r2": {"o": []}},
        failed_requests={"r2": "boom"},
    )
    w._drop_failed_rids(pending, output)

    assert node_batch.request_ids == ["r1"]
    assert set(output.per_request_output_tensors) == {"r1"}
    assert set(scheduled.node_objects) == {"r1"}
    assert set(scheduled.request_to_worker_graph) == {"r1"}
    assert set(node_batch.per_request_info) == {"r1"}


def _pending_batch(rids, future=None):
    return PendingBatch(
        batch=ScheduledBatch(
            node_name="node", graph_walk="walk",
            node_objects={
                rid: SimpleNamespace(_speculatively_scheduled=True) for rid in rids
            },
            request_to_worker_graph={rid: "wg" for rid in rids},
        ),
        node_batch=_node_batch(rids), node_name="node",
        partition="default", graph_walk="walk", future=future,
    )


def test_crashed_forward_fails_the_whole_batch():
    """A raise the worker can't attribute to one rid fails every request the
    iteration touched, rather than dropping them into the request timeout."""
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        w = _worker()
        w._in_flight_rids = {"r1"}
        pending = _pending_batch(["r1"], future=executor.submit(lambda: 1 / 0))
        try:
            pending.future.result()
        except ZeroDivisionError as exc:
            w._handle_main_loop_error(exc, (pending, None), None)

        assert len(w.sent) == 1
        errors = w.sent[0][1].body.errors
        assert set(errors) == {"r1"}
        assert "ZeroDivisionError" in errors["r1"]
        # The node must not stay flagged as speculatively scheduled, or it can
        # never be re-queued.
        assert all(
            not n._speculatively_scheduled
            for n in pending.batch.node_objects.values()
        )
    finally:
        executor.shutdown(wait=True)


def test_error_handler_drains_the_speculative_batch_still_on_the_gpu():
    """The spec batch's future was submitted before the crash; abandoning it
    would leave the GPU thread writing into a batch nobody collects."""
    executor = ThreadPoolExecutor(max_workers=1)
    started = threading.Event()
    release = threading.Event()

    def _slow():
        started.set()
        release.wait(10)
        return "done"

    try:
        w = _worker()
        w._in_flight_rids = set()
        spec = _pending_batch(["r2"], future=executor.submit(_slow))
        started.wait(10)
        release.set()
        w._handle_main_loop_error(RuntimeError("boom"), (None, spec), None)

        assert spec.future.done()
        assert set(w.sent[0][1].body.errors) == {"r2"}
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_error_handler_also_fails_the_batch_built_this_iteration():
    w = _worker()
    w._in_flight_rids = {"r1"}
    scheduled = ScheduledBatch(
        node_name="node", graph_walk="walk",
        node_objects={"r2": SimpleNamespace(_speculatively_scheduled=True)},
        request_to_worker_graph={"r2": "wg"},
    )
    w._handle_main_loop_error(RuntimeError("boom"), (None, None), scheduled)
    assert set(w.sent[0][1].body.errors) == {"r1", "r2"}


# ── scheduler ──────────────────────────────────────────────────────────────


def _scheduler():
    s = MicroScheduler.__new__(MicroScheduler)
    s.failed_rids = set()
    s.held_until = {}
    s.pending_removes = set()
    s.tp_batches_pending_schedule = []
    s.pending_tp_follow_count = {}
    s.engine_manager = SimpleNamespace(
        get_engine=lambda name: SimpleNamespace(check_ready=lambda *a: True)
    )
    return s


def _graphs_manager(rids):
    return SimpleNamespace(
        per_request_info={rid: object() for rid in rids},
        queues={"wg": SimpleNamespace(
            get_ready_node_names=lambda: {rid: ["node"] for rid in rids}
        )},
        get_partition_for_node=lambda name: "default",
        get_graph_walk=lambda rid, part: "walk",
        get_fwd_info=lambda rid, part: None,
    )


def test_failed_rid_is_not_reported_as_ready_work():
    s = _scheduler()
    mgr = _graphs_manager(["r1"])
    assert s.has_ready_excluding(mgr, None) is True
    s.fail_rids({"r1"})
    assert s.has_ready_excluding(mgr, None) is False


def test_clear_rid_releases_a_failed_request():
    s = _scheduler()
    s.fail_rids({"r1"})
    s.held_until["r1"] = time.monotonic() + 100
    s.clear_rid("r1")
    assert s.failed_rids == set() and s.held_until == {}


# ── conductor ──────────────────────────────────────────────────────────────


def _conductor(rids):
    c = Conductor.__new__(Conductor)
    c.sent = []
    c.communicator = SimpleNamespace(
        send=lambda entity_id, msg: c.sent.append((entity_id, msg))
    )
    c.requests = {
        rid: SimpleNamespace(worker_graph_to_workers={"wg": ["w0"]}) for rid in rids
    }
    c.draining = {}
    c._early_reads_done = {}
    c.waiting_queue = []
    c._try_admit_waiting = lambda: None
    return c


def _drain_all(c, rid):
    """Deliver READS_DONE from every participant so the barrier finalizes."""
    for entity in ["w0", "api_server_preprocess_worker"]:
        c._handle_reads_done(ReadsDone(request_id=rid, entity_id=entity))


def test_conductor_tears_down_and_notifies_per_rid():
    c = _conductor(["r1", "r2"])
    c._fail_requests(FailRequests(errors={"r1": "boom", "r2": "bang"}))

    # Phase 1: drain the readers first — nothing torn down or notified yet.
    assert set(c.requests) == {"r1", "r2"}
    drains = {
        m.body.request_id for e, m in c.sent
        if m.message_type == WorkerMessageType.DRAIN_REQUEST
    }
    assert drains == {"r1", "r2"}
    assert not [m for e, m in c.sent if e == "api_server"]

    # Phase 2: once every reader has drained, hard-remove and notify the client.
    c.sent.clear()
    _drain_all(c, "r1")
    _drain_all(c, "r2")
    assert c.requests == {}
    removes = {
        m.body.request_id for e, m in c.sent
        if m.message_type == WorkerMessageType.REMOVE_REQUEST
    }
    assert removes == {"r1", "r2"}
    failures = {
        m.body.request_id: m.body for e, m in c.sent
        if e == "api_server" and m.message_type == "request_failed"
    }
    assert failures["r1"].error_message == "boom"
    assert failures["r2"].error_message == "bang"
    assert failures["r1"].status == 500


def test_duplicate_failure_report_is_ignored():
    """Under TP every rank raises symmetrically and each reports it; only the
    first report may start the teardown."""
    c = _conductor(["r1"])
    c._fail_requests(FailRequests(errors={"r1": "boom"}))
    c.sent.clear()
    c._fail_requests(FailRequests(errors={"r1": "boom"}))  # already draining
    assert c.sent == []


def test_one_unknown_rid_does_not_block_the_others():
    c = _conductor(["r2"])
    c._fail_requests(FailRequests(errors={"gone": "boom", "r2": "bang"}))
    drains = {
        m.body.request_id for e, m in c.sent
        if m.message_type == WorkerMessageType.DRAIN_REQUEST
    }
    assert drains == {"r2"}


# ── api server ─────────────────────────────────────────────────────────────


def _api_server(messages):
    s = APIServer.__new__(APIServer)
    s.pending_requests = {}
    s.recently_completed = {}
    s._recently_completed_ttl = 15.0
    s.request_lock = threading.Lock()
    s.running = True
    s.log_stats = False
    s.cleaned = []
    s.communicator = SimpleNamespace(
        get_all_new_messages=lambda: messages.pop(0) if messages else []
    )
    s.preprocess_worker = SimpleNamespace(
        get_profile_updates=lambda: [],
        get_result_chunks=lambda: [],
        has_pending_tensors=lambda rid: False,
        received_final_chunks=lambda rid, outs: False,
        cleanup_request=s.cleaned.append,
        new_result_tensors=lambda body: None,
        discard_result_tensors=lambda body: None,
    )
    return s


def _pending_request(streaming=False):
    return PendingRequest(
        streaming=streaming,
        input_modalities=["text"],
        output_modalities=["text"],
        profile=RequestProfile(rid="r1", timing=RequestTiming(recv_time=0.0)),
    )


def _drain(server):
    thread = threading.Thread(target=server._process_messages, daemon=True)
    thread.start()
    try:
        yield_deadline = time.time() + 10
        while time.time() < yield_deadline:
            if server.pending_requests["r1"].event.is_set():
                return
            time.sleep(0.005)
        raise AssertionError("request was never released")
    finally:
        server.running = False
        thread.join(timeout=10)


def test_engine_failure_releases_the_client_with_an_error():
    server = _api_server([[APIServerMessage(
        message_type="request_failed",
        body=RequestFailed(request_id="r1", error_message="boom", status=500),
    )]])
    server.pending_requests["r1"] = _pending_request()
    _drain(server)

    req = server.pending_requests["r1"]
    assert req.error == "boom"
    assert req.error_status == 500
    # Parked for cleanup so the data worker's per-request state is reclaimed.
    assert "r1" in server.recently_completed


def test_engine_failure_does_not_clobber_an_earlier_error():
    server = _api_server([[APIServerMessage(
        message_type="request_failed",
        body=RequestFailed(request_id="r1", error_message="downstream", status=500),
    )]])
    req = _pending_request()
    req.error = "bad input"
    req.error_status = 400
    server.pending_requests["r1"] = req
    _drain(server)
    assert req.error == "bad input" and req.error_status == 400


# ── data worker ────────────────────────────────────────────────────────────


def _preprocess_thread(model):
    from mstar.api_server.data_worker import PreprocessWorkerThread

    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    wt.out_queue = queue.Queue()
    wt.model = model
    wt.request_model_kwargs = {}
    wt.tensor_uuid_to_metadata_per_request = {"r1": {"u1": {}}}
    wt.enable_prof = False
    return wt


def test_output_postprocess_failure_becomes_an_error_chunk():
    """A model that blows up postprocessing a result tensor must fail that
    request, not silently drop the chunk and leave the client hanging."""

    class _BadModel:
        def postprocess(self, tensor, modality, request_kwargs=None):
            raise RuntimeError("decode failed")

    wt = _preprocess_thread(_BadModel())
    dereferenced = []
    edge = SimpleNamespace(
        name="text_output", tensor_info=[SimpleNamespace(uuid="u1")],
    )
    wt.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {"r1": [edge]},
        get_tensor=lambda request_id, uuid: object(),
        dereference=lambda request_id, uuid: dereferenced.append(uuid),
    )

    assert wt._process_read_tensors() is True
    chunk: ResultChunk = wt.out_queue.get_nowait()
    assert chunk.request_id == "r1"
    assert chunk.modality == "error"
    assert b"decode failed" in chunk.data
    assert chunk.metadata["status"] == 500
    # The tensor is still released even though postprocessing died.
    assert dereferenced == ["u1"]


def test_result_transfer_failure_answers_for_every_queued_tensor():
    """The API server decrements its outstanding-read count once per chunk, so
    a read that never starts owes one error chunk per tensor it dropped."""
    wt = _preprocess_thread(model=None)
    edge = SimpleNamespace(
        name="text_output",
        tensor_info=[SimpleNamespace(uuid="u1"), SimpleNamespace(uuid="u2")],
    )
    result = ResultTensors(
        request_id="r1", modality="text", graph_edge=edge, loop_indices=None,
    )
    try:
        raise RuntimeError("arena full")
    except RuntimeError as exc:
        wt._fail_request(
            "r1", exc, "text output transfer",
            count=len(result.graph_edge.tensor_info),
        )
    assert wt.out_queue.qsize() == 2
    assert b"arena full" in wt.out_queue.get_nowait().data
