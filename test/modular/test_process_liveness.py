"""A worker or conductor process that dies must take the deployment down
instead of leaving it hanging.

A dead process sends nothing, so the conductor's startup wait and main loop
poll the worker handles, and the API server polls the conductor handle. Every
waiting client gets a 503 naming the dead process instead of sitting out the
request timeout, new requests are refused, and the processes exit non-zero.
"""

import collections
import os
import queue
import signal
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi import HTTPException

from mstar.api_server.entrypoint import APIServer, DeadConductorError, PendingRequest
from mstar.api_server.request_types import APIServerMessage
from mstar.conductor.conductor import (
    Conductor,
    DeadWorkerError,
    DrainingRequest,
    RequestData,
    _exit_when_orphaned,
)
from mstar.utils.exitcode import describe_exitcode
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    NewRequestConductor,
    SetupDone,
)


class _Proc:
    """Stand-in for a multiprocessing.Process handle."""

    def __init__(self, pid, alive=True, exitcode=None):
        self.pid = pid
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self):
        return self._alive


def _inbox(messages):
    queue = list(messages)

    def get_all_new_messages():
        batch = list(queue)
        queue.clear()
        return batch

    return get_all_new_messages


def _setup_done(worker_id):
    return ConductorMessage(
        message_type=ConductorMessageType.SETUP_DONE, body=SetupDone(worker_id=worker_id)
    )


def _request_data(workers):
    return RequestData(
        persist_signals={},
        persist_signal_ref_cnt={},
        worker_graph_to_workers={0: list(workers)},
        all_worker_graph_ids={0},
        max_output_tokens=1,
        random_seed=0,
        resource_configs={},
    )


def _queued(rid):
    return NewRequestConductor(
        request_id=rid,
        initial_signals={},
        initial_input_modalities=["text"],
        initial_output_modalities=["text"],
        input_metadata={},
        model_kwargs={},
    )


def _conductor(procs, inbox=()):
    c = Conductor.__new__(Conductor)
    c.sent = []
    c.communicator = SimpleNamespace(
        send=lambda entity, msg: c.sent.append((entity, msg)),
        get_all_new_messages=_inbox(inbox),
    )
    c.worker_ids = [f"worker_{i}" for i in range(len(procs))]
    c._worker_processes = list(procs)
    c._per_worker_graphs = {}
    c._startup_message_backlog = []
    c._liveness_interval_s = 0.0
    c._next_liveness_check = 0.0
    c.requests = {}
    c.draining = {}
    c.waiting_queue = []
    return c


def _failed(c):
    return {
        m.body.request_id: m.body
        for entity, m in c.sent
        if entity == "api_server" and m.message_type == "request_failed"
    }


def test_describe_exitcode():
    assert describe_exitcode(None) == "an unknown status"
    assert describe_exitcode(0) == "exit code 0"
    assert describe_exitcode(1) == "exit code 1"
    assert describe_exitcode(-9) == "signal SIGKILL"
    assert describe_exitcode(-200) == "signal 200"


def test_startup_wait_returns_once_every_worker_reports(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    c = _conductor(
        [_Proc(11), _Proc(12)], inbox=[_setup_done("worker_1"), _setup_done("worker_0")]
    )
    c._wait_for_workers_ready()
    assert c.sent == []


def test_startup_wait_raises_when_a_worker_dies_before_reporting(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    c = _conductor([_Proc(11), _Proc(12, alive=False, exitcode=1)], inbox=[_setup_done("worker_0")])
    with pytest.raises(DeadWorkerError) as exc:
        c._wait_for_workers_ready()
    assert (exc.value.worker_id, exc.value.pid, exc.value.exitcode) == ("worker_1", 12, 1)
    assert "worker_1 (pid 12) exited with exit code 1" in str(exc.value)
    # Noticed on the first poll, with no waiting on a SETUP_DONE that will never come.
    assert sleeps == []


def test_liveness_poll_is_throttled():
    c = _conductor([_Proc(11, alive=False, exitcode=-9)])
    c._liveness_interval_s = 60.0
    c._next_liveness_check = time.perf_counter() + 60.0
    c._poll_worker_liveness()  # not due yet
    c._next_liveness_check = 0.0
    with pytest.raises(DeadWorkerError, match="signal SIGKILL"):
        c._poll_worker_liveness()


def test_liveness_poll_is_quiet_while_workers_live():
    c = _conductor([_Proc(11), _Proc(12)])
    c.requests = {"r": _request_data(["worker_0"])}
    c._poll_worker_liveness()
    assert c.sent == []


def test_dead_worker_fails_every_waiting_client_with_503():
    c = _conductor([_Proc(11), _Proc(12, alive=False, exitcode=-11)])
    c.requests = {
        "on_dead": _request_data(["worker_1"]),
        "elsewhere": _request_data(["worker_0"]),
        "failing": _request_data(["worker_1"]),
        "completed": _request_data(["worker_0"]),
    }
    c.draining = {
        # Fail path. The client is only told once the barrier completes, and
        # the dead participant will never ACK, so tell it now.
        "failing": DrainingRequest(
            expected_acks={"worker_1"}, participants={"worker_1"}, failure_error="boom"
        ),
        # Happy path, request_complete already went out.
        "completed": DrainingRequest(expected_acks={"pp"}, participants={"pp"}),
    }
    c.waiting_queue = [_queued("queued")]

    with pytest.raises(DeadWorkerError):
        c._poll_worker_liveness()

    failed = _failed(c)
    assert set(failed) == {"on_dead", "elsewhere", "failing", "queued"}
    for body in failed.values():
        assert body.status == 503
        assert "worker_1 (pid 12) exited with signal SIGSEGV" in body.error_message


def test_orphaned_worker_leaves_gracefully_then_hard(monkeypatch):
    calls = []
    monkeypatch.setattr(time, "sleep", lambda s: calls.append(("sleep", s)))
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append(("kill", pid, sig)))

    def _exit(code):
        calls.append(("_exit", code))
        raise SystemExit(code)  # os._exit never returns, so stand in for it

    monkeypatch.setattr(os, "_exit", _exit)
    alive = [True, True, False]
    parent = SimpleNamespace(is_alive=lambda: alive.pop(0))

    with pytest.raises(SystemExit):
        _exit_when_orphaned("worker_0", parent=parent, poll_s=0.0)

    kill_at = calls.index(("kill", os.getpid(), signal.SIGTERM))
    assert calls[-1] == ("_exit", 1)
    assert kill_at < len(calls) - 1
    assert ("sleep", 5.0) in calls[kill_at:]


def test_orphaned_conductor_leaves_gracefully_then_hard(monkeypatch):
    """Symmetric with the worker watchdog: an API server killed outright runs
    no cleanup, so the conductor must notice and take its workers down with it
    rather than leave them holding GPU memory."""
    from mstar.utils.orphan import exit_when_orphaned

    calls = []
    monkeypatch.setattr(time, "sleep", lambda s: calls.append(("sleep", s)))
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append(("kill", pid, sig)))

    def _exit(code):
        calls.append(("_exit", code))
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", _exit)
    alive = [True, False]
    parent = SimpleNamespace(is_alive=lambda: alive.pop(0))

    with pytest.raises(SystemExit):
        exit_when_orphaned(
            "Conductor", "API server", signal.SIGINT, parent=parent, poll_s=0.0,
        )

    # SIGINT, not SIGTERM: run() unwinds into shutdown(), which stops the workers.
    kill_at = calls.index(("kill", os.getpid(), signal.SIGINT))
    assert calls[-1] == ("_exit", 1)
    assert ("sleep", 5.0) in calls[kill_at:]


def test_conductor_process_target_watches_the_api_server():
    """The watchdog has to start before the model load, so an API server that
    dies during weight loading is still caught."""
    import inspect

    from mstar.api_server import entrypoint

    source = inspect.getsource(entrypoint._conductor_process_target)
    watch_at = source.index("watch_parent(")
    assert "signal.SIGINT" in source[watch_at:source.index("\n", watch_at)]
    assert watch_at < source.index("get_model_class")


def test_conductor_process_target_rearms_sigint_before_watching():
    """An inherited SIGINT ignore would make the watchdog's graceful stop a
    no-op, so the handler is installed before the watchdog starts."""
    import inspect

    from mstar.api_server import entrypoint

    source = inspect.getsource(entrypoint._conductor_process_target)
    install_at = source.index(
        "signal.signal(signal.SIGINT, signal.default_int_handler)"
    )
    assert install_at < source.index("watch_parent(")


def test_worker_without_a_multiprocessing_parent_keeps_running():
    # A test process has no multiprocessing parent, so the watchdog is a no-op.
    assert _exit_when_orphaned("worker_0") is None


# ---------------------------------------------------------------- API server


def test_stopping_data_worker_drops_tracked_requests():
    """No RemoveRequest is coming for an in-flight request once the server
    stops, so the thread hard-cleans what it still tracks on its way out.

    ``r1`` is the case that matters: a request whose input signals were
    written but which never produced an output tensor, so it appears only in
    ``in_flight_requests``. Tracking teardown off the output-metadata dict
    misses exactly that request and leaks its signals into /dev/shm.
    """
    from mstar.api_server.data_worker import PreprocessWorkerThread

    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    for name in ("in_queue", "out_queue", "result_tensor_queue", "cleanup_request_queue",
                 "abort_request_queue", "reads_done_queue", "discard_tensor_queue"):
        setattr(wt, name, queue.Queue())
    wt.stop_event = threading.Event()
    wt.stop_event.set()
    wt.communicator = SimpleNamespace(get_all_new_messages=lambda: [])
    cleaned = []
    wt.tensor_manager = SimpleNamespace(
        force_cleanup_request=cleaned.append,
        has_inflight_reads=lambda rid: False,
        get_ready_tensors=lambda: {},
    )
    wt.in_flight_requests = {"r1", "r2"}
    wt.tensor_uuid_to_metadata_per_request = {"r2": {}}
    wt.request_model_kwargs = {"r1": {}}
    wt._draining_rids = {"r1"}
    wt._reads_done_sent = set()

    wt.run()

    assert sorted(cleaned) == ["r1", "r2"]
    assert wt.in_flight_requests == set()
    assert wt.tensor_uuid_to_metadata_per_request == {}
    assert wt.request_model_kwargs == {}


def _pending_request():
    return PendingRequest(
        streaming=True, input_modalities=["text"], output_modalities=["text"], profile=None
    )


def _api_server(conductor_proc, inbox=()):
    server = object.__new__(APIServer)
    server.pending_requests = {}
    server.recently_completed = collections.OrderedDict()
    server._recently_completed_ttl = 15.0
    server.request_lock = threading.Lock()
    server.running = True
    server.conductor_proc = conductor_proc
    server.fatal_error = None
    server.on_fatal = None
    server._liveness_interval_s = 0.0
    server.communicator = SimpleNamespace(get_all_new_messages=_inbox(inbox))
    server.preprocess_worker = SimpleNamespace(
        get_profile_updates=lambda: [], get_result_chunks=lambda: []
    )
    server.started = []
    server._msg_thread = SimpleNamespace(start=lambda: server.started.append(True))
    return server


def test_finalize_setup_returns_on_setup_done(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    server = _api_server(_Proc(5), inbox=[APIServerMessage(message_type="setup_done")])
    server.finalize_setup()
    assert server.started == [True]


def test_finalize_setup_raises_when_the_conductor_dies_first(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    server = _api_server(_Proc(5, alive=False, exitcode=1))
    with pytest.raises(DeadConductorError, match="exit code 1"):
        server.finalize_setup()
    assert server.started == []
    assert sleeps == []


def test_message_loop_runs_while_the_conductor_lives():
    server = _api_server(_Proc(5))
    thread = threading.Thread(target=server._process_messages, daemon=True)
    thread.start()
    time.sleep(0.05)
    server.running = False
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert server.fatal_error is None


def test_dead_conductor_releases_pending_requests_and_stops_the_server():
    stopped = []
    server = _api_server(_Proc(5, alive=False, exitcode=-9))
    server.on_fatal = lambda: stopped.append(True)
    waiting = _pending_request()
    already_failed = _pending_request()
    already_failed.error, already_failed.error_status = "bad knob", 400
    server.pending_requests = {"r1": waiting, "r2": already_failed}

    server._process_messages()  # returns once it has acted

    assert waiting.event.is_set()
    assert waiting.error_status == 503
    assert "conductor process exited with signal SIGKILL" in waiting.error
    # An earlier, more specific error is kept, and the client is still released.
    assert (already_failed.error, already_failed.error_status) == ("bad knob", 400)
    assert already_failed.event.is_set()
    assert stopped == [True]
    assert server.fatal_error is not None
    with pytest.raises(HTTPException) as exc:
        server.submit_request(input_modalities=["text"], output_modalities=["text"])
    assert exc.value.status_code == 503


def test_stop_callback_registered_after_the_conductor_died_still_runs():
    """main() registers what stops the HTTP server after finalize_setup has
    started the message thread, so the thread can find the conductor dead
    before anything is registered. The late registration has to stop the
    server itself, or the process would keep answering 503 forever."""
    stopped = []
    server = _api_server(_Proc(5, alive=False, exitcode=1))
    server._process_messages()  # found dead, nothing to call yet
    assert server.fatal_error is not None
    server.set_on_fatal(lambda: stopped.append("late"))
    assert stopped == ["late"]

    # Registered first, the message thread runs it, once.
    server = _api_server(_Proc(5, alive=False, exitcode=1))
    server.set_on_fatal(lambda: stopped.append("early"))
    assert stopped == ["late"]
    server._process_messages()
    assert stopped == ["late", "early"]


def test_health_reports_unhealthy_once_fatal():
    """A load balancer must stop routing here: /health has to fail as soon as
    the deployment is going down, not stay 200 until the process exits."""
    from fastapi.testclient import TestClient

    import mstar.api_server.entrypoint as ep

    client = TestClient(ep.app)
    previous = ep.api_server
    try:
        ep.api_server = SimpleNamespace(fatal_error=None)
        assert client.get("/health").status_code == 200

        ep.api_server = SimpleNamespace(
            fatal_error="worker worker_0 (pid 7) exited with signal SIGKILL"
        )
        response = client.get("/health")
        assert response.status_code == 503
        assert "worker_0" in response.json()["detail"]
    finally:
        ep.api_server = previous


def test_dynamo_worker_exits_nonzero_when_conductor_dies():
    """The Dynamo entrypoint runs the same APIServer, so a dead conductor has
    to fail that process too rather than leave it registered and serving."""
    from mstar.api_server.entrypoint import DeadConductorError
    from mstar.integrations.dynamo import worker as dynamo_worker

    conductor = _Proc(11, alive=False, exitcode=1)
    server = SimpleNamespace(
        conductor_proc=None,
        fatal_error=None,
        cleanup=lambda: None,
        finalize_setup=lambda: (_ for _ in ()).throw(
            DeadConductorError("conductor process exited with exit code 1")
        ),
    )
    argv = ["--config", "c.yaml", "--model-path", "/tmp/m"]
    with (
        mock.patch.object(dynamo_worker, "build_server",
                          return_value=(server, conductor, "whisper_large")),
        mock.patch.object(dynamo_worker, "serve"),
        mock.patch("mstar.api_server.entrypoint._shutdown_conductor_process"),
        pytest.raises(SystemExit) as exc,
    ):
        dynamo_worker.main(argv)
    assert exc.value.code == 1
