"""``Conductor.run`` waits on its communicator's poll between passes, over the
real transports: a message ends the wait at once, each timed check ends it on
its own deadline with no message at all, without the loop spinning, and while a
request is in flight the wait is at most ``MSTAR_CONDUCTOR_TICK_MS``."""
import threading
import time

import pytest

pytest.importorskip("torch")

import mstar.communication.wire_types  # noqa: E402,F401  (registers the wire tags)
import mstar.conductor.conductor as conductor_mod  # noqa: E402
from mstar.api_server.request_types import ConductorStatus  # noqa: E402
from mstar.communication.communicator import ZMQCommunicator  # noqa: E402
from mstar.conductor.conductor import Conductor  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    ConductorMessage,
    ConductorMessageType,
    ReadsDone,
)

FAR_S = 60.0  # no check but the one under test comes due during a test
DUE_S = 0.4

TIMED_CHECKS = {
    "liveness": lambda c, at: setattr(c, "_next_liveness_check", at),
    "status": lambda c, at: setattr(c, "_next_status", at),
    "drain": lambda c, at: c._draining_deadlines.append((at, "r")),
    "early_reads_done": lambda c, at: c._early_reads_done_deadlines.append((at, "r")),
    "early_abort": lambda c, at: c._early_abort_deadlines.append((at, "r")),
}


class _StopLoop(BaseException):
    pass


def _communicator(kind, my_id, prefix):
    if kind == "rust":
        pytest.importorskip("mstar_rust")
        from mstar.communication.rust_communicator import RustZMQCommunicator as Comm
    else:
        Comm = ZMQCommunicator
    return Comm(my_id, push_ids=[], ipc_socket_path_prefix=prefix)


def _conductor(comm):
    """The state ``run`` reads; ``_worker_processes`` empty, so nothing is dead."""
    c = Conductor.__new__(Conductor)
    c.communicator = comm
    c.worker_ids, c._worker_processes = [], []
    c.enable_nvtx = False
    c._startup_message_backlog = []
    c.requests, c.draining, c.waiting_queue = {}, {}, []
    c._early_reads_done, c._early_abort_requests = {}, set()
    c._draining_deadlines = conductor_mod.deque()
    c._early_reads_done_deadlines = conductor_mod.deque()
    c._early_abort_deadlines = conductor_mod.deque()
    c._drain_ttl_s = FAR_S
    c._tick_ms = 1
    c._liveness_interval_s = c._status_interval_s = FAR_S
    c._status = lambda: ConductorStatus(metrics=[])
    return c


def _wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True


@pytest.mark.parametrize("kind", ["python", "rust"])
@pytest.mark.parametrize("check", TIMED_CHECKS)
def test_a_message_or_a_timed_check_ends_the_wait(kind, check, tmp_path, monkeypatch):
    monkeypatch.setattr(conductor_mod, "freeze_after_setup", lambda _name: None)
    prefix = str(tmp_path) + "/"
    comm = _communicator(kind, "conductor", prefix)
    peer = _communicator(kind, "peer", prefix)
    comm.send = lambda entity, msg: None
    c = _conductor(comm)

    passes = []
    drain = comm.get_all_new_messages

    def counted_drain():
        passes.append(time.perf_counter())
        return drain()

    comm.get_all_new_messages = counted_drain

    stop = threading.Event()
    liveness = c._poll_worker_liveness

    def stop_when_told():
        liveness()
        if stop.is_set():
            raise _StopLoop

    c._poll_worker_liveness = stop_when_told

    def run():
        try:
            c.run()
        except _StopLoop:
            pass

    c._next_liveness_check = c._next_status = 0.0
    assert c._ms_to_next_timed_check() == 1, "a check already due must not spin the loop"

    now = time.perf_counter()
    c._next_liveness_check = c._next_status = now + FAR_S
    due = now + DUE_S
    TIMED_CHECKS[check](c, due)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: len(passes) == 1)

        peer.send("conductor", ConductorMessage(ConductorMessageType.READS_DONE, ReadsDone("x", "peer")))
        assert _wait_until(lambda: "x" in c._early_reads_done)
        assert passes[1] < due and len(passes) == 2

        # no message from here on: only the deadline can end this wait
        assert _wait_until(lambda: len(passes) >= 3)
        assert passes[2] >= due - 0.001
        time.sleep(0.05)
        assert len(passes) <= 4, "the loop must go back to waiting"
    finally:
        stop.set()
        peer.send("conductor", ConductorMessage(ConductorMessageType.READS_DONE, ReadsDone("y", "peer")))
        thread.join(timeout=5.0)
    assert not thread.is_alive()


class _Loop:
    """``c.run`` on its own thread, with its passes counted."""

    def __init__(self, c):
        self.c, self.passes, self._stop = c, [], threading.Event()
        drain = c.communicator.get_all_new_messages

        def counted_drain():
            self.passes.append(time.perf_counter())
            return drain()

        c.communicator.get_all_new_messages = counted_drain
        liveness = c._poll_worker_liveness

        def stop_when_told():
            liveness()
            if self._stop.is_set():
                raise _StopLoop

        c._poll_worker_liveness = stop_when_told
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            self.c.run()
        except _StopLoop:
            pass

    def __enter__(self):
        self._thread.start()
        assert _wait_until(lambda: self.passes)
        return self

    def __exit__(self, *_exc):
        self._stop.set()

    def passes_over(self, seconds):
        before = len(self.passes)
        time.sleep(seconds)
        return len(self.passes) - before


def _idle_loop(kind, tmp_path, monkeypatch):
    """A conductor with nothing in flight and no check due, and a peer to end its wait."""
    monkeypatch.setattr(conductor_mod, "freeze_after_setup", lambda _name: None)
    prefix = str(tmp_path) + "/"
    comm = _communicator(kind, "conductor", prefix)
    peer = _communicator(kind, "peer", prefix)
    comm.send = lambda entity, msg: None
    c = _conductor(comm)
    c._next_liveness_check = c._next_status = time.perf_counter() + FAR_S
    return c, peer, _Loop(c)


def _end_the_wait(peer):
    peer.send("conductor", ConductorMessage(ConductorMessageType.READS_DONE, ReadsDone("x", "peer")))


@pytest.mark.parametrize("kind", ["python", "rust"])
def test_the_tick_runs_only_while_a_request_is_in_flight(kind, tmp_path, monkeypatch):
    monkeypatch.delenv("MSTAR_CONDUCTOR_TICK_MS", raising=False)
    c, peer, loop = _idle_loop(kind, tmp_path, monkeypatch)
    with loop:
        assert loop.passes_over(0.05) <= 1, "idle, the wait runs to the next timed check"

        c.requests["r"] = object()
        _end_the_wait(peer)
        assert loop.passes_over(0.1) >= 10, "in flight, the loop passes on the tick"

        del c.requests["r"]
        time.sleep(0.01)  # the wait in progress ends on its tick
        assert loop.passes_over(0.05) <= 1, "idle again, back to the long wait"
        _end_the_wait(peer)


@pytest.mark.parametrize("kind", ["python", "rust"])
def test_tick_zero_keeps_the_long_wait_with_a_request_in_flight(kind, tmp_path, monkeypatch):
    monkeypatch.setenv("MSTAR_CONDUCTOR_TICK_MS", "0")
    c, peer, loop = _idle_loop(kind, tmp_path, monkeypatch)
    c.requests["r"] = object()
    with loop:
        assert c._tick_ms == 0
        assert loop.passes_over(0.05) <= 1
        _end_the_wait(peer)


def test_the_tick_is_a_cap_on_the_wait():
    c = _conductor(None)
    now = time.perf_counter()
    c._next_liveness_check = c._next_status = now + FAR_S
    c._tick_ms = 5
    assert c._ms_to_next_timed_check() > 5, "nothing in flight"

    c.requests["r"] = object()
    assert c._ms_to_next_timed_check() == 5

    c._next_liveness_check = now + 0.002
    assert c._ms_to_next_timed_check() <= 3, "a timed check due sooner still ends the wait"
