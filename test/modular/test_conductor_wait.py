"""``Conductor.run`` waits on its communicator's poll between passes, over the
real transports: a message ends the wait at once, and each timed check ends it
on its own deadline with no message at all, without the loop spinning."""
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
