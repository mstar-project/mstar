"""Regression tests for ZMQCommunicator non-blocking send / backpressure.

The worker<->conductor topology is a PUSH/PULL cycle: each drains its PULL
in the same loop that issues PUSH sends. If a PUSH blocked when the peer's
receive buffer filled, the caller would stop draining its own PULL, so two
peers could each block sending to the other while neither drained — a
deadlock observed under concurrent Ming-flash-omni serving load.

The fix makes send() non-blocking: when a peer is full, the message is
queued in-process and flushed opportunistically. These tests pin that
behavior: send never blocks, and queued messages deliver in FIFO order
with no loss once the peer drains.
"""
import os
import time

from mstar.communication.communicator import CommProtocol, ZMQCommunicator


def _prefix(tag: str) -> str:
    p = f"/tmp/mstar_bptest_{tag}_{os.getpid()}/"
    os.makedirs(p, exist_ok=True)
    return p


def test_send_does_not_block_without_receiver():
    s = ZMQCommunicator(
        "sender", ["peer"], protocol=CommProtocol.IPC,
        ipc_socket_path_prefix=_prefix("noblock"),
    )
    t0 = time.time()
    for i in range(500):
        s.send("peer", {"i": i, "pad": "x" * 10_000})
    assert time.time() - t0 < 5.0, "send blocked with no receiver"


def test_overflow_queues_locally_then_delivers_in_order(monkeypatch):
    # Force a tiny HWM so the zmq.Again -> local-queue path triggers
    # deterministically. _SNDHWM is read at module load, so patch the
    # already-imported constant rather than the env var.
    import mstar.communication.communicator as comm
    monkeypatch.setattr(comm, "_SNDHWM", 10)

    prefix = _prefix("overflow")
    s = comm.ZMQCommunicator(
        "sender", ["peer"], protocol=CommProtocol.IPC,
        ipc_socket_path_prefix=prefix,
    )
    n = 200
    t0 = time.time()
    for i in range(n):
        s.send("peer", {"i": i, "pad": "x" * 5_000})
    assert time.time() - t0 < 5.0, "send blocked on overflow"
    assert len(s.outbound.get("peer", [])) > 0, "expected local queueing"

    r = comm.ZMQCommunicator(
        "peer", [], protocol=CommProtocol.IPC,
        ipc_socket_path_prefix=prefix,
    )
    got = []
    deadline = time.time() + 30
    while len(got) < n and time.time() < deadline:
        s._flush_outbound()
        got += r.get_all_new_messages()
        time.sleep(0.01)

    assert len(got) == n, f"lost messages: {len(got)}/{n}"
    assert [m["i"] for m in got] == list(range(n)), "FIFO order broken"
    assert not s.outbound.get("peer"), "backlog not fully drained"
