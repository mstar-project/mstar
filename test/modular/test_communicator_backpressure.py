"""Regression tests for ZMQCommunicator non-blocking send / backpressure.

The worker<->conductor topology is a PUSH/PULL cycle: each drains its PULL in
the same loop that issues PUSH sends. If a PUSH blocked when the peer's receive
buffer filled, the caller would stop draining its own PULL, so two peers could
each block sending to the other while neither drained — a deadlock observed
under concurrent omni serving load.

The fix makes send() non-blocking: when a peer is full, the message is queued
in-process and flushed opportunistically. These tests pin that behavior: send
never blocks, and queued messages deliver in FIFO order with no loss once the
peer drains.
"""
import os
import shutil
import tempfile
import time

import pytest

import mstar.communication.communicator as comm
from mstar.communication.communicator import CommProtocol, ZMQCommunicator


@pytest.fixture
def socket_dir():
    """A private IPC socket namespace, removed with the test.

    Per-test rather than per-pid: two tests sharing a prefix share endpoints,
    so one test's undelivered backlog would land in the other's receiver.
    """
    path = tempfile.mkdtemp(prefix="mstar_bptest_")
    yield path + "/"
    shutil.rmtree(path, ignore_errors=True)


def test_send_does_not_block_without_receiver(socket_dir):
    """No peer is bound, so nothing is ever consumed — send must still return.

    This is the deadlock's shape in miniature: with a blocking send, the caller
    parks here forever instead of getting back to its own receive loop. The
    count has to exceed zmq's default 1000-message HWM, or the messages fit in
    the socket's own queue and even a blocking send returns.
    """
    sender = ZMQCommunicator(
        "sender", ["peer"], protocol=CommProtocol.IPC,
        ipc_socket_path_prefix=socket_dir,
    )
    start = time.monotonic()
    for i in range(3000):
        sender.send("peer", {"i": i, "pad": "x" * 10_000})
    assert time.monotonic() - start < 5.0, "send blocked with no receiver"


def test_overflow_queues_locally_then_delivers_in_order(socket_dir, monkeypatch):
    """Overflow goes to the in-process queue and still arrives, in order."""
    # Force a tiny HWM so the zmq.Again -> local-queue path triggers
    # deterministically. _SNDHWM is read at import time and applied when the
    # socket is created, so patch the module constant before constructing.
    monkeypatch.setattr(comm, "_SNDHWM", 10)

    sender = ZMQCommunicator(
        "sender", ["peer"], protocol=CommProtocol.IPC,
        ipc_socket_path_prefix=socket_dir,
    )
    n = 200
    start = time.monotonic()
    for i in range(n):
        sender.send("peer", {"i": i, "pad": "x" * 5_000})
    assert time.monotonic() - start < 5.0, "send blocked on overflow"
    assert sender.outbound.get("peer"), "expected messages to queue in-process"

    receiver = ZMQCommunicator(
        "peer", [], protocol=CommProtocol.IPC,
        ipc_socket_path_prefix=socket_dir,
    )
    got = []
    deadline = time.monotonic() + 30
    while len(got) < n and time.monotonic() < deadline:
        sender._flush_outbound()
        got += receiver.get_all_new_messages()
        time.sleep(0.01)

    assert len(got) == n, f"lost messages: {len(got)}/{n}"
    assert [m["i"] for m in got] == list(range(n)), "FIFO order broken"
    assert not sender.outbound.get("peer"), "backlog not fully drained"


def test_poll_points_flush_the_backlog(socket_dir, monkeypatch):
    """A backlog drains from the receive path alone — no further send() needed.

    That is what unblocks a stalled peer in production: the loop that stopped
    sending is still polling, and each poll retries the queue.
    """
    monkeypatch.setattr(comm, "_SNDHWM", 10)

    sender = ZMQCommunicator(
        "sender", ["peer"], protocol=CommProtocol.IPC,
        ipc_socket_path_prefix=socket_dir,
    )
    n = 100
    for i in range(n):
        sender.send("peer", {"i": i, "pad": "x" * 5_000})
    assert sender.outbound.get("peer"), "expected messages to queue in-process"

    receiver = ZMQCommunicator(
        "peer", [], protocol=CommProtocol.IPC,
        ipc_socket_path_prefix=socket_dir,
    )
    got = []
    deadline = time.monotonic() + 30
    while len(got) < n and time.monotonic() < deadline:
        # The sender only ever polls its own inbox here.
        sender.get_all_new_messages()
        got += receiver.get_all_new_messages()
        time.sleep(0.01)

    assert [m["i"] for m in got] == list(range(n))
    assert not sender.outbound.get("peer"), "backlog not fully drained"


def test_sndhwm_is_env_overridable():
    """Deployments tune the burst headroom without a code change."""
    assert comm._SNDHWM == int(os.getenv("MSTAR_ZMQ_SNDHWM", "100000"))
