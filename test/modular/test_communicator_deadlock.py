"""End-to-end proof that the PUSH/PULL cycle deadlock is real, and fixed.

``test_communicator_backpressure.py`` pins the *primitive* (send doesn't block).
This file reproduces the actual failure: two processes wired into a PUSH/PULL
cycle — the worker<->conductor shape, where each peer drains its PULL in the
same loop that issues its PUSH sends.

Both variants run the identical peer program; only ``send`` differs:

* ``legacy`` — the pre-fix blocking ``send_pyobj``. Each peer fills the other's
  queue before either reaches its drain loop, so both park in ``send`` and
  neither ever drains. Deadlock: neither peer finishes its send burst.
* ``current`` — the shipped non-blocking ``send``. Overflow is queued
  in-process and flushed from the drain loop, so both bursts finish and
  everything is delivered.

CPU-only and a few seconds; no GPU, no model, no server. That matters because
the original incident was only ever reproduced under live multi-request serving
load, which CI cannot run.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

# The peers have to overflow the transport, and the capacity they must overflow
# is SNDHWM (the sender's queue) + RCVHWM (the receiver's, serviced by zmq's own
# IO thread whether or not the app drains) + kernel buffers. SNDHWM is pinned
# small via the env var below; RCVHWM stays at its 1000-message default, so the
# burst has to comfortably exceed that.
_TEST_SNDHWM = "50"
_BURST = 3000
_PAYLOAD = "x" * 1024

#: How long to let the deadlocking variant run before calling it deadlocked.
#: The peers park within ~0.2s of the barrier, so this only has to outlast
#: process startup.
_DEADLOCK_WINDOW_S = 5.0
#: Budget for the fixed variant (it completes in ~1s). Generous: it fails only
#: on a real hang.
_COMPLETION_BUDGET_S = 60.0


def _peer(my_id: str, peer_id: str, prefix: str, legacy: bool, barrier) -> None:
    """One side of the cycle. Marker files report how far it got.

    Sequence: bind -> barrier -> send a burst at the peer -> drain until the
    peer's burst has fully arrived -> ``fin`` handshake. The deadlock lands
    between the barrier and ``<id>.burst_done``.
    """
    import mstar.communication.communicator as comm
    from mstar.communication.communicator import CommProtocol, ZMQCommunicator

    if legacy:
        # The pre-fix send: blocking, no local queue. Socket setup is shared
        # with the current implementation (so SNDHWM is pinned identically in
        # both variants) — the blocking call is the only difference under test.
        def _blocking_send(self, entity_id, msg):
            self._socket_for(entity_id).send_pyobj(msg)

        comm.ZMQCommunicator.send = _blocking_send

    marker = Path(prefix).parent
    conn = ZMQCommunicator(
        my_id, [peer_id], protocol=CommProtocol.IPC,
        ipc_socket_path_prefix=prefix,
    )

    # Both PULL sockets are bound before any burst starts, and both bursts
    # start together. Without the barrier the peers stagger: whoever starts
    # first drains the other's messages while the other is still setting up,
    # and the cycle never closes. A barrier (not a message handshake) keeps the
    # drain loops untouched until the burst.
    barrier.wait(timeout=30)
    (marker / f"{my_id}.started").touch()

    # The cycle: a send burst issued BEFORE this peer drains anything. With a
    # blocking send both peers stop here, each waiting on the other.
    for i in range(_BURST):
        conn.send(peer_id, {"kind": "burst", "i": i, "pad": _PAYLOAD})
    (marker / f"{my_id}.burst_done").touch()

    seen = 0
    deadline = time.monotonic() + _COMPLETION_BUDGET_S
    while seen < _BURST and time.monotonic() < deadline:
        # Also flushes this peer's own backlog — the property that lets a
        # stalled cycle recover with no further send() calls.
        seen += sum(
            1 for m in conn.get_all_new_messages() if m.get("kind") == "burst"
        )
        time.sleep(0.001)
    if seen != _BURST:
        raise SystemExit(f"{my_id}: received {seen}/{_BURST}")

    # Don't exit while the peer still needs us: LINGER is 0, so anything left
    # in our socket (or our backlog) dies with the process.
    conn.send(peer_id, {"kind": "fin"})
    peer_done = False
    deadline = time.monotonic() + _COMPLETION_BUDGET_S
    while not peer_done and time.monotonic() < deadline:
        peer_done = any(
            m.get("kind") == "fin" for m in conn.get_all_new_messages()
        )
        time.sleep(0.001)
    if not peer_done:
        raise SystemExit(f"{my_id}: peer never finished")
    (marker / f"{my_id}.done").touch()


@pytest.fixture
def cycle_env(monkeypatch):
    """Private socket dir + a small SNDHWM, inherited by the spawned peers."""
    root = tempfile.mkdtemp(prefix="mstar_cycle_")
    monkeypatch.setenv("MSTAR_ZMQ_SNDHWM", _TEST_SNDHWM)
    # ``spawn`` re-imports in the child, so the env var is read there; the
    # parent never builds a communicator.
    yield Path(root)
    shutil.rmtree(root, ignore_errors=True)


def _run_cycle(root: Path, legacy: bool, wait_s: float) -> list[mp.Process]:
    prefix = str(root / "sock") + "/"
    os.makedirs(prefix, exist_ok=True)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    peers = [
        ctx.Process(
            target=_peer, args=("peer_a", "peer_b", prefix, legacy, barrier),
        ),
        ctx.Process(
            target=_peer, args=("peer_b", "peer_a", prefix, legacy, barrier),
        ),
    ]
    for p in peers:
        p.start()
    deadline = time.monotonic() + wait_s
    for p in peers:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    return peers


def _cleanup(peers: list[mp.Process]) -> None:
    for p in peers:
        if p.is_alive():
            p.terminate()
    for p in peers:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join(timeout=5)


def test_blocking_send_deadlocks_the_cycle(cycle_env):
    """The negative control: with the pre-fix blocking send, the cycle wedges.

    Without this, the fix has no evidence it prevents anything — the failure it
    targets was only ever seen on a GPU box under live serving load.
    """
    peers = _run_cycle(cycle_env, legacy=True, wait_s=_DEADLOCK_WINDOW_S)
    try:
        assert (cycle_env / "peer_a.started").exists(), "peers never started"
        assert (cycle_env / "peer_b.started").exists(), "peers never started"
        # The deadlock's signature: both up, both stuck mid-burst.
        assert not (cycle_env / "peer_a.burst_done").exists()
        assert not (cycle_env / "peer_b.burst_done").exists()
        assert all(p.is_alive() for p in peers), "expected both peers wedged"
    finally:
        _cleanup(peers)


def test_non_blocking_send_completes_the_cycle(cycle_env):
    """The same cycle on the shipped send: both peers finish and deliver."""
    peers = _run_cycle(cycle_env, legacy=False, wait_s=_COMPLETION_BUDGET_S)
    try:
        assert not [p for p in peers if p.is_alive()], (
            "cycle deadlocked with the non-blocking send"
        )
        assert [p.exitcode for p in peers] == [0, 0], (
            f"peers exited {[p.exitcode for p in peers]} "
            "(non-zero = burst not fully received)"
        )
        for name in ("peer_a", "peer_b"):
            assert (cycle_env / f"{name}.burst_done").exists()
            assert (cycle_env / f"{name}.done").exists()
    finally:
        _cleanup(peers)
