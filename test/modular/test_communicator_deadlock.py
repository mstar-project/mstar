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
# Over the pinned capacity (SNDHWM 50 + the receiver's 1000 RCVHWM) with room
# to spare, and also over the *default* capacity (1000 + 1000) — so the same
# reproduction works against a tree that predates MSTAR_ZMQ_SNDHWM, which is
# how this test was checked to actually discriminate.
_BURST = 3000
_PAYLOAD = "x" * 1024

#: How long to let the deadlocking variant run before calling it deadlocked.
#: The peers park within ~0.2s of the barrier, so this only has to outlast
#: process startup.
_DEADLOCK_WINDOW_S = 5.0
#: The peers' own budget for finishing (they complete in ~1s). Generous: it
#: trips only on a real hang.
_COMPLETION_BUDGET_S = 60.0
#: The parent waits longer than the peers do, so a peer that gives up reports
#: *where* it stopped (exit code + reason file) instead of being killed mid-way
#: and looking like a deadlock.
_PARENT_BUDGET_S = _COMPLETION_BUDGET_S + 20.0


def _peer(
    my_id: str, peer_id: str, prefix: str, legacy: bool,
    barrier, my_done, peer_done,
) -> None:
    """One side of the cycle. Marker files report how far it got.

    Sequence: bind -> barrier -> send a burst at the peer -> drain until the
    peer's burst has fully arrived -> keep draining until the peer is done
    too. The deadlock lands between the barrier and ``<id>.burst_done``.
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
        (marker / f"{my_id}.stalled").write_text(
            f"received {seen}/{_BURST} bursts"
        )
        raise SystemExit(1)

    # Receiving everything is not the end: this peer's own backlog only moves
    # while it keeps polling (the flush hangs off the receive path), and LINGER
    # is 0 so whatever is still queued dies with the process. So keep polling
    # until the peer reports it has everything — which is what the production
    # loops do anyway. Out-of-band events, not a "fin" message: an in-band
    # handshake has the same problem it is trying to solve, since a fin queued
    # behind 1500 burst messages never arrives.
    my_done.set()
    deadline = time.monotonic() + _COMPLETION_BUDGET_S
    while not peer_done.is_set() and time.monotonic() < deadline:
        conn.get_all_new_messages()  # flushes our backlog toward the peer
        time.sleep(0.001)
    if not peer_done.is_set():
        (marker / f"{my_id}.stalled").write_text("peer never finished receiving")
        raise SystemExit(1)
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
    done_a, done_b = ctx.Event(), ctx.Event()
    peers = [
        ctx.Process(
            target=_peer,
            args=("peer_a", "peer_b", prefix, legacy, barrier, done_a, done_b),
        ),
        ctx.Process(
            target=_peer,
            args=("peer_b", "peer_a", prefix, legacy, barrier, done_b, done_a),
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
    peers = _run_cycle(cycle_env, legacy=False, wait_s=_PARENT_BUDGET_S)
    try:
        stalls = {
            f.name: f.read_text() for f in cycle_env.glob("*.stalled")
        }
        assert not [p for p in peers if p.is_alive()], (
            f"cycle deadlocked with the non-blocking send; peer reports: {stalls}"
        )
        assert [p.exitcode for p in peers] == [0, 0], (
            f"peers exited {[p.exitcode for p in peers]}; reports: {stalls}"
        )
        for name in ("peer_a", "peer_b"):
            assert (cycle_env / f"{name}.burst_done").exists()
            assert (cycle_env / f"{name}.done").exists()
    finally:
        _cleanup(peers)
