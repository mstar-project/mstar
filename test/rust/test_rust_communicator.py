"""RustZMQCommunicator interop with the pyzmq ZMQCommunicator (RFC #130
Step 1): same mesh, pickle wire, eventfd wakeup, lossless readiness polls.
Skipped unless the ``mstar_rust`` extension (built from ``rust/``) is
installed."""
import os
import tempfile
import threading
import time

import pytest

pytest.importorskip("mstar_rust")

from mstar.communication.communicator import ZMQCommunicator
from mstar.communication.event import EventWakeup
from mstar.communication.rust_communicator import RustZMQCommunicator


def _wait_msgs(comm, n=1, timeout=5.0):
    out = []
    deadline = time.time() + timeout
    while len(out) < n and time.time() < deadline:
        out.extend(comm.get_all_new_messages())
        time.sleep(0.005)
    return out


@pytest.fixture()
def pair():
    prefix = tempfile.mkdtemp(prefix="mstar_wrap_test_")
    orig = ZMQCommunicator("orig", push_ids=["rust"], ipc_socket_path_prefix=prefix)
    rust = RustZMQCommunicator("rust", push_ids=["orig"], ipc_socket_path_prefix=prefix)
    return orig, rust


def test_pickle_interop_both_directions(pair):
    orig, rust = pair
    payload = {"op": "execute", "rids": [1, 2, 3], "nested": {"f": 1.5}}
    orig.send("rust", payload)
    assert _wait_msgs(rust) == [payload]
    rust.send("orig", ("done", 42))
    assert _wait_msgs(orig) == [("done", 42)]


def test_eventfd_wakeup_cuts_wait_short(pair):
    _, rust = pair
    ev = EventWakeup()
    rust.register_event_for_poll(ev)
    threading.Thread(target=lambda: (time.sleep(0.05),
                                     os.eventfd_write(ev.fd, 1))).start()
    t0 = time.time()
    rust.wait_for_work(timeout_ms=2000)
    assert time.time() - t0 < 0.5, "wake must beat the poll timeout"


def test_readiness_poll_never_drops_or_reorders(pair):
    orig, rust = pair
    orig.send("rust", "first")
    assert any(rust.poll_for_messages(timeout_ms=10) for _ in range(500))
    orig.send("rust", "second")
    time.sleep(0.1)
    assert rust.get_all_new_messages() == ["first", "second"]


@pytest.mark.parametrize("receiver", ["rust", "orig"])
def test_blocking_receive_waits_for_a_message(pair, receiver):
    """get_all_new_messages(blocking=True) waits instead of returning [] —
    on both communicators (the pyzmq one had the same latent bug)."""
    orig, rust = pair
    dst, src = (rust, orig) if receiver == "rust" else (orig, rust)
    threading.Thread(target=lambda: (time.sleep(0.1),
                                     src.send(receiver, "late"))).start()
    assert dst.get_all_new_messages(blocking=True) == ["late"]


def test_make_communicator_flag(monkeypatch, tmp_path):
    from mstar.communication.communicator import make_communicator

    def make(value):
        monkeypatch.setenv("MSTAR_RUST_ZMQ", value)
        return make_communicator(
            f"m_{value}", push_ids=[], ipc_socket_path_prefix=str(tmp_path))

    assert isinstance(make("0"), ZMQCommunicator)
    assert isinstance(make("1"), RustZMQCommunicator)
    assert isinstance(make("AUTO"), RustZMQCommunicator)  # extension installed
    with pytest.raises(ValueError):
        make("yes")
