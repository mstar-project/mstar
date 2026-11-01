"""RustZMQCommunicator: data-only wire, eventfd wakeup, lossless readiness
polls. Skipped unless the ``mstar_rust`` extension (built from ``rust/``) is
installed."""
import os
import tempfile
import threading
import time

import pytest

pytest.importorskip("mstar_rust")

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
    a = RustZMQCommunicator("a", push_ids=["b"], ipc_socket_path_prefix=prefix)
    b = RustZMQCommunicator("b", push_ids=["a"], ipc_socket_path_prefix=prefix)
    return a, b


def test_unregistered_payload_both_directions(pair):
    a, b = pair
    payload = {"op": "execute", "rids": [1, 2, 3], "nested": {"f": 1.5}}
    a.send("b", payload)
    assert _wait_msgs(b) == [payload]
    b.send("a", ("done", 42))
    assert _wait_msgs(a) == [("done", 42)]


def test_eventfd_wakeup_cuts_wait_short(pair):
    _, b = pair
    ev = EventWakeup()
    b.register_event_for_poll(ev)
    threading.Thread(target=lambda: (time.sleep(0.05),
                                     os.eventfd_write(ev.fd, 1))).start()
    t0 = time.time()
    b.wait_for_work(timeout_ms=2000)
    assert time.time() - t0 < 0.5, "wake must beat the poll timeout"


def test_readiness_poll_never_drops_or_reorders(pair):
    a, b = pair
    a.send("b", "first")
    assert any(b.poll_for_messages(timeout_ms=10) for _ in range(500))
    a.send("b", "second")
    time.sleep(0.1)
    assert b.get_all_new_messages() == ["first", "second"]


def test_blocking_receive_waits_for_a_message(pair):
    """get_all_new_messages(blocking=True) waits instead of returning []."""
    a, b = pair
    threading.Thread(target=lambda: (time.sleep(0.1),
                                     a.send("b", "late"))).start()
    assert b.get_all_new_messages(blocking=True) == ["late"]


def test_make_communicator_flag(monkeypatch, tmp_path):
    from mstar.communication.communicator import ZMQCommunicator, make_communicator

    def make(value):
        monkeypatch.setenv("MSTAR_RUST_ZMQ", value)
        return make_communicator(
            f"m_{value}", push_ids=[], ipc_socket_path_prefix=str(tmp_path))

    assert isinstance(make("0"), ZMQCommunicator)
    assert isinstance(make("1"), RustZMQCommunicator)
    assert isinstance(make("AUTO"), RustZMQCommunicator)  # extension installed
    with pytest.raises(ValueError):
        make("yes")
    # default is AUTO: with the extension installed, unset -> Rust
    monkeypatch.delenv("MSTAR_RUST_ZMQ", raising=False)
    assert isinstance(
        make_communicator("m_default", push_ids=[],
                          ipc_socket_path_prefix=str(tmp_path)),
        RustZMQCommunicator)


def test_blocking_receive_timeout_bounds_the_wait(pair):
    _, b = pair
    t0 = time.monotonic()
    assert b.get_all_new_messages(blocking=True, timeout_s=0.15) == []
    assert 0.1 < time.monotonic() - t0 < 2.0


def test_poll_for_messages_leaves_the_message_queued(tmp_path):
    """True from poll_for_messages leaves the message queued for
    get_all_new_messages; with nothing sent it times out with False."""
    a = RustZMQCommunicator("pfm_a", ["pfm_b"], ipc_socket_path_prefix=str(tmp_path) + "/")
    b = RustZMQCommunicator("pfm_b", ["pfm_a"], ipc_socket_path_prefix=str(tmp_path) + "/")
    assert b.poll_for_messages(timeout_ms=10) is False
    a.send("pfm_b", {"hello": "rs"})
    assert b.poll_for_messages(timeout_ms=2000) is True
    assert b.get_all_new_messages() == [{"hello": "rs"}]
