"""``try_send_bytes`` on the control-mesh transport: it never blocks, says
False with nothing sent while the peer is at its high-water mark, and loses or
reorders nothing. The API server's preprocess thread sends everything through
it (``OutboxCommunicator``). Skipped unless the ``mstar_rust`` extension is
installed."""
import tempfile
import time

import pytest

pytest.importorskip("mstar_rust")

from mstar.communication.rust_communicator import RustZMQCommunicator


def test_try_send_bytes_refuses_a_full_peer_and_loses_nothing():
    prefix = tempfile.mkdtemp(prefix="mstar_ts_", dir="/tmp")
    dst = RustZMQCommunicator("dst", push_ids=[], ipc_socket_path_prefix=prefix)  # reads only when asked
    src = RustZMQCommunicator("src", push_ids=["dst"], ipc_socket_path_prefix=prefix)
    encode = src.codec.encode

    # the connection comes up asynchronously; until it does the peer can't take anything
    deadline = time.time() + 5
    while not src.try_send_bytes("dst", encode(0)):
        assert time.time() < deadline, "never connected"
        time.sleep(0.01)
    sent = 1
    while src.try_send_bytes("dst", encode(sent)):    # fill the peer's queues
        sent += 1
        assert sent < 1_000_000, "a peer that reads nothing never filled up"

    received: list = []
    deadline = time.time() + 10
    while len(received) < sent and time.time() < deadline:
        received.extend(dst.get_all_new_messages())
    assert received == list(range(sent))               # each exactly once, in order

    assert src.try_send_bytes("dst", encode("again"))  # drained: it takes more
    deadline = time.time() + 5
    late: list = []
    while not late and time.time() < deadline:
        late = dst.get_all_new_messages()
    assert late == ["again"]
