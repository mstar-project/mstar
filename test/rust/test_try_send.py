"""``try_send_bytes`` on both control-mesh transports (pyzmq and the Rust
binding): it never blocks, says False with nothing sent while the peer is at
its high-water mark, and loses or reorders nothing. The API server's
preprocess thread sends everything through it (``OutboxCommunicator``). The
Rust case is skipped unless the ``mstar_rust`` extension is installed."""
import tempfile
import time

import pytest

from mstar.communication.communicator import ZMQCommunicator


def _sender(kind, prefix):
    if kind == "pyzmq":
        return ZMQCommunicator("src", push_ids=["dst"], ipc_socket_path_prefix=prefix)
    pytest.importorskip("mstar_rust")
    from mstar.communication.rust_communicator import RustZMQCommunicator
    return RustZMQCommunicator("src", push_ids=["dst"], ipc_socket_path_prefix=prefix)


@pytest.mark.parametrize("kind", ["pyzmq", "rust"])
def test_try_send_bytes_refuses_a_full_peer_and_loses_nothing(kind):
    prefix = tempfile.mkdtemp(prefix="mstar_ts_", dir="/tmp")
    dst = ZMQCommunicator("dst", push_ids=[], ipc_socket_path_prefix=prefix)  # reads only when asked
    src = _sender(kind, prefix)
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
