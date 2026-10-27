"""``OutboxCommunicator``: the API server's preprocess thread sends without
blocking, queueing what a full peer cannot take, in order per peer. A message
that cannot be sent must fail on its own and never stop the queue behind it."""
from __future__ import annotations

import queue
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from mstar.api_server.data_worker import PreprocessWorkerThread  # noqa: E402
from mstar.api_server.request_types import PreprocessInput  # noqa: E402
from mstar.communication.codec import default_codec  # noqa: E402
from mstar.communication.communicator import OutboxCommunicator  # noqa: E402
from mstar.utils.ipc_format import ConductorMessageType  # noqa: E402


class _Peer:
    """A transport whose peers are full for ``full[peer]`` more sends, with the
    real wire codec; what it delivers is decoded back for the test."""

    def __init__(self, full: dict | None = None):
        self.codec = default_codec()
        self.full = dict(full or {})
        self.delivered: list = []
        self.fail: set[bytes] = set()

    def try_send_bytes(self, entity, data):
        if data in self.fail:
            raise RuntimeError("socket error")
        if self.full.get(entity, 0) > 0:
            self.full[entity] -= 1
            return False
        self.delivered.append((entity, self.codec.decode(data)))
        return True

    def get_all_new_messages(self):
        return []


def test_the_outbox_queues_what_a_full_peer_cannot_take_and_keeps_order():
    peer = _Peer(full={"conductor": 2})
    out = OutboxCommunicator(peer)
    out.send("conductor", 1)
    out.send("worker_0", "a")          # another peer is not held up
    out.send("conductor", 2)           # queued behind 1, even if the peer could take it
    assert peer.delivered == [("worker_0", "a")]
    assert out.flush() is True          # the peer is still full once
    assert out.flush() is False
    assert peer.delivered == [("worker_0", "a"), ("conductor", 1), ("conductor", 2)]


def test_an_unencodable_message_fails_its_sender_even_behind_a_backlog():
    peer = _Peer(full={"conductor": 1})
    out = OutboxCommunicator(peer)
    out.send("conductor", {"n": 1})                 # queued: the peer is full
    with pytest.raises(OverflowError):
        out.send("conductor", {"seed": 2**64})      # the codec refuses it, here
    out.send("conductor", {"n": 2})
    assert out.flush() is False
    assert [msg for _, msg in peer.delivered] == [{"n": 1}, {"n": 2}]


def test_a_queued_message_that_fails_to_send_is_dropped_not_retried():
    peer = _Peer(full={"conductor": 1})
    out = OutboxCommunicator(peer)
    out.send("conductor", "first")
    out.send("conductor", "poisoned")
    out.send("conductor", "last")
    peer.fail.add(peer.codec.encode("poisoned"))
    assert out.flush() is False
    assert [msg for _, msg in peer.delivered] == ["first", "last"]


class _Model:
    def process_prompt(self, *args, **kwargs):
        return {"text_inputs": [torch.tensor([1, 2, 3])]}


class _TensorManager:
    def store_and_return_tensor_info(self, request_id, tensors, **kwargs):
        return {name: [] for name in tensors}

    def register_for_send(self, *args, **kwargs):
        pass

    def set_persist(self, *args, **kwargs):
        pass

    def get_ready_tensors(self):
        return {}

    def take_failed_reads(self):
        return []

    def has_inflight_reads(self, request_id):
        return False

    def force_cleanup_request(self, request_id):
        pass


def _request(request_id, **model_kwargs):
    return PreprocessInput(
        request_id=request_id, text="hi", file_paths=None, input_modalities=["text"],
        output_modalities=["text"], model_kwargs=model_kwargs,
    )


def _thread(peer, tensor_manager=None):
    queues = {name: queue.Queue() for name in (
        "in_queue", "result_tensor_queue", "out_queue", "profile_queue", "cleanup_request_queue",
        "abort_request_queue", "reads_done_queue", "discard_tensor_queue",
    )}
    out = OutboxCommunicator(peer)
    wt = PreprocessWorkerThread(
        **queues, stop_event=threading.Event(), communicator=out,
        tensor_manager=tensor_manager or _TensorManager(), model=_Model(),
    )
    return wt, out


def test_a_request_the_codec_refuses_fails_alone_after_a_backlog():
    """The conductor is full while a request the codec refuses (a client's
    out-of-range seed) waits behind another: once it drains, that request
    fails with an error to its client, and the requests around it still reach
    the conductor."""
    peer = _Peer(full={"conductor": 10**9})
    wt, out = _thread(peer)
    wt.in_queue.put(_request("first"))
    wt.in_queue.put(_request("bad", seed=2**64))
    wt.in_queue.put(_request("good"))
    runner = threading.Thread(target=wt.run, daemon=True)
    runner.start()
    try:
        deadline = time.time() + 10
        while not any(out._outbox.values()) and time.time() < deadline:
            time.sleep(0.01)
        peer.full.clear()                           # the conductor catches up
        error = wt.out_queue.get(timeout=10)
        while len(peer.delivered) < 2 and time.time() < deadline:
            time.sleep(0.01)
    finally:
        wt.stop_event.set()
        runner.join(timeout=10)

    assert error.request_id == "bad" and error.modality == "error"
    new = [msg.body.request_id for _, msg in peer.delivered
           if msg.message_type == ConductorMessageType.NEW_REQUEST]
    assert new == ["first", "good"]


def test_no_new_input_is_taken_while_a_peer_cannot_take_what_was_sent():
    """A full conductor holds back new requests, as a blocking send did,
    rather than growing the outbox; they go once it drains."""
    peer = _Peer(full={"conductor": 10**9})
    wt, out = _thread(peer)
    for request_id in ("a", "b", "c"):
        wt.in_queue.put(_request(request_id))
    runner = threading.Thread(target=wt.run, daemon=True)
    runner.start()
    try:
        deadline = time.time() + 10
        while sum(map(len, out._outbox.values())) < 1 and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.2)                          # many passes while the peer stays full
        held = (sum(map(len, out._outbox.values())), wt.in_queue.qsize())
        peer.full.clear()
        while len(peer.delivered) < 3 and time.time() < deadline:
            time.sleep(0.01)
    finally:
        wt.stop_event.set()
        runner.join(timeout=10)

    assert held == (1, 2)                        # "a" queued; "b" and "c" not even preprocessed
    assert [msg.body.request_id for _, msg in peer.delivered] == ["a", "b", "c"]


def test_a_release_that_cannot_be_sent_does_not_lose_the_request_taken_with_it():
    """An abort the codec refuses is released in the same pass as a new
    request, just before it is preprocessed: that abort fails alone, and the
    request still reaches the conductor."""
    class _AbortsOnce(_TensorManager):
        def get_ready_tensors(self):
            # polled after the pass's releases and before it takes new input
            if not wt.in_queue.empty() and not self.done:
                self.done = True
                wt.abort_request_queue.put("\udc80")   # no message can carry it
            return {}

    peer = _Peer()
    tm = _AbortsOnce()
    tm.done = False
    wt, _out = _thread(peer, tm)
    wt.in_queue.put(_request("next"))
    runner = threading.Thread(target=wt.run, daemon=True)
    runner.start()
    try:
        deadline = time.time() + 10
        while not peer.delivered and time.time() < deadline:
            time.sleep(0.01)
    finally:
        wt.stop_event.set()
        runner.join(timeout=10)

    assert tm.done and wt.abort_request_queue.empty()
    assert [(msg.message_type, msg.body.request_id) for _, msg in peer.delivered] == [
        (ConductorMessageType.NEW_REQUEST, "next"),
    ]


def _factory(monkeypatch, choice, *, has_try_send):
    """``make_communicator`` over a fake ``mstar_rust`` build, with each
    transport's constructor returning its name."""
    import sys
    from types import SimpleNamespace

    # imported before the fake goes in: this module binds mstar_rust on import
    from mstar.communication import communicator, rust_communicator

    class _Zmq:
        def send(self, peer_id, data):
            pass

    if has_try_send:
        _Zmq.try_send = lambda self, peer_id, data: True
    version = communicator.EXPECTED_MSTAR_RUST_VERSION if has_try_send else "0.1.0"
    monkeypatch.setitem(sys.modules, "mstar_rust", SimpleNamespace(ZmqCommunicator=_Zmq, __version__=version))
    monkeypatch.setattr(rust_communicator, "RustZMQCommunicator", lambda *args, **kwargs: "rust")
    monkeypatch.setattr(communicator, "ZMQCommunicator", lambda *args, **kwargs: "pyzmq")
    monkeypatch.setenv("MSTAR_RUST_ZMQ", choice)
    return communicator.make_communicator


@pytest.mark.parametrize("choice", ["AUTO", "1"])
def test_a_rust_build_without_try_send_still_carries_the_other_senders(monkeypatch, choice):
    """Only the outbox needs try_send: the conductor's and the workers'
    communicators stay Rust, which a worker's Rust graph runtime requires."""
    make = _factory(monkeypatch, choice, has_try_send=False)
    assert make("conductor", []) == "rust"


def test_a_rust_build_without_try_send_puts_the_outbox_on_pyzmq_under_auto(monkeypatch, caplog):
    make = _factory(monkeypatch, "AUTO", has_try_send=False)
    with caplog.at_level("WARNING"):
        assert make("api_server_preprocess_worker", [], nonblocking_send=True) == "pyzmq"
    assert "try_send" in caplog.text


def test_a_rust_build_without_try_send_fails_the_outbox_when_rust_is_required(monkeypatch):
    make = _factory(monkeypatch, "1", has_try_send=False)
    with pytest.raises(RuntimeError, match="rebuild"):
        make("api_server_preprocess_worker", [], nonblocking_send=True)


@pytest.mark.parametrize("choice", ["AUTO", "1"])
def test_a_rust_build_with_try_send_carries_the_outbox(monkeypatch, choice):
    make = _factory(monkeypatch, choice, has_try_send=True)
    assert make("api_server_preprocess_worker", [], nonblocking_send=True) == "rust"
