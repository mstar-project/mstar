"""A ChunkSink takes its requests' chunks once per message pass.

The Rust-frontend bridge relays every chunk the message thread drained in a
pass as one message, instead of each request's handler waking per chunk.
The handler still owns completion, errors and the timeout."""
import asyncio
import sys

import pytest
from api_server_helpers import make_test_api_server

from mstar.api_server import entrypoint as ep
from mstar.api_server.entrypoint import PendingRequest
from mstar.api_server.request_types import ResultChunk
from mstar.communication.event import EventWakeup
from mstar.profile.format import RequestProfile


class _PreprocessWorker:
    """Hands the message thread a fixed list of chunks on its next pass."""

    def __init__(self):
        self.chunks: list[ResultChunk] = []
        self.aborted: list[str] = []
        self.results_wakeup = EventWakeup()

    def get_profile_updates(self):
        return []

    def get_result_chunks(self):
        out, self.chunks = self.chunks, []
        return out

    def has_pending_tensors(self, rid):
        return False

    def received_final_chunks(self, rid, counts):
        return True

    def finished_reading(self, rid, drained=True):
        pass

    def abort_request(self, rid):
        pass

    def forward_abort(self, rid):
        self.aborted.append(rid)


class _Comm:
    """Ends ``_process_messages`` after one pass, at its end-of-pass wait."""

    def __init__(self):
        self.server = None

    def get_all_new_messages(self):
        return []

    def register_event_for_poll(self, event):
        pass

    def poll_for_messages(self, timeout_ms=20):
        self.server.running = False
        return False


class _Sink:
    def __init__(self, loop):
        self.loop = loop
        self.calls: list[tuple[list, list[bytes]]] = []

    def relay(self, tags, chunks):
        self.calls.append((list(tags), [c.data for c in chunks]))


def _server():
    comm = _Comm()
    comm.server = make_test_api_server(preprocess_worker=_PreprocessWorker(), communicator=comm)
    return comm.server


def _register(server, key):
    req = PendingRequest(
        streaming=True, input_modalities=["text"], output_modalities=["text"], profile=None,
    )
    req.profile = RequestProfile(rid=key)
    server.pending_requests[key] = req
    server._live[key] = key
    server._client_ids[key] = key
    return req


def _chunk(key, data, modality="text", **metadata):
    return ResultChunk(request_id=key, modality=modality, data=data, metadata=metadata)


def _one_pass(server, monkeypatch):
    """``_process_messages`` for one pass: its end-of-pass wait stops it. An
    error in the pass is raised here instead of logged and the pass retried."""
    def reraise(*_args, **_kwargs):
        raise sys.exception()

    with monkeypatch.context() as m:
        m.setattr(ep.logger, "exception", reraise)
        server.running = True
        server._process_messages()


def test_a_pass_hands_the_sink_every_request_s_chunks_in_one_call(monkeypatch):
    async def run():
        server = _server()
        sink = _Sink(asyncio.get_running_loop())
        a, b, c = (_register(server, k) for k in ("A", "B", "C"))
        server.relay_chunks_to("A", sink, 1)
        server.relay_chunks_to("B", sink, 2)
        c.loop, c.wake = asyncio.get_running_loop(), asyncio.Event()
        a.loop, a.wake = asyncio.get_running_loop(), asyncio.Event()
        server.preprocess_worker.chunks = [
            _chunk("A", b"a1"), _chunk("B", b"b1"), _chunk("C", b"c1"), _chunk("A", b"a2"),
        ]
        _one_pass(server, monkeypatch)
        await asyncio.sleep(0)
        assert sink.calls == [([1, 1, 2], [b"a1", b"a2", b"b1"])]
        assert a.chunks == [] and b.chunks == []
        # a sink request's handler is not woken per chunk; anyone else's is
        assert not a.wake.is_set() and c.wake.is_set()
        assert [ch.data for ch in c.chunks] == [b"c1"]

    asyncio.run(run())


def test_an_error_chunk_and_what_follows_stay_for_the_handler(monkeypatch):
    async def run():
        server = _server()
        sink = _Sink(asyncio.get_running_loop())
        _register(server, "A")
        server.relay_chunks_to("A", sink, 1)
        server.preprocess_worker.chunks = [
            _chunk("A", b"a1"), _chunk("A", b"boom", modality="error", status=400),
        ]
        handler = server.iter_result_chunks("A")
        _one_pass(server, monkeypatch)
        await asyncio.sleep(0)  # the sink's callback
        rest = [c async for c in handler]
        assert sink.calls == [([1], [b"a1"])]
        assert [(c.modality, c.data) for c in rest] == [("error", b"boom")]
        assert "A" not in server.pending_requests

    asyncio.run(run())


def test_the_handler_finishes_a_relayed_request_with_nothing_left(monkeypatch):
    async def run():
        server = _server()
        sink = _Sink(asyncio.get_running_loop())
        req = _register(server, "A")
        server.relay_chunks_to("A", sink, 1)
        handler = server.iter_result_chunks("A")
        first = asyncio.ensure_future(handler.__anext__())
        await asyncio.sleep(0)  # the handler is waiting
        server.preprocess_worker.chunks = [_chunk("A", b"a1"), _chunk("A", b"a2")]
        _one_pass(server, monkeypatch)
        await asyncio.sleep(0)
        assert sink.calls == [([1, 1], [b"a1", b"a2"])]
        # completion, as _prune_recently_completed signals it
        with server.request_lock:
            req.event.set()
            req.notify()
        try:
            leftover = [await first] + [c async for c in handler]
        except StopAsyncIteration:
            leftover = []
        assert leftover == []
        assert "A" not in server.pending_requests

    asyncio.run(run())


def test_chunks_the_handler_already_took_are_not_relayed_again(monkeypatch):
    async def run():
        server = _server()
        sink = _Sink(asyncio.get_running_loop())
        req = _register(server, "A")
        server.relay_chunks_to("A", sink, 1)
        server.preprocess_worker.chunks = [_chunk("A", b"a1")]
        _one_pass(server, monkeypatch)
        # the handler's backstop wake runs before the sink's callback
        batches = server.iter_result_chunk_batches("A")
        got = await batches.__anext__()
        assert [c.data for c in got] == [b"a1"]
        await asyncio.sleep(0)
        assert sink.calls == []
        assert req.chunks == []
        await batches.aclose()

    asyncio.run(run())


def test_a_relay_that_raises_fails_its_requests(monkeypatch):
    """The relay runs as a loop callback, whose exception asyncio only logs,
    and its chunks are already off the requests: they fail in-band, and the
    engine requests are aborted, rather than finishing with output missing."""
    async def run():
        server = _server()
        sink = _Sink(asyncio.get_running_loop())

        def broken(tags, chunks):
            raise RuntimeError("bridge gone")

        sink.relay = broken
        _register(server, "A")
        server.relay_chunks_to("A", sink, 1)
        server.preprocess_worker.chunks = [_chunk("A", b"a1")]
        handler = server.iter_result_chunks("A")
        _one_pass(server, monkeypatch)
        await asyncio.sleep(0)  # the sink's callback
        rest = [c async for c in handler]
        assert [(c.modality, c.metadata.get("status")) for c in rest] == [("error", 500)]
        assert server.preprocess_worker.aborted == ["A"]

    asyncio.run(run())


def test_an_aborted_request_releases_its_waiting_handler_at_once():
    """The Rust frontend aborts a dropped submission from the bridge, not from
    the handler: the abort wakes the handler, rather than leaving it to its
    next backstop wake."""
    async def run():
        server = _server()
        _register(server, "A")
        handler = server.iter_result_chunks("A")
        waiting = asyncio.ensure_future(handler.__anext__())
        await asyncio.sleep(0)  # the handler is waiting
        server.abort_submission("A")
        for _ in range(10):  # a few loop turns, well inside the backstop
            await asyncio.sleep(0)
            if waiting.done():
                break
        assert waiting.done()
        with pytest.raises(StopAsyncIteration):
            waiting.result()

    asyncio.run(run())
