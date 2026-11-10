"""The SDK's reader for binary-framed result streams (no live server, no GPU).

The NDJSON form base64s the payload so it can live inside a JSON string, which
at 720p costs ~85 ms of CPU per chunk against a 66.67 ms budget. Binary framing
sends a one-line JSON header carrying ``nbytes`` and then the payload untouched.
These tests pin the framing and the reader's tolerance for short socket reads;
mstar-server's side of the negotiation is tested in test/rust/.
"""

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from stream_frames import binary_frame

from mstar.api_server.entrypoint import BINARY_STREAM_MEDIA_TYPE
from mstar.api_server.request_types import ResultChunk
from mstar.client.client import MStarClient
from mstar.client.media import BINARY_STREAM_MEDIA_TYPE as CLIENT_BINARY_MEDIA_TYPE
from mstar.client.media import iter_binary_frames


def _chunks():
    return [
        ResultChunk(
            request_id="r",
            modality="video_frame",
            # Spans every byte value, including b"\n" and bytes that are not
            # valid UTF-8 — precisely what base64 existed to work around.
            data=bytes(range(256)) * 64,
            metadata={"frame_index": i, "width": 8, "height": 8},
        )
        for i in range(3)
    ]


def _wire(chunks):
    return b"".join(b"".join(binary_frame(c)) for c in chunks)


class _Dribble:
    """A reader that returns at most ``limit`` bytes, like a real socket."""

    def __init__(self, data: bytes, limit: int = 7):
        self._data = data
        self._pos = 0
        self._limit = limit

    def read(self, size: int) -> bytes:
        end = self._pos + min(size, self._limit)
        out = self._data[self._pos:end]
        self._pos += len(out)
        return out


def test_media_type_constants_agree():
    """The server and the SDK duplicate this string; drift would silently
    disable negotiation and leave 720p slow with no failing test."""
    assert BINARY_STREAM_MEDIA_TYPE == CLIENT_BINARY_MEDIA_TYPE


def test_binary_frames_round_trip():
    chunks = _chunks()
    frames = list(iter_binary_frames(io.BytesIO(_wire(chunks))))

    assert len(frames) == len(chunks)
    for frame, chunk in zip(frames, chunks, strict=True):
        assert frame["bytes"] == chunk.data
        assert frame["modality"] == chunk.modality
        assert frame["metadata"] == chunk.metadata


def test_binary_frames_yield_bytes_not_bytearray():
    """``VideoFrameChunk.__post_init__`` rejects ``bytearray`` outright."""
    (frame,) = iter_binary_frames(io.BytesIO(_wire(_chunks()[:1])))
    assert type(frame["bytes"]) is bytes


def test_binary_frames_survive_short_reads():
    chunks = _chunks()
    frames = list(iter_binary_frames(_Dribble(_wire(chunks))))
    assert [f["bytes"] for f in frames] == [c.data for c in chunks]


def test_binary_frames_arrive_incrementally_over_a_real_stream():
    """PR-review regression: the header scan used ``raw.read(read_size)``,
    which blocks until ``read_size`` bytes or EOF. Small frames sent slowly
    over a real connection therefore all arrived at once when the server
    closed the stream, instead of as each one landed. ``BytesIO`` and
    ``_Dribble`` can't expose this (neither one blocks like a real socket),
    so this drives ``iter_binary_frames`` over an actual local HTTP server
    with a chunked body.
    """
    n_frames = 6
    gap = 0.05
    all_sent = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def do_GET(self):
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(n_frames):
                header = json.dumps(
                    {"modality": "audio", "nbytes": 4, "metadata": {"i": i}}
                ).encode() + b"\n"
                body = header + b"abcd"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(body), body))
                self.wfile.flush()
                if i < n_frames - 1:
                    time.sleep(gap)
            all_sent.set()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def log_message(self, *args):
            pass

    class _Server(ThreadingHTTPServer):
        daemon_threads = True

    server = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        resp = requests.get(
            f"http://127.0.0.1:{server.server_address[1]}/", stream=True, timeout=5
        )
        try:
            frames = iter_binary_frames(resp.raw)
            first = next(frames)
            # The bug buffered every frame until the connection closed; this
            # must fire while the server is still mid-stream.
            assert not all_sent.is_set()
            assert first["metadata"] == {"i": 0}
            assert [f["metadata"]["i"] for f in frames] == list(range(1, n_frames))
        finally:
            resp.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_binary_frames_reject_truncated_payload():
    wire = _wire(_chunks())
    with pytest.raises(RuntimeError, match="ended 10 bytes short"):
        list(iter_binary_frames(io.BytesIO(wire[:-10])))


def test_binary_frames_reject_truncated_header():
    header, _ = binary_frame(_chunks()[0])
    with pytest.raises(RuntimeError, match="ended mid-header"):
        list(iter_binary_frames(io.BytesIO(header[:20])))


def test_binary_frames_end_cleanly_on_empty_stream():
    assert list(iter_binary_frames(io.BytesIO(b""))) == []


def test_binary_frame_header_is_always_one_line():
    """The reader splits the header on the first newline, so metadata that
    contains one must not be able to forge a frame boundary."""
    chunk = ResultChunk(
        request_id="r",
        modality="text",
        data=b"payload\nwith\nnewlines",
        metadata={"note": "line\nbreak", "unicode": "café é"},
    )
    header, payload = binary_frame(chunk)

    assert header.count(b"\n") == 1 and header.endswith(b"\n")
    (frame,) = iter_binary_frames(io.BytesIO(header + payload))
    assert frame["bytes"] == chunk.data
    assert frame["metadata"] == chunk.metadata


def test_binary_frames_raise_on_top_level_error_envelope():
    """Mirrors ``parse_ndjson_line``: a header with no ``modality`` but an
    ``error`` key is a failure envelope, not a zero-length chunk."""
    wire = json.dumps({"error": "bridge failed"}).encode() + b"\n"
    with pytest.raises(RuntimeError, match="Server stream failed: bridge failed"):
        list(iter_binary_frames(io.BytesIO(wire)))


def test_binary_error_chunk_reaches_the_sdk_error_path():
    chunk = ResultChunk(
        request_id="r", modality="error", data=b"capture failed", metadata={"status": 500}
    )
    (frame,) = iter_binary_frames(io.BytesIO(_wire([chunk])))
    with pytest.raises(RuntimeError, match=r"Server stream failed \(status 500\): capture failed"):
        MStarClient._to_event(frame)
