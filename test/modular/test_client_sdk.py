"""Unit tests for the mstar Python SDK parsing/encoding (no live server)."""

import base64
import io
import json
from unittest import mock

import pytest

pytest.importorskip("requests")
np = pytest.importorskip("numpy")

from mstar.client import AudioBuffer, MStarClient  # noqa: E402
from mstar.client.media import parse_ndjson_line  # noqa: E402


def test_parse_ndjson_line():
    line = json.dumps({"modality": "text", "data": base64.b64encode(b"hi").decode(), "metadata": {}})
    p = parse_ndjson_line(line)
    assert p["modality"] == "text" and p["bytes"] == b"hi"
    assert parse_ndjson_line("") is None
    assert parse_ndjson_line("not json") is None


def test_parse_result_groups_modalities():
    pcm = np.array([0, 1000, -1000, 32767], dtype="<i2").tobytes()  # 4 int16 samples
    payload = {
        "request_id": "r1",
        "outputs": {
            "text": [
                {"data": base64.b64encode(b"hello ").decode()},
                {"data": base64.b64encode(b"world").decode()},
            ],
            "image": [{"data": base64.b64encode(b"\x89PNG").decode()}],
            "audio": [{"data": base64.b64encode(pcm).decode(), "metadata": {"sample_rate": 24000}}],
        },
    }
    res = MStarClient._parse_result(payload)
    assert res.text == "hello world"
    assert res.images == [b"\x89PNG"]
    assert res.audio is not None and res.audio.sample_rate == 24000 and len(res.audio) == 4
    assert res.audio.pcm == pcm  # raw int16 preserved
    assert len(res.raw) == 4


def test_to_event_typing():
    pcm = np.array([123, -123], dtype="<i2").tobytes()
    assert MStarClient._to_event({"modality": "text", "bytes": b"hi", "metadata": {}}).text == "hi"
    audio = MStarClient._to_event({"modality": "audio", "bytes": pcm, "metadata": {"sample_rate": 16000}})
    assert audio.sample_rate == 16000
    assert MStarClient._to_event({"modality": "image", "bytes": b"P", "metadata": {}}).data == b"P"


def test_coerce_and_build_files():
    c = MStarClient("http://x")
    assert c._coerce_file("images", 0, b"\x89PNG") == ("image_0.png", b"\x89PNG")
    assert c._build_files(None, b"\x00\x01", None) == [("files", ("audio_0.wav", b"\x00\x01"))]
    assert c._build_files(("seed.png", b"\x89PNG"), None, None) == [
        ("files", ("seed.png", b"\x89PNG")),
    ]


def test_stream_without_content_type_charset():
    """Content-Type without a charset must still yield events (issue #163)."""
    requests = pytest.importorskip("requests")
    line = json.dumps({"modality": "text", "data": base64.b64encode(b"hi").decode(), "metadata": {}})

    resp = requests.Response()
    resp.status_code = 200
    resp.headers["Content-Type"] = "application/x-ndjson"  # no charset
    resp.raw = io.BytesIO((line + "\n").encode())
    assert resp.encoding is None

    c = MStarClient("http://x")
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = resp
    with (
        mock.patch.object(c._session, "post", return_value=ctx),
        mock.patch.object(resp, "iter_lines", wraps=resp.iter_lines) as iter_lines,
    ):
        events = list(c._stream("http://x/generate", {}, None))
    assert [e.text for e in events] == ["hi"]
    iter_lines.assert_called_once_with(
        chunk_size=1024 * 1024,
        decode_unicode=True,
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "modality": "error",
                "data": base64.b64encode(b"capture failed").decode(),
                "metadata": {"status": 500},
            },
            r"Server stream failed \(status 500\): capture failed",
        ),
        ({"error": "bridge failed"}, "Server stream failed: bridge failed"),
    ],
)
def test_stream_raises_in_band_server_errors(payload, message):
    requests = pytest.importorskip("requests")
    resp = requests.Response()
    resp.status_code = 200
    resp.headers["Content-Type"] = "application/x-ndjson"
    resp.raw = io.BytesIO((json.dumps(payload) + "\n").encode())

    client = MStarClient("http://x")
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = resp
    with mock.patch.object(client._session, "post", return_value=ctx):
        with pytest.raises(RuntimeError, match=message):
            list(client._stream("http://x/generate", {}, None))


@pytest.mark.parametrize("modality", ["action", "scalar", "tensor", "video"])
def test_to_event_preserves_text_compatible_modality_fallback(modality):
    event = MStarClient._to_event({
        "modality": modality,
        "bytes": b"[1, 2, 3]",
        "metadata": {"sequence": 4},
    })
    assert event.text == "[1, 2, 3]"
    assert event.metadata == {"sequence": 4}


def test_audiobuffer_wav_bytes():
    pcm = np.array([0, 16000, -16000], dtype="<i2").tobytes()
    wav = AudioBuffer(pcm, 24000).wav_bytes()
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE" and wav[44:] == pcm


def _fake_response(content_type: str, body: bytes, extra_headers=None):
    requests = pytest.importorskip("requests")
    resp = requests.Response()
    resp.status_code = 200
    resp.headers["Content-Type"] = content_type
    for key, value in (extra_headers or {}).items():
        resp.headers[key] = value
    resp.raw = io.BytesIO(body)
    return resp


def _stream_with(client, resp):
    ctx = mock.MagicMock()
    ctx.__enter__.return_value = resp
    with mock.patch.object(client._session, "post", return_value=ctx) as post:
        events = list(client._stream("http://x/generate", {}, None))
    return events, post


def test_stream_takes_the_binary_path_when_the_server_says_so():
    from mstar.api_server.entrypoint import _chunk_to_binary_frame
    from mstar.api_server.request_types import ResultChunk
    from mstar.client.media import BINARY_STREAM_MEDIA_TYPE

    payload = bytes(range(256)) * 8
    header, body = _chunk_to_binary_frame(
        ResultChunk(request_id="r", modality="text", data=payload, metadata={})
    )
    resp = _fake_response(BINARY_STREAM_MEDIA_TYPE, header + body)

    events, post = _stream_with(MStarClient("http://x", prefer_binary=True), resp)

    assert [e.text for e in events] == [payload.decode("utf-8", "replace")]
    headers = post.call_args.kwargs["headers"]
    assert BINARY_STREAM_MEDIA_TYPE in headers["Accept"]
    assert headers["Accept-Encoding"] == "identity"


def test_stream_falls_back_to_ndjson_when_the_server_ignores_accept():
    """An older server or the Rust frontend answers x-ndjson regardless of what
    was asked for; the client keys off the response, so nothing breaks."""
    line = json.dumps({"modality": "text", "data": base64.b64encode(b"hi").decode(), "metadata": {}})
    resp = _fake_response("application/x-ndjson", (line + "\n").encode())

    with mock.patch.object(resp, "iter_lines", wraps=resp.iter_lines) as iter_lines:
        events, _ = _stream_with(MStarClient("http://x", prefer_binary=True), resp)

    assert [e.text for e in events] == ["hi"]
    iter_lines.assert_called_once_with(chunk_size=1024 * 1024, decode_unicode=True)


def test_stream_sends_no_negotiation_headers_when_binary_is_disabled():
    line = json.dumps({"modality": "text", "data": base64.b64encode(b"hi").decode(), "metadata": {}})
    resp = _fake_response("application/x-ndjson", (line + "\n").encode())

    events, post = _stream_with(MStarClient("http://x"), resp)

    assert [e.text for e in events] == ["hi"]
    assert post.call_args.kwargs["headers"] == {}


def test_stream_names_content_encoding_as_the_cause_on_a_compressed_binary_body():
    """``resp.raw.read`` bypasses urllib3's decoder, so a gzipped body would
    otherwise surface as an unreadable frame header."""
    from mstar.client.media import BINARY_STREAM_MEDIA_TYPE

    resp = _fake_response(
        BINARY_STREAM_MEDIA_TYPE, b"\x1f\x8b garbage", {"Content-Encoding": "gzip"}
    )
    with pytest.raises(RuntimeError, match="Content-Encoding 'gzip'"):
        _stream_with(MStarClient("http://x", prefer_binary=True), resp)
