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
    with mock.patch.object(c._session, "post", return_value=ctx):
        events = list(c._stream("http://x/generate", {}, None))
    assert [e.text for e in events] == ["hi"]


def test_audiobuffer_wav_bytes():
    pcm = np.array([0, 16000, -16000], dtype="<i2").tobytes()
    wav = AudioBuffer(pcm, 24000).wav_bytes()
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE" and wav[44:] == pcm
