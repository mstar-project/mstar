"""The nemotron_duplex standalone Realtime server's audio framing, over a
fake model (no weights, no GPU)."""

import asyncio
import base64

import numpy as np

from mstar.model.nemotron_duplex.realtime_api import RealtimeConnection


class _Stream:
    def __init__(self):
        self.fed = bytearray()

    def append_audio(self, pcm: bytes):
        assert len(pcm) % 2 == 0 and pcm, "the stream takes whole int16 samples"
        self.fed.extend(pcm)
        return []

    def flush(self):
        return []


class _Model:
    def __init__(self):
        self.streams: list[_Stream] = []

    def create_stream(self, **_):
        self.streams.append(_Stream())
        return self.streams[-1]


def _run(events):
    model, sent = _Model(), []

    async def send(event):
        sent.append(event)

    async def go():
        conn = RealtimeConnection(model, send)
        await conn.start()
        for event in events:
            await conn.handle(event)

    asyncio.run(go())
    return model, sent


def _append(raw: bytes) -> dict:
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(raw).decode("ascii")}


def test_odd_length_appends_are_joined_into_whole_samples():
    raw = (np.sin(np.arange(3200) * 0.05) * 8000).astype("<i2").tobytes()
    events = [_append(raw[i:i + 1001]) for i in range(0, len(raw), 1001)]
    model, sent = _run(events + [{"type": "input_audio_buffer.commit"}, _append(b"\x01")])
    assert bytes(model.streams[-1].fed) == raw
    assert [e["type"] for e in sent] == ["session.created"]


def test_clear_drops_the_odd_byte_and_bad_base64_is_an_error():
    model, sent = _run([
        _append(b"\x01\x02\x03"),
        {"type": "input_audio_buffer.clear"},
        _append(b"\x04\x05"),
        {"type": "input_audio_buffer.append", "audio": "%%%"},
    ])
    assert bytes(model.streams[-1].fed) == b"\x04\x05"
    assert sent[-1]["type"] == "error" and sent[-1]["error"]["message"] == "audio is not base64"


def test_cancel_and_session_update_keep_the_odd_byte():
    """A barge-in (response.cancel) or a prompt change opens a fresh stream,
    but the client's byte stream goes on, so the pending byte is kept."""
    raw = (np.sin(np.arange(1600) * 0.05) * 8000).astype("<i2").tobytes()
    model, _ = _run([
        _append(raw[:1001]),
        {"type": "response.cancel"},
        {"type": "session.update", "session": {"voice": "Other"}},
        _append(raw[1001:]),
    ])
    assert bytes(model.streams[-1].fed) == raw[1000:]


def test_line_wrapped_base64_is_accepted_and_a_non_string_is_an_error():
    raw = (np.sin(np.arange(1600) * 0.05) * 8000).astype("<i2").tobytes()
    model, sent = _run([
        {"type": "input_audio_buffer.append", "audio": base64.encodebytes(raw).decode("ascii")},
        {"type": "input_audio_buffer.append", "audio": 12345},
    ])
    assert bytes(model.streams[-1].fed) == raw
    assert sent[-1]["type"] == "error" and sent[-1]["error"]["message"] == "audio is not base64"


def test_null_audio_is_an_invalid_audio_error():
    raw = (np.sin(np.arange(1600) * 0.05) * 8000).astype("<i2").tobytes()
    model, sent = _run([
        _append(raw[:1001]),
        {"type": "input_audio_buffer.append", "audio": None},
        _append(raw[1001:]),
    ])
    assert bytes(model.streams[-1].fed) == raw
    assert [e["type"] for e in sent] == ["session.created", "error"]
    assert sent[-1]["error"] == {"message": "audio is not base64", "code": "invalid_audio"}
