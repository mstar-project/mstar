"""Unit tests for mminf.api_server.media_io (numpy only — no server deps)."""

import base64
import struct

import pytest

np = pytest.importorskip("numpy")

from mminf.api_server import media_io  # noqa: E402


def test_pcm16_to_wav_header_rate_and_data():
    pcm = np.array([0, 16000, -16000, 32767], dtype="<i2").tobytes()
    wav = media_io.pcm16_to_wav_bytes(pcm, 24000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert struct.unpack("<I", wav[24:28])[0] == 24000  # sample rate field
    assert wav[44:] == pcm  # 44-byte header, then the int16 samples pass through unchanged


def test_container_pcm_passthrough():
    pcm = np.array([1, -1, 100], dtype="<i2").tobytes()
    out, mime = media_io.pcm16_to_container(pcm, 24000, "pcm")
    assert mime == "audio/pcm" and out == pcm  # already PCM_16, returned as-is


def test_container_wav_default():
    pcm = np.array([0, 123, -123], dtype="<i2").tobytes()
    out, mime = media_io.pcm16_to_container(pcm, 16000, "wav")
    assert mime == "audio/wav" and out[:4] == b"RIFF" and out[44:] == pcm


def test_data_url_roundtrip(tmp_path):
    raw = b"\x89PNG hello world"
    url = "data:image/png;base64," + base64.b64encode(raw).decode()
    modality, path = media_io.save_data_url(url, tmp_path)
    assert modality == "image"
    assert path.endswith(".png")
    with open(path, "rb") as f:
        assert f.read() == raw


def test_save_base64_audio(tmp_path):
    raw = b"RIFFfake"
    modality, path = media_io.save_base64(base64.b64encode(raw).decode(), "wav", "audio", tmp_path)
    assert modality == "audio" and path.endswith(".wav")


def test_png_data_url():
    assert media_io.png_to_data_url(b"x").startswith("data:image/png;base64,")


def test_wav_stream_header():
    h = media_io.wav_stream_header(24000)
    assert h[:4] == b"RIFF" and h[8:12] == b"WAVE" and len(h) == 44


def test_modality_from_mime():
    assert media_io.modality_from_mime("image/png") == "image"
    assert media_io.modality_from_mime("audio/wav") == "audio"
    assert media_io.modality_from_mime("text/plain") == "unknown"
