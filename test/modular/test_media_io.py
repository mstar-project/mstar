"""Unit tests for mstar.api_server.media_io (numpy only — no server deps)."""

import base64
import struct

import pytest

np = pytest.importorskip("numpy")

from mstar.api_server import media_io  # noqa: E402


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


def test_check_audio_format_defaults_and_refusals():
    assert media_io.check_audio_format(None) == "wav"
    assert media_io.check_audio_format("PCM", stream=True) == "pcm"
    for fmt, kwargs in [("xyz", {}), ("aac", {}), ("mp3", {"stream": True}), ("flac", {"stream": True})]:
        with pytest.raises(ValueError, match=f"response_format '{fmt}' cannot be"):
            media_io.check_audio_format(fmt, **kwargs)


def test_compressed_containers_encode_for_real():
    sf = pytest.importorskip("soundfile")
    import io

    pcm = (np.sin(np.arange(24000) / 10) * 8000).astype("<i2").tobytes()
    for fmt, mime in [("mp3", "audio/mpeg"), ("flac", "audio/flac"), ("opus", "audio/ogg")]:
        out, got = media_io.pcm16_to_container(pcm, 24000, fmt)
        assert got == mime and out[:4] != b"RIFF"
        audio, rate = sf.read(io.BytesIO(out))
        assert rate == 24000 and abs(len(audio) - 24000) < 2400  # decodes back to ~1 s
    # Opus has fixed rates; 44.1 kHz (e.g. Zonos2) is refused, not encoded as WAV
    with pytest.raises(ValueError, match="'opus' cannot be produced"):
        media_io.pcm16_to_container(pcm, 44100, "opus")
    assert media_io.pcm16_to_container(pcm, 44100, "mp3")[1] == "audio/mpeg"


def test_data_url_roundtrip(tmp_path):
    raw = b"\x89PNG hello world"
    url = "data:image/png;base64," + base64.b64encode(raw).decode()
    modality, path = media_io.save_data_url(url, tmp_path)
    assert modality == "image"
    assert path.endswith(".png")
    with open(path, "rb") as f:
        assert f.read() == raw


@pytest.mark.parametrize("raw", [b"R", b"RI", b"\x89PNG hello world"])
@pytest.mark.parametrize("padded", [True, False])
@pytest.mark.parametrize("wrap", [0, 4, 76])
@pytest.mark.parametrize("trailing", ["", "\n", "\r\n"])
def test_data_url_accepts_unpadded_and_whitespace(raw, padded, wrap, trailing, tmp_path):
    encoded = base64.b64encode(raw).decode()
    if not padded:
        encoded = encoded.rstrip("=")
    if wrap:
        encoded = "\n".join(encoded[index : index + wrap] for index in range(0, len(encoded), wrap))
    modality, path = media_io.save_data_url(
        "data:image/png;base64," + encoded + trailing,
        tmp_path,
    )
    assert modality == "image" and path.endswith(".png")
    with open(path, "rb") as f:
        assert f.read() == raw


def test_save_base64_audio(tmp_path):
    raw = b"RIFFfake"
    modality, path = media_io.save_base64(base64.b64encode(raw).decode(), "wav", "audio", tmp_path)
    assert modality == "audio" and path.endswith(".wav")


@pytest.mark.parametrize("raw", [b"R", b"RI"])
def test_save_base64_audio_accepts_unpadded(raw, tmp_path):
    # Some OpenAI clients strip the trailing "=" from input_audio.data; a
    # 1- or 2-byte payload is the case that needs 2 or 1 pad chars back.
    encoded = base64.b64encode(raw).decode().rstrip("=")
    modality, path = media_io.save_base64(encoded, "wav", "audio", tmp_path)
    assert modality == "audio" and path.endswith(".wav")
    with open(path, "rb") as f:
        assert f.read() == raw


@pytest.mark.parametrize("raw", [b"R", b"RI", b"RIFFfake"])
@pytest.mark.parametrize("padded", [True, False])
@pytest.mark.parametrize("wrap", [0, 4, 76])
@pytest.mark.parametrize("trailing", ["", "\n", "\r\n"])
def test_save_base64_audio_accepts_whitespace(raw, padded, wrap, trailing, tmp_path):
    # b64decode ignores whitespace, so counting it when restoring padding
    # computes the padding from the wrong length. A 1-byte payload with one
    # trailing newline is the smallest case that breaks: the raw length is a
    # multiple of 4 plus 3, so one "=" is appended to a body that needs two.
    # MIME encoders wrap at 76 columns, so this arrives from ordinary clients.
    encoded = base64.b64encode(raw).decode()
    if not padded:
        encoded = encoded.rstrip("=")
    if wrap:
        encoded = "\n".join(encoded[index : index + wrap] for index in range(0, len(encoded), wrap))
    modality, path = media_io.save_base64(encoded + trailing, "wav", "audio", tmp_path)
    assert modality == "audio" and path.endswith(".wav")
    with open(path, "rb") as f:
        assert f.read() == raw


def test_png_data_url():
    assert media_io.png_to_data_url(b"x").startswith("data:image/png;base64,")


def test_wav_stream_header():
    h = media_io.wav_stream_header(24000)
    assert h[:4] == b"RIFF" and h[8:12] == b"WAVE" and len(h) == 44


def test_modality_from_mime():
    assert media_io.modality_from_mime("image/png") == "image"
    assert media_io.modality_from_mime("audio/wav") == "audio"
    assert media_io.modality_from_mime("text/plain") == "unknown"


def test_remove_uploads_leaves_a_file_outside_the_upload_dir(tmp_path):
    upload_dir, elsewhere = tmp_path / "uploads", tmp_path / "voices"
    upload_dir.mkdir()
    elsewhere.mkdir()
    saved, preset = upload_dir / "clip.wav", elsewhere / "preset.wav"
    saved.write_bytes(b"RIFF")
    preset.write_bytes(b"RIFF")

    media_io.remove_uploads({"audio": [str(saved), str(preset)]}, upload_dir)

    assert not saved.exists() and preset.exists()


# G.711 reference pairs from CPython's audioop (ulaw2lin / alaw2lin / lin2ulaw /
# lin2alaw), the same tables as ITU-T G.711 and Sun's g711.c
_ULAW_DECODE = {0x00: -32124, 0x01: -31100, 0x0F: -16764, 0x10: -15996, 0x3F: -1980, 0x70: -120,
                0x7E: -8, 0x7F: 0, 0x80: 32124, 0x8F: 16764, 0xC0: 1884, 0xF0: 120, 0xFE: 8, 0xFF: 0}
_ALAW_DECODE = {0x00: -5504, 0x0F: -6784, 0x10: -2752, 0x2A: -32256, 0x54: -24, 0x55: -8, 0x7F: -848,
                0x80: 5504, 0xAA: 32256, 0xD4: 24, 0xD5: 8, 0xFF: 848}
_PCM = [-32768, -32124, -8159, -100, -1, 0, 1, 7, 8, 100, 1000, 8158, 32000, 32767]
_ULAW_ENCODE = [0, 0, 31, 114, 126, 255, 255, 254, 254, 242, 206, 159, 128, 128]
_ALAW_ENCODE = [42, 42, 10, 83, 85, 213, 213, 213, 213, 211, 250, 138, 170, 170]


@pytest.mark.parametrize("law, decode, encode", [
    ("audio/pcmu", _ULAW_DECODE, _ULAW_ENCODE),
    ("audio/pcma", _ALAW_DECODE, _ALAW_ENCODE),
])
def test_g711_matches_the_reference_tables(law, decode, encode):
    codes = bytes(decode)
    assert media_io.g711_decode(codes, law).tolist() == list(decode.values())
    assert list(media_io.g711_encode(np.array(_PCM, dtype="<i2"), law)) == encode
    # every code decodes to a level that encodes back to a code of the same level
    levels = media_io.g711_decode(bytes(range(256)), law)
    assert (media_io.g711_decode(media_io.g711_encode(levels, law), law) == levels).all()
