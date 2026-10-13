"""Request media must not name files on the server, not even under upload_dir:
only data URLs, uploads, base64 and, when the server allows it, http(s) URLs."""
import base64
import io

import numpy as np
import pytest

from mstar.api_server import media_io
from mstar.api_server.openai import adapters
from mstar.api_server.openai.protocol import ChatCompletionRequest, SpeechRequest, VideoGenerationRequest

REFUSED = "server file paths are refused"


@pytest.fixture
def dirs(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"\x89PNG\r\n\x1a\n")
    (uploads / "escape.png").symlink_to(secret)
    (uploads / "loop.png").symlink_to(uploads / "loop.png")
    (uploads / "mine.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return uploads, secret


def escapes(uploads, secret):
    return [
        str(secret),                                  # absolute
        str(uploads / ".." / "secret.png"),           # traversal out of upload_dir
        "../secret.png",                              # relative to the server's cwd
        str(uploads / "escape.png"),                  # symlink inside upload_dir pointing out
        f"file://{secret}",                           # file URL
        str(uploads.parent / "missing.png"),          # refused the same way whether or not it exists
        str(uploads / "loop.png"),                    # symlink loop
        str(uploads / "mine.png"),                    # even a file under upload_dir
        "mine.png",
    ]


def test_resolve_media_ref_refuses_every_server_path(dirs, monkeypatch):
    uploads, secret = dirs
    monkeypatch.chdir(uploads)
    for ref in escapes(uploads, secret):
        with pytest.raises(ValueError, match=REFUSED):
            media_io.resolve_media_ref(ref, uploads)
    data_url = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    modality, path = media_io.resolve_media_ref(data_url, uploads)
    assert modality == "image" and path.startswith(str(uploads))


def test_chat_media_parts_refuse_server_paths(dirs):
    uploads, secret = dirs
    for part in ({"type": "image_url", "image_url": {"url": str(secret)}},
                 {"type": "audio_url", "audio_url": {"url": str(secret)}},
                 {"type": "video_url", "video_url": {"url": str(secret)}}):
        req = ChatCompletionRequest(model="qwen3_omni", messages=[{"role": "user", "content": [part]}])
        with pytest.raises(ValueError, match=REFUSED):
            adapters.Qwen3OmniAdapter().chat_to_request(req, uploads)


@pytest.mark.parametrize("adapter", [adapters.Qwen3TTSAdapter, adapters.OmniVoiceAdapter])
def test_voice_clone_references_refuse_server_paths(adapter, dirs):
    uploads, secret = dirs
    before = sorted(uploads.iterdir())
    # the suffix-less paths are valid base64 ("//..." even decodes to an MPEG frame sync),
    # and so are bytes that only start like a WAV; none of them is audio the model can load
    fake_wav = base64.b64encode(b"RIFF....WAVEfmt " + bytes(64)).decode()
    for ref in (str(secret), "/etc/passwd", "/etc/shadow", "//etc/passwd", "//etc/shadow", "secret", "///x",
                f"file://{secret}", fake_wav):
        req = SpeechRequest(input="hi", ref_audio=ref, ref_text="hi")
        with pytest.raises(ValueError, match=REFUSED):
            adapter().speech_to_request(req, uploads)
    assert sorted(uploads.iterdir()) == before


def test_omnivoice_fetches_a_url_only_when_the_server_allows(dirs, monkeypatch):
    uploads, _ = dirs
    fetched = []
    monkeypatch.setattr(media_io, "save_remote_url", lambda url, d: fetched.append(url) or ("audio", "/x.wav"))
    monkeypatch.delenv("MSTAR_ALLOW_REMOTE", raising=False)
    req = SpeechRequest(input="hi", ref_audio="http://169.254.169.254/latest/meta-data", ref_text="hi")
    with pytest.raises(ValueError, match="Remote media fetch is disabled"):
        adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
    assert fetched == []


@pytest.mark.parametrize("adapter", [
    adapters.BagelAdapter, adapters.Qwen3OmniAdapter, adapters.Qwen3_5Adapter, adapters.Cosmos3EdgeAdapter,
])
def test_chat_media_parts_fetch_a_url_only_when_the_server_allows(adapter, dirs, monkeypatch):
    uploads, _ = dirs
    fetched = []
    monkeypatch.setattr(media_io, "save_remote_url", lambda url, d: fetched.append(url) or ("image", "/x.png"))
    monkeypatch.delenv("MSTAR_ALLOW_REMOTE", raising=False)
    url = "http://169.254.169.254/latest/meta-data"
    for kind in ("image_url", "audio_url", "video_url"):
        part = {"type": kind, kind: {"url": url}}
        req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": [part]}])
        with pytest.raises(ValueError, match="Remote media fetch is disabled"):
            adapter().chat_to_request(req, uploads)
    assert fetched == []
    monkeypatch.setenv("MSTAR_ALLOW_REMOTE", "1")
    content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "describe"}]
    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": content}])
    adapter().chat_to_request(req, uploads)
    assert fetched == [url]


# (format, endian, subtype) as soundfile writes them; RIFX is WAV with endian BIG
FFMPEG_READS = [
    ("WAV", "FILE", None), ("WAV", "BIG", None), ("NIST", "FILE", None), ("NIST", "BIG", None),
    ("AIFF", "FILE", None), ("AU", "FILE", None), ("CAF", "FILE", None), ("FLAC", "FILE", None),
    ("MP3", "FILE", None), ("OGG", "FILE", "VORBIS"), ("OGG", "FILE", "OPUS"), ("RF64", "FILE", None),
    ("W64", "FILE", None), ("WAVEX", "FILE", None), ("IRCAM", "FILE", None), ("VOC", "FILE", None),
    ("PVF", "FILE", None), ("AVR", "FILE", None), ("PAF", "FILE", None), ("SVX", "FILE", None),
]
SOUNDFILE_ONLY = [
    ("HTK", "FILE", None), ("MAT4", "FILE", None), ("MAT5", "FILE", None), ("MPC2K", "FILE", None),
    ("WVE", "FILE", None), ("XI", "FILE", None),
]


def _clip(fmt, endian, subtype):
    sf = pytest.importorskip("soundfile")
    if fmt not in sf.available_formats():
        pytest.skip(f"this libsndfile cannot write {fmt}")
    rate = 48000 if subtype == "OPUS" else 24000
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate)
    buf = io.BytesIO()
    sf.write(buf, tone.astype(np.float32), rate, format=fmt, subtype=subtype, endian=endian)
    return base64.b64encode(buf.getvalue()).decode()


def _audio_decoder():
    try:
        from torchcodec.decoders import AudioDecoder

        AudioDecoder(base64.b64decode(_clip("WAV", "FILE", None)))
    except Exception as exc:  # noqa: BLE001 — torchcodec raises RuntimeError without FFmpeg
        pytest.skip(f"torchcodec cannot load here: {exc}")
    return AudioDecoder


@pytest.mark.parametrize("clip", FFMPEG_READS + SOUNDFILE_ONLY)
def test_qwen3_tts_accepts_bare_base64_that_soundfile_reads(clip, dirs):
    uploads, _ = dirs
    req = SpeechRequest(input="hi", ref_audio=_clip(*clip), ref_text="hi")
    (path,) = adapters.Qwen3TTSAdapter().speech_to_request(req, uploads).file_paths["audio"]
    import soundfile

    assert path.startswith(str(uploads)) and soundfile.read(path)[0].size


@pytest.mark.parametrize("clip", FFMPEG_READS)
def test_omnivoice_accepts_bare_base64_that_torchcodec_reads(clip, dirs):
    uploads, _ = dirs
    decoder = _audio_decoder()
    req = SpeechRequest(input="hi", ref_audio=_clip(*clip), ref_text="hi")
    (path,) = adapters.OmniVoiceAdapter().speech_to_request(req, uploads).file_paths["audio"]
    assert path.startswith(str(uploads))
    assert decoder(path, sample_rate=24000, num_channels=1).get_all_samples().data.numel()


@pytest.mark.parametrize("clip", SOUNDFILE_ONLY)
def test_omnivoice_refuses_audio_its_decoder_cannot_read(clip, dirs):
    uploads, _ = dirs
    _audio_decoder()
    req = SpeechRequest(input="hi", ref_audio=_clip(*clip), ref_text="hi")
    with pytest.raises(ValueError, match=REFUSED):
        adapters.OmniVoiceAdapter().speech_to_request(req, uploads)


@pytest.mark.parametrize("adapter", [adapters.Cosmos3Adapter, adapters.Wan22Adapter])
def test_video_generation_conditioning_refuses_server_paths(adapter, dirs):
    uploads, secret = dirs
    with pytest.raises(ValueError, match=REFUSED):
        adapter().video_to_request(VideoGenerationRequest(prompt="x", image=str(secret)), uploads)
