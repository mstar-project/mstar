"""Request media must not name files on the server, not even under upload_dir:
only data URLs, uploads, base64 and, when the server allows it, http(s) URLs."""
import base64

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
    req = SpeechRequest(input="hi", ref_audio=str(secret), ref_text="hi")
    with pytest.raises(ValueError, match=REFUSED):
        adapter().speech_to_request(req, uploads)


def test_omnivoice_fetches_a_url_only_when_the_server_allows(dirs, monkeypatch):
    uploads, _ = dirs
    fetched = []
    monkeypatch.setattr(media_io, "save_remote_url", lambda url, d: fetched.append(url) or ("audio", "/x.wav"))
    monkeypatch.delenv("MSTAR_ALLOW_REMOTE", raising=False)
    req = SpeechRequest(input="hi", ref_audio="http://169.254.169.254/latest/meta-data", ref_text="hi")
    with pytest.raises(ValueError, match="Remote media fetch is disabled"):
        adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
    assert fetched == []


def test_omnivoice_accepts_bare_base64_reference(dirs):
    uploads, _ = dirs
    req = SpeechRequest(input="hi", ref_audio=base64.b64encode(b"RIFF....WAVE").decode(), ref_text="hi")
    (path,) = adapters.OmniVoiceAdapter().speech_to_request(req, uploads).file_paths["audio"]
    assert path.startswith(str(uploads))


@pytest.mark.parametrize("adapter", [adapters.Cosmos3Adapter, adapters.Wan22Adapter])
def test_video_generation_conditioning_refuses_server_paths(adapter, dirs):
    uploads, secret = dirs
    with pytest.raises(ValueError, match=REFUSED):
        adapter().video_to_request(VideoGenerationRequest(prompt="x", image=str(secret)), uploads)
