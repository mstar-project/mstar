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
TOO_LONG = "ref_audio must be 60 s or shorter"
WAV_BYTES = b"RIFF....WAVEfmt " + bytes(64)


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


@pytest.mark.parametrize("adapter", [adapters.Qwen3TTSAdapter, adapters.OmniVoiceAdapter, adapters.ChatterboxAdapter])
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


def _clip(fmt, endian, subtype, rate=None, seconds=0.5):
    sf = pytest.importorskip("soundfile")
    if fmt not in sf.available_formats():
        pytest.skip(f"this libsndfile cannot write {fmt}")
    rate = rate or (48000 if subtype == "OPUS" else 24000)
    n = max(1, int(rate * seconds))
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / rate)
    buf = io.BytesIO()
    try:
        sf.write(buf, tone.astype(np.float32), rate, format=fmt, subtype=subtype, endian=endian)
    except Exception as exc:  # noqa: BLE001 — e.g. Opus has no 44.1 kHz mode
        pytest.skip(f"this libsndfile cannot write {fmt} at {rate} Hz: {exc}")
    return base64.b64encode(buf.getvalue()).decode()


def _model_media():
    """The decoders OmniVoice's ``load_audio`` uses: torchcodec, else PyAV then libsndfile."""
    from mstar.model.components import media_io as model_media

    if not model_media.uses_torchcodec():
        pytest.importorskip("av")
    return model_media


def _audio_decoder():
    """torchcodec's AudioDecoder, where OmniVoice decodes with it."""
    try:
        from torchcodec.decoders import AudioDecoder

        AudioDecoder(base64.b64decode(_clip("WAV", "FILE", None)))
    except Exception as exc:  # noqa: BLE001 — torchcodec raises RuntimeError without FFmpeg
        pytest.skip(f"torchcodec cannot load here: {exc}")
    if not _model_media().uses_torchcodec():
        pytest.skip("torchcodec loads but cannot decode here, so OmniVoice uses PyAV")
    return AudioDecoder


def _without_torchcodec(monkeypatch):
    """OmniVoice's decoders where torchcodec does not work (ROCm, no FFmpeg)."""
    model_media = _model_media()
    monkeypatch.setattr(
        model_media, "_decoders",
        lambda: model_media._Decoders(model_media._FallbackAudio(), model_media._PyAVVideo()),
    )
    return model_media


@pytest.mark.parametrize("clip", FFMPEG_READS + SOUNDFILE_ONLY)
def test_qwen3_tts_accepts_bare_base64_that_soundfile_reads(clip, dirs):
    uploads, _ = dirs
    req = SpeechRequest(input="hi", ref_audio=_clip(*clip), ref_text="hi")
    (path,) = adapters.Qwen3TTSAdapter().speech_to_request(req, uploads).file_paths["audio"]
    import soundfile

    assert path.startswith(str(uploads)) and soundfile.read(path)[0].size


@pytest.mark.parametrize("clip", FFMPEG_READS)
def test_omnivoice_accepts_bare_base64_that_its_decoder_reads(clip, dirs):
    uploads, _ = dirs
    model_media = _model_media()
    req = SpeechRequest(input="hi", ref_audio=_clip(*clip), ref_text="hi")
    (path,) = adapters.OmniVoiceAdapter().speech_to_request(req, uploads).file_paths["audio"]
    assert path.startswith(str(uploads))
    assert model_media.decode_audio(path, sample_rate=24000, num_channels=1).data.numel()


@pytest.mark.parametrize("clip", SOUNDFILE_ONLY)
@pytest.mark.parametrize("adapter", [adapters.OmniVoiceAdapter, adapters.ChatterboxAdapter])
def test_libsndfile_first_voice_clones_accept_audio_only_libsndfile_reads(adapter, clip, dirs):
    """Their loaders try libsndfile before FFmpeg, as Model.load_audio does."""
    uploads, _ = dirs
    req = SpeechRequest(input="hi", ref_audio=_clip(*clip), ref_text="hi")
    (path,) = adapter().speech_to_request(req, uploads).file_paths["audio"]
    import soundfile

    assert path.startswith(str(uploads)) and soundfile.read(path)[0].size


@pytest.mark.parametrize("decodes", [True, False])
def test_omnivoice_refuses_audio_whose_header_opens_but_samples_do_not_decode(decodes, dirs, monkeypatch):
    """Some FFmpeg builds open a MAT4 or MAT5 header and then fail on the first
    frame. This stub decoder does the same, so the check runs without FFmpeg."""
    decoded = _stub_decoder(monkeypatch, rate=24000, decodes=decodes)
    uploads, _ = dirs
    before = set(uploads.iterdir())
    req = SpeechRequest(input="hi", ref_audio=base64.b64encode(b"MATLAB 5.0 MAT-file").decode(), ref_text="hi")
    if decodes:
        (path,) = adapters.OmniVoiceAdapter().speech_to_request(req, uploads).file_paths["audio"]
        assert path.startswith(str(uploads))
    else:
        with pytest.raises(ValueError, match=REFUSED):
            adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
        assert set(uploads.iterdir()) == before
    # decoded size must not scale with the channel count, rate or length the clip declares
    assert decoded and all(
        d["num_channels"] == 1 and d["sample_rate"] and d["stop"] == media_io.MAX_REF_AUDIO_SECONDS + 1.0
        for d in decoded
    )


def _stub_torchcodec(monkeypatch, decoder_cls):
    import sys
    import types

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.AudioDecoder = decoder_cls
    package = types.ModuleType("torchcodec")
    package.decoders = decoders
    monkeypatch.setitem(sys.modules, "torchcodec", package)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    # and OmniVoice's decoders take it, as where torchcodec works
    from mstar.model.components import media_io as model_media

    monkeypatch.setattr(
        model_media, "_decoders",
        lambda: model_media._Decoders(model_media._TorchcodecAudio(), model_media._TorchcodecVideo()),
    )


@pytest.mark.parametrize("rate", [None, 0, 1, 7999])
def test_omnivoice_refuses_a_clip_below_the_rate_floor_before_decoding(rate, dirs, monkeypatch):
    """The rate comes from the header; nothing is decoded, so a 1 Hz clip cannot
    expand 16,000x on its way to the probe's 16 kHz. Stub decoder, no FFmpeg."""
    decoded = _stub_decoder(monkeypatch, rate)
    uploads, _ = dirs
    before = set(uploads.iterdir())
    req = SpeechRequest(input="hi", ref_audio=base64.b64encode(b"RIFF....WAVEfmt " + bytes(64)).decode(), ref_text="hi")
    with pytest.raises(ValueError, match=REFUSED):
        adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
    assert decoded == [] and set(uploads.iterdir()) == before


@pytest.mark.parametrize("rate", [1, 4000, 7999])
def test_omnivoice_refuses_a_low_rate_wav_without_decoding_it(rate, dirs, monkeypatch):
    """Real torchcodec: 64 KB of 1 Hz PCM would decode to 2 GB at 16 kHz."""
    decoder = _audio_decoder()
    import torchcodec.decoders

    decodes = []

    class Spy(decoder):
        def get_samples_played_in_range(self, *args, **kwargs):
            decodes.append(True)
            return super().get_samples_played_in_range(*args, **kwargs)

    monkeypatch.setattr(torchcodec.decoders, "AudioDecoder", Spy)
    uploads, _ = dirs
    before = set(uploads.iterdir())
    seconds = 32000 / rate  # 32,000 samples, 64 KB of PCM16
    req = SpeechRequest(input="hi", ref_audio=_clip("WAV", "FILE", None, rate=rate, seconds=seconds), ref_text="hi")
    with pytest.raises(ValueError, match=REFUSED):
        adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
    assert decodes == [] and set(uploads.iterdir()) == before


@pytest.mark.parametrize("rate", [1, 7999])
def test_qwen3_tts_refuses_a_clip_below_the_rate_floor(rate, dirs):
    uploads, _ = dirs
    before = set(uploads.iterdir())
    clip = _clip("WAV", "FILE", None, rate=rate, seconds=32000 / rate)
    req = SpeechRequest(input="hi", ref_audio=clip, ref_text="hi")
    with pytest.raises(ValueError, match=REFUSED):
        adapters.Qwen3TTSAdapter().speech_to_request(req, uploads)
    assert set(uploads.iterdir()) == before


UNDECODABLE = "ref_audio must be audio the model can decode"


def _stub_decoder(monkeypatch, rate, decodes=True, header_seconds=0.5, seconds=0.5):
    """A torchcodec stand-in whose header reports ``rate`` and ``header_seconds``
    and whose decode yields ``seconds`` of audio, cut at the requested stop.
    Returns the list each sample decode appends its arguments to."""
    import types

    decoded = []

    class AudioDecoder:
        def __init__(self, source, **kwargs):
            open(source, "rb").close()
            self.kwargs = kwargs
            self.metadata = types.SimpleNamespace(sample_rate=rate, duration_seconds_from_header=header_seconds)

        def get_samples_played_in_range(self, start_seconds=0.0, stop_seconds=None):
            decoded.append({**self.kwargs, "stop": stop_seconds})
            if not decodes:
                raise RuntimeError("decode_av_frame failed")
            played = seconds if stop_seconds is None else min(seconds, stop_seconds)
            return types.SimpleNamespace(pts_seconds=start_seconds, duration_seconds=played)

    _stub_torchcodec(monkeypatch, AudioDecoder)
    return decoded


@pytest.mark.parametrize("rate, decodes, accepted", [(24000, True, True), (1, True, False), (24000, False, False)])
def test_omnivoice_checks_a_data_url_like_bare_base64(rate, decodes, accepted, dirs, monkeypatch):
    """A data URL gets the rate floor and the decode check too. Stub decoder, no FFmpeg."""
    decoded = _stub_decoder(monkeypatch, rate, decodes)
    uploads, _ = dirs
    before = set(uploads.iterdir())
    ref = "data:audio/wav;base64," + base64.b64encode(b"RIFF....WAVEfmt " + bytes(64)).decode()
    req = SpeechRequest(input="hi", ref_audio=ref, ref_text="hi")
    if accepted:
        (path,) = adapters.OmniVoiceAdapter().speech_to_request(req, uploads).file_paths["audio"]
        assert path.startswith(str(uploads))
        return
    with pytest.raises(ValueError, match=UNDECODABLE):
        adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
    assert set(uploads.iterdir()) == before
    assert len(decoded) == (0 if rate < media_io.MIN_REF_AUDIO_SAMPLE_RATE else 1)


def test_omnivoice_checks_an_allowed_url_like_bare_base64(dirs, monkeypatch):
    decoded = _stub_decoder(monkeypatch, rate=1)
    uploads, _ = dirs
    before = set(uploads.iterdir())
    monkeypatch.setenv("MSTAR_ALLOW_REMOTE", "1")
    monkeypatch.setattr(media_io, "save_remote_url",
                        lambda url, d: media_io._save_bytes(b"RIFF....WAVEfmt " + bytes(64), "audio/wav", d))
    req = SpeechRequest(input="hi", ref_audio="https://example.com/voice.wav", ref_text="hi")
    with pytest.raises(ValueError, match=UNDECODABLE):
        adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
    assert decoded == [] and set(uploads.iterdir()) == before


@pytest.mark.parametrize("rate", [1, 24000])
@pytest.mark.parametrize("adapter", [adapters.OmniVoiceAdapter, adapters.Qwen3TTSAdapter])
def test_a_wav_data_url_is_held_to_the_rate_floor(adapter, rate, dirs):
    """Real decoders: a 1 Hz clip used to reach the data worker's 24 kHz decode."""
    if adapter is adapters.OmniVoiceAdapter:
        _audio_decoder()
    uploads, _ = dirs
    before = set(uploads.iterdir())
    clip = _clip("WAV", "FILE", None, rate=rate, seconds=32000 / rate)
    req = SpeechRequest(input="hi", ref_audio="data:audio/wav;base64," + clip, ref_text="hi")
    if rate >= media_io.MIN_REF_AUDIO_SAMPLE_RATE:
        (path,) = adapter().speech_to_request(req, uploads).file_paths["audio"]
        assert path.startswith(str(uploads))
        return
    with pytest.raises(ValueError, match=UNDECODABLE):
        adapter().speech_to_request(req, uploads)
    assert set(uploads.iterdir()) == before


@pytest.mark.parametrize("adapter", [adapters.OmniVoiceAdapter, adapters.ChatterboxAdapter, adapters.Qwen3TTSAdapter])
def test_a_mat5_data_url_is_checked_by_the_model_decoder(adapter, dirs):
    """torchcodec cannot decode MAT5 samples, and FFmpeg's header for it claims
    over a minute; libsndfile, which all three loaders try first, reads it."""
    uploads, _ = dirs
    req = SpeechRequest(input="hi", ref_audio="data:audio/wav;base64," + _clip("MAT5", "FILE", None), ref_text="hi")
    (path,) = adapter().speech_to_request(req, uploads).file_paths["audio"]
    assert path.startswith(str(uploads))


@pytest.mark.parametrize("form", ["base64", "data URL"])
def test_omnivoice_refuses_a_clip_whose_header_is_too_long_without_decoding(form, dirs, monkeypatch):
    """A long clip is a 400 from its header alone. Stub decoder, no FFmpeg."""
    decoded = _stub_decoder(monkeypatch, rate=48000, header_seconds=media_io.MAX_REF_AUDIO_SECONDS + 1)
    uploads, _ = dirs
    before = set(uploads.iterdir())
    ref = base64.b64encode(WAV_BYTES).decode()
    if form == "data URL":
        ref = "data:audio/wav;base64," + ref
    req = SpeechRequest(input="hi", ref_audio=ref, ref_text="hi")
    with pytest.raises(media_io.RefAudioTooLongError, match=TOO_LONG):
        adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
    assert decoded == [] and set(uploads.iterdir()) == before


@pytest.mark.parametrize("header_seconds", [None, 5.0])
@pytest.mark.parametrize("seconds, accepted", [(5.0, True), (3600.0, False)])
def test_omnivoice_caps_the_decode_when_the_header_hides_the_length(
    header_seconds, seconds, accepted, dirs, monkeypatch
):
    """No header duration, or one that understates the clip: the decode stops just
    past the cap, so it never holds more than about 61 s of samples."""
    decoded = _stub_decoder(monkeypatch, rate=48000, header_seconds=header_seconds, seconds=seconds)
    uploads, _ = dirs
    before = set(uploads.iterdir())
    req = SpeechRequest(input="hi", ref_audio=base64.b64encode(WAV_BYTES).decode(), ref_text="hi")
    if accepted:
        (path,) = adapters.OmniVoiceAdapter().speech_to_request(req, uploads).file_paths["audio"]
        assert path.startswith(str(uploads))
    else:
        with pytest.raises(media_io.RefAudioTooLongError, match=TOO_LONG):
            adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
        assert set(uploads.iterdir()) == before
    assert [d["stop"] for d in decoded] == [media_io.MAX_REF_AUDIO_SECONDS + 1.0]


@pytest.mark.parametrize("seconds", [59, 61])
@pytest.mark.parametrize("adapter", [adapters.OmniVoiceAdapter, adapters.ChatterboxAdapter, adapters.Qwen3TTSAdapter])
def test_a_wav_longer_than_the_cap_is_refused_before_decoding(adapter, seconds, dirs, monkeypatch):
    """Real decoders, 8 kHz WAV. torchcodec is spied on: libsndfile reads a WAV,
    so neither the check nor the refusal decodes it with FFmpeg."""
    decodes = []
    if adapter is adapters.OmniVoiceAdapter:
        decoder = _audio_decoder()
        import torchcodec.decoders

        class Spy(decoder):
            def get_samples_played_in_range(self, *args, **kwargs):
                decodes.append(True)
                return super().get_samples_played_in_range(*args, **kwargs)

        monkeypatch.setattr(torchcodec.decoders, "AudioDecoder", Spy)
    uploads, _ = dirs
    before = set(uploads.iterdir())
    req = SpeechRequest(input="hi", ref_audio=_clip("WAV", "FILE", None, rate=8000, seconds=seconds), ref_text="hi")
    if seconds <= media_io.MAX_REF_AUDIO_SECONDS:
        (path,) = adapter().speech_to_request(req, uploads).file_paths["audio"]
        assert path.startswith(str(uploads))
        assert decodes == []
        return
    with pytest.raises(media_io.RefAudioTooLongError, match=TOO_LONG):
        adapter().speech_to_request(req, uploads)
    assert decodes == [] and set(uploads.iterdir()) == before


def test_ref_audio_probe_decodes_are_bounded(tmp_path, monkeypatch):
    """Twice as many concurrent checks as decode slots: at most the slot count
    decode at once, and the rest run once a slot frees. Stub decoder, no FFmpeg."""
    import threading
    import time
    import types

    slots = media_io.REF_AUDIO_DECODE_SLOTS
    release = threading.Event()
    lock = threading.Lock()
    active = peak = 0

    class AudioDecoder:
        def __init__(self, source, **kwargs):
            self.metadata = types.SimpleNamespace(sample_rate=24000, duration_seconds_from_header=1.0)

        def get_samples_played_in_range(self, start_seconds=0.0, stop_seconds=None):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            release.wait(10)
            with lock:
                active -= 1
            return types.SimpleNamespace(pts_seconds=0.0, duration_seconds=1.0)

    _stub_torchcodec(monkeypatch, AudioDecoder)
    paths = []
    for i in range(2 * slots):
        paths.append(tmp_path / f"clip{i}.wav")
        paths[-1].write_bytes(WAV_BYTES)
    errors = []

    def check(path):
        try:
            media_io.check_ref_audio(str(path), "torchcodec")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=check, args=(p,)) for p in paths]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 5
    while active < slots and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    assert active == slots and peak == slots
    release.set()
    for t in threads:
        t.join(10)
    assert errors == [] and peak == slots and not any(t.is_alive() for t in threads)


RATES = [8000, 16000, 24000, 44100, 48000]


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("clip", FFMPEG_READS)
def test_omnivoice_accepts_clips_at_common_rates(clip, rate, dirs):
    uploads, _ = dirs
    _audio_decoder()
    req = SpeechRequest(input="hi", ref_audio=_clip(*clip, rate=rate), ref_text="hi")
    (path,) = adapters.OmniVoiceAdapter().speech_to_request(req, uploads).file_paths["audio"]
    assert path.startswith(str(uploads))


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("clip", FFMPEG_READS + SOUNDFILE_ONLY)
def test_qwen3_tts_accepts_clips_at_common_rates(clip, rate, dirs):
    uploads, _ = dirs
    req = SpeechRequest(input="hi", ref_audio=_clip(*clip, rate=rate), ref_text="hi")
    (path,) = adapters.Qwen3TTSAdapter().speech_to_request(req, uploads).file_paths["audio"]
    assert path.startswith(str(uploads))


@pytest.mark.parametrize("rate", [1, 4000, 7999])
def test_omnivoice_without_torchcodec_refuses_a_low_rate_wav_without_decoding_it(rate, dirs, monkeypatch):
    """PyAV and libsndfile take the same rate floor, from the header."""
    model_media = _without_torchcodec(monkeypatch)
    decodes = []
    monkeypatch.setattr(model_media._FallbackAudio, "decode", lambda *a, **k: decodes.append(True))
    uploads, _ = dirs
    before = set(uploads.iterdir())
    clip = _clip("WAV", "FILE", None, rate=rate, seconds=32000 / rate)
    req = SpeechRequest(input="hi", ref_audio=clip, ref_text="hi")
    with pytest.raises(ValueError, match=REFUSED):
        adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
    assert decodes == [] and set(uploads.iterdir()) == before


@pytest.mark.parametrize("seconds", [59, 61])
@pytest.mark.parametrize("clip", [("WAV", "FILE", None), ("OGG", "FILE", "VORBIS")])
def test_omnivoice_without_torchcodec_refuses_a_clip_past_the_cap(clip, seconds, dirs, monkeypatch):
    """A clip libsndfile cannot open goes to the FFmpeg decoders (PyAV here): the
    cap holds from the header too, and an accepted clip's probe decode is mono
    at 16 kHz and stops just past the cap."""
    model_media = _without_torchcodec(monkeypatch)

    def not_a_libsndfile_container(path):
        raise RuntimeError("Error opening: Format not recognised.")

    monkeypatch.setattr(media_io, "_check_ref_audio_soundfile", not_a_libsndfile_container)
    real_decode = model_media._FallbackAudio.decode
    probes = []

    def spy(self, source, sample_rate, num_channels, max_seconds=None):
        probes.append((sample_rate, num_channels, max_seconds))
        return real_decode(self, source, sample_rate, num_channels, max_seconds)

    monkeypatch.setattr(model_media._FallbackAudio, "decode", spy)
    uploads, _ = dirs
    before = set(uploads.iterdir())
    req = SpeechRequest(input="hi", ref_audio=_clip(*clip, rate=8000, seconds=seconds), ref_text="hi")
    if seconds <= media_io.MAX_REF_AUDIO_SECONDS:
        (path,) = adapters.OmniVoiceAdapter().speech_to_request(req, uploads).file_paths["audio"]
        assert path.startswith(str(uploads))
        assert probes == [(16000, 1, media_io.MAX_REF_AUDIO_SECONDS + 1.0)]
        return
    with pytest.raises(media_io.RefAudioTooLongError, match=TOO_LONG):
        adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
    assert probes == [] and set(uploads.iterdir()) == before


@pytest.mark.parametrize("clip", SOUNDFILE_ONLY)
def test_omnivoice_without_torchcodec_accepts_what_libsndfile_reads(clip, dirs, monkeypatch):
    """FFmpeg misreads some of these headers, so the header comes from
    libsndfile, which also decodes them for the model. At 8 kHz, which all of
    them store exactly: HTK keeps a sample period in 100 ns units, so 24 kHz
    reads back as 24,038 Hz, a rate check_resample_rate refuses."""
    model_media = _without_torchcodec(monkeypatch)
    uploads, _ = dirs
    req = SpeechRequest(input="hi", ref_audio=_clip(*clip, rate=8000), ref_text="hi")
    (path,) = adapters.OmniVoiceAdapter().speech_to_request(req, uploads).file_paths["audio"]
    assert model_media.decode_audio(path, sample_rate=24000, num_channels=1).data.numel()


def test_omnivoice_without_torchcodec_refuses_bytes_neither_decoder_reads(dirs, monkeypatch):
    _without_torchcodec(monkeypatch)
    uploads, _ = dirs
    before = set(uploads.iterdir())
    req = SpeechRequest(input="hi", ref_audio=base64.b64encode(WAV_BYTES).decode(), ref_text="hi")
    with pytest.raises(ValueError, match=REFUSED):
        adapters.OmniVoiceAdapter().speech_to_request(req, uploads)
    assert set(uploads.iterdir()) == before


@pytest.mark.parametrize("adapter", [adapters.Cosmos3Adapter, adapters.Wan22Adapter])
def test_video_generation_conditioning_refuses_server_paths(adapter, dirs):
    uploads, secret = dirs
    with pytest.raises(ValueError, match=REFUSED):
        adapter().video_to_request(VideoGenerationRequest(prompt="x", image=str(secret)), uploads)
