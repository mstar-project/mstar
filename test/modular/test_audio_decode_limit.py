"""Request audio is bounded by what it decodes to, not by its header (#508).

Every decoder of request audio stops just past ``media_io.MAX_AUDIO_SECONDS``
(or a tighter per-call bound) and refuses the file, whatever the container
declares. The stub-decoder cases run without soundfile or torchcodec (CI);
the real-decoder cases skip when those are not installed.
"""

import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from mstar.api_server import media_io
from mstar.api_server.media_io import MAX_AUDIO_SECONDS, AudioTooLongError

# a minute past the limit
TOO_LONG = MAX_AUDIO_SECONDS + 60


# --------------------------------------------------------------------------
# stub decoders: a container whose header says one length and whose samples
# run on for another
# --------------------------------------------------------------------------


class _FakeSoundfile(types.ModuleType):
    """``soundfile`` whose files decode to ``seconds`` of silence at ``rate``
    while the header claims ``header_seconds``; counts the frames handed out."""

    class SoundFileError(Exception):
        pass

    def __init__(self, seconds, rate=16000, header_seconds=None, channels=1):
        super().__init__("soundfile")
        self.total = int(seconds * rate)
        self.rate = rate
        self.channels = channels
        self.header_frames = self.total if header_seconds is None else int(header_seconds * rate)
        self.frames_read = 0
        fake = self

        class SoundFile:
            def __init__(self, path, *a, **k):
                self.samplerate = fake.rate
                self._pos = 0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, frames=-1, dtype="float64", always_2d=False, **k):
                left = fake.total - self._pos
                n = left if frames < 0 else min(frames, left)
                self._pos += n
                fake.frames_read += n
                return np.zeros((n, fake.channels), dtype=np.float32)

        self.SoundFile = SoundFile

    def info(self, path):
        return types.SimpleNamespace(frames=self.header_frames, samplerate=self.rate, channels=self.channels)

    def read(self, path, dtype="float64", always_2d=False, **k):
        # the whole-file read main used: hands out every frame
        self.frames_read += self.total
        return np.zeros((self.total, self.channels), dtype=np.float32), self.rate


class _NoSoundfile(types.ModuleType):
    """``soundfile`` that cannot read the container (an m4a, a webm)."""

    class SoundFileError(Exception):
        pass

    def __init__(self):
        super().__init__("soundfile")

    def _fail(self, *a, **k):
        raise self.SoundFileError("Format not recognised")

    info = read = SoundFile = _fail


def _fake_torchcodec(monkeypatch, seconds, header_seconds=None):
    """``torchcodec.decoders.AudioDecoder`` decoding ``seconds`` of audio at
    the requested rate; the header gives ``header_seconds`` (None: unknown)."""
    calls = {"all": 0, "stops": []}

    class AudioDecoder:
        def __init__(self, path, sample_rate=16000, num_channels=None):
            self.rate = sample_rate or 16000
            self.metadata = types.SimpleNamespace(
                duration_seconds_from_header=header_seconds, duration_seconds=header_seconds,
                sample_rate=self.rate,
            )

        def _samples(self, play):
            return types.SimpleNamespace(
                data=torch.zeros(1, int(play * self.rate)), pts_seconds=0.0, duration_seconds=play,
                sample_rate=self.rate,
            )

        def get_all_samples(self):
            calls["all"] += 1
            return self._samples(seconds)

        def get_samples_played_in_range(self, start, stop):
            calls["stops"].append(stop)
            return self._samples(min(seconds, stop))

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.AudioDecoder = AudioDecoder
    pkg = types.ModuleType("torchcodec")
    pkg.decoders = decoders
    monkeypatch.setitem(sys.modules, "torchcodec", pkg)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    return calls


def _use_soundfile(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "soundfile", fake)
    return fake


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF")
    return str(path)


# --------------------------------------------------------------------------
# media_io's decoders
# --------------------------------------------------------------------------


def test_decode_audio_stops_a_soundfile_read_at_the_limit(monkeypatch, clip):
    sf = _use_soundfile(monkeypatch, _FakeSoundfile(TOO_LONG, header_seconds=10))
    with pytest.raises(AudioTooLongError) as e:
        media_io.decode_audio(clip, 16000)
    assert e.value.status_code == 413
    # at most one frame past the limit is decoded
    assert sf.frames_read <= MAX_AUDIO_SECONDS * 16000 + 1


def test_a_soundfile_rate_past_the_range_is_refused_before_any_read(monkeypatch, clip):
    # the frame cap is seconds times the file's rate, so the rate is bounded
    # first, and no loader falls back to torchcodec for a refused file
    from mstar.model.chatterbox.chatterbox_model import ChatterboxModel
    from mstar.model.components.audio_features import load_speech_file
    from mstar.model.nemotron_duplex.nemotron_duplex_model import NemotronDuplexModel

    for load in (
        lambda: media_io.read_soundfile_capped(clip),
        lambda: media_io.decode_audio(clip, 16000),
        lambda: load_speech_file(clip, 16000, MAX_AUDIO_SECONDS),
        lambda: ChatterboxModel._decode_audio(clip),
        lambda: NemotronDuplexModel.load_audio(None, clip, "cpu"),
    ):
        sf = _use_soundfile(monkeypatch, _FakeSoundfile(1, rate=media_io.MAX_SAMPLE_RATE + 1))
        calls = _fake_torchcodec(monkeypatch, 1)
        with pytest.raises(ValueError, match="unsupported sample rate"):
            load()
        assert sf.frames_read == 0
        assert calls["all"] == 0 and not calls["stops"]


def test_decode_audio_stops_a_torchcodec_decode_at_the_limit(monkeypatch, clip):
    _use_soundfile(monkeypatch, _NoSoundfile())
    calls = _fake_torchcodec(monkeypatch, TOO_LONG)
    with pytest.raises(AudioTooLongError):
        media_io.decode_audio(clip, 16000)
    assert calls["all"] == 0 and calls["stops"] == [MAX_AUDIO_SECONDS + 1.0]


def test_decode_audio_keeps_audio_under_the_limit(monkeypatch, clip):
    _use_soundfile(monkeypatch, _FakeSoundfile(600, rate=16000, channels=2))
    assert media_io.decode_audio(clip, 16000).shape == (600 * 16000,)
    _use_soundfile(monkeypatch, _NoSoundfile())
    _fake_torchcodec(monkeypatch, 600)
    assert media_io.decode_audio(clip, 16000).shape == (600 * 16000,)


def test_torchcodec_counts_decoded_samples_not_play_time(monkeypatch, clip):
    # a stream whose timestamps stay inside the window but whose frames
    # decode to more samples than the limit allows
    calls = _fake_torchcodec(monkeypatch, 5)

    class Overfull(sys.modules["torchcodec.decoders"].AudioDecoder):
        def get_samples_played_in_range(self, start, stop):
            calls["stops"].append(stop)
            return types.SimpleNamespace(data=torch.zeros(1, 20 * self.rate), pts_seconds=0.0, duration_seconds=1.0)

    monkeypatch.setattr(sys.modules["torchcodec.decoders"], "AudioDecoder", Overfull)
    with pytest.raises(AudioTooLongError):
        media_io.decode_torchcodec_capped(clip, 16000, max_seconds=10)


@pytest.mark.parametrize("play, refused", [(60, False), (MAX_AUDIO_SECONDS, False), (MAX_AUDIO_SECONDS + 1, True)])
def test_torchcodec_play_time_counts_from_the_first_timestamp(monkeypatch, clip, play, refused):
    # a stream whose first frame is at 1800 s plays from there; its length is
    # what is capped, not its timestamps
    calls = _fake_torchcodec(monkeypatch, 0)

    class Offset(sys.modules["torchcodec.decoders"].AudioDecoder):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.metadata.begin_stream_seconds_from_header = 1800.0

        def get_samples_played_in_range(self, start, stop):
            calls["stops"].append((start, stop))
            first, last = max(start, 1800.0), min(stop, 1800.0 + play)
            return types.SimpleNamespace(
                data=torch.zeros(1, int((last - first) * self.rate)), pts_seconds=first, duration_seconds=last - first,
            )

    monkeypatch.setattr(sys.modules["torchcodec.decoders"], "AudioDecoder", Offset)
    if refused:
        with pytest.raises(AudioTooLongError):
            media_io.decode_torchcodec_capped(clip, 16000)
    else:
        assert media_io.decode_torchcodec_capped(clip, 16000).numel() == play * 16000
    assert calls["stops"] == [(1800.0, 1800.0 + MAX_AUDIO_SECONDS + 1.0)]


# --------------------------------------------------------------------------
# the data worker's loaders (chat and /generate audio, reference clips)
# --------------------------------------------------------------------------


def test_base_load_audio_is_capped(monkeypatch, clip):
    from mstar.model.base import Model

    model = types.SimpleNamespace(AUDIO_INPUT_RATE=16000, MAX_AUDIO_SECONDS=MAX_AUDIO_SECONDS)
    calls = _fake_torchcodec(monkeypatch, TOO_LONG)
    with pytest.raises(AudioTooLongError):
        Model.load_audio(model, clip, "cpu")
    assert calls["all"] == 0
    _fake_torchcodec(monkeypatch, 12)
    assert Model.load_audio(model, clip, "cpu").data.shape == (12 * 16000,)
    # a model's own cap is the decode's bound
    model.MAX_AUDIO_SECONDS = 10
    with pytest.raises(AudioTooLongError, match="10 s"):
        Model.load_audio(model, clip, "cpu")


def test_soundfile_loaders_are_capped(monkeypatch, clip):
    from mstar.model.chatterbox.chatterbox_model import ChatterboxModel
    from mstar.model.components.audio_features import load_speech_file
    from mstar.model.nemotron_duplex.nemotron_duplex_model import NemotronDuplexModel

    duplex = object.__new__(NemotronDuplexModel)
    for load, error, match, seconds in (
        (lambda: load_speech_file(clip, 16000, MAX_AUDIO_SECONDS), AudioTooLongError, None, MAX_AUDIO_SECONDS),
        (lambda: ChatterboxModel._decode_audio(clip), AudioTooLongError, None, MAX_AUDIO_SECONDS),
        # capped tighter, at its longest session, and refused with that error
        (lambda: duplex.load_audio(clip, "cpu"), ValueError, "longest session", duplex.max_session_s),
    ):
        sf = _use_soundfile(monkeypatch, _FakeSoundfile(TOO_LONG, header_seconds=10))
        # chatterbox falls back to torchcodec on a soundfile failure; the cap must not
        calls = _fake_torchcodec(monkeypatch, TOO_LONG)
        with pytest.raises(error, match=match):
            load()
        assert sf.frames_read <= seconds * 16000 + 1
        assert calls["all"] == 0 and not calls["stops"]


def test_qwen3_tts_reference_decode_stops_at_its_bound(monkeypatch, clip):
    from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel

    # the header says 5 s and passes the model's check; the samples run on
    sf = _use_soundfile(monkeypatch, _FakeSoundfile(60, rate=24000, header_seconds=5))
    config = types.SimpleNamespace(
        reference_min_seconds=1.0, reference_max_seconds=30.0,
        codec=types.SimpleNamespace(input_sample_rate=24000),
    )
    model = types.SimpleNamespace(config=config)
    model._check_reference_duration = lambda s: Qwen3TTSModel._check_reference_duration(model, s)
    with pytest.raises(AudioTooLongError, match="30 s"):
        Qwen3TTSModel.load_audio(model, clip, "cpu")
    assert sf.frames_read <= 30 * 24000 + 1


def test_omnilingual_asr_decode_stops_at_its_clip_bound(monkeypatch, clip):
    from mstar.model.omnilingual_asr import audio
    from mstar.model.omnilingual_asr.config import MAX_AUDIO_SECONDS as CLIP_SECONDS

    # the header says 10 s; the samples run on for an hour
    sf = _use_soundfile(monkeypatch, _FakeSoundfile(3600, header_seconds=10))
    calls = _fake_torchcodec(monkeypatch, 3600)
    with pytest.raises(AudioTooLongError, match="40 s"):
        audio.decode_file(clip)
    assert sf.frames_read <= CLIP_SECONDS * 16000 + 1
    assert calls["all"] == 0 and not calls["stops"]
    # a container soundfile cannot read goes to torchcodec, capped the same way
    _use_soundfile(monkeypatch, _NoSoundfile())
    calls = _fake_torchcodec(monkeypatch, 3600)
    with pytest.raises(AudioTooLongError, match="40 s"):
        audio.decode_file(clip)
    assert calls["all"] == 0 and calls["stops"] == [CLIP_SECONDS + 1.0]


def test_omnilingual_asr_decode_keeps_a_clip_under_its_bound(monkeypatch, clip):
    from mstar.model.omnilingual_asr import audio

    _use_soundfile(monkeypatch, _FakeSoundfile(30))
    decoded, rate = audio.decode_file(clip)
    assert rate == 16000 and decoded.shape[0] == 30 * 16000
    _use_soundfile(monkeypatch, _NoSoundfile())
    _fake_torchcodec(monkeypatch, 30)
    decoded, rate = audio.decode_file(clip)
    assert rate == 16000 and decoded.shape == (30 * 16000,)


def test_omnilingual_asr_refuses_a_rate_past_the_range_before_reading(monkeypatch, clip):
    from mstar.model.omnilingual_asr import audio

    sf = _use_soundfile(monkeypatch, _FakeSoundfile(1, rate=media_io.MAX_SAMPLE_RATE + 1))
    calls = _fake_torchcodec(monkeypatch, 1)
    with pytest.raises(media_io.AudioRefusedError):
        audio.decode_file(clip)
    assert sf.frames_read == 0 and calls["all"] == 0 and not calls["stops"]


# --------------------------------------------------------------------------
# /v1/audio/transcriptions
# --------------------------------------------------------------------------


class _StubAPI:
    def __init__(self):
        self.model_name = "whisper_large"
        self.model = None
        self.upload_dir = Path(tempfile.mkdtemp())
        self.submits: list = []

    def submit_request(self, **kw):
        self.submits.append(kw)
        return kw["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return [types.SimpleNamespace(modality="text", data=b" hi", metadata={})]

    async def iter_result_chunks(self, request_id):
        yield types.SimpleNamespace(modality="text", data=b" hi", metadata={})

    def release_request(self, request_id):
        pass


@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import mstar.api_server

    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    fake_ep.api_server = _StubAPI()
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)
    from mstar.api_server.openai.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), fake_ep.api_server


def _transcribe(client, content=b"RIFFaudio", filename="speech.webm", model="whisper_large"):
    return client.post(
        "/v1/audio/transcriptions", data={"model": model}, files={"file": (filename, content, "audio/webm")},
    )


@pytest.mark.parametrize("model", ["whisper_large", "higgs_audio"])
def test_transcription_without_a_header_duration_is_bounded_by_its_decode(monkeypatch, client, model):
    # webm-like: soundfile cannot open it and FFmpeg reports no duration; on
    # main the upload went to the data worker undecoded, whatever its length
    http, api = client
    api.model_name = model
    _use_soundfile(monkeypatch, _NoSoundfile())
    calls = _fake_torchcodec(monkeypatch, MAX_AUDIO_SECONDS + 60)
    r = _transcribe(http, model=model)
    assert r.status_code == 413
    assert "1800 s" in r.json()["error"]["message"]
    assert api.submits == [] and calls["all"] == 0
    assert list(api.upload_dir.iterdir()) == []


def test_windowed_transcription_with_an_understated_header_is_refused(monkeypatch, client):
    # the header says 60 s, so the header check passes; the decode runs on
    http, api = client
    _use_soundfile(monkeypatch, _FakeSoundfile(TOO_LONG, header_seconds=60))
    _fake_torchcodec(monkeypatch, TOO_LONG)
    r = _transcribe(http, filename="speech.wav")
    assert r.status_code == 413
    assert api.submits == []


def test_transcription_under_the_limit_still_runs(monkeypatch, client):
    http, api = client
    _use_soundfile(monkeypatch, _NoSoundfile())
    _fake_torchcodec(monkeypatch, 20)
    r = _transcribe(http)
    assert r.status_code == 200, r.text
    assert len(api.submits) == 1


def test_a_headerless_upload_longer_than_the_clip_is_windowed(monkeypatch, client):
    # once the decode gives the length, a 45 s upload is cut into Whisper's
    # 30 s windows like one whose header gives it (main sent it whole)
    http, api = client
    _use_soundfile(monkeypatch, _NoSoundfile())
    _fake_torchcodec(monkeypatch, 45)
    r = _transcribe(http)
    assert r.status_code == 200, r.text
    assert len(api.submits) == 2


def test_an_unknown_soundfile_length_is_not_a_duration(monkeypatch, tmp_path):
    from mstar.api_server.openai.serving_transcriptions import audio_duration_seconds

    sf = _FakeSoundfile(120)
    sf.header_frames = 2**63 - 1  # libsndfile's value for "length not in the header"
    _use_soundfile(monkeypatch, sf)
    _fake_torchcodec(monkeypatch, 120, header_seconds=None)
    path = tmp_path / "a.flac"
    path.write_bytes(b"fLaC")
    assert audio_duration_seconds(str(path)) is None


# --------------------------------------------------------------------------
# real decoders
# --------------------------------------------------------------------------


def test_real_soundfile_decode_stops_at_the_limit(tmp_path):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "a.wav"
    rng = np.random.default_rng(0)
    stereo = (0.1 * rng.standard_normal((8000 * 3, 2))).astype(np.float32)
    sf.write(path, stereo, 8000)
    with pytest.raises(AudioTooLongError):
        media_io.read_soundfile_capped(str(path), max_seconds=2)
    audio, rate = media_io.read_soundfile_capped(str(path), max_seconds=3)
    whole, _ = sf.read(path, dtype="float32", always_2d=True)
    assert rate == 8000 and np.array_equal(audio, whole.mean(axis=1))


def test_blockwise_mp3_decode_equals_one_read(tmp_path, monkeypatch):
    """A read that splits an MPEG frame glitches the samples after it (0.027
    on a 45 s, 44.1 kHz speech upload with 10 s blocks), so blocks are whole frames."""
    sf = pytest.importorskip("soundfile")
    if "MPEG_LAYER_III" not in sf.available_subtypes("MP3"):
        pytest.skip("this libsndfile cannot write MP3")
    monkeypatch.setattr(media_io, "_SOUNDFILE_BLOCK_SECONDS", 1)
    rate = 22_050
    t = np.arange(8 * rate) / rate
    signal = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.default_rng(0).standard_normal(t.size)
    path = tmp_path / "clip.mp3"
    sf.write(path, signal.astype(np.float32), rate, format="MP3", subtype="MPEG_LAYER_III")
    whole, _ = sf.read(path, dtype="float32", always_2d=True)
    audio, got = media_io.read_soundfile_capped(str(path))
    assert got == rate and audio.shape == (whole.shape[0],)
    np.testing.assert_allclose(audio, whole.mean(axis=1), atol=1e-6, rtol=0)


def _flac_without_length(path: Path, seconds: float, rate: int = 16000) -> None:
    """A FLAC whose STREAMINFO leaves the total sample count out (0), which
    the format allows for streamed encoders."""
    sf = pytest.importorskip("soundfile")
    tone = (0.1 * np.sin(np.arange(int(seconds * rate)) * 2 * np.pi * 440 / rate)).astype(np.float32)
    sf.write(path, tone, rate)
    data = bytearray(path.read_bytes())
    assert data[:4] == b"fLaC"
    streaminfo = int.from_bytes(data[8:42], "big")
    shift = 34 * 8 - 108 - 36  # the 36-bit total sample count
    streaminfo &= ~(((1 << 36) - 1) << shift)
    data[8:42] = streaminfo.to_bytes(34, "big")
    path.write_bytes(bytes(data))


def test_headerless_flac_decodes_without_torchcodec(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "speech.flac"
    _flac_without_length(path, 5)
    assert sf.info(path).frames == 2**63 - 1
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)

    from mstar.model.components.audio_features import load_speech_file

    audio = load_speech_file(str(path), 16000, max_seconds=5)
    assert len(audio) == 5 * 16000
    reference = tmp_path / "reference.flac"
    tone = (0.1 * np.sin(np.arange(5 * 16000) * 2 * np.pi * 440 / 16000)).astype(np.float32)
    sf.write(reference, tone, 16000)
    expected, _ = sf.read(reference, dtype="float32")
    np.testing.assert_array_equal(audio.numpy(), expected)


def test_headerless_flac_still_obeys_the_decoded_limit(tmp_path):
    path = tmp_path / "speech.flac"
    _flac_without_length(path, 5)
    with pytest.raises(AudioTooLongError):
        media_io.read_soundfile_capped(str(path), max_seconds=1)


def test_headerless_flac_still_obeys_the_30_minute_cap(monkeypatch):
    class LongFile:
        format = "FLAC"
        samplerate = 1
        channels = 1
        frames = 2**63 - 1

        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, frames, **kwargs):
            return np.zeros((frames, 1), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(SoundFile=LongFile))
    with pytest.raises(AudioTooLongError):
        media_io.read_soundfile_capped("long.flac", max_seconds=MAX_AUDIO_SECONDS)


@pytest.mark.parametrize("format_, frames", [
    ("FLAC", 16000), ("WAV", 2**63 - 1), ("FLAC", 2**63 - 1),
])
def test_soundfile_read_errors_outside_headerless_flac_still_raise(monkeypatch, format_, frames):
    class FailingFile:
        format = format_
        samplerate = 16000
        channels = 1

        def __init__(self, path):
            self.frames = frames

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, *args, **kwargs):
            raise RuntimeError("decode failed")

    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(SoundFile=FailingFile))
    with pytest.raises(RuntimeError, match="decode failed"):
        media_io.read_soundfile_capped("corrupt.flac", max_seconds=10)


def test_stereo_24bit_headerless_flac_decodes_without_torchcodec(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "stereo24.flac"
    rate = 16000
    t = np.arange(5 * rate) / rate
    stereo = np.stack([
        0.15 * np.sin(2 * np.pi * 440 * t),
        0.15 * np.cos(2 * np.pi * 660 * t),
    ], axis=1).astype(np.float32)
    sf.write(path, stereo, rate, subtype="PCM_24")
    reference = tmp_path / "reference24.flac"
    reference.write_bytes(path.read_bytes())

    data = bytearray(path.read_bytes())
    assert data[:4] == b"fLaC"
    streaminfo = int.from_bytes(data[8:42], "big")
    shift = 34 * 8 - 108 - 36
    streaminfo &= ~(((1 << 36) - 1) << shift)
    data[8:42] = streaminfo.to_bytes(34, "big")
    path.write_bytes(bytes(data))

    assert sf.info(path).frames == 2**63 - 1
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)

    from mstar.model.components.audio_features import load_speech_file

    expected, _ = sf.read(reference, dtype="float32", always_2d=True)
    expected_mono = expected.mean(axis=1)
    decoded, got_rate = media_io.read_soundfile_capped(str(path), max_seconds=5)
    assert got_rate == rate
    np.testing.assert_array_equal(decoded, expected_mono)
    tensor = load_speech_file(str(path), rate, max_seconds=5)
    np.testing.assert_array_equal(tensor.numpy(), expected_mono)


def test_corrupt_headerless_flac_still_raises(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "corrupt.flac"
    _flac_without_length(path, 2)
    data = bytearray(path.read_bytes())
    for idx in range(1000, min(1200, len(data))):
        data[idx] = 0
    path.write_bytes(bytes(data))
    assert sf.info(path).frames == 2**63 - 1
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)

    with pytest.raises(sf.LibsndfileError, match="lost sync"):
        media_io.read_soundfile_capped(str(path), max_seconds=5)


def test_flac_without_a_length_is_measured_by_decoding_it(client, tmp_path, monkeypatch):
    # main read libsndfile's "unknown" frame count as a duration of about
    # 5.8e14 s and refused a 2 minute clip with a 413
    pytest.importorskip("soundfile")
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    http, api = client
    path = tmp_path / "speech.flac"
    _flac_without_length(path, 120)
    r = _transcribe(http, content=path.read_bytes(), filename="speech.flac")
    assert r.status_code == 200, r.text
    assert api.submits


def test_live_http_serving_decodes_headerless_flac_without_torchcodec(tmp_path, monkeypatch):
    """CPU live HTTP serving check over uvicorn + APIServer + PreprocessWorkerThread
    + Model.load_audio with torchcodec disabled: both /v1/audio/transcriptions
    (windowed >30s headerless FLAC) and /generate (headerless FLAC upload) decode
    without FFmpeg or torchcodec."""
    import base64
    import queue
    import socket
    import threading
    import time

    sf = pytest.importorskip("soundfile")
    httpx = pytest.importorskip("httpx")
    uvicorn = pytest.importorskip("uvicorn")
    from api_server_helpers import StandInSocket, make_test_api_server

    from mstar.api_server import entrypoint
    from mstar.api_server.data_worker import PreprocessWorker, PreprocessWorkerThread, ResultQueue
    from mstar.api_server.request_types import ResultChunk
    from mstar.communication.event import EventWakeup
    from mstar.model.base import Model
    from mstar.testing_utils.model import StubModel
    from mstar.utils.ipc_format import ConductorMessageType

    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)

    class _AudioModel(StubModel):
        AUDIO_INPUT_RATE = 16000
        MAX_AUDIO_SECONDS = MAX_AUDIO_SECONDS

        def __init__(self):
            self.recorded_audio = []

        def process_prompt(
            self, text, input_modalities, output_modalities,
            tensors=None, input_metadata=None, **kwargs,
        ):
            for wav in (tensors or {}).get("audio_inputs", ()):
                self.recorded_audio.append(wav.clone())
            return {}

    class _TensorManager:
        def store_and_return_tensor_info(self, request_id, tensors):
            return {}

        def register_for_send(self, request_id, tensor_infos):
            pass

        def set_persist(self, uuid, persist=True):
            pass

        def take_failed_reads(self):
            return []

        def get_ready_tensors(self):
            return {}

        def has_inflight_reads(self, request_id):
            return False

        def force_cleanup_request(self, request_id):
            pass

        def ack_unread_tensors(self, graph_edges):
            pass

    class _WorkerSocket:
        codec = types.SimpleNamespace(encode=lambda msg: msg)

        def __init__(self, on_new_request):
            self.on_new_request = on_new_request

        def get_all_new_messages(self):
            return []

        def send(self, entity, msg):
            if entity == "conductor" and msg.message_type == ConductorMessageType.NEW_REQUEST:
                self.on_new_request(msg.body.request_id)

        def try_send_bytes(self, entity, msg):
            self.send(entity, msg)
            return True

    model = _AudioModel()
    # Explicitly verify Model.load_audio is inherited unchanged.
    assert type(model).load_audio is Model.load_audio

    stop_event = threading.Event()
    out_queue = ResultQueue(EventWakeup())
    profile_queue = ResultQueue(EventWakeup())
    server_holder = {}

    def complete_request(rid):
        srv = server_holder["api"]
        with srv.request_lock:
            srv.pending_requests[rid].output_chunk_counts = {"text_inputs": 1}
            srv.recently_completed[rid] = time.time()
        out_queue.put(ResultChunk(request_id=rid, modality="text", data=b"ok", stopped=True))
        out_queue.flush()

    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(),
        result_tensor_queue=queue.Queue(),
        out_queue=out_queue,
        profile_queue=profile_queue,
        cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(),
        stop_event=stop_event,
        communicator=_WorkerSocket(complete_request),
        tensor_manager=_TensorManager(),
        model=model,
    )
    facade = PreprocessWorker.__new__(PreprocessWorker)
    facade.request_input_queue = worker.in_queue
    facade.result_tensor_input_queue = worker.result_tensor_queue
    facade.abort_request_queue = worker.abort_request_queue
    facade.reads_done_queue = worker.reads_done_queue
    facade.cleanup_request_queue = worker.cleanup_request_queue
    facade.discard_tensor_queue = worker.discard_tensor_queue
    facade.output_queue = out_queue
    facade.results_wakeup = out_queue.wakeup
    facade.received_chunk_counts = {}
    facade.per_request_reading_tensors = {}
    facade.get_result_chunks = lambda: [out_queue.get_nowait()] if not out_queue.empty() else []
    facade.get_profile_updates = lambda: [profile_queue.get_nowait()] if not profile_queue.empty() else []
    facade.has_pending_tensors = lambda rid: False
    facade.received_final_chunks = lambda rid, counts: True
    facade.finished_reading = lambda rid, drained=True: None

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    api = make_test_api_server(
        preprocess_worker=facade,
        communicator=StandInSocket(),
        model=model,
        model_name="whisper_large",
        upload_dir=upload_dir,
    )
    server_holder["api"] = api
    monkeypatch.setattr(entrypoint, "api_server", api)

    worker_thread = threading.Thread(target=worker.run, daemon=True)
    msg_thread = threading.Thread(target=api._process_messages, daemon=True)
    worker_thread.start()
    msg_thread.start()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    http_server = uvicorn.Server(uvicorn.Config(entrypoint.app, host="127.0.0.1", port=port, log_level="warning"))
    http_thread = threading.Thread(target=http_server.run, daemon=True)
    http_thread.start()

    deadline = time.monotonic() + 15.0
    while not http_server.started:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15.0) as http:
            # 1. >30s headerless FLAC on /v1/audio/transcriptions -> WindowPlanner splits into 2 windows.
            long_flac = tmp_path / "long_headerless.flac"
            _flac_without_length(long_flac, 45)
            assert sf.info(long_flac).frames == 2**63 - 1
            r1 = http.post(
                "/v1/audio/transcriptions",
                data={"model": "whisper_large"},
                files={"file": ("long_headerless.flac", long_flac.read_bytes(), "audio/flac")},
            )
            assert r1.status_code == 200, r1.text
            assert r1.json()["text"] == "ok ok"
            assert len(model.recorded_audio) == 2

            # 2. 5s headerless FLAC upload on /generate -> Model.load_audio -> read_soundfile_capped.
            short_flac = tmp_path / "short_headerless.flac"
            _flac_without_length(short_flac, 5)
            assert sf.info(short_flac).frames == 2**63 - 1
            ref_flac = tmp_path / "short_ref.flac"
            tone = (0.1 * np.sin(np.arange(5 * 16000) * 2 * np.pi * 440 / 16000)).astype(np.float32)
            sf.write(ref_flac, tone, 16000)
            expected, _ = sf.read(ref_flac, dtype="float32")

            r2 = http.post(
                "/generate",
                data={"text": "transcribe", "output_modalities": "text", "streaming": "false"},
                files={"files": ("short_headerless.flac", short_flac.read_bytes(), "audio/flac")},
            )
            assert r2.status_code == 200, r2.text
            assert [base64.b64decode(o["data"]) for o in r2.json()["outputs"]["text"]] == [b"ok"]
            assert len(model.recorded_audio) == 3
            np.testing.assert_array_equal(model.recorded_audio[-1].numpy(), expected)
    finally:
        http_server.should_exit = True
        http_thread.join(timeout=10)
        api.running = False
        out_queue.wakeup.signal()
        msg_thread.join(timeout=5)
        stop_event.set()
        worker_thread.join(timeout=5)
