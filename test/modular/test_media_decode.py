"""media_io without torchcodec, as on ROCm. Runs the PyAV/soundfile
fallbacks on small generated clips and checks they return torchcodec's
layouts and metadata keys."""

import io
import sys
import types
import wave

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")

from mstar.model.components import media_io  # noqa: E402

# torchcodec 0.16's VideoStreamMetadata fields
TORCHCODEC_VIDEO_KEYS = {
    "duration_seconds_from_header", "begin_stream_seconds_from_header", "bit_rate", "codec",
    "stream_index", "duration_seconds", "begin_stream_seconds", "begin_stream_seconds_from_content",
    "end_stream_seconds_from_content", "width", "height", "num_frames_from_header",
    "num_frames_from_content", "average_fps_from_header", "pixel_aspect_ratio", "rotation",
    "color_primaries", "color_space", "color_transfer_characteristic", "pixel_format",
    "end_stream_seconds", "num_frames", "average_fps",
}


@pytest.fixture(autouse=True)
def without_torchcodec(monkeypatch):
    """Pick the decoders again with torchcodec unimportable."""
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    media_io._decoders.cache_clear()
    yield
    media_io._decoders.cache_clear()


def _clip(path, frames=8, fps=4, rotation=None):
    # a bar that grows 8 columns per frame to check order, and a white
    # square in the top-left corner to check orientation
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        if rotation is not None:
            stream.set_display_rotation(rotation)
        for t in range(frames):
            img = np.zeros((48, 64, 3), np.uint8)
            img[16:, : 8 * (t + 1)] = 200
            img[:6, :6] = 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return str(path)


def _bar_widths(frames: torch.Tensor) -> list[int]:
    return (frames[:, :, 16:].float().mean(dim=(1, 2)) > 100).sum(dim=-1).tolist()


def _wav(rate=8000, seconds=1.0, channels=2) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    pcm = (np.repeat(tone[:, None], channels, axis=1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_falls_back_without_torchcodec():
    assert not media_io.uses_torchcodec()
    assert media_io._decoders().audio.name == "pyav+soundfile"
    assert media_io._decoders().video.name == "pyav"


def test_uses_a_working_torchcodec(monkeypatch):
    calls = []

    class Samples:
        data = torch.zeros(1, 4)
        sample_rate = 16000

    class AudioDecoder:
        def __init__(self, source, sample_rate=None, num_channels=None):
            calls.append((source, sample_rate, num_channels))

        def get_all_samples(self):
            return Samples()

    fake = types.ModuleType("torchcodec.decoders")
    fake.AudioDecoder = AudioDecoder
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", fake)
    media_io._decoders.cache_clear()
    assert media_io.uses_torchcodec()
    out = media_io.decode_audio("x.wav", 16000, 1)
    assert calls[-1] == ("x.wav", 16000, 1) and out.data.shape == (1, 4)


def test_a_torchcodec_that_cannot_decode_is_not_used(monkeypatch):
    class AudioDecoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Could not load libtorchcodec")

    fake = types.ModuleType("torchcodec.decoders")
    fake.AudioDecoder = AudioDecoder
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", fake)
    media_io._decoders.cache_clear()
    assert not media_io.uses_torchcodec()


def test_decode_video_all_frames(tmp_path):
    frames, metadata = media_io.decode_video(_clip(tmp_path / "a.mp4"))
    assert frames.shape == (8, 3, 48, 64) and frames.dtype == torch.uint8
    # lossy codec: the bar's edge may blur by a column
    assert all(abs(n - 8 * (t + 1)) <= 2 for t, n in enumerate(_bar_widths(frames)))
    assert set(metadata) == TORCHCODEC_VIDEO_KEYS
    assert metadata["num_frames"] == metadata["num_frames_from_content"] == 8
    assert metadata["average_fps"] == 4.0 and metadata["duration_seconds"] == 2.0
    assert (metadata["width"], metadata["height"]) == (64, 48)
    assert metadata["codec"] == "h264" and metadata["pixel_format"] == "yuv420p"


def test_metadata_alone_matches_a_full_decode(tmp_path):
    path = _clip(tmp_path / "a.mp4")
    assert media_io.video_metadata(path) == media_io.decode_video(path).metadata


def test_decode_video_at_indices(tmp_path):
    path = _clip(tmp_path / "a.mp4")
    everything = media_io.decode_video(path).data
    picked = media_io.decode_video(path, indices=[7, 0, 3, 3]).data
    assert torch.equal(picked, everything[[7, 0, 3, 3]])
    with pytest.raises(IndexError):
        media_io.decode_video(path, indices=[8])


def test_sample_video_chooses_from_metadata(tmp_path):
    path = _clip(tmp_path / "a.mp4")
    seen = []

    def choose(metadata):
        seen.append(metadata["num_frames"])
        return list(range(0, metadata["num_frames"], 3))

    frames = media_io.sample_video(path, choose)
    assert seen == [8] and frames.data.shape[0] == 3
    assert torch.equal(frames.data, media_io.decode_video(path).data[[0, 3, 6]])


def test_on_frames_hears_of_each_kept_frame_and_can_stop_the_decode(tmp_path):
    path = _clip(tmp_path / "a.mp4")
    told = []
    media_io.decode_video(path, on_frames=lambda *frames: told.append(frames))
    assert told == [(48, 64, 1)] * 8
    told.clear()
    media_io.sample_video(path, lambda _meta: [7, 0, 3, 3], on_frames=lambda *frames: told.append(frames))
    assert told == [(48, 64, 1)] * 3

    def over_two(*frames):
        told.append(frames)
        if len(told) > 2:
            raise ValueError("over budget")

    told.clear()
    with pytest.raises(ValueError, match="over budget"):
        media_io.decode_video(path, on_frames=over_two)
    assert len(told) == 3


def test_decode_video_from_bytes(tmp_path):
    path = _clip(tmp_path / "a.mp4")
    with open(path, "rb") as f:
        data = f.read()
    assert torch.equal(media_io.decode_video(data).data, media_io.decode_video(path).data)


def test_rotated_video_comes_out_upright(tmp_path):
    upright = media_io.decode_video(_clip(tmp_path / "a.mp4")).data
    frames, metadata = media_io.decode_video(_clip(tmp_path / "r.mp4", rotation=90))
    # rotated 90 degrees counter-clockwise like torchcodec does, so width and height swap
    assert frames.shape == (8, 3, 64, 48)
    assert metadata["rotation"] == 90.0 and (metadata["width"], metadata["height"]) == (48, 64)
    assert torch.equal(frames, torch.rot90(upright, 1, dims=(2, 3)))


def test_encode_video_round_trips(tmp_path):
    frames = media_io.decode_video(_clip(tmp_path / "a.mp4")).data
    media_io.encode_video(frames, 4.0, tmp_path / "b.mp4")
    again, metadata = media_io.decode_video(tmp_path / "b.mp4")
    assert again.shape == frames.shape and metadata["average_fps"] == 4.0
    assert (again.float() - frames.float()).abs().mean() < 4


def test_decode_audio_resamples_and_mixes_like_ffmpeg():
    samples = media_io.decode_audio(_wav(channels=2), sample_rate=16000, num_channels=1)
    assert samples.sample_rate == 16000 and samples.data.shape == (1, 16000)
    assert samples.data.dtype == torch.float32
    # FFmpeg's stereo-to-mono matrix weights each channel by 1/sqrt(2)
    assert abs(samples.data.abs().max().item() - 0.5 * 2 ** 0.5) < 0.01
    native = media_io.decode_audio(_wav(channels=2))
    assert native.sample_rate == 8000 and native.data.shape == (2, 8000)


def test_audio_needing_no_conversion_is_read_by_libsndfile(monkeypatch):
    pytest.importorskip("soundfile")
    decoders = media_io._decoders().audio
    used = []
    for name in ("_pyav", "_soundfile"):
        backend = getattr(decoders, name)
        monkeypatch.setattr(backend, "decode", lambda *a, _n=name, _f=backend.decode: used.append(_n) or _f(*a))
    as_is = media_io.decode_audio(_wav(rate=16000, channels=1), sample_rate=16000, num_channels=1)
    converted = media_io.decode_audio(_wav(rate=8000, channels=1), sample_rate=16000, num_channels=1)
    assert used == ["_soundfile", "_pyav"]
    assert as_is.data.shape == (1, 16000) and converted.data.shape == (1, 16000)


def test_audio_duration():
    assert media_io.audio_duration(_wav(seconds=1.5)) == pytest.approx(1.5)


def test_audio_header_reads_the_rate_and_length_without_decoding(monkeypatch):
    decoders = media_io._decoders().audio
    for name in ("_pyav", "_soundfile"):
        monkeypatch.setattr(getattr(decoders, name), "decode", lambda *a: pytest.fail("decoded"))
    assert media_io.audio_header(_wav(rate=8000, seconds=1.5)) == (8000, pytest.approx(1.5))


def test_a_header_without_a_sample_count_gives_no_length(tmp_path):
    """libsndfile reports 2**63 - 1 frames for a FLAC whose STREAMINFO has no
    sample count (one written to a pipe); that is no length, not an enormous one."""
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "pipe.flac"
    sf.write(str(path), np.zeros(8000, np.int16), 8000, format="FLAC", subtype="PCM_16")
    data = bytearray(path.read_bytes())
    # STREAMINFO's 36-bit sample count ends at byte 25, then 16 bytes of MD5
    data[21] &= 0xF0
    data[22:42] = bytes(20)
    path.write_bytes(bytes(data))
    assert media_io.audio_header(str(path)) == (8000, None)


@pytest.mark.parametrize(("rate", "path"), [(8000, "_pyav"), (16000, "_soundfile")])
def test_a_bounded_decode_stops_at_max_seconds(monkeypatch, rate, path):
    """Through PyAV (resampled) and libsndfile (read as is) alike."""
    pytest.importorskip("soundfile")
    decoders = media_io._decoders().audio
    used = []
    backend = getattr(decoders, path)
    monkeypatch.setattr(backend, "decode", lambda *a, _f=backend.decode: used.append(path) or _f(*a))
    samples = media_io.decode_audio(
        _wav(rate=rate, seconds=5.0, channels=1), sample_rate=16000, num_channels=1, max_seconds=2.0,
    )
    assert used == [path] and samples.data.shape == (1, 32000)
    whole = media_io.decode_audio(_wav(rate=rate, seconds=1.0, channels=1), 16000, 1, max_seconds=2.0)
    assert whole.data.shape == (1, 16000)


def test_audio_only_libsndfile_reads_falls_through_to_soundfile():
    sf = pytest.importorskip("soundfile")
    if "MAT5" not in sf.available_formats():
        pytest.skip("this libsndfile cannot write MAT5")
    buf = io.BytesIO()
    sf.write(buf, np.zeros(800, np.float32), 8000, format="MAT5")
    samples = media_io.decode_audio(buf.getvalue(), sample_rate=16000, num_channels=1)
    assert samples.data.shape == (1, 1600)


def test_undecodable_audio_is_a_value_error():
    with pytest.raises(ValueError, match="cannot decode audio"):
        media_io.decode_audio(b"hello world" * 100)


def test_model_load_audio_and_video(tmp_path):
    from mstar.model.base import Model

    (tmp_path / "a.wav").write_bytes(_wav(channels=1))
    out = Model.load_audio(None, str(tmp_path / "a.wav"), "cpu")
    assert out.data.shape == (16000,) and out.data.dtype == torch.float32
    assert out.metadata == dict(sample_rate=16000, num_channels=1)

    out = Model.load_video(None, _clip(tmp_path / "a.mp4"), "cpu")
    assert out.data.shape == (8, 3, 48, 64) and out.data.dtype == torch.float32
    assert 0.0 <= out.data.min() and out.data.max() <= 1.0
    assert out.metadata["average_fps"] == 4.0


def test_vjepa2_samples_frames_uniformly(tmp_path):
    from mstar.model.vjepa2.vjepa2_model import VJepa2Model

    model = VJepa2Model.__new__(VJepa2Model)
    model.config = types.SimpleNamespace(frames_per_clip=4)
    out = model.load_video(_clip(tmp_path / "a.mp4"), "cpu")
    assert out.metadata["sampled_indices"] == [0, 2, 4, 6]
    assert out.metadata["original_num_frames"] == 8
    widths = _bar_widths((out.data * 255).round().to(torch.uint8))
    assert all(abs(n - 8 * (t + 1)) <= 2 for t, n in zip([0, 2, 4, 6], widths, strict=True))


def test_qwen3_omni_video_matches_qwen_omni_utils(tmp_path, monkeypatch):
    """Our path gives the same frames as fetch_video's torchcodec reader, run
    here against a fake torchcodec backed by media_io."""
    vp = pytest.importorskip("qwen_omni_utils.v2_5.vision_process")
    from mstar.model.qwen3_omni.qwen3_omni_model import _fetch_video_without_torchcodec

    path = _clip(tmp_path / "a.mp4", frames=24, fps=8)

    class VideoDecoder:
        def __init__(self, source, **kwargs):
            self.source = source
            self.metadata = types.SimpleNamespace(**media_io.video_metadata(source))

        def get_frames_at(self, indices):
            return types.SimpleNamespace(data=media_io.decode_video(self.source, indices=indices).data)

    fake = types.ModuleType("torchcodec.decoders")
    fake.VideoDecoder = VideoDecoder
    monkeypatch.setitem(sys.modules, "torchcodec", types.ModuleType("torchcodec"))
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", fake)
    monkeypatch.setattr(vp, "get_video_reader_backend", lambda: "torchcodec")
    expected, expected_fps = vp.fetch_video(
        {"video": path}, return_video_sample_fps=True, image_patch_size=14, return_video_metadata=False,
    )
    video, fps = _fetch_video_without_torchcodec(path, image_patch_size=14)
    assert fps == expected_fps
    assert video.shape == expected.shape and torch.equal(video, expected)
