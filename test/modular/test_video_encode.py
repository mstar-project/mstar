"""``encode_mp4`` converts frames a few ahead of the encoder on a thread of its own.

The conversion hands the encoder the frames ``stream.encode`` would have made
(checked through ``rawvideo``, which stores them as they come). libx264 at its
defaults writes different bytes for one clip from run to run, so the mp4 bytes
are compared under ``ultrafast``, which reproduces them: for the writer itself
and through each model's ``postprocess``.
"""

from __future__ import annotations

import io
from fractions import Fraction

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")

from mstar.model.components.video_encode import converted_frames, encode_mp4  # noqa: E402

# 30000/1001 exactly, as encode_mp4 writes it
NTSC = 30000 / 1001
# LTX-2.5's settings, and Cosmos3's PyAV path at its default preset
ULTRAFAST = {"crf": "18", "preset": "ultrafast", "threads": "0"}


def _plain(frames: np.ndarray, fps, options, pad_even: bool = False) -> bytes:
    """The models' writer before ``encode_mp4``: their own stream setup and a plain
    ``stream.encode`` loop."""
    if pad_even and (frames.shape[1] % 2 or frames.shape[2] % 2):
        frames = np.pad(frames, ((0, 0), (0, frames.shape[1] % 2), (0, frames.shape[2] % 2), (0, 0)), mode="edge")
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
    stream.height, stream.width = frames.shape[1], frames.shape[2]
    stream.pix_fmt = "yuv420p"
    if options:
        stream.options = options
    for frame in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buffer.getvalue()


def _clip(frames: int, height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(frames * height + width)
    # a moving gradient plus noise, so the encoder has motion to search
    y, x = np.mgrid[:height, :width]
    clip = np.stack([(x + 3 * t + y // 2) % 256 for t in range(frames)])[..., None].repeat(3, -1)
    noise = rng.integers(0, 24, size=clip.shape)
    return ((clip + noise) % 256).astype(np.uint8)


def _x264_settings(data: bytes) -> bytes:
    """The settings libx264 writes into the stream (preset, crf, threads, lookahead...)."""
    start = data.index(b"x264 - core")
    return data[start:data.index(b"\x00", start)]


def _stored_frames(frames: np.ndarray, height: int, width: int, converted: bool) -> list[bytes]:
    """The frames a ``rawvideo`` yuv420p encoder of ``height`` x ``width`` stores, from
    ``converted_frames`` or from the RGB frames as the plain loop hands them over."""
    context = av.CodecContext.create("rawvideo", "w")
    context.width, context.height, context.pix_fmt = width, height, "yuv420p"
    context.time_base = Fraction(1, 24)
    if converted:
        source = converted_frames(context, frames)
    else:
        source = (av.VideoFrame.from_ndarray(frame, format="rgb24") for frame in frames)
    return [bytes(packet) for frame in source for packet in context.encode(frame)]


@pytest.mark.parametrize(("height", "width"), [(48, 64), (32, 48), (66, 98)])
def test_the_encoder_gets_the_frames_its_own_conversion_makes(height, width):
    clip = _clip(9, 48, 64)  # (32, 48) and (66, 98) make the conversion scale too
    stored = _stored_frames(clip, height, width, converted=True)
    assert len(stored) == len(clip)
    assert stored == _stored_frames(clip, height, width, converted=False)


@pytest.mark.parametrize("fps", [24, NTSC, 12.0])
@pytest.mark.parametrize(("frames", "height", "width"), [(1, 16, 16), (9, 48, 64), (17, 66, 98)])
def test_encode_mp4_writes_the_plain_loops_bytes(frames, height, width, fps):
    clip = _clip(frames, height, width)
    new = encode_mp4(clip, fps, options=ULTRAFAST)
    assert new == _plain(clip, fps, ULTRAFAST)


def test_odd_sizes_pad_to_even_ones():
    clip = _clip(5, 47, 63)
    new = encode_mp4(clip, 24, options=ULTRAFAST, pad_even=True)
    assert new == _plain(clip, 24, ULTRAFAST, pad_even=True)
    with av.open(io.BytesIO(new)) as container:
        assert (container.streams.video[0].width, container.streams.video[0].height) == (64, 48)


def test_the_conversion_runs_a_bounded_few_frames_ahead():
    clip = _clip(12, 16, 16)
    read = []

    def frames():
        for i, frame in enumerate(clip):
            read.append(i)
            yield frame

    context = av.CodecContext.create("rawvideo", "w")
    context.width, context.height, context.pix_fmt = 16, 16, "yuv420p"
    converted = converted_frames(context, frames(), lookahead=3)
    for taken in range(1, 6):
        next(converted)
        assert len(read) == taken + 3
    assert len(list(converted)) == len(clip) - 5


def test_a_conversion_failure_reaches_the_caller():
    with pytest.raises(ValueError):
        encode_mp4(np.zeros((2, 16, 16, 3), dtype=np.float32), 24)


# Each model's postprocess against its old writer.

def _wan22():
    from mstar.model.wan22.config import Wan22Config
    from mstar.model.wan22.wan22_model import Wan22Model

    model = object.__new__(Wan22Model)
    model.config = Wan22Config()
    return model


def test_wan22_writes_the_old_bytes():
    clip = _clip(9, 48, 64)
    data = _wan22().postprocess(torch.from_numpy(clip), "video", {"fps": NTSC})
    assert data == _plain(clip, NTSC, ULTRAFAST)
    assert b" crf=18.0 " in _x264_settings(data)


def test_ltx2_5_writes_the_old_bytes():
    from mstar.model.ltx2_5.config import LTX25Config
    from mstar.model.ltx2_5.ltx2_5_model import LTX25Model

    model = LTX25Model(model_path_hf="test/ltx2_5", skip_weight_loading=True)
    model.set_config(LTX25Config())
    clip = _clip(9, 48, 64)
    data = model.postprocess(torch.from_numpy(clip), "video", {"fps": NTSC})
    assert data == _plain(clip, NTSC, ULTRAFAST)


@pytest.mark.parametrize(("height", "width"), [(48, 64), (47, 63)])
def test_cosmos3_pyav_path_writes_an_exact_ntsc_rate(monkeypatch, height, width):
    import sys

    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model

    class _Broken:  # torchcodec without FFmpeg libraries: the PyAV fallback
        def __getattr__(self, name):
            raise RuntimeError("Could not load libtorchcodec")

    monkeypatch.setitem(sys.modules, "torchcodec", _Broken())
    monkeypatch.setitem(sys.modules, "torchcodec.encoders", None)
    monkeypatch.delenv("COSMOS3_X264_PRESET", raising=False)
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    clip = _clip(9, height, width)
    data = model.postprocess(torch.from_numpy(clip).permute(0, 3, 1, 2), "video", {"fps": NTSC})
    assert data == _plain(clip, NTSC, ULTRAFAST, pad_even=True)
