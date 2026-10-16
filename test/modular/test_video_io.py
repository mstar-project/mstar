"""PyAV video decode, the fallback where torchcodec cannot load (its PyPI
wheels link CUDA libraries, so on ROCm the import fails)."""

import sys

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")


def _clip(path, frames=8, fps=4):
    # a bar that widens by 8 columns a frame, so frame order is checkable
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for t in range(frames):
            img = np.zeros((48, 64, 3), np.uint8)
            img[:, : 8 * (t + 1)] = 200
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def test_decode_video_pyav(tmp_path):
    from mstar.model.components.video_io import decode_video_pyav

    _clip(tmp_path / "a.mp4")
    frames, metadata = decode_video_pyav(str(tmp_path / "a.mp4"))
    assert frames.shape == (8, 3, 48, 64) and frames.dtype == torch.uint8
    assert metadata == dict(num_frames=8, average_fps=4.0, duration_seconds=2.0, height=48, width=64)
    lit = (frames.float().mean(dim=(1, 2)) > 100).sum(dim=-1)
    # lossy codec: the bar's edge may blur by a column
    assert all(abs(n - 8 * (t + 1)) <= 2 for t, n in enumerate(lit.tolist()))


def test_model_load_video_without_torchcodec(tmp_path, monkeypatch):
    from mstar.model.base import Model

    _clip(tmp_path / "a.mp4")
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    out = Model.load_video(None, str(tmp_path / "a.mp4"), "cpu")
    assert out.data.shape == (8, 3, 48, 64) and out.data.dtype == torch.float32
    assert 0.0 <= out.data.min() and out.data.max() <= 1.0
    assert out.metadata["average_fps"] == 4.0
