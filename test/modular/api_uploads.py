"""A request as the API hands it to a model: uploaded files through the data worker.

A count test that shapes its own tensors tests the shape it chose, not the one
the API sends: Pi0.5's took its cameras as one stacked tensor, while every
uploaded file arrives as its own entry of ``<modality>_inputs``.
"""

from __future__ import annotations

import functools
import io
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

import av
import numpy as np
import pytest
import soundfile
import torch
import torchaudio
from test_kv_prefix_request_keys import _StubTensorManager, _worker
from torchvision.io import encode_png

from mstar.api_server.entrypoint import _decode_ws_files
from mstar.api_server.request_types import PreprocessInput
from mstar.model.multimodal import PromptPart


class _KeptTensors(_StubTensorManager):
    """Keeps the tensors the worker would have sent the first node."""

    def store_and_return_tensor_info(self, request_id, tensors, **kwargs):
        self.tensors = tensors
        return {}


class SoundfileAudioDecoder:
    """torchcodec's ``AudioDecoder`` for a WAV upload, where torchcodec cannot load FFmpeg's shared libraries."""

    def __init__(self, source: str, sample_rate: int, num_channels: int):
        data, rate = soundfile.read(source, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        if num_channels == 1:
            wav = wav.mean(dim=0, keepdim=True)
        self._data = torchaudio.functional.resample(wav, rate, sample_rate)

    def get_all_samples(self) -> SimpleNamespace:
        return SimpleNamespace(data=self._data)


class PyAVVideoDecoder:
    """torchcodec's ``VideoDecoder`` over PyAV's own FFmpeg, where torchcodec cannot load FFmpeg's shared libraries."""

    def __init__(self, source: str, **options):
        with av.open(source) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        self._frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        self.metadata = SimpleNamespace(num_frames=len(frames), average_fps=fps)

    def __len__(self) -> int:
        return len(self._frames)

    def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
        return SimpleNamespace(data=self._frames[indices])


@functools.cache
def _torchcodec_decodes() -> bool:
    from torchcodec.decoders import AudioDecoder

    try:
        AudioDecoder(wav(0.1, 16000)[1])
    except RuntimeError:
        return False
    return True


def decode_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have `Model.load_audio` and the video loaders decode with torchcodec, or with the stand-ins where it cannot.

    torchcodec raises RuntimeError, not ImportError, on a host without FFmpeg's
    shared libraries; the stand-ins keep the rest of the upload path, and the
    count, tested on such a host.
    """
    if not _torchcodec_decodes():
        monkeypatch.setattr("torchcodec.decoders.AudioDecoder", SoundfileAudioDecoder)
        monkeypatch.setattr("torchcodec.decoders.VideoDecoder", PyAVVideoDecoder)


def png(height: int, width: int) -> tuple[str, bytes]:
    return "frame.png", encode_png(torch.randint(0, 256, (3, height, width), dtype=torch.uint8)).numpy().tobytes()


def wav(seconds: float, sample_rate: int, channels: int = 1) -> tuple[str, bytes]:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    data = io.BytesIO()
    soundfile.write(data, np.stack([tone] * channels, axis=1), sample_rate, format="WAV", subtype="PCM_16")
    return "clip.wav", data.getvalue()


def mp4(frames: int, height: int, width: int, tmp_path: Path) -> tuple[str, bytes]:
    # PyAV, not torchcodec's encoder: it ships its own FFmpeg, as the decode fallback does
    path = tmp_path / f"clip{frames}x{height}x{width}.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.height, stream.width, stream.pix_fmt = height, width, "yuv420p"
        for _ in range(frames):
            rgb = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8).numpy()
            container.mux(stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")))
        container.mux(stream.encode())
    return "clip.mp4", path.read_bytes()


def preprocess(
    model, deployment: dict | None, text: str | None, files: list[tuple[str, bytes]],
    output_modalities: list[str], model_kwargs: dict, upload_dir: Path,
) -> tuple[dict, dict]:
    """(the tensors, the model_kwargs) the data worker sends on for one upload."""
    file_paths, parts = _decode_ws_files([{"name": name, "data": data} for name, data in files], upload_dir)
    if text:
        parts.append(PromptPart(modality="text", text=text))
    worker = _worker(model, deployment)
    worker.tensor_manager = _KeptTensors()
    worker._process_input(PreprocessInput(
        request_id="r", text=text, file_paths=file_paths or None,
        input_modalities=[part.modality for part in parts], output_modalities=output_modalities,
        model_kwargs=model_kwargs, prompt_parts=parts,
    ))
    return worker.tensor_manager.tensors, worker.communicator.sent[-1].body.model_kwargs
