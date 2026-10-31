"""Qwen3-Omni Code2Wav streaming with the real vocoder weights.

Streams real Talker codes (six TTS replies served by M*, in
``data/qwen3_omni_talker_codes.safetensors``) through the edge's
``ScheduledLeftContextChunkPolicy`` and ``Code2WavSubmodule``, and compares
the concatenated audio with one decode of each whole utterance, for the
default first-chunk ramp and for 25-frame first chunks.

Streaming can't match the full decode exactly: 25 frames of left context cut
into the vocoder's 72-frame attention window (24-28 dB SNR over an utterance).
What it must do is produce the full decode's length, chunk by chunk, and stay
about as close to it at chunk boundaries as mid-chunk. A boundary that drops
or repeats samples misaligns the rest of the stream.

The module never downloads weights: it skips unless CUDA is available and the
Qwen3-Omni checkpoint is in a local Hugging Face cache. Only the Code2Wav
weights (one shard) are loaded.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HF_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
CODES = Path(__file__).parent / "data" / "qwen3_omni_talker_codes.safetensors"
# 20 ms either side of a sample position
HALF_WINDOW = 480


def _find_cached_snapshot() -> Path | None:
    repo_dir = f"models--{HF_REPO.replace('/', '--')}"
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    for root in roots:
        snapshots = root / repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in snapshots.iterdir():
            index = snapshot / "model.safetensors.index.json"
            if not (snapshot / "config.json").is_file() or not index.is_file():
                continue
            shards = {
                shard for key, shard in json.loads(index.read_text())["weight_map"].items()
                if key.startswith("code2wav.")
            }
            if shards and all((snapshot / shard).is_file() for shard in shards):
                return snapshot
    return None


# Skip before importing the model: the GPU stack it needs isn't installed in
# CPU environments.
if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)
pytest.importorskip("flashinfer")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
SNAPSHOT = _find_cached_snapshot()
if SNAPSHOT is None:
    pytest.skip(f"{HF_REPO} Code2Wav weights are not in the local Hugging Face cache", allow_module_level=True)

from safetensors.torch import load_file  # noqa: E402

from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav  # noqa: E402
from mstar.model.qwen3_omni.config import Code2WavConfig, Qwen3OmniModelConfig  # noqa: E402
from mstar.model.qwen3_omni.submodules import Code2WavSubmodule  # noqa: E402
from mstar.model.submodule_base import InputMetadata, ModelInputsFromEngine  # noqa: E402
from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards  # noqa: E402
from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy  # noqa: E402
from mstar.streaming.stream_buffer import StreamBuffer, StreamChunkInfo  # noqa: E402


@pytest.fixture(scope="module")
def code2wav():
    """Built and loaded the way ``Qwen3OmniModel`` builds the Code2Wav node."""
    config = Code2WavConfig.from_dict(json.loads((SNAPSHOT / "config.json").read_text())["code2wav_config"])
    model = Qwen3OmniMoeCode2Wav(config)
    load_weights_from_hf_shards(repo_dir=SNAPSHOT, modules=[ModuleAndPrefix(model, prefix="code2wav")], device="cuda:0")
    benchmark = torch.backends.cudnn.benchmark
    # what _create_code2wav_submodule sets before the warm-up and capture
    torch.backends.cudnn.benchmark = True
    model.eval()
    eager_forward = type(model).forward.__get__(model)
    model.consolidate()
    # what the engine manager does with a disable_autocast node: device only
    model.to(device="cuda:0")
    try:
        yield model, eager_forward, config
    finally:
        torch.backends.cudnn.benchmark = benchmark


def stream(submodule: Code2WavSubmodule, frames: torch.Tensor) -> list[tuple[StreamChunkInfo, torch.Tensor]]:
    """Per-step window and audio for ``frames`` ``[T, 16]`` streamed through
    the edge's policy and the submodule."""
    cfg = submodule.config.code2wav
    buffer = StreamBuffer(
        request_id="r", edge_name="codec_tokens", from_partition="Talker",
        policy=ScheduledLeftContextChunkPolicy(
            schedule=cfg.codec_chunk_schedule, chunk=cfg.codec_chunk_frames, left_context=cfg.codec_left_context_frames,
        ),
    )
    for i, frame in enumerate(frames):
        buffer.pre_read_register(f"t{i}")
        buffer.put(f"t{i}", frame)
    buffer.signal_done()

    fwd_info = SimpleNamespace(request_id="r", rid_handle="r")
    chunks = []
    while buffer.has_chunk_ready():
        chunk = buffer.pop_chunk()
        meta = {"r": InputMetadata(stream_chunks={"codec_tokens": chunk.info})}
        key = submodule.cg_key_info("code2wav_chunk", {"r": fwd_info}, per_request_input_metadata=meta)
        engine_inputs = ModelInputsFromEngine(
            request_ids=["r"], per_request_info={}, per_request_input_metadata=meta, captured=key is None,
        )
        item = submodule.prepare_inputs(
            "code2wav_chunk", fwd_info, {"codec_tokens": [chunk.data["data"]]}, input_metadata=meta["r"],
        )
        with torch.no_grad():
            outputs = submodule.forward_batched(
                "code2wav_chunk", engine_inputs, **submodule.preprocess("code2wav_chunk", engine_inputs, [item]),
            )["r"]
        submodule.postprocess("r", fwd_info, outputs, inputs=item)
        chunks.append((chunk.info, outputs["audio_chunk"][0].cpu()))
    submodule.cleanup_request("r")
    return chunks


def window_snr_db(audio: torch.Tensor, reference: torch.Tensor, center: int) -> float:
    """The utterance's mean power over the error power within 20 ms of
    ``center``. Against the whole utterance's power, so a quiet stretch does
    not read as a dip."""
    lo, hi = max(center - HALF_WINDOW, 0), min(center + HALF_WINDOW, reference.numel(), audio.numel())
    signal = reference.double().pow(2).mean().item()
    error = (audio[lo:hi].double() - reference[lo:hi].double()).pow(2).mean().item()
    return 10 * math.log10(signal / max(error, 1e-9))


def boundary_and_mid_snr(
    steps: list[tuple[StreamChunkInfo, torch.Tensor]], audio: torch.Tensor, reference: torch.Tensor,
):
    """SNR around every chunk boundary of the stream, and halfway into every
    window long enough to have a middle."""
    boundaries, middles = [], []
    end = min(audio.numel(), reference.numel()) - HALF_WINDOW
    position = 0
    for info, chunk_audio in steps:
        if info.context_items and position < end:
            boundaries.append(window_snr_db(audio, reference, position))
        middle = position + chunk_audio.numel() // 2
        if info.context_items and middle < end and chunk_audio.numel() >= 4 * HALF_WINDOW:
            middles.append(window_snr_db(audio, reference, middle))
        position += chunk_audio.numel()
    return boundaries, middles


@pytest.mark.parametrize("schedule, left_context", [
    pytest.param((12, 13), 25, id="ramp-12-13-25"),
    pytest.param((4, 8, 13), 25, id="ramp-4-8-13-25"),
    pytest.param((4, 8, 16), 25, id="ramp-4-8-16-25"),
    pytest.param((4, 5, 6, 8, 10, 13, 16, 20), 25, id="ramp-4-5-6-8-10-13-16-20"),
    pytest.param((8, 10, 13, 16, 20), 25, id="early-audio-8-10-13-16-20"),
    pytest.param((), 25, id="first-chunk-25"),
    pytest.param((), 50, id="first-chunk-25-context-50"),
])
def test_real_codes_stream_to_the_full_decode_length_without_boundary_dips(code2wav, schedule, left_context):
    model, eager_forward, config = code2wav
    config = replace(config, codec_chunk_schedule=schedule, codec_left_context_frames=left_context)
    submodule = Code2WavSubmodule(model, Qwen3OmniModelConfig(code2wav=config))
    upsample = submodule.total_upsample
    all_boundaries, all_middles = [], []
    for name, frames in sorted(load_file(str(CODES)).items()):
        frames = frames.long().cuda()
        num_frames = frames.shape[0]
        with torch.no_grad():
            full = eager_forward(frames.t()[None], torch.arange(num_frames, device="cuda")[None])[0, 0]
        reference = (full.clamp(-1, 1) * 32767).to(torch.int16).cpu()
        tail = num_frames * upsample - reference.numel()

        steps = stream(submodule, frames)
        audio = torch.cat([chunk_audio for _, chunk_audio in steps])
        boundaries, middles = boundary_and_mid_snr(steps, audio, reference)
        print(
            f"{name}: samples per chunk {[a.numel() for _, a in steps]}, total {audio.numel()} "
            f"(full decode {reference.numel()}), boundary SNR {[round(s, 1) for s in boundaries]} dB, "
            f"mid-chunk SNR {[round(s, 1) for s in middles]} dB, "
            f"utterance SNR {utterance_snr_db(audio, reference):.1f} dB"
        )
        assert tail == 555
        assert audio.numel() == reference.numel() == num_frames * upsample - tail
        # the first window ends ``tail`` short; every later one carries its new frames
        expected = [
            (info.num_items - info.context_items) * upsample - (0 if info.context_items else tail) for info, _ in steps
        ]
        assert [a.numel() for _, a in steps] == [max(n, 0) for n in expected]
        all_boundaries += boundaries
        all_middles += middles

    boundary_median = statistics.median(all_boundaries)
    middle_median = statistics.median(all_middles)
    print(
        f"{schedule}/{left_context}: SNR vs full decode: boundary median {boundary_median:.1f} dB "
        f"(min {min(all_boundaries):.1f}), mid-chunk median {middle_median:.1f} dB (min {min(all_middles):.1f})"
    )
    assert len(all_boundaries) >= 10 and len(all_middles) >= 5
    # A boundary starts a window's new frames on the least attention history
    # (25 frames), so it runs a few dB under mid-chunk (14-56 dB vs 20-49 dB
    # on an H200). A dropped or repeated sample there reads near 0 dB, and so
    # does everything after it.
    assert boundary_median >= middle_median - 12
    assert min(all_boundaries) >= 10


def utterance_snr_db(audio: torch.Tensor, reference: torch.Tensor) -> float:
    error = (audio.double() - reference.double()).pow(2).mean().item()
    return 10 * math.log10(reference.double().pow(2).mean().item() / max(error, 1e-9))
