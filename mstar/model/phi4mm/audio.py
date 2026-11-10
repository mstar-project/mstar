"""Phi-4-multimodal's audio: the intake that turns a waveform into the
reference's log-mel features, and the ``audio_encoder`` node.

The intake runs where the prompt is processed and reproduces
``Phi4MMAudioFeatureExtractor`` in numpy, op for op, so the features match
the reference's bit for bit. Each clip fills ``ceil(frames / 8)`` audio pads;
its mel frame count rides the walk's ``MEDIA_LAYOUT`` as the run's grid, so
the node lays out its batch from host values.
"""

from __future__ import annotations

from collections.abc import Collection
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.multimodal import MEDIA_LAYOUT, MediaEncoder, media_grids
from mstar.model.multimodal_decoder import PreparedMedia
from mstar.model.phi4mm.components.audio import (
    SPEECH,
    VISION,
    AudioBatch,
    Phi4MMAudioConfig,
    Phi4MMAudioTower,
    audio_batch,
)
from mstar.model.phi4mm.config import AUDIO_PAD_ID
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch

AUDIO_FEATURES = "audio_features"
AUDIO_EMBEDS = "audio_embeds"
# the media walk of a prompt with an image and audio, whose audio rows take
# the ``vision`` projection
VISION_AUDIO_WALK = "prefill_vision_audio"

SAMPLE_RATE = 16_000
# the feature extractor's 8 kHz path: an 8 kHz spectrum in the low half of a
# 16 kHz one's bins
NARROWBAND_RATE = 8_000
PREEMPHASIS = 0.97


def audio_encoder(pad_id: int = AUDIO_PAD_ID) -> MediaEncoder:
    """The ``audio_encoder`` node: a clip's log-mel frames in, one LLM row
    per audio pad out."""
    return MediaEncoder("audio", "audio_encoder", (AUDIO_FEATURES,), AUDIO_EMBEDS, (pad_id,))


@cache
def _mel_filters() -> np.ndarray:
    """SpeechLib's 80 triangular filters over a 512-point FFT at 16 kHz, up
    to 7690 Hz, ``(257, 80)``: ``speechlib_mel(16000, 512, 80, fmax=7690).T``."""
    sample_rate, n_fft, n_mels, fmin, fmax = 16000, 512, 80, 0.0, 7690.0

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    klo = int(fmin * n_fft / sample_rate + 0.5) + 1
    khi = max(int(fmax * n_fft / sample_rate + 0.5), klo)
    mlo, mhi = mel(fmin), mel(fmax)
    centers = np.linspace(mlo, mhi, n_mels + 2)
    width = (mhi - mlo) / (n_mels + 1)
    matrix = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = centers[m], centers[m + 1], centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / width
    return matrix.T


def log_mel(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """``(frames, 80)`` float32 log-mel features of a mono ``wav``, as the
    reference extracts them: 25 ms Hamming frames every 10 ms (a tail short
    of a hop dropped), preemphasis within each frame, x32768, power
    spectrum, mel, log clipped at 1. 8 kHz audio is framed at its own rate
    and its spectrum zero-filled above 4 kHz."""
    if sample_rate == SAMPLE_RATE:
        n_fft, win_length, hop_length = 512, 400, 160
    elif sample_rate == NARROWBAND_RATE:
        n_fft, win_length, hop_length = 256, 200, 80
    else:
        raise ValueError(f"Phi-4-multimodal takes audio at 16 or 8 kHz, not {sample_rate} Hz")
    frames = (wav.shape[0] - win_length) // hop_length + 1
    if frames < 1:
        raise ValueError(
            f"the audio is {wav.shape[0] / sample_rate * 1000:.1f} ms long; "
            f"Phi-4-multimodal needs at least {win_length / sample_rate * 1000:.0f} ms"
        )
    y = np.lib.stride_tricks.sliding_window_view(wav, win_length)[::hop_length][:frames].astype(np.float32)
    prev = np.roll(y, 1, axis=1)
    prev[:, 0] = prev[:, 1]
    y = (y - PREEMPHASIS * prev) * 32768
    spec = np.fft.rfft(np.hamming(win_length) * y, n=n_fft, axis=1).astype(np.complex64)
    if sample_rate == NARROWBAND_RATE:
        # the Nyquist bin goes too, as the reference's fill does
        spec = np.concatenate((spec[:, :-1], np.zeros((frames, n_fft // 2 + 1))), axis=1)
    power = np.abs(spec).astype(np.float32) ** 2
    return np.log(np.clip(power.dot(_mel_filters()), 1.0, None)).astype(np.float32)


def audio_intake(
    waveform: torch.Tensor, metadata: dict, config: Phi4MMAudioConfig | None = None,
) -> PreparedMedia:
    """A clip as the encoder reads it: its log-mel frames under
    ``audio_features``, filling one run of ``ceil(frames / 8)`` pads, with
    ``[frames]`` as the run's grid.

    ``waveform`` is mono float32 at ``metadata["sample_rate"]``, as
    `Model.load_audio` decodes it (16 kHz by default). The reference's own
    resampling of other rates is not reproduced: the loader resamples them."""
    config = config or Phi4MMAudioConfig()
    wav = waveform.squeeze()
    if wav.dim() != 1:
        raise ValueError(f"Phi-4-multimodal takes mono audio; got a waveform of shape {tuple(waveform.shape)}")
    mel = log_mel(wav.to(torch.float32).cpu().numpy(), int(metadata.get("sample_rate", SAMPLE_RATE)))
    frames = mel.shape[0]
    return PreparedMedia({AUDIO_FEATURES: torch.from_numpy(mel)}, [config.tokens(frames)], [[frames]])


class Phi4MMAudioEncoderSubmodule(NodeSubmodule):
    """The conformer and a projection over a step's clips, stateless.

    Every clip of every request in the step is subsampled in one padded
    batch, then cut into attention windows that run as one batch of rows
    (`plan_windows`), each masked past its own frames. The rows come out
    clip by clip, so each request's ``audio_embeds`` fill its audio pads in
    prompt order. A prompt that also holds an image (`vision_walks`) takes
    the ``vision`` projection, any other the ``speech`` one.
    """

    # eager for now; compiled, every new clip length would recompile
    disable_torch_compile = True
    MAX_BATCH_SIZE = 16
    # attention windows a step batches across requests; one request's clip may hold
    # more (a 30-minute clip is 45), and runs alone
    MAX_WINDOWS = 32

    def __init__(self, tower: Phi4MMAudioTower, vision_walks: Collection[str] = (VISION_AUDIO_WALK,)):
        super().__init__()
        self.tower = tower
        self.config = tower.config
        self.vision_walks = frozenset(vision_walks)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        clips = inputs[AUDIO_FEATURES]
        frames = [grid[0] for grid in media_grids(fwd_info.step_metadata[MEDIA_LAYOUT], AUDIO_EMBEDS)]
        if [int(c.shape[0]) for c in clips] != frames:
            raise ValueError(f"audio features of {[c.shape[0] for c in clips]} frames for a layout of {frames}")
        tokens = [self.config.tokens(f) for f in frames]
        return NodeInputs(
            # one tensor a request: its clips end to end, split again by `frames`
            tensor_inputs={AUDIO_FEATURES: clips[0] if len(clips) == 1 else torch.cat(clips)},
            kwargs={"frames": frames, "windows": sum(self.config.windows(n) for n in tokens)},
            input_seq_len=sum(tokens),
        )

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return sum(inp.kwargs["windows"] for inp in model_inputs) <= self.MAX_WINDOWS

    def max_batch_size(self, graph_walk: str) -> int:
        return self.MAX_BATCH_SIZE

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        clips = [
            clip for inp in inputs
            for clip in inp.tensor_inputs[AUDIO_FEATURES].split(inp.kwargs["frames"])
        ]
        dtype = self.tower.encoder.rel_bias.weight.dtype
        return {
            "batch": audio_batch(clips, self.config, self.get_device(), dtype),
            "rows_per_request": [inp.input_seq_len for inp in inputs],
        }

    def projection_mode(self, graph_walk: str) -> str:
        return VISION if graph_walk in self.vision_walks else SPEECH

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        batch: AudioBatch,
        rows_per_request: list[int],
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        embeds = self.tower.project(self.tower.encode(batch), self.projection_mode(graph_walk))
        return {
            rid: {AUDIO_EMBEDS: [rows]}
            for rid, rows in zip(engine_inputs.request_ids, embeds.split(rows_per_request), strict=True)
        }

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]
