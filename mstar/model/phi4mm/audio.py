"""Phi-4-multimodal's audio: the intake that turns a waveform into the
reference's log-mel features, and the ``audio_encoder`` node.

The intake runs where the prompt is processed and reproduces
``Phi4MMAudioFeatureExtractor`` in numpy, op for op, so the features match
the reference's bit for bit. Each clip fills ``ceil(frames / 8)`` audio pads;
its mel frame count rides the walk's ``MEDIA_LAYOUT`` as the run's grid, so
the node lays out its batch from host values.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from contextlib import nullcontext
from functools import cache, partial
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
)
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


def audio_encoder() -> MediaEncoder:
    """The ``audio_encoder`` node: a clip's log-mel frames in, one LLM row
    per token of its ``MEDIA_LAYOUT`` runs out."""
    return MediaEncoder("audio", "audio_encoder", (AUDIO_FEATURES,), AUDIO_EMBEDS)


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


# Captured shapes. The subsampling runs per clip, by the step's longest mel
# (10, 20, 41 s); the blocks per window, by its width in encoder frames (10, 20,
# 40 s; a clip past one window runs at 500).
MEL_BUCKETS = (1024, 2048, 4096)
WIDTH_BUCKETS = (128, 256, 500)
CLIP_BATCH_SIZES = (1, 2, 4, 8, 16)
WINDOW_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
# SDPA sends the dense T5 bias to cuDNN, which builds a graph per new shape: ~0.5 s of
# host time a step when every eager step's clips differ. Captured blocks keep cuDNN,
# whose builds happen once at capture and which runs 1.2-3.7x faster in the graph.
_EAGER_ATTENTION_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
# mel frames per subsampling replay, which bounds its activations (the first
# conv's output is 1024 channels x half the frames x 40)
MAX_CAPTURE_MEL_FRAMES = 32 * 1024


def _subsample_region(bucket: int) -> str:
    return f"subsample_T{bucket}"


def _blocks_region(width: int) -> str:
    return f"blocks_W{width}"


def _clip_batch_sizes(bucket: int) -> list[int]:
    return [bs for bs in CLIP_BATCH_SIZES if bs * bucket <= MAX_CAPTURE_MEL_FRAMES]


class StepRegions(NamedTuple):
    """The captured regions a step replays, chosen with its buckets in
    ``preprocess``; None runs that part eagerly at the step's own shape."""

    subsample: str | None
    blocks: str | None


class Phi4MMAudioEncoderSubmodule(NodeSubmodule):
    """The conformer and a projection over a step's clips, stateless.

    Every clip of every request in the step is subsampled in one padded
    batch, then cut into attention windows that run as one batch of rows
    (`plan_windows`), each masked past its own frames. The rows come out
    clip by clip, so each request's ``audio_embeds`` fill its audio pads in
    prompt order. A prompt that also holds an image (`vision_walks`) takes
    the ``vision`` projection, any other the ``speech`` one.

    The subsampling replays a graph per (clips, mel bucket) and the blocks
    per (windows, width bucket); padding is exact in both (zeroed before
    each conv, masked in attention). A step past every bucket runs that part
    eagerly at its own shape. The projection and the row gather stay eager.
    """

    # the captured regions carry the speed; compiled, every new clip length would recompile
    disable_torch_compile = True
    # the tower runs in its own bf16, as the reference does: autocast would run
    # every LayerNorm in fp32 behind a cast copy each way
    disable_autocast = True
    MAX_BATCH_SIZE = 16
    # attention windows a step batches across requests; one request's clip may hold
    # more (a 30-minute clip is 45), and runs alone
    MAX_WINDOWS = max(WINDOW_BATCH_SIZES)

    def __init__(
        self, tower: Phi4MMAudioTower, vision_walks: Collection[str] = (VISION_AUDIO_WALK,),
    ):
        super().__init__()
        self.tower = tower
        self.config = tower.config
        self.vision_walks = frozenset(vision_walks)

    def _dtype(self) -> torch.dtype:
        return self.tower.encoder.rel_bias.weight.dtype

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

    # -- capture ---------------------------------------------------------------

    def _capture_subsample(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        s = inp.static_inputs
        # padding rows replay zeroed: one frame keeps them a well-formed clip
        return {"x": self.tower.encoder.subsample(s["mel"], s["mel_lengths"].clamp(min=1))}

    def _capture_blocks(self, inp: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        s = inp.static_inputs
        # padding windows replay zeroed: one key keeps their softmax over a real key
        return {"x": self.tower.encoder.blocks(s["windows"], s["key_lengths"].clamp(min=1))}

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs: Any,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """The subsampling by mel bucket and the blocks by width bucket; the
        T5 bias and the masks are built in the graph from the length buffers."""
        del autocast_dtype, tp_world_size, kwargs
        dtype, c = self._dtype(), self.config

        def mel(shape: PiecewiseCaptureShape, bucket: int) -> dict[str, torch.Tensor]:
            return {
                "mel": torch.zeros(shape.bs, bucket, c.num_mel_bins, dtype=dtype, device=device),
                "mel_lengths": torch.zeros(shape.bs, dtype=torch.long, device=device),
            }

        def windows(shape: PiecewiseCaptureShape, width: int) -> dict[str, torch.Tensor]:
            return {
                "windows": torch.zeros(shape.bs, width, c.hidden_size, dtype=dtype, device=device),
                "key_lengths": torch.zeros(shape.bs, dtype=torch.long, device=device),
            }

        configs: dict[str, PiecewiseAcceleratorGraphConfig] = {
            _subsample_region(bucket): PiecewiseBatchedConfig(
                capture_fn=self._capture_subsample,
                make_static_inputs=partial(mel, bucket=bucket),
                seq_len=bucket,
                capture_batch_sizes=_clip_batch_sizes(bucket),
            )
            for bucket in MEL_BUCKETS
        }
        configs.update({
            _blocks_region(width): PiecewiseBatchedConfig(
                capture_fn=self._capture_blocks,
                make_static_inputs=partial(windows, width=width),
                seq_len=width,
                capture_batch_sizes=list(WINDOW_BATCH_SIZES),
            )
            for width in WIDTH_BUCKETS
        })
        return configs

    @staticmethod
    def _region(runners: dict, labels: Iterable[tuple[int, str]], needed: int, rows: int) -> tuple[int, str] | None:
        """The smallest ``(bucket, label)`` holding ``needed`` whose runner
        replays ``rows`` rows."""
        return next(
            ((bucket, label) for bucket, label in labels
             if bucket >= needed and label in runners and runners[label].can_run(rows)),
            None,
        )

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
        runners = engine_inputs.piecewise_runners
        longest = max(int(clip.shape[0]) for clip in clips)
        tokens = self.config.tokens(longest)
        windows = sum(self.config.windows(self.config.tokens(int(clip.shape[0]))) for clip in clips)
        subsample = self._region(runners, ((b, _subsample_region(b)) for b in MEL_BUCKETS), longest, len(clips))
        blocks = self._region(
            runners, ((w, _blocks_region(w)) for w in WIDTH_BUCKETS), min(tokens, self.config.window), windows,
        )
        batch = audio_batch(
            clips, self.config, self.get_device(), self._dtype(),
            mel_frames=None if subsample is None else subsample[0],
            width=None if blocks is None else blocks[0],
        )
        return {
            "batch": batch,
            "regions": StepRegions(
                None if subsample is None else subsample[1], None if blocks is None else blocks[1],
            ),
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
        regions: StepRegions = StepRegions(None, None),
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        runners = engine_inputs.piecewise_runners

        def subsample(mel: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            replay = runners[regions.subsample].run(
                static_inputs={"mel": mel, "mel_lengths": lengths}, real_bs=mel.shape[0],
            )
            return replay.get_view("x")

        def blocks(windows: torch.Tensor, key_lengths: torch.Tensor | None) -> torch.Tensor:
            # a bucketed batch always carries its key lengths (`audio_batch`); without
            # them the replay would read the last step's
            assert key_lengths is not None
            replay = runners[regions.blocks].run(
                static_inputs={"windows": windows, "key_lengths": key_lengths}, real_bs=windows.shape[0],
            )
            return replay.get_view("x")

        with sdpa_kernel(_EAGER_ATTENTION_BACKENDS) if regions.blocks is None else nullcontext():
            encoded = self.tower.encode(
                batch,
                subsample=subsample if regions.subsample is not None else None,
                blocks=blocks if regions.blocks is not None else None,
            )
        embeds = self.tower.project(encoded, self.projection_mode(graph_walk))
        return {
            rid: {AUDIO_EMBEDS: [rows]}
            for rid, rows in zip(engine_inputs.request_ids, embeds.split(rows_per_request), strict=True)
        }

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]
