"""Whisper-family log-mel front end, in torch.

Whisper, Qwen3-ASR's AuT and the Higgs audio tower all consume the same
features: a 16 kHz waveform, 25 ms Hann-windowed STFT at a 10 ms hop, a
Slaney-scaled mel filterbank (80 or 128 bins), ``log10``, an 8-unit dynamic
range clamp under the clip's maximum, then ``(x + 4) / 4``. This module
reproduces HF's ``WhisperFeatureExtractor`` torch path to ~1e-5 so a
natively served encoder sees the features its HF oracle was tested with,
without ``transformers`` in the serving path. It runs wherever the waveform
is, on the CPU of the API-server data worker or on the GPU inside an
encoder's forward (every op is fixed-shape for a fixed clip length, so it
captures into a CUDA graph).

NVIDIA's FastConformer models (Canary, Parakeet, Nemotron ASR) take NeMo's
recipe instead, ``NemoLogMel``: preemphasis, natural log, and per-bin
standardization over the clip.
"""
from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mstar.api_server.media_io import check_resample_rate
from mstar.engine.fp32_precision import Fp32Precision, fp32_precision_scope
from mstar.model.components.dft import stft_as_real
from mstar.utils.h2d import to_device_nonblocking


def _hertz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    """Slaney mel scale (linear below 1 kHz, logarithmic above), as in
    librosa and HF ``audio_utils.hertz_to_mel(mel_scale="slaney")``."""
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * np.asarray(freq, dtype=np.float64) / 200.0
    high = np.asarray(freq, dtype=np.float64) >= min_log_hertz
    mels = np.where(
        high,
        min_log_mel + np.log(np.maximum(np.asarray(freq, dtype=np.float64), 1e-12) / min_log_hertz) * logstep,
        mels,
    )
    return mels


def _mel_to_hertz(mels: np.ndarray) -> np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    return np.where(mels >= min_log_mel, min_log_hertz * np.exp(logstep * (mels - min_log_mel)), freq)


def slaney_mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    sampling_rate: int,
    min_frequency: float = 0.0,
    max_frequency: float = 8000.0,
) -> np.ndarray:
    """Area-normalized triangular mel filters, ``(num_mel_filters,
    num_frequency_bins)`` in float64 — the transpose of HF's layout, so the
    forward is one ``filters @ power_spectrogram`` matmul."""
    fft_freqs = np.linspace(0.0, sampling_rate / 2.0, num_frequency_bins)
    mel_min = float(_hertz_to_mel(min_frequency))
    mel_max = float(_hertz_to_mel(max_frequency))
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = _mel_to_hertz(mel_freqs)

    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    filters = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

    # Slaney normalization: each filter's area is 2 / bandwidth.
    enorm = 2.0 / (filter_freqs[2:num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    filters *= np.expand_dims(enorm, 0)
    return filters.T


class LogMelSpectrogram(nn.Module):
    """``(B, samples)`` or ``(samples,)`` float waveform -> ``(B, n_mels,
    frames)`` log-mel features, ``frames = samples // hop_length``.

    Matches HF's ``_torch_extract_fbank_features``: ``torch.stft`` with
    ``center=True`` yields ``samples // hop + 1`` frames and the last is
    dropped, and the dynamic-range clamp is per clip. Whisper pads or
    truncates to ``chunk_length`` seconds *before* this transform; use
    :meth:`pad_or_trim` for that.
    """

    def __init__(
        self,
        num_mel_bins: int = 128,
        sampling_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        chunk_length: int = 30,
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        filters = slaney_mel_filter_bank(1 + n_fft // 2, num_mel_bins, sampling_rate)
        # float32, as HF casts its float64 bank before the matmul
        self.register_buffer("mel_filters", torch.from_numpy(filters).to(torch.float32), persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True), persistent=False)

    def pad_or_trim(self, waveform: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """Zero-pad or truncate the last dim to ``length`` samples (the 30 s
        window by default), as Whisper's extractor does before the STFT."""
        length = self.n_samples if length is None else length
        n = waveform.shape[-1]
        if n > length:
            return waveform[..., :length]
        if n < length:
            return torch.nn.functional.pad(waveform, (0, length - n))
        return waveform

    def num_frames(self, num_samples: int) -> int:
        return num_samples // self.hop_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        squeeze = waveform.dim() == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(torch.float32)
        stft = torch.stft(
            waveform, self.n_fft, self.hop_length, window=self.window, return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self.mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_val = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.squeeze(0) if squeeze else log_spec

    def extra_repr(self) -> str:
        return (
            f"num_mel_bins={self.num_mel_bins}, sampling_rate={self.sampling_rate}, "
            f"n_fft={self.n_fft}, hop_length={self.hop_length}, chunk_length={self.chunk_length}"
        )


def librosa_mel_filters(sampling_rate: int, n_fft: int, num_mel_bins: int) -> np.ndarray:
    """``librosa.filters.mel(norm="slaney")`` as float32 ``(n_mels, 1 + n_fft // 2)``,
    with librosa's own arithmetic and its float32 rounding of each row before
    the area normalization, so NeMo-trained encoders see their exact filters."""
    f_sp = 200.0 / 3
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / (200.0 / 3), np.log(6.4) / 27.0

    def hz_to_mel(f: float) -> float:
        return min_log_mel + np.log(f / min_log_hz) / logstep if f >= min_log_hz else f / f_sp

    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(sampling_rate / 2.0), num_mel_bins + 2)
    mel_f = f_sp * mels
    log_t = mels >= min_log_mel
    mel_f[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sampling_rate)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((num_mel_bins, 1 + n_fft // 2), dtype=np.float32)
    for i in range(num_mel_bins):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights *= (2.0 / (mel_f[2:num_mel_bins + 2] - mel_f[:num_mel_bins]))[:, None]
    return weights


class NemoLogMel:
    """FastConformer front end (Canary, Parakeet, Nemotron ASR): ``(samples,)``
    at 16 kHz -> ``(num_frames(samples), n_mels)``.

    Preemphasis, a power spectrum from a 25 ms symmetric Hann window zero-padded
    to ``n_fft``, Slaney mel, natural log with a 2^-24 floor, then (offline
    checkpoints) per-bin standardization over the clip. Matches HF's
    ``ParakeetFeatureExtractor`` for one clip; the STFT's last frame is
    padding there and is dropped here.

    ``reflect_edges``: the STFT reflects the clip's edges instead of
    zero-padding them, and the last frame is kept, ``samples // hop + 1``
    frames. That is the featurizer of checkpoints whose reference predates NeMo
    2.5 (Nemotron VoiceChat's); NeMo changed both in one commit, so they go
    together.

    A plain object rather than a module, so an engine casting its node to
    bf16 leaves the filters and window in fp32.
    """

    def __init__(
        self,
        num_mel_bins: int = 128,
        sampling_rate: int = 16_000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        preemphasis: float = 0.97,
        normalize: bool = True,
        reflect_edges: bool = False,
        device: torch.device | str = "cpu",
        mel_filters: torch.Tensor | None = None,
        window: torch.Tensor | None = None,
    ):
        """``mel_filters`` / ``window`` override the computed ones, for a
        checkpoint that ships its own (rounded) copies."""
        self.sampling_rate = sampling_rate
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.preemphasis = preemphasis
        self.normalize = normalize
        self.reflect_edges = reflect_edges
        if mel_filters is None:
            mel_filters = torch.from_numpy(librosa_mel_filters(sampling_rate, n_fft, num_mel_bins))
        if window is None:
            window = torch.hann_window(win_length, periodic=False)
        self.mel_filters = mel_filters.reshape(num_mel_bins, n_fft // 2 + 1).to(device=device, dtype=torch.float32)
        self.window = window.to(device=device, dtype=torch.float32)

    @staticmethod
    def frames(num_samples: int, hop_length: int, reflect_edges: bool) -> int:
        """``num_frames`` from the settings alone, for a caller with no featurizer."""
        return num_samples // hop_length + reflect_edges

    def num_frames(self, num_samples: int) -> int:
        return self.frames(num_samples, self.hop_length, self.reflect_edges)

    @property
    def min_samples(self) -> int:
        """The shortest clip it featurizes: two frames, and (``reflect_edges``)
        more than ``n_fft/2`` samples, which the reflection reads past each edge."""
        two_frames = (2 - self.reflect_edges) * self.hop_length
        return max(two_frames, self.n_fft // 2 + 1) if self.reflect_edges else two_frames

    def check_length(self, num_samples: int) -> None:
        if num_samples < self.min_samples:
            raise ValueError(f"audio of {num_samples} samples is shorter than the {self.min_samples} it needs")

    def complete_frames(self, num_samples: int, final: bool) -> int:
        """Frames of a stream ``num_samples`` long that no later audio changes:
        frame m spans ``[m * hop - n_fft/2, m * hop + n_fft/2)``, and the first
        reflects ``n_fft/2`` samples past the stream's first, so it needs more
        than that. At the stream's end, every frame of the clip. ``reflect_edges``
        streams only."""
        assert self.reflect_edges
        if num_samples <= self.n_fft // 2:
            return 0
        if final:
            return self.num_frames(num_samples)
        return (num_samples - self.n_fft // 2) // self.hop_length + 1

    def stream_span(self, x: torch.Tensor, start: int, lo: int, hi: int, final: bool) -> torch.Tensor:
        """The samples ``windows`` reads for a stream's frames ``lo .. hi - 1``,
        from ``x``, its preemphasized samples from sample ``start`` on (from its
        first, while its first frames are cut): reflected at the stream's edges
        as ``batch`` reflects a clip's. Only ``reflect_edges`` streams."""
        assert self.reflect_edges
        pad, hop = self.n_fft // 2, self.hop_length
        if lo * hop < pad:  # frame lo reads past the stream's first sample
            assert start == 0, "the reflection needs the stream's first samples"
            x, start = torch.cat([x[1: pad + 1].flip(0), x]), start - pad
        if final:
            x = torch.cat([x, x[-pad - 1: -1].flip(0)])
        return x[lo * hop - pad - start: (hi - 1) * hop + pad - start]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.batch([waveform])[0]

    def batch(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        """``__call__`` for several clips in one pass and one host-to-device
        copy: each clip gets what it gets alone. A clip is preemphasized, then
        zero past its end, which is what the STFT pads a lone clip with (or,
        ``reflect_edges``, reflected at its own edges); and standardized over its
        own frames."""
        lengths = [w.numel() for w in waveforms]
        self.check_length(min(lengths))
        frames = [self.num_frames(n) for n in lengths]
        pad = self.n_fft // 2
        device = self.window.device
        source = waveforms[0].device
        # host clips are padded straight into pinned memory: one copy to the device
        pin = source.type == "cpu" and device.type == "cuda"
        x = torch.zeros(len(waveforms), max(lengths), dtype=torch.float32, device=source, pin_memory=pin)
        for row, w in enumerate(waveforms):
            x[row, :lengths[row]] = w.reshape(-1)
        x = to_device_nonblocking(x, device)
        n_samples = to_device_nonblocking(torch.tensor(lengths), device)
        if self.preemphasis:
            x = torch.cat([x[:, :1], x[:, 1:] - self.preemphasis * x[:, :-1]], dim=1)
            x = x * (torch.arange(x.shape[1], device=device) < n_samples[:, None])
        if self.reflect_edges:
            # each clip reflected at its own end, as alone, not at the batch's zeros
            padded = x.new_zeros(x.shape[0], x.shape[1] + 2 * pad)
            for row, n in enumerate(lengths):
                padded[row, :n + 2 * pad] = F.pad(x[row:row + 1, :n], (pad, pad), mode="reflect")[0]
            stft = torch.stft(
                padded, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                window=self.window, center=False, return_complex=True,
            )
        else:
            stft = torch.stft(
                x, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                window=self.window, return_complex=True, pad_mode="constant",
            )
        mel = self._log_mel(torch.view_as_real(stft))[:, :, :max(frames)]
        if self.normalize:
            # the mean taken as an offset from the first frame: the plain mean up
            # to rounding, and exact for a constant bin, so digital silence stays
            # zero where the reference's rounding amplifies it into noise
            n = to_device_nonblocking(torch.tensor(frames), device)[:, None, None]
            valid = torch.arange(mel.shape[2], device=device) < n
            ref = mel[:, :, :1]
            mean = ref + ((mel - ref) * valid).sum(dim=2, keepdim=True) / n
            std = (((mel - mean) * valid).pow(2).sum(dim=2, keepdim=True) / (n - 1)).sqrt()
            mel = (mel - mean) / (std + 1e-5)
        return [mel[row, :, :f].T for row, f in enumerate(frames)]

    def windows(self, x: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, samples)`` preemphasized samples -> ``(B, T, n_mels)`` log-mel
        frames, one per ``n_fft`` window at each hop, with no padding (a
        stream's front end cuts the windows itself). ``basis``
        (``stft_basis(self.window, n_fft)``) takes the STFT as a matmul, for a
        captured graph, which must not hold a cuFFT plan (see ``dft``)."""
        if basis is not None:
            spec = stft_as_real(x, basis, self.n_fft, self.hop_length)
        else:
            spec = torch.view_as_real(torch.stft(
                x, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                window=self.window, center=False, return_complex=True,
            ))
        return self._log_mel(spec).transpose(1, 2).to(torch.float32)

    def _log_mel(self, spec: torch.Tensor) -> torch.Tensor:
        """``(B, F, T, 2)`` STFT -> ``(B, n_mels, T)`` log mel power."""
        # sqrt then square, as the reference does: it rounds differently
        power = spec.pow(2).sum(-1).sqrt().pow(2)
        # full fp32 rather than the process's TF32 default, as the reference, when
        # autocast is off (offline); under the engine's autocast it runs in its dtype
        with fp32_precision_scope(Fp32Precision(matmul="highest", cudnn_allow_tf32=None)):
            return torch.log(self.mel_filters @ power + 2.0 ** -24)


# NeMo ``preprocessor`` settings ``NemoLogMel`` runs for every checkpoint
NEMO_MEL_SETTINGS = dict(
    window="hann", log=True, frame_splicing=1, mag_power=2.0, exact_pad=False,
    log_zero_guard_type="add", log_zero_guard_value=2 ** -24,
)
# NeMo's ``AudioToMelSpectrogramPreprocessor`` defaults for the checked settings
# its configs usually leave out
_NEMO_PREPROCESSOR_DEFAULTS = dict(
    mag_power=2.0, exact_pad=False, log_zero_guard_type="add", log_zero_guard_value=2 ** -24,
    preemph=0.97, pad_to=16,
)


def check_nemo_preprocessor(preprocessor: dict[str, Any], **model_settings: Any) -> None:
    """Refuse a NeMo ``preprocessor`` config whose settings differ from
    ``NEMO_MEL_SETTINGS`` and the model's own, the only ones compared. A key it
    leaves out reads as NeMo's default where ``_NEMO_PREPROCESSOR_DEFAULTS``
    has one, and is refused otherwise."""
    expected = {**NEMO_MEL_SETTINGS, **model_settings}
    got = {k: preprocessor.get(k, _NEMO_PREPROCESSOR_DEFAULTS.get(k)) for k in expected}
    unsupported = {k: v for k, v in got.items() if v != expected[k]}
    if unsupported:
        raise ValueError(f"unsupported NeMo preprocessor settings {unsupported}; expected {expected}")


@functools.cache
def _ffmpeg_decoder_loads() -> bool:
    """Whether torchcodec, and the FFmpeg libraries it loads, work here."""
    try:
        from torchcodec.decoders import AudioDecoder  # noqa: F401
    except Exception:  # noqa: BLE001 -- ImportError, or RuntimeError without FFmpeg
        return False
    return True


def load_audio_file(path: str, sample_rate: int = 16_000, max_seconds: float | None = None) -> torch.Tensor:
    """Decode an audio file to a float32 mono waveform at ``sample_rate``.

    libsndfile (``soundfile``) first — it needs no FFmpeg and covers WAV,
    FLAC, OGG and MP3 — then torchcodec for anything else (M4A, WebM, ...).
    Resampling goes through torchaudio. Audio longer than ``max_seconds``
    (default ``media_io.MAX_AUDIO_SECONDS``) is refused while it decodes.
    """
    from mstar.api_server.media_io import MAX_AUDIO_SECONDS

    if max_seconds is None:
        max_seconds = MAX_AUDIO_SECONDS
    return _resample(*_decode(path, sample_rate, max_seconds), sample_rate)


def _decode(path: str, sample_rate: int, max_seconds: float) -> tuple[torch.Tensor, int]:
    """The waveform and its rate: libsndfile at the file's own rate, else
    FFmpeg, which decodes straight to ``sample_rate``."""
    from mstar.api_server.media_io import AudioRefusedError, decode_torchcodec_capped, read_soundfile_capped

    try:
        audio, sr = read_soundfile_capped(path, max_seconds)
    except AudioRefusedError:
        raise
    except Exception:  # noqa: BLE001 — not a libsndfile container; try FFmpeg
        return decode_torchcodec_capped(path, sample_rate, max_seconds).to(torch.float32), sample_rate
    return torch.from_numpy(audio), sr


def _resample(waveform: torch.Tensor, rate: int, sample_rate: int) -> torch.Tensor:
    check_resample_rate(rate, sample_rate)
    if rate != sample_rate:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, rate, sample_rate)
    return waveform.contiguous()


def load_speech_file(path: str, sample_rate: int, max_seconds: float) -> torch.Tensor:
    """:func:`load_audio_file` for request uploads, refused past the model's
    ``max_seconds`` as it decodes, so a small file whose header understates
    its length cannot expand past it. An undecodable file is the client's
    error, so it surfaces as ``ValueError`` (HTTP 400). A missing decoder or
    memory, and anything after the decode, is the server's, and stays as it is."""
    try:
        # decoded at its own rate, so a resampling failure isn't blamed on the file
        waveform, rate = _decode(path, sample_rate, max_seconds)
    except (ImportError, MemoryError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001 -- libsndfile and FFmpeg both refused it
        if not _ffmpeg_decoder_loads():
            raise
        # FFmpeg's message (paths, internals) stays out of the response
        raise ValueError("could not decode the audio upload") from exc
    return _resample(waveform, rate, sample_rate)


SAMPLE_RATE = 16_000
# the mel hop of every FastConformer checkpoint, 10 ms
HOP_LENGTH = 160


def clip_samples(tensors: dict | None, max_seconds: float, model: str) -> torch.Tensor:
    """A request's one audio input as float32 samples at :data:`SAMPLE_RATE`;
    ValueError unless there is exactly one, at most ``max_seconds`` long and
    at least two mel frames."""
    audio = (tensors or {}).get("audio_inputs", [])
    if len(audio) != 1:
        raise ValueError(f"{model} expects exactly one audio input per request; got {len(audio)}")
    samples = audio[0].reshape(-1).to(torch.float32)
    seconds = samples.shape[0] / SAMPLE_RATE
    if seconds > max_seconds:
        # the loader stops a second past the limit, so the length may be clipped
        raise ValueError(
            f"audio is longer than {max_seconds:.0f} s; {model} takes up to {max_seconds:.0f} s per "
            "request (the transcription endpoint splits longer uploads)"
        )
    if samples.shape[0] < 2 * HOP_LENGTH:
        raise ValueError("audio is shorter than 20 ms")
    return samples


def sinusoid_positions(length: int, channels: int, max_timescale: float = 10_000.0) -> torch.Tensor:
    """Whisper's fixed sinusoidal position table, ``(length, channels)``:
    ``[sin | cos]`` halves over geometrically spaced timescales. The
    checkpoint ships it as ``embed_positions.weight``; this is for models
    (Qwen3-ASR's AuT) that compute it instead."""
    if channels % 2:
        raise ValueError(f"channels must be even for sinusoid positions; got {channels}")
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32))
    scaled_time = torch.arange(length, dtype=torch.float32)[:, None] * inv_timescales[None, :]
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)
