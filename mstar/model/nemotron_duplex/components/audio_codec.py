"""EarTTS RVQ audio codec (``tts_model.audio_codec.*``).

A DAC/Vocos-style ConvNeXt codec: a downsampling encoder and an upsampling
decoder, each a heterogeneous stack of strided resample convs and ConvNeXt
blocks, plus a projected-RVQ quantizer (``prvq``).

Channel schedule (from the checkpoint):
    encoder: 18 ->[384]x3 ->768 ->[768]x3 ->1536 ->[1536]x3 ->512(latent)
    decoder: 512 ->1536 ->[1536]x3 ->768 ->[768]x3 ->384 ->[384]x3 ->18

Forward is a faithful port of NeMo ``RVQVAEModel`` / ``Wav2Latent`` /
``Latent2Wav`` / ``PreTrainedProbabilisticVQ``
(``nemo/collections/speechlm2/modules/ear_tts_vae_codec.py``, branch
``nemotron-labs-voicechat``). Config (from the checkpoint ``codec_config``):
``n_fft=16 hop_length=4 base_hidden_size=384 channel_mult=(1,2,4)
rates=(7,7,9) num_blocks=3 kernel_size=7 latent_size=512 codebook_size=1024
num_quantizers=31 wav_to_token_ratio=1764``.

The encoder's spectrogram/STFT and the decoder's inverse-STFT overlap-add are
ported verbatim. ConvNeXt blocks are *causal* (left-padded) and use a
channel-dim LayerNorm with eps 1e-6.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from mstar.engine.resources.recurrent import RecurrentBlockConfig
from mstar.model.nemotron_duplex.components._util import RawWeight, param
from mstar.model.nemotron_duplex.components.dft import irfft_dim1
from mstar.model.nemotron_duplex.config import CodecConfig

# The encoder/decoder layer specs — (kind, weight_shape, stride, transpose) for
# conv layers, ("block", dim) for ConvNeXt blocks — are derived from
# ``CodecConfig`` (channel schedule + rates + num_blocks), not hardcoded.


class ConvNeXtBlock(nn.Module):
    """Causal 1D ConvNeXt block (ports NeMo ``ConvNeXt1d``).

    Depthwise conv (left-padded for causality) -> channel-dim LayerNorm ->
    pointwise expand -> GELU -> pointwise project, with a residual add.
    """

    def __init__(self, dim: int, expand: int = 4, kernel: int = 7):
        super().__init__()
        self.kernel = kernel
        # No conv padding: causality handled by explicit left pad in forward.
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel, groups=dim)
        # eps 1e-6 to match NeMo ``LayerNormNd``.
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv1d(dim, expand * dim, kernel_size=1)
        self.pwconv2 = nn.Conv1d(expand * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, tail: torch.Tensor | None = None) -> torch.Tensor:
        """``tail``, for a stream, is the block's previous ``kernel - 1`` inputs
        ``[B, C, kernel - 1]``; without one the block pads with zeros, as at the
        stream's start."""
        residual = x
        x = F.pad(x, [self.kernel - 1, 0]) if tail is None else torch.cat([tail.to(x.dtype), x], dim=-1)
        x = self.dwconv(x)
        # Channel-dim LayerNorm: normalize over C (== nn.LayerNorm over the
        # transposed last dim), eps 1e-6, biased variance.
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.pwconv2(F.gelu(self.pwconv1(x)))
        return residual + x


def _build_stack(spec, kernel: int) -> tuple[nn.ModuleList, list]:
    layers: list[nn.Module] = []
    meta: list = []
    for entry in spec:
        if entry[0] == "conv":
            _, shape, stride, transpose = entry
            layers.append(RawWeight(*shape))
            meta.append(("conv", stride, transpose))
        else:
            layers.append(ConvNeXtBlock(entry[1], kernel=kernel))
            meta.append(("block", None, None))
    return nn.ModuleList(layers), meta


class _CodecHalf(nn.Module):
    def __init__(self, spec, kernel: int):
        super().__init__()
        self.layers, self._meta = _build_stack(spec, kernel)

    def run(self, x: torch.Tensor, tails: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Apply the conv/block stack to a ``[B, C, T]`` tensor.

        For a stream, ``tails`` holds each block's previous ``kernel - 1`` inputs,
        in order, and the run replaces them with its own last ones. The resample
        convs have kernel == stride, so the blocks hold all the state."""
        block = 0
        for layer, (kind, stride, transpose) in zip(self.layers, self._meta, strict=True):
            if kind == "block":
                if tails is None:
                    x = layer(x)
                    continue
                tail, keep = tails[block], tails[block].shape[-1]
                tails[block] = x[..., -keep:] if x.shape[-1] >= keep else (
                    torch.cat([tail.to(x.dtype), x], dim=-1)[..., -keep:])
                x = layer(x, tail)
                block += 1
            elif transpose:
                x = F.conv_transpose1d(x, layer.weight, stride=stride)
            else:
                x = F.conv1d(x, layer.weight, stride=stride)
        return x


class _Variance(nn.Module):
    def __init__(self):
        super().__init__()
        self.variance = param()  # scalar


class ProjectedRVQ(nn.Module):
    """Projected residual vector quantizer (ports NeMo ``PreTrainedProbabilisticVQ``).

    Per-quantizer codebooks (``mus_list``) and running variances
    (``_variance_list``).  ``encode`` greedily selects the nearest codebook
    entry per residual level; ``decode`` sums the selected embeddings.
    """

    def __init__(self, num_quantizers: int, codebook: int, dim: int):
        super().__init__()
        self.mus_list = nn.ParameterList([param(codebook, dim) for _ in range(num_quantizers)])
        self._variance_list = nn.ModuleList([_Variance() for _ in range(num_quantizers)])

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """``[B, T, dim]`` continuous latents -> ``[B, T, num_quantizers]`` codes."""
        r = z
        ids = []
        for mus in self.mus_list:
            # squared distance to each codebook entry: |r|^2 + |mus|^2 - 2 r.mus
            dist = (
                r.pow(2).sum(-1, keepdim=True)
                + mus.pow(2).sum(-1)
                - 2.0 * (r.unsqueeze(-2) @ mus.transpose(-1, -2)).squeeze(-2)
            )
            idx = dist.argmin(-1)
            r = r - F.embedding(idx, mus)
            ids.append(idx)
        return torch.stack(ids, -1)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """``[B, T, num_quantizers]`` codes -> ``[B, T, dim]`` continuous latents."""
        mus0 = self.mus_list[0]
        z = torch.zeros((*code.shape[:-1], mus0.shape[-1]), device=code.device, dtype=mus0.dtype)
        for i, mus in enumerate(self.mus_list):
            z = z + F.embedding(code[..., i], mus)
        return z


def _spectrogram(wav: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
    """STFT with manual centering pad (ports NeMo ``spectrogram``)."""
    pad_l = (n_fft - hop_length) // 2
    pad_r = (n_fft - hop_length) - pad_l
    with torch.autocast(device_type=wav.device.type, enabled=False):
        wav = F.pad(wav.float(), (pad_l, pad_r))
        window = torch.hann_window(win_length, dtype=torch.float, device=wav.device)
        spec = torch.stft(
            wav, n_fft, hop_length=hop_length, win_length=win_length, window=window,
            center=False, normalized=False, onesided=True, return_complex=True,
        )
    return spec


def _spec_to_wav(
    spec: torch.Tensor, n_fft: int, hop_length: int, win_length: int, constrain_value_range: bool = True,
    by_matmul: bool = False,
) -> torch.Tensor:
    """Inverse STFT via windowed overlap-add (ports NeMo ``spec_to_wav``).
    ``by_matmul`` takes the inverse DFT as a matmul, for a captured graph (see ``dft``)."""
    with torch.autocast(device_type=spec.device.type, enabled=False):
        pad = (win_length - hop_length) // 2
        T = spec.size(-1)
        window = torch.hann_window(win_length, device=spec.device)

        if by_matmul:
            ifft = irfft_dim1(spec, n_fft)
        else:
            ifft = torch.fft.irfft(spec, n=n_fft, dim=-2, norm="backward")
        window_unsqz = window.unsqueeze(-1)

        if constrain_value_range:
            ifft = torch.where(
                ifft >= 0,
                torch.minimum(ifft, window_unsqz),
                torch.maximum(ifft, -window_unsqz),
            )

        ifft = ifft * window_unsqz

        output_size = (T - 1) * hop_length + win_length
        wav = F.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, win_length), stride=(1, hop_length),
        )[..., 0, 0, pad:-pad]

        # The envelope is ~0 only where the window and hop leave a gap, which
        # check_window_envelope refuses when the codec is built; a device-side
        # guard keeps the decode free of host syncs, for CUDA-graph capture.
        env = _window_envelope(window, T, hop_length, win_length)
        return torch.where(env > _ENVELOPE_FLOOR, wav / env, torch.zeros_like(wav))


_ENVELOPE_FLOOR = 1e-11


def _window_envelope(window: torch.Tensor, T: int, hop_length: int, win_length: int) -> torch.Tensor:
    """The overlap-added squared window of ``T`` frames, trimmed as ``_spec_to_wav``
    trims the signal."""
    pad = (win_length - hop_length) // 2
    output_size = (T - 1) * hop_length + win_length
    return F.fold(
        window.square().expand(T, -1).transpose(0, 1), output_size=(1, output_size),
        kernel_size=(1, win_length), stride=(1, hop_length),
    ).squeeze()[pad:-pad]


def check_window_envelope(window: torch.Tensor, hop_length: int) -> None:
    """Refuse a window and hop whose overlap-add has a gap at any frame count.
    A sample within one window of either end sees the same frames for every
    count of at least ``ceil(win / hop)``, and an interior one the full periodic
    overlap, so the counts up to twice that and one more cover them all."""
    win_length = window.numel()
    k = -(-win_length // hop_length)
    for T in range(1, 2 * k + 2):
        if not (_window_envelope(window, T, hop_length, win_length) > _ENVELOPE_FLOOR).all():
            raise ValueError(f"the codec's window overlap-adds to ~0 at hop {hop_length} ({T} frames): "
                             "its inverse STFT cannot normalize")


class CodecStreamState(NamedTuple):
    """What a decode stream carries from one step to the next."""

    # each ConvNeXt block's last ``kernel - 1`` inputs ``[B, C, kernel - 1]``, in decoder order
    tails: list[torch.Tensor]
    # ``[B, 2, n_fft - hop]`` float32: the overlap-added signal and squared window
    # of the samples past the last final one, which the next frames still add to
    overlap: torch.Tensor


class CodecStreamAudio(NamedTuple):
    """One decode step's samples, float32. ``end_before`` / ``end_after`` are the
    stream's last ``stream_edge`` samples had it ended before / after this step's
    frames: the full decode normalizes its right edge by the partial window sum."""

    final: torch.Tensor        # [B, frames * wav_to_token_ratio]
    end_before: torch.Tensor   # [B, stream_edge]
    end_after: torch.Tensor    # [B, stream_edge]


def codec_state_blocks(config: CodecConfig, dtype: torch.dtype | None = None) -> dict[str, RecurrentBlockConfig]:
    """A decode stream's state as recurrent-pool blocks: per decoder stage, its
    blocks' conv tails ``(num_blocks, channels, kernel - 1)`` (by default in the
    engine's compute dtype, which the blocks compute in), and the inverse STFT's
    overlap (``CodecStreamState.overlap``, float32)."""
    blocks = {
        f"stage{i}": RecurrentBlockConfig(shape=(config.num_blocks, ch, config.kernel_size - 1), dtype=dtype)
        for i, ch in enumerate(reversed(config.channels))
    }
    blocks["overlap"] = RecurrentBlockConfig(shape=(2, config.n_fft - config.hop_length), dtype=torch.float32)
    return blocks


def _overlap_add_stream(
    spec: torch.Tensor, overlap: torch.Tensor, n_fft: int, hop_length: int,
) -> tuple[CodecStreamAudio, torch.Tensor]:
    """``_spec_to_wav`` (window == n_fft, value range constrained, by matmul) for
    a stream: ``spec``'s frames overlap-add onto the carried ``overlap``. The
    samples no later frame reaches are final; the next ``n_fft - hop`` become
    the new overlap."""
    with torch.autocast(device_type=spec.device.type, enabled=False):
        frames, held = spec.size(-1), n_fft - hop_length
        size = frames * hop_length + held
        window = torch.hann_window(n_fft, device=spec.device).unsqueeze(-1)
        ifft = irfft_dim1(spec, n_fft)
        ifft = torch.where(ifft >= 0, torch.minimum(ifft, window), torch.maximum(ifft, -window)) * window

        def fold(x: torch.Tensor) -> torch.Tensor:
            return F.fold(x, output_size=(1, size), kernel_size=(1, n_fft), stride=(1, hop_length))

        carried = F.pad(overlap, (0, size - held))
        sig = fold(ifft)[:, 0, 0] + carried[:, 0]
        env = fold(window.square().expand(-1, frames)).reshape(1, size) + carried[:, 1]
        out = _normalize(sig, env)
        done, edge = frames * hop_length, held // 2
        audio = CodecStreamAudio(
            final=out[:, :done], end_before=_normalize(overlap[:, 0, :edge], overlap[:, 1, :edge]),
            end_after=out[:, done:done + edge],
        )
        return audio, torch.stack([sig[:, done:], env[:, done:]], dim=1)


def _normalize(sig: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    """The overlap-added signal over its squared-window sum, 0 where that is ~0
    (as ``_spec_to_wav``; a fresh stream's empty overlap)."""
    return torch.where(env > _ENVELOPE_FLOOR, sig / env, torch.zeros_like(sig))


class AudioCodec(nn.Module):
    def __init__(self, config: CodecConfig | None = None):
        super().__init__()
        cfg = config or CodecConfig()
        self.config = cfg
        self.encoder = _CodecHalf(cfg.encoder_layers(), kernel=cfg.kernel_size)
        self.decoder = _CodecHalf(cfg.decoder_layers(), kernel=cfg.kernel_size)
        self.prvq = ProjectedRVQ(cfg.num_quantizers, cfg.codebook_size, cfg.latent_size)
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.wav_to_token_ratio = cfg.wav_to_token_ratio
        self.max_magnitude = cfg.max_magnitude
        # on the host: the model builds the codec under torch.device("meta")
        check_window_envelope(torch.hann_window(cfg.n_fft, device="cpu"), cfg.hop_length)
        if (cfg.n_fft - cfg.hop_length) % 2:
            # the stream trims half the overlap at each end, as the full decode does
            raise ValueError(
                f"the codec's stream needs an even overlap; n_fft {cfg.n_fft} - hop {cfg.hop_length} is odd"
            )

    # ---- autoencoder halves -------------------------------------------------
    def ae_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Waveform ``[B, 1, T]`` -> continuous latents ``[B, T', latent]``."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = _spectrogram(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft)
            mag, ph = torch.view_as_real(spec).chunk(2, dim=-1)
            x = torch.cat([mag, ph], 1).squeeze(-1)
        x = self.encoder.run(x)
        return x.transpose(-1, -2)

    def ae_decode(self, x: torch.Tensor, constrain_value_range: bool = True, by_matmul: bool = False) -> torch.Tensor:
        """Continuous latents ``[B, T', latent]`` -> waveform ``[B, 1, T]``; ``by_matmul``
        as ``_spec_to_wav`` takes it."""
        spec = self._spectrum(self.decoder.run(x.transpose(-1, -2)))
        with torch.autocast(device_type=spec.device.type, enabled=False):
            return _spec_to_wav(
                spec, self.n_fft, self.hop_length, self.n_fft, constrain_value_range=constrain_value_range,
                by_matmul=by_matmul,
            ).unsqueeze(1)

    def _spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """The decoder stack's output ``[B, 2 * bins, frames]`` (log-magnitude,
        phase) -> the complex one-sided spectrum ``[B, bins, frames]``, float32."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            mag, ph = x.float().chunk(2, dim=1)
            mag = self.max_magnitude * torch.exp(-F.softplus(-mag + math.log(self.max_magnitude)))

            mag_dc, mag_mid, mag_nyquist = mag.split([1, mag.size(1) - 2, 1], dim=1)
            ph_dc, ph_mid, ph_nyquist = torch.cos(ph).split([1, ph.size(1) - 2, 1], dim=1)
            ph_imag = torch.sin(ph[:, 1:-1, :])

            spec_real = mag_mid * ph_mid
            spec_imag = mag_mid * ph_imag
            return torch.cat([mag_dc * ph_dc, spec_real + 1j * spec_imag, mag_nyquist * ph_nyquist], 1)

    # ---- streaming decode ---------------------------------------------------
    @property
    def stream_edge(self) -> int:
        """Samples the full decode trims off each end of its overlap-add: a
        stream's first frame emits that many fewer, and its end that many more."""
        return (self.n_fft - self.hop_length) // 2

    def decode_stream(self, code: torch.Tensor, state: CodecStreamState) -> tuple[CodecStreamAudio, CodecStreamState]:
        """Decode a stream's next code frames ``[B, T, num_quantizers]`` given its
        ``state`` (zeros for a stream's first frames, which is the causal zero
        padding of the full decode): the samples these frames make final, and
        the state after them. Over a stream, ``audio.final`` (less
        ``stream_edge`` samples at the first frame) followed by the last step's
        ``audio.end_after`` is the full decode of all its frames.

        Every step is the same fixed-shape work, so a CUDA graph captures it; the
        inverse DFT is a matmul (see ``dft``)."""
        tails = list(state.tails)
        spec = self._spectrum(self.decoder.run(self.dequantize(code).transpose(-1, -2), tails))
        audio, overlap = _overlap_add_stream(spec, state.overlap, self.n_fft, self.hop_length)
        return audio, CodecStreamState(tails=tails, overlap=overlap)

    # ---- quantizer wrappers -------------------------------------------------
    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        return self.prvq.encode(z)

    def dequantize(self, code: torch.Tensor) -> torch.Tensor:
        return self.prvq.decode(code)

    # ---- public API ---------------------------------------------------------
    def encode(self, x: torch.Tensor, x_len: torch.Tensor | None = None):
        """Waveform ``[B, 1, T]`` -> (codes ``[B, T', num_quantizers]``, code_len)."""
        z_e = self.ae_encode(x)
        code = self.quantize(z_e)
        code_len = x_len // self.wav_to_token_ratio if x_len is not None else None
        return code, code_len

    def decode(self, code: torch.Tensor, code_len: torch.Tensor | None = None, constrain_value_range: bool = True,
               by_matmul: bool = False):
        """Codes ``[B, T', num_quantizers]`` -> (waveform ``[B, 1, T]``, wav_len); ``by_matmul``
        as ``_spec_to_wav`` takes it."""
        z_q = self.dequantize(code)
        x_hat = self.ae_decode(z_q, constrain_value_range=constrain_value_range, by_matmul=by_matmul)
        wav_len = code_len * self.wav_to_token_ratio if code_len is not None else None
        return x_hat, wav_len

    @staticmethod
    def remap(name: str) -> str | None:
        prefix = "tts_model.audio_codec."
        return name[len(prefix):] if name.startswith(prefix) else None

    def load_weights(self, weights):
        """Load the ``tts_model.audio_codec.*`` subtree."""
        from mstar.model.loader import load_hf_weights

        return load_hf_weights(self, weights, stacked_params=[], name_remapper=self.remap)
