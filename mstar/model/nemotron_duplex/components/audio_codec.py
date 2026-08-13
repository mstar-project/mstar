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

import torch
from torch import nn
from torch.nn import functional as F

from mstar.model.nemotron_duplex.components._util import RawWeight, param
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, [self.kernel - 1, 0])  # causal left pad
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

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the conv/block stack to a ``[B, C, T]`` tensor."""
        for layer, (kind, stride, transpose) in zip(self.layers, self._meta, strict=True):
            if kind == "block":
                x = layer(x)
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
) -> torch.Tensor:
    """Inverse STFT via windowed overlap-add (ports NeMo ``spec_to_wav``)."""
    with torch.autocast(device_type=spec.device.type, enabled=False):
        pad = (win_length - hop_length) // 2
        T = spec.size(-1)
        window = torch.hann_window(win_length, device=spec.device)

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

        window_sq = window.square().expand(T, -1).transpose(0, 1)
        window_envelope = F.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, win_length), stride=(1, hop_length),
        ).squeeze()[pad:-pad]

        assert (window_envelope > 1e-11).all(), "Window envelope has zero values, cannot normalize."
        wav = wav / window_envelope
        return wav


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

    # ---- autoencoder halves -------------------------------------------------
    def ae_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Waveform ``[B, 1, T]`` -> continuous latents ``[B, T', latent]``."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = _spectrogram(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft)
            mag, ph = torch.view_as_real(spec).chunk(2, dim=-1)
            x = torch.cat([mag, ph], 1).squeeze(-1)
        x = self.encoder.run(x)
        return x.transpose(-1, -2)

    def ae_decode(self, x: torch.Tensor, constrain_value_range: bool = True) -> torch.Tensor:
        """Continuous latents ``[B, T', latent]`` -> waveform ``[B, 1, T]``."""
        x = x.transpose(-1, -2)
        x = self.decoder.run(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            mag, ph = x.float().chunk(2, dim=1)
            mag = self.max_magnitude * torch.exp(-F.softplus(-mag + math.log(self.max_magnitude)))

            mag_dc, mag_mid, mag_nyquist = mag.split([1, mag.size(1) - 2, 1], dim=1)
            ph_dc, ph_mid, ph_nyquist = torch.cos(ph).split([1, ph.size(1) - 2, 1], dim=1)
            ph_imag = torch.sin(ph[:, 1:-1, :])

            spec_real = mag_mid * ph_mid
            spec_imag = mag_mid * ph_imag
            spec = torch.cat([mag_dc * ph_dc, spec_real + 1j * spec_imag, mag_nyquist * ph_nyquist], 1)

            x = _spec_to_wav(
                spec, self.n_fft, self.hop_length, self.n_fft, constrain_value_range=constrain_value_range,
            ).unsqueeze(1)
        return x

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

    def decode(self, code: torch.Tensor, code_len: torch.Tensor | None = None, constrain_value_range: bool = True):
        """Codes ``[B, T', num_quantizers]`` -> (waveform ``[B, 1, T]``, wav_len)."""
        z_q = self.dequantize(code)
        x_hat = self.ae_decode(z_q, constrain_value_range=constrain_value_range)
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
