"""HiFT vocoder: 80-bin mel at 50 Hz -> 24 kHz waveform.

Reference: ``chatterbox/models/s3gen/hifigan.py`` (``HiFTGenerator``, a
neural-source-filter HiFi-GAN with an iSTFT head, from CosyVoice / HiFTNet)
and ``chatterbox/models/s3gen/f0_predictor.py``.

The checkpoint stores the convolutions with weight-norm parametrizations;
the loader folds them into plain weights (``loader.fold_weight_norm``), so
nothing here is parametrized at serve time.

Randomness: the harmonic source draws one uniform phase per harmonic and two
Gaussian noise fields per call, in the order ``rand([B, H+1, 1])``,
``randn([B, H+1, L])``, ``randn([B, 1, L])`` (``L`` = output samples). Pass a
``torch.Generator`` to make a call reproducible; with ``None`` the global RNG
is consumed exactly like the reference, which is what the parity test relies on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.chatterbox.config import S3GenHiFTConfig


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class Snake(nn.Module):
    """``x + sin^2(alpha x) / alpha`` with one learned ``alpha`` per channel."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        return x + (1.0 / (alpha + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class ResBlock(nn.Module):
    """Snake-activated dilated residual stack (BigVGAN style)."""

    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=_get_padding(kernel_size, d))
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))
            for _ in dilations
        ])
        self.activations1 = nn.ModuleList([Snake(channels) for _ in dilations])
        self.activations2 = nn.ModuleList([Snake(channels) for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = zip(self.activations1, self.convs1, self.activations2, self.convs2, strict=True)
        for act1, conv1, act2, conv2 in layers:
            xt = conv1(act1(x))
            xt = conv2(act2(xt))
            x = xt + x
        return x


class F0Predictor(nn.Module):
    """Five conv+ELU layers and a linear head; returns ``|f0|`` per mel frame, ``[B, T]``."""

    def __init__(self, in_channels: int, cond_channels: int):
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(5):
            layers += [nn.Conv1d(channels, cond_channels, kernel_size=3, padding=1), nn.ELU()]
            channels = cond_channels
        self.condnet = nn.Sequential(*layers)
        self.classifier = nn.Linear(cond_channels, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.condnet(mel).transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


@dataclass
class HiFTNoise:
    """The random draws one vocoder call consumes, taken in the reference's
    order (harmonic phase offsets, harmonic noise, then a field the reference
    draws and discards) so they can be made before the forward and the
    forward stays deterministic and graph-capturable."""

    phase: torch.Tensor  # [B, H+1, 1] uniform in [0, 1)
    harmonic: torch.Tensor  # [B, H+1, L] standard normal


class HarmonicSource(nn.Module):
    """Sine harmonics + noise excitation from an upsampled f0 track (``SourceModuleHnNSF``)."""

    def __init__(
        self, sampling_rate: int, harmonic_num: int, sine_amp: float, noise_std: float, voiced_threshold: float,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def draw_noise(
        self, batch: int, length: int, dtype: torch.dtype, device, generator: torch.Generator | None,
    ) -> HiFTNoise:
        """The draws ``forward`` used to make inline, in the same order: the
        phase offsets, the harmonic noise, then the field the reference draws
        and discards (so a shared generator keeps the reference's state)."""
        phase = torch.rand((batch, self.harmonic_num + 1, 1), dtype=dtype, device=device, generator=generator)
        harmonic = torch.randn((batch, self.harmonic_num + 1, length), dtype=dtype, device=device, generator=generator)
        torch.randn((batch, length, 1), dtype=dtype, device=device, generator=generator)
        return HiFTNoise(phase=phase, harmonic=harmonic)

    def _sines(self, f0: torch.Tensor, noise: HiFTNoise) -> tuple[torch.Tensor, torch.Tensor]:
        """``f0``: ``[B, 1, L]`` Hz -> (sine harmonics ``[B, H+1, L]``, voiced mask ``[B, 1, L]``)."""
        harmonics = torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype).view(1, -1, 1)
        f_mat = f0 * harmonics / self.sampling_rate
        theta_mat = 2 * math.pi * (torch.cumsum(f_mat, dim=-1) % 1)
        # torch.distributions.Uniform(-pi, pi).sample(): low + u*high - u*low
        u = noise.phase
        low = torch.tensor(-math.pi, dtype=f0.dtype, device=f0.device)
        high = torch.tensor(math.pi, dtype=f0.dtype, device=f0.device)
        phase_vec = low + u * high - u * low
        phase_vec = torch.cat([torch.zeros_like(phase_vec[:, :1]), phase_vec[:, 1:]], dim=1)
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)
        uv = (f0 > self.voiced_threshold).to(torch.float32)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        return sine_waves * uv + noise_amp * noise.harmonic, uv

    def forward(
        self, f0_upsampled: torch.Tensor, noise: HiFTNoise | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """``[B, L, 1]`` f0 -> merged excitation ``[B, L, 1]``; ``noise`` from
        ``draw_noise`` (drawn here from ``generator`` when not given)."""
        b, length, _ = f0_upsampled.shape
        if noise is None:
            noise = self.draw_noise(b, length, f0_upsampled.dtype, f0_upsampled.device, generator)
        with torch.no_grad():
            sine_waves, _uv = self._sines(f0_upsampled.transpose(1, 2), noise)
            sine_waves = sine_waves.transpose(1, 2)
        return torch.tanh(self.l_linear(sine_waves))


class HiFTGenerator(nn.Module):
    """NSF HiFi-GAN with an iSTFT output head. ``forward(mel [B, 80, T]) -> wav [B, T * upsample_factor]``."""

    def __init__(self, config: S3GenHiFTConfig):
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.f0_predictor = F0Predictor(config.in_channels, config.f0_cond_channels)
        self.m_source = HarmonicSource(
            config.sampling_rate, config.nb_harmonics, config.nsf_alpha, config.nsf_sigma, config.nsf_voiced_threshold,
        )
        self.conv_pre = nn.Conv1d(config.in_channels, config.base_channels, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes, strict=True)):
            self.ups.append(nn.ConvTranspose1d(
                config.base_channels // (2 ** i), config.base_channels // (2 ** (i + 1)), k, u, padding=(k - u) // 2,
            ))

        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + list(config.upsample_rates[::-1][:-1])
        cum = 1
        downsample_cum_rates = []
        for rate in downsample_rates:
            cum *= rate
            downsample_cum_rates.append(cum)
        n_fft_bins = config.istft_n_fft + 2
        for i, (u, k, d) in enumerate(zip(
            downsample_cum_rates[::-1], config.source_resblock_kernel_sizes,
            config.source_resblock_dilation_sizes, strict=True,
        )):
            ch = config.base_channels // (2 ** (i + 1))
            if u == 1:
                self.source_downs.append(nn.Conv1d(n_fft_bins, ch, 1, 1))
            else:
                self.source_downs.append(nn.Conv1d(n_fft_bins, ch, u * 2, u, padding=u // 2))
            self.source_resblocks.append(ResBlock(ch, k, tuple(d)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.base_channels // (2 ** (i + 1))
            for k, d in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes, strict=True):
                self.resblocks.append(ResBlock(ch, k, tuple(d)))
        self.conv_post = nn.Conv1d(ch, n_fft_bins, 7, 1, padding=3)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        # scipy's periodic Hann in float64, cast once, as the reference does
        window = torch.hann_window(config.istft_n_fft, periodic=True, dtype=torch.float64).float()
        self.register_buffer("stft_window", window, persistent=False)

    @property
    def upsample_factor(self) -> int:
        return self.config.upsample_factor

    def _stft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = torch.stft(
            x, self.config.istft_n_fft, self.config.istft_hop_len, self.config.istft_n_fft,
            window=self.stft_window, return_complex=True,
        )
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return torch.istft(
            torch.complex(real, imag), self.config.istft_n_fft, self.config.istft_hop_len, self.config.istft_n_fft,
            window=self.stft_window,
        )

    def decode(self, mel: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        s_real, s_imag = self._stft(source.squeeze(1))
        s_stft = torch.cat([s_real, s_imag], dim=1)

        x = self.conv_pre(mel)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.config.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            si = self.source_resblocks[i](self.source_downs[i](s_stft))
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        half = self.config.istft_n_fft // 2 + 1
        magnitude = torch.exp(x[:, :half, :])
        phase = torch.sin(x[:, half:, :])
        x = self._istft(magnitude, phase)
        return torch.clamp(x, -self.config.audio_limit, self.config.audio_limit)

    def draw_noise(self, mel: torch.Tensor, generator: torch.Generator | None) -> HiFTNoise:
        """The excitation draws for vocoding ``mel`` (``[B, 80, T]``)."""
        return self.m_source.draw_noise(
            mel.shape[0], mel.shape[-1] * self.upsample_factor, mel.dtype, mel.device, generator,
        )

    def vocode(
        self,
        mel: torch.Tensor,
        generator: torch.Generator | None = None,
        cache_source: torch.Tensor | None = None,
        noise: HiFTNoise | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(wav [B, 480T], source [B, 1, 480T])``; ``cache_source`` overwrites
        the head of the excitation so consecutive chunks share their harmonics
        (reference ``inference(cache_source=...)``). ``noise`` is the excitation's
        random draws (``draw_noise``), taken from ``generator`` when not given."""
        if noise is None:
            noise = self.draw_noise(mel, generator)
        return self.vocode_with(mel, noise.phase, noise.harmonic, cache_source)

    def vocode_with(
        self, mel: torch.Tensor, phase: torch.Tensor, harmonic: torch.Tensor, cache_source: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``vocode`` with every random draw given: deterministic, graph-capturable."""
        f0 = self.f0_predictor(mel)
        source = F.interpolate(f0[:, None], scale_factor=float(self.upsample_factor), mode="nearest").transpose(1, 2)
        source = self.m_source(source, HiFTNoise(phase=phase, harmonic=harmonic)).transpose(1, 2)
        if cache_source is not None and cache_source.shape[-1] > 0:
            n = min(cache_source.shape[-1], source.shape[-1])
            head = cache_source[:, :, :n].to(source.dtype)
            source = torch.cat([head, source[:, :, n:]], dim=-1)
        return self.decode(mel, source), source

    def forward(self, mel: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.vocode(mel, generator=generator)[0]
