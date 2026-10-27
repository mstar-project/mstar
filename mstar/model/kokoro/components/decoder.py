"""iSTFTNet decoder: prosody-conditioned convolutional trunk, harmonic-plus-noise
excitation, two upsampling stages with Snake residual blocks, and an inverse
STFT head (StyleTTS2's ``Modules/istftnet.py`` as adapted by Kokoro).

Every block carries per-row frame lengths so a padded batch decodes each row
exactly as the single-request reference would.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.kokoro.components.adain import AdaIN1d, AdainResBlk1d
from mstar.model.kokoro.components.masking import length_mask, mask_channels
from mstar.model.kokoro.components.stft import RealSTFT
from mstar.model.kokoro.config import KokoroModelConfig

GENERATOR_LEAKY_SLOPE = 0.1
_MASK32 = 0xFFFFFFFF


def _mul32(x: torch.Tensor, c: int) -> torch.Tensor:
    """``x * c mod 2**32`` for int64 ``x`` in [0, 2**32), split so no product overflows."""
    return (x * (c & 0xFFFF) + (((x * (c >> 16)) & 0xFFFF) << 16)) & _MASK32


def _hash32(x: torch.Tensor) -> torch.Tensor:
    """The ``lowbias32`` integer hash on int64 values in [0, 2**32)."""
    x = _mul32(x ^ (x >> 16), 0x7FEB352D)
    x = _mul32(x ^ (x >> 15), 0x846CA68B)
    return x ^ (x >> 16)


def keyed_normal(keys: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Standard normals ``[B, rows, cols]``, a pure function of each row's key and position.

    ``keys`` is int64 ``[B, 2]`` (two 32-bit words). A row's values do not
    depend on the batch or its padding, and the draw is CUDA-graph safe.
    """
    counter = torch.arange(rows * cols, device=keys.device, dtype=torch.int64).view(1, rows, cols)
    k1, k2 = keys[:, 0, None, None], keys[:, 1, None, None]
    bits = _hash32(_hash32((counter + k1) & _MASK32) ^ k2)
    # two 16-bit uniforms in (0, 1) per value, then Box-Muller
    u1 = ((bits >> 16).float() + 0.5) / 65536
    u2 = ((bits & 0xFFFF).float() + 0.5) / 65536
    return torch.sqrt(-2 * torch.log(u1)) * torch.cos((2 * math.pi) * u2)


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Snake activation ``x + sin^2(alpha x) / alpha`` (zero stays zero)."""
    return x + (1 / alpha) * torch.sin(alpha * x).square()


class SnakeResBlock(nn.Module):
    """HiFi-GAN ``ResBlock1`` with AdaIN and Snake activations (``AdaINResBlock1``)."""

    def __init__(self, channels: int, kernel_size: int, dilations: list[int], style_dim: int):
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=(kernel_size * d - d) // 2)
            for d in dilations
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2)
            for _ in dilations
        )
        self.adain1 = nn.ModuleList(AdaIN1d(style_dim, channels) for _ in dilations)
        self.adain2 = nn.ModuleList(AdaIN1d(style_dim, channels) for _ in dilations)
        self.alpha1 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilations)
        self.alpha2 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilations)

    def forward(self, x: torch.Tensor, style: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2, strict=True
        ):
            xt = mask_channels(c1(snake(n1(x, style, mask), a1)), mask)
            xt = mask_channels(c2(snake(n2(xt, style, mask), a2)), mask)
            x = xt + x
        return x


class SineSource(nn.Module):
    """Harmonic-plus-noise excitation from the F0 curve (``SourceModuleHnNSF``).

    ``deterministic`` silences the noise branch; tests use it to compare a
    padded batch against single rows, which draw noise of different shapes.

    Serving passes ``noise_keys`` (per row, from the request seed); the noise
    is then ``keyed_normal`` and the global RNG is untouched. Without keys the
    noise comes from the global RNG, as in the reference, which also draws an
    initial harmonic phase and an unused noise tensor; both are drawn then too
    so a seeded run consumes the RNG stream exactly as the reference does.
    """

    def __init__(self, config: KokoroModelConfig):
        super().__init__()
        self.sample_rate = config.sample_rate
        self.upsample = config.istftnet.upsample_factor
        self.num_harmonics = config.source_harmonics + 1
        self.sine_amp = config.source_sine_amp
        self.noise_std = config.source_noise_std
        self.voiced_threshold = config.source_voiced_threshold
        self.l_linear = nn.Linear(self.num_harmonics, 1)
        self.register_buffer(
            "harmonic_index", torch.arange(1, self.num_harmonics + 1, dtype=torch.float32), persistent=False
        )
        self.deterministic = False

    def forward(
        self, f0: torch.Tensor, frame_lengths: torch.Tensor, noise_keys: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``[B, n]`` F0 (Hz) per generator frame -> ``[B, n * upsample]`` excitation."""
        bsz, num_frames = f0.shape
        num_samples = num_frames * self.upsample
        # Phase increment per frame, in cycles, for the fundamental and its
        # harmonics, time-major ``[B, n, H]`` like the reference: the frame-rate
        # value equals its downsampled sample-rate value because F0 is constant
        # within a frame, and summing over the same layout keeps the GPU scan's
        # rounding identical (the vocoder amplifies one-ulp phase differences).
        cycles = (f0[:, :, None] * self.harmonic_index / self.sample_rate) % 1
        if noise_keys is None:
            torch.rand(bsz, self.num_harmonics, device=f0.device)  # reference's initial phase (has no effect)
        phase = (torch.cumsum(cycles, dim=1) * (2 * math.pi)).transpose(1, 2)
        # Hold each row's last valid phase through its padding: the linear
        # upsample then computes every valid sample with exactly the operands
        # the reference's single-row call uses, including the clamped tail.
        frame_index = torch.arange(num_frames, device=f0.device)[None, None, :]
        edge_index = torch.minimum(frame_index, (frame_lengths - 1)[:, None, None]).expand_as(phase)
        phase = F.interpolate(phase.gather(2, edge_index) * self.upsample, scale_factor=self.upsample, mode="linear")
        # ``[B, L, H]`` as a transposed view: the reference's noise is drawn
        # with ``randn_like`` on a tensor of exactly these strides, and the
        # normal kernel's output order depends on them.
        sines = torch.sin(phase).transpose(1, 2)

        f0_samples = f0.repeat_interleave(self.upsample, dim=1)
        voiced = (f0_samples > self.voiced_threshold).to(f0.dtype)[:, :, None]
        noise_amp = voiced * self.noise_std + (1 - voiced) * self.sine_amp / 3
        if noise_keys is None:
            normal = torch.randn_like(sines)
        else:
            normal = keyed_normal(noise_keys, num_samples, self.num_harmonics)
        noise = noise_amp * normal
        if self.deterministic:
            noise = torch.zeros_like(noise)
        sine_waves = sines * self.sine_amp * voiced + noise
        if noise_keys is None:
            torch.randn_like(voiced)  # reference's unused noise branch

        source = torch.tanh(self.l_linear(sine_waves))[:, :, 0]
        return source.masked_fill(~length_mask(frame_lengths * self.upsample, num_samples), 0.0)


class Generator(nn.Module):
    """Two transposed-conv upsampling stages mixed with the STFT of the
    harmonic source, then an inverse-STFT head."""

    def __init__(self, config: KokoroModelConfig):
        super().__init__()
        cfg = config.istftnet
        style_dim = config.style_dim
        self.rates = list(cfg.upsample_rates)
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.num_upsamples = len(self.rates)
        self.m_source = SineSource(config)
        self.stft = RealSTFT(cfg.gen_istft_n_fft, cfg.gen_istft_hop_size)
        source_channels = cfg.gen_istft_n_fft + 2

        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        base = cfg.upsample_initial_channel
        for i, (rate, kernel) in enumerate(zip(self.rates, cfg.upsample_kernel_sizes, strict=True)):
            if kernel != 2 * rate:
                # With padding (k - u) // 2 only k == 2u upsamples by exactly
                # u; the frame bookkeeping below relies on that.
                raise ValueError(f"upsample kernel {kernel} must be twice the rate {rate}")
            channels = base // (2 ** (i + 1))
            self.ups.append(
                nn.ConvTranspose1d(base // (2**i), channels, kernel, rate, padding=(kernel - rate) // 2)
            )
            for k, dilations in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes, strict=True):
                self.resblocks.append(SnakeResBlock(channels, k, dilations, style_dim))
            if i + 1 < self.num_upsamples:
                stride = math.prod(self.rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        source_channels, channels, kernel_size=stride * 2, stride=stride, padding=(stride + 1) // 2
                    )
                )
                self.noise_res.append(SnakeResBlock(channels, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(source_channels, channels, kernel_size=1))
                self.noise_res.append(SnakeResBlock(channels, 11, [1, 3, 5], style_dim))
        self.conv_post = nn.Conv1d(channels, source_channels, 7, 1, padding=3)

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        f0: torch.Tensor,
        frame_lengths: torch.Tensor,
        noise_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, C, n]`` features and ``[B, n]`` F0 -> ``[B, n * upsample]`` waveform."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            source = self.m_source(f0.float(), frame_lengths, noise_keys)
            magnitude, phase, _ = self.stft.transform(source, frame_lengths * self.m_source.upsample)
        harmonics = torch.cat([magnitude, phase], dim=1).to(x.dtype)

        lengths = frame_lengths
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, GENERATOR_LEAKY_SLOPE))
            lengths = lengths * self.rates[i]
            if i == self.num_upsamples - 1:
                x = F.pad(x, (1, 0), mode="reflect")
                lengths = lengths + 1
            mask = length_mask(lengths, x.shape[-1])
            x = mask_channels(x, mask)
            x_source = mask_channels(self.noise_convs[i](harmonics), mask)
            x = x + self.noise_res[i](x_source, style, mask)
            acc = None
            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x, style, mask)
                acc = out if acc is None else acc + out
            x = acc / self.num_kernels

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = mask_channels(self.conv_post(F.leaky_relu(x.float())), mask)
            spec = torch.exp(x[:, : self.stft.bins])
            phase = torch.sin(x[:, self.stft.bins :])
            return self.stft.inverse(spec, phase, lengths)


class Decoder(nn.Module):
    """Aligned text features + F0 + energy -> waveform."""

    def __init__(self, config: KokoroModelConfig):
        super().__init__()
        hidden, style_dim = config.hidden_dim, config.style_dim
        trunk, res = config.decoder_hidden, config.asr_res_dim
        self.encode = AdainResBlk1d(hidden + 2, trunk, style_dim)
        self.decode = nn.ModuleList([
            AdainResBlk1d(trunk + 2 + res, trunk, style_dim),
            AdainResBlk1d(trunk + 2 + res, trunk, style_dim),
            AdainResBlk1d(trunk + 2 + res, trunk, style_dim),
            AdainResBlk1d(trunk + 2 + res, config.istftnet.upsample_initial_channel, style_dim, upsample=True),
        ])
        self.F0_conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        self.N_conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        self.asr_res = nn.Sequential(nn.Conv1d(hidden, res, kernel_size=1))
        self.generator = Generator(config)
        # Reduced precision for the convolutional trunk and the generator's
        # upsampling stages; the source and the STFT head stay fp32 regardless.
        dtypes = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}
        self.autocast_dtype = dtypes[config.decoder_dtype]

    def forward(
        self,
        asr: torch.Tensor,
        f0_curve: torch.Tensor,
        energy_curve: torch.Tensor,
        style: torch.Tensor,
        frame_lengths: torch.Tensor,
        noise_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``asr [B, hidden, F]``, curves ``[B, 2F]`` -> ``[B, F * samples_per_frame]``."""
        autocast = torch.autocast(
            device_type=asr.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None
        )
        with autocast:
            mask = length_mask(frame_lengths, asr.shape[-1])
            f0 = mask_channels(self.F0_conv(f0_curve[:, None]), mask)
            energy = mask_channels(self.N_conv(energy_curve[:, None]), mask)
            x, lengths = self.encode(torch.cat([asr, f0, energy], dim=1), style, frame_lengths)
            asr_res = mask_channels(self.asr_res(asr), mask)
            for block in self.decode:
                x, lengths = block(torch.cat([x, asr_res, f0, energy], dim=1), style, lengths)
            return self.generator(x, style, f0_curve, lengths, noise_keys)
