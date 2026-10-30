# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
# Copyright (c) 2024 Alibaba Inc (Xiang Lyu)
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: rebuilt from causal pieces on
# Chatterbox's HiFT, with bounded-window streaming.
#
# Original files (FunAudioLLM/CosyVoice, cosyvoice/hifigan/generator.py,
# cosyvoice/hifigan/f0_predictor.py and cosyvoice/transformer/convolution.py)
# were released under Apache-2.0, with the full license text available at
# https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE. The generator is
# modified from jik876/hifi-gan (MIT).
#
# This modified file is released under the same license.

"""CosyVoice3's causal HiFT vocoder: 80-bin mel at 50 Hz -> 24 kHz waveform.

The same neural-source-filter HiFi-GAN with an iSTFT head as Chatterbox's
HiFT (``Snake`` activations and the STFT helpers are shared), rebuilt from
causal pieces so it can stream:

* every convolution pads on the left only, except the F0 predictor's first one
  (three frames of look-ahead) and ``conv_pre`` (four frames of look-ahead);
* the harmonic source is a sample-and-hold sine per 20 ms frame whose phase is
  the running sum of the frame frequencies, plus noise read from one fixed
  field by absolute output sample (the reference draws ``rand(1, 300 s, 9)``
  once and slices it from sample 0 on every call);
* the F0 predictor runs in float64, as the reference insists.

A non-final call on ``N`` frames returns the first ``(N - 8) * 480`` samples,
which no later frame can change; a final call returns all ``N * 480``.

``stream`` vocodes a window instead of the whole history: the frames from
``start`` on, the F0 predictor fed its eight frames of left history, the
phase sum carried in from before ``start`` and the noise read at the window's
absolute offset. Given enough left context (``start`` far enough before the
first sample to emit) the samples match a whole-history call to float
rounding, at a cost that no longer grows with the utterance.

The reference also adds a random phase offset to sample 0 of each harmonic;
its linear resampling to frame rate never reads sample 0, so it has no effect
and is left out.

Ported from ``cosyvoice/hifigan/generator.py`` (``CausalHiFTGenerator``,
``SineGen2``, ``SourceModuleHnNSF``), ``cosyvoice/hifigan/f0_predictor.py``
(``CausalConvRNNF0Predictor``) and ``cosyvoice/transformer/convolution.py``.
Parameter paths match ``hift.pt`` (weight norm folded).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.chatterbox.components.s3gen_hift import Snake
from mstar.model.chatterbox.loader import fold_weight_norm, load_component
from mstar.model.cosyvoice3.config import CausalHiFTConfig


def _causal_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size * dilation - dilation) // 2 * 2 + (kernel_size + 1) % 2


class CausalConv1d(nn.Conv1d):
    """Stride-1 convolution padded on one side only: ``left`` sees the past,
    ``right`` the next ``kernel_size - 1`` steps (zeros, or ``context``)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, side: str = "left"):
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation)
        self.causal_padding = _causal_padding(kernel_size, dilation)
        self.side = side

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.side == "left":
            return super().forward(F.pad(x, (self.causal_padding, 0)))
        if context is not None:
            x_ctx = torch.cat([x, context], dim=2)
            return super().forward(F.pad(x_ctx, (0, self.causal_padding - context.shape[2])))
        return super().forward(F.pad(x, (0, self.causal_padding)))


class CausalUpsample(nn.Conv1d):
    """Nearest x``stride`` upsample, then a left-padded convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__(in_channels, out_channels, kernel_size)
        self.stride_up = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=float(self.stride_up), mode="nearest")
        return super().forward(F.pad(x, (self.kernel_size[0] - 1, 0)))


class CausalDownsample(nn.Conv1d):
    """Strided convolution with ``stride - 1`` zeros on the left."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.stride[0] - 1, 0)))


class CausalResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList([CausalConv1d(channels, channels, kernel_size, d) for d in dilations])
        self.convs2 = nn.ModuleList([CausalConv1d(channels, channels, kernel_size, 1) for _ in dilations])
        self.activations1 = nn.ModuleList([Snake(channels) for _ in dilations])
        self.activations2 = nn.ModuleList([Snake(channels) for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = zip(self.activations1, self.convs1, self.activations2, self.convs2, strict=True)
        for act1, conv1, act2, conv2 in layers:
            x = conv2(act2(conv1(act1(x)))) + x
        return x


class CausalF0Predictor(nn.Module):
    """Five conv + ELU layers (the first reads three frames ahead) and a linear head."""

    def __init__(self, in_channels: int, cond_channels: int):
        super().__init__()
        layers: list[nn.Module] = [CausalConv1d(in_channels, cond_channels, 4, side="right"), nn.ELU()]
        for _ in range(4):
            layers += [CausalConv1d(cond_channels, cond_channels, 3), nn.ELU()]
        self.condnet = nn.Sequential(*layers)
        self.classifier = nn.Linear(cond_channels, 1)

    @property
    def look_right(self) -> int:
        return self.condnet[0].causal_padding

    @property
    def left_history(self) -> int:
        return sum(m.causal_padding for m in self.condnet[2::2])

    def forward(self, mel: torch.Tensor, finalize: bool = True) -> torch.Tensor:
        """``[B, 80, T]`` -> ``|f0|`` ``[B, T]``, or ``[B, T - 3]`` when not final
        (the last three frames are only look-ahead)."""
        if finalize:
            x = self.condnet[0](mel)
        else:
            r = self.look_right
            x = self.condnet[0](mel[:, :, :-r], context=mel[:, :, -r:])
        for layer in self.condnet[1:]:
            x = layer(x)
        return torch.abs(self.classifier(x.transpose(1, 2)).squeeze(-1))


@dataclass
class VocoderWindow:
    """Where a call's frames sit in the utterance (all counts global)."""

    start: int = 0  # first mel frame of the window
    f0_history: int = 0  # frames before ``start`` fed to the F0 predictor only
    phase_carry: torch.Tensor | None = None  # [B, H+1] cycle sum of the frames before ``start``


class CausalHiFTGenerator(nn.Module):
    def __init__(self, config: CausalHiFTConfig):
        super().__init__()
        self.config = config
        cfg = config
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.num_upsamples = len(cfg.upsample_rates)
        self.m_source = nn.Module()
        self.m_source.l_linear = nn.Linear(cfg.nb_harmonics + 1, 1)
        self.f0_predictor = CausalF0Predictor(cfg.in_channels, cfg.f0_cond_channels)
        self.conv_pre = CausalConv1d(cfg.in_channels, cfg.base_channels, cfg.conv_pre_look_right + 1, side="right")
        self.ups = nn.ModuleList(
            [
                CausalUpsample(cfg.base_channels // (2**i), cfg.base_channels // (2 ** (i + 1)), k, u)
                for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes, strict=True))
            ]
        )
        rates = [1] + list(cfg.upsample_rates[::-1][:-1])
        cum = [math.prod(rates[: i + 1]) for i in range(len(rates))][::-1]
        n_bins = cfg.istft_n_fft + 2
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        for i, (u, k, d) in enumerate(
            zip(
                cum,
                cfg.source_resblock_kernel_sizes,
                cfg.source_resblock_dilation_sizes,
                strict=True,
            )
        ):
            ch = cfg.base_channels // (2 ** (i + 1))
            self.source_downs.append(CausalConv1d(n_bins, ch, 1) if u == 1 else CausalDownsample(n_bins, ch, u * 2, u))
            self.source_resblocks.append(CausalResBlock(ch, k, tuple(d)))
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = cfg.base_channels // (2 ** (i + 1))
            for k, d in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes, strict=True):
                self.resblocks.append(CausalResBlock(ch, k, tuple(d)))
        self.conv_post = CausalConv1d(ch, n_bins, 7)
        window = torch.hann_window(cfg.istft_n_fft, periodic=True, dtype=torch.float64).float()
        self.register_buffer("stft_window", window, persistent=False)
        self.register_buffer(
            "harmonics",
            torch.arange(1, cfg.nb_harmonics + 2, dtype=torch.float32),
            persistent=False,
        )
        # The excitation noise, by output sample (about 260 MB for 300 s); it
        # moves with the module, since a per-call copy from the host waits on it
        generator = torch.Generator().manual_seed(cfg.noise_seed)
        self.register_buffer(
            "source_noise",
            torch.rand(1, int(cfg.noise_seconds * cfg.sampling_rate), cfg.nb_harmonics + 1, generator=generator),
            persistent=False,
        )
        # torch.istft checks the window envelope on the host every call; the
        # envelope depends only on the window, so it is checked here once
        frames = cfg.istft_n_fft // cfg.istft_hop_len + 1
        if window.device.type != "meta" and self._window_envelope(frames, window).min() < 1e-11:
            raise ValueError("the iSTFT window does not overlap-add to a nonzero envelope")

    @property
    def upsample_factor(self) -> int:
        return self.config.upsample_factor

    # -- excitation ------------------------------------------------------------

    def harmonic_source(
        self,
        f0: torch.Tensor,
        window: VocoderWindow,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``f0`` ``[B, F]`` of the window's frames -> (excitation ``[B, 1, 480F]``,
        cycle sums ``[B, F, H+1]`` through each frame)."""
        cfg = self.config
        up = self.upsample_factor
        b, frames = f0.shape
        rad = torch.remainder(f0.unsqueeze(-1) * self.harmonics / cfg.sampling_rate, 1.0)  # [B, F, H+1]
        carry = window.phase_carry
        if carry is None:
            carry = torch.zeros(b, rad.shape[-1], dtype=rad.dtype, device=rad.device)
        cycles = torch.cumsum(torch.cat([carry.to(rad.dtype).unsqueeze(1), rad], dim=1), dim=1)[:, 1:]
        phase = (cycles * 2 * math.pi * up).repeat_interleave(up, dim=1)  # [B, 480F, H+1]
        sines = torch.sin(phase) * cfg.nsf_alpha
        f0_samples = f0.repeat_interleave(up, dim=1).unsqueeze(-1)
        uv = (f0_samples > cfg.nsf_voiced_threshold).to(sines.dtype)
        offset = window.start * up
        length = frames * up
        if offset + length > self.source_noise.shape[1]:
            raise ValueError(f"the vocoder noise covers {cfg.noise_seconds} s; this call reaches past it")
        noise = self.source_noise[:, offset : offset + length].to(sines.device, sines.dtype)
        noise_amp = uv * cfg.nsf_sigma + (1 - uv) * cfg.nsf_alpha / 3
        excitation = sines * uv + noise_amp * noise
        merged = torch.tanh(self.m_source.l_linear(excitation)).transpose(1, 2)
        return merged, cycles

    # -- spectrum and waveform -------------------------------------------------

    def _stft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        spec = torch.stft(
            x,
            cfg.istft_n_fft,
            cfg.istft_hop_len,
            cfg.istft_n_fft,
            window=self.stft_window,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _overlap_add(self, frames: torch.Tensor) -> torch.Tensor:
        """``[B, T, n_fft]`` frames -> ``[B, n_fft + hop (T - 1)]`` samples less the
        centre padding, summed as ``torch.istft`` sums them (the backward of
        ``unfold``), so the result is the same to the bit."""
        n_fft, hop = self.config.istft_n_fft, self.config.istft_hop_len
        length = n_fft + hop * (frames.shape[1] - 1)
        out = torch.ops.aten.unfold_backward(frames, [frames.shape[0], length], 1, n_fft, hop)
        return out[:, n_fft // 2 : length - n_fft // 2]

    def _window_envelope(self, num_frames: int, window: torch.Tensor) -> torch.Tensor:
        return self._overlap_add((window**2).expand(1, num_frames, -1).contiguous())[0]

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """``torch.istft`` (centred, one-sided, unnormalized) without its host-side envelope check."""
        cfg = self.config
        magnitude = torch.clip(magnitude, max=1e2)
        spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
        frames = torch.fft.irfft(spec.transpose(1, 2), n=cfg.istft_n_fft, dim=-1) * self.stft_window
        return self._overlap_add(frames) / self._window_envelope(frames.shape[1], self.stft_window)

    def decode(self, mel: torch.Tensor, source: torch.Tensor, finalize: bool) -> torch.Tensor:
        cfg = self.config
        s_real, s_imag = self._stft(source.squeeze(1))
        if finalize:
            x = self.conv_pre(mel)
        else:
            r = cfg.conv_pre_look_right
            x = self.conv_pre(mel[:, :, :-r], context=mel[:, :, -r:])
            cut = math.prod(cfg.upsample_rates) * r
            s_real, s_imag = s_real[:, :, :-cut], s_imag[:, :, :-cut]
        s_stft = torch.cat([s_real, s_imag], dim=1)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, cfg.lrelu_slope))
            if i == self.num_upsamples - 1:
                x = F.pad(x, (1, 0), mode="reflect")
            x = x + self.source_resblocks[i](self.source_downs[i](s_stft))
            xs = None
            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        half = cfg.istft_n_fft // 2 + 1
        wav = self._istft(torch.exp(x[:, :half, :]), torch.sin(x[:, half:, :]))
        if not finalize:
            wav = wav[:, : -self.upsample_factor]
        return torch.clamp(wav, -cfg.audio_limit, cfg.audio_limit)

    @torch.no_grad()
    def vocode(
        self,
        mel: torch.Tensor,
        finalize: bool = True,
        window: VocoderWindow | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``mel`` ``[B, 80, history + F]`` (the window's frames after
        ``window.f0_history`` frames of F0-only history) -> (waveform from the
        window's first frame on, cycle sums ``[B, F', H+1]`` of its frames).

        Whole-utterance call (no window): the reference ``inference``."""
        window = window or VocoderWindow()
        f0 = self.f0_predictor(mel.to(torch.float64), finalize=finalize).to(mel.dtype)
        f0 = f0[:, window.f0_history :]
        mel = mel[:, :, window.f0_history :]
        source, cycles = self.harmonic_source(f0, window)
        if not finalize:
            mel = mel[:, :, : -self.f0_predictor.look_right]
        return self.decode(mel, source, finalize), cycles

    def keep_f0_predictor_in_float64(self) -> None:
        self.f0_predictor.double()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream ``hift.pt`` (weight-norm pairs folded) into this module."""
        return load_component(self, fold_weight_norm(weights), component="CosyVoice3 HiFT")


@dataclass
class HiftStreamState:
    """One request's streaming vocoder state: the mel frames still needed (from
    global frame ``mel_start``), the phase carried to ``carry_frame`` and how
    many samples went out."""

    mel: torch.Tensor  # [80, F]
    mel_start: int = 0
    carry: torch.Tensor | None = None  # [H+1]
    carry_frame: int = 0
    emitted: int = 0  # samples
    done: bool = False

    @property
    def mel_end(self) -> int:
        return self.mel_start + int(self.mel.shape[-1])


def stream_vocode(
    hift: CausalHiFTGenerator,
    state: HiftStreamState,
    new_mel: torch.Tensor,
    final: bool,
    context_frames: int,
) -> torch.Tensor:
    """Append ``new_mel`` ``[80, F]`` and return the samples that are final now
    (all of them when ``final``). ``context_frames <= 0`` re-vocodes the whole
    history every call, as the reference does."""
    up = hift.upsample_factor
    if state.done:
        return new_mel.new_zeros(0)
    state.mel = torch.cat([state.mel, new_mel.to(state.mel)], dim=-1)
    end = state.mel_end
    if end == 0:
        state.done = final
        return new_mel.new_zeros(0)
    holdback = hift.config.stream_holdback_frames
    emit_to = end * up if final else max(end - holdback, 0) * up
    if emit_to <= state.emitted and not final:
        return new_mel.new_zeros(0)
    emitted_frames = state.emitted // up
    start = 0 if context_frames <= 0 else max(emitted_frames - context_frames, 0)
    if start != state.carry_frame:
        raise RuntimeError(f"vocoder phase carried to frame {state.carry_frame}, window starts at {start}")
    history = min(hift.f0_predictor.left_history, start)
    lo = start - history - state.mel_start
    if lo < 0:
        raise RuntimeError("streaming vocoder dropped mel it still needs")
    window = VocoderWindow(
        start=start,
        f0_history=history,
        phase_carry=None if state.carry is None else state.carry[None],
    )
    wav, cycles = hift.vocode(state.mel[None, :, lo:], finalize=final, window=window)
    out = wav[0, state.emitted - start * up : emit_to - start * up]
    state.emitted = emit_to
    if final:
        state.done = True
        return out
    next_start = 0 if context_frames <= 0 else max(state.emitted // up - context_frames, 0)
    if next_start > start:
        state.carry = cycles[0, next_start - start - 1]
        state.carry_frame = next_start
    keep_from = next_start - min(hift.f0_predictor.left_history, next_start)
    if keep_from > state.mel_start:
        state.mel = state.mel[:, keep_from - state.mel_start :]
        state.mel_start = keep_from
    return out
