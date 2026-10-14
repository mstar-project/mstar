"""Speaker embedding for T3's conditioning (Real-Time-Voice-Cloning encoder).

A 3-layer LSTM reads 40-bin mel partials of the reference utterance; the last
hidden state is projected, rectified and L2-normalised per partial, and the
utterance embedding is the normalised mean over partials. Read from
``chatterbox/models/voice_encoder/{voice_encoder.py,melspec.py,config.py}``;
the mel front end is the librosa recipe of ``melspec.melspectrogram`` (power
spectrum, slaney mel filters, no dB), the partial layout is
``get_num_wins`` / ``get_frame_step`` with the utterance rate the TTS class
passes (1.3 partials per second), and the leading/trailing silence trim is the
``librosa.effects.trim(top_db=20)`` that ``embeds_from_wavs`` applies.
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.chatterbox.components.audio_frontend import (
    mel_filter_bank,
    stft_power,
    trim_silence,
)
from mstar.model.chatterbox.config import VoiceEncoderConfig
from mstar.model.chatterbox.loader import DEFAULT_SKIP_FRAGMENTS, WeightStream, load_component


class VoiceEncoder(nn.Module):
    # training-only scalars in ``ve.safetensors``
    _SKIP = DEFAULT_SKIP_FRAGMENTS + ("similarity_weight", "similarity_bias")

    def __init__(self, config: VoiceEncoderConfig | None = None):
        super().__init__()
        self.config = config or VoiceEncoderConfig()
        cfg = self.config
        self.lstm = nn.LSTM(
            cfg.num_mels, cfg.hidden_size, num_layers=cfg.num_layers, batch_first=True,
        )
        self.proj = nn.Linear(cfg.hidden_size, cfg.speaker_embed_size)
        self.register_buffer(
            "mel_basis",
            mel_filter_bank(cfg.sample_rate, cfg.n_fft, cfg.num_mels, cfg.fmin, cfg.fmax),
            persistent=False,
        )
        self.register_buffer(
            "window", torch.hann_window(cfg.win_size, periodic=True), persistent=False,
        )
        # frames between the starts of two partials, for the configured rate
        self.frame_step = int(round((cfg.sample_rate / cfg.partials_rate) / cfg.hop_size))
        assert 0 < self.frame_step <= cfg.partial_frames

    # ---------------------------------------------------------------- front end

    def mel(self, wav16: torch.Tensor) -> torch.Tensor:
        """``[T]`` 16 kHz waveform -> ``[frames, num_mels]`` power mel (no log)."""
        cfg = self.config
        power = stft_power(
            wav16.float().unsqueeze(0), cfg.n_fft, cfg.hop_size, cfg.win_size, self.window,
        )[0]
        if cfg.mel_power != 2.0:
            power = power.sqrt().pow(cfg.mel_power)
        return (self.mel_basis @ power).transpose(0, 1)

    def num_partials(self, n_frames: int) -> tuple[int, int]:
        """(partials, frames those partials cover) for an utterance of
        ``n_frames`` mel frames; the reference ``get_num_wins``."""
        cfg = self.config
        win, step = cfg.partial_frames, self.frame_step
        n_wins, remainder = divmod(max(n_frames - win + step, 0), step)
        if n_wins == 0 or (remainder + (win - step)) / win >= cfg.min_coverage:
            n_wins += 1
        return n_wins, win + step * (n_wins - 1)

    def mel_partials(self, wav16: torch.Tensor) -> torch.Tensor:
        """``[T]`` -> ``[n_partials, partial_frames, num_mels]`` overlapping
        windows of the mel, zero-padded (or trimmed) to whole partials."""
        mel = self.mel(wav16)
        n_partials, target = self.num_partials(mel.shape[0])
        if target > mel.shape[0]:
            mel = torch.cat([mel, mel.new_zeros(target - mel.shape[0], mel.shape[1])])
        elif target < mel.shape[0]:
            mel = mel[:target]
        return mel.unfold(0, self.config.partial_frames, self.frame_step).transpose(1, 2)

    # ---------------------------------------------------------------- network

    def forward(self, partials: torch.Tensor) -> torch.Tensor:
        """``[N, partial_frames, num_mels]`` -> ``[N, speaker_embed_size]``,
        one L2-normalised embedding per partial."""
        _, (hidden, _) = self.lstm(partials)
        embeds = self.proj(hidden[-1])
        if self.config.final_relu:
            embeds = torch.relu(embeds)
        return embeds / embeds.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def embed_utterance(self, wav16: torch.Tensor, trim_top_db: float | None = 20.0) -> torch.Tensor:
        """``[T]`` 16 kHz waveform -> ``[speaker_embed_size]`` speaker embedding
        (mean of the partial embeddings, re-normalised)."""
        if trim_top_db:
            wav16 = trim_silence(wav16, top_db=trim_top_db)
        partials = self.mel_partials(wav16)
        embed = self(partials).mean(dim=0)
        return embed / embed.norm()

    # ---------------------------------------------------------------- weights

    def load_weights(self, weights: WeightStream) -> set[str]:
        return load_component(
            self, weights, component="Chatterbox voice encoder", skip_fragments=self._SKIP,
        )
