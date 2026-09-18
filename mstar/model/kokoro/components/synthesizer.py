"""The whole Kokoro network: phonemes + style -> waveform, batched and masked.

Two halves, split where the frame count becomes known:

* ``encode_text`` is shaped by the phoneme count ``T``: PL-BERT, the prosody
  encoder, the duration head and the text encoder.
* ``align`` expands phoneme features to frames once ``F`` is known, and
  ``decode_frames`` is shaped by ``F``: the F0 and energy heads and the decoder.

``forward`` joins them with the single host read of ``F``. The submodule
captures ``encode_text`` and ``decode_frames`` as CUDA graphs per bucket.
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.kokoro.components.albert import PLBert
from mstar.model.kokoro.components.decoder import Decoder
from mstar.model.kokoro.components.masking import length_mask, mask_channels, mask_features
from mstar.model.kokoro.components.prosody import ProsodyPredictor, TextEncoder
from mstar.model.kokoro.config import KokoroModelConfig


class KokoroTTS(nn.Module):
    def __init__(self, config: KokoroModelConfig):
        super().__init__()
        self.config = config
        self.bert = PLBert(config.plbert, config.n_token)
        self.bert_encoder = nn.Linear(config.plbert.hidden_size, config.hidden_dim)
        self.predictor = ProsodyPredictor(config)
        self.text_encoder = TextEncoder(config)
        self.decoder = Decoder(config)

    def _split_style(self, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A pack row is [decoder style, predictor style]."""
        return style[:, : self.config.style_dim], style[:, self.config.style_dim :]

    def encode_text(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        style: torch.Tensor,
        speed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, T]`` ids -> prosody states ``[B, T, hidden + style]``, text features
        ``[B, hidden, T]`` and integer durations ``[B, T]``."""
        _, predictor_style = self._split_style(style)
        d_en = self.bert_encoder(self.bert(input_ids, lengths))
        d = self.predictor.encode(d_en, predictor_style, lengths)
        pred_dur = self.predictor.durations(d, lengths, speed)
        t_en = self.text_encoder(input_ids, lengths)
        return d, t_en, pred_dur

    @staticmethod
    def alignment(pred_dur: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Index of the phoneme that owns each frame, ``[B, num_frames]``.

        Equal to the reference's ``repeat_interleave`` + one-hot matmul on the
        valid frames; frames past a row's total point at its last phoneme.
        """
        cumulative = pred_dur.cumsum(dim=1)
        frames = torch.arange(num_frames, device=pred_dur.device)[None, :].expand(pred_dur.shape[0], -1)
        return torch.searchsorted(cumulative, frames.contiguous(), right=True).clamp(max=pred_dur.shape[1] - 1)

    def align(
        self,
        d: torch.Tensor,
        t_en: torch.Tensor,
        pred_dur: torch.Tensor,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand phoneme features to ``num_frames`` frames.

        Returns the frame-aligned prosody states ``[B, F, hidden + style]``, the
        aligned text features ``[B, hidden, F]`` and each row's valid frame count.
        """
        frame_lengths = pred_dur.sum(dim=1)
        token_index = self.alignment(pred_dur, num_frames)
        mask = length_mask(frame_lengths, num_frames)
        en = mask_features(d.gather(1, token_index[:, :, None].expand(-1, -1, d.shape[-1])), mask)
        asr = mask_channels(t_en.gather(2, token_index[:, None, :].expand(-1, t_en.shape[1], -1)), mask)
        return en, asr, frame_lengths

    def decode_frames(
        self,
        en: torch.Tensor,
        asr: torch.Tensor,
        frame_lengths: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        """Frame-aligned features -> waveform ``[B, F * samples_per_frame]``."""
        decoder_style, predictor_style = self._split_style(style)
        f0, energy = self.predictor.f0n(en, predictor_style, frame_lengths)
        return self.decoder(asr, f0, energy, decoder_style, frame_lengths)

    def synthesize_frames(
        self,
        d: torch.Tensor,
        t_en: torch.Tensor,
        pred_dur: torch.Tensor,
        style: torch.Tensor,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``align`` then ``decode_frames``: the waveform and each row's frame count."""
        en, asr, frame_lengths = self.align(d, t_en, pred_dur, num_frames)
        return self.decode_frames(en, asr, frame_lengths, style), frame_lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        style: torch.Tensor,
        speed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager end-to-end synthesis: (waveform, frame lengths, durations)."""
        d, t_en, pred_dur = self.encode_text(input_ids, lengths, style, speed)
        num_frames = int(pred_dur.sum(dim=1).max().item())
        audio, frame_lengths = self.synthesize_frames(d, t_en, pred_dur, style, num_frames)
        return audio, frame_lengths, pred_dur
