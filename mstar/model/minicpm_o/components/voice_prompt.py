"""Token2wav's voice prompt: a reference clip -> speech tokens, speaker x-vector and 24 kHz mel.

Ported from Step-Audio2's ``Token2wav._prepare_prompt`` (``minicpmo-utils``, Apache-2.0). The
three encoders are Chatterbox's native ports of the same CosyVoice-family networks, loaded
from the ONNX files MiniCPM-o ships (``speech_tokenizer_v2_25hz.onnx``, ``campplus.onnx``):

- tokens: ``S3Tokenizer`` with the ``s3tokenizer`` package's current quantizer, which clamps
  the tanh output before rounding (Chatterbox's ``FSQuantizer`` scales it by 0.999 instead),
  and its attention LayerNorm eps of 1e-6 (the ONNX graph and Chatterbox use 1e-5);
- x-vector: ``CAMPPlus`` on the mean-normalised Kaldi fbank;
- mel: ``MelSpectrogram24k`` of the clip resampled 16 -> 24 kHz, replicate-padded (or cut) to
  two frames per prompt token, which is what the flow's prompt condition must be.

Both ONNX files load through CosyVoice3's readers (``cosyvoice3.components.onnx_weights``),
which need no ``onnx`` package.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from mstar.model.chatterbox.components.audio import MelSpectrogram24k, kaldi_fbank_80
from mstar.model.chatterbox.components.s3_tokenizer import S3Tokenizer
from mstar.model.chatterbox.components.s3gen_xvector import CAMPPlus
from mstar.model.chatterbox.config import S3GenMelConfig, S3GenXVectorConfig, S3TokenizerConfig

PROMPT_SR = 16000
MEL_SR = 24000
FRAMES_PER_TOKEN = 2


class VoicePrompt(NamedTuple):
    """One voice's conditioning, computed once per reference clip."""

    tokens: torch.Tensor  # [1, P] int32 s3 tokens
    spk_emb: torch.Tensor  # [1, 192] raw CAMPPlus x-vector
    mel: torch.Tensor  # [1, 2P, 80] 24 kHz log-mel


class ClampedFSQuantizer(nn.Module):
    """``s3tokenizer``'s FSQ: tanh, clamp to ``(-1, 1)``, round to ``{0, 1, 2}``, base-3 code."""

    def __init__(self, dim: int, fsq_dim: int = 8, levels: int = 3):
        super().__init__()
        self.project_down = nn.Linear(dim, fsq_dim)
        self.levels = levels
        self.register_buffer("powers", levels ** torch.arange(fsq_dim, dtype=torch.int64), persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        h = torch.tanh(self.project_down(hidden).float())
        h = torch.clamp(h, -1 + eps, 1 - eps)
        h = ((h + 1.0) * (self.levels - 1) / 2.0).round().to(torch.int64)
        return (h * self.powers).sum(dim=-1)


class PromptTokenizer(S3Tokenizer):
    """Chatterbox's S3TokenizerV2 with the quantizer and LayerNorm eps the reference runs."""

    def __init__(self, config: S3TokenizerConfig | None = None):
        super().__init__(config)
        self.quantizer = ClampedFSQuantizer(self.config.n_state, self.config.fsq_dim, self.config.fsq_levels)
        for block in self.encoder.blocks:
            block.attn_ln.eps = 1e-6


class VoicePromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = PromptTokenizer()
        self.speaker = CAMPPlus(S3GenXVectorConfig())
        self.mel = MelSpectrogram24k(S3GenMelConfig())
        self.resample = torchaudio.transforms.Resample(PROMPT_SR, MEL_SR)

    @torch.inference_mode()
    def forward(self, wav16: torch.Tensor) -> VoicePrompt:
        """``wav16 [T]`` float32 at 16 kHz. Each part runs on its module's device."""
        tok_dev = self.tokenizer.window.device
        log_mel = self.tokenizer.log_mel(wav16.to(tok_dev))
        tokens, _ = self.tokenizer.encode_mel(log_mel, torch.tensor([log_mel.shape[-1]], device=tok_dev))
        tokens = tokens.int()

        spk_dev = next(self.speaker.parameters()).device
        spk_emb = self.speaker(kaldi_fbank_80(wav16.to(spk_dev))[None])

        mel_dev = self.mel.window.device
        mel = self.mel(self.resample(wav16.to(mel_dev)[None])).transpose(1, 2)
        mel = F.pad(mel, (0, 0, 0, tokens.shape[1] * FRAMES_PER_TOKEN - mel.shape[1]), mode="replicate")
        return VoicePrompt(tokens=tokens, spk_emb=spk_emb, mel=mel)
