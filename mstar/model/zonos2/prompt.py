"""Byte-level prompt construction for Zonos2 TTS.

Ported from ``../ZONOS2/python/zonos2/tts/prompt.py``. It is trimmed to
the default path: UTF-8 byte tokenization plus a sheared silence tail. It
omits the optional speaking-rate, quality, and speaker-background
conditioning tokens.

# TODO: add the optional speaking-rate, quality, and speaker-background
conditioning

A prompt is a 2-D int tensor of shape ``(num_frames, n_codebooks + 1)``.
For each byte token the audio columns hold ``audio_pad_id``. The final
(text) column carries the byte-token id. The code appends a pre-computed
0.2 s silence tail (17 frames) with the inter-codebook shear pattern. So
generation starts from silence.
"""
from __future__ import annotations

import torch

# Text vocabulary layout: 192 legacy symbol ids, then 256 UTF-8 byte ids.
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
LEGACY_SYMBOL_VOCAB_SIZE = 192
BYTE_VOCAB_SIZE = 256
BYTE_TEXT_VOCAB_SIZE = LEGACY_SYMBOL_VOCAB_SIZE + BYTE_VOCAB_SIZE  # 448

# Pre-computed silence tokens for 0.2 s at 44.1 kHz (17 frames x 9 codebooks).
_SILENCE_TOKENS_0_2S = [
    [568, 778, 338, 524, 967, 360, 728, 550, 90],
    [568, 778, 10, 674, 364, 981, 741, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 804, 10, 674, 364, 981, 568, 378, 731],
    [568, 778, 721, 842, 264, 974, 989, 507, 308],
]


def text_to_byte_ids(text: str) -> list[int]:
    """BOS + UTF-8 bytes (offset past the legacy symbol block) + EOS."""
    return [BOS_ID, *(b + LEGACY_SYMBOL_VOCAB_SIZE for b in text.encode("utf-8")), EOS_ID]


def shear(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Apply the inter-codebook delay: shift column ``j`` down by ``j``.

    ``x`` is ``(T, C)``. The output is ``(C - 1 + T, C)``. ``pad`` fills the
    delayed positions. This is the inverse of ``vocoder.shear_up``.
    """
    T, C = x.shape
    padded = x.new_full((C - 1 + T, C), pad)
    padded[C - 1:] = x
    row_idx = (C - 1) + torch.arange(T, device=x.device).unsqueeze(1) - torch.arange(
        C, device=x.device
    )
    return padded.gather(0, row_idx)


def silence_prompt_tokens(
    n_codebooks: int, audio_pad_id: int, text_vocab: int,
) -> torch.Tensor:
    """Sheared 0.2 s silence tail with a text-padding column."""
    silence = torch.tensor(_SILENCE_TOKENS_0_2S, dtype=torch.int32)
    sheared = shear(silence[:, :n_codebooks], audio_pad_id)
    text_col = torch.full((sheared.shape[0], 1), text_vocab, dtype=torch.int32)
    return torch.cat([sheared, text_col], dim=1)


class TTSPromptBuilder:
    """Build the 2-D prompt frame tensor for a text string."""

    def __init__(
        self,
        n_codebooks: int = 9,
        audio_pad_id: int = 1025,
        text_vocab: int = BYTE_TEXT_VOCAB_SIZE,
        prepend_silence: bool = True,
    ):
        if text_vocab < BYTE_TEXT_VOCAB_SIZE:
            raise ValueError(
                f"text_vocab ({text_vocab}) must be >= byte vocab {BYTE_TEXT_VOCAB_SIZE}."
            )
        self.n_codebooks = n_codebooks
        self.audio_pad_id = audio_pad_id
        self.text_vocab = text_vocab
        self._silence = (
            silence_prompt_tokens(n_codebooks, audio_pad_id, text_vocab)
            if prepend_silence
            else None
        )

    def build_text_prompt(self, text: str) -> torch.Tensor:
        rows = [
            [self.audio_pad_id] * self.n_codebooks + [token]
            for token in text_to_byte_ids(text)
        ]
        return torch.tensor(rows, dtype=torch.int32)

    def build(self, text: str) -> torch.Tensor:
        """Return ``(num_frames, n_codebooks + 1)`` int32 prompt frames."""
        prompt = self.build_text_prompt(text)
        if self._silence is not None:
            prompt = torch.cat([prompt, self._silence], dim=0)
        return prompt
