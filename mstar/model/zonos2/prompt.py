"""Byte-level prompt construction for Zonos2 TTS.

This is a port of ``../ZONOS2/python/zonos2/tts/prompt.py``.

A prompt is a 2-D int tensor of shape ``(num_frames, n_codebooks + 1)``. For
each byte token, the audio columns hold ``audio_pad_id`` and the last (text)
column holds the byte-token id. The builder appends a pre-computed 0.2 s
silence tail of 17 frames with the inter-codebook shear pattern. Generation
therefore starts from silence.

The conditioning tokens occupy the tail of the text vocabulary in this order:
the speaking-rate buckets, the quality buckets (one block for each feature),
the clean/noisy background pair, then the accurate-mode marker. ``text_vocab``
is the text padding id. Each count shifts the ids after it, so
:func:`_conditioning_base_text_vocab` subtracts all four counts before it
locates any token.

With a speaker embedding, the frame layout agrees with training. See
``../ZONOS2/python/zonos2/tokenizer/server.py:121`` for the slot, and
``.../scheduler/scheduler.py:372`` for the markers::

    0   speaker slot        [pad x C, text_vocab]  <- model overwrites the state
    1   speaker background  [pad x C, clean|noisy]
    2   accurate mode       [pad x C, accurate]    (only on request)
    3   speaking rate       [pad x C, rate_tok]    (optional)
    4+  quality buckets     [pad x C, qual_tok] x features
        BOS, UTF-8 bytes, EOS
        17 frames of sheared 0.2 s silence

The speaker slot holds no information. It is an all-padding row, and the model
replaces its hidden state with the projected speaker embedding. See
``Zonos2ForCausalLM.forward``.
"""
from __future__ import annotations

import torch

# The text vocabulary holds 192 legacy symbol ids, then 256 UTF-8 byte ids.
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
LEGACY_SYMBOL_VOCAB_SIZE = 192
BYTE_VOCAB_SIZE = 256
BYTE_TEXT_VOCAB_SIZE = LEGACY_SYMBOL_VOCAB_SIZE + BYTE_VOCAB_SIZE  # 448

# Pre-computed silence tokens for 0.2 s at 44.1 kHz: 17 frames x 9 codebooks.
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
    """Return BOS, then the UTF-8 bytes, then EOS.

    The byte ids start after the legacy symbol block.
    """
    return [BOS_ID, *(b + LEGACY_SYMBOL_VOCAB_SIZE for b in text.encode("utf-8")), EOS_ID]


def _normalize_quality_bucket_counts(quality_bucket_counts) -> tuple[int, ...]:
    counts = tuple(int(count) for count in (quality_bucket_counts or ()))
    if any(count < 0 for count in counts):
        raise ValueError("quality_bucket_counts must be non-negative.")
    return counts


def _conditioning_base_text_vocab(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts=(),
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
    *,
    context: str,
) -> int:
    """Return the first conditioning token id.

    Every id below it belongs to the normal text vocabulary.
    """
    if text_vocab is None:
        raise ValueError(f"text_vocab is required for {context}.")

    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base_text_vocab = (
        int(text_vocab)
        - int(speaking_rate_num_buckets)
        - sum(counts)
        - int(speaker_background_num_buckets)
        - int(accurate_mode_num_buckets)
    )
    if base_text_vocab < 0:
        raise ValueError(
            "text_vocab is smaller than the configured conditioning buckets; "
            "cannot locate conditioning tokens."
        )
    return base_text_vocab


def speaking_rate_token_id(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    speaking_rate_bucket: int,
    quality_bucket_counts=(),
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> int:
    """Return the token id of a speaking-rate bucket.

    Speaking rate is the first conditioning block.
    """
    num_buckets = int(speaking_rate_num_buckets)
    if num_buckets <= 0:
        raise ValueError("Current model does not define speaking-rate buckets.")

    bucket = int(speaking_rate_bucket)
    if bucket < 0 or bucket >= num_buckets:
        raise ValueError(
            f"speaking_rate_bucket must be in [0, {num_buckets - 1}], got {bucket}."
        )

    return _conditioning_base_text_vocab(
        text_vocab,
        num_buckets,
        quality_bucket_counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        context="speaking-rate conditioning",
    ) + bucket


def quality_token_id(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts,
    feature_idx: int,
    quality_bucket: int,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> int:
    """Return the token id of the quality bucket of one feature.

    The quality blocks follow the ``quality_features`` order. The block of a
    feature therefore starts after the buckets of every preceding feature.
    """
    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    if not counts:
        raise ValueError("Current model does not define quality buckets.")

    feature = int(feature_idx)
    if feature < 0 or feature >= len(counts):
        raise ValueError(
            f"quality feature index must be in [0, {len(counts) - 1}], got {feature}."
        )

    num_buckets = counts[feature]
    if num_buckets <= 0:
        raise ValueError(f"quality feature {feature} does not define buckets.")

    bucket = int(quality_bucket)
    if bucket < 0 or bucket >= num_buckets:
        raise ValueError(
            f"quality bucket for feature {feature} must be in "
            f"[0, {num_buckets - 1}], got {bucket}."
        )

    base_text_vocab = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        speaker_background_num_buckets,
        accurate_mode_num_buckets,
        context="quality conditioning",
    )
    return base_text_vocab + int(speaking_rate_num_buckets) + sum(counts[:feature]) + bucket


def speaker_background_token_id(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts,
    clean: bool,
    speaker_background_num_buckets: int = 2,
    accurate_mode_num_buckets: int = 0,
) -> int:
    """Return the token id of the clean/noisy background marker."""
    num_buckets = int(speaker_background_num_buckets)
    if num_buckets < 2:
        raise ValueError(
            "speaker_background_num_buckets must be at least 2 for background tokens."
        )

    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base_text_vocab = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        num_buckets,
        accurate_mode_num_buckets,
        context="speaker-background conditioning",
    )
    return (
        base_text_vocab
        + int(speaking_rate_num_buckets)
        + sum(counts)
        + (0 if bool(clean) else 1)
    )


def accurate_mode_token_id(
    text_vocab: int | None,
    speaking_rate_num_buckets: int,
    quality_bucket_counts,
    speaker_background_num_buckets: int = 2,
    accurate_mode_num_buckets: int = 1,
) -> int:
    """Return the token id of the accurate-mode marker.

    If the marker is absent, the model uses expressive mode.
    """
    accurate_count = int(accurate_mode_num_buckets)
    if accurate_count <= 0:
        raise ValueError("accurate_mode_num_buckets must be positive.")
    background_count = int(speaker_background_num_buckets)
    if background_count < 2:
        raise ValueError(
            "speaker_background_num_buckets must be at least 2 for accurate-mode tokens."
        )

    counts = _normalize_quality_bucket_counts(quality_bucket_counts)
    base_text_vocab = _conditioning_base_text_vocab(
        text_vocab,
        speaking_rate_num_buckets,
        counts,
        background_count,
        accurate_count,
        context="accurate-mode conditioning",
    )
    return (
        base_text_vocab
        + int(speaking_rate_num_buckets)
        + sum(counts)
        + background_count
    )


def shear(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Apply the inter-codebook delay. Shift column ``j`` down by ``j`` rows.

    ``x`` is ``(T, C)``, and the result is ``(C - 1 + T, C)``. ``pad`` fills the
    delayed positions. ``vocoder.shear_up`` is the inverse.
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
    """Return the sheared 0.2 s silence tail with a text-padding column."""
    silence = torch.tensor(_SILENCE_TOKENS_0_2S, dtype=torch.int32)
    sheared = shear(silence[:, :n_codebooks], audio_pad_id)
    text_col = torch.full((sheared.shape[0], 1), text_vocab, dtype=torch.int32)
    return torch.cat([sheared, text_col], dim=1)


def make_speaker_slot(
    n_codebooks: int,
    audio_pad_id: int,
    text_vocab: int,
    *,
    dtype: torch.dtype = torch.int32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the reserved all-padding frame that carries the speaker embedding.

    The result is ``(1, n_codebooks + 1)``. The frame is identical to padding,
    because the model replaces its embedding. The token ids therefore never
    reach the transformer.
    """
    slot = torch.full((1, n_codebooks + 1), audio_pad_id, dtype=dtype, device=device)
    slot[:, n_codebooks] = text_vocab
    return slot


class TTSPromptBuilder:
    """Build the 2-D prompt frame tensor for a text string.

    The bucket counts come from the ``params.json`` of the checkpoint. See
    :class:`~mstar.model.zonos2.config.Zonos2Config`. The builder needs them
    even when the caller emits no rate token and no quality token, because each
    count shifts the ids of the conditioning blocks after it.
    """

    def __init__(
        self,
        n_codebooks: int = 9,
        audio_pad_id: int = 1025,
        text_vocab: int = BYTE_TEXT_VOCAB_SIZE,
        prepend_silence: bool = True,
        speaking_rate_num_buckets: int = 0,
        quality_bucket_counts: tuple[int, ...] = (),
        speaker_background_num_buckets: int = 0,
        accurate_mode_num_buckets: int = 0,
    ):
        if text_vocab < BYTE_TEXT_VOCAB_SIZE:
            raise ValueError(
                f"text_vocab ({text_vocab}) must be >= byte vocab {BYTE_TEXT_VOCAB_SIZE}."
            )
        self.n_codebooks = n_codebooks
        self.audio_pad_id = audio_pad_id
        self.text_vocab = text_vocab
        self.speaking_rate_num_buckets = int(speaking_rate_num_buckets)
        self.quality_bucket_counts = _normalize_quality_bucket_counts(quality_bucket_counts)
        self.speaker_background_num_buckets = int(speaker_background_num_buckets)
        self.accurate_mode_num_buckets = int(accurate_mode_num_buckets)
        self._silence = (
            silence_prompt_tokens(n_codebooks, audio_pad_id, text_vocab)
            if prepend_silence
            else None
        )

    def _row(self, text_token: int) -> list[int]:
        return [self.audio_pad_id] * self.n_codebooks + [int(text_token)]

    def speaker_slot(
        self,
        *,
        dtype: torch.dtype = torch.int32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return make_speaker_slot(
            self.n_codebooks, self.audio_pad_id, self.text_vocab,
            dtype=dtype, device=device,
        )

    def build(
        self,
        text: str,
        *,
        speaker: bool = False,
        clean_speaker_background: bool = True,
        accurate_mode: bool = False,
        speaking_rate_bucket: int | None = None,
        quality_buckets: list[int | None] | None = None,
    ) -> torch.Tensor:
        """Return the int32 prompt frames ``(num_frames, n_codebooks + 1)``.

        ``speaker=True`` reserves the leading speaker slot (frame 0) and its
        markers. The caller must then supply the matching embedding. The speaker
        token position is always 0.
        """
        rows: list[list[int]] = []

        if speaker:
            # Frame 0 is the slot. The audio columns hold padding, and the text
            # column holds the text padding id. This agrees with
            # make_speaker_slot.
            rows.append(self._row(self.text_vocab))
            if self.speaker_background_num_buckets:
                rows.append(self._row(speaker_background_token_id(
                    self.text_vocab,
                    self.speaking_rate_num_buckets,
                    self.quality_bucket_counts,
                    clean_speaker_background,
                    self.speaker_background_num_buckets,
                    self.accurate_mode_num_buckets,
                )))
                # The reference emits the accurate-mode token only together
                # with the background token (scheduler.py:409-412). Keep it
                # nested here.
                if self.accurate_mode_num_buckets and accurate_mode:
                    rows.append(self._row(accurate_mode_token_id(
                        self.text_vocab,
                        self.speaking_rate_num_buckets,
                        self.quality_bucket_counts,
                        self.speaker_background_num_buckets,
                        self.accurate_mode_num_buckets,
                    )))

        if speaking_rate_bucket is not None:
            rows.append(self._row(speaking_rate_token_id(
                self.text_vocab,
                self.speaking_rate_num_buckets,
                speaking_rate_bucket,
                self.quality_bucket_counts,
                self.speaker_background_num_buckets,
                self.accurate_mode_num_buckets,
            )))

        if quality_buckets is not None:
            for feature_idx, bucket in enumerate(quality_buckets):
                if bucket is None:
                    continue
                rows.append(self._row(quality_token_id(
                    self.text_vocab,
                    self.speaking_rate_num_buckets,
                    self.quality_bucket_counts,
                    feature_idx,
                    bucket,
                    self.speaker_background_num_buckets,
                    self.accurate_mode_num_buckets,
                )))

        rows.extend(self._row(token) for token in text_to_byte_ids(text))

        prompt = torch.tensor(rows, dtype=torch.int32)
        if self._silence is not None:
            prompt = torch.cat([prompt, self._silence], dim=0)
        return prompt
