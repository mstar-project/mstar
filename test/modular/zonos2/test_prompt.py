"""Unit tests for Zonos2 prompt construction (``prompt``).

Covers the byte-level text tokenization, the inter-codebook delay pattern
(``shear`` and its inverse ``vocoder.shear_up`` — the machinery behind the
delayed-EOS / trailing-audio behaviour), and the assembled prompt tensor.
All pure-CPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mstar.model.zonos2.prompt import (
    BOS_ID,
    BYTE_TEXT_VOCAB_SIZE,
    EOS_ID,
    LEGACY_SYMBOL_VOCAB_SIZE,
    TTSPromptBuilder,
    shear,
    silence_prompt_tokens,
    text_to_byte_ids,
)
from mstar.model.zonos2.vocoder import shear_up


# -- byte tokenization ------------------------------------------------------
def test_text_to_byte_ids_wraps_and_offsets():
    ids = text_to_byte_ids("A")  # 'A' == 0x41 == 65
    assert ids == [BOS_ID, 65 + LEGACY_SYMBOL_VOCAB_SIZE, EOS_ID]


def test_text_to_byte_ids_roundtrips_utf8():
    text = "héllo"  # multi-byte to exercise UTF-8
    ids = text_to_byte_ids(text)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    decoded = bytes(b - LEGACY_SYMBOL_VOCAB_SIZE for b in ids[1:-1]).decode("utf-8")
    assert decoded == text


# -- inter-codebook delay (shear / shear_up) --------------------------------
def test_shear_applies_per_column_delay():
    # shear[i, j] = x[i - j, j] for i >= j, else pad.
    T, C, pad = 5, 3, -1
    x = torch.arange(1, T * C + 1).reshape(T, C)  # nonzero, distinct entries
    s = shear(x, pad)
    assert s.shape == (T, C)
    for i in range(T):
        for j in range(C):
            expected = x[i - j, j].item() if i >= j else pad
            assert s[i, j].item() == expected


def test_shear_up_inverts_shear_in_valid_triangle():
    # shear then shear_up recovers x where i + j < T (the un-padded region).
    T, C, pad = 6, 4, -1
    x = torch.arange(1, T * C + 1).reshape(T, C)
    recovered = shear_up(shear(x, pad), pad)
    assert recovered.shape == (T, C)
    for i in range(T):
        for j in range(C):
            if i + j < T:
                assert recovered[i, j].item() == x[i, j].item()


# -- assembled prompt -------------------------------------------------------
def test_build_shape_and_columns():
    n_cb, text_vocab, pad = 9, 512, 1025
    builder = TTSPromptBuilder(n_codebooks=n_cb, audio_pad_id=pad, text_vocab=text_vocab)
    byte_ids = text_to_byte_ids("hi")
    silence_rows = silence_prompt_tokens(n_cb, pad, text_vocab).shape[0]

    prompt = builder.build("hi")
    assert prompt.shape == (len(byte_ids) + silence_rows, n_cb + 1)
    # Text rows: audio columns are padding, text column carries the byte id.
    text_block = prompt[: len(byte_ids)]
    assert (text_block[:, :n_cb] == pad).all()
    assert text_block[:, -1].tolist() == byte_ids


def test_build_without_silence_is_text_only():
    builder = TTSPromptBuilder(n_codebooks=9, text_vocab=512, prepend_silence=False)
    prompt = builder.build("hi")
    assert prompt.shape == (len(text_to_byte_ids("hi")), 10)


def test_text_vocab_below_byte_vocab_raises():
    with pytest.raises(ValueError):
        TTSPromptBuilder(text_vocab=BYTE_TEXT_VOCAB_SIZE - 1)
