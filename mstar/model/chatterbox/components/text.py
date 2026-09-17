"""Text normalisation and tokenisation for Chatterbox.

Two front ends ship with the checkpoints (reference: ``chatterbox/tts.py``,
``chatterbox/tts_turbo.py``, ``chatterbox/models/tokenizers/tokenizer.py``):

* Chatterbox uses a small BPE vocabulary (``tokenizer.json``, 704 entries)
  with ``[SPACE]`` standing in for blanks and the text wrapped in
  ``[START]`` / ``[STOP]`` (ids 255 and 0).
* Chatterbox-Turbo uses a GPT-2 tokenizer extended with paralinguistic tags
  such as ``[laugh]`` (50276 entries) and no wrapping tokens.

Both normalise punctuation first. The two ``punc_norm`` tables differ
slightly in the reference and are kept apart here for parity.
"""

from __future__ import annotations

from pathlib import Path

import torch

_SPACE = "[SPACE]"
_START = "[START]"
_STOP = "[STOP]"

EMPTY_TEXT_FALLBACK = "You need to add some text for me to talk."

_PUNC_REPLACEMENTS = [
    ("...", ", "),
    ("…", ", "),   # …
    (":", ","),
    (" - ", ", "),
    (";", ", "),
    ("—", "-"),    # —
    ("–", "-"),    # –
    (" ,", ","),
    ("“", '"'),    # “
    ("”", '"'),    # ”
    ("‘", "'"),    # ‘
    ("’", "'"),    # ’
]

# Turbo keeps ellipses, spaced dashes and semicolons as they are.
_PUNC_REPLACEMENTS_TURBO = [
    pair for pair in _PUNC_REPLACEMENTS if pair[0] not in ("...", " - ", ";")
]

_SENTENCE_ENDERS = (".", "!", "?", "-", ",")


def punc_norm(text: str, *, turbo: bool = False) -> str:
    """Capitalise, collapse whitespace, replace unusual punctuation and make
    sure the text ends with a sentence ender (reference ``punc_norm``)."""
    if len(text) == 0:
        return EMPTY_TEXT_FALLBACK
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old, new in (_PUNC_REPLACEMENTS_TURBO if turbo else _PUNC_REPLACEMENTS):
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not text.endswith(_SENTENCE_ENDERS):
        text += "."
    return text


class ChatterboxTextTokenizer:
    """The 704-token BPE front end of Chatterbox (English)."""

    def __init__(self, vocab_file: str | Path, start_token: int, stop_token: int):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(vocab_file))
        vocab = self.tokenizer.get_vocab()
        if _START not in vocab or _STOP not in vocab:
            raise ValueError(f"{vocab_file} lacks the {_START}/{_STOP} tokens")
        self.start_token = start_token
        self.stop_token = stop_token

    def encode(self, text: str) -> list[int]:
        """Token ids without the wrapping start/stop tokens."""
        return self.tokenizer.encode(text.replace(" ", _SPACE)).ids

    def __call__(self, text: str) -> torch.Tensor:
        """``[START] + ids + [STOP]`` after punctuation normalisation."""
        ids = self.encode(punc_norm(text))
        return torch.tensor(
            [self.start_token, *ids, self.stop_token], dtype=torch.long
        )

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        return text.replace(" ", "").replace(_SPACE, " ").replace(_STOP, "")


class TurboTextTokenizer:
    """GPT-2 BPE plus paralinguistic tags, as shipped with Chatterbox-Turbo."""

    def __init__(self, snapshot_dir: str | Path):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(punc_norm(text, turbo=True))["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)
