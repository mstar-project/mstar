"""Tokenize a long prompt as parallel chunks, byte-identical to one encode.

An 8k-token prompt costs the fast tokenizer ~15 ms single-threaded, a quarter
of the request's time to first token. The Rust backend encodes a batch in
parallel, so a long text is cut into chunks and encoded as one batch; the
cuts are chosen so every pre-token stays whole, which keeps the BPE merges,
and so the ids, identical to encoding the whole text.

The cut rule assumes a GPT-style pre-tokenizer (the Qwen2 family, GPT-2,
Llama 3): a space starts a pre-token unless it sits inside a run of
whitespace, so a cut is placed only right before a space whose previous
character is not whitespace. Runs of whitespace, newline runs and the
punctuation-plus-newline pre-tokens are never cut, and special-token strings
contain no space. The identity is checked by test/modular/test_fast_tokenize.py
against the real tokenizer over the benchmark prompts.
"""
from __future__ import annotations

import os

# Below this many characters a single encode is as fast; above it chunking
# pays. ~4 KB of text is roughly 1k tokens.
DEFAULT_CHUNK_CHARS = int(os.environ.get("MSTAR_TOKENIZE_CHUNK_CHARS", "4096"))
DEFAULT_MAX_CHUNKS = int(os.environ.get("MSTAR_TOKENIZE_MAX_CHUNKS", "8"))


def cut_points(
    text: str, chunk_chars: int = DEFAULT_CHUNK_CHARS, max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[int]:
    """Indices where ``text`` may be cut: each is a space that starts a
    whitespace run, at or after the next chunk target. Empty when the text is
    short or no safe cut exists."""
    n = len(text)
    if chunk_chars <= 0 or n <= chunk_chars:
        return []
    want = min(max_chunks, max(2, n // chunk_chars))
    size = n // want
    cuts: list[int] = []
    at = 0
    for _ in range(want - 1):
        target = at + size
        if target >= n:
            break
        i = text.find(" ", target)
        while i > 0 and text[i - 1].isspace():
            i = text.find(" ", i + 1)
        if i <= 0 or i >= n - 1:
            break
        cuts.append(i)
        at = i
    return cuts


def split_at(text: str, cuts: list[int]) -> list[str]:
    parts = []
    at = 0
    for c in cuts:
        parts.append(text[at:c])
        at = c
    parts.append(text[at:])
    return parts


def _parallelism_on() -> None:
    """The Rust backend encodes a batch on its thread pool only while
    ``TOKENIZERS_PARALLELISM`` reads true; serving environments often export
    it false (to silence the fork warning), which makes the chunked encode
    no faster than one encode (18 vs 19 ms for 8.8k tokens on H200 hosts).
    ``MSTAR_TOKENIZE_PARALLEL=0`` leaves the variable alone."""
    if os.environ.get("MSTAR_TOKENIZE_PARALLEL", "1") != "1":
        return
    if os.environ.get("TOKENIZERS_PARALLELISM", "").lower() not in ("true", "1"):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


def encode_ids(
    tokenizer, text: str, chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[int]:
    """Token ids of ``text`` with the tokenizer's defaults for special
    tokens: a long text goes through the Rust backend as a batch of chunks,
    a short one (or a tokenizer without a Rust backend) through ``tokenizer``
    itself."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    cuts = cut_points(text, chunk_chars, max_chunks) if backend is not None else []
    if not cuts:
        return list(tokenizer(text).input_ids)
    _parallelism_on()
    encodings = backend.encode_batch(split_at(text, cuts), add_special_tokens=False)
    ids: list[int] = []
    for enc in encodings:
        ids.extend(enc.ids)
    return ids
