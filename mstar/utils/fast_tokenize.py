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

import json
import os
import re

# Below this many characters a single encode is as fast; above it chunking
# pays. ~4 KB of text is roughly 1k tokens.
DEFAULT_CHUNK_CHARS = int(os.environ.get("MSTAR_TOKENIZE_CHUNK_CHARS", "4096"))
DEFAULT_MAX_CHUNKS = int(os.environ.get("MSTAR_TOKENIZE_MAX_CHUNKS", "8"))
_SPACE_BOUNDARY = re.compile(r"\S ")


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
        match = _SPACE_BOUNDARY.search(text, max(0, target - 1))
        if match is None:
            break
        i = match.start() + 1
        if i >= n - 1:
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


def _byte_bpe_token_bound(backend, text: str) -> int | None:
    """Maximum input characters per token, only when preprocessing preserves bytes."""
    state = json.loads(backend.to_str())
    normalizer = state.get("normalizer")
    if normalizer is not None and not (
        normalizer.get("type") in ("NFC", "NFKC", "NFD", "NFKD") and text.isascii()
    ):
        return None
    model = state.get("model", {})
    if model.get("type") != "BPE" or model.get("fuse_unk"):
        return None
    pending = [state.get("pre_tokenizer")]
    byte_level = False
    while pending:
        part = pending.pop()
        if not part:
            return None
        kind = part.get("type")
        if kind == "Sequence":
            pending.extend(part["pretokenizers"])
        elif kind == "ByteLevel":
            byte_level = True
        elif kind != "Split" or part.get("behavior") == "Removed":
            return None
    added = state.get("added_tokens", [])
    if not byte_level or any(t.get("lstrip") or t.get("rstrip") for t in added):
        return None
    # ByteLevel represents each source byte with one vocabulary character.
    # Special tokens instead match their literal content; both give upper bounds.
    return max([0, *(len(t) for t in model["vocab"]), *(len(t["content"]) for t in added)]) or None


def encode_ids(
    tokenizer, text: str, chunk_chars: int = DEFAULT_CHUNK_CHARS,
    max_chunks: int = DEFAULT_MAX_CHUNKS, *, max_tokens: int | None = None,
) -> list[int]:
    """Token ids of ``text`` with the tokenizer's defaults for special
    tokens: a long text goes through the Rust backend as a batch of chunks,
    a short one (or a tokenizer without a Rust backend) through ``tokenizer``
    itself. With ``max_tokens``, stop once the context is exceeded instead of
    encoding and retaining the rest of an oversized prompt."""
    backend = getattr(tokenizer, "backend_tokenizer", None)

    def check(count):
        if max_tokens is not None and count > max_tokens:
            raise ValueError(f"prompt exceeds the model context of {max_tokens} tokens")

    if max_tokens is not None and backend is not None and len(text) > max_tokens * 8:
        bound = _byte_bpe_token_bound(backend, text)
        if bound is not None and len(text) > max_tokens * bound:
            check(max_tokens + 1)
    # Fixed-size chunks let an oversized request stop after a bounded batch,
    # instead of making eight equally huge encodings before checking the limit.
    chunks = max_chunks
    if max_tokens is not None and chunk_chars > 0:
        chunks = max(2, len(text) // chunk_chars)
    cuts = cut_points(text, chunk_chars, chunks) if backend is not None else []
    if not cuts:
        ids = list(tokenizer(text).input_ids)
        check(len(ids))
        return ids
    _parallelism_on()
    ids: list[int] = []
    ends = [0, *cuts, len(text)]
    for at in range(0, len(ends) - 1, max(1, max_chunks)):
        parts = [text[ends[i]:ends[i + 1]] for i in range(at, min(at + max(1, max_chunks), len(ends) - 1))]
        for enc in backend.encode_batch(parts, add_special_tokens=False):
            check(len(ids) + len(enc.ids))
            ids.extend(enc.ids)
    return ids
