"""The encoder's chunk-causal visibility rule.

This is the mask the engine cannot express yet, so it lives here as a plain
predicate over token codes rather than as a resource declaration. Writing it
this way keeps it checkable against the reference and keeps the eventual
resource honest: whatever kind ends up carrying this has to reproduce
exactly these four behaviours.

The sequence is ``[registers | anchor | text | noised | clean]``, so the
globals are a contiguous prefix. Each token gets one integer code:

* ``0`` for every global,
* ``2c + 1`` for a clean token of chunk ``c``,
* ``2c + 2`` for a noised token of chunk ``c``.

A key ``k`` is visible to a query ``q`` when

    ``(k == q) | (k == 0) | (((k & 1) == 1) & (k <= hi(q)))``

with ``hi(q) = q`` for a clean query and ``q - 3`` for a noised one, which
for a global query is ``-3`` and so admits no clean key at all.

Spelled out, that is four rules at once, and only the second is what most
people picture when they hear "chunk causal":

1. every token attends the globals, and a global attends nothing else,
2. a clean token attends the clean tokens of its own and earlier chunks,
3. a noised token attends the noised tokens of **its own chunk only**,
   bidirectionally within it, and no other chunk's,
4. a noised token attends the clean tokens of **strictly earlier** chunks.

Rule 3 is the one that breaks the fit with a block-causal resource: under
block causality a noised query would also reach earlier chunks' noised
tokens, which this mask forbids. Rule 4 is asymmetric and has no analogue
in a single-stream kind at all.
"""

from __future__ import annotations

import torch

# What a noised query subtracts from its own code to reach the clean tokens
# of strictly earlier chunks. Clean c is 2c+1 and noised c is 2c+2, so a
# noised query of chunk c reaching 2c-1 lands exactly on clean chunk c-1.
NOISED_CLEAN_LAG = 3


def token_codes(
    chunk: torch.Tensor, is_clean: torch.Tensor, is_block_causal: torch.Tensor
) -> torch.Tensor:
    """The per-token codes, ``(S,)`` int32.

    ``is_block_causal`` is false for the globals, which collapse to 0
    whatever chunk index they carry.
    """
    codes = torch.where(is_clean, 2 * chunk + 1, 2 * chunk + 2)
    return torch.where(is_block_causal, codes, torch.zeros_like(codes)).to(torch.int32)


def visible(code: torch.Tensor) -> torch.Tensor:
    """The dense ``(S, S)`` boolean mask for one sample's codes.

    Dense is fine here: this is the reference for the predicate and the
    thing the tests compare against, not the path a served model takes. A
    served model wants the kernel's block mask, which is what the missing
    resource would build.
    """
    q = code[:, None].to(torch.int64)
    k = code[None, :].to(torch.int64)
    hi = torch.where((q & 1) == 1, q, q - NOISED_CLEAN_LAG)
    return (k == q) | (k == 0) | (((k & 1) == 1) & (k <= hi))


def encoder_codes(
    n_registers: int,
    n_anchor: int,
    n_text: int,
    tokens_per_chunk: int,
    n_chunks: int,
    n_clean_chunks: int,
) -> torch.Tensor:
    """Codes for one encoder sequence laid out the way the tower packs it.

    ``[registers | anchor | text | noised | clean]``, with ``n_clean_chunks``
    chunks of clean tokens appended after the noised ones. A window of one
    chunk carries no clean tokens at all.
    """
    n_global = n_registers + n_anchor + n_text
    parts = [torch.zeros(n_global, dtype=torch.int64)]
    noised_chunk = torch.arange(n_chunks).repeat_interleave(tokens_per_chunk)
    parts.append(2 * noised_chunk + 2)
    if n_clean_chunks:
        clean_chunk = torch.arange(n_clean_chunks).repeat_interleave(tokens_per_chunk)
        parts.append(2 * clean_chunk + 1)
    return torch.cat(parts).to(torch.int32)
