"""DeepSeek-V4.1 Engram n-gram hashing, compressed token mapping, and batch-isolated lookback."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sympy import nextprime


def build_compressed_token_map(
    tokenizer,
    expected_compressed_vocab_size: int | None = None,
) -> np.ndarray:
    """Map each token id to its normalized canonical id via HuggingFace normalizers."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    vocab_len = len(tokenizer)
    ids: dict[str, int] = {}
    lookup = np.zeros(vocab_len, dtype=np.int64)
    for token_id in range(vocab_len):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        lookup[token_id] = ids.setdefault(key, len(ids))
    if expected_compressed_vocab_size is not None and len(ids) != expected_compressed_vocab_size:
        raise ValueError(
            f"tokenizer compressed to {len(ids)} tokens, expected {expected_compressed_vocab_size}"
        )
    return lookup


def bucket_primes(
    engram_layer_ids: Sequence[int],
    max_ngram_size: int,
    engram_vocab_size: int,
    engram_n_heads: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Allocate sequential disjoint prime bucket counts per ``[engram_layer][ngram_size][head]``."""
    primes: list[tuple[tuple[int, ...], ...]] = []
    seen: set[int] = set()
    for _ in engram_layer_ids:
        per_ngram: list[tuple[int, ...]] = []
        for _ in range(max_ngram_size - 1):
            sizes: list[int] = []
            current = engram_vocab_size - 1
            for _ in range(engram_n_heads):
                current = int(nextprime(current))
                while current in seen:
                    current = int(nextprime(current))
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    return tuple(primes)


def compute_hash_multipliers(
    engram_layer_ids: Sequence[int],
    max_ngram_size: int,
    compressed_vocab_size: int,
) -> torch.Tensor:
    """Deterministic odd int64 multipliers ``[n_engram_layers, max_ngram_size]`` per layer."""
    bound = max(1, (np.iinfo(np.int64).max // max(1, compressed_vocab_size)) // 2)
    rows = []
    for layer_id in engram_layer_ids:
        draws = np.random.default_rng(10007 * int(layer_id)).integers(
            low=0, high=bound, size=(max_ngram_size,), dtype=np.int64
        )
        rows.append(torch.from_numpy(draws * 2 + 1))
    return torch.stack(rows)


@dataclass(frozen=True)
class EngramLayout:
    """Precomputed primes, offsets, multipliers, and token map for DeepSeek-V4.1 Engram."""

    token_map: torch.Tensor
    multipliers: torch.Tensor
    primes: torch.Tensor
    offsets: torch.Tensor
    pad_id: int
    max_ngram_size: int
    n_heads: int
    num_embeddings: tuple[int, ...]

    @classmethod
    def from_config(
        cls,
        config,
        token_map: torch.Tensor | np.ndarray | None = None,
    ) -> EngramLayout:
        if token_map is None:
            tmap = torch.arange(config.vocab_size, dtype=torch.long, device="cpu")
            compressed_vocab_size = max(config.vocab_size, config.engram_compressed_vocab_size)
            pad_id = int(config.engram_pad_token_id)
        else:
            tmap = torch.as_tensor(token_map, dtype=torch.long, device="cpu")
            compressed_vocab_size = int(tmap.max().item()) + 1 if tmap.numel() else config.engram_compressed_vocab_size
            pad_id = int(tmap[config.engram_pad_token_id].item())
        primes_nested = bucket_primes(
            config.engram_layer_ids,
            config.engram_max_ngram_size,
            config.engram_vocab_size,
            config.engram_n_heads,
        )
        flat = [[p for sizes in layer for p in sizes] for layer in primes_nested]
        num_embeddings = tuple(sum(sizes) for sizes in flat)
        offsets = torch.tensor(
            [np.cumsum([0, *sizes[:-1]]).tolist() for sizes in flat], dtype=torch.long, device="cpu"
        )
        primes = torch.tensor(primes_nested, dtype=torch.long, device="cpu")
        multipliers = compute_hash_multipliers(
            config.engram_layer_ids,
            config.engram_max_ngram_size,
            compressed_vocab_size,
        )
        return cls(
            token_map=tmap,
            multipliers=multipliers,
            primes=primes,
            offsets=offsets,
            pad_id=pad_id,
            max_ngram_size=config.engram_max_ngram_size,
            n_heads=config.engram_n_heads,
            num_embeddings=num_embeddings,
        )


def compute_engram_hashes(
    input_ids: torch.Tensor,
    layout: EngramLayout,
    *,
    pos: torch.Tensor | None = None,
    row_req: torch.Tensor | None = None,
    req_first: torch.Tensor | None = None,
    prev_tail: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute Engram hash ids for packed 1D ``input_ids [T]`` or batched ``[B, S]``.

    When ``input_ids`` is 2D ``[B, S]`` with optional ``prev_tail [B, max_ngram - 1]``,
    each batch row is hashed independently (no cross-request leakage).
    When ``input_ids`` is 1D ``[T]`` (packed requests), ``pos [T]``, ``row_req [T]``,
    ``req_first [T]``, and ``prev_tail [R, max_ngram - 1]`` isolate each request's lookback.
    Returns ``(hashes, updated_tail)``.
    """
    n = layout.max_ngram_size
    tail_len = n - 1
    device = input_ids.device
    tmap = layout.token_map.to(device)
    mult = layout.multipliers.to(device)
    primes = layout.primes.to(device)
    offsets = layout.offsets.to(device)
    pad_id = layout.pad_id

    if input_ids.ndim == 2:
        bsz, seq = input_ids.shape
        compressed = tmap[input_ids]
        if token_mask is not None:
            compressed = compressed.masked_fill(~token_mask, -1)
        if prev_tail is None:
            history = compressed.new_full((bsz, tail_len), -1)
        else:
            history = prev_tail.to(device=device, dtype=torch.long)
        combined = torch.cat([history, compressed], dim=1)  # [B, tail_len + S]
        # Shift 0 is current token, shift s is s tokens back; stop at -1 (DEAD/BOS).
        cols = []
        blocked = torch.zeros((bsz, seq), dtype=torch.bool, device=device)
        for shift in range(n):
            idx = tail_len - shift + torch.arange(seq, device=device)
            tok = combined[:, idx]
            blocked = blocked | (tok < 0)
            cols.append(torch.where(blocked, pad_id, tok))
        tokens = torch.stack(cols, dim=-1)  # [B, S, n]
        products = tokens.unsqueeze(2) * mult[None, None, :, :]  # [B, S, L, n]
        rolling = products[..., 0]
        hashes = []
        for i in range(1, n):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1][None, None, :, :])
        out = torch.cat(hashes, dim=-1) + offsets[None, None, :, :]
        next_tail = combined[:, -tail_len:].clone() if tail_len > 0 else None
        return out, next_tail

    if input_ids.ndim != 1:
        raise ValueError(f"expected 1D or 2D input_ids, got shape {tuple(input_ids.shape)}")

    t_count = input_ids.shape[0]
    compressed = tmap[input_ids]
    if token_mask is not None:
        compressed = compressed.masked_fill(~token_mask, -1)
    if pos is None:
        pos = torch.arange(t_count, dtype=torch.long, device=device)
    if row_req is None:
        row_req = torch.zeros(t_count, dtype=torch.long, device=device)
    if req_first is None:
        req_first = torch.zeros(t_count, dtype=torch.long, device=device)
    num_reqs = int(row_req.max().item()) + 1 if t_count > 0 else (0 if prev_tail is None else prev_tail.shape[0])
    if prev_tail is None:
        held = compressed.new_full((max(1, num_reqs), tail_len), -1)
    else:
        held = prev_tail.to(device=device, dtype=torch.long)

    row = torch.arange(t_count, dtype=torch.long, device=device)
    in_span = row - req_first
    cols = []
    blocked = torch.zeros(t_count, dtype=torch.bool, device=device)
    for shift in range(n):
        from_step = compressed[(row - shift).clamp_min(0)]
        # prev_tail[:, -1] is pos_first-1, so shift tokens back from in_span reads tail_len - (shift - in_span).
        if tail_len > 0 and shift > 0:
            tail_idx = (tail_len - (shift - in_span)).clamp(0, tail_len - 1)
            from_held = held[row_req, tail_idx]
        else:
            from_held = compressed.new_full((t_count,), -1)
        src = torch.where(in_span >= shift, from_step, from_held)
        blocked = blocked | (pos < shift) | (src < 0)
        cols.append(torch.where(blocked, pad_id, src))
    tokens = torch.stack(cols, dim=-1)  # [T, n]
    products = tokens.unsqueeze(1) * mult[None, :, :]  # [T, L, n]
    rolling = products[..., 0]
    hashes = []
    for i in range(1, n):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1][None, :, :])
    out = torch.cat(hashes, dim=-1) + offsets[None, :, :]

    if tail_len == 0:
        return out, None
    next_tail = held.clone()
    for r in range(num_reqs):
        mask_r = row_req == r
        if mask_r.any():
            seq_r = compressed[mask_r]
            combined_r = torch.cat([held[r], seq_r], dim=0)
            next_tail[r] = combined_r[-tail_len:]
    return out, next_tail
