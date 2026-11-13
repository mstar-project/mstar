"""DeepSeek-V4.1's Engram: n-gram hash lookups gated into the residual streams.

Each position is hashed as the 2- to ``max_ngram``-grams ending there, over a compressed
token vocabulary in which tokens that normalize alike share an id. Every (n-gram size, head)
pair owns a prime-sized bucket range of the layer's table; the rows it fetches become one key
per stream and a shared value, and a gate from the streams' match with the key scales the
value in. The hashes depend only on token ids, so a request carries just its last
``max_ngram - 1`` ids between steps (``NGRAM`` in the recurrent pool).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.deepseek_v41.components.linear import RoundedLinear
from mstar.model.deepseek_v41.config import CSA, LABEL, NGRAM, STATE, DeepseekV41Config
from mstar.utils.fp8_block import E8M0_DTYPE, FP8_DTYPE


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def bucket_primes(config: DeepseekV41Config) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """``[engram layer][n-gram size][head]`` bucket counts: successive primes from
    ``engram_vocab_size`` up, none used twice, so no two ranges collide."""
    primes, seen = [], set()
    for _ in config.engram_layer_ids:
        per_ngram = []
        for _ in range(config.engram_max_ngram_size - 1):
            # each n-gram size starts again from engram_vocab_size and skips the primes taken
            sizes, current = [], config.engram_vocab_size - 1
            for _ in range(config.engram_n_heads):
                current += 1
                while not _is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    return tuple(primes)


def table_rows(config: DeepseekV41Config) -> tuple[int, ...]:
    """Rows of each engram layer's table: its bucket ranges end to end."""
    return tuple(sum(p for sizes in layer for p in sizes) for layer in bucket_primes(config))


def hash_multipliers(config: DeepseekV41Config) -> torch.Tensor:
    """``[engram layer, max_ngram]`` odd int64 multipliers, one per lookback, from a generator
    seeded by the layer; bounded so a token id times one stays in int64."""
    bound = max(1, (np.iinfo(np.int64).max // config.engram_compressed_vocab_size) // 2)
    rows = []
    for layer in config.engram_layer_ids:
        draws = np.random.default_rng(10007 * layer).integers(
            low=0, high=bound, size=(config.engram_max_ngram_size,), dtype=np.int64)
        rows.append(torch.from_numpy(draws * 2 + 1))
    return torch.stack(rows)


def compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Each token id's compressed id, and how many there are: tokens whose text is the same
    after Unicode folding, accent stripping, lowercasing and whitespace collapsing share one.
    A token that is not whole UTF-8 keeps its own. The hash multipliers derive from the count."""
    from tokenizers import Regex, normalizers

    # a private-use stand-in, so a token that is exactly one space survives Strip()
    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    ids: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        lookup[token_id] = ids.setdefault(key, len(ids))
    return lookup, len(ids)


class NgramHasher(nn.Module):
    """The step's hash ids ``(T, engram layers, hash cols)`` from its token ids and each
    request's ``NGRAM`` tail."""

    def __init__(self, config: DeepseekV41Config, token_map: torch.Tensor) -> None:
        super().__init__()
        if int(token_map.max()) + 1 != config.engram_compressed_vocab_size:
            # every multiplier derives from the count, so another count rehashes the tables
            raise ValueError(
                f"the tokenizer compresses to {int(token_map.max()) + 1} ids; the checkpoint's "
                f"Engram tables were hashed over {config.engram_compressed_vocab_size}")
        self.config = config
        self.max_ngram = config.engram_max_ngram_size
        self._token_map = token_map
        self.pad_id = int(token_map[config.engram_pad_token_id])
        # the hash tables, built on the model's device once it is known
        self._dev: dict[str, torch.Tensor] = {}
        self._csa = self._state = None

    def process_weights_after_loading(self, device) -> None:
        primes = bucket_primes(self.config)
        flat = [[p for sizes in layer for p in sizes] for layer in primes]
        host = {
            "token_map": self._token_map.to(torch.long),
            "primes": torch.tensor(primes, dtype=torch.long),
            # a layer's bucket ranges lie end to end in its table
            "offsets": torch.tensor([np.cumsum([0, *sizes[:-1]]).tolist() for sizes in flat], dtype=torch.long),
            "multipliers": hash_multipliers(self.config),
        }
        self._dev = {name: t.to(device) for name, t in host.items()}

    def bind_resources(self, resources: dict) -> None:
        self._csa, self._state = resources[CSA], resources[STATE]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows, n = self._csa.current(LABEL), self.max_ngram
        compressed = self._dev["token_map"][input_ids]
        held = self._state.gather(NGRAM, None, label=LABEL)  # (requests, n - 1), the newest last
        row = torch.arange(input_ids.shape[0], device=input_ids.device)
        in_span = row - rows.first
        tokens, blocked = [], torch.zeros_like(row, dtype=torch.bool)
        for shift in range(n):
            source = torch.where(
                in_span >= shift, compressed[(row - shift).clamp_min(0)],
                held[rows.row_req, (n - 1 + in_span - shift).clamp(0, n - 2)])
            # an n-gram never reaches before the sequence's first token
            blocked = blocked | (rows.pos < shift)
            tokens.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(tokens, dim=-1)[:, None, :] * self._dev["multipliers"]
        rolling, hashes = products[..., 0], []
        for i in range(1, n):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self._dev["primes"][:, i - 1])
        # each request's last n - 1 ids, for its next step
        slot = torch.arange(n - 1, device=input_ids.device)[None, :]
        at = rows.req_rows[:, None] - (n - 1) + slot  # index in the request's rows of the step
        held = torch.where(
            at >= 0, compressed[(rows.req_first[:, None] + at).clamp(0, max(input_ids.shape[0] - 1, 0))],
            held.gather(1, (n - 1 + at).clamp(0, n - 2)))
        self._state.scatter_(NGRAM, held, None, label=LABEL)
        return torch.cat(hashes, dim=-1) + self._dev["offsets"]


class Engram(nn.Module):
    def __init__(self, config: DeepseekV41Config, layer: int, comm_group: CommGroup | None = None) -> None:
        super().__init__()
        self.comm_group = comm_group or CommGroup.trivial()
        self.index = config.engram_layer_ids.index(layer)
        self.hidden_size, self.hc_mult = config.hidden_size, config.hc_mult
        self.eps, self.block = config.rms_norm_eps, config.fp8_block
        # the table stays fp8 as stored, sharded by rows; a row is dequantized when it is read
        self.total_rows = config.engram_num_embeddings[self.index]
        self.rows = -(-self.total_rows // self.comm_group.world_size)
        self.row_start = self.comm_group.rank * self.rows
        width = config.engram_head_dim
        self.table = nn.Parameter(torch.empty(self.rows, width, dtype=torch.uint8), requires_grad=False)
        self.table_scale = nn.Parameter(
            torch.empty(self.rows, width // self.block, dtype=torch.uint8), requires_grad=False)
        self.wkv = RoundedLinear(config.engram_hash_cols * width, config.hidden_size * (config.hc_mult + 1))
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def lookup(self, ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Table rows ``ids (T, cols)`` dequantized: ``(T, cols, head_dim)``."""
        local = ids - self.row_start
        mine = (local >= 0) & (local < self.rows)
        local = torch.where(mine, local, 0)
        values = self.table[local].view(FP8_DTYPE).float().unflatten(-1, (-1, self.block))
        scales = self.table_scale[local].view(E8M0_DTYPE).float()
        values = ((values * scales.unsqueeze(-1)).flatten(-2) * mine.unsqueeze(-1)).to(dtype)
        if self.comm_group.world_size > 1:
            values = self.comm_group.all_reduce(values)
        return values

    def forward(self, streams: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """``streams (T, hc, hidden)`` with the lookup of ``ids (T, cols)`` gated in."""
        kv = self.wkv(self.lookup(ids, streams.dtype).flatten(-2))
        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        weight = self.q_weight.float() * self.k_weight.float()
        h = streams.float()
        # normalized per stream, not jointly over them
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.hidden_size ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(streams.dtype)
