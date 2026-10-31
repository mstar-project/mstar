"""A row's top-k/top-p distribution from one read of its logits.

FlashInfer's renorms search a whole row for each cut, which is a dozen reads of a
248k-token vocabulary per launch. What the filters keep is nearly always a few
tokens, so here each tile of the vocabulary hands up its top tokens (its logit in
the high bits of a key, its index in the low ones), and one program per row
selects the row's top ones from those and filters them in registers. The tiles'
softmax sums go up with them, so a kept token's probability is the one a
vocabulary-wide softmax gives it.

Ties stay together, as in FlashInfer: keeping a prefix of the sorted tokens by mass
and then everything at or above its last value keeps a token iff the mass strictly
above it is under the cut.

What is kept can run past the selection: more tokens than it holds, or more of them
in one tile than the tile handed up. A row then says so in ``complete`` and holds
the kept tokens it saw, renormalised, and `FILTER_PAST_CANDIDATES` gives the exact
row for the callers that need it, at no cost to the other rows. The vocabulary is
sorted by frequency, so tiles do crowd, but with what top-k keeps at its tail.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.kernels.numerics import Tolerance
from mstar.kernels.registry import KernelOp, ShapePolicy

# Tokens of a tile a key can say apart, and what a tile hands up. At temperature 0.7
# and top-p 0.9 the most kept tokens seen in one tile of 2048 was 17, at the 99th
# percentile 6; at 1.0 and 0.95, 64 and 11.
TILE = 2048
TILE_TOP = 16
MAX_VOCAB = 1 << 20


@triton.jit
def _tiles_kernel(
    logits_ptr, temperature_ptr, keys_ptr, max_ptr, sum_ptr,
    V, stride_b, NSPLIT, BLOCK: tl.constexpr, M: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs = split * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    x = tl.load(logits_ptr + row * stride_b + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = tl.where(x == x, x, -float("inf"))  # noqa: PLR0124  (NaN test)
    temp = tl.load(temperature_ptr + row)
    scaled = x * tl.where(temp > 0, 1.0 / tl.maximum(temp, 1e-30), 1.0)
    top = tl.max(scaled, axis=0)
    # -inf - -inf is NaN, in a tile with nothing finite
    terms = tl.where(scaled > -float("inf"), tl.exp(scaled - top), 0.0)
    tl.store(max_ptr + row * NSPLIT + split, top)
    tl.store(sum_ptr + row * NSPLIT + split, tl.sum(terms, axis=0))
    # IEEE754 made order-preserving, then the index, so equal logits go to the lowest one, as an argmax does
    bits = x.to(tl.int32, bitcast=True).to(tl.int64)
    order = tl.where(bits < 0, -(bits & 0x7FFFFFFF), bits)
    key = (order << 20) + (0xFFFFF - offs.to(tl.int64))
    tl.store(keys_ptr + (row * NSPLIT + split) * M + tl.arange(0, M), tl.topk(key, M))


@triton.jit
def _select_kernel(
    keys_ptr, max_ptr, sum_ptr, temperature_ptr, top_k_ptr, top_p_ptr, ids_ptr, probs_ptr, complete_ptr,
    V, NSPLIT, M, BLOCK_KEYS: tl.constexpr, BLOCK_PARTS: tl.constexpr, C: tl.constexpr,
):
    row = tl.program_id(0)
    parts = tl.arange(0, BLOCK_PARTS)
    in_parts = parts < NSPLIT
    tile_max = tl.load(max_ptr + row * NSPLIT + parts, mask=in_parts, other=-float("inf"))
    tile_sum = tl.load(sum_ptr + row * NSPLIT + parts, mask=in_parts, other=0.0)
    peak = tl.max(tile_max, axis=0)
    # each tile's sum is relative to its own max
    total = tl.sum(tl.where(in_parts, tile_sum * tl.exp(tile_max - peak), 0.0), axis=0)

    n = tl.arange(0, BLOCK_KEYS)
    in_keys = n < NSPLIT * M
    keys = tl.load(keys_ptr + row * NSPLIT * M + n, mask=in_keys, other=0)
    keys = tl.where(in_keys, keys, -(1 << 62))
    # nothing a tile left out lies above its last key
    bound = tl.max(tl.where((n % M) == M - 1, keys, -(1 << 62)), axis=0) >> 20
    top = tl.topk(keys, C)
    pos = tl.arange(0, C)
    order = top >> 20
    index = 0xFFFFF - (top & 0xFFFFF)
    x = tl.where(order < 0, (-order) | 0x80000000, order).to(tl.int32).to(tl.float32, bitcast=True)

    temp = tl.load(temperature_ptr + row)
    scaled = x * tl.where(temp > 0, 1.0 / tl.maximum(temp, 1e-30), 1.0)
    p = tl.where(x > -float("inf"), tl.exp(scaled - peak) / tl.maximum(total, 1e-30), 0.0)

    k = tl.load(top_k_ptr + row)
    top_p = tl.load(top_p_ptr + row)
    off = k >= V
    wide = k > C
    order_k = tl.sum(tl.where(pos == k - 1, order, 0), axis=0)
    in_k = off | wide | (order >= order_k)
    held = tl.where(in_k, p, 0.0)
    before = tl.cumsum(held, axis=0) - held
    # Top-p is a share of the whole row, or with top-k on of what that keeps. With
    # top-k's cut out of sight that mass is only known to lie between what is seen
    # above `bound` and the whole row. Frequent tokens crowd a tile, so this is usual
    # at a top-k of 40, and the cut still lands on the same token from both ends.
    sure = off | (~wide & (order_k > bound))
    low = tl.where(off, 1.0, tl.sum(tl.where(sure | (order > bound), held, 0.0), axis=0))
    high = tl.where(sure, low, 1.0)
    inf = float("inf")
    floor = tl.min(tl.where(in_k & (before < tl.where(top_p >= 1.0, inf, top_p * high)), order, 1 << 61), axis=0)
    floor_low = tl.min(tl.where(in_k & (before < tl.where(top_p >= 1.0, inf, top_p * low)), order, 1 << 61), axis=0)
    kept = tl.where(in_k & (order >= floor), p, 0.0)
    out = kept / tl.maximum(tl.sum(kept, axis=0), 1e-30)
    lowest = tl.min(tl.where(kept > 0, order, 1 << 61), axis=0)
    fits = (tl.sum((kept > 0).to(tl.int32), axis=0) < C) & (lowest > bound) & (floor == floor_low)

    greedy = temp == 0
    out = tl.where(greedy, tl.where(pos == 0, 1.0, 0.0), out)
    tl.store(ids_ptr + row * C + pos, tl.where(out > 0, index, 0))
    tl.store(probs_ptr + row * C + pos, out)
    tl.store(complete_ptr + row, (greedy | fits).to(tl.int32))


def _candidates_impl():
    def filtered_candidates(logits, temperature, top_k, top_p, ids, probs, complete) -> None:
        rows, vocab = logits.shape
        width = ids.shape[1]
        if vocab > MAX_VOCAB:
            raise ValueError(f"a key holds a token index of 20 bits, not one of a {vocab}-token vocabulary")
        if logits.stride(1) != 1:
            logits = logits.contiguous()
        nsplit = -(-vocab // TILE)
        # a small vocabulary still fills the selection
        tile_top = max(TILE_TOP, triton.next_power_of_2(-(-width // nsplit)))
        device = logits.device
        keys = torch.empty((rows, nsplit, tile_top), dtype=torch.int64, device=device)
        part_max = torch.empty((rows, nsplit), dtype=torch.float32, device=device)
        part_sum = torch.empty((rows, nsplit), dtype=torch.float32, device=device)
        with torch.cuda.device(device):
            _tiles_kernel[(rows, nsplit)](
                logits, temperature, keys, part_max, part_sum, vocab, logits.stride(0), nsplit,
                BLOCK=TILE, M=tile_top, num_warps=8,
            )
            _select_kernel[(rows,)](
                keys, part_max, part_sum, temperature, top_k, top_p, ids, probs, complete,
                vocab, nsplit, tile_top,
                BLOCK_KEYS=triton.next_power_of_2(nsplit * tile_top), BLOCK_PARTS=triton.next_power_of_2(nsplit),
                C=width, num_warps=4,
            )

    return filtered_candidates


@triton.jit
def _row_probs(logits_ptr, row, stride_in, offs, mask, inv_temp, peak, scale):
    """One tile of a row's softmax, and its probabilities' float bits."""
    x = tl.load(logits_ptr + row * stride_in + offs, mask=mask, other=-float("inf")).to(tl.float32)
    p = tl.where(mask & (x == x), tl.exp(x * inv_temp - peak) * scale, 0.0)  # noqa: PLR0124  (NaN test)
    return p, p.to(tl.int32, bitcast=True)


@triton.jit
def _past_kernel(
    logits_ptr, temperature_ptr, top_k_ptr, top_p_ptr, complete_ptr, out_ptr,
    V, stride_in, stride_out, BLOCK: tl.constexpr,
):
    """The exact filter of each row that isn't complete, and nothing for the rest.
    A greedy row always is, so these all have a temperature."""
    row = tl.program_id(0)
    if tl.load(complete_ptr + row) == 0:
        inv_temp = 1.0 / tl.load(temperature_ptr + row)
        k = tl.load(top_k_ptr + row)
        top_p = tl.load(top_p_ptr + row)

        peak = -float("inf")
        for v_start in range(0, V, BLOCK):
            offs = v_start + tl.arange(0, BLOCK)
            x = tl.load(logits_ptr + row * stride_in + offs, mask=offs < V, other=-float("inf")).to(tl.float32)
            x = tl.where(x == x, x, -float("inf"))  # noqa: PLR0124  (NaN test)
            peak = tl.maximum(peak, tl.max(x * inv_temp, axis=0))
        total = tl.zeros([], dtype=tl.float32)
        for v_start in range(0, V, BLOCK):
            offs = v_start + tl.arange(0, BLOCK)
            p, _ = _row_probs(logits_ptr, row, stride_in, offs, offs < V, inv_temp, peak, 1.0)
            total += tl.sum(p, axis=0)
        scale = 1.0 / total

        # top-k: the largest threshold with at least k tokens at or above it
        first = tl.zeros([], dtype=tl.int32)
        if k < V:
            lo = tl.zeros([], dtype=tl.int32)
            hi = tl.zeros([], dtype=tl.int32) + 0x7F800000
            while lo < hi:
                mid = lo + (hi - lo + 1) // 2
                count = tl.zeros([], dtype=tl.int32)
                for v_start in range(0, V, BLOCK):
                    offs = v_start + tl.arange(0, BLOCK)
                    _, bits = _row_probs(logits_ptr, row, stride_in, offs, offs < V, inv_temp, peak, scale)
                    count += tl.sum((bits >= mid).to(tl.int32), axis=0)
                ok = count >= k
                lo = tl.where(ok, mid, lo)
                hi = tl.where(ok, hi, mid - 1)
            first = lo
        mass = tl.zeros([], dtype=tl.float32)
        for v_start in range(0, V, BLOCK):
            offs = v_start + tl.arange(0, BLOCK)
            p, bits = _row_probs(logits_ptr, row, stride_in, offs, offs < V, inv_temp, peak, scale)
            mass += tl.sum(tl.where(bits >= first, p, 0.0), axis=0)

        # top-p over what top-k left: the least value with less than the cut strictly
        # above it. Each pass pulls the bracket in to the values next to the split.
        floor = first
        if top_p < 1.0:
            cut = top_p * mass
            lo = first - 1  # the mass strictly above lo reaches the cut
            hi = tl.zeros([], dtype=tl.int32) + 0x7F800000  # and above hi it doesn't
            while lo + 1 < hi:
                mid = lo + (hi - lo) // 2
                above = tl.zeros([], dtype=tl.float32)
                next_up = tl.zeros([], dtype=tl.int32) + 0x7F800000
                next_down = lo
                for v_start in range(0, V, BLOCK):
                    offs = v_start + tl.arange(0, BLOCK)
                    p, bits = _row_probs(logits_ptr, row, stride_in, offs, offs < V, inv_temp, peak, scale)
                    gt = bits > mid
                    above += tl.sum(tl.where(gt, p, 0.0), axis=0)
                    next_up = tl.minimum(next_up, tl.min(tl.where(gt, bits, 0x7F800000), axis=0))
                    next_down = tl.maximum(next_down, tl.max(tl.where(gt | (bits < first), lo, bits), axis=0))
                ok = above < cut
                hi = tl.where(ok, next_down, hi)
                lo = tl.where(ok, lo, next_up - 1)
            floor = hi
        kept_mass = tl.zeros([], dtype=tl.float32)
        for v_start in range(0, V, BLOCK):
            offs = v_start + tl.arange(0, BLOCK)
            p, bits = _row_probs(logits_ptr, row, stride_in, offs, offs < V, inv_temp, peak, scale)
            kept_mass += tl.sum(tl.where(bits >= floor, p, 0.0), axis=0)
        for v_start in range(0, V, BLOCK):
            offs = v_start + tl.arange(0, BLOCK)
            p, bits = _row_probs(logits_ptr, row, stride_in, offs, offs < V, inv_temp, peak, scale)
            tl.store(out_ptr + row * stride_out + offs, tl.where(bits >= floor, p / kept_mass, 0.0), mask=offs < V)


def _past_impl():
    def filter_past_candidates(logits, temperature, top_k, top_p, complete, probs) -> None:
        rows, vocab = logits.shape
        if logits.stride(1) != 1:
            logits = logits.contiguous()
        with torch.cuda.device(logits.device):
            _past_kernel[(rows,)](
                logits, temperature, top_k, top_p, complete, probs, vocab, logits.stride(0), probs.stride(0),
                BLOCK=8192, num_warps=8,
            )

    return filter_past_candidates


# --- torch references: for CPU tensors, and what the kernels are checked against ---


def _softmax_reference(logits, temperature):
    x = logits.float()
    if temperature == 0:
        return torch.nn.functional.one_hot(x.argmax(), x.numel()).to(torch.float32)
    return torch.softmax(x / temperature.float(), dim=-1)


def keep_reference(probs, top_k, top_p):
    """One row's filter by the kernels' rule: a token stays while fewer than k
    tokens, then less than ``top_p`` of the mass left, lie strictly above it."""
    if int(top_k) < probs.numel():
        floor = probs.topk(int(top_k)).values[-1]
        probs = torch.where(probs >= floor, probs, 0.0)
    if float(top_p) < 1.0:
        values = probs.unique()
        above = torch.tensor([float(probs[probs > v].sum()) for v in values], dtype=probs.dtype)
        floor = values[above < float(top_p) * float(probs.sum())].min()
        probs = torch.where(probs >= floor, probs, 0.0)
    return probs / probs.sum()


def _candidates_reference(logits, temperature, top_k, top_p, ids, probs, complete) -> None:
    vocab, width = logits.shape[1], ids.shape[1]
    for r in range(logits.shape[0]):
        row = _softmax_reference(logits[r], temperature[r])
        # stable, so equal values keep the lowest index first
        order = torch.argsort(row, descending=True, stable=True)[:width]
        values = row[order]
        off = int(top_k[r]) >= vocab
        share = top_p[r] / values.sum() if off else top_p[r]
        kept = keep_reference(values, top_k[r], share)
        probs[r] = kept.to(probs.dtype)
        ids[r] = torch.where(kept > 0, order, 0)
        fits = int((kept > 0).sum()) < width and (off or int(top_k[r]) <= width)
        complete[r] = int(fits or temperature[r] == 0)


def _past_reference(logits, temperature, top_k, top_p, complete, probs) -> None:
    for r in range(logits.shape[0]):
        if not complete[r]:
            probs[r] = keep_reference(_softmax_reference(logits[r], temperature[r]), top_k[r], top_p[r]).to(probs.dtype)


def _fake(*args) -> None:
    return None


def _candidate_inputs(shape, dtype, device, generator=None):
    rows, vocab = shape["rows"], shape["vocab"]
    logits = torch.randn(rows, vocab, device=device, generator=generator, dtype=torch.float32)
    # a few plausible tokens over a noise floor, spread out so no tile has more than it hands up
    picks = torch.arange(32, dtype=torch.int64, device=device) * (vocab // 32) + 5
    logits[:, picks] = 8.0 + torch.arange(32, device=device, dtype=torch.float32) * 0.25
    temperature = torch.full((rows,), 0.7, dtype=torch.float32, device=device)
    temperature[0] = 0.0
    top_k = torch.full((rows,), vocab, dtype=torch.int32, device=device)
    top_k[rows // 2:] = 8
    top_p = torch.full((rows,), 0.9, dtype=torch.float32, device=device)
    width = 64
    return (logits.to(dtype), temperature, top_k, top_p,
            torch.zeros(rows, width, dtype=torch.int64, device=device),
            torch.zeros(rows, width, dtype=torch.float32, device=device),
            torch.zeros(rows, dtype=torch.int32, device=device))


def _past_inputs(shape, dtype, device, generator=None):
    logits, temperature, top_k, top_p, _, _, complete = _candidate_inputs(shape, dtype, device, generator)
    # flat rows, whose kept sets a selection can't hold
    logits = logits.float() * 0.05
    temperature = temperature.fill_(2.0)
    return (logits.to(dtype), temperature, top_k, top_p, complete,
            torch.zeros(logits.shape, dtype=torch.float32, device=device))


FILTERED_CANDIDATES = KernelOp(
    "filtered_candidates",
    "(Tensor logits, Tensor temperature, Tensor top_k, Tensor top_p, Tensor(a!) ids, Tensor(b!) probs, "
    "Tensor(c!) complete) -> ()",
    fake=_fake,
    reference=_candidates_reference,
    impls={"rocm": _candidates_impl},
    shapes=ShapePolicy(
        dims=("rows", "vocab"),
        key=lambda logits, *args, **kwargs: {"rows": logits.shape[0], "vocab": logits.shape[1]},
        inputs=_candidate_inputs,
        warm=({"rows": 3, "vocab": 248320}, {"rows": 192, "vocab": 248320}),
        dtypes=(torch.bfloat16, torch.float32),
    ),
    tolerance=Tolerance.accumulated(rtol=1e-5, atol_scale=1e-6),
)

FILTER_PAST_CANDIDATES = KernelOp(
    "filter_past_candidates",
    "(Tensor logits, Tensor temperature, Tensor top_k, Tensor top_p, Tensor complete, Tensor(a!) probs) -> ()",
    fake=_fake,
    reference=_past_reference,
    impls={"rocm": _past_impl},
    shapes=ShapePolicy(
        dims=("rows", "vocab"),
        key=lambda logits, *args, **kwargs: {"rows": logits.shape[0], "vocab": logits.shape[1]},
        inputs=_past_inputs,
        warm=({"rows": 2, "vocab": 4096},),
        dtypes=(torch.float32,),
    ),
    tolerance=Tolerance.accumulated(rtol=1e-5, atol_scale=1e-6),
)
