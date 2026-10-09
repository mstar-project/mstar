"""Generic token sampling utilities.

Uses device-specific fused top-k/top-p sampling kernels: FlashInfer on CUDA and
``vllm-xpu-kernels`` on XPU. Model-agnostic — any AR model returns logits, this
module selects the next token.

Supports per-request sampling parameters (different temperature/top_k/top_p
for each request in a batch) via tensor parameters.

Supports CUDA and XPU graph capture with explicit device seed/offset tensors
after warmup. Uses masking for mixed greedy and sampled requests.

Usage:
    from mstar.engine.resources.sampler.utils import sample_tokens
    tokens = sample_tokens(logits, temperature=0.7, top_p=0.9)
"""

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl

from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)


def _rng_offset_stride(device_type: str, vocab_size: int) -> int:
    """Number of Philox offset units consumed by one sampled row."""
    return vocab_size if device_type == "xpu" else 1


def _xpu_generator_seed_offset(
    generator: torch.Generator,
    num_random_values: int,
) -> tuple[int, int]:
    """Read XPU generator state and reserve an aligned Philox offset range."""
    state = generator.get_state()
    state_values = state.view(torch.int64)
    seed = int(state_values[0])
    offset = int(state_values[1])
    if num_random_values:
        # PyTorch requires the stored XPU generator offset to be a multiple of 4.
        next_offset = (offset + num_random_values + 3) // 4 * 4
        state_values[1] = next_offset
        generator.set_state(state)
    return seed, offset


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 32768}, num_warps=32, num_stages=2),
    ],
    key=["V", "APPLY_PENALTY", "INCLUDE_GREEDY", "WINDOW", "NUM_STOP"],
)
@triton.jit
def _fused_sampling_prep_kernel(
    logits_ptr,        # [B, V] input
    temperature_ptr,   # [B]
    penalty_ptr,       # [B] (only read when APPLY_PENALTY=True)
    seen_mask_ptr,     # [B, V] bool (only read when APPLY_PENALTY=True)
    hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,  # see _history_adjust
    probs_ptr,         # [B, V] float32 output
    V,
    stride_b, stride_v,
    out_stride_b, out_stride_v,
    mask_stride_b, mask_stride_v,
    BLOCK_SIZE: tl.constexpr,
    APPLY_PENALTY: tl.constexpr,
    INCLUDE_GREEDY: tl.constexpr,
    WINDOW: tl.constexpr,
    NUM_STOP: tl.constexpr,
):
    """Fused (optional rep penalty) + (logits/temperature) + softmax.

    When INCLUDE_GREEDY is True and a row's temperature == 0, the kernel
    emits a one-hot distribution at the argmax instead of a temperature-scaled
    softmax — so a downstream multinomial sampler deterministically returns
    the argmax token (replaces the separate torch.argmax + torch.where pair).

    Both constexprs specialize at compile time; the unused branches compile out.
    """
    row = tl.program_id(0)
    temp = tl.load(temperature_ptr + row)
    if INCLUDE_GREEDY:
        is_greedy = temp == 0
        # Safe inv_temp so the softmax branch doesn't produce NaN for greedy
        # rows (their output is overwritten by the one-hot anyway).
        inv_temp = tl.where(is_greedy, 1.0, 1.0 / tl.maximum(temp, 1e-30))
    else:
        inv_temp = 1.0 / temp

    if APPLY_PENALTY:
        penalty = tl.load(penalty_ptr + row)

    # Pass 1: scan over V, compute max of raw vals (post-penalty) + argmax.
    # argmax is only used by the greedy one-hot path; still tracked when
    # INCLUDE_GREEDY is True regardless of per-row temp.
    max_raw = -float("inf")
    max_idx = tl.zeros([], dtype=tl.int32)
    for v_start in range(0, V, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=-float("inf"),
        )
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            penalized = tl.where(vals > 0, vals / penalty, vals * penalty)
            vals = tl.where(seen, penalized, vals)
        vals = _history_adjust(
            vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
            WINDOW, NUM_STOP,
        )
        masked_vals = tl.where(mask, vals, -float("inf"))
        block_max = tl.max(masked_vals)
        if INCLUDE_GREEDY:
            block_argmax = tl.argmax(masked_vals, axis=0)
            is_new = block_max > max_raw
            max_idx = tl.where(is_new, v_start + block_argmax.to(tl.int32), max_idx)
        max_raw = tl.maximum(max_raw, block_max)

    max_scaled = max_raw * inv_temp

    # Pass 2: exp(scaled - max_scaled), accumulate sum
    sum_exp = tl.zeros([], dtype=tl.float32)
    for v_start in range(0, V, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=0.0,
        )
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            penalized = tl.where(vals > 0, vals / penalty, vals * penalty)
            vals = tl.where(seen, penalized, vals)
        vals = _history_adjust(
            vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
            WINDOW, NUM_STOP,
        )
        scaled = vals * inv_temp
        exp_val = tl.exp(scaled - max_scaled)
        exp_val = tl.where(mask, exp_val, 0.0)
        sum_exp += tl.sum(exp_val)

    # Avoid div-by-zero in the greedy rows (their output is overwritten).
    inv_sum = 1.0 / tl.maximum(sum_exp, 1e-30)

    # Pass 3: write the output — softmax probs for non-greedy rows,
    # one-hot at argmax for greedy rows.
    for v_start in range(0, V, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=0.0,
        )
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            penalized = tl.where(vals > 0, vals / penalty, vals * penalty)
            vals = tl.where(seen, penalized, vals)
        vals = _history_adjust(
            vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
            WINDOW, NUM_STOP,
        )
        scaled = vals * inv_temp
        softmax_val = tl.exp(scaled - max_scaled) * inv_sum
        if INCLUDE_GREEDY:
            is_max = offs == max_idx
            one_hot = tl.where(is_max, 1.0, 0.0)
            probs = tl.where(is_greedy, one_hot, softmax_val)
        else:
            probs = softmax_val
        tl.store(
            probs_ptr + row * out_stride_b + offs * out_stride_v,
            probs, mask=mask,
        )


# ---------------------------------------------------------------------------
# Split-V variant of the above.
#
# The single-kernel form runs one block per row (16 of 132 SMs at bs=16) and
# reads the vocab three times. Splitting the vocab across blocks scales
# occupancy with B x NSPLIT, and an online softmax reads the logits twice.
# Measured well under the single kernel for Qwen3.5's 248k vocab.
# ---------------------------------------------------------------------------


@triton.jit
def _history_adjust(
    vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
    WINDOW: tl.constexpr, NUM_STOP: tl.constexpr,
):
    """The ``HistoryRows`` processors on one tile of a row's logits: the windowed
    frequency penalty (``penalty ** n``, n the token's count among the row's
    ring of recent tokens, sign-aware like the presence penalty), and the
    stop-id floor for greedy rows (sampled rows are floored after the filter,
    in ``apply_min_tokens_floor``). Both compile out when off."""
    if WINDOW > 0:
        pen = tl.load(wpen_ptr + row)
        factor = tl.zeros_like(vals) + 1.0
        for w in tl.static_range(WINDOW):
            tok = tl.load(hist_ptr + row * hist_stride + w)
            factor = tl.where(offs == tok, factor * pen, factor)
        vals = tl.where(vals > 0, vals / factor, vals * factor)
    if NUM_STOP > 0:
        floor = (temp == 0) & (tl.load(count_ptr + row) < tl.load(min_tok_ptr + row))
        for s in tl.static_range(NUM_STOP):
            vals = tl.where(floor & (offs == tl.load(stop_ptr + s)), -float("inf"), vals)
    return vals


@triton.jit
def _penalize(vals, seen, penalty):
    """Repetition penalty: divide positive logits, multiply negative ones."""
    return tl.where(seen, tl.where(vals > 0, vals / penalty, vals * penalty), vals)


@triton.jit
def _split_partials_kernel(
    logits_ptr, temperature_ptr, penalty_ptr, seen_mask_ptr,
    hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
    chunk_max_ptr, chunk_sum_ptr, chunk_arg_ptr,
    V, CHUNK,
    stride_b, stride_v, mask_stride_b, mask_stride_v, part_stride_b,
    BLOCK_SIZE: tl.constexpr,
    APPLY_PENALTY: tl.constexpr,
    INCLUDE_GREEDY: tl.constexpr,
    WINDOW: tl.constexpr,
    NUM_STOP: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    temp = tl.load(temperature_ptr + row)
    if INCLUDE_GREEDY:
        inv_temp = tl.where(temp == 0, 1.0, 1.0 / tl.maximum(temp, 1e-30))
    else:
        inv_temp = 1.0 / temp
    if APPLY_PENALTY:
        penalty = tl.load(penalty_ptr + row)

    lo = split * CHUNK
    hi = tl.minimum(lo + CHUNK, V)

    # Online softmax over this chunk: one read, running max and sum.
    run_max = -float("inf")
    run_sum = tl.zeros([], dtype=tl.float32)
    arg = tl.zeros([], dtype=tl.int32)
    for v_start in range(lo, hi, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hi
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=-float("inf"),
        ).to(tl.float32)
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            vals = _penalize(vals, seen, penalty)
        vals = _history_adjust(
            vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
            WINDOW, NUM_STOP,
        )
        scaled = tl.where(mask, vals * inv_temp, -float("inf"))
        block_max = tl.max(scaled)
        if INCLUDE_GREEDY:
            is_new = block_max > run_max
            arg = tl.where(
                is_new, v_start + tl.argmax(scaled, axis=0).to(tl.int32), arg,
            )
        new_max = tl.maximum(run_max, block_max)
        # rescale what we had, then fold this block in. Guard both terms while
        # no finite logit has been seen: `-inf - -inf` is NaN, and an all -inf
        # leading block would carry a NaN sum into the combine step.
        rescale = tl.where(run_max == -float("inf"), 0.0, tl.exp(run_max - new_max))
        terms = tl.where(
            mask & (scaled > -float("inf")), tl.exp(scaled - new_max), 0.0
        )
        run_sum = run_sum * rescale + tl.sum(terms)
        run_max = new_max

    base = row * part_stride_b + split
    tl.store(chunk_max_ptr + base, run_max)
    tl.store(chunk_sum_ptr + base, run_sum)
    if INCLUDE_GREEDY:
        tl.store(chunk_arg_ptr + base, arg)


@triton.jit
def _split_combine_kernel(
    chunk_max_ptr, chunk_sum_ptr, chunk_arg_ptr,
    row_max_ptr, row_inv_sum_ptr, row_arg_ptr,
    NSPLIT, part_stride_b,
    BLOCK_N: tl.constexpr,
    INCLUDE_GREEDY: tl.constexpr,
):
    """Fold the per-chunk partials into one max/sum/argmax per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < NSPLIT
    base = row * part_stride_b + offs
    cmax = tl.load(chunk_max_ptr + base, mask=mask, other=-float("inf"))
    csum = tl.load(chunk_sum_ptr + base, mask=mask, other=0.0)
    gmax = tl.max(cmax)
    # each chunk's sum is relative to its own max, so rescale before adding
    gsum = tl.sum(tl.where(mask, csum * tl.exp(cmax - gmax), 0.0))
    tl.store(row_max_ptr + row, gmax)
    tl.store(row_inv_sum_ptr + row, 1.0 / tl.maximum(gsum, 1e-30))
    if INCLUDE_GREEDY:
        carg = tl.load(chunk_arg_ptr + base, mask=mask, other=0)
        winner = tl.argmax(tl.where(mask, cmax, -float("inf")), axis=0)
        tl.store(row_arg_ptr + row, tl.sum(tl.where(offs == winner, carg, 0)))


@triton.jit
def _split_write_kernel(
    logits_ptr, temperature_ptr, penalty_ptr, seen_mask_ptr,
    hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
    row_max_ptr, row_inv_sum_ptr, row_arg_ptr, probs_ptr,
    V, CHUNK,
    stride_b, stride_v, out_stride_b, out_stride_v,
    mask_stride_b, mask_stride_v,
    BLOCK_SIZE: tl.constexpr,
    APPLY_PENALTY: tl.constexpr,
    INCLUDE_GREEDY: tl.constexpr,
    WINDOW: tl.constexpr,
    NUM_STOP: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    temp = tl.load(temperature_ptr + row)
    if INCLUDE_GREEDY:
        is_greedy = temp == 0
        inv_temp = tl.where(is_greedy, 1.0, 1.0 / tl.maximum(temp, 1e-30))
        arg = tl.load(row_arg_ptr + row)
    else:
        inv_temp = 1.0 / temp
    if APPLY_PENALTY:
        penalty = tl.load(penalty_ptr + row)
    gmax = tl.load(row_max_ptr + row)
    inv_sum = tl.load(row_inv_sum_ptr + row)

    lo = split * CHUNK
    hi = tl.minimum(lo + CHUNK, V)
    for v_start in range(lo, hi, BLOCK_SIZE):
        offs = v_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < hi
        vals = tl.load(
            logits_ptr + row * stride_b + offs * stride_v,
            mask=mask, other=0.0,
        ).to(tl.float32)
        if APPLY_PENALTY:
            seen = tl.load(
                seen_mask_ptr + row * mask_stride_b + offs * mask_stride_v,
                mask=mask, other=0,
            ).to(tl.int1)
            vals = _penalize(vals, seen, penalty)
        vals = _history_adjust(
            vals, offs, row, temp, hist_ptr, hist_stride, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
            WINDOW, NUM_STOP,
        )
        out = tl.exp(vals * inv_temp - gmax) * inv_sum
        if INCLUDE_GREEDY:
            out = tl.where(is_greedy, tl.where(offs == arg, 1.0, 0.0), out)
        tl.store(
            probs_ptr + row * out_stride_b + offs * out_stride_v,
            out, mask=mask,
        )


# A fixed chunk, not one derived from the batch, so the block count scales with
# B and no batch size turns the split into a long serial walk. 248k vocab is 16
# chunks: 256 blocks at bs=16.
_SPLIT_CHUNK = 16384
_SPLIT_MAX_BLOCK = 8192


def _split_count(batch: int, vocab: int, device: torch.device) -> int:
    """How many chunks to cut the vocab into.

    The vocab's call alone: any vocab wider than one chunk splits, so for real
    LLM vocabularies the fused kernel is only reached from tests. ``batch`` is
    accepted for a future rule.
    """
    del batch, device
    return max(1, -(-vocab // _SPLIT_CHUNK))


def _split_v_softmax(
    logits, temperature, pen_ptr, mask_ptr, probs, nsplit,
    apply_penalty, include_greedy, mask_stride_b, mask_stride_v, hist,
):
    """The three-kernel path; see the kernels above for why."""
    B, V = logits.shape
    chunk = -(-V // nsplit)
    # sized to the chunk: a small block makes a long chunk many serial iterations
    block = min(_SPLIT_MAX_BLOCK, triton.next_power_of_2(chunk))
    hist_args, hist_flags = hist
    opts = dict(
        BLOCK_SIZE=block,
        APPLY_PENALTY=apply_penalty,
        INCLUDE_GREEDY=include_greedy,
        **hist_flags,
        num_warps=8,
        num_stages=2,
    )
    f32 = dict(device=logits.device, dtype=torch.float32)
    cmax = torch.empty((B, nsplit), **f32)
    csum = torch.empty((B, nsplit), **f32)
    carg = torch.empty((B, nsplit), device=logits.device, dtype=torch.int32)
    rmax = torch.empty(B, **f32)
    rinv = torch.empty(B, **f32)
    rarg = torch.empty(B, device=logits.device, dtype=torch.int32)
    with torch.cuda.device(logits.device):
        _split_partials_kernel[(B, nsplit)](
            logits, temperature, pen_ptr, mask_ptr, *hist_args, cmax, csum, carg,
            V, chunk,
            logits.stride(0), logits.stride(1), mask_stride_b, mask_stride_v,
            cmax.stride(0), **opts,
        )
        _split_combine_kernel[(B,)](
            cmax, csum, carg, rmax, rinv, rarg,
            nsplit, cmax.stride(0),
            BLOCK_N=triton.next_power_of_2(nsplit),
            INCLUDE_GREEDY=include_greedy,
            num_warps=4,
        )
        _split_write_kernel[(B, nsplit)](
            logits, temperature, pen_ptr, mask_ptr, *hist_args, rmax, rinv, rarg, probs,
            V, chunk,
            logits.stride(0), logits.stride(1),
            probs.stride(0), probs.stride(1),
            mask_stride_b, mask_stride_v, **opts,
        )


def _history_kernel_args(history: "HistoryRows | None", dummy: torch.Tensor):
    """``(pointer args, constexprs)`` for ``_history_adjust``; ``dummy`` stands
    in for the pointers of a processor that is off."""
    window = history is not None and history.tokens is not None and history.tokens.shape[1] > 0
    floor = history is not None and history.stop_ids is not None
    args = (
        history.tokens if window else dummy,
        history.tokens.stride(0) if window else 0,
        history.window_penalty if window else dummy,
        history.count if floor else dummy,
        history.min_tokens if floor else dummy,
        history.stop_ids if floor else dummy,
    )
    flags = dict(
        WINDOW=history.tokens.shape[1] if window else 0,
        NUM_STOP=history.stop_ids.numel() if floor else 0,
    )
    return args, flags


def fused_temperature_softmax(
    logits: torch.Tensor,       # [B, V]
    temperature: torch.Tensor,  # [B]
    penalty: torch.Tensor | None = None,    # [B]
    seen_mask: torch.Tensor | None = None,  # [B, V] bool
    include_greedy: bool = False,
    history: "HistoryRows | None" = None,
) -> torch.Tensor:
    """softmax(apply_penalty(logits) / temperature) fused, returns [B, V] float32.

    When include_greedy=True, rows with temperature == 0 produce a one-hot
    distribution at argmax (equivalent to argmax sampling via multinomial).
    ``history`` adds its windowed penalty, and its stop-id floor for greedy
    rows, inside the same kernels.
    """
    B, V = logits.shape
    probs = torch.empty_like(logits, dtype=torch.float32)
    apply_penalty = penalty is not None and seen_mask is not None
    pen_ptr = penalty if apply_penalty else logits
    mask_ptr = seen_mask if apply_penalty else logits
    mask_stride_b = seen_mask.stride(0) if apply_penalty else 0
    mask_stride_v = seen_mask.stride(1) if apply_penalty else 0
    hist = _history_kernel_args(history, logits)

    nsplit = _split_count(B, V, logits.device)
    if nsplit > 1:
        _split_v_softmax(
            logits, temperature, pen_ptr, mask_ptr, probs, nsplit,
            apply_penalty, include_greedy, mask_stride_b, mask_stride_v, hist,
        )
        return probs

    grid = (B,)
    with torch.cuda.device(logits.device):
        # BLOCK_SIZE is picked by @triton.autotune (not passed here). The first
        # launch for a given key benchmarks every config (do_bench), which can
        # leave probs in a state not ordered on the current stream relative to
        # the downstream FlashInfer read -> garbage. We can't gate the sync on
        # autotune alone (that wasn't enough on its own); pairing it with the
        # device context above is what fixes it. Detect the autotune call by the
        # config cache growing, and sync only then -- steady state stays sync-free.
        cache = getattr(_fused_sampling_prep_kernel, "cache", None)
        cache_size_before = len(cache) if cache is not None else 0
        _fused_sampling_prep_kernel[grid](
            logits, temperature, pen_ptr, mask_ptr, *hist[0], probs,
            V,
            logits.stride(0), logits.stride(1),
            probs.stride(0), probs.stride(1),
            mask_stride_b, mask_stride_v,
            APPLY_PENALTY=apply_penalty,
            INCLUDE_GREEDY=include_greedy,
            **hist[1],
        )
        if cache is not None and len(cache) > cache_size_before:
            torch.cuda.current_stream().synchronize()
    return probs


def apply_min_p(probs: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """Min-p filter on a ``[B, V]`` distribution: zero every probability below
    ``min_p`` times the row's largest, then renormalise.

    The same tokens HF's ``MinPLogitsWarper`` keeps (``probs >= min_p * max``,
    so the argmax always survives). Rows with ``min_p == 0`` and one-hot
    (greedy) rows come back unchanged. No CPU branches or data-dependent
    shapes, so it can sit inside a captured graph.
    """
    threshold = probs.amax(dim=-1, keepdim=True) * min_p[:, None]
    kept = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
    return kept / kept.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Generation-aware processors
#
# Two independent concerns, each off unless the node's SamplerSpec enables it:
#
# * What a request has generated (``GenerationHistory`` / ``HistoryRows``):
#   the windowed frequency penalty (applied in ``fused_temperature_softmax``)
#   and the min-tokens floor on stop ids (``apply_min_tokens_floor``).
# * The filter order (``FilterOrder``): top-p before top-k, with a minimum to
#   keep (``filter_top_k_top_p``).
#
# All of it is tensor ops on per-row device state, so the same code runs
# eagerly and inside a captured graph.
# ---------------------------------------------------------------------------


@dataclass
class HistoryRows:
    """A step's rows of ``GenerationHistory`` and the settings that read them,
    ``[B]`` unless noted. ``None`` leaves a processor out; on the graph path
    that is decided by the node's ``SamplerSpec``, not the batch."""
    count: torch.Tensor                         # int32, tokens generated so far
    tokens: torch.Tensor | None = None          # [B, W] int64 ring, -1 where empty
    window: torch.Tensor | None = None          # int32, 0 = no windowed penalty
    window_penalty: torch.Tensor | None = None  # float32
    min_tokens: torch.Tensor | None = None      # int32
    stop_ids: torch.Tensor | None = None        # [S] int64, the ids ``min_tokens`` bars


@dataclass
class FilterOrder:
    """Per-row top-k/top-p order: HF's top-p first with a minimum to keep."""
    top_p_first: torch.Tensor   # [B] bool
    min_keep: torch.Tensor      # [B] int32


def _kept(probs: torch.Tensor) -> torch.Tensor:
    return (probs > 0).sum(dim=-1)


# Vocabularies up to this size are filtered by `_fused_filter_kernel`, one
# program per row holding the whole row; larger ones by FlashInfer's renorms.
FUSED_FILTER_MAX_VOCAB = 16384
FUSED_FILTER_WARPS = 4


@triton.jit
def _count_ge(x_bits, t):
    return tl.sum((x_bits >= t).to(tl.int32), axis=0)


@triton.jit
def _kth_largest_bits(x_bits, k):
    """The largest threshold ``t`` (float bits) with at least ``k`` entries >= t:
    the k-th largest value's bits. Non-negative floats order like their bits."""
    lo = 0
    hi = 0x7F800000
    for _ in range(31):
        mid = lo + (hi - lo + 1) // 2
        ok = _count_ge(x_bits, mid) >= k
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid - 1)
    return lo


@triton.jit
def _fused_filter_kernel(
    probs_ptr, out_ptr, top_k_ptr, top_p_ptr, first_k_ptr, min_keep_ptr,
    V, stride_in, stride_out,
    HAS_MIN_KEEP: tl.constexpr, BLOCK: tl.constexpr,
):
    """``filter_top_k_top_p`` for one row, in registers: the top-p count of the
    (top-k-first rows: renormalised top-k) distribution by a bitwise search for
    its cutoff, raised to ``min_keep``, clipped to ``k``; then the top-``keep``
    renorm. No sort and one launch for the batch."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    valid = offs < V
    p = tl.load(probs_ptr + row * stride_in + offs, mask=valid, other=0.0)
    p = tl.maximum(p, 0.0)
    bits = p.to(tl.int32, bitcast=True)
    k = tl.load(top_k_ptr + row)
    first_k = tl.load(first_k_ptr + row)
    top_p = tl.load(top_p_ptr + row)

    # the set top-p runs over: the top-`first_k` (all of it for top-p-first rows)
    t_first = 0
    if first_k < V:  # a branch, not tl.where: skip the search when nothing is cut
        t_first = _kth_largest_bits(bits, first_k)
    in_first = bits >= t_first
    mass_first = tl.sum(tl.where(in_first, p, 0.0), axis=0)
    # top-p keeps token i iff the mass strictly above it is < p * mass. That mass
    # falls as i's value rises, so the kept tokens are those >= the smallest
    # threshold t with mass(> t) < p * mass: search for it on the bits
    target = top_p * mass_first
    lo = 0
    hi = 0x7F800000
    for _ in range(31):
        mid = lo + (hi - lo) // 2
        above = tl.sum(tl.where(in_first & (bits > mid), p, 0.0), axis=0)
        ok = above < target
        hi = tl.where(ok, mid, hi)
        lo = tl.where(ok, lo, mid + 1)
    n = _count_ge(bits, tl.maximum(lo, t_first))
    # p = 1 removes nothing; the search can't say so, since "the mass above" the
    # smallest tokens rounds to the whole mass in float32
    n = tl.where(top_p >= 1.0, _count_ge(bits, t_first), n)
    if HAS_MIN_KEEP:
        n = tl.maximum(n, tl.load(min_keep_ptr + row))
    keep = tl.minimum(n, k)
    t_keep = 0
    if keep < V:
        t_keep = _kth_largest_bits(bits, keep)
    out = tl.where(valid & (bits >= t_keep), p, 0.0)
    total = tl.sum(out, axis=0)
    tl.store(out_ptr + row * stride_out + offs, out / tl.maximum(total, 1e-30), mask=valid)


def filter_top_k_top_p(
    probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor,
    order: FilterOrder | None = None, fused: bool = True,
) -> torch.Tensor:
    """The top-k/top-p filtered, renormalised distribution, in either order.

    Both filters keep a prefix of the tokens sorted by probability, so each
    order reduces to one per-row count ``L`` and a single top-``L`` renorm:

    * top-k first (FlashInfer's order): ``L`` = what top-p keeps of the
      renormalised top-k.
    * top-p first (HF ``TopPLogitsWarper`` then ``TopKLogitsWarper``):
      ``L = min(k, n_p)``, ``n_p`` what top-p keeps of the full distribution,
      raised to ``min_keep`` as HF's ``min_tokens_to_keep`` does.
    """
    import flashinfer

    vocab = probs.shape[1]
    k = torch.where(top_k > 0, top_k, vocab).to(torch.int32)
    # one top-p pass for both orders: a top-p-first row's top-k here is the
    # whole vocabulary, which leaves its distribution as it is
    first_k = (k if order is None else torch.where(order.top_p_first, vocab, k)).to(torch.int32)
    if fused and probs.is_cuda and vocab <= FUSED_FILTER_MAX_VOCAB and probs.dtype == torch.float32:
        out = torch.empty_like(probs)
        _fused_filter_kernel[(probs.shape[0],)](
            probs, out, k, top_p.to(torch.float32), first_k,
            order.min_keep if order is not None else k,  # k: a placeholder pointer
            vocab, probs.stride(0), out.stride(0),
            HAS_MIN_KEEP=order is not None,
            BLOCK=triton.next_power_of_2(vocab), num_warps=FUSED_FILTER_WARPS,
        )
        return out
    n = _kept(flashinfer.sampling.top_p_renorm_probs(
        flashinfer.sampling.top_k_renorm_probs(probs, first_k), top_p,
    ))
    if order is not None:
        n = torch.maximum(n, order.min_keep)
    return flashinfer.sampling.top_k_renorm_probs(probs, torch.minimum(n.to(torch.int32), k))


def apply_min_tokens_floor(probs: torch.Tensor, rows: HistoryRows) -> torch.Tensor:
    """Zero the stop ids for rows that have generated fewer than ``min_tokens``
    tokens, and renormalise. Upstream floors after top-k/top-p, so this runs on
    the filtered distribution; a row whose whole kept mass is stop ids keeps
    its unfloored distribution."""
    floor = rows.count < rows.min_tokens
    cols = probs.index_select(1, rows.stop_ids)
    floored = probs.index_copy(1, rows.stop_ids, torch.where(floor[:, None], 0.0, cols))
    mass = floored.sum(dim=-1, keepdim=True)
    return torch.where(mass > 0, floored / mass.clamp_min(1e-30), probs)


def advance_history(rows: HistoryRows, tokens: torch.Tensor) -> None:
    """Fold this step's sampled tokens into ``rows`` in place: into the ring at
    ``count % window`` (order inside the window doesn't matter to a frequency
    penalty), then the count."""
    if rows.tokens is not None:
        window = rows.window.clamp_min(1)
        pos = (rows.count % window).long()[:, None]
        new = torch.where(
            (rows.window > 0)[:, None], tokens.reshape(-1, 1).to(rows.tokens.dtype),
            rows.tokens.gather(1, pos),
        )
        rows.tokens.scatter_(1, pos, new)
    rows.count.add_(1)


class GenerationHistory:
    """Per-request device state for the history-aware processors: how many
    tokens each request has generated, and a ring of its last ``window``.

    Shared by the eager sampler and the graph path, so a request that samples
    its first token eagerly and decodes under a graph keeps one history. The
    master rows (``count``, ``tokens``) live on the device and are only ever
    written there: the eager sampler reads and writes them around its draw; the
    graph path gathers this step's rows into fixed per-step buffers inline
    (``gather_step``), the replay advances those, and ``scatter_step`` writes
    the real rows back after it -- the RNG offset's round trip.

    Slot bookkeeping is host-side and may run on the main thread; every device
    write (growing the masters, resetting a new request's row) is deferred to
    the next gather on the GPU thread, so it is ordered behind the commits of
    the request that held the slot before. Slot 0 is a scratch row: padding
    rows read it, and nothing real writes it.
    """

    def __init__(self, window: int, device: torch.device, capacity: int = 64):
        self.window = window
        self.device = device
        self._lock = threading.Lock()
        self._rid_to_slot: dict[Any, int] = {}
        self._capacity = capacity
        self._free = list(range(capacity - 1, 0, -1))
        self._pending_reset: set[int] = set()
        self.count = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.tokens = torch.full((capacity, window), -1, dtype=torch.long, device=device)
        self._stager = PinnedStager(torch.long, numel=capacity)
        # graph path, allocated by ``allocate_step_buffers``
        self.step_count: torch.Tensor | None = None
        self.step_tokens: torch.Tensor | None = None
        self._step_idx: torch.Tensor | None = None
        self._step_real_bs = 0

    def register(self, rid) -> None:
        with self._lock:
            if rid in self._rid_to_slot:
                return
            if not self._free:
                old = self._capacity
                self._capacity *= 2
                self._free = list(range(self._capacity - 1, old - 1, -1))
            slot = self._free.pop()
            self._rid_to_slot[rid] = slot
            self._pending_reset.add(slot)

    def unregister(self, rid) -> None:
        with self._lock:
            slot = self._rid_to_slot.pop(rid, None)
            if slot is not None:
                self._pending_reset.discard(slot)
                self._free.append(slot)

    def _sync_device(self) -> None:
        """Apply the deferred device writes. GPU thread only."""
        with self._lock:
            capacity = self._capacity
            resets, self._pending_reset = self._pending_reset, set()
        if capacity > self.count.shape[0]:
            n = self.count.shape[0]
            count = torch.zeros(capacity, dtype=self.count.dtype, device=self.device)
            tokens = torch.full((capacity, self.window), -1, dtype=torch.long, device=self.device)
            count[:n].copy_(self.count)
            tokens[:n].copy_(self.tokens)
            self.count, self.tokens = count, tokens
        for slot in resets:
            self.count[slot:slot + 1].zero_()
            if self.window:
                self.tokens[slot:slot + 1].fill_(-1)

    def _slots(self, request_ids, padded_bs: int) -> list[int]:
        get = self._rid_to_slot.get
        rows = [get(rid, 0) for rid in request_ids]
        rows.extend([0] * (padded_bs - len(rows)))
        return rows

    # -- eager --------------------------------------------------------------

    def read(self, request_ids) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(rows, tokens, count)`` for ``request_ids``: copies, which the
        caller advances and hands back to ``write``."""
        self._sync_device()
        rows = torch.empty(len(request_ids), dtype=torch.long, device=self.device)
        self._stager.copy_(rows, self._slots(request_ids, len(request_ids)))
        return rows, self.tokens.index_select(0, rows), self.count.index_select(0, rows)

    def write(self, rows: torch.Tensor, tokens: torch.Tensor, count: torch.Tensor) -> None:
        self.count.index_copy_(0, rows, count)
        if self.window:
            self.tokens.index_copy_(0, rows, tokens)

    # -- graph --------------------------------------------------------------

    def allocate_step_buffers(self, max_bs: int) -> None:
        if self.step_count is not None and self.step_count.shape[0] >= max_bs:
            return
        self.step_count = torch.zeros(max_bs, dtype=torch.int32, device=self.device)
        self.step_tokens = torch.full((max_bs, self.window), -1, dtype=torch.long, device=self.device)
        self._step_idx = torch.zeros(max_bs, dtype=torch.long, device=self.device)

    def gather_step(self, request_ids, padded_bs: int) -> None:
        """This step's rows into the per-step buffers. Inline on the GPU
        thread, after the previous step's ``scatter_step``: like the RNG
        offset, these are single-buffered and read what that step wrote."""
        self._sync_device()
        idx = self._step_idx[:padded_bs]
        self._stager.copy_(idx, self._slots(request_ids, padded_bs))
        torch.index_select(self.count, 0, idx, out=self.step_count[:padded_bs])
        if self.window:
            torch.index_select(self.tokens, 0, idx, out=self.step_tokens[:padded_bs])
        self._step_real_bs = len(request_ids)

    def scatter_step(self) -> None:
        """The replay's advanced rows back to their masters; real rows only."""
        n = self._step_real_bs
        idx = self._step_idx[:n]
        self.count.index_copy_(0, idx, self.step_count[:n])
        if self.window:
            self.tokens.index_copy_(0, idx, self.step_tokens[:n])


@dataclass
class SamplingConfig:
    # Sizes the per-request seen-token mask for the repetition penalty. When set,
    # it MUST equal the model's logit width (lm_head/codec_head output dim): the
    # mask is indexed as ``[B, vocab_size]`` against ``logits[B, V]``, and on the
    # CUDA-graph path it also gates allocation of the in-graph penalty buffers.
    vocab_size: int | None = None
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 1
    ignore_eos: bool = False # used for benchmark parity
    repetition_penalty: float = 1
    min_p: float = 0.0  # 0 = disabled; see ``SamplingReqConfig.min_p``
    # see the fields of the same names on ``SamplingReqConfig``
    repetition_window: int = 0
    min_tokens: int = 0
    top_p_first: bool = False
    top_p_min_keep: int = 1
    _seed: int = 0 # set by the conductor

    def set_seed(self, seed: int):
        self._seed = seed

    @property
    def presence_penalty(self) -> float:
        """The penalty the seen-token mask applies: a windowed request's
        penalty is count-based (``HistoryRows``) instead."""
        return self.repetition_penalty if self.repetition_window == 0 else 1.0

    @property
    def window_penalty(self) -> float:
        return self.repetition_penalty if self.repetition_window > 0 else 1.0

    @property
    def uses_history(self) -> bool:
        return self.repetition_window > 0 or self.min_tokens > 0

    @property
    def uses_filter_order(self) -> bool:
        return self.top_p_first or self.top_p_min_keep > 1

    @property
    def seed(self):
        return self._seed


@dataclass
class BaseSampler(ABC):
    def _broadcast_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """In-place broadcast of ``tokens`` from rank 0 to all TP ranks.

        No-op for ``tp_group`` of size 1 (trivial group / non-TP) or
        unset. Subclasses set ``self.tp_group`` so all TP ranks agree
        on the sampled token (otherwise per-rank RNG diverges →
        mid-sequence garbage, hangs on EOS, KV drift).
        """
        tp_group = getattr(self, "tp_group", None)
        if tp_group is None or tp_group.world_size == 1:
            return tokens
        return tp_group.broadcast(tokens, src=0)

    @abstractmethod
    def sample(
        self, request_ids: list[str], logits: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        pass


@dataclass
class SeenTokenMask:
    request_id: str
    _seen_token_mask: torch.Tensor | None

    @classmethod
    def new(cls, request_id: str, vocab_size: int | None, device):
        return cls(
            request_id=request_id,
            _seen_token_mask=torch.zeros(
                vocab_size, dtype=torch.bool, device=device
            ) if vocab_size is not None else None,

        )

    def add_tokens(self, tokens: torch.Tensor | int):
        if self._seen_token_mask is None:
            logger.warning(
                "Calling add_tokens on an uninitialized SeenTokenMask, i.e., "
                "one where the vocab_size was provided in the SamplingConfig or "
                "the SamplingConfig has not yet been registered with the Sampler.s"
            )
            return
        idx = torch.as_tensor(
            tokens, dtype=torch.long, device=self._seen_token_mask.device,
        ).reshape(-1)
        self._seen_token_mask.scatter_(0, idx, True)


@dataclass
class Sampler(BaseSampler):
    device: torch.device
    _sampling_config: dict[str, SamplingConfig] = field(default_factory=dict)
    _seen_token_mask: dict[str, SeenTokenMask]= field(default_factory=dict)
    # Per-request RNG offset, advanced once per sampled step. Paired with the
    # request's fixed seed, this steps the philox stream so deterministic
    # (seeded) sampling draws a fresh number each step — otherwise identical
    # (seed, offset=0) draws repeat forever and stable logits never reach EOS.
    _step_offset: dict[str, int] = field(default_factory=dict)
    tp_group: "CommGroup | None" = None  # noqa: F821
    # Set by a ``SamplerResource`` whose spec enables a generation-aware
    # processor; ``None`` keeps every step on the plain path.
    history: GenerationHistory | None = None
    stop_ids: torch.Tensor | None = None
    enable_top_p_first: bool = False
    _settings_stager: PinnedStager | None = None

    def add_request(self, request_id: str):
        self._sampling_config[request_id] = SamplingConfig()
        self._seen_token_mask[request_id] =  SeenTokenMask.new(
            request_id,
            vocab_size=None,
            device=self.device
        )
        self._step_offset[request_id] = 0
        # lazy init _seen_token_mask, taking vocab size from logits or cfg

    def get_token_mask(self, request_id: str):
        return self._seen_token_mask[request_id]

    def remove_request(self, request_id: str):
        if request_id in self._sampling_config:
            del self._sampling_config[request_id]
        if request_id in self._seen_token_mask:
            del self._seen_token_mask[request_id]
        self._step_offset.pop(request_id, None)

    def set_config(self, request_id: str, **kwargs):
        old_vocab_size = self._sampling_config[request_id].vocab_size
        curr_config = asdict(self._sampling_config[request_id])
        kwargs = {k: arg for k, arg in kwargs.items() if k in curr_config.keys()}
        self._sampling_config[request_id] = SamplingConfig(**{
            **curr_config, **kwargs
        })

        new_vocab_size = self._sampling_config[request_id].vocab_size
        if old_vocab_size != new_vocab_size:
            self._seen_token_mask[request_id] = SeenTokenMask.new(
                request_id=request_id,
                vocab_size=new_vocab_size,
                device=self.device
            )

    # sampling runs inside the forward; nothing here is worth tracing
    @torch.compiler.disable
    def sample(
        self, request_ids: list[str], logits: torch.Tensor, apply_filters: bool = True, **kwargs
    ) -> torch.Tensor:
        """Return the sampled tokens as a single [B] int tensor.

        Callers that want a per-rid mapping can slice `tokens[i:i+1]` using
        the rid order in `request_ids`. We return the raw tensor (instead of
        a dict of views) because constructing the dict adds Python overhead
        the hot path doesn't need.
        """
        configs = [self._sampling_config[rid] for rid in request_ids]
        history, rows = self._gather_history(request_ids, configs)
        order = self._gather_filter(configs)
        temperature = torch.tensor([c.temperature for c in configs], device=logits.device)
        top_k = torch.tensor([c.top_k if apply_filters else 0 for c in configs],
                             device=logits.device, dtype=torch.int32)
        top_p = torch.tensor([c.top_p if apply_filters else 1.0 for c in configs], device=logits.device)
        if self.history is not None:
            # a windowed request's penalty is in ``history``, not the mask
            penalties = [c.presence_penalty for c in configs]
        else:
            penalties = [c.repetition_penalty for c in configs]
        r_pen = torch.tensor(penalties, device=logits.device)
        min_p = (
            torch.tensor([c.min_p for c in configs], device=logits.device)
            if any(c.min_p > 0 for c in configs) else None
        )
        seed = torch.tensor([c.seed for c in configs], device=logits.device, dtype=torch.long)
        rand_offset = torch.tensor(
            [self._step_offset.get(rid, 0) for rid in request_ids],
            device=logits.device, dtype=torch.long,
        )

        any_rep_pen = any(p != 1.0 for p in penalties)
        any_greedy = any(c.temperature == 0 for c in configs)
        top_k_zero_count = sum(c.top_k == 0 or not apply_filters for c in configs)

        for rid in request_ids:
            if self._seen_token_mask[rid]._seen_token_mask is None:
                self._seen_token_mask[rid] = SeenTokenMask.new(
                    rid, vocab_size=logits.shape[1],
                    device=self.device
                )

        seen_mask = None
        if any_rep_pen:
            seen_mask = torch.stack(
                [self._seen_token_mask[rid]._seen_token_mask for rid in request_ids], dim=0,
            )

        tokens = sample_tokens(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=r_pen,
            seen_token_mask=seen_mask,
            any_greedy=any_greedy,
            top_k_zero_count=top_k_zero_count,
            seed=seed,
            rand_offset=rand_offset,
            min_p=min_p,
            history=history,
            order=order,
        )

        # TODO: make this scatter async. Currently runs 2 kernels per rid
        # (broadcast-True + index_put) on the default stream, serializing N=bs
        # small launches that add up (~500 µs at bs=8 for Orpheus with
        # repetition_penalty=1.3). Two options to fix:
        #   (a) Shared [max_concurrent, V] buffer with rid→slot mapping; replace
        #       the loop with a single batched `buf[slots, tokens] = True`
        #       scatter — one launch instead of N.
        #   (b) Issue the updates on a side CUDA stream so the main stream
        #       (next prefill/decode) doesn't wait. The next sample() for the
        #       same rid would need to sync, but amortized over a full
        #       generation this is cheap.
        tokens = self._broadcast_tokens(tokens)

        if history is not None:
            advance_history(history, tokens)
            self.history.write(rows, history.tokens, history.count)

        if any_rep_pen:
            for i, rid in enumerate(request_ids):
                self._seen_token_mask[rid].add_tokens(tokens[i:i+1])

        # FlashInfer consumes one offset unit per sampled row. The XPU kernel
        # consumes one Philox region per logit in its row, so advancing by one
        # would make consecutive decode steps reuse overlapping RNG regions.
        offset_stride = _rng_offset_stride(
            logits.device.type, logits.shape[-1],
        )
        for rid in request_ids:
            self._step_offset[rid] = (
                self._step_offset.get(rid, 0) + offset_stride
            )

        return tokens

    def _stage(self, rows: list[list[float]]) -> torch.Tensor:
        """Per-request settings ``[len(rows), B]`` on the device in one pinned H2D
        (float32 holds every int a setting can take exactly)."""
        values = np.asarray(rows, dtype=np.float32)
        if self._settings_stager is None:
            self._settings_stager = PinnedStager(torch.float32, numel=values.size)
        dev = torch.empty(values.shape, dtype=torch.float32, device=self.device)
        self._settings_stager.copy_(dev, values.reshape(-1))
        return dev

    def _gather_history(
        self, request_ids: list[str], configs: list[SamplingConfig],
    ) -> tuple["HistoryRows | None", torch.Tensor | None]:
        """This batch's ``HistoryRows`` and the master rows they came from;
        ``None`` when the node keeps no history or no row uses it."""
        if self.history is None or not any(c.uses_history for c in configs):
            return None, None
        rows, tokens, count = self.history.read(request_ids)
        dev = self._stage([
            [c.repetition_window for c in configs],
            [c.window_penalty for c in configs],
            [c.min_tokens for c in configs],
        ])
        windowed = self.history.window > 0
        return HistoryRows(
            count=count,
            tokens=tokens if windowed else None,
            window=dev[0].to(torch.int32) if windowed else None,
            window_penalty=dev[1] if windowed else None,
            min_tokens=dev[2].to(torch.int32) if self.stop_ids is not None else None,
            stop_ids=self.stop_ids,
        ), rows

    def _gather_filter(self, configs: list[SamplingConfig]) -> "FilterOrder | None":
        """This batch's ``FilterOrder``; ``None`` when the node lacks it or no
        row asks for HF's order."""
        if not self.enable_top_p_first or not any(c.uses_filter_order for c in configs):
            return None
        dev = self._stage([[c.top_p_first for c in configs], [c.top_p_min_keep for c in configs]])
        return FilterOrder(top_p_first=dev[0] != 0, min_keep=dev[1].to(torch.int32))


def _sample_cuda(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: float | torch.Tensor,
    seen_token_mask: torch.Tensor | None,
    run_greedy: bool,
    top_k_zero_count: int | None,
    seed: torch.Tensor | None,
    rand_offset: torch.Tensor | None,
    min_p: torch.Tensor | None = None,
    history: HistoryRows | None = None,
    order: FilterOrder | None = None,
) -> torch.Tensor:
    """Sample normalized CUDA inputs with FlashInfer."""
    import flashinfer

    # Pin the Triton prep kernel (writes probs) and the FlashInfer sampler
    # (reads probs) to the same device/stream so the write-before-read is
    # ordered without an explicit sync. Otherwise FlashInfer runs on the
    # worker's current-device stream while probs lives off-device (e.g. BAGEL
    # LLM on rank 1) — a cross-stream race that yields garbage.
    with torch.cuda.device(logits.device):
        # One Triton kernel fuses (optional rep-penalty, and the history's
        # windowed penalty) + (temperature-scaled softmax) + (argmax → one-hot
        # for greedy rows). FlashInfer's sample-from-probs then deterministically
        # picks argmax on one-hot rows, matching greedy semantics.
        probs = fused_temperature_softmax(
            logits, temperature,
            penalty=repetition_penalty if seen_token_mask is not None else None,
            seen_mask=seen_token_mask,
            include_greedy=run_greedy,
            history=history,
        )
        if min_p is not None:
            probs = apply_min_p(probs, min_p)
        if _needs_explicit_filter(history, order):
            return _filter_floor_and_draw(probs, top_k, top_p, seed, rand_offset, history, order)
        if top_k_zero_count == logits.shape[0]:
            # top-k is off for every row
            result = flashinfer.sampling.top_p_sampling_from_probs(
                probs, top_p,
                deterministic=True,
                seed=seed, offset=rand_offset,
            )
        else:
            result = flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, top_k, top_p,
                deterministic=True,
                seed=seed, offset=rand_offset,
            )
        return result[0] if isinstance(result, tuple) else result


def _needs_explicit_filter(history: HistoryRows | None, order: FilterOrder | None) -> bool:
    """Whether the draw needs the filtered distribution in hand: FlashInfer's
    fused sampler has neither HF's order nor a floor applied after filtering."""
    return order is not None or (history is not None and history.stop_ids is not None)


def _filter_floor_and_draw(
    probs: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor | None,
    offset: torch.Tensor | None,
    history: HistoryRows | None,
    order: FilterOrder | None,
) -> torch.Tensor:
    """Filter, apply the min-tokens floor, and draw; graph-safe."""
    import flashinfer

    probs = filter_top_k_top_p(probs, top_k, top_p, order)
    if history is not None and history.stop_ids is not None:
        probs = apply_min_tokens_floor(probs, history)
    return flashinfer.sampling.sampling_from_probs(probs, deterministic=True, seed=seed, offset=offset)


def _sample_xpu(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: float | torch.Tensor,
    seen_token_mask: torch.Tensor | None,
    run_greedy: bool,
    top_k_zero_count: int | None,
    seed: torch.Tensor | None,
    rand_offset: torch.Tensor | None,
    min_p: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample a whole batch with vllm-xpu-kernels >= 0.1.15.

    The kernel reads one device-side [seed, offset] pair per row. Explicit
    RNG tensors never round-trip through the CPU, including under XPUGraph.
    """
    import vllm_xpu_kernels._xpu_C  # noqa: F401

    batch_size = logits.shape[0]
    scores = logits.float()
    if seen_token_mask is not None:
        penalty = repetition_penalty[:, None]
        penalized = torch.where(
            scores < 0, scores * penalty, scores / penalty,
        )
        scores = torch.where(seen_token_mask, penalized, scores)

    greedy = temperature == 0
    safe_temperature = torch.where(
        greedy, torch.ones_like(temperature), temperature,
    )
    scores = (scores / safe_temperature[:, None]).contiguous()
    greedy_tokens = scores.argmax(dim=-1) if run_greedy else None
    if min_p is not None:
        # the kernel samples from raw logits, so the filter masks those
        probs = scores.softmax(dim=-1)
        threshold = probs.amax(dim=-1, keepdim=True) * min_p[:, None]
        scores = scores.masked_fill(probs < threshold, float("-inf")).contiguous()

    # Raw callers may omit seed and/or offset. Match CUDA's behavior by filling
    # missing values from the default XPU generator. When offset is omitted,
    # reserve one Philox region per logit and advance the generator state.
    # This fallback runs eagerly; graph callers supply explicit RNG tensors.
    if seed is None or rand_offset is None:
        device_index = logits.device.index
        if device_index is None:
            device_index = torch.xpu.current_device()
        generator = torch.xpu.default_generators[device_index]
        default_seed, default_offset = _xpu_generator_seed_offset(
            generator,
            logits.numel() if rand_offset is None else 0,
        )
        if seed is None:
            seed = torch.full(
                (batch_size,), default_seed, dtype=torch.int64, device=logits.device,
            )
        if rand_offset is None:
            rand_offset = (
                torch.arange(batch_size, dtype=torch.int64, device=logits.device)
                * logits.shape[1]
                + default_offset
            )

    seed_offsets = torch.stack(
        (seed.to(device=logits.device, dtype=torch.int64),
         rand_offset.to(device=logits.device, dtype=torch.int64)),
        dim=1,
    ).contiguous()
    kernel_top_k = None
    if top_k_zero_count != batch_size:
        kernel_top_k = top_k.to(torch.int64)
        if top_k_zero_count != 0:
            kernel_top_k = torch.where(kernel_top_k == 0, logits.shape[1], kernel_top_k)
    sampled = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    torch.ops._xpu_C.topk_topp_sampler(
        sampled, None, scores, kernel_top_k, top_p,
        "raw_logits", seed_offsets, 1.0,
    )
    if greedy_tokens is not None:
        sampled = torch.where(greedy, greedy_tokens, sampled)
    return sampled


def sample_tokens(
    logits: torch.Tensor,
    temperature: float | torch.Tensor = 0.6,
    top_k: int | torch.Tensor = 0,
    top_p: float | torch.Tensor = 1.0,
    repetition_penalty: float | torch.Tensor= 1.0,
    seen_token_mask: torch.Tensor | None = None,
    any_greedy: bool | None = None,
    top_k_zero_count: int | None = None,
    seed: torch.Tensor | None = None,
    rand_offset: torch.Tensor | None = None,
    min_p: float | torch.Tensor | None = None,
    history: HistoryRows | None = None,
    order: FilterOrder | None = None,
) -> torch.Tensor:
    """Sample tokens from logits with temperature, top-k, top-p, and repetition penalty.

    Args:
        logits: [batch_size, vocab_size] raw logits from lm_head.
        temperature: Scalar or per-request tensor [batch_size].
            0 = greedy (argmax) for that request. >0 = scaled sampling.
        top_k: Scalar or per-request tensor [batch_size]. 0 = disabled.
        top_p: Scalar or per-request tensor [batch_size]. 1.0 = disabled.
        repetition_penalty: vLLM-style sign-aware penalty (1.0 = disabled).
        seen_token_mask: [batch_size, vocab_size] bool. None = penalty skipped.
        any_greedy: CPU-side hint. When False, skips the argmax/masked_fill/where
            branch entirely. None = unknown → run the full path.
        top_k_zero_count: CPU-side count of requests with top_k == 0.
            0 skips zero-to-vocabulary normalization; batch_size disables top-k
            filtering for the entire batch. Intermediate counts mean mixed
            rows. None = unknown → use the conservative device path.
        seed: Optional per-request int64 tensor [batch_size].
        rand_offset: Optional per-request int64 tensor [batch_size]. On XPU,
            omit seed/offset to use the default generator eagerly; provide both
            on device to capture sampling in an XPUGraph.
        min_p: Scalar or per-request tensor [batch_size]; None/0 = disabled.
            Applied to the temperature-scaled, penalised distribution before
            top-k/top-p (the HF processor order).
        history: Optional ``HistoryRows`` (CUDA only): the windowed penalty and
            the min-tokens floor.
        order: Optional ``FilterOrder`` (CUDA only): HF's top-p-first order.

    Returns:
        tokens: [batch_size] sampled token IDs.
    """
    batch_size, vocab_size = logits.shape

    # Normalize params to tensors [batch_size] for uniform handling
    temperature = _to_tensor(temperature, batch_size, logits.device)
    top_k = _to_tensor(top_k, batch_size, logits.device, dtype=torch.int32)
    top_p = _to_tensor(top_p, batch_size, logits.device)
    if seen_token_mask is not None:
        repetition_penalty = _to_tensor(repetition_penalty, batch_size, logits.device)
    if min_p is not None:
        min_p = _to_tensor(min_p, batch_size, logits.device)

    # Default to the conservative "unknown → do the work" path.
    run_greedy = True if any_greedy is None else any_greedy

    if logits.device.type == "cuda":
        return _sample_cuda(
            logits,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seen_token_mask,
            run_greedy,
            top_k_zero_count,
            seed,
            rand_offset,
            min_p=min_p,
            history=history,
            order=order,
        )
    elif logits.device.type == "xpu":
        if history is not None or order is not None:
            raise ValueError("generation-aware sampling controls are CUDA-only")
        return _sample_xpu(
            logits,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seen_token_mask,
            run_greedy,
            top_k_zero_count,
            seed,
            rand_offset,
            min_p=min_p,
        )
    else:
        raise ValueError(
            f"Sampling is unsupported on device type {logits.device.type!r}; "
            "expected 'cuda' or 'xpu'."
        )


def _to_tensor(
    value: float | int | torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert scalar or tensor to [batch_size] tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype).reshape(-1)
    return torch.full((batch_size,), value, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Graph-safe sampler
# ---------------------------------------------------------------------------
#
# Reads top_k / top_p / temperature from preallocated device tensors so the
# call can sit inside a CUDA graph capture region without allocating, syncing,
# or branching on CPU-side values. The full ``Sampler`` class is *not* graph
# capturable (repetition-penalty state, ``@torch.compiler.disable``, the
# device-context switch inside ``sample_tokens``), so the unrolled MTP loop uses
# this narrower path. ``deterministic=True`` disables the CPU-RNG-seeded path that
# FlashInfer would otherwise take.

def sample_cuda_graphable_gpu(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    apply_penalty: bool = False,
    rep_penalty: torch.Tensor | None = None,
    seen_tokens: torch.Tensor | None = None,
    min_p: torch.Tensor | None = None,
    history: HistoryRows | None = None,
    order: FilterOrder | None = None,
) -> torch.Tensor:
    """Deterministic per-batch top-k/top-p sampling for graph-captured code.

    Routes through the fused Triton prep kernel (``fused_temperature_softmax``)
    so the CUDA-graph path can apply the same vLLM-style repetition penalty as
    the regular ``Sampler``, then samples with
    ``flashinfer.sampling.top_k_top_p_sampling_from_probs`` (``deterministic=True``
    — the graph-safe variant that avoids CPU-seeded RNG paths). Greedy rows keep
    ``temperature == 0`` in the device buffer and the prep kernel turns them into
    a one-hot at argmax on-device (``include_greedy=True`` is a constexpr, no CPU
    branch), so ``from_probs`` returns the argmax regardless of the philox draw.
    Encoding greedy as ``(temperature=1.0, top_k=1)`` instead is NOT greedy:
    top-k rejection sampling accepts any token with no strictly-greater
    probability, so exact top-1 ties (common in bf16 logits) are broken by the
    per-request seed and identical greedy requests diverge.

    The autotune sync inside ``fused_temperature_softmax`` only fires the first
    time a kernel key is seen, which happens during eager warmup — by capture
    time the config is cached, so the captured launch is sync-free.

    Args:
        logits: ``[batch_size, vocab_size]`` raw logits from the codebook head.
        temperature: ``[batch_size]`` float tensor.
        top_k: ``[batch_size]`` int32 tensor. Use ``vocab_size`` to disable.
        top_p: ``[batch_size]`` float tensor. Use ``1.0`` to disable.
        apply_penalty: when True, ``rep_penalty`` + ``seen_tokens`` are applied.
        rep_penalty: ``[batch_size]`` float tensor (1.0 = disabled per row).
        seen_tokens: ``[batch_size, vocab_size]`` bool mask of seen tokens.
        min_p: ``[batch_size]`` float tensor (0.0 = disabled per row); None
            leaves the filter out of the captured graph entirely.
        history, order: the generation-aware processors (``HistoryRows``,
            ``FilterOrder``); None leaves each out of the captured graph. The
            caller advances ``history`` with the sampled tokens.

    Returns:
        ``[batch_size]`` int64 sampled token IDs. FlashInfer's default
        output is int32; we cast to int64 so the caller can index
        ``nn.Embedding`` modules (which require int64 indices) directly.
    """
    import flashinfer

    with torch.cuda.device(logits.device):
        probs = fused_temperature_softmax(
            logits, temperature,
            penalty=rep_penalty if apply_penalty else None,
            seen_mask=seen_tokens if apply_penalty else None,
            include_greedy=True,
            history=history,
        )
        if min_p is not None:
            probs = apply_min_p(probs, min_p)
        if _needs_explicit_filter(history, order):
            return _filter_floor_and_draw(probs, top_k, top_p, seed, offset, history, order).to(torch.int64)
        top_k = torch.where(top_k > 0, top_k, logits.shape[1])
        # NOTE: this is NOT batch-invariant — flashinfer's deterministic RNG
        # folds the batch row index into philox, so identical (probs, seed,
        # offset) yield different tokens at different batch positions. Sampling
        # is thus reproducible only within a fixed batch layout; under
        # continuous batching (shifting positions) a request's stream is not.
        # Measured in test/sampling_test/flashinfer_batch_test.py.
        samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, top_k, top_p, deterministic=True,
            seed=seed, offset=offset,
        )
    return samples.to(torch.int64)


@dataclass
class CudaGraphableSampler(BaseSampler):
    temperature_buf: torch.Tensor
    top_k_buf: torch.Tensor
    top_p_buf: torch.Tensor
    seed_buf: torch.Tensor
    offset_buf: torch.Tensor
    # Repetition-penalty state for the CUDA-graph path. ``None`` for submodules
    # that don't opt into seen-token tracking (then ``apply_penalty`` is a no-op).
    rep_penalty_buf: torch.Tensor | None = None
    seen_tokens_buf: torch.Tensor | None = None  # [bs, V] bool
    # ``None`` for submodules whose ``SamplerSpec`` leaves ``enable_min_p`` off.
    min_p_buf: torch.Tensor | None = None
    # Views into ``SamplerBuffers``' per-step rows; each ``None`` unless the
    # node's spec enables it.
    history: HistoryRows | None = None
    order: FilterOrder | None = None
    tp_group: "CommGroup | None" = None  # noqa: F821

    # Set during graph capture, and used by the cuda graph runner to determine
    # whether requests' seen token buffers should be synced post-replay
    applied_penalty_in_graph: bool = False

    @torch.compiler.disable
    def sample(
        self, request_ids: list[str], logits: torch.Tensor,
        apply_penalty: bool = False,
        apply_filters: bool = True,
    ):
        top_k, top_p = self.top_k_buf, self.top_p_buf
        if not apply_filters:
            # top-k 0 is "off"; captured, these are constant fills
            top_k, top_p = torch.zeros_like(top_k), torch.ones_like(top_p)
        codes = sample_cuda_graphable_gpu(
            logits, self.temperature_buf,
            top_k, top_p,
            self.seed_buf, self.offset_buf,
            apply_penalty=apply_penalty,
            rep_penalty=self.rep_penalty_buf,
            seen_tokens=self.seen_tokens_buf,
            min_p=self.min_p_buf,
            history=self.history,
            order=self.order,
        )
        self.offset_buf += 1
        codes = self._broadcast_tokens(codes)
        if self.history is not None:
            # the TP-agreed token, so every rank's history stays the same
            advance_history(self.history, codes)
        if apply_penalty and self.seen_tokens_buf is not None:
            self.applied_penalty_in_graph = True
            # Record the (broadcast, TP-agreed) token in the seen-token buffer so
            # the next step penalises it. ``scatter_`` with a scalar value is
            # CUDA-graph capturable; advanced-index assignment
            # (``buf[rows, codes] = True``) is not — it trips "operation not
            # permitted when stream is capturing".
            self.seen_tokens_buf.scatter_(1, codes.unsqueeze(1), True)
        return codes

    @torch.compiler.disable
    def sync_seen_token_masks(
        self, seen_masks: "Iterable[SeenTokenMask]",
    ) -> None:
        """Copy the in-graph seen-token rows back into canonical ``SeenTokenMask``s.

        Called eagerly after graph replay (the captured ``sample`` scattered the
        newly sampled token into ``seen_tokens_buf``). ``seen_masks`` are in
        request order; padding rows beyond ``len(seen_masks)`` are ignored, and
        not-yet-sized masks (``_seen_token_mask is None``) are skipped.
        """
        if self.seen_tokens_buf is None:
            return
        # One multi-tensor-apply instead of bs separate copy_ launches. The
        # masks are separately-owned tensors, so a single index_copy_ would
        # need them to be views into one buffer; _foreach_copy_ fuses the
        # launches without changing that ownership.
        dsts = []
        srcs = []
        for i, m in enumerate(seen_masks):
            mask = m._seen_token_mask
            if mask is not None:
                dsts.append(mask)
                srcs.append(self.seen_tokens_buf[i])
        if dsts:
            torch._foreach_copy_(dsts, srcs)


@dataclass
class Buffer:
    """Three-tier storage for one per-request scalar sampling parameter.

    - ``buf``     ``[max_bs]``   per-step tensor, sliced to ``padded_bs`` and read
      by ``CudaGraphableSampler`` (its address must stay stable across replays).
    - ``master``  ``[capacity]`` slot-indexed cache, one row per active request.
    """
    # ``[cg_slots, max_bs]``: one per-step row per double-buffer slot, so a
    # pre-plan gathering into one slot doesn't clobber the replay reading another
    buf: torch.Tensor
    master: torch.Tensor
    default: float
    dtype: torch.dtype

    @classmethod
    def allocate(
        cls, max_bs: int, capacity: int, device: torch.device,
        dtype: torch.dtype, default: float, cg_slots: int = 1,
    ) -> "Buffer":
        return cls(
            buf=torch.full((cg_slots, max_bs), default, dtype=dtype, device=device),
            master=torch.full((capacity,), default, dtype=dtype, device=device),
            default=default,
            dtype=dtype,
        )

    def write_master_row(self, slot: int, value) -> None:
        # Staging through a shared pinned row +
        # non_blocking H2D raced: the copy runs behind step N-1's kernels, so a
        # second rid registered in the same step overwrote the row first.
        self.master[slot:slot + 1].fill_(value)

    def grow_master(self, new_capacity: int) -> None:
        new = torch.full(
            (new_capacity,), self.default, dtype=self.dtype, device=self.master.device,
        )
        new[: self.master.shape[0]].copy_(self.master)
        self.master = new

    def slot_view(self, cg_slot: int, bs: int) -> torch.Tensor:
        """This slot's per-step row (clamped: a single-buffered buffer — the
        RNG offset, gathered inline and used serialized — ignores ``cg_slot``)."""
        return self.buf[cg_slot if self.buf.shape[0] > 1 else 0, :bs]

    def gather(self, idx_view: torch.Tensor, padded_bs: int, cg_slot: int) -> None:
        torch.index_select(self.master, 0, idx_view, out=self.slot_view(cg_slot, padded_bs))

    def scatter(self, idx_view: torch.Tensor, real_bs: int, cg_slot: int) -> None:
        """Persist the per-step rows back to their slots (GPU-only, no CPU).

        Inverse of ``gather`` — for buffers whose per-step value is advanced in
        graph (the RNG offset). REAL rows only: padding rows all gather from
        slot 0 and get advanced too, so scattering them would clobber slot 0.
        """
        self.master.index_copy_(0, idx_view[:real_bs], self.slot_view(cg_slot, real_bs))


@dataclass
class HostBuffer:
    """Storage for one per-request scalar sampling parameter that only changes
    on a config update (temperature, top-k, top-p, seed, penalty).

    - ``buf``    ``[cg_slots, max_bs]`` per-step device tensor the graph reads
      (its address must stay stable across replays).
    - ``master`` ``[capacity]`` slot-indexed HOST array, one row per request.

    The master is on the host because the gather runs in the pre-plan beside a
    live graph, so it must be one H2D copy with no device ops (``mstar.utils.h2d``).
    """
    buf: torch.Tensor
    master: np.ndarray
    default: float
    dtype: torch.dtype
    stager: PinnedStager

    @classmethod
    def allocate(
        cls, max_bs: int, capacity: int, device: torch.device,
        dtype: torch.dtype, default: float, cg_slots: int = 1,
    ) -> "HostBuffer":
        np_dtype = torch.empty((), dtype=dtype).numpy().dtype
        return cls(
            buf=torch.full((cg_slots, max_bs), default, dtype=dtype, device=device),
            master=np.full(capacity, default, dtype=np_dtype),
            default=default,
            dtype=dtype,
            stager=PinnedStager(dtype, numel=max_bs),
        )

    def write_master_row(self, slot: int, value) -> None:
        self.master[slot] = value

    def grow_master(self, new_capacity: int) -> None:
        new = np.full(new_capacity, self.default, dtype=self.master.dtype)
        new[: self.master.shape[0]] = self.master
        self.master = new

    def slot_view(self, cg_slot: int, bs: int) -> torch.Tensor:
        """This slot's per-step row."""
        return self.buf[cg_slot if self.buf.shape[0] > 1 else 0, :bs]

    def gather(self, rows: np.ndarray, padded_bs: int, cg_slot: int) -> None:
        """Rows ``rows`` of the master into ``cg_slot``'s per-step buffer."""
        self.stager.copy_(self.slot_view(cg_slot, padded_bs), self.master[rows])


@dataclass
class MaskBuffer:
    """Three-tier storage for the per-request seen-token mask ``[*, V]`` (bool).

    Mirrors ``Buffer`` but 2-D and sourced from on-device ``SeenTokenMask``
    tensors, so the master-row write is a GPU->GPU copy.
    """
    buf: torch.Tensor       # [cg_slots, max_bs, V] bool
    master: torch.Tensor    # [capacity, V] bool
    vocab_size: int

    @classmethod
    def allocate(
        cls, max_bs: int, capacity: int, vocab_size: int, device: torch.device,
        cg_slots: int = 1,
    ) -> "MaskBuffer":
        return cls(
            buf=torch.zeros(cg_slots, max_bs, vocab_size, dtype=torch.bool, device=device),
            master=torch.zeros(capacity, vocab_size, dtype=torch.bool, device=device),
            vocab_size=vocab_size,
        )

    def write_master_row(self, slot: int, mask: torch.Tensor) -> None:
        # ``mask`` is the [V] bool tensor owned by a SeenTokenMask (on device).
        self.master[slot].copy_(mask)

    def clear_master_row(self, slot: int) -> None:
        self.master[slot].zero_()

    def grow_master(self, new_capacity: int) -> None:
        new = torch.zeros(
            new_capacity, self.vocab_size, dtype=torch.bool, device=self.master.device,
        )
        new[: self.master.shape[0]].copy_(self.master)
        self.master = new

    def slot_view(self, cg_slot: int, bs: int) -> torch.Tensor:
        """This slot's per-step rows (clamped: single-buffered — the seen-token
        mask, gathered inline and used serialized — ignores ``cg_slot``)."""
        return self.buf[cg_slot if self.buf.shape[0] > 1 else 0, :bs]

    def gather(self, idx_view: torch.Tensor, padded_bs: int, cg_slot: int) -> None:
        torch.index_select(self.master, 0, idx_view, out=self.slot_view(cg_slot, padded_bs))


@dataclass
class SamplerBuffers:
    """Pre-allocated static buffers for graph-safe sampling.

    Each per-request scalar parameter (temperature, top_k, top_p, seed,
    repetition_penalty) is a ``Buffer`` owning a per-step slice and a slot-indexed
    master cache. The RNG ``offset`` is also a ``Buffer``
    but round-trips on the GPU with no CPU middleman: gathered from its slot
    master before sampling, advanced in graph (``offset_buf += 1`` per sample),
    then scattered back to the master after replay — so a request's RNG stream
    follows its slot (batch-position invariant) and reproduces under a fixed
    seed. The optional ``seen_tokens`` ``MaskBuffer`` (allocated only when
    ``vocab_size`` is given) carries the per-request repetition-penalty mask.

    ``gather_static`` / ``gather_dynamic`` build a pinned slot-index tensor, async-copy it
    to GPU, and ``index_select``s each master into the per-step buffers — one
    cheap gather per buffer instead of the old per-element item-assignments.
    """
    max_batch_size: int
    temperature: HostBuffer
    top_k: HostBuffer
    top_p: HostBuffer
    seed: HostBuffer
    rep_penalty: HostBuffer
    # Per-request RNG offset: gathered by slot, advanced in graph, scattered
    # back after replay (all on GPU). Not a config value, so it's kept out of
    # ``_scalar_buffers`` (never written from a SamplingConfig) and reset to 0
    # on register instead.
    offset: Buffer

    # TP communicator for the submodule that owns these buffers. Passed
    # through ``slice_for_bs`` into every per-step ``CudaGraphableSampler``
    # so its ``_broadcast_tokens`` aligns the sampled token across ranks.
    # Without this, ``sample`` would build a
    # sampler with ``tp_group=None``, the broadcast would silently no-op,
    # and TP ranks would drift apart on the first tied-logit sample —
    # garbled audio for Talker, premature EOS for Thinker. Defaults to
    # ``None`` for non-TP submodules (trivial broadcast is a cheap no-op).
    tp_group: "CommGroup | None" = None  # noqa: F821
    # Per-request seen-token mask buffer for the repetition penalty. Present
    # only for submodules that opt in by declaring a vocab size (e.g. the
    # Qwen3-Omni Talker). ``None`` => the CUDA-graph path applies no penalty.
    seen_tokens: "MaskBuffer | None" = None
    # Per-request min-p; allocated only for submodules whose spec enables it,
    # so every other node's captured sampler is unchanged.
    min_p: "HostBuffer | None" = None
    # Generation-aware processors, each allocated only when the spec enables
    # it. The per-request settings are static rows like the ones above; the
    # per-step state (count, recent tokens) is the resource's
    # ``GenerationHistory``, gathered inline like the RNG offset.
    history: "GenerationHistory | None" = None
    window: "HostBuffer | None" = None
    window_penalty: "HostBuffer | None" = None
    min_tokens: "HostBuffer | None" = None
    stop_ids: torch.Tensor | None = None
    top_p_first: "HostBuffer | None" = None
    top_p_min_keep: "HostBuffer | None" = None
    # Master cache capacity (grown by doubling when more requests are
    # concurrently registered than the per-step buffer holds).
    _master_capacity: int = field(default=0, repr=False)
    # Number of double-buffer slots (per-step buffers carry this leading dim).
    cg_slots: int = 1
    # Per-step slot-index staging, per cg slot. ``_slot_idx_cpu`` is pinned so
    # the H2D copy can be issued non-blocking; ``_slot_idx_gpu`` is the
    # device-side index tensor that ``index_select`` reads from.
    _slot_idx_cpu: torch.Tensor = field(default=None, repr=False)
    _slot_idx_gpu: torch.Tensor = field(default=None, repr=False)
    # Zero-copy numpy view of ``_slot_idx_cpu``, so staging is one slice write
    # rather than a tensor __setitem__ per row. Built lazily; the tensor is
    # allocated once and never rebound, so the view stays valid.
    _slot_idx_np: Any = field(default=None, repr=False)
    # Real (unpadded) batch size of the last gather per cg slot — the
    # scatter-back writes only these rows (padding rows all map to slot 0).
    _last_real_bs: list[int] = field(default_factory=list, repr=False)
    # cg slot -> the (rids, padded_bs) its pinned index row currently holds, so
    # gather_dynamic can re-issue the H2D without redoing the O(bs) CPU fill
    _staged_rids: dict[int, tuple[tuple[str, ...], int]] = field(
        default_factory=dict, repr=False
    )
    # Slot bookkeeping (CPU-only).
    _rid_to_slot: dict[str, int] = field(default_factory=dict, repr=False)
    _free_slots: list[int] = field(default_factory=list, repr=False)
    # Slots awaiting init, consumed by the next gather so the rows are written
    # on the gathering thread rather than racing it from the main one.
    _pending_init: set[int] = field(default_factory=set, repr=False)
    # Slots whose RNG offset must be zeroed before it is next gathered: a
    # device write, so the plan thread queues it and ``gather_dynamic`` (GPU
    # thread, default stream) drains it.
    _pending_offset_reset: set[int] = field(default_factory=set, repr=False)
    _offset_reset_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Last-known config per rid — change-detect for ``update_request_config``
    # so steady-state per-step calls do zero GPU work (for the scalar rows).
    _cached_config: dict[str, SamplingConfig] = field(default_factory=dict, repr=False)
    # Bumped on every scalar master-row write, so ``gather_static`` can tell
    # a config change apart from a batch it already gathered.
    _config_version: int = field(default=0, repr=False)
    # cg slot -> (rids, padded_bs, _config_version) of the last
    # ``gather_static``, so a steady batch can skip it.
    _static_key: dict[int, tuple[tuple[str, ...], int, int]] = field(
        default_factory=dict, repr=False
    )

    @property
    def tracks_seen_tokens(self) -> bool:
        return self.seen_tokens is not None

    def _scalar_buffers(self) -> list[HostBuffer]:
        bufs = [self.temperature, self.top_k, self.top_p, self.seed, self.rep_penalty]
        for opt in (
            self.min_p, self.window, self.window_penalty, self.min_tokens,
            self.top_p_first, self.top_p_min_keep,
        ):
            if opt is not None:
                bufs.append(opt)
        return bufs

    @classmethod
    def allocate(
        cls,
        max_batch_size: int,
        device: torch.device,
        tp_group: "CommGroup | None" = None,  # noqa: F821
        vocab_size: int | None = None,
        cg_slots: int = 1,
        enable_min_p: bool = False,
        history: GenerationHistory | None = None,
        stop_ids: torch.Tensor | None = None,
        enable_top_p_first: bool = False,
    ) -> "SamplerBuffers":
        """Allocate sampling buffers for ``max_batch_size``.

        ``vocab_size`` (when not None) enables the seen-token mask buffer for the
        repetition penalty. ``history`` (with ``stop_ids`` and
        ``enable_top_p_first``) enables the generation-aware processors; its
        per-step buffers are sized here too. The master rows default to a ``SamplingConfig()`` row
        (temp=1, top_k=0, top_p=1, rep_penalty=1) — what an unregistered slot
        would surface if accidentally indexed. ``cg_slots`` double-buffers the
        per-step tensors so the sampler can pre-plan.
        """
        pinned = torch.cuda.is_available() and device.type == "cuda"
        cap = max_batch_size

        def mk(dtype: torch.dtype, default: float) -> HostBuffer:
            return HostBuffer.allocate(
                max_batch_size, cap, device, dtype, default, cg_slots
            )

        # seen token mask is not double-buffered, as it depends on the GPU
        # value of the previous step. Same goes for offset
        seen_tokens = (
            MaskBuffer.allocate(max_batch_size, cap, vocab_size, device, cg_slots=1)
            if vocab_size is not None else None
        )
        windowed = history is not None and history.window > 0
        if history is not None:
            history.allocate_step_buffers(max_batch_size)
        return cls(
            max_batch_size=max_batch_size,
            temperature=mk(torch.float32, 1.0),
            top_k=mk(torch.int32, 0),
            top_p=mk(torch.float32, 1.0),
            seed=mk(torch.long, 0),
            rep_penalty=mk(torch.float32, 1.0),
            offset=Buffer.allocate(max_batch_size, cap, device, torch.long, 0, 1),
            tp_group=tp_group,
            seen_tokens=seen_tokens,
            min_p=mk(torch.float32, 0.0) if enable_min_p else None,
            history=history,
            window=mk(torch.int32, 0) if windowed else None,
            window_penalty=mk(torch.float32, 1.0) if windowed else None,
            min_tokens=mk(torch.int32, 0) if stop_ids is not None else None,
            stop_ids=stop_ids,
            top_p_first=mk(torch.bool, False) if enable_top_p_first else None,
            top_p_min_keep=mk(torch.int32, 1) if enable_top_p_first else None,
            _master_capacity=cap,
            cg_slots=cg_slots,
            _slot_idx_cpu=torch.zeros(cg_slots, max_batch_size, dtype=torch.long, pin_memory=pinned),
            _slot_idx_gpu=torch.zeros(cg_slots, max_batch_size, dtype=torch.long, device=device),
            _last_real_bs=[0] * cg_slots,
            _free_slots=list(range(cap)),
        )

    def slice_for_bs(self, bs: int, cg_slot: int = 0) -> dict[str, Any]:
        """Return bs-sized views into one slot's buffers (zero-copy slices) plus
        the owning submodule's ``tp_group`` so the constructed sampler
        broadcasts across TP ranks."""
        # slot_view clamps single-buffered buffers (offset, seen mask) to slot 0
        return {
            "temperature_buf": self.temperature.slot_view(cg_slot, bs),
            "top_k_buf": self.top_k.slot_view(cg_slot, bs),
            "top_p_buf": self.top_p.slot_view(cg_slot, bs),
            "seed_buf": self.seed.slot_view(cg_slot, bs),
            "offset_buf": self.offset.slot_view(cg_slot, bs),
            "rep_penalty_buf": self.rep_penalty.slot_view(cg_slot, bs),
            "seen_tokens_buf": self.seen_tokens.slot_view(cg_slot, bs) if self.seen_tokens is not None else None,
            "min_p_buf": self.min_p.slot_view(cg_slot, bs) if self.min_p is not None else None,
            "history": self._history_rows(bs, cg_slot) if self.history is not None else None,
            "order": (
                FilterOrder(self.top_p_first.slot_view(cg_slot, bs), self.top_p_min_keep.slot_view(cg_slot, bs))
                if self.top_p_first is not None else None
            ),
            "tp_group": self.tp_group,
        }

    def _history_rows(self, bs: int, cg_slot: int) -> HistoryRows:
        def view(buf: "HostBuffer | None") -> torch.Tensor | None:
            return buf.slot_view(cg_slot, bs) if buf is not None else None

        return HistoryRows(
            count=self.history.step_count[:bs],
            tokens=self.history.step_tokens[:bs] if self.window is not None else None,
            window=view(self.window),
            window_penalty=view(self.window_penalty),
            min_tokens=view(self.min_tokens),
            stop_ids=self.stop_ids,
        )

    # ------------------------------------------------------------------
    # Master-cache lifecycle: register / unregister / update per request
    # ------------------------------------------------------------------

    def _write_master_row(self, slot: int, cfg: SamplingConfig) -> None:
        """Push one config row into each scalar master buffer.

        Host writes only; only runs on register or actual config change
        (change-detection lives in ``update_request_config``). The seen-token
        mask is NOT written here (it changes every step — see
        ``update_request_config``).
        """
        if cfg.temperature > 0:
            t = float(cfg.temperature)
            k = int(cfg.top_k)
            p = float(cfg.top_p) if cfg.top_p else 1.0
        else:
            # Greedy: temperature 0 -> the fused prep kernel emits a one-hot at
            # argmax (first index on ties), which from_probs returns for any
            # seed. See ``sample_cuda_graphable_gpu``.
            t, k, p = 0.0, 0, 1.0
        self.temperature.write_master_row(slot, t)
        self.top_k.write_master_row(slot, k)
        self.top_p.write_master_row(slot, p)
        self.seed.write_master_row(slot, cfg.seed)
        self.rep_penalty.write_master_row(slot, float(cfg.repetition_penalty))
        if self.min_p is not None:
            self.min_p.write_master_row(slot, float(cfg.min_p) if cfg.temperature > 0 else 0.0)
        if self.window is not None:
            # a windowed request's penalty is count-based, so the mask is inert for it
            self.rep_penalty.write_master_row(slot, float(cfg.presence_penalty))
            self.window.write_master_row(slot, int(cfg.repetition_window))
            self.window_penalty.write_master_row(slot, float(cfg.window_penalty))
        if self.min_tokens is not None:
            self.min_tokens.write_master_row(slot, int(cfg.min_tokens))
        if self.top_p_first is not None:
            self.top_p_first.write_master_row(slot, bool(cfg.top_p_first))
            self.top_p_min_keep.write_master_row(slot, int(cfg.top_p_min_keep))
        self._config_version += 1

    def _grow_master(self, new_capacity: int) -> None:
        """Double-and-copy the master buffers up to at least ``new_capacity``.

        Triggered when concurrently-registered requests exceed the current
        master capacity. Per-step buffers (sized to the cuda-graph max_bs) are
        NOT resized — the gather only reads ``padded_bs`` rows from master.
        """
        for buf in self._scalar_buffers():
            buf.grow_master(new_capacity)
        self.offset.grow_master(new_capacity)
        if self.seen_tokens is not None:
            self.seen_tokens.grow_master(new_capacity)
        self._free_slots.extend(range(self._master_capacity, new_capacity))
        self._master_capacity = new_capacity

    def register_request(
        self, rid: str, sampling_config: SamplingConfig | None = None,
    ) -> None:
        """Allocate a slot for ``rid`` and seed its master row."""
        if rid in self._rid_to_slot:
            # Re-registration: just refresh the config in place.
            if sampling_config is not None:
                self.update_request_config(rid, sampling_config)
            return
        if not self._free_slots:
            self._grow_master(self._master_capacity * 2)
        slot = self._free_slots.pop()
        self._rid_to_slot[rid] = slot
        # CPU-only; every GPU write for this slot is deferred to the next gather.
        self._pending_init.add(slot)
        self._cached_config[rid] = (
            sampling_config if sampling_config is not None else SamplingConfig()
        )

    def unregister_request(self, rid: str) -> None:
        """Free the slot owned by ``rid`` (no GPU writes)."""
        slot = self._rid_to_slot.pop(rid, None)
        if slot is None:
            return
        self._cached_config.pop(rid, None)
        self._pending_init.discard(slot)
        self._free_slots.append(slot)

    def update_request_config(
        self, rid: str, sampling_config: SamplingConfig,
    ) -> None:
        """Update the master row for ``rid`` only when its config changed.

        AR engine calls this every step (mirroring ``Sampler.set_config``).
        Steady-state requests have identical configs across steps, so the
        change-check skips the H2D path entirely. The seen-token mask is staged
        separately (see ``stage_seen_token_masks``) because it grows every step.
        """
        slot = self._rid_to_slot.get(rid)
        if slot is None:
            # Request not yet registered for this submodule (e.g. ar_engine
            # may invoke set_config for a node that doesn't own a runner /
            # SamplerBuffers). Silently no-op.
            return
        prev = self._cached_config.get(rid)
        if prev == sampling_config:
            return
        self._cached_config[rid] = sampling_config
        # A slot awaiting init writes this row at gather time instead.
        if slot not in self._pending_init:
            self._write_master_row(slot, sampling_config)

    def stage_seen_token_masks(
        self, request_ids: list[str], seen_masks: "Iterable[SeenTokenMask]",
    ) -> None:
        """Copy each request's current seen-token mask into its master row.

        Called every step (before ``gather_dynamic``) for submodules that
        sample with a penalty in-graph, so the gathered per-step buffer reflects
        the live prompt + generated tokens. No-op when seen-token tracking is off.
        """
        if self.seen_tokens is None:
            return
        # Batched for the same reason as sync_seen_token_masks: bs launches
        # become one. _init_slot still has to run per slot (it is CPU-side
        # bookkeeping plus a master-row write) before anything is staged.
        dsts = []
        srcs = []
        for rid, m in zip(request_ids, seen_masks, strict=False):
            slot = self._rid_to_slot.get(rid)
            if slot is None:
                continue
            # Runs before the gather, so init here or the clear would wipe it.
            if slot in self._pending_init:
                self._init_slot(slot, rid)
            mask = m._seen_token_mask
            if mask is not None:
                dsts.append(self.seen_tokens.master[slot])
                srcs.append(mask)
        if dsts:
            torch._foreach_copy_(dsts, srcs)

    def _init_slot(self, slot: int, rid: str) -> None:
        """Init for a newly registered slot, on the gathering thread so rows are
        written before the gather reads them. No device work (pre-plan): the
        offset reset is queued for ``gather_dynamic``."""
        with self._offset_reset_lock:
            self._pending_offset_reset.add(slot)
        self._write_master_row(slot, self._cached_config[rid])
        # mask row not cleared: always staged fresh before gather, and clearing
        # here would race the default-stream stage from the plan stream
        self._pending_init.discard(slot)

    # ------------------------------------------------------------------
    # Per-step gather: pinned-H2D slot-index → index_select into per-step bufs
    # ------------------------------------------------------------------

    def _stage_slot_idx(
        self, request_ids: list[str], padded_bs: int, cg_slot: int,
    ) -> None:
        """Write this batch's master-slot indices into ``cg_slot``'s pinned row.

        Padding slots (``i >= len(request_ids)``) reuse slot 0's row — the
        captured graph forwards them through the same kernels as real slots,
        but their outputs are discarded by the runner's dummy-rid remap.
        Unregistered rids fall back to slot 0 (the config defaults).
        """
        assert padded_bs <= self.max_batch_size, (
            f"padded_bs={padded_bs} exceeds SamplerBuffers.max_batch_size="
            f"{self.max_batch_size}"
        )
        # Batched for the same reason as stage_seen_token_masks: _init_slot has
        # to run per slot, but the staging writes become one. Elementwise, this
        # was ~46us at bs=16 — all Python, and the plan thread pays it inline.
        if self._slot_idx_np is None:
            self._slot_idx_np = self._slot_idx_cpu.numpy()
        get = self._rid_to_slot.get
        pending = self._pending_init
        rows = []
        for rid in request_ids:
            slot = get(rid)
            if slot is None:
                slot = 0
            elif slot in pending:
                self._init_slot(slot, rid)
            rows.append(slot)
        if len(rows) < padded_bs:
            rows.extend([0] * (padded_bs - len(rows)))
        self._slot_idx_np[cg_slot, :padded_bs] = rows
        self._staged_rids[cg_slot] = (tuple(request_ids), padded_bs)

    def _upload_slot_idx(self, padded_bs: int, cg_slot: int) -> torch.Tensor:
        """H2D the staged row on the CURRENT stream, and hand back its view."""
        idx_view = self._slot_idx_gpu[cg_slot, :padded_bs]
        idx_view.copy_(self._slot_idx_cpu[cg_slot, :padded_bs], non_blocking=True)
        return idx_view

    def gather_static(
        self, request_ids: list[str], padded_bs: int, cg_slot: int,
    ) -> None:
        """Gather the per-request scalar config (temp/top_k/top_p/seed/penalty)
        into ``cg_slot``. Safe to pre-plan: these change only on a config
        update, never step to step.

        Skipped when ``cg_slot`` last gathered this same batch and no master row
        changed since. Never skipped with a slot awaiting init, which also
        catches a request id re-registered onto a new slot."""
        rids = tuple(request_ids)
        if (
            not self._pending_init
            and self._static_key.get(cg_slot)
            == (rids, padded_bs, self._config_version)
        ):
            self._last_real_bs[cg_slot] = len(request_ids)
            return
        self._stage_slot_idx(request_ids, padded_bs, cg_slot)
        # H2D copies only (pre-plan, see HostBuffer); gather_dynamic uploads
        # the device-side index row
        rows = self._slot_idx_np[cg_slot, :padded_bs]
        for buf in self._scalar_buffers():
            buf.gather(rows, padded_bs, cg_slot)
        self._last_real_bs[cg_slot] = len(request_ids)
        # read after staging: an init it ran bumped the version
        self._static_key[cg_slot] = (rids, padded_bs, self._config_version)

    def gather_dynamic(
        self, request_ids: list[str], padded_bs: int, cg_slot: int,
        gather_seen_tokens: bool = True,
    ) -> None:
        """Gather the per-step state — RNG offset and seen-token mask — into
        ``cg_slot``. Must run inline (default stream) AFTER the previous step's
        commit scattered the offset / synced the mask; pre-planning it reads
        stale state across the plan/default stream boundary.

        Only the H2D is re-issued (the pinned row ``gather_static`` staged for
        this slot is stream-agnostic, and this step's batch is the one it was
        staged for) — restaging is the fallback for a caller that reached here
        without a matching ``gather_static``.

        The seen-token mask is large ([bs, V] bool); only gathered when the
        caller's graph actually applies the penalty in-graph (the Talker)."""
        if self._staged_rids.get(cg_slot) != (tuple(request_ids), padded_bs):
            self._stage_slot_idx(request_ids, padded_bs, cg_slot)
        with self._offset_reset_lock:
            resets, self._pending_offset_reset = self._pending_offset_reset, set()
        for slot in resets:
            self.offset.master[slot:slot + 1].zero_()
        idx_view = self._upload_slot_idx(padded_bs, cg_slot)
        self.offset.gather(idx_view, padded_bs, cg_slot)
        if self.seen_tokens is not None and gather_seen_tokens:
            self.seen_tokens.gather(idx_view, padded_bs, cg_slot)

    def sampler_for(self, padded_bs: int, cg_slot: int) -> "CudaGraphableSampler":
        """A sampler bound to ``cg_slot``'s per-step buffer views (zero-copy).
        Valid regardless of when the buffers are (re)gathered into."""
        return CudaGraphableSampler(**self.slice_for_bs(padded_bs, cg_slot))

    def scatter_offset(self, cg_slot: int = 0) -> None:
        """Persist the (in-graph advanced) per-step offsets back to their slot
        masters. Call once AFTER the graph replay for the gather on ``cg_slot``;
        GPU-only, real rows only (padding rows all map to slot 0)."""
        self.offset.scatter(
            self._slot_idx_gpu[cg_slot], self._last_real_bs[cg_slot], cg_slot
        )
