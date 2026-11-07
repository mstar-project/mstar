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
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np
import torch
import triton
import triton.language as tl

from mstar.utils.device import DeviceCapability
from mstar.utils.h2d import H2DMirror, PinnedStager, to_device_nonblocking
from mstar.utils.knobs import sampler_slots

if TYPE_CHECKING:
    from mstar.engine.resources.sampler.resident import SamplerCells

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
    hist_ptr, hist_stride, window_ptr, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,  # see _history_adjust
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
            vals, offs, row, temp, hist_ptr, hist_stride, window_ptr, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
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
            vals, offs, row, temp, hist_ptr, hist_stride, window_ptr, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
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
            vals, offs, row, temp, hist_ptr, hist_stride, window_ptr, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
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
    vals, offs, row, temp, hist_ptr, hist_stride, window_ptr, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
    WINDOW: tl.constexpr, NUM_STOP: tl.constexpr,
):
    """The ``HistoryRows`` processors on one tile of a row's logits: the windowed
    frequency penalty (``penalty ** n``, n the token's count among the row's
    ring of recent tokens, sign-aware like the presence penalty), and the
    stop-id floor for greedy rows (sampled rows are floored after the filter,
    in ``apply_min_tokens_floor``). Both compile out when off."""
    if WINDOW > 0:
        pen = tl.load(wpen_ptr + row)
        count = tl.load(count_ptr + row)
        window = tl.minimum(tl.load(window_ptr + row), count)
        factor = tl.zeros_like(vals) + 1.0
        for w in tl.static_range(WINDOW):
            tok = tl.load(hist_ptr + row * hist_stride + w)
            age = (count - 1 - w + WINDOW) % WINDOW
            factor = tl.where((offs == tok) & (age < window), factor * pen, factor)
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
    hist_ptr, hist_stride, window_ptr, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
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
            vals, offs, row, temp, hist_ptr, hist_stride, window_ptr, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
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
    hist_ptr, hist_stride, window_ptr, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
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
            vals, offs, row, temp, hist_ptr, hist_stride, window_ptr, wpen_ptr, count_ptr, min_tok_ptr, stop_ptr,
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
        history.window if window else dummy,
        history.window_penalty if window else dummy,
        history.count if window or floor else dummy,
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
    tokens: torch.Tensor | None = None          # [B, W] fixed-capacity ring, -1 where empty
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


@triton.jit
def _kth_largest_key(key, k, HI: tl.constexpr, ITERS: tl.constexpr):
    """The largest ``t`` in ``[0, HI]`` with at least ``k`` entries of ``key`` >= t."""
    lo = 0
    hi = HI
    for _ in range(ITERS):
        mid = lo + (hi - lo + 1) // 2
        ok = tl.sum((key >= mid).to(tl.int32), axis=0) >= k
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid - 1)
    return lo


@triton.jit
def _fused_sample_kernel(
    logits_ptr, temperature_ptr, top_k_ptr, top_p_ptr, seed_ptr, offset_ptr,
    penalty_ptr, seen_ptr, out_ptr,
    V, stride_l, seen_stride_b, seen_stride_v,
    APPLY_PENALTY: tl.constexpr, BLOCK: tl.constexpr,
):
    """One row's token, from its logits, in registers: the repetition penalty,
    temperature softmax, top-k then top-p over the renormalised top-k
    (FlashInfer's order), and an inverse-CDF draw from Philox at the row's
    own (seed, offset), so a request draws the same token at any batch
    position. Temperature 0 takes the argmax.

    The cuts rank tokens by their probability's top 16 bits (bf16
    precision, truncated): a search over 2^14 keys instead of 2^31, so
    tokens within one bf16 step of the k-th probability are kept with it."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    valid = offs < V
    x = tl.load(logits_ptr + row * stride_l + offs, mask=valid, other=-float("inf")).to(tl.float32)
    if APPLY_PENALTY:
        seen = tl.load(
            seen_ptr + row * seen_stride_b + offs * seen_stride_v, mask=valid, other=0,
        ).to(tl.int1)
        x = _penalize(x, seen, tl.load(penalty_ptr + row))
    temp = tl.load(temperature_ptr + row)
    greedy = temp == 0
    x_max, arg = tl.max(x, axis=0, return_indices=True)
    inv_temp = tl.where(greedy, 1.0, 1.0 / tl.maximum(temp, 1e-30))
    # unnormalised: the largest is 1, so every key is at most 1.0's, 0x3F80
    e = tl.where(valid, tl.exp((x - x_max) * inv_temp), 0.0)
    key = e.to(tl.int32, bitcast=True) >> 16
    k = tl.load(top_k_ptr + row)
    t = 0
    if k > 0:
        if k < V:  # a branch, not tl.where: skip the search when nothing is cut
            t = _kth_largest_key(key, k, 0x3F80, 14)
    top_p = tl.load(top_p_ptr + row)
    if top_p < 1.0:
        # top-p keeps token i iff the mass strictly above it is < p * mass. That
        # mass falls as i's key rises: search for the smallest key t' with
        # mass(> t') < p * mass
        target = top_p * tl.sum(tl.where(key >= t, e, 0.0), axis=0)
        lo = t
        hi = 0x3F80
        for _ in range(14):
            mid = lo + (hi - lo) // 2
            ok = tl.sum(tl.where(key > mid, e, 0.0), axis=0) < target
            hi = tl.where(ok, mid, hi)
            lo = tl.where(ok, lo, mid + 1)
        t = lo
    q = tl.where(key >= t, e, 0.0)
    # u in (0, mass]: the first token whose running mass reaches u, never one
    # with no mass
    u = (1.0 - tl.rand(tl.load(seed_ptr + row), tl.load(offset_ptr + row))) * tl.sum(q, axis=0)
    drawn = tl.sum((tl.cumsum(q, axis=0) < u).to(tl.int32), axis=0)
    drawn = tl.minimum(drawn, tl.max(tl.where(q > 0, offs, 0), axis=0))
    tl.store(out_ptr + row, tl.where(greedy, arg, drawn).to(tl.int64))


# Vocabularies up to this size draw in one ``_fused_sample_kernel`` launch, one
# program per row holding the whole row; larger ones through FlashInfer.
FUSED_SAMPLE_MAX_VOCAB = 4096


def fused_sample(
    logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor,
    seed: torch.Tensor, offset: torch.Tensor,
    rep_penalty: torch.Tensor | None = None, seen_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[B]`` int64 tokens drawn from ``logits [B, V]`` (V up to
    ``FUSED_SAMPLE_MAX_VOCAB``) in one launch: the work of
    ``fused_temperature_softmax`` plus FlashInfer's top-k/top-p draw, which
    is a handful of launches of a few microseconds each at a small vocab.
    Graph-safe."""
    B, V = logits.shape
    out = torch.empty(B, dtype=torch.int64, device=logits.device)
    penalty = rep_penalty is not None and seen_tokens is not None
    _fused_sample_kernel[(B,)](
        logits, temperature, top_k, top_p, seed, offset,
        rep_penalty if penalty else temperature, seen_tokens if penalty else logits, out,
        V, logits.stride(0),
        seen_tokens.stride(0) if penalty else 0, seen_tokens.stride(1) if penalty else 0,
        APPLY_PENALTY=penalty, BLOCK=triton.next_power_of_2(V), num_warps=4 if V <= 2048 else 8,
    )
    return out


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
    first = flashinfer.sampling.top_k_renorm_probs(probs, first_k)
    # FlashInfer's top-p expects p in (0, 1); p = 1 removes nothing
    n = torch.where(top_p >= 1.0, _kept(first), _kept(flashinfer.sampling.top_p_renorm_probs(first, top_p)))
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
    """Keep the node's full history ring, even while its processors are inert."""
    if rows.tokens is not None and rows.tokens.shape[1]:
        pos = (rows.count % rows.tokens.shape[1]).long()[:, None]
        rows.tokens.scatter_(1, pos, tokens.reshape(-1, 1).to(rows.tokens.dtype))
    rows.count.add_(1)


def token_logprobs(logits: torch.Tensor, tokens: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Raw log-probs of ``tokens`` under ``softmax(logits / T)``, before the
    penalty and filters, with T = 1 for greedy rows; graph-safe."""
    if logits.is_cuda:
        return _token_logprobs_cuda(logits, tokens, temperature)
    t = torch.where(temperature > 0, temperature, 1.0).float().unsqueeze(1)
    # fp32 by promotion against ``t``, without a second [B, V] copy
    scaled = logits / t
    norm = scaled.logsumexp(dim=-1, keepdim=True)
    return (scaled.gather(1, tokens.long().unsqueeze(1)) - norm).squeeze(1)


def processed_logprobs(
    probs: torch.Tensor, tokens: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor,
    history: "HistoryRows | None" = None, order: "FilterOrder | None" = None,
) -> torch.Tensor:
    """Log-probs of ``tokens`` under the distribution they were drawn from:
    ``probs`` (after the penalty, temperature and min-p) filtered and floored
    as ``_filter_floor_and_draw`` does. 0 for greedy rows; graph-safe."""
    import flashinfer

    tokens = tokens.long().view(-1, 1)
    floor = history is not None and history.stop_ids is not None
    if order is not None or floor:
        filtered = filter_top_k_top_p(probs, top_k, top_p, order)
        if floor:
            filtered = apply_min_tokens_floor(filtered, history)
        picked = filtered.gather(1, tokens)
    else:
        # FlashInfer's own order, as the draw: two renorms and no counting pass
        first = flashinfer.sampling.top_k_renorm_probs(probs, torch.where(top_k > 0, top_k, probs.shape[1]))
        then = flashinfer.sampling.top_p_renorm_probs(first, top_p)
        picked = torch.where(top_p[:, None] < 1.0, then.gather(1, tokens), first.gather(1, tokens))
    # a renormed near-certain token can land a ulp above 1
    return picked.squeeze(1).log().clamp_max(0)


@triton.jit
def _token_logprob_kernel(
    chunk_max_ptr, chunk_sum_ptr, logits_ptr, tokens_ptr, temperature_ptr, out_ptr,
    NSPLIT, part_stride_b, stride_b, stride_v,
    BLOCK_N: tl.constexpr,
):
    """Fold a row's partials into its logsumexp and score its token, scaled
    as ``_split_partials_kernel`` scales a greedy row (T = 1)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < NSPLIT
    base = row * part_stride_b + offs
    cmax = tl.load(chunk_max_ptr + base, mask=mask, other=-float("inf"))
    csum = tl.load(chunk_sum_ptr + base, mask=mask, other=0.0)
    gmax = tl.max(cmax)
    gsum = tl.sum(tl.where(mask, csum * tl.exp(cmax - gmax), 0.0))
    temp = tl.load(temperature_ptr + row)
    inv_temp = tl.where(temp == 0, 1.0, 1.0 / tl.maximum(temp, 1e-30))
    tok = tl.load(tokens_ptr + row)
    picked = tl.load(logits_ptr + row * stride_b + tok * stride_v).to(tl.float32) * inv_temp
    # the partials' approximate exp can put a near-certain token a hair above 0
    tl.store(out_ptr + row, tl.minimum(picked - gmax - tl.log(gsum), 0.0))


def _token_logprobs_cuda(logits: torch.Tensor, tokens: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """``token_logprobs`` in one fp32 pass, on the split-V softmax's partials.
    Plain torch doubled a captured sampler step at bs 128 on a 248k vocab."""
    B, V = logits.shape
    nsplit = _split_count(B, V, logits.device)
    chunk = -(-V // nsplit)
    f32 = dict(device=logits.device, dtype=torch.float32)
    cmax = torch.empty((B, nsplit), **f32)
    csum = torch.empty((B, nsplit), **f32)
    carg = torch.empty((B, nsplit), device=logits.device, dtype=torch.int32)
    out = torch.empty(B, **f32)
    hist_args, hist_flags = _history_kernel_args(None, logits)
    with torch.cuda.device(logits.device):
        _split_partials_kernel[(B, nsplit)](
            logits, temperature, logits, logits, *hist_args, cmax, csum, carg,
            V, chunk, logits.stride(0), logits.stride(1), 0, 0, cmax.stride(0),
            BLOCK_SIZE=min(_SPLIT_MAX_BLOCK, triton.next_power_of_2(chunk)),
            # for its greedy rows' T = 1; the argmax it also takes goes unread
            APPLY_PENALTY=False, INCLUDE_GREEDY=True, **hist_flags,
            num_warps=8, num_stages=2,
        )
        _token_logprob_kernel[(B,)](
            cmax, csum, logits, tokens, temperature, out,
            nsplit, cmax.stride(0), logits.stride(0), logits.stride(1),
            BLOCK_N=triton.next_power_of_2(nsplit), num_warps=1,
        )
    return out


class GenerationHistory:
    """Row-indexed generated-token counts and history, owned by SamplerRows."""

    def __init__(self, window: int, device: torch.device):
        self.window = window
        self.count = torch.zeros(1, dtype=torch.int32, device=device)
        self.tokens = torch.full((1, window), -1, dtype=torch.long, device=device)

    def grow(self, capacity: int) -> None:
        n = self.count.shape[0]
        if capacity <= n:
            return
        count = self.count.new_zeros(capacity)
        tokens = self.tokens.new_full((capacity, self.window), -1)
        count[:n].copy_(self.count)
        tokens[:n].copy_(self.tokens)
        self.count, self.tokens = count, tokens

    def reset(self, slot: int) -> None:
        self.count[slot:slot + 1].zero_()
        self.tokens[slot:slot + 1].fill_(-1)

    def read(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokens.index_select(0, rows), self.count.index_select(0, rows)

    def gather(self, rows: torch.Tensor, count: torch.Tensor, tokens: torch.Tensor | None) -> None:
        torch.index_select(self.count, 0, rows, out=count)
        if tokens is not None:
            torch.index_select(self.tokens, 0, rows, out=tokens)

    def write(self, rows: torch.Tensor, tokens: torch.Tensor | None, count: torch.Tensor) -> None:
        self.count.index_copy_(0, rows, count)
        if tokens is not None:
            self.tokens.index_copy_(0, rows, tokens)


class SamplerRows:
    """The sampler's admitted row identity and persistent device state.

    Allocation and release are host-only. Reset and growth run before the
    first GPU-thread read, behind the previous occupant's writes. Captures
    that read masters directly bound them before capture, so addresses stay
    fixed while ordinary resource admission waits for a free row.
    """

    def __init__(
        self, device: torch.device, history: GenerationHistory | None = None, capacity: int = 1,
        vocab_size: int | None = None,
    ):
        if vocab_size is not None and (
            isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0
        ):
            raise ValueError("sampler presence vocabulary must be a positive integer")
        self.vocab_size = vocab_size
        self.seen_words = (torch.zeros(capacity + 1, -(-vocab_size // 32), dtype=torch.int32, device=device)
                           if vocab_size is not None else None)
        self._bit_shifts = torch.arange(32, dtype=torch.int32, device=device) if vocab_size is not None else None
        self.device = device
        self.history = history
        self.capacity = capacity
        self.limit: int | None = None
        self._rid_to_slot: dict[Any, int] = {}
        self._free_slots = list(range(capacity, 0, -1))
        self._pending_reset: set[int] = set()
        self._lock = threading.Lock()
        self.offset = torch.zeros(capacity + 1, dtype=torch.long, device=device)
        if history is not None:
            history.grow(capacity + 1)

    def bound(self, capacity: int) -> None:
        """Before capture: reserve every master address a graph may read."""
        with self._lock:
            if max(self._rid_to_slot.values(), default=0) > capacity:
                raise ValueError("sampler capture capacity is below an allocated row")
            self.limit = capacity
            if self.capacity > capacity:
                self.capacity = capacity
                self._free_slots = [slot for slot in self._free_slots if slot <= capacity]
        self.sync()

    @property
    def free_rows(self) -> int:
        with self._lock:
            return len(self._free_slots) + (self.limit or self.capacity) - self.capacity

    def register_request(self, rid) -> int:
        if isinstance(rid, int) and rid < 0:
            return 0
        with self._lock:
            if rid in self._rid_to_slot:
                return self._rid_to_slot[rid]
            if not self._free_slots:
                capacity = self.capacity * 2
                if self.limit is not None:
                    capacity = min(capacity, self.limit)
                if capacity == self.capacity:
                    raise ValueError(f"all {capacity} sampler rows are held (MSTAR_SAMPLER_SLOTS)")
                self._free_slots.extend(range(capacity, self.capacity, -1))
                self.capacity = capacity
            slot = self._free_slots.pop()
            self._rid_to_slot[rid] = slot
            self._pending_reset.add(slot)
            return slot

    def register_trash_row(self, rid) -> None:
        with self._lock:
            self._rid_to_slot.setdefault(rid, 0)

    def unregister_request(self, rid) -> int | None:
        with self._lock:
            slot = self._rid_to_slot.pop(rid, None)
            if slot:
                self._pending_reset.discard(slot)
                self._free_slots.append(slot)
            return slot

    def has_slot(self, rid) -> bool:
        return rid in self._rid_to_slot

    def slot_of(self, rid) -> int:
        if isinstance(rid, int) and rid < 0:
            return 0
        if rid not in self._rid_to_slot:
            raise ValueError(f"request {rid!r} has no admitted sampler row")
        return self._rid_to_slot[rid]

    def indices(self, request_ids) -> torch.Tensor:
        with self._lock:
            # An unadmitted real request must not silently sample on trash.
            slots = [self.slot_of(rid) for rid in request_ids]
        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def sync(self) -> None:
        """Drain deferred state writes on the GPU thread before any read."""
        with self._lock:
            capacity = (self.limit or self.capacity) + 1
            resets, self._pending_reset = self._pending_reset, set()
        if capacity > self.offset.shape[0]:
            offset = self.offset.new_zeros(capacity)
            offset[:self.offset.shape[0]].copy_(self.offset)
            self.offset = offset
        if self.history is not None:
            self.history.grow(capacity)
        if self.seen_words is not None and capacity > self.seen_words.shape[0]:
            words = self.seen_words.new_zeros(capacity, self.seen_words.shape[1])
            words[:self.seen_words.shape[0]].copy_(self.seen_words)
            self.seen_words = words
        for slot in resets:
            self.offset[slot:slot + 1].zero_()
            if self.history is not None:
                self.history.reset(slot)
            if self.seen_words is not None:
                self.seen_words[slot:slot + 1].zero_()

    def check_vocab(self, width: int) -> None:
        if self.vocab_size is not None and width != self.vocab_size:
            raise ValueError(f"sampler logits width {width} differs from declared vocabulary {self.vocab_size}")

    def read_seen(self, rows: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        words = self.seen_words.index_select(0, rows)
        mask = ((words.unsqueeze(-1) >> self._bit_shifts) & 1).bool().flatten(1)[:, :self.vocab_size]
        if out is not None:
            out.copy_(mask)
            return out
        return mask

    def _pack_seen(self, mask: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(mask, (0, -mask.shape[-1] % 32))
        bits = padded.reshape(mask.shape[0], -(-self.vocab_size // 32), 32).to(torch.int32)
        return (bits << self._bit_shifts).sum(-1, dtype=torch.int32)

    def seed_seen(self, rid, tokens: torch.Tensor) -> None:
        """Inline prompt OR, after reset; duplicate tokens/words are idempotent."""
        self.sync()
        slot = self.slot_of(rid)
        if not slot or self.seen_words is None:
            return
        mask = torch.zeros((1, self.vocab_size), dtype=torch.bool, device=self.device)
        mask.scatter_(1, tokens.to(device=self.device, dtype=torch.long).reshape(1, -1), True)
        self.seen_words[slot:slot + 1].bitwise_or_(self._pack_seen(mask))

    def record_seen(self, rows: torch.Tensor, tokens: torch.Tensor) -> None:
        """One token per unique real row; duplicate writes can only hit trash."""
        index = rows * self.seen_words.shape[1] + tokens // 32
        flat = self.seen_words.flatten()
        bits = torch.ones_like(tokens, dtype=torch.int32) << (tokens % 32).to(torch.int32)
        flat.scatter_(0, index, flat.index_select(0, index) | bits)

    def write_seen(self, rows: torch.Tensor, mask: torch.Tensor, keep: torch.Tensor | None = None) -> None:
        words = self._pack_seen(mask)
        if keep is not None:
            words = torch.where(keep[:, None], words, self.seen_words.index_select(0, rows))
        self.seen_words.index_copy_(0, rows, words)

    def eager_offsets(self, request_ids) -> tuple[torch.Tensor, torch.Tensor]:
        self.sync()
        rows = self.indices(request_ids)
        return rows, self.offset.index_select(0, rows)

    def advance_offsets(self, rows: torch.Tensor, offsets: torch.Tensor, stride: int,
                        keep: torch.Tensor | None = None) -> None:
        advanced = offsets + stride
        if keep is not None:
            advanced = torch.where(keep, advanced, offsets)
        self.offset.index_copy_(0, rows, advanced)


# Temperatures below this are greedy: past it 1/T overflows fp32 and the paths disagree.
GREEDY_TEMPERATURE = 1e-5


def canonical_sampling_values(
    temperature, top_k, top_p, min_p,
) -> tuple[float, int, float, float]:
    """The one meaning of each knob, applied before either sampler reads it.

    ``temperature`` below ``GREEDY_TEMPERATURE`` is greedy (0). A greedy row
    ignores top-k / top-p / min-p, so they are written as off.
    """
    if temperature < GREEDY_TEMPERATURE:
        return 0.0, 0, 1.0, 0.0
    # any top_k past the vocab is off; past int32 it would not even fit the buffers
    top_k = int(top_k) if int(top_k) < 2**31 else 0
    return float(temperature), top_k, float(top_p), float(min_p)


class ScalarField(NamedTuple):
    """One per-request scalar a captured sampler reads per row: a ``HostBuffer``
    of ``SamplerBuffers`` by this name, and a cell of the resident rows."""
    name: str
    dtype: torch.dtype
    default: float


def scalar_fields(
    enable_min_p: bool = False, windowed: bool = False, min_tokens: bool = False,
    enable_top_p_first: bool = False, enable_logprobs: bool = False,
    enable_processed_logprobs: bool = False,
) -> tuple[ScalarField, ...]:
    """The scalars a sampler's spec enables, in the order its per-step rows
    pack them. Every field defaults to what ``SamplingConfig()`` reads as."""
    fields = [
        ScalarField("temperature", torch.float32, 1.0),
        ScalarField("top_k", torch.int32, 0),
        ScalarField("top_p", torch.float32, 1.0),
        ScalarField("seed", torch.long, 0),
        ScalarField("rep_penalty", torch.float32, 1.0),
    ]
    if enable_min_p:
        fields.append(ScalarField("min_p", torch.float32, 0.0))
    if windowed:
        fields += [ScalarField("window", torch.int32, 0), ScalarField("window_penalty", torch.float32, 1.0)]
    if min_tokens:
        fields.append(ScalarField("min_tokens", torch.int32, 0))
    if enable_top_p_first:
        fields += [ScalarField("top_p_first", torch.bool, False), ScalarField("top_p_min_keep", torch.int32, 1)]
    if enable_logprobs:
        fields.append(ScalarField("return_logprobs", torch.bool, False))
    if enable_processed_logprobs:
        fields.append(ScalarField("processed_logprobs", torch.bool, False))
    return tuple(fields)


def scalar_values(cfg: "SamplingConfig", windowed: bool) -> dict[str, float | int | bool]:
    """Each scalar's value for ``cfg``: the eager sampler's canonical values
    (greedy: the prep kernel's one-hot at argmax, filters off), so both paths
    read one meaning. A field the spec doesn't enable is never read."""
    t, k, p, m = canonical_sampling_values(cfg.temperature, cfg.top_k, cfg.top_p, cfg.min_p)
    return {
        "temperature": t, "top_k": k, "top_p": p, "seed": cfg.seed,
        # a windowed request's penalty is count-based, so the mask is inert for it
        "rep_penalty": float(cfg.presence_penalty if windowed else cfg.repetition_penalty),
        "min_p": m,
        "window": int(cfg.repetition_window),
        "window_penalty": float(cfg.window_penalty),
        "min_tokens": int(cfg.min_tokens),
        "top_p_first": bool(cfg.top_p_first),
        "top_p_min_keep": int(cfg.top_p_min_keep),
        "return_logprobs": bool(cfg.return_logprobs),
        "processed_logprobs": cfg.logprobs_mode == "processed",
    }


@dataclass
class SamplingConfig:
    # Optional declared-width metadata used by existing model config helpers.
    # Storage width belongs to SamplerSpec and must agree when this is set.
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
    # whether the penalty also counts the prompt's tokens; see ``SamplingReqConfig``
    penalize_prompt: bool = True
    return_logprobs: bool = False
    logprobs_mode: str = "raw"
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
class Sampler(BaseSampler):
    device: torch.device
    rows: SamplerRows
    _sampling_config: dict[str, SamplingConfig] = field(default_factory=dict)
    tp_group: "CommGroup | None" = None  # noqa: F821
    stop_ids: torch.Tensor | None = None
    enable_top_p_first: bool = False
    # The last ``sample``'s sampled-token log-probs, NaN in a row that did not
    # ask; None when no row asked.
    logprobs: torch.Tensor | None = None
    _settings_stager: PinnedStager | None = None

    def add_request(self, request_id: str):
        self._sampling_config[request_id] = SamplingConfig()

    def remove_request(self, request_id: str):
        if request_id in self._sampling_config:
            del self._sampling_config[request_id]

    def set_config(self, request_id: str, **kwargs):
        curr_config = asdict(self._sampling_config[request_id])
        # None is "not set": it keeps the current value rather than replacing it
        kwargs = {
            k: arg for k, arg in kwargs.items()
            if k in curr_config.keys() and arg is not None
        }
        merged = {**curr_config, **kwargs}
        (merged["temperature"], merged["top_k"], merged["top_p"],
         merged["min_p"]) = canonical_sampling_values(
            merged["temperature"], merged["top_k"], merged["top_p"], merged["min_p"],
        )
        width = merged["vocab_size"]
        if width is not None and width != self.rows.vocab_size:
            raise ValueError("sampling config vocabulary must match the declared sampler vocabulary")
        self._sampling_config[request_id] = SamplingConfig(**merged)

    # sampling runs inside the forward; nothing here is worth tracing
    @torch.compiler.disable
    def sample(
        self, request_ids: list[str], logits: torch.Tensor,
        apply_penalty: bool = True, apply_filters: bool = True,
        kept_rids: "frozenset | None" = None,
    ) -> torch.Tensor:
        """Return the sampled tokens as a single [B] int tensor.

        Callers that want a per-rid mapping can slice `tokens[i:i+1]` using
        the rid order in `request_ids`. We return the raw tensor (instead of
        a dict of views) because constructing the dict adds Python overhead
        the hot path doesn't need.

        ``apply_penalty`` False skips the repetition penalty, as the captured
        sampler does for a node or step without it. Only ``kept_rids`` (None:
        all) advance their RNG offset and seen-token mask.

        """
        self.rows.check_vocab(logits.shape[-1])
        configs = [self._sampling_config[rid] for rid in request_ids]
        slots, rand_offset = self.rows.eager_offsets(request_ids)
        history, rows = self._gather_history(slots, configs)
        order = self._gather_filter(configs)
        temperature = torch.tensor([c.temperature for c in configs], device=logits.device)
        top_k = torch.tensor([c.top_k if apply_filters else 0 for c in configs],
                             device=logits.device, dtype=torch.int32)
        top_p = torch.tensor([c.top_p if apply_filters else 1.0 for c in configs], device=logits.device)
        if self.rows.history is not None:
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

        any_rep_pen = apply_penalty and any(p != 1.0 for p in penalties)
        any_greedy = any(c.temperature == 0 for c in configs)
        top_k_zero_count = sum(c.top_k == 0 or not apply_filters for c in configs)

        seen_mask = self.rows.read_seen(slots) if any_rep_pen and self.rows.seen_words is not None else None

        processed = [c.return_logprobs and c.logprobs_mode == "processed" for c in configs]
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
            history=history if logits.device.type == "cuda" else None,
            order=order,
            return_probs=any(processed),
        )
        tokens, probs = tokens if any(processed) else (tokens, None)

        tokens = self._broadcast_tokens(tokens)

        want = [c.return_logprobs for c in configs]
        self.logprobs = None
        if any(want):
            chosen = token_logprobs(logits, tokens, temperature)
            if probs is not None:
                chosen = torch.where(
                    torch.tensor(processed, device=logits.device),
                    processed_logprobs(probs, tokens, top_k, top_p, history, order), chosen,
                )
            self.logprobs = torch.where(torch.tensor(want, device=logits.device), chosen, torch.nan)

        if history is not None:
            advance_history(history, tokens)
            new_tokens, count = history.tokens, history.count
            if kept_rids is not None:
                # only the kept rows' draws count
                sel = torch.tensor([i for i, rid in enumerate(request_ids) if rid in kept_rids],
                                   dtype=torch.long, device=rows.device)
                rows, count = rows.index_select(0, sel), count.index_select(0, sel)
                new_tokens = None if new_tokens is None else new_tokens.index_select(0, sel)
            self.rows.history.write(rows, new_tokens, count)

        keep = None if kept_rids is None else torch.tensor(
            [rid in kept_rids for rid in request_ids], dtype=torch.bool, device=logits.device,
        )
        if apply_penalty and self.rows.seen_words is not None:
            scatter = slots if keep is None else torch.where(keep, slots, 0)
            self.rows.record_seen(scatter, tokens)
        self.rows.advance_offsets(slots, rand_offset, _rng_offset_stride(logits.device.type, logits.shape[-1]), keep)

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
        self, rows: torch.Tensor, configs: list[SamplingConfig],
    ) -> tuple["HistoryRows | None", torch.Tensor | None]:
        """This batch's ``HistoryRows`` and the master rows they came from;
        ``None`` when the node keeps no history."""
        if self.rows.history is None:
            return None, None
        tokens, count = self.rows.history.read(rows)
        dev = self._stage([
            [c.repetition_window for c in configs],
            [c.window_penalty for c in configs],
            [c.min_tokens for c in configs],
        ])
        windowed = self.rows.history.window > 0
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


class Draw(NamedTuple):
    tokens: torch.Tensor
    probs: torch.Tensor  # the distribution the filters ran on


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
    return_probs: bool = False,
) -> torch.Tensor | Draw:
    """Sample normalized CUDA inputs with FlashInfer; with ``return_probs``,
    also the distribution the filters ran on."""
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
            result = _filter_floor_and_draw(probs, top_k, top_p, seed, rand_offset, history, order)
        elif top_k_zero_count == logits.shape[0]:
            # top-k is off for every row
            result = flashinfer.sampling.top_p_sampling_from_probs(
                probs, top_p,
                deterministic=True,
                seed=seed, offset=rand_offset,
            )
        else:
            # top_k 0 is off: FlashInfer reads it as k=0, so a mixed batch's 0 rows drew token 0
            top_k = torch.where(top_k > 0, top_k, logits.shape[1])
            result = flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, top_k, top_p,
                deterministic=True,
                seed=seed, offset=rand_offset,
            )
        result = result[0] if isinstance(result, tuple) else result
        return Draw(result, probs) if return_probs else result


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
    return_probs: bool = False,
) -> torch.Tensor | Draw:
    """Sample tokens from logits with temperature, top-k, top-p, and repetition penalty.

    Args:
        logits: [batch_size, vocab_size] raw logits from lm_head.
        temperature: Scalar or per-request tensor [batch_size].
            0 = greedy (argmax) for that request. >0 = scaled sampling.
        top_k: Scalar or per-request tensor [batch_size]. 0 = disabled.
        top_p: Scalar or per-request tensor [batch_size]. 1.0 = disabled.
        repetition_penalty: Sign-aware penalty: divides positive logits and
            multiplies negative ones (1.0 = disabled).
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
        return_probs: (CUDA only) also return the distribution the filters
            ran on, for ``processed_logprobs``.

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
            return_probs=return_probs,
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

def sample_accelerator_graphable_gpu(
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
    return_probs: bool = False,
) -> torch.Tensor | Draw:
    """Deterministic per-batch top-k/top-p sampling for graph-captured code.

    On CUDA, a vocabulary of at most ``FUSED_SAMPLE_MAX_VOCAB`` without
    ``min_p``, ``history``, ``order`` or ``return_probs`` draws in one
    ``fused_sample`` launch:
    per-row seeds, so batch-invariant, but its cuts rank on bf16-truncated
    keys (see ``_fused_sample_kernel``), and the eager ``Sampler`` path
    (``_sample_cuda``) keeps FlashInfer, so for those vocabularies an eager
    and a captured step draw differently for the same seed.

    Otherwise it routes through the fused Triton prep kernel
    (``fused_temperature_softmax``) so the CUDA-graph path can apply the same
    sign-aware repetition penalty as the regular ``Sampler``, then samples with
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
        return_probs: (CUDA only) also return the distribution the filters
            ran on, for ``processed_logprobs``.

    Returns:
        ``[batch_size]`` int64 sampled token IDs. FlashInfer's default
        output is int32; we cast to int64 so the caller can index
        ``nn.Embedding`` modules (which require int64 indices) directly.
    """
    if logits.device.type == "xpu":
        if history is not None or order is not None:
            raise ValueError("generation-aware sampling controls are CUDA-only")
        with torch.xpu.device(logits.device):
            return _sample_xpu(
                logits, temperature, top_k, top_p,
                rep_penalty if apply_penalty else 1.0,
                seen_tokens if apply_penalty else None,
                run_greedy=True,
                top_k_zero_count=None,
                seed=seed,
                rand_offset=offset,
                min_p=min_p,
            )
    elif logits.device.type == "cuda":
        import flashinfer

        with torch.cuda.device(logits.device):
            if (
                logits.shape[1] <= FUSED_SAMPLE_MAX_VOCAB and min_p is None
                and history is None and order is None and not return_probs
            ):
                return fused_sample(
                    logits, temperature, top_k, top_p, seed, offset,
                    rep_penalty if apply_penalty else None, seen_tokens if apply_penalty else None,
                )
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
                samples = _filter_floor_and_draw(probs, top_k, top_p, seed, offset, history, order)
            else:
                # NOTE: this FlashInfer path (not ``fused_sample``'s) is NOT
                # batch-invariant. FlashInfer reads only row 0's seed
                # and offset and folds the row index into philox, so identical (probs,
                # seed, offset) yield different tokens at different batch positions.
                # Sampling is thus reproducible only within a fixed batch layout; under
                # continuous batching (shifting positions) a request's stream is not.
                # Measured in test/sampling_test/flashinfer_batch_test.py (raw FlashInfer).
                samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
                    probs, torch.where(top_k > 0, top_k, logits.shape[1]), top_p, deterministic=True,
                    seed=seed, offset=offset,
                )
        samples = samples.to(torch.int64)
        return Draw(samples, probs) if return_probs else samples
    else:
        raise ValueError(
            f"Graph-captured sampling is unsupported on {logits.device.type!r}; "
            "expected 'cuda' or 'xpu'."
        )


@dataclass
class AcceleratorGraphableSampler(BaseSampler):
    rows: SamplerRows
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
    # ``[bs]`` per-request ``return_logprobs``, and the per-step row the
    # captured sample writes each row's log-prob to (NaN where it did not
    # ask); both ``None`` unless the node's spec enables log-probs.
    return_logprobs_buf: torch.Tensor | None = None
    logprob_buf: torch.Tensor | None = None
    # ``[bs]`` per-request ``logprobs_mode == "processed"``; ``None`` unless
    # the spec enables processed log-probs.
    processed_logprobs_buf: torch.Tensor | None = None
    tp_group: "CommGroup | None" = None  # noqa: F821
    # In-graph gather/scatter (``MSTAR_SAMPLER_INGRAPH_SCATTER``): the slot
    # masters and this slot's padded index rows, all static, so the captured
    # sample reads its offsets from the master and writes the advanced
    # offsets back without a launch after the replay.
    # The scatter row is the gather row with the step's dropped rows
    # (``SamplerStep.kept_rids``) sent to the trash row, as padding rows are.
    slot_idx_view: torch.Tensor | None = None
    scatter_idx_view: torch.Tensor | None = None
    ingraph_scatter: bool = False

    def gather_in_graph(self) -> None:
        """Offsets of this step's rows off the master, inside the graph."""
        torch.index_select(
            self.rows.offset, 0, self.slot_idx_view, out=self.offset_buf,
        )
        if self.history is not None:
            self.rows.history.gather(self.slot_idx_view, self.history.count, self.history.tokens)

    def scatter_in_graph(self) -> None:
        """Advanced offsets back to the master, inside the graph: every row,
        padding and dropped rows into the trash row."""
        self.rows.offset.index_copy_(0, self.scatter_idx_view, self.offset_buf)
        if self.history is not None:
            self.rows.history.write(self.scatter_idx_view, self.history.tokens, self.history.count)

    @torch.compiler.disable
    def sample(
        self, request_ids: list[str], logits: torch.Tensor,
        apply_penalty: bool = False,
        apply_filters: bool = True,
    ):
        self.rows.check_vocab(logits.shape[-1])
        top_k, top_p = self.top_k_buf, self.top_p_buf
        if not apply_filters:
            # top-k 0 is "off"; captured, these are constant fills
            top_k, top_p = torch.zeros_like(top_k), torch.ones_like(top_p)
        if self.ingraph_scatter:
            self.gather_in_graph()
            if apply_penalty and self.seen_tokens_buf is not None:
                self.rows.read_seen(self.slot_idx_view, out=self.seen_tokens_buf)
        processed = self.processed_logprobs_buf is not None
        codes = sample_accelerator_graphable_gpu(
            logits, self.temperature_buf,
            top_k, top_p,
            self.seed_buf, self.offset_buf,
            apply_penalty=apply_penalty,
            rep_penalty=self.rep_penalty_buf,
            seen_tokens=self.seen_tokens_buf,
            min_p=self.min_p_buf,
            history=self.history if logits.device.type == "cuda" else None,
            order=self.order,
            return_probs=processed,
        )
        codes, probs = codes if processed else (codes, None)
        self.offset_buf.add_(_rng_offset_stride(logits.device.type, logits.shape[-1]))
        codes = self._broadcast_tokens(codes)
        if self.logprob_buf is not None:
            # before the history moves on: the min-tokens floor reads its count
            chosen = token_logprobs(logits, codes, self.temperature_buf)
            if processed:
                chosen = torch.where(
                    self.processed_logprobs_buf,
                    processed_logprobs(probs, codes, top_k, top_p, self.history, self.order), chosen,
                )
            self.logprob_buf.copy_(torch.where(self.return_logprobs_buf, chosen, torch.nan))
        if self.history is not None:
            # the TP-agreed token, so every rank's history stays the same
            advance_history(self.history, codes)
        if self.ingraph_scatter:
            self.scatter_in_graph()
        if apply_penalty and self.seen_tokens_buf is not None:
            if self.ingraph_scatter:
                self.rows.record_seen(self.scatter_idx_view, codes)
            else:
                self.seen_tokens_buf.scatter_(1, codes.unsqueeze(1), True)
        return codes


@dataclass
class HostBuffer:
    """Storage for one per-request scalar sampling parameter that only changes
    on a config update (temperature, top-k, top-p, seed, penalty).

    - ``buf``    ``[cg_slots, max_bs]`` per-step device tensor the graph reads
      (its address must stay stable across replays).
    - ``master`` ``[capacity]`` slot-indexed HOST array, one row per request.

    The master is on the host because the gather runs in the pre-plan beside a
    live graph, so it must be one H2D copy with no device ops (``mstar.utils.h2d``).
    A sampler's buffers share that copy: ``ScalarRows`` packs their ``buf`` rows.
    """
    buf: torch.Tensor
    master: np.ndarray
    default: float
    dtype: torch.dtype

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


class ScalarRows:
    """The per-step rows of a sampler's ``HostBuffer``s in one device byte row
    per cg slot, each buffer's ``buf`` a typed view into it, so a gather stages
    them all in one H2D copy and skips it when the slot already holds the
    bytes (a copy per buffer cost more than the gathers themselves)."""

    def __init__(self, buffers: list[HostBuffer], max_bs: int, device: torch.device):
        self.buffers = buffers
        cg_slots = buffers[0].buf.shape[0]
        # 8-byte aligned regions, so every typed view divides its row stride
        self.regions: list[tuple[int, int]] = []
        size = 0
        for buf in buffers:
            nbytes = max_bs * buf.master.itemsize
            self.regions.append((size, nbytes))
            size += -(-nbytes // 8) * 8
        self.rows = torch.empty(cg_slots, size, dtype=torch.uint8, device=device)
        for buf, (start, nbytes) in zip(buffers, self.regions, strict=True):
            view = self.rows[:, start:start + nbytes].view(buf.dtype)
            view.copy_(buf.buf)
            buf.buf = view
        # per slot: a pre-plan may gather one slot while the gpu thread gathers the other
        self._host = [np.zeros(size, np.uint8) for _ in range(cg_slots)]
        self._mirrors = [H2DMirror() for _ in range(cg_slots)]
        # room for several rows a segment, so most gathers leave none (each
        # segment left costs an event record)
        self._stager = PinnedStager(torch.uint8, numel=8 * size)

    def gather(self, rows: np.ndarray, padded_bs: int, cg_slot: int) -> None:
        """Rows ``rows`` of every buffer's master into ``cg_slot``'s row. The
        bytes past ``padded_bs`` in each region are whatever the slot held: a
        step reads only its first ``padded_bs``."""
        slot = cg_slot if self.rows.shape[0] > 1 else 0
        host = self._host[slot]
        for buf, (start, _) in zip(self.buffers, self.regions, strict=True):
            master = buf.master
            np.take(master, rows, out=host[start:start + padded_bs * master.itemsize].view(master.dtype))
        self._stager.copy_(self.rows[slot], host, mirror=self._mirrors[slot])


@dataclass
class SamplerBuffers:
    """Capture scratch and config staging, indexed by the sampler's owned rows.

    Each capture slot keeps its indices and RNG/history scratch. In-graph
    sampling reads and writes the owner's masters per draw; host-scattered
    sampling does the same around replay. Presence masks unpack from the
    same owner into per-capture scratch; there is no second content authority.
    """
    max_batch_size: int
    rows: SamplerRows
    temperature: HostBuffer
    top_k: HostBuffer
    top_p: HostBuffer
    seed: HostBuffer
    rep_penalty: HostBuffer
    # Per-capture scratch; the authoritative counter belongs to rows.
    offset: torch.Tensor

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
    seen_tokens: torch.Tensor | None = None
    # Per-request min-p; allocated only for submodules whose spec enables it,
    # so every other node's captured sampler is unchanged.
    min_p: "HostBuffer | None" = None
    # Generation-aware processors, each allocated only when the spec enables
    # it. The per-request settings are static rows like the ones above; the
    # per-step state (count, recent tokens) is the resource's
    # ``GenerationHistory``, gathered and persisted with the RNG offset.
    history_count: torch.Tensor | None = None
    history_tokens: torch.Tensor | None = None
    window: "HostBuffer | None" = None
    window_penalty: "HostBuffer | None" = None
    min_tokens: "HostBuffer | None" = None
    stop_ids: torch.Tensor | None = None
    top_p_first: "HostBuffer | None" = None
    top_p_min_keep: "HostBuffer | None" = None
    # Per-request ``return_logprobs`` and the ``[cg_slots, max_bs]`` per-step
    # log-prob rows the captured sample writes; only when the spec enables them.
    return_logprobs: "HostBuffer | None" = None
    logprob: torch.Tensor | None = None
    # Per-request ``logprobs_mode == "processed"``; only when the spec enables it.
    processed_logprobs: "HostBuffer | None" = None
    # Request rows in the masters, trash row excluded; doubles when registrations outgrow it.
    _master_capacity: int = field(default=0, repr=False)
    ingraph_scatter: bool = False
    # Master row 0, never given to a request, so its config stays SamplingConfig().
    # Padding and capture rows read it; in graph they write it.
    _trash_slot: ClassVar[int] = 0
    # Number of double-buffer slots (per-step buffers carry this leading dim).
    cg_slots: int = 1
    # Per-step slot-index staging, per cg slot. ``_slot_idx_cpu`` is pinned so
    # the H2D copy can be issued non-blocking; ``_slot_idx_gpu`` is the
    # device-side index tensor that ``index_select`` reads from. A step of
    # ``padded_bs`` rows uses ``[:padded_bs]`` as its gather index and, with
    # ``ingraph_scatter``, ``[padded_bs:2 * padded_bs]`` as its scatter index,
    # so one copy uploads both.
    _slot_idx_cpu: torch.Tensor = field(default=None, repr=False)
    _slot_idx_gpu: torch.Tensor = field(default=None, repr=False)
    # Zero-copy numpy view of ``_slot_idx_cpu``, so staging is one slice write
    # rather than a tensor __setitem__ per row. Built lazily; the tensor is
    # allocated once and never rebound, so the view stays valid.
    _slot_idx_np: Any = field(default=None, repr=False)
    # Real (unpadded) batch size of the last gather per cg slot — the
    # scatter-back writes only these rows (padding rows all map to slot 0).
    _last_real_bs: list[int] = field(default_factory=list, repr=False)
    # memoized per (slot, real rows): see ``_scatter_idx``
    _scatter_idx_views: dict = field(default_factory=dict, repr=False)
    # (cg slot, padded_bs) -> (device dst, pinned src, gather view) of ``_upload_slot_idx``
    _upload_views: dict = field(default_factory=dict, repr=False)
    # cg slot -> the (rids, padded_bs) its pinned index row currently holds, so
    # gather_dynamic can re-issue the H2D without redoing the O(bs) CPU fill
    _staged_rids: dict[int, tuple[tuple[str, ...], int]] = field(
        default_factory=dict, repr=False
    )
    # Slots awaiting init, consumed by the next gather so the rows are written
    # on the gathering thread rather than racing it from the main one.
    _pending_init: set[int] = field(default_factory=set, repr=False)
    # Host master-row writes vs their regrow; a plan-thread write between copy and swap was lost.
    _host_master_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
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
    # (padded_bs, cg slot) -> the sampler over that slot's views; see sampler_for
    _samplers: dict = field(default_factory=dict, repr=False)
    # (bucket, cg slot) -> a resident capture's sampler; see resident_sampler_for
    _resident_samplers: dict = field(default_factory=dict, repr=False)
    # the scalar buffers' per-step rows, staged together; set by ``allocate``
    _scalar_rows: ScalarRows | None = field(default=None, repr=False)
    # the scalars the spec enables, each a ``HostBuffer`` field by its name
    scalar_fields: tuple[ScalarField, ...] = ()
    # The node's resident rows, when its planner binds this sampler: a
    # request's scalars as cells by resident slot (`sampler/resident.py`).
    cells: "SamplerCells | None" = field(default=None, repr=False)

    @property
    def history(self) -> GenerationHistory | None:
        return self.rows.history

    def _scalar_buffers(self) -> list[HostBuffer]:
        return [getattr(self, f.name) for f in self.scalar_fields]

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
        rows: SamplerRows | None = None,
        stop_ids: torch.Tensor | None = None,
        enable_top_p_first: bool = False,
        ingraph_scatter: bool = False,
        slots: int | None = None,
        enable_logprobs: bool = False,
        enable_processed_logprobs: bool = False,
    ) -> "SamplerBuffers":
        """Allocate sampling buffers for ``max_batch_size``.

        The masters hold the trash row (row 0, ``_trash_slot``) plus one row per
        request, ``max_batch_size`` to start with, doubling as registrations
        outgrow them. With ``ingraph_scatter`` the RNG/history masters hold ``slots``
        request rows (the knob's 4096 by default), allocated once: the captured
        graphs keep their addresses, so they never grow, and config/mask caches
        grow up to that bound. A registration past it raises, so the caller
        takes rows only while ``rows.free_rows`` says one is left.

        ``vocab_size`` (when not None) enables the seen-token mask buffer for the
        repetition penalty. ``history`` (with ``stop_ids`` and
        ``enable_top_p_first``) enables the generation-aware processors; its
        per-step buffers are sized here too. The master rows default to a
        ``SamplingConfig()`` row (temp=1, top_k=0, top_p=1, rep_penalty=1),
        which the trash row keeps for good. ``cg_slots`` double-buffers the
        per-step tensors so the sampler can pre-plan.
        """
        pinned = DeviceCapability.of(device).pin_memory
        owner = rows if rows is not None else SamplerRows(device, history, max_batch_size, vocab_size)
        if history is not None and owner.history is not history:
            raise ValueError("sampler history must belong to its row owner")
        if vocab_size != owner.vocab_size:
            raise ValueError("sampler capture vocabulary must match its row owner")
        history = owner.history
        cap = max(max_batch_size, owner.capacity)
        if ingraph_scatter:
            owner.bound(max(max_batch_size, slots if slots is not None else sampler_slots()))
        master_rows = cap + 1
        # the in-graph scatter's index follows the gather index (see ``_slot_idx_cpu``)
        idx_cols = 2 * max_batch_size if ingraph_scatter else max_batch_size

        def mk(dtype: torch.dtype, default: float) -> HostBuffer:
            return HostBuffer.allocate(
                max_batch_size, master_rows, device, dtype, default, cg_slots
            )

        seen_tokens = (torch.zeros(cg_slots, max_batch_size, vocab_size, dtype=torch.bool, device=device)
                       if vocab_size is not None else None)
        fields = cls.fields_for(
            enable_min_p=enable_min_p, history=history, stop_ids=stop_ids,
            enable_top_p_first=enable_top_p_first, enable_logprobs=enable_logprobs,
            enable_processed_logprobs=enable_processed_logprobs,
        )
        buffers = cls(
            max_batch_size=max_batch_size, rows=owner,
            **{f.name: mk(f.dtype, f.default) for f in fields},
            scalar_fields=fields,
            offset=torch.zeros(cg_slots, max_batch_size, dtype=torch.long, device=device),
            tp_group=tp_group,
            seen_tokens=seen_tokens,
            history_count=(torch.zeros(cg_slots, max_batch_size, dtype=torch.int32, device=device)
                           if history is not None else None),
            history_tokens=(torch.full((cg_slots, max_batch_size, history.window), -1, dtype=torch.long, device=device)
                            if history is not None and history.window else None),
            stop_ids=stop_ids,
            logprob=torch.zeros(cg_slots, max_batch_size, device=device) if enable_logprobs else None,
            _master_capacity=cap,
            ingraph_scatter=ingraph_scatter,
            cg_slots=cg_slots,
            _slot_idx_cpu=torch.zeros(cg_slots, idx_cols, dtype=torch.long, pin_memory=pinned),
            _slot_idx_gpu=torch.zeros(cg_slots, idx_cols, dtype=torch.long, device=device),
            _last_real_bs=[0] * cg_slots,
        )
        buffers._scalar_rows = ScalarRows(buffers._scalar_buffers(), max_batch_size, device)
        return buffers

    @staticmethod
    def fields_for(
        enable_min_p: bool = False,
        history: GenerationHistory | None = None,
        stop_ids: torch.Tensor | None = None,
        enable_top_p_first: bool = False,
        enable_logprobs: bool = False,
        enable_processed_logprobs: bool = False,
    ) -> tuple[ScalarField, ...]:
        """The scalars ``allocate`` gives these options a buffer each."""
        return scalar_fields(
            enable_min_p=enable_min_p, windowed=history is not None and history.window > 0,
            min_tokens=stop_ids is not None, enable_top_p_first=enable_top_p_first,
            enable_logprobs=enable_logprobs, enable_processed_logprobs=enable_processed_logprobs,
        )

    def slice_for_bs(
        self, bs: int, cg_slot: int = 0,
        scalars: Mapping[str, torch.Tensor] | None = None, index: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Return bs-sized views into one slot's buffers (zero-copy slices) plus
        the owning submodule's ``tp_group`` so the constructed sampler
        broadcasts across TP ranks. A resident capture passes its own
        per-step ``scalars`` (by field name), which its graph fills, and
        ``index`` row (gather half, then scatter half), which its stage uploads."""
        def scalar(name: str) -> torch.Tensor | None:
            if scalars is not None:
                return scalars[name][:bs] if name in scalars else None
            buf = getattr(self, name)
            return buf.slot_view(cg_slot, bs) if buf is not None else None

        if index is None:
            index = self._slot_idx_gpu[cg_slot]
        return {
            "rows": self.rows,
            "temperature_buf": scalar("temperature"),
            "top_k_buf": scalar("top_k"),
            "top_p_buf": scalar("top_p"),
            "seed_buf": scalar("seed"),
            "offset_buf": self.offset[cg_slot, :bs],
            "rep_penalty_buf": scalar("rep_penalty"),
            "seen_tokens_buf": self.seen_tokens[cg_slot, :bs] if self.seen_tokens is not None else None,
            "min_p_buf": scalar("min_p"),
            "history": self._history_rows(bs, cg_slot, scalar) if self.history is not None else None,
            "order": (
                FilterOrder(scalar("top_p_first"), scalar("top_p_min_keep"))
                if self.top_p_first is not None else None
            ),
            "return_logprobs_buf": scalar("return_logprobs"),
            "logprob_buf": self.logprob[cg_slot, :bs] if self.logprob is not None else None,
            "processed_logprobs_buf": scalar("processed_logprobs"),
            "tp_group": self.tp_group,
            "slot_idx_view": index[:bs],
            "scatter_idx_view": index[bs:2 * bs] if self.ingraph_scatter else None,
            "ingraph_scatter": self.ingraph_scatter,
        }

    def _history_rows(self, bs: int, cg_slot: int, scalar) -> HistoryRows:
        return HistoryRows(
            count=self.history_count[cg_slot, :bs],
            tokens=self.history_tokens[cg_slot, :bs] if self.window is not None else None,
            window=scalar("window"),
            window_penalty=scalar("window_penalty"),
            min_tokens=scalar("min_tokens"),
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
        values = scalar_values(cfg, windowed=self.window is not None)
        with self._host_master_lock:
            for f in self.scalar_fields:
                getattr(self, f.name).write_master_row(slot, values[f.name])
        self._config_version += 1

    def _grow_master(self, new_capacity: int) -> None:
        """Double-and-copy the master buffers up to at least ``new_capacity``.

        Triggered when concurrently-registered requests exceed the current
        master capacity. Per-step buffers (sized to the cuda-graph max_bs) are
        NOT resized — the gather only reads ``padded_bs`` rows from master.

        Host rows only, under the lock ``_write_master_row`` takes, so a plan-thread
        ``_init_slot`` cannot land in the old array; device masters grow on the gpu thread.
        """
        if new_capacity <= self._master_capacity:
            return
        rows = new_capacity + 1  # plus the trash row
        with self._host_master_lock:
            for buf in self._scalar_buffers():
                buf.grow_master(rows)
        self._master_capacity = new_capacity

    def register_request(
        self, rid: str, sampling_config: SamplingConfig | None = None,
    ) -> None:
        """Allocate a slot for ``rid`` and seed its master row."""
        # a capture padding handle (negative) takes no slot; it reads slot 0
        if isinstance(rid, int) and rid < 0:
            return
        slot = self.rows.register_request(rid)
        if rid in self._cached_config:
            if sampling_config is not None:
                self.update_request_config(rid, sampling_config)
            return
        self._grow_master(self.rows.capacity)
        self._pending_init.add(slot)
        self._cached_config[rid] = (
            sampling_config if sampling_config is not None else SamplingConfig()
        )
        if self.cells is not None:
            self.cells.set_config(rid, self._cached_config[rid], windowed=self.window is not None)

    def register_trash_row(self, rid: str) -> None:
        """Point ``rid`` at the trash row instead of a slot of its own.

        For capture padding rows: they hold no request state, so they take no
        master row, and their in-graph writes land on the trash row.
        """
        self.rows.register_trash_row(rid)

    def unregister_request(self, rid: str) -> None:
        """Free the slot owned by ``rid`` (no GPU writes)."""
        slot = self.rows.unregister_request(rid)
        if slot is None or slot == self._trash_slot:
            return
        self._cached_config.pop(rid, None)
        if self.cells is not None:
            self.cells.drop(rid)
        self._pending_init.discard(slot)
        self._staged_rids.clear()
        self._static_key.clear()

    def update_request_config(
        self, rid: str, sampling_config: SamplingConfig,
    ) -> None:
        """Update the master row for ``rid`` only when its config changed.

        AR engine calls this every step (mirroring ``Sampler.set_config``).
        Steady-state requests have identical configs across steps, so the
        change-check skips the H2D path entirely. Dynamic contents stay with
        the row owner and never come from this config cache.
        """
        slot = self.rows._rid_to_slot.get(rid)
        if slot is None or slot == self._trash_slot:
            # Request not yet registered for this submodule (e.g. ar_engine
            # may invoke set_config for a node that doesn't own a runner /
            # SamplerBuffers), or a capture row whose trash row stays defaults.
            return
        prev = self._cached_config.get(rid)
        if prev == sampling_config:
            return
        self._cached_config[rid] = sampling_config
        # A slot awaiting init writes this row at gather time instead.
        if slot not in self._pending_init:
            self._write_master_row(slot, sampling_config)
        if self.cells is not None:
            # its resident row is rewritten whole by the next step that reads it
            self.cells.set_config(rid, sampling_config, windowed=self.window is not None, stale=True)

    def _init_slot(self, slot: int, rid: str) -> None:
        """Init for a newly registered slot, on the gathering thread so rows are
        written before the gather reads them. No device work (pre-plan); the
        offset reset is queued at registration instead."""
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

        Padding slots (``i >= len(request_ids)``) read the trash row: the
        captured graph forwards them through the same kernels as real slots,
        but their outputs are discarded by the runner's dummy-rid remap.
        Unadmitted real requests cannot use the trash row.
        """
        assert padded_bs <= self.max_batch_size, (
            f"padded_bs={padded_bs} exceeds SamplerBuffers.max_batch_size="
            f"{self.max_batch_size}"
        )
        # _init_slot runs per slot, but the staging writes become one. Elementwise, this
        # was ~46us at bs=16 — all Python, and the plan thread pays it inline.
        if self._slot_idx_np is None:
            self._slot_idx_np = self._slot_idx_cpu.numpy()
        get = self.rows.slot_of
        pending = self._pending_init
        rows = []
        for rid in request_ids:
            slot = get(rid)
            if slot in pending:
                self._init_slot(slot, rid)
            rows.append(slot)
        if len(rows) < padded_bs:
            rows.extend([self._trash_slot] * (padded_bs - len(rows)))
        self._slot_idx_np[cg_slot, :padded_bs] = rows
        self._staged_rids[cg_slot] = (tuple(request_ids), padded_bs)

    def _stage_scatter_idx(
        self, request_ids: list[str], padded_bs: int, cg_slot: int, kept: "frozenset | None",
    ) -> None:
        """The in-graph scatter's index, behind the staged gather index: the same
        rows, except that a row ``kept`` leaves out writes the trash row, so its
        offset stays as it was (``SamplerStep.kept_rids``)."""
        row = self._slot_idx_np[cg_slot]
        scatter = row[padded_bs:2 * padded_bs]
        scatter[:] = row[:padded_bs]
        if kept is not None:
            for i, rid in enumerate(request_ids):
                if rid not in kept:
                    scatter[i] = self._trash_slot

    def _upload_slot_idx(self, padded_bs: int, cg_slot: int) -> torch.Tensor:
        """H2D the staged row on the CURRENT stream, and hand back its gather view."""
        key = (cg_slot, padded_bs)
        views = self._upload_views.get(key)
        if views is None:
            cols = 2 * padded_bs if self.ingraph_scatter else padded_bs
            views = self._upload_views[key] = (
                self._slot_idx_gpu[cg_slot, :cols], self._slot_idx_cpu[cg_slot, :cols],
                self._slot_idx_gpu[cg_slot, :padded_bs],
            )
        dst, src, idx_view = views
        dst.copy_(src, non_blocking=True)
        return idx_view

    def gather_static(
        self, request_ids: list[str], padded_bs: int, cg_slot: int,
    ) -> None:
        """Gather the per-request scalar config (temp/top_k/top_p/seed/penalty)
        into ``cg_slot``. Safe to pre-plan: these change only on a config
        update, never step to step.

        Skipped when ``cg_slot`` last gathered this same batch and no master row
        changed since. Never skipped while a row of this batch awaits init,
        which also catches a request id re-registered onto a new slot. Slots
        awaiting init that belong to requests outside the batch (registered,
        still queued) do not block the skip: their rows are not read here, and
        their init runs on the gather that first includes them."""
        rids = tuple(request_ids)
        if self._static_key.get(cg_slot) == (rids, padded_bs, self._config_version):
            pending = self._pending_init
            get = self.rows._rid_to_slot.get
            if not pending or not any(get(rid) in pending for rid in request_ids):
                self._last_real_bs[cg_slot] = len(request_ids)
                return
        self._stage_slot_idx(request_ids, padded_bs, cg_slot)
        # one H2D copy (pre-plan, see ScalarRows); gather_dynamic uploads
        # the device-side index row
        self._scalar_rows.gather(self._slot_idx_np[cg_slot, :padded_bs], padded_bs, cg_slot)
        self._last_real_bs[cg_slot] = len(request_ids)
        # read after staging: an init it ran bumped the version
        self._static_key[cg_slot] = (rids, padded_bs, self._config_version)

    def gather_dynamic(
        self, request_ids: list[str], padded_bs: int, cg_slot: int,
        gather_seen_tokens: bool = True, kept: "frozenset | None" = None,
    ) -> None:
        """Prepare the per-step RNG/history and seen-token mask in ``cg_slot``.
        Must run inline (default stream) after the previous step's writes;
        pre-planning it reads stale state across the plan/default stream boundary.

        Only the H2D is re-issued (the pinned row ``gather_static`` staged for
        this slot is stream-agnostic, and this step's batch is the one it was
        staged for) — restaging is the fallback for a caller that reached here
        without a matching ``gather_static``. With ``ingraph_scatter`` the
        scatter index for this step's ``kept`` rows is staged here, on the
        step it is for, and goes up in the same copy.

        The seen-token mask is large ([bs, V] bool); only gathered when the
        caller's graph actually applies the penalty in-graph (the Talker)."""
        self.rows.sync()
        if self._staged_rids.get(cg_slot) != (tuple(request_ids), padded_bs):
            self._stage_slot_idx(request_ids, padded_bs, cg_slot)
        if self.ingraph_scatter:
            self._stage_scatter_idx(request_ids, padded_bs, cg_slot, kept)
        idx_view = self._upload_slot_idx(padded_bs, cg_slot)
        if not self.ingraph_scatter:
            # with it on, the captured sample gathers off the master itself
            self._gather_state(idx_view, padded_bs, cg_slot)
        if not self.ingraph_scatter and self.seen_tokens is not None and gather_seen_tokens:
            self.rows.read_seen(idx_view, out=self.seen_tokens[cg_slot, :padded_bs])

    def resident_sampler_for(
        self, key, cg_slot: int, bs: int, scalars: Mapping[str, torch.Tensor], index: torch.Tensor,
    ) -> "AcceleratorGraphableSampler":
        """``sampler_for`` a resident capture ``key``: its per-step scalars and
        index row are its own, filled by its graph and its resident stage."""
        sampler = self._resident_samplers.get((key, cg_slot))
        if sampler is None:
            sampler = AcceleratorGraphableSampler(**self.slice_for_bs(bs, cg_slot, scalars=scalars, index=index))
            self._resident_samplers[(key, cg_slot)] = sampler
        return sampler

    def attach_cells(self, cells: "SamplerCells") -> None:
        """Keep ``cells`` with every registration and config update from now on,
        and give them the requests registered so far."""
        self.cells = cells
        for rid, cfg in self._cached_config.items():
            cells.set_config(rid, cfg, windowed=self.window is not None)

    def _gather_state(self, index: torch.Tensor, padded_bs: int, cg_slot: int) -> None:
        torch.index_select(self.rows.offset, 0, index, out=self.offset[cg_slot, :padded_bs])
        if self.history is not None:
            self.history.gather(index, self.history_count[cg_slot, :padded_bs],
                                self.history_tokens[cg_slot, :padded_bs] if self.history_tokens is not None else None)

    def sampler_for(self, padded_bs: int, cg_slot: int) -> "AcceleratorGraphableSampler":
        """A sampler bound to ``cg_slot``'s per-step buffer views (zero-copy).
        Valid regardless of when the buffers are (re)gathered into.

        Cached per (bucket, slot): the per-step buffers are allocated once and
        never rebound, so the views are the same every step, and the sampler
        itself keeps no per-step state the next step could read."""
        key = (padded_bs, cg_slot)
        sampler = self._samplers.get(key)
        if sampler is None:
            sampler = AcceleratorGraphableSampler(**self.slice_for_bs(padded_bs, cg_slot))
            self._samplers[key] = sampler
        return sampler

    def _scatter_idx(self, cg_slot: int) -> tuple[torch.Tensor, int]:
        """The host-planned slot's index narrowed to its real rows."""
        real_bs = self._last_real_bs[cg_slot]
        key = (cg_slot, real_bs)
        idx = self._scatter_idx_views.get(key)
        if idx is None:
            idx = self._scatter_idx_views[key] = self._slot_idx_gpu[cg_slot, :real_bs]
        return idx, real_bs

    def keep_mask(self, keep: list[bool] | None) -> torch.Tensor | None:
        """``keep`` (request order) on the device for ``scatter_*``; None when
        every row is kept."""
        if keep is None or all(keep):
            return None
        return to_device_nonblocking(keep, self._slot_idx_gpu.device, dtype=torch.bool)

    def scatter_state(
        self, cg_slot: int = 0, keep: torch.Tensor | None = None,
        seen_tokens: bool = True,
    ) -> None:
        """Persist scratch after a host-scattered replay, using its capture's index."""
        idx, real_bs = self._scatter_idx(cg_slot)
        offsets = self.offset[cg_slot, :real_bs]
        if keep is not None:
            offsets = torch.where(keep, offsets, self.rows.offset.index_select(0, idx))
        self.rows.offset.index_copy_(0, idx, offsets)
        if self.history is not None:
            count = self.history_count[cg_slot, :real_bs]
            tokens = None if self.history_tokens is None else self.history_tokens[cg_slot, :real_bs]
            if keep is not None:
                old_tokens, old_count = self.history.read(idx)
                count = torch.where(keep, count, old_count)
                if tokens is not None:
                    tokens = torch.where(keep[:, None], tokens, old_tokens)
            self.history.write(idx, tokens, count)
        if seen_tokens and self.seen_tokens is not None:
            self.rows.write_seen(idx, self.seen_tokens[cg_slot, :real_bs], keep)
