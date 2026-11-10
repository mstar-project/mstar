"""Triton LoRA kernels grouped by slot.

The pool lays a step's tokens out grouped by slot (``token_order``, ``slot_indptr``), so one
program tile multiplies tokens of many requests that share an adapter against that
adapter's weights, and decode rows do not each pad a tile on their own.

Programs are laid out over compact row tiles: tile ``i`` belongs to the slot whose tiles
cover it, counting ``cdiv(rows, BLOCK_M)`` tiles for each slot that has rows and a nonzero
rank, slot after slot (``_locate``). So base-model rows (the zero adapter) and slots with
nothing for this target launch no work beyond reading the slot table. The grid depends
only on the token count, the slot count, the shapes, the dtype and the device, never on the
batch's contents (there are never more tiles than ``min(tokens, cdiv(tokens, BLOCK_M) +
slots - 1)``), so a captured replay launches the same grid every step.

At decode width a shrink has a few row tiles and a long contraction, so ``lora_apply_``
splits the contraction over programs into fp32 partials. The rank is held in tiles of at
most 64: when the max rank fits one tile the expand loads a row tile's ``h`` once, summing
the partials, and walks a strip of output columns with it; above that it reloads ``h``
per column block, one rank tile at a time and only the tiles under the slot's own rank,
so a small kernel first sums the partials into one ``h`` (``_lora_reduce_kernel``):
summed inside that loop they would be reloaded for every column block.

The launch config (``_choose``) depends only on the token count, the max rank, the element
size and the device: small row tiles at decode width, where the work is streaming the
adapter weights, larger ones at prefill width, where it is the GEMMs, each falling back to
fewer pipeline stages or smaller tiles when the device's shared memory is too small.

The shrink writes zeros from a slot's rank up to the max rank, so ``lora_apply_``'s expand
masks ``h`` and ``b`` by the max rank, the same for every slot, rather than by the slot's
rank. The max rank is a runtime argument, but Triton specializes integer arguments on
divisibility by 16, so for a max rank that is a multiple of 16 the mask is constant over
runs of contiguous elements and the loads compile to 16-byte ``cp.async`` copies; masked
by a rank loaded from memory (or a max rank off the multiple), they are scalar loads. So
``b[slot, :, rank:]`` is read, against zeros, and must be finite (``ops.py``). The same
specialization on ``in_features`` vectorizes the shrink's loads of ``x`` and ``a``.

These are the CUDA kernels of the ``mstar::lora_shrink`` / ``mstar::lora_expand_`` /
``mstar::lora_apply_`` custom ops (``ops.py``), so ``torch.compile`` treats each as one
opaque node, and of ``mstar::lora_experts_apply_``, an experts target's grouped GEMMs.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from mstar.engine.resources.lora.backend import experts_reference_
from mstar.engine.resources.lora.ops import (
    experts_config,
    lora_apply_,
    lora_expand_,
    lora_experts_apply_,
    lora_shrink,
)
from mstar.utils.fused_moe.kernels import invoke_fused_moe_kernel
from mstar.utils.fused_moe.runner import tl_compute_type

# fused slices the expand kernel takes in one launch (q/k/v, gate/up, a delta-net's
# q / k / v / z / a / b)
_MAX_SLICES = 8
# a program holds at most this many of a slot's rank at once; larger ranks are tiled
_MAX_BLOCK_R = 64
# a split-K shrink gives each program at least this much of the contraction, in at most
# this many splits, and splits only until its programs fill the device this many times
_SPLIT_K_CHUNK = 256
_MAX_SPLITS = 16
_SHRINK_WAVES = 2


class _Tiles(NamedTuple):
    """One kernel's launch config: rows per tile, its other tiled extent (the shrink's
    contraction, the expand's output columns) and its software-pipeline stages."""
    m: int
    n: int
    stages: int


class _Config(NamedTuple):
    shrink: _Tiles
    expand: _Tiles


# preferred first: a launch takes the first whose shared memory fits the device's, so an
# fp32 pool or a smaller GPU gets fewer stages, then smaller tiles
_DECODE_SHRINK = (_Tiles(16, 128, 3), _Tiles(16, 128, 2), _Tiles(16, 64, 2), _Tiles(16, 64, 1))
_DECODE_EXPAND = (_Tiles(16, 64, 3), _Tiles(16, 64, 2), _Tiles(16, 64, 1))
_PREFILL_SHRINK = (_Tiles(64, 128, 3), _Tiles(64, 128, 2), _Tiles(64, 64, 2), _Tiles(32, 64, 2), _Tiles(16, 64, 1))
_PREFILL_EXPAND = (_Tiles(64, 128, 3), _Tiles(64, 128, 2), _Tiles(64, 64, 2), _Tiles(32, 64, 2), _Tiles(16, 64, 1))
# from these many tokens, or these many over a tiled rank, the prefill tiles are faster
_PREFILL_TOKENS = 256
_PREFILL_TOKENS_TILED_RANK = 64


def _shrink_smem(t: _Tiles, block_r: int, elsize: int) -> int:
    """An upper bound on the shrink's shared memory: ``stages`` buffers of its two operand
    tiles and one of its output tile (``test_lora_triton_high_rank`` checks it against the
    compiled kernels)."""
    return (t.stages * (t.m * t.n + t.n * block_r) + t.m * block_r) * elsize


def _expand_smem(t: _Tiles, block_r: int, elsize: int) -> int:
    """As ``_shrink_smem``, plus the ``h`` tile a one-tile rank holds across the loop."""
    return (t.stages * (t.m * block_r + block_r * t.n) + t.m * block_r + t.m * t.n) * elsize


@functools.cache
def _smem_limit(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


@functools.cache
def _choose(tokens: int, max_rank: int, elsize: int, smem: int) -> _Config:
    """The launch config: a function of the token count, the max rank, the element size and
    the device, so a capture bucket launches the same grid every replay."""
    block_r = _rank_tiles(max_rank)[0]
    tiled = max_rank > _MAX_BLOCK_R
    prefill = tokens >= (_PREFILL_TOKENS_TILED_RANK if tiled else _PREFILL_TOKENS)
    shrink = _PREFILL_SHRINK if prefill else _DECODE_SHRINK
    expand = _PREFILL_EXPAND if prefill else _DECODE_EXPAND
    # the last candidate when none fits, which then fails to launch as any would
    return _Config(
        next((t for t in shrink if _shrink_smem(t, block_r, elsize) <= smem), shrink[-1]),
        next((t for t in expand if _expand_smem(t, block_r, elsize) <= smem), expand[-1]),
    )


def _tiles(tokens: int, max_rank: int, dtype: torch.dtype, device: torch.device) -> _Config:
    return _choose(tokens, max_rank, dtype.itemsize, _smem_limit(device.index))


@triton.jit
def _locate(pid_m, indptr_ptr, ranks_ptr, NUM_SLOTS: tl.constexpr, SLOTS_P2: tl.constexpr,
            BLOCK_M: tl.constexpr):
    """``(slot, first sorted row, end row, rank, valid)`` of compact row tile ``pid_m``."""
    s = tl.arange(0, SLOTS_P2)
    smask = s < NUM_SLOTS
    lo = tl.load(indptr_ptr + s, mask=smask, other=0)
    hi = tl.load(indptr_ptr + s + 1, mask=smask, other=0)
    rk = tl.load(ranks_ptr + s, mask=smask, other=0)
    tiles = tl.where(rk > 0, (hi - lo + BLOCK_M - 1) // BLOCK_M, 0)
    incl = tl.cumsum(tiles, 0)
    excl = incl - tiles
    hit = (pid_m >= excl) & (pid_m < incl)
    valid = tl.sum(hit.to(tl.int32), 0) > 0
    slot = tl.sum(tl.where(hit, s, 0), 0)
    start = tl.sum(tl.where(hit, lo, 0), 0)
    end = tl.sum(tl.where(hit, hi, 0), 0)
    rank = tl.sum(tl.where(hit, rk, 0), 0)
    first = tl.sum(tl.where(hit, excl, 0), 0)
    return slot, start + (pid_m - first) * BLOCK_M, end, rank, valid


@triton.jit
def _lora_shrink_kernel(
    x_ptr, a_ptr, h_ptr, order_ptr, indptr_ptr, ranks_ptr,
    in_features, max_rank, k_per_split,
    stride_xm, stride_xk, stride_as, stride_ar, stride_ak, stride_hs, stride_hm, stride_hn,
    NUM_SLOTS: tl.constexpr, SLOTS_P2: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr, R_TILES: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """``h[split, t, slice cols] = x[t, split's k] @ a[slot, slice rows, split's k]^T``, one
    tile of ``BLOCK_R`` of the rank per program, and zeros from the slot's rank up to
    ``max_rank``."""
    pid_m = tl.program_id(0)
    slice_id = tl.program_id(1) // R_TILES
    r0 = (tl.program_id(1) % R_TILES) * BLOCK_R
    split = tl.program_id(2)
    slot, m0, end, rank, valid = _locate(pid_m, indptr_ptr, ranks_ptr, NUM_SLOTS, SLOTS_P2, BLOCK_M)
    if not valid:
        return
    offs_m = m0 + tl.arange(0, BLOCK_M)
    m_mask = offs_m < end
    rows = tl.load(order_ptr + offs_m, mask=m_mask, other=0).to(tl.int64)
    offs_n = r0 + tl.arange(0, BLOCK_R)
    n_mask = offs_n < rank
    cols = slice_id * max_rank + offs_n
    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    if r0 < rank:
        k_lo = split * k_per_split
        k_hi = tl.minimum(k_lo + k_per_split, in_features)
        for k0 in range(k_lo, k_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < k_hi
            x = tl.load(
                x_ptr + rows[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )
            a = tl.load(
                a_ptr + slot * stride_as + cols[None, :] * stride_ar + offs_k[:, None] * stride_ak,
                mask=n_mask[None, :] & k_mask[:, None], other=0.0,
            )
            acc += tl.dot(x, a)
    tl.store(
        h_ptr + split * stride_hs + rows[:, None] * stride_hm + cols[None, :] * stride_hn,
        acc.to(h_ptr.dtype.element_ty),
        mask=m_mask[:, None] & (offs_n < max_rank)[None, :],
    )


@triton.jit
def _lora_reduce_kernel(parts_ptr, h_ptr, numel, stride_s, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    """``h = sum over splits of parts[split]``, in fp32 and in split order."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for split in tl.static_range(SPLITS):
        acc += tl.load(parts_ptr + split * stride_s + offs, mask=mask, other=0.0)
    tl.store(h_ptr + offs, acc.to(h_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _bound(i, b0, b1, b2, b3, b4, b5, b6, b7, b8):
    """``(b0, ..., b8)[i]`` for a runtime ``i``: slice ``i`` starts at bound ``i``."""
    return tl.where(i == 0, b0, tl.where(i == 1, b1, tl.where(i == 2, b2, tl.where(
        i == 3, b3, tl.where(i == 4, b4, tl.where(i == 5, b5, tl.where(
            i == 6, b6, tl.where(i == 7, b7, b8))))))))


@triton.jit
def _load_h(h_ptr, rows, m_mask, col0, r0, rank, max_rank, stride_hs, stride_hm, stride_hn,
            SPLITS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr,
            ZEROS_PAST_RANK: tl.constexpr, dtype: tl.constexpr):
    """``h`` over ranks ``[r0, r0 + BLOCK_R)`` of a slice starting at column ``col0``, its
    split partials summed in fp32, as ``dtype``. Loaded up to ``max_rank``; past the slot's
    rank it is zeroed unless ``h`` already holds zeros there (``ZEROS_PAST_RANK``)."""
    offs_k = r0 + tl.arange(0, BLOCK_R)
    ptrs = h_ptr + rows[:, None] * stride_hm + (col0 + offs_k)[None, :] * stride_hn
    mask = m_mask[:, None] & (offs_k < max_rank)[None, :]
    if SPLITS == 1:
        h = tl.load(ptrs, mask=mask, other=0.0)
    else:
        h = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for split in tl.static_range(SPLITS):
            h += tl.load(ptrs + split * stride_hs, mask=mask, other=0.0).to(tl.float32)
    if not ZEROS_PAST_RANK:
        # a select, not a multiply: h past a slot's rank may hold anything
        h = tl.where((offs_k < rank)[None, :], h, 0.0)
    return h.to(dtype)


@triton.jit
def _load_b(b_ptr, slot, offs_n, n_mask, r0, max_rank, stride_bs, stride_bn, stride_bk, BLOCK_R: tl.constexpr):
    """``b[slot]`` over ranks ``[r0, r0 + BLOCK_R)`` and columns ``offs_n``, as ``[BLOCK_R, N]``;
    ranks past the slot's meet zeros in ``h``."""
    offs_k = r0 + tl.arange(0, BLOCK_R)
    return tl.load(
        b_ptr + slot * stride_bs + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk,
        mask=(offs_k < max_rank)[:, None] & n_mask[None, :], other=0.0,
    )


@triton.jit
def _lora_expand_kernel(
    h_ptr, b_ptr, y_ptr, order_ptr, indptr_ptr, ranks_ptr, scalings_ptr,
    max_rank,
    bound0, bound1, bound2, bound3, bound4, bound5, bound6, bound7, bound8,
    stride_hs, stride_hm, stride_hn,
    stride_bs, stride_bn, stride_bk,
    stride_ym, stride_yn,
    NUM_SLOTS: tl.constexpr, SLOTS_P2: tl.constexpr, SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr, R_TILES: tl.constexpr,
    ZEROS_PAST_RANK: tl.constexpr,
):
    """``y[t, slice] += scaling * (sum over splits of h[split, t, slice ranks]) @ b^T``,
    each program over a strip of the slice's columns."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    slice_id = tl.program_id(2)
    n_programs = tl.num_programs(1)
    dtype = b_ptr.dtype.element_ty
    lo = _bound(slice_id, bound0, bound1, bound2, bound3, bound4, bound5, bound6, bound7, bound8)
    hi = _bound(slice_id + 1, bound0, bound1, bound2, bound3, bound4, bound5, bound6, bound7, bound8)
    if lo + pid_n * BLOCK_N >= hi:
        return
    slot, m0, end, rank, valid = _locate(pid_m, indptr_ptr, ranks_ptr, NUM_SLOTS, SLOTS_P2, BLOCK_M)
    if not valid:
        return
    offs_m = m0 + tl.arange(0, BLOCK_M)
    m_mask = offs_m < end
    rows = tl.load(order_ptr + offs_m, mask=m_mask, other=0).to(tl.int64)
    scaling = tl.load(scalings_ptr + slot).to(tl.float32)
    col0 = slice_id * max_rank
    if R_TILES == 1:
        # the whole rank in one tile: its h is loaded once, for every column strip
        h = _load_h(h_ptr, rows, m_mask, col0, 0, rank, max_rank, stride_hs, stride_hm, stride_hn,
                    SPLITS, BLOCK_M, BLOCK_R, ZEROS_PAST_RANK, dtype)
    for n0 in range(lo + pid_n * BLOCK_N, hi, n_programs * BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < hi
        if R_TILES == 1:
            acc = tl.dot(h, _load_b(b_ptr, slot, offs_n, n_mask, 0, max_rank, stride_bs, stride_bn, stride_bk,
                                    BLOCK_R))
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            # bounded by the slot's rank, not max_rank: a small adapter in a large-rank pool
            # pays for its own rank
            for r0 in range(0, rank, BLOCK_R):
                acc += tl.dot(
                    _load_h(h_ptr, rows, m_mask, col0, r0, rank, max_rank, stride_hs, stride_hm, stride_hn,
                            SPLITS, BLOCK_M, BLOCK_R, ZEROS_PAST_RANK, dtype),
                    _load_b(b_ptr, slot, offs_n, n_mask, r0, max_rank, stride_bs, stride_bn, stride_bk, BLOCK_R),
                )
        y_ptrs = y_ptr + rows[:, None] * stride_ym + offs_n[None, :] * stride_yn
        mask = m_mask[:, None] & n_mask[None, :]
        y = tl.load(y_ptrs, mask=mask, other=0.0)
        tl.store(y_ptrs, (y.to(tl.float32) + acc * scaling).to(y_ptr.dtype.element_ty), mask=mask)


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _row_tiles(tokens: int, slots: int, block_m: int) -> int:
    """The most compact row tiles any grouping of ``tokens`` over ``slots`` needs."""
    return min(tokens, triton.cdiv(tokens, block_m) + slots - 1)


def _rank_tiles(max_rank: int) -> tuple[int, int]:
    """``(BLOCK_R, tiles)``: the rank in tiles of at most ``_MAX_BLOCK_R`` (tl.dot contracts
    over at least 16), so a large max_rank does not hold its whole rank in one tile."""
    block = min(max(16, triton.next_power_of_2(max_rank)), _MAX_BLOCK_R)
    return block, triton.cdiv(max_rank, block)


def _shrink_into(h, x, a, token_order, slot_indptr, ranks, num_slices, splits, k_per_split, tiles):
    slots, width, _ = a.shape
    block_r, r_tiles = _rank_tiles(width // num_slices)
    return _lora_shrink_kernel[(_row_tiles(x.shape[0], slots, tiles.m), num_slices * r_tiles, splits)](
        x, a, h, token_order, slot_indptr, ranks,
        x.shape[1], width // num_slices, k_per_split,
        x.stride(0), x.stride(1),
        a.stride(0), a.stride(1), a.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        NUM_SLOTS=slots, SLOTS_P2=triton.next_power_of_2(slots),
        BLOCK_M=tiles.m, BLOCK_R=block_r, R_TILES=r_tiles, BLOCK_K=tiles.n,
        num_stages=tiles.stages,
    )


def _reduce(parts: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``[splits, tokens, width]`` fp32 partials summed into ``[1, tokens, width]``."""
    h = parts.new_empty(1, *parts.shape[1:], dtype=dtype)
    block = 128
    _lora_reduce_kernel[(triton.cdiv(h.numel(), block),)](
        parts, h, h.numel(), parts.stride(0), SPLITS=parts.shape[0], BLOCK=block,
    )
    return h


def _expand_from(y, h, b, token_order, slot_indptr, ranks, scalings, out_slices, tiles, zeros_past_rank):
    """``h`` is ``[splits, tokens, slices * R]``, summed over its splits; ``zeros_past_rank``
    says it holds zeros from each slot's rank up to ``R`` (as ``_lora_shrink_kernel``
    writes it)."""
    if len(out_slices) > _MAX_SLICES:
        raise ValueError(f"at most {_MAX_SLICES} fused slices; got {len(out_slices)}")
    slots, _, max_rank = b.shape
    bounds = [0]
    for width in out_slices:
        bounds.append(bounds[-1] + width)
    bounds += [bounds[-1]] * (_MAX_SLICES + 1 - len(bounds))
    row_tiles = _row_tiles(y.shape[0], slots, tiles.m)
    block_r, r_tiles = _rank_tiles(max_rank)
    # enough strips to fill the device a few times over when the row tiles do not
    strips = max(1, min(
        triton.cdiv(max(out_slices), tiles.n),
        4 * _num_sms(y.device.index) // (row_tiles * len(out_slices)),
    ))
    return _lora_expand_kernel[(row_tiles, strips, len(out_slices))](
        h, b, y, token_order, slot_indptr, ranks, scalings,
        max_rank, *bounds,
        h.stride(0), h.stride(1), h.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        y.stride(0), y.stride(1),
        NUM_SLOTS=slots, SLOTS_P2=triton.next_power_of_2(slots), SPLITS=h.shape[0],
        BLOCK_M=tiles.m, BLOCK_N=tiles.n, BLOCK_R=block_r, R_TILES=r_tiles,
        ZEROS_PAST_RANK=zeros_past_rank, num_stages=tiles.stages,
    )


def _shrink_cuda(x, a, token_order, slot_indptr, token_slot, ranks, num_slices):
    tokens, in_features = x.shape
    h = x.new_empty(tokens, a.shape[1])
    if tokens == 0:
        return h
    _shrink_into(h.unsqueeze(0), x, a, token_order, slot_indptr, ranks, num_slices, 1, in_features,
                 _tiles(tokens, a.shape[1] // num_slices, x.dtype, x.device).shrink)
    return h


def _expand_cuda(y, h, b, token_order, slot_indptr, token_slot, ranks, scalings, out_slices):
    if y.shape[0] == 0:
        return
    _expand_from(y, h.unsqueeze(0), b, token_order, slot_indptr, ranks, scalings, out_slices,
                 _tiles(y.shape[0], b.shape[2], b.dtype, b.device).expand, zeros_past_rank=False)


def _apply_cuda(y, x, a, b, token_order, slot_indptr, token_slot, ranks, scalings, out_slices):
    tokens, in_features = x.shape
    if tokens == 0:
        return
    slots, width, _ = a.shape
    num_slices = len(out_slices)
    tiles = _tiles(tokens, width // num_slices, x.dtype, x.device)
    r_tiles = _rank_tiles(width // num_slices)[1]
    programs = _row_tiles(tokens, slots, tiles.shrink.m) * num_slices * r_tiles
    splits = max(1, min(_MAX_SPLITS, triton.cdiv(in_features, _SPLIT_K_CHUNK),
                        triton.cdiv(_SHRINK_WAVES * _num_sms(x.device.index), programs)))
    k_per_split = triton.cdiv(triton.cdiv(in_features, splits), tiles.shrink.n) * tiles.shrink.n
    splits = triton.cdiv(in_features, k_per_split)
    # partials, written only for rows the expand reads, so never cleared; one split is
    # written in the weights' dtype, as the expand would round it
    h = torch.empty(splits, tokens, width, dtype=torch.float32 if splits > 1 else x.dtype, device=x.device)
    _shrink_into(h, x, a, token_order, slot_indptr, ranks, num_slices, splits, k_per_split, tiles.shrink)
    if splits > 1 and r_tiles > 1:
        h = _reduce(h, x.dtype)
    _expand_from(y, h, b, token_order, slot_indptr, ranks, scalings, out_slices, tiles.expand,
                 zeros_past_rank=True)


lora_shrink.register_kernel("cuda")(_shrink_cuda)
lora_expand_.register_kernel("cuda")(_expand_cuda)
lora_apply_.register_kernel("cuda")(_apply_cuda)



def _experts_apply_cuda(x, y, a, b, ids, row_weight, sorted_ids, expert_ids, num_post_padded, top_k,
                        out_slices, block_m):
    """An experts target on the fused-MoE grouped GEMM (``invoke_fused_moe_kernel``) over
    combined ids: one align per MoE layer (``sorted_ids`` .. ``block_m``) groups its rows by
    (slot, expert), a shrink writes ``h``, and an expand per slice adds into the layer's
    cache (``accumulate``) scaled per row. One opaque op for ``torch.compile``: traced as
    raw Triton launches into column views of ``y``, compiled output differed from eager
    and graph capture crashed."""
    if sorted_ids is None:  # no alignment: the rows were never grouped
        experts_reference_(x, y, a, b, ids, row_weight, top_k, out_slices)
        return
    if y.shape[0] == 0:
        return
    config = experts_config(ids, a.shape[0])
    assert config["BLOCK_SIZE_M"] == block_m
    launch = dict(
        topk_ids=ids, sorted_token_ids=sorted_ids, expert_ids=expert_ids,
        num_tokens_post_padded=num_post_padded, config=config, compute_type=tl_compute_type(x.dtype),
    )
    max_rank = b.shape[-1]
    h = x.new_empty(y.shape[0], a.shape[1])
    invoke_fused_moe_kernel(x, a, h, row_weight, mul_routed_weight=False, top_k=top_k, **launch)
    offset = 0
    for i, width in enumerate(out_slices):
        invoke_fused_moe_kernel(
            h[:, i * max_rank:(i + 1) * max_rank], b[:, offset:offset + width], y[:, offset:offset + width],
            row_weight, mul_routed_weight=True, top_k=1, accumulate=True, **launch,
        )
        offset += width


lora_experts_apply_.register_kernel("cuda")(_experts_apply_cuda)
