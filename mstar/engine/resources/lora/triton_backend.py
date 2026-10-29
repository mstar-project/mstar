"""Triton LoRA kernels grouped by slot.

The pool lays a step's tokens out grouped by slot (``token_order``, ``slot_indptr``), so one
program tile multiplies tokens of many requests that share an adapter against that
adapter's weights, and decode rows do not each pad a tile on their own.

Programs are laid out over compact row tiles: tile ``i`` belongs to the slot whose tiles
cover it, counting ``cdiv(rows, BLOCK_M)`` tiles for each slot that has rows and a nonzero
rank, slot after slot (``_locate``). So base-model rows (the zero adapter) and slots with
nothing for this target launch no work beyond reading the slot table. The grid depends
only on the token count, the slot count and the shapes, never on the batch's contents
(there are never more tiles than ``min(tokens, cdiv(tokens, BLOCK_M) + slots - 1)``), so a
captured replay launches the same grid every step.

At decode width a shrink has a few row tiles and a long contraction, so ``lora_apply_``
splits the contraction over programs into fp32 partials that its expand sums when it
loads them. The rank is held in tiles of at most 64: when the max rank fits one tile the
expand loads a row tile's ``h`` once and walks a strip of output columns with it; above
that it reloads ``h`` per column block, one rank tile at a time, and only the tiles under
the slot's own rank.

These are the CUDA kernels of the ``mstar::lora_shrink`` / ``mstar::lora_expand_`` /
``mstar::lora_apply_`` custom ops (``ops.py``), so ``torch.compile`` treats each as one
opaque node.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from mstar.engine.resources.lora.ops import LoraOpsBackend, lora_apply_, lora_expand_, lora_shrink

# fused slices the expand kernel takes in one launch (q/k/v, gate/up, a delta-net's
# q / k / v / z / a / b)
_MAX_SLICES = 8
_BLOCK_M = 16
# a program holds at most this many of a slot's rank at once; larger ranks are tiled
_MAX_BLOCK_R = 64
_SHRINK_BLOCK_K = 128
_EXPAND_BLOCK_N = 64
# a split-K shrink gives each program about this much of the contraction, in at most
# this many splits; only when the row tiles alone would not fill the device
_SPLIT_K_CHUNK = 256
_MAX_SPLITS = 16


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
    tile of ``BLOCK_R`` of the rank per program."""
    pid_m = tl.program_id(0)
    slice_id = tl.program_id(1) // R_TILES
    r0 = (tl.program_id(1) % R_TILES) * BLOCK_R
    split = tl.program_id(2)
    slot, m0, end, rank, valid = _locate(pid_m, indptr_ptr, ranks_ptr, NUM_SLOTS, SLOTS_P2, BLOCK_M)
    if not valid:
        return
    if r0 >= rank:
        return
    offs_m = m0 + tl.arange(0, BLOCK_M)
    m_mask = offs_m < end
    rows = tl.load(order_ptr + offs_m, mask=m_mask, other=0).to(tl.int64)
    offs_n = r0 + tl.arange(0, BLOCK_R)
    n_mask = offs_n < rank
    cols = slice_id * max_rank + offs_n
    k_lo = split * k_per_split
    k_hi = tl.minimum(k_lo + k_per_split, in_features)
    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
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
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _bound(i, b0, b1, b2, b3, b4, b5, b6, b7, b8):
    """``(b0, ..., b8)[i]`` for a runtime ``i``: slice ``i`` starts at bound ``i``."""
    return tl.where(i == 0, b0, tl.where(i == 1, b1, tl.where(i == 2, b2, tl.where(
        i == 3, b3, tl.where(i == 4, b4, tl.where(i == 5, b5, tl.where(
            i == 6, b6, tl.where(i == 7, b7, b8))))))))


@triton.jit
def _load_h(h_ptr, rows, m_mask, col0, r0, rank, stride_hs, stride_hm, stride_hn,
            SPLITS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr):
    """``h`` over ranks ``[r0, r0 + BLOCK_R)`` of a slice starting at column ``col0``, its
    split partials summed in fp32."""
    offs_k = r0 + tl.arange(0, BLOCK_R)
    ptrs = h_ptr + rows[:, None] * stride_hm + (col0 + offs_k)[None, :] * stride_hn
    mask = m_mask[:, None] & (offs_k < rank)[None, :]
    h = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    for split in tl.static_range(SPLITS):
        h += tl.load(ptrs + split * stride_hs, mask=mask, other=0.0).to(tl.float32)
    return h


@triton.jit
def _load_b(b_ptr, slot, offs_n, n_mask, r0, rank, stride_bs, stride_bn, stride_bk, BLOCK_R: tl.constexpr):
    """``b[slot]`` over ranks ``[r0, r0 + BLOCK_R)`` and columns ``offs_n``, as ``[BLOCK_R, N]``."""
    offs_k = r0 + tl.arange(0, BLOCK_R)
    return tl.load(
        b_ptr + slot * stride_bs + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk,
        mask=(offs_k < rank)[:, None] & n_mask[None, :], other=0.0,
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
):
    """``y[t, slice] += scaling * (sum over splits of h[split, t, slice ranks]) @ b^T``,
    each program over a strip of the slice's columns."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    slice_id = tl.program_id(2)
    n_programs = tl.num_programs(1)
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
    if R_TILES == 1:
        # the whole rank in one tile: its h is loaded once, for every column strip
        h = _load_h(h_ptr, rows, m_mask, slice_id * max_rank, 0, rank, stride_hs, stride_hm, stride_hn,
                    SPLITS, BLOCK_M, BLOCK_R).to(b_ptr.dtype.element_ty)
    for n0 in range(lo + pid_n * BLOCK_N, hi, n_programs * BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < hi
        if R_TILES == 1:
            acc = tl.dot(h, _load_b(b_ptr, slot, offs_n, n_mask, 0, rank, stride_bs, stride_bn, stride_bk, BLOCK_R))
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            # bounded by the slot's rank, not max_rank: a small adapter in a large-rank pool
            # pays for its own rank
            for r0 in range(0, rank, BLOCK_R):
                h = _load_h(h_ptr, rows, m_mask, slice_id * max_rank, r0, rank,
                            stride_hs, stride_hm, stride_hn, SPLITS, BLOCK_M, BLOCK_R)
                acc += tl.dot(
                    h.to(b_ptr.dtype.element_ty),
                    _load_b(b_ptr, slot, offs_n, n_mask, r0, rank, stride_bs, stride_bn, stride_bk, BLOCK_R),
                )
        y_ptrs = y_ptr + rows[:, None] * stride_ym + offs_n[None, :] * stride_yn
        mask = m_mask[:, None] & n_mask[None, :]
        y = tl.load(y_ptrs, mask=mask, other=0.0)
        tl.store(y_ptrs, (y.to(tl.float32) + acc * scaling).to(y_ptr.dtype.element_ty), mask=mask)


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _row_tiles(tokens: int, slots: int) -> int:
    """The most compact row tiles any grouping of ``tokens`` over ``slots`` needs."""
    return min(tokens, triton.cdiv(tokens, _BLOCK_M) + slots - 1)


def _rank_tiles(max_rank: int) -> tuple[int, int]:
    """``(BLOCK_R, tiles)``: the rank in tiles of at most ``_MAX_BLOCK_R`` (tl.dot contracts
    over at least 16), so a large max_rank does not hold its whole rank in one tile."""
    block = min(max(16, triton.next_power_of_2(max_rank)), _MAX_BLOCK_R)
    return block, triton.cdiv(max_rank, block)


def _shrink_into(h, x, a, token_order, slot_indptr, ranks, num_slices, splits, k_per_split):
    slots, width, _ = a.shape
    block_r, r_tiles = _rank_tiles(width // num_slices)
    _lora_shrink_kernel[(_row_tiles(x.shape[0], slots), num_slices * r_tiles, splits)](
        x, a, h, token_order, slot_indptr, ranks,
        x.shape[1], width // num_slices, k_per_split,
        x.stride(0), x.stride(1),
        a.stride(0), a.stride(1), a.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        NUM_SLOTS=slots, SLOTS_P2=triton.next_power_of_2(slots),
        BLOCK_M=_BLOCK_M, BLOCK_R=block_r, R_TILES=r_tiles, BLOCK_K=_SHRINK_BLOCK_K,
    )


def _expand_from(y, h, b, token_order, slot_indptr, ranks, scalings, out_slices):
    """``h`` is ``[splits, tokens, slices * R]``, summed over its splits."""
    if len(out_slices) > _MAX_SLICES:
        raise ValueError(f"at most {_MAX_SLICES} fused slices; got {len(out_slices)}")
    slots, _, max_rank = b.shape
    bounds = [0]
    for width in out_slices:
        bounds.append(bounds[-1] + width)
    bounds += [bounds[-1]] * (_MAX_SLICES + 1 - len(bounds))
    row_tiles = _row_tiles(y.shape[0], slots)
    block_r, r_tiles = _rank_tiles(max_rank)
    # enough strips to fill the device a few times over when the row tiles do not
    strips = max(1, min(
        triton.cdiv(max(out_slices), _EXPAND_BLOCK_N),
        4 * _num_sms(y.device.index) // (row_tiles * len(out_slices)),
    ))
    _lora_expand_kernel[(row_tiles, strips, len(out_slices))](
        h, b, y, token_order, slot_indptr, ranks, scalings,
        max_rank, *bounds,
        h.stride(0), h.stride(1), h.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        y.stride(0), y.stride(1),
        NUM_SLOTS=slots, SLOTS_P2=triton.next_power_of_2(slots), SPLITS=h.shape[0],
        BLOCK_M=_BLOCK_M, BLOCK_N=_EXPAND_BLOCK_N, BLOCK_R=block_r, R_TILES=r_tiles,
        # a tiled rank reloads its h tiles per column strip: pipelining those loads would
        # stage every split of every tile in shared memory
        **({"num_stages": 1} if r_tiles > 1 else {}),
    )


def _shrink_cuda(x, a, token_order, slot_indptr, ranks, num_slices):
    tokens, in_features = x.shape
    h = x.new_empty(tokens, a.shape[1])
    if tokens == 0:
        return h
    _shrink_into(h.unsqueeze(0), x, a, token_order, slot_indptr, ranks, num_slices, 1, in_features)
    return h


def _expand_cuda(y, h, b, token_order, slot_indptr, ranks, scalings, out_slices):
    if y.shape[0] == 0:
        return
    _expand_from(y, h.unsqueeze(0), b, token_order, slot_indptr, ranks, scalings, out_slices)


def _apply_cuda(y, x, a, b, token_order, slot_indptr, ranks, scalings, out_slices):
    tokens, in_features = x.shape
    if tokens == 0:
        return
    slots, width, _ = a.shape
    num_slices = len(out_slices)
    splits = 1
    if _row_tiles(tokens, slots) * num_slices < _num_sms(x.device.index):
        splits = max(1, min(_MAX_SPLITS, triton.cdiv(in_features, _SPLIT_K_CHUNK)))
    k_per_split = triton.cdiv(triton.cdiv(in_features, splits), _SHRINK_BLOCK_K) * _SHRINK_BLOCK_K
    splits = triton.cdiv(in_features, k_per_split)
    # fp32 partials, written only for rows the expand reads, so never cleared
    h = torch.empty(splits, tokens, width, dtype=torch.float32, device=x.device)
    _shrink_into(h, x, a, token_order, slot_indptr, ranks, num_slices, splits, k_per_split)
    _expand_from(y, h, b, token_order, slot_indptr, ranks, scalings, out_slices)


lora_shrink.register_kernel("cuda")(_shrink_cuda)
lora_expand_.register_kernel("cuda")(_expand_cuda)
lora_apply_.register_kernel("cuda")(_apply_cuda)


class TritonLoraBackend(LoraOpsBackend):
    """The custom ops with the kernels above registered for CUDA."""
