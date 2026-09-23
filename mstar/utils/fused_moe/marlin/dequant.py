"""A layer's experts back to bf16 from Marlin's tiles, one launch, at memory speed.

Marlin dequantizes its 4-bit weights inside the GEMM's inner loop, once per 64-row block of activations,
so a prefill of thousands of tokens dequantizes every expert tens of times over and the kernel runs
compute-bound at a quarter of the tensor cores' rate (plan section 11, 2026-09-21). For those steps the
routed experts run as bf16 grouped GEMMs on weights this kernel unpacks once per layer: about 0.9 ms a
layer for the 1.85 GB it writes (924 M weights a rank) against the 2 to 4 ms the in-loop dequantization
costs Marlin at 4k to 8k tokens.

One program per Marlin tile (16 rows of K by 64 of N): its 1024 codes sit in 128 consecutive int32s of
the repacked tensor (``layout.py`` derives the position of each weight inside the block, the same for
every block), so the program loads the block once, picks each weight's nibble through that table, decodes
E2M1 (sign, two exponent bits, one mantissa bit), scales it by the row's E8M0 group scale (also through
a table) and stores the tile with its K axis contiguous. Bit-identical to ``dequant_mxfp4`` of the
original codes (fp32 products rounded once to bf16).
"""
from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from mstar.utils.fused_moe.marlin.layout import code_positions, scale_positions

TILE_K, TILE_N = 16, 64  # one Marlin block: 1024 codes in 128 int32s


@functools.lru_cache(maxsize=None)
def _block_tables(size_k: int, size_n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``fwd [TILE_N, TILE_K]`` int32: the nibble position inside the 128-word block of weight
    ``(n, k)`` of a tile (the same for every tile); ``spos [K/32, N]`` int32: the scale positions."""
    pos = code_positions(size_k, size_n)  # [K, N]
    fwd = pos[:TILE_K, :TILE_N].t().contiguous()  # positions inside block 0 of tile row 0: < 1024
    assert int(fwd.max()) < 1024 and int(fwd.min()) >= 0
    # every tile is the same block layout shifted by its block's first nibble
    check = pos[TILE_K: 2 * TILE_K, TILE_N: 2 * TILE_N].t() - fwd
    assert int(check.min()) == int(check.max()), "Marlin's block layout is not shift-invariant here"
    return fwd.to(device), scale_positions(size_k, size_n).to(device)


@triton.jit
def _dequant_marlin_kernel(
    w, s, fwd, spos, out,
    N, K, words_per_row, groups,
    TILE_K: tl.constexpr, TILE_N: tl.constexpr, TILES: tl.constexpr,
):
    """One program: ``TILES`` consecutive K tiles of one N block of one expert, a tile at a time, so each
    output row receives ``TILES x 32`` contiguous bytes from the program (decoding the tiles as one wide
    tile spilled registers and ran slower)."""
    pid = tl.program_id(0).to(tl.int64)
    n_blocks = N // TILE_N
    k_tiles = K // TILE_K
    k_groups = k_tiles // TILES
    e = pid // (k_groups * n_blocks)
    rem = pid % (k_groups * n_blocks)
    rg = rem // n_blocks  # the group of K tiles
    b = rem % n_blocks  # the N block
    nn = tl.arange(0, TILE_N)[:, None]
    kk = tl.arange(0, TILE_K)[None, :]
    p = tl.load(fwd + nn * TILE_K + kk)  # [TILE_N, TILE_K] nibble positions inside a block
    n_idx = b * TILE_N + tl.arange(0, TILE_N)
    out_rows = out + (e * N + n_idx[:, None]) * K
    for t in tl.static_range(TILES):
        r = rg * TILES + t
        block = w + (e * k_tiles + r) * words_per_row + b * (TILE_K * TILE_N // 8)
        word = tl.load(block + p // 8)
        code = (word >> (4 * (p % 8))) & 0xF
        mag = code & 7
        ex = mag >> 1
        man = (mag & 1).to(tl.float32)
        val = tl.where(ex == 0, 0.5 * man, (1.0 + 0.5 * man) * tl.exp2((ex - 1).to(tl.float32)))
        val = tl.where((code >> 3) == 1, -val, val)
        g = (r * TILE_K) // 32  # a scale group spans two K tiles
        sp = tl.load(spos + g * N + n_idx)
        bits = tl.load(s + e * (groups * N) + sp).to(tl.float32)
        y = val * tl.exp2(bits - 127.0)[:, None]
        tl.store(out_rows + r * TILE_K + kk, y.to(out.dtype.element_ty))


def dequant_marlin_experts(w_marlin: torch.Tensor, s_marlin: torch.Tensor, size_n: int, size_k: int,
                           out: torch.Tensor | None = None) -> torch.Tensor:
    """``w_marlin [E, K/16, 2N]`` int32 and ``s_marlin [E, K/32, N]`` E8M0 (``repack_experts`` /
    ``prepare_scales``) -> ``[E, N, K]`` bf16 (into ``out`` when given)."""
    e = w_marlin.shape[0]
    assert w_marlin.shape[1:] == (size_k // TILE_K, 2 * size_n) and w_marlin.dtype == torch.int32, w_marlin.shape
    assert s_marlin.shape[1:] == (size_k // 32, size_n), s_marlin.shape
    assert size_k % 32 == 0 and size_n % TILE_N == 0
    fwd, spos = _block_tables(size_k, size_n, w_marlin.device)
    if out is None:
        out = torch.empty(e, size_n, size_k, dtype=torch.bfloat16, device=w_marlin.device)
    assert out.shape == (e, size_n, size_k) and out.is_contiguous()
    k_tiles = size_k // TILE_K
    tiles = 8  # 224 tiles (K 3584) and 24 (K 384) both divide by 8; 1.57 ms a layer at the per-rank shapes
    while k_tiles % tiles:
        tiles //= 2
    grid = (e * (k_tiles // tiles) * (size_n // TILE_N),)
    _dequant_marlin_kernel[grid](
        w_marlin, s_marlin.view(torch.uint8), fwd, spos, out,
        size_n, size_k, 2 * size_n, size_k // 32,
        TILE_K=TILE_K, TILE_N=TILE_N, TILES=tiles, num_warps=4,
    )
    return out
