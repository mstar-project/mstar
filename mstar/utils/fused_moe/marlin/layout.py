"""Where Marlin keeps each weight: the inverse of the repack, as lookup tables.

``repack_experts`` and ``prepare_scales`` turn an expert's MXFP4 codes ``[N, K/2]`` (two per byte) and E8M0
scales ``[N, K/32]`` into Marlin's tile order (vLLM's ``gptq_marlin_repack`` and ``marlin_permute_scales``
followed by the MXFP4 pair swap), and the module's parameters are rebound to those tensors, so the codes
in the checkpoint's order are gone from the GPU. A path that wants the weights as bf16 again (a large
prefill, where a bf16 grouped GEMM beats Marlin's in-loop dequantization) needs to know, for every
weight ``(n, k)``, which nibble of which int32 holds its code and which entry holds its scale. The
permutation is a fixed function of the shape, so it is derived once by pushing index tensors through the
same reshapes and gathers vLLM's Python reference of the repack applies (``marlin_permute_weights`` with
``get_weight_perm(4)`` and the 8-per-int32 packing), and inverted into two tables per shape:

* ``code_pos [K, N]`` int32: the nibble position (``pos // 8`` the int32, ``pos % 8`` the nibble) of
  weight ``(n, k)`` in the flattened repacked tensor;
* ``scale_pos [K/32, N]`` int32: the position of group ``k // 32`` of row ``n`` in the flattened
  processed scale tensor (E8M0 bits).

``dequant_marlin_reference`` applies them in torch (the check against ``dequant_mxfp4`` of the original
codes is bit-exact); the Triton kernel in ``dequant.py`` does the same per layer at bandwidth speed.
"""
from __future__ import annotations

import functools

import torch

TILE = 16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def weight_perm_4bit() -> torch.Tensor:
    """vLLM's ``get_weight_perm(4)``: the 1024-entry column permutation inside each 16 x 64 tile row."""
    perm: list[int] = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in (0, 1):
            for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1):
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend(p + 256 * j for p in perm1)
    p = torch.tensor(perm, dtype=torch.long)
    interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    return p.view(-1, 8)[:, interleave].reshape(-1)


def _permute_weights(q: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    """vLLM's ``marlin_permute_weights`` on a ``[size_k, size_n]`` tensor of anything (indices here)."""
    perm = weight_perm_4bit()
    q = q.reshape(size_k // TILE, TILE, size_n // TILE, TILE).permute(0, 2, 1, 3).reshape(size_k // TILE, size_n * TILE)
    return q.reshape(-1, perm.numel())[:, perm].reshape(size_k // TILE, size_n * TILE)


@functools.lru_cache(maxsize=None)
def code_positions(size_k: int, size_n: int) -> torch.Tensor:
    """``[K, N]`` int32 (CPU): the nibble position of weight ``(n, k)`` in the repacked int32 tensor."""
    origin = torch.arange(size_k * size_n, dtype=torch.long).view(size_k, size_n)  # GPTQ order: [K, N]
    permuted = _permute_weights(origin, size_k, size_n)  # position (r, c) holds weight permuted[r, c]
    table = torch.empty(size_k * size_n, dtype=torch.long)
    table[permuted.reshape(-1)] = torch.arange(permuted.numel(), dtype=torch.long)
    return table.view(size_k, size_n).to(torch.int32)


@functools.lru_cache(maxsize=None)
def scale_positions(size_k: int, size_n: int, group: int = 32) -> torch.Tensor:
    """``[K/group, N]`` int32 (CPU): the position of the scale of group ``g`` of row ``n`` in the flattened
    processed scale tensor (``marlin_permute_scales`` then the MXFP4 pair swap)."""
    groups = size_k // group
    s = torch.arange(groups * size_n, dtype=torch.long).view(groups, size_n)
    perm = torch.tensor([i + 8 * j for i in range(8) for j in range(8)])
    s = s.reshape(-1, 64)[:, perm].reshape(-1, size_n)
    s = s.view(-1, 4)[:, [0, 2, 1, 3]].reshape(groups, size_n)
    table = torch.empty(groups * size_n, dtype=torch.long)
    table[s.reshape(-1)] = torch.arange(s.numel(), dtype=torch.long)
    return table.view(groups, size_n).to(torch.int32)


def dequant_marlin_reference(w_marlin: torch.Tensor, s_marlin: torch.Tensor, size_n: int, size_k: int,
                             dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """One expert's weights ``[N, K]`` in ``dtype`` from its Marlin tensors (``repack_experts`` /
    ``prepare_scales`` outputs), through the tables: the check for the kernel."""
    dev = w_marlin.device
    pos = code_positions(size_k, size_n).to(dev).long().t()  # [N, K]
    words = w_marlin.reshape(-1)
    codes = (words[pos // 8] >> (4 * (pos % 8))) & 0xF
    vals = torch.tensor(E2M1, dtype=torch.float32, device=dev)[codes.long()]
    spos = scale_positions(size_k, size_n).to(dev).long().t()  # [N, K/32]
    bits = s_marlin.view(torch.uint8).reshape(-1)[spos].to(torch.float32)
    scales = torch.exp2(bits - 127.0)
    return (vals.view(size_n, size_k // 32, 32) * scales.unsqueeze(-1)).view(size_n, size_k).to(dtype)
