"""Device-side helpers the decode kernels share: the counter wait and a row pick.

Counters are plain int32 words in global memory. A producer CTA makes its stores visible with a CTA
barrier and then one `release` atomic at GPU scope; a consumer spins on an `acquire` load of the word,
barriers, and reads the produced data with `.cg` loads (L2, never a stale L1 line).
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def ld_acquire(ptr):
    """One 32-bit acquire load at GPU scope: no read-modify-write traffic while spinning."""
    return tl.inline_asm_elementwise(
        "ld.global.acquire.gpu.b32 $0, [$1];", "=r,l", [ptr],
        dtype=tl.int32, is_pure=False, pack=1,
    )


@triton.jit
def wait_geq(ptr, target):
    """Spin until the counter at ``ptr`` reaches ``target``."""
    v = ld_acquire(ptr)
    while v < target:
        v = ld_acquire(ptr)


@triton.jit
def row(t, bi, b):
    """Row ``b`` of a [BB, N] register tile, as an [N] tensor."""
    return tl.sum(tl.where(bi[:, None] == b, t, 0.0), axis=0)
