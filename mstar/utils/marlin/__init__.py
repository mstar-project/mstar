"""Vendored Marlin W4A16 (INT4) CUDA kernels for mstar.

JIT-compiled from Apache-2.0 vLLM sources under ``csrc/`` on first use, with a
Triton fallback (see :mod:`mstar.utils.marlin.loader`). Exposes the repack + GEMM
launchers (:mod:`mstar.utils.marlin.ops`) used by the compressed-tensors W4A16
routed-expert path.
"""
from mstar.utils.marlin.loader import is_marlin_available

__all__ = ["is_marlin_available"]
