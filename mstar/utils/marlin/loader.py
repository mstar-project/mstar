"""JIT build + load of the vendored Marlin W4A16 CUDA ops.

Mirrors :mod:`mstar.utils.fused_moe.align`: the Marlin CUDA sources (vendored
Apache-2.0 from vLLM under ``csrc/``) are JIT-compiled with
``torch.utils.cpp_extension.load`` on first use and registered as
``torch.ops._mstar_marlin_C.*`` — no ``vllm`` / ``sgl_kernel`` runtime dependency.

If the build fails (no ``nvcc`` / no ``ninja`` / sm<80 / ABI mismatch) the loader
logs and returns ``False``; callers fall back to the Triton W4A16 path
(``fused_moe_kernel_w4a16``). Marlin is a *speed* layer over a correctness path
that already exists — exactly the CUDA-op-or-torch-fallback pattern ``align.py``
uses for ``moe_align_block_size``.
"""
from __future__ import annotations

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_CSRC = os.path.join(os.path.dirname(__file__), "csrc")
_MARLIN = os.path.join(_CSRC, "libtorch_stable", "quantization", "marlin")
_MARLIN_MOE = os.path.join(_CSRC, "libtorch_stable", "moe", "marlin_moe_wna16")

# Sources compiled into the ``_mstar_marlin_C`` extension: the repack op, the MoE
# GEMM host+device shim, and the pre-generated per-config kernel instantiations
# (``sm80_kernel_*.cu`` + ``kernel_selector.h``, produced once by the trimmed
# ``generate_kernels.py`` and vendored — GPTQ symmetric INT4, fp16/bf16 only).
_SOURCES = [
    os.path.join(_MARLIN, "gptq_marlin_repack.cu"),
    os.path.join(_MARLIN_MOE, "marlin_moe.cu"),
    os.path.join(_MARLIN_MOE, "sm80_kernel_bfloat16_u4b8_bfloat16.cu"),
    os.path.join(_MARLIN_MOE, "sm80_kernel_float16_u4b8_float16.cu"),
]

# Marlin's device code (cp.async, m16n8k16 MMA, bf16) requires sm80+.
_MIN_CAPABILITY = (8, 0)


@functools.lru_cache(maxsize=1)
def is_marlin_available() -> bool:
    """JIT-build the Marlin ops once per process; return whether they are usable.

    Cached so compilation is attempted at most once. Any failure (missing
    toolchain, unsupported GPU, compile/ABI error) is logged and the caller uses
    the Triton W4A16 fallback.
    """
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    if capability < _MIN_CAPABILITY:
        logger.warning(
            "Marlin W4A16 needs sm%d%d+ (device is sm%d%d); using the Triton "
            "in-kernel-dequant fallback.",
            _MIN_CAPABILITY[0], _MIN_CAPABILITY[1], capability[0], capability[1],
        )
        return False
    try:
        from torch.utils.cpp_extension import load

        load(
            name="_mstar_marlin_C",
            sources=list(_SOURCES),
            is_python_module=False,
            extra_include_paths=[_CSRC],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
            verbose=False,
        )
        # Touch an op so a registration failure surfaces here, not at call time.
        _ = torch.ops._mstar_marlin_C.moe_wna16_marlin_gemm
        return True
    except Exception as e:  # pragma: no cover -- depends on the build toolchain
        logger.warning(
            "Marlin W4A16: could not build the CUDA ops (%s); using the Triton "
            "in-kernel-dequant fallback.",
            e,
        )
        return False
