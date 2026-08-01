"""JIT build + load of the vendored Marlin W4A16 CUDA ops."""
from __future__ import annotations

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_CSRC = os.path.join(os.path.dirname(__file__), "csrc")
_MARLIN = os.path.join(_CSRC, "libtorch_stable", "quantization", "marlin")
_MARLIN_MOE = os.path.join(_CSRC, "libtorch_stable", "moe", "marlin_moe_wna16")

_SOURCES = [
    os.path.join(_MARLIN, "gptq_marlin_repack.cu"),
    os.path.join(_MARLIN_MOE, "marlin_moe.cu"),
    os.path.join(_MARLIN_MOE, "sm80_kernel_bfloat16_u4b8_bfloat16.cu"),
    os.path.join(_MARLIN_MOE, "sm80_kernel_float16_u4b8_float16.cu"),
]

_MIN_CAPABILITY = (8, 0)


@functools.lru_cache(maxsize=1)
def is_marlin_available() -> bool:
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
        _ = torch.ops._mstar_marlin_C.moe_wna16_marlin_gemm
        return True
    except Exception as e:  # pragma: no cover -- depends on the build toolchain
        logger.warning(
            "Marlin W4A16: could not build the CUDA ops (%s); using the Triton "
            "in-kernel-dequant fallback.",
            e,
        )
        return False
