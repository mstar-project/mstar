"""FlashInfer on AMD GPUs, which means AMD's ROCm fork (``amd-flashinfer``).

The fork keeps upstream FlashInfer's Python API, so a kernel binding written for
NVIDIA often carries over; upstream FlashInfer imports on ROCm too but fails on
its first kernel build, so callers on AMD GPUs import through here to get a
clear error instead.
"""

from types import ModuleType


def import_flashinfer_rocm(purpose: str) -> ModuleType:
    """``flashinfer``, checked to be the ROCm fork; ``purpose`` names the caller."""
    try:
        import flashinfer
        import flashinfer.rocm  # noqa: F401  (only the fork has it)
    except ImportError as e:
        raise RuntimeError(
            f"{purpose} on AMD GPUs needs amd-flashinfer, FlashInfer's ROCm fork "
            "(see docs/installation.rst)"
        ) from e
    return flashinfer
