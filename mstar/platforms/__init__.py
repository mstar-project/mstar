"""Which hardware platform this process runs on.

``current_platform`` says whether the process drives NVIDIA (CUDA), AMD
(ROCm), Intel (XPU) or no accelerator, and the GPU's architecture. Code that
branches on hardware asks it rather than reading ``torch.version.hip``, the
device type, or a raw compute capability. Which kernel to run stays the
choice of the resource that runs it; the platform only says what the
hardware is.
"""

from __future__ import annotations

import functools
from typing import cast

import torch

from mstar.platforms.interface import (
    CpuPlatform,
    CudaPlatform,
    Platform,
    RocmPlatform,
    XpuPlatform,
)

__all__ = [
    "CpuPlatform",
    "CudaPlatform",
    "Platform",
    "RocmPlatform",
    "XpuPlatform",
    "current_platform",
    "resolve_platform",
]


@functools.cache
def resolve_platform() -> Platform:
    """The stack whose devices this process can see, decided on first use."""
    if torch.cuda.is_available():
        return RocmPlatform() if torch.version.hip is not None else CudaPlatform()
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return XpuPlatform()
    return CpuPlatform()


class _CurrentPlatform:
    """Forwards to ``resolve_platform()``, so importing this probes no device."""

    def __getattr__(self, name: str):
        return getattr(resolve_platform(), name)

    def __repr__(self) -> str:
        return repr(resolve_platform())


current_platform = cast(Platform, _CurrentPlatform())
