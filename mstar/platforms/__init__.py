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
import os
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
    "apply_process_env_defaults",
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
    """Forwards to ``resolve_platform()``, so importing this probes no device.

    ``__class__`` is forwarded too, as ``unittest.mock`` does, so
    ``isinstance(current_platform, RocmPlatform)`` holds on ROCm.
    """

    @property
    def __class__(self):
        return type(resolve_platform())

    def __getattr__(self, name: str):
        return getattr(resolve_platform(), name)

    def __repr__(self) -> str:
        return repr(resolve_platform())

    def __reduce__(self):
        # by name: pickling through the forwarded __class__ would fail
        return "current_platform"


current_platform = cast(Platform, _CurrentPlatform())


def _built_for() -> Platform:
    """The platform torch was built for, which takes no device call to know."""
    if torch.version.hip is not None:
        return RocmPlatform()
    if torch.version.cuda is not None:
        return CudaPlatform()
    if getattr(torch.version, "xpu", None) is not None:
        return XpuPlatform()
    return CpuPlatform()


def apply_process_env_defaults() -> None:
    """Set the platform's ``process_env_defaults``, keeping any value already set.

    Call before the process's first device call: the HIP and CUDA runtimes
    read much of their environment when they initialize, and resolving
    ``current_platform`` counts devices, which initializes them. So the
    platform asked here is the one torch was built for. A process that spawns
    workers calls it first too, since a child inherits ``os.environ``.
    """
    for key, value in _built_for().process_env_defaults().items():
        os.environ.setdefault(key, value)
