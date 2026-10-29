"""Which hardware platform this process runs on.

``current_platform`` says whether the process drives NVIDIA (CUDA), AMD
(ROCm), Intel (XPU) or no accelerator, and the GPU's architecture. Code that
branches on the GPU vendor or architecture asks it rather than reading
``torch.version.hip``, the ``cuda`` device type, or a raw compute capability.
``device.type == "cuda"`` is still the way to ask "is this a GPU". Which
kernel to run stays the choice of the resource that runs it; the platform
only says what the hardware is.
"""

from __future__ import annotations

import functools
import os
from typing import cast

import torch

from mstar.platforms.interface import (
    CpuPlatform,
    CudaPlatform,
    HardwareQueueColocation,
    Platform,
    RocmPlatform,
    WorkerColocation,
    XpuPlatform,
)

__all__ = [
    "CpuPlatform",
    "CudaPlatform",
    "HardwareQueueColocation",
    "Platform",
    "RocmPlatform",
    "WorkerColocation",
    "XpuPlatform",
    "apply_process_env_defaults",
    "apply_visible_devices",
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
    """Forwards to ``resolve_platform()``, so importing this touches no device.
    Ask it ``is_rocm()`` and friends rather than ``isinstance``."""

    def __getattr__(self, name: str):
        return getattr(resolve_platform(), name)

    def __repr__(self) -> str:
        return repr(resolve_platform())

    def __reduce__(self):
        # pickle by name so a round trip gives back this singleton
        return "current_platform"


current_platform = cast(Platform, _CurrentPlatform())


def _built_for() -> Platform:
    """The platform torch was built for. Needs no device call."""
    if torch.version.hip is not None:
        return RocmPlatform()
    if torch.version.cuda is not None:
        return CudaPlatform()
    if getattr(torch.version, "xpu", None) is not None:
        return XpuPlatform()
    return CpuPlatform()


def apply_process_env_defaults() -> None:
    """Set the platform's ``process_env_defaults``, keeping any value already set.

    Call it before the first device call, since the HIP and CUDA runtimes read
    much of their environment at init. Resolving ``current_platform`` counts
    devices, which initializes the runtime, so this asks the platform torch was
    built for instead.
    """
    for key, value in _built_for().process_env_defaults().items():
        os.environ.setdefault(key, value)


def apply_visible_devices(devices: str | None) -> None:
    """Limit this process and its children to ``devices`` (comma-separated
    indices). With None, keep the env's choice after the platform checks it.
    Call before the first device call, like apply_process_env_defaults.
    """
    os.environ.update(_built_for().visible_devices_env(devices, os.environ))
