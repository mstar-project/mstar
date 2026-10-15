"""Device capabilities shared by workers, graph runners, and staging utilities."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache

import torch

from mstar.engine.accelerator_graph_backend import (
    AcceleratorGraphBackend,
    get_accelerator_graph_backend,
)


class DeviceCapability:
    """Runtime operations for one device, without initializing its hardware.

    Kernel dispatch remains with the resource that owns the kernel. This
    abstraction covers streams, events, host staging, and graph capture.
    """

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self._runtime = getattr(torch, self.device.type, None)
        self.graph_backend: AcceleratorGraphBackend | None
        try:
            self.graph_backend = get_accelerator_graph_backend(self.device)
        except ValueError:
            self.graph_backend = None

    @staticmethod
    def of(device: torch.device | str) -> "DeviceCapability":
        return _device_capability(torch.device(device))

    @property
    def is_accelerator(self) -> bool:
        return self.device.type != "cpu" and all(
            hasattr(self._runtime, name)
            for name in ("Stream", "Event", "stream", "is_available")
        )

    @property
    def is_available(self) -> bool:
        return self.is_accelerator and bool(self._runtime.is_available())

    @property
    def pin_memory(self) -> bool:
        return self.is_available

    def is_capturing(self) -> bool:
        return (
            self.graph_backend is not None
            and self.is_available
            and self.graph_backend.is_current_stream_capturing()
        )

    def set_device(self) -> None:
        if self.is_accelerator and self.device.index is not None:
            torch.accelerator.set_device_index(self.device)

    def synchronize(self) -> None:
        if self.is_available:
            torch.accelerator.synchronize(self.device)

    def current_stream(self):
        return torch.accelerator.current_stream(self.device)

    def new_stream(self):
        return self._runtime.Stream(device=self.device)

    def stream_context(self, stream):
        return nullcontext() if stream is None else self._runtime.stream(stream)

    def new_event(self, **kwargs):
        return self._runtime.Event(**kwargs)


@lru_cache(maxsize=None)
def _device_capability(device: torch.device) -> DeviceCapability:
    return DeviceCapability(device)
