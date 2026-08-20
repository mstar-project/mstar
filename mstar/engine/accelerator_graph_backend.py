from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol

import torch


class CapturedGraph(Protocol):
    def replay(self) -> None: ...

    def reset(self) -> None: ...

    def pool(self) -> Any: ...


class AcceleratorGraphBackend:
    """Device-specific graph capture operations used by generic runners."""

    def __init__(self, device: torch.device):
        if device.type not in {"cuda", "xpu"}:
            raise ValueError(
                f"Accelerator graphs are unsupported on device type {device.type!r}"
            )
        self.device = device
        self.device_module = getattr(torch, device.type)

    @property
    def device_type(self) -> str:
        return self.device.type

    def is_available(self) -> bool:
        return bool(self.device_module.is_available())

    def set_device(self) -> None:
        torch.accelerator.set_device_index(self.device)

    def synchronize(self) -> None:
        torch.accelerator.synchronize(self.device)

    def create_graph(self) -> CapturedGraph:
        graph_type = (
            self.device_module.CUDAGraph
            if self.device_type == "cuda"
            else self.device_module.XPUGraph
        )
        return graph_type()

    def graph_pool_handle(self) -> Any:
        return self.device_module.graph_pool_handle()

    def capture(
        self,
        graph: CapturedGraph,
        *,
        pool: Any = None,
        stream: Any = None,
    ) -> AbstractContextManager:
        kwargs: dict[str, Any] = {"pool": pool}
        if stream is not None:
            kwargs["stream"] = stream
        return self.device_module.graph(graph, **kwargs)

    def new_stream(self):
        return self.device_module.Stream(device=self.device)

    def current_stream(self):
        return torch.accelerator.current_stream(self.device)

    def stream_context(self, stream) -> AbstractContextManager:
        return self.device_module.stream(stream)

    def memory_allocated(self) -> int:
        memory_allocated = getattr(self.device_module, "memory_allocated", None)
        if memory_allocated is None:
            return 0
        return int(memory_allocated(self.device))

    def is_current_stream_capturing(self) -> bool:
        return bool(self.device_module.is_current_stream_capturing())


def get_accelerator_graph_backend(
    device: torch.device,
) -> AcceleratorGraphBackend:
    return AcceleratorGraphBackend(device)
