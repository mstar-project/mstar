from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Protocol

import torch


class CapturedGraph(Protocol):
    def replay(self) -> None: ...

    def reset(self) -> None: ...

    def pool(self) -> Any: ...


@dataclass
class AcceleratorGraphPool:
    handle: Any
    failed_graph: CapturedGraph | None = field(default=None, repr=False)


class AcceleratorGraphBackend(ABC):

    def __init__(self, device: torch.device):
        self.device = device

    @property
    def device_type(self) -> str:
        return self.device.type

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def set_device(self) -> None: ...

    @abstractmethod
    def synchronize(self) -> None: ...

    @abstractmethod
    def create_graph(self) -> CapturedGraph: ...

    @abstractmethod
    def graph_pool_handle(self) -> AcceleratorGraphPool: ...

    @abstractmethod
    def capture(
        self,
        graph: CapturedGraph,
        *,
        pool: AcceleratorGraphPool | None = None,
        stream: Any = None,
    ) -> AbstractContextManager: ...

    @abstractmethod
    def new_stream(self) -> Any: ...

    @abstractmethod
    def current_stream(self) -> Any: ...

    @abstractmethod
    def stream_context(self, stream: Any) -> AbstractContextManager: ...

    @abstractmethod
    def memory_allocated(self) -> int: ...

    @abstractmethod
    def is_current_stream_capturing(self) -> bool: ...

    @abstractmethod
    def recover_failed_capture(self, pool: AcceleratorGraphPool, stream: Any) -> None:
        pass


class _TorchGraphBackend(AcceleratorGraphBackend):

    def __init__(
        self,
        device: torch.device,
        *,
        device_type: str,
        runtime: ModuleType,
    ):
        if device.type != device_type:
            raise ValueError(
                f"{type(self).__name__} requires a {device_type!r} device, "
                f"got {device.type!r}"
            )
        super().__init__(device)
        self._runtime = runtime

    def is_available(self) -> bool:
        return bool(self._runtime.is_available())

    def set_device(self) -> None:
        torch.accelerator.set_device_index(self.device)

    def synchronize(self) -> None:
        torch.accelerator.synchronize(self.device)

    def graph_pool_handle(self) -> AcceleratorGraphPool:
        return AcceleratorGraphPool(self._runtime.graph_pool_handle())

    def capture(
        self,
        graph: CapturedGraph,
        *,
        pool: AcceleratorGraphPool | None = None,
        stream: Any = None,
    ) -> AbstractContextManager:
        kwargs: dict[str, Any] = {
            "pool": None if pool is None else pool.handle,
        }
        if stream is not None:
            kwargs["stream"] = stream
        return self._runtime.graph(graph, **kwargs)

    def new_stream(self) -> Any:
        return self._runtime.Stream(device=self.device)

    def current_stream(self) -> Any:
        return torch.accelerator.current_stream(self.device)

    def stream_context(self, stream: Any) -> AbstractContextManager:
        return self._runtime.stream(stream)

    def memory_allocated(self) -> int:
        memory_allocated = getattr(self._runtime, "memory_allocated", None)
        if memory_allocated is None:
            return 0
        return int(memory_allocated(self.device))

    def is_current_stream_capturing(self) -> bool:
        return bool(self._runtime.is_current_stream_capturing())

    def recover_failed_capture(self, pool: AcceleratorGraphPool, stream: Any) -> None:
        self._runtime.set_stream(stream)
        self.synchronize()
        end = getattr(torch._C, f"_{self.device_type}_endAllocateToPool", None)
        if end is None:
            return
        index = self.device.index
        if index is None:
            index = self._runtime.current_device()
        try:
            end(index, pool.handle)
        except RuntimeError:
            pass


class CUDAGraphBackend(_TorchGraphBackend):

    def __init__(self, device: torch.device):
        super().__init__(device, device_type="cuda", runtime=torch.cuda)

    def create_graph(self) -> CapturedGraph:
        return self._runtime.CUDAGraph()


class XPUGraphBackend(_TorchGraphBackend):

    def __init__(self, device: torch.device):
        super().__init__(device, device_type="xpu", runtime=torch.xpu)

    def create_graph(self) -> CapturedGraph:
        return self._runtime.XPUGraph()


_GRAPH_BACKENDS: dict[str, type[AcceleratorGraphBackend]] = {
    "cuda": CUDAGraphBackend,
    "xpu": XPUGraphBackend,
}


def get_accelerator_graph_backend(
    device: torch.device,
) -> AcceleratorGraphBackend:
    backend = _GRAPH_BACKENDS.get(device.type)
    if backend is None:
        raise ValueError(
            f"Accelerator graphs are unsupported on device type {device.type!r}"
        )
    return backend(device)
