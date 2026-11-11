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

    def on_failure(self, graph: CapturedGraph) -> None:
        pass

    def on_success(self) -> None:
        pass


@dataclass
class XPUGraphPool(AcceleratorGraphPool):
    # XPU cannot reopen a pool after its last graph is destroyed. Retain a
    # failed capture until a successful graph owns the pool; CUDA needs no hold.
    # TODO: Remove this hold once the minimum supported PyTorch version fixes
    # XPU graph pool reuse: https://github.com/pytorch/pytorch/issues/198794.
    failed_graph: CapturedGraph | None = field(default=None, repr=False)

    def on_failure(self, graph: CapturedGraph) -> None:
        self.failed_graph = graph

    def on_success(self) -> None:
        self.failed_graph = None


class AcceleratorGraphBackend(ABC):
    """Graph and stream operations required by the accelerator graph runners.

    Implementations own their native graph API and capture recovery. Runners
    do not need to know which runtime supplies those operations.
    """

    def __init__(self, device: torch.device):
        self.device = device

    @property
    def device_type(self) -> str:
        return self.device.type

    def while_refusal(self) -> str | None:
        """Why this backend has no WHILE run (``mstar.engine.device_run``), or None."""
        return f"{type(self).__name__} has no WHILE graph node"

    def while_runtime(self) -> Any:
        """What a WHILE run calls: a ``device_run.CudaWhile``."""
        raise NotImplementedError(self.while_refusal())

    def loop(self, child: CapturedGraph, predicate: torch.Tensor) -> CapturedGraph:
        """WHILE(``predicate``) { ``child`` }, one launch."""
        raise NotImplementedError(self.while_refusal())

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def set_device(self) -> None: ...

    @abstractmethod
    def synchronize(self) -> None: ...

    @abstractmethod
    def create_graph(self, keep_graph: bool = False) -> CapturedGraph: ...

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
            # not recording: it never started, or capture_end stopped it and the graph releases the pool
            return
        # still recording, so the capture never ended and no graph will release the pool
        self._release_pool(index, pool)

    def _release_pool(self, index: int, pool: AcceleratorGraphPool) -> None:
        getattr(torch._C, f"_{self.device_type}_releasePool")(index, pool.handle)


class CUDAGraphBackend(_TorchGraphBackend):

    def __init__(self, device: torch.device):
        super().__init__(device, device_type="cuda", runtime=torch.cuda)
        self._while = None

    def create_graph(self, keep_graph: bool = False) -> CapturedGraph:
        return self._runtime.CUDAGraph(keep_graph=keep_graph)

    def raw_stream(self) -> int:
        """The current ``cudaStream_t``, which ``torch.Stream`` doesn't expose."""
        return int(self._runtime.current_stream(self.device).cuda_stream)

    def while_refusal(self) -> str | None:
        if not torch.version.cuda:
            return "this torch has no CUDA runtime"
        if tuple(int(part) for part in torch.version.cuda.split(".")[:2]) < (12, 4):
            return f"CUDA {torch.version.cuda} has no conditional graph node (12.4 and up)"
        from mstar.engine.device_run import CudaWhile

        try:
            nvrtc_dir = CudaWhile.nvrtc()[1]
        except OSError as error:
            return f"no NVRTC to compile its stop kernel with: {error}"
        nvjitlink, devrt = CudaWhile.link_libraries(nvrtc_dir)
        if nvjitlink is None or devrt is None:
            return (
                "no one CUDA release has both libnvJitLink and libcudadevrt.a to link the stop kernel: "
                f"looked next to torch's NVRTC ({nvrtc_dir}), its sibling wheel directories, then CUDA_HOME "
                "and /usr/local/cuda"
            )
        return None

    def while_runtime(self) -> Any:
        from mstar.engine.device_run import CudaWhile

        if self._while is None:
            self._while = CudaWhile()
        return self._while

    def loop(self, child: CapturedGraph, predicate: torch.Tensor) -> CapturedGraph:
        from mstar.engine.device_run import WhileGraph

        return WhileGraph(self.while_runtime(), child, predicate, self.raw_stream)


class XPUGraphBackend(_TorchGraphBackend):

    def __init__(self, device: torch.device):
        super().__init__(device, device_type="xpu", runtime=torch.xpu)

    def create_graph(self, keep_graph: bool = False) -> CapturedGraph:
        return self._runtime.XPUGraph(keep_graph=keep_graph)

    def graph_pool_handle(self) -> AcceleratorGraphPool:
        return XPUGraphPool(self._runtime.graph_pool_handle())

    def _release_pool(self, index: int, pool: AcceleratorGraphPool) -> None:
        # the pool holds the failed graph instead (see `XPUGraphPool`)
        pass


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
