from __future__ import annotations

import gc
import threading
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterator, Protocol

import torch


class CapturedGraph(Protocol):
    def replay(self) -> None: ...

    def reset(self) -> None: ...

    def pool(self) -> Any: ...


class SplitGraph:
    """A capture `split_capture` split: a replay runs the first graph, ``fn`` on the host, then the
    second, even when ``fn`` raises, so a rank whose ``fn`` raised still runs the second's collectives."""

    def __init__(self, graphs: list[CapturedGraph], fn: Callable[[], None]):
        self.graphs = graphs
        self.fn = fn

    def replay(self, launching: Callable[[], None] | None = None) -> None:
        """``launching`` runs just before the second graph's launch, after ``fn``."""
        first, second = self.graphs
        first.replay()
        try:
            self.fn()
        finally:
            if launching is not None:
                launching()
            second.replay()

    def reset(self) -> None:
        for graph in self.graphs:
            graph.reset()

    def pool(self) -> Any:
        return self.graphs[0].pool()


def replay_graph(graph: CapturedGraph | SplitGraph, launching: Callable[[], None] | None = None) -> None:
    """Replay ``graph``, running ``launching`` just before its last launch: before
    the only one of a plain graph, before the second of a split one."""
    if isinstance(graph, SplitGraph):
        graph.replay(launching)
        return
    if launching is not None:
        launching()
    graph.replay()


@dataclass
class GraphCapture:
    """A capture running on this thread (`AcceleratorGraphBackend.capture`): the graphs it
    records into, in order, and the host call `split_capture` put between them."""
    backend: AcceleratorGraphBackend
    graphs: list[CapturedGraph]
    keep_graph: bool
    fn: Callable[[], None] | None = None

    def graph(self) -> CapturedGraph:
        return self.graphs[0] if self.fn is None else SplitGraph(self.graphs, self.fn)


_capturing = threading.local()


def split_capture(fn: Callable[[], None]) -> None:
    """End the graph this thread is capturing and capture on into a new one in the same pool, so
    every replay runs ``fn`` on the host between the two (`SplitGraph`): host work that needs what
    the step before wrote, done while the first graph runs. ``fn`` runs on the stream the replay
    is on; the second graph replays whether or not it raises. Once per capture; a second split is
    a ``ValueError``."""
    capture: GraphCapture | None = getattr(_capturing, "capture", None)
    if capture is None or not capture.backend.is_current_stream_capturing():
        raise ValueError(
            "split_capture runs only inside a splittable graph capture (AcceleratorGraphBackend.capture_split)"
        )
    if capture.fn is not None:
        raise ValueError("a graph capture splits once, and this one already split")
    capture.backend.split(capture)
    capture.fn = fn


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
    def capture_split(
        self,
        graph: CapturedGraph,
        *,
        pool: AcceleratorGraphPool | None = None,
        stream: Any = None,
        keep_graph: bool = False,
    ) -> AbstractContextManager[GraphCapture]:
        """`capture` for a node whose capture may `split_capture`; every other capture uses `capture`."""

    @abstractmethod
    def split(self, capture: GraphCapture) -> None:
        """End ``capture``'s current graph and begin its next, in the same pool."""

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
        # the side stream a capture runs on when its caller names none
        self._capture_stream: Any = None

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

    @contextmanager
    def capture_split(
        self,
        graph: CapturedGraph,
        *,
        pool: AcceleratorGraphPool | None = None,
        stream: Any = None,
        keep_graph: bool = False,
    ) -> Iterator[GraphCapture]:
        """What the runtime's own ``graph`` context does, which ends only the graph it began: after
        a `split_capture` the capture ends in another. Only for a node whose resources split their
        capture (``Resource.splits_capture``); every other capture is the runtime's own (`capture`)."""
        # free as much memory as can be for the graph; a collect only on request, as torch's own
        # context does, since it is slow and a capture comes many to a boot
        self.synchronize()
        if torch.compiler.config.force_cudagraph_gc:
            gc.collect()
        self._runtime.empty_cache()
        if stream is None:
            if self._capture_stream is None:
                self._capture_stream = self.new_stream()
            stream = self._capture_stream
        capture = GraphCapture(self, [graph], keep_graph)
        with self.stream_context(stream):
            graph.capture_begin(*(() if pool is None else (pool.handle,)))
            _capturing.capture = capture
            try:
                yield capture
            except BaseException:
                # as the runtime's context does, which would raise its own error in place of the first
                with suppress(RuntimeError):
                    capture.graphs[-1].capture_end()
                raise
            finally:
                _capturing.capture = None
            capture.graphs[-1].capture_end()

    def split(self, capture: GraphCapture) -> None:
        graph = capture.graphs[-1]
        graph.capture_end()
        following = self.create_graph(capture.keep_graph)
        capture.graphs.append(following)
        following.capture_begin(graph.pool())

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

    def create_graph(self, keep_graph: bool = False) -> CapturedGraph:
        return self._runtime.CUDAGraph(keep_graph=keep_graph)


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
