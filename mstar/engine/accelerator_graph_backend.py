from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, NamedTuple, Protocol, Sequence

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


class RawLaunch(NamedTuple):
    """The addresses a raw run of a graph goes through."""
    graph_launch: int
    memcpy_2d: int
    exec: int
    stream: int


def copy_words(target: torch.Tensor, source: torch.Tensor) -> list[int] | None:
    """A same-dtype copy as destination/pitch, source/pitch, byte width/rows."""
    if target.shape != source.shape or target.dtype != source.dtype or target.numel() == 0:
        return None
    if target.is_contiguous() and source.is_contiguous():
        size = target.numel() * target.element_size()
        return [target.data_ptr(), size, source.data_ptr(), size, size, 1]
    if target.dim() == 0 or not (target[0].is_contiguous() and source[0].is_contiguous()):
        return None
    item = target.element_size()
    return [target.data_ptr(), target.stride(0) * item, source.data_ptr(), source.stride(0) * item,
            target[0].numel() * item, target.shape[0]]


class _PitchedPointer(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("pitch", ctypes.c_size_t),
                ("xsize", ctypes.c_size_t), ("ysize", ctypes.c_size_t)]


class _Dimensions(ctypes.Structure):
    _fields_ = [("x", ctypes.c_size_t), ("y", ctypes.c_size_t), ("z", ctypes.c_size_t)]


class _Memcpy3D(ctypes.Structure):
    _fields_ = [("srcArray", ctypes.c_void_p), ("srcPos", _Dimensions), ("srcPtr", _PitchedPointer),
                ("dstArray", ctypes.c_void_p), ("dstPos", _Dimensions), ("dstPtr", _PitchedPointer),
                ("extent", _Dimensions), ("kind", ctypes.c_int)]


class _CUDAComposedGraph:
    """One native graph over existing captures and copies; owns every binding."""

    def __init__(self, backend, ops):
        self._backend, self._cuda = backend, backend.while_runtime()
        self._ops = tuple(ops)
        self._graph, self._exec = ctypes.c_void_p(), ctypes.c_void_p()
        try:
            self._cuda._call("cudaGraphCreate", ctypes.byref(self._graph), ctypes.c_uint(0))
            previous = None
            for op in self._ops:
                node = ctypes.c_void_p()
                deps = None if previous is None else ctypes.byref(previous)
                count = ctypes.c_size_t(int(previous is not None))
                if isinstance(op, tuple):
                    target, source = op
                    words = copy_words(target, source)
                    if (words is None or source.device != target.device or target.device.type != backend.device.type
                            or (backend.device.index is not None and target.device.index != backend.device.index)):
                        raise ValueError("a composed graph needs same-device contiguous-row copies without casts")
                    dst, dst_pitch, src, src_pitch, width, rows = words
                    params = _Memcpy3D(
                        srcPtr=_PitchedPointer(src, src_pitch, width, rows),
                        dstPtr=_PitchedPointer(dst, dst_pitch, width, rows),
                        extent=_Dimensions(width, rows, 1), kind=3,  # cudaMemcpyDeviceToDevice
                    )
                    self._cuda._call("cudaGraphAddMemcpyNode", ctypes.byref(node), self._graph,
                                     deps, count, ctypes.byref(params))
                else:
                    graph = int(op.raw_cuda_graph())
                    if reason := self._cuda.node_refusal(graph):
                        raise ValueError(f"a composed child capture is unsupported: {reason}")
                    self._cuda._call("cudaGraphAddChildGraphNode", ctypes.byref(node), self._graph,
                                     deps, count, ctypes.c_void_p(graph))
                previous = node
            self._cuda._call("cudaGraphInstantiateWithFlags", ctypes.byref(self._exec), self._graph,
                             ctypes.c_ulonglong(0))
        except BaseException:
            self.reset()
            raise

    def replay(self) -> None:
        self._cuda.launch(self._exec.value, self._backend.raw_stream())

    def pool(self):
        return None  # child captures own their allocation pools

    def reset(self) -> None:
        if self._exec.value:
            self._cuda._call("cudaGraphExecDestroy", self._exec)
            self._exec = ctypes.c_void_p()
        if self._graph.value:
            self._cuda._call("cudaGraphDestroy", self._graph)
            self._graph = ctypes.c_void_p()
        self._ops = ()

    def __del__(self):
        try:
            self.reset()
        except Exception:
            pass  # the CUDA runtime may already be gone during interpreter teardown


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

    def launch_refusal(self) -> str | None:
        """Why a graph can't be launched raw (``raw_launch``), or None."""
        return f"{type(self).__name__} launches no raw graphs"

    def raw_launch(self, graph: CapturedGraph) -> RawLaunch:
        """What ``mstar_rust.launch_run`` needs to launch ``graph`` on the current stream."""
        raise NotImplementedError(self.launch_refusal())

    def compose(self, ops: Sequence[CapturedGraph | tuple[torch.Tensor, torch.Tensor]]) -> CapturedGraph:
        """Lower ordered captured members/copies to one backend graph."""
        raise NotImplementedError(f"{type(self).__name__} cannot compose captured members")

    def generator_offset(self) -> int:
        """Where the device's RNG generator stands: an eager step that moves
        it draws from it, which a graph's raw launch wouldn't advance."""
        return 0

    def while_refusal(self) -> str | None:
        """Why this backend has no WHILE run (``mstar.engine.device_run``), or None."""
        return f"{type(self).__name__} has no WHILE graph node"

    def while_runtime(self) -> Any:
        """What a WHILE run calls: a ``device_run.CudaWhile``."""
        raise NotImplementedError(self.while_refusal())


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

    def compose(self, ops: Sequence[CapturedGraph | tuple[torch.Tensor, torch.Tensor]]) -> CapturedGraph:
        return _CUDAComposedGraph(self, ops)

    def _runtime_entry(self, name: str) -> int:
        """A libcudart entry point's address, by the soname of torch's CUDA
        major, which reaches the copy torch already loaded."""
        lib = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")
        return ctypes.cast(getattr(lib, name), ctypes.c_void_p).value

    def launch_refusal(self) -> str | None:
        if not torch.version.cuda or torch.version.hip is not None:
            return "this torch has no CUDA runtime to launch through"
        return None

    def raw_launch(self, graph: CapturedGraph) -> RawLaunch:
        return RawLaunch(
            self._runtime_entry("cudaGraphLaunch"), self._runtime_entry("cudaMemcpy2DAsync"),
            int(graph.raw_cuda_graph_exec()), self.raw_stream(),
        )

    def raw_stream(self) -> int:
        """The current ``cudaStream_t``, which ``torch.Stream`` doesn't expose."""
        return int(self._runtime.current_stream(self.device).cuda_stream)

    def generator_offset(self) -> int:
        return torch.cuda.default_generators[self.device.index or 0].get_offset()

    def while_refusal(self) -> str | None:
        if not torch.version.cuda or torch.version.hip is not None:
            return "this torch has no CUDA runtime"
        if tuple(int(part) for part in torch.version.cuda.split(".")[:2]) < (12, 4):
            return f"CUDA {torch.version.cuda} has no conditional graph node (12.4 and up)"
        from mstar.engine.device_run import CudaWhile

        if CudaWhile.devrt_path() is None:
            return "no libcudadevrt.a to link its stop kernel against (set CUDA_HOME)"
        return None

    def while_runtime(self) -> Any:
        from mstar.engine.device_run import CudaWhile

        if self._while is None:
            self._while = CudaWhile()
        return self._while


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
