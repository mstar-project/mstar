"""The CUDA driver calls mstar makes through ctypes, answered for the emulated GPU.

mstar counts a captured graph's nodes and measures its executable memory with
``libcuda`` directly (``cuGraphGetNodes``, ``cuGraphInstantiateWithFlags``),
and opens a context to settle a stream's placement. ``ctypes.CDLL`` returns
this stand-in for ``libcuda``; graphs are found by the handle
:meth:`mstar_sim.cuda.CUDAGraph.raw_cuda_graph` gives out. Executable copies
take no simulated memory.
"""

from __future__ import annotations

import ctypes
import functools
import itertools
import weakref

GRAPHS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_handles = itertools.count(1)


def register(graph) -> int:
    handle = next(_handles)
    GRAPHS[handle] = graph
    return handle


def _write(ref, value) -> None:
    obj = getattr(ref, "_obj", ref)
    obj.value = value


class _Fn:
    """A driver entry point: settable ``argtypes``/``restype``, returns CUDA_SUCCESS."""

    def __init__(self, fn):
        self.fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.fn(*args)


def _value(x):
    return x.value if hasattr(x, "value") else x


def _graph_nodes(graph_ptr, nodes, count) -> int:
    graph = GRAPHS.get(_value(graph_ptr))
    tape = getattr(graph, "tape", None)
    _write(count, 0 if tape is None else max(1, tape.kernels + sum(cost is not None for cost, _ in tape.nodes)))
    return 0


def _instantiate(exec_ref, graph_ptr, flags) -> int:
    _write(exec_ref, next(_handles))
    return 0


def _ok(*_):
    return 0


def _device_get(ref, index) -> int:
    _write(ref, int(_value(index)))
    return 0


def _ctx_create(ref, flags, device) -> int:
    _write(ref, next(_handles))
    return 0


class LibCuda:
    _name = "libcuda.so.1"

    def __init__(self):
        self._fns = {
            "cuGraphGetNodes": _Fn(_graph_nodes),
            "cuGraphInstantiateWithFlags": _Fn(_instantiate),
            "cuGraphExecDestroy": _Fn(_ok),
            "cuDeviceGet": _Fn(_device_get),
            "cuCtxCreate_v2": _Fn(_ctx_create),
            "cuCtxDestroy_v2": _Fn(_ok),
            "cuInit": _Fn(_ok),
        }

    def __getattr__(self, name):
        try:
            return self._fns[name]
        except KeyError:
            raise AttributeError(f"mstar_sim: libcuda.{name} has no stand-in yet") from None

    __getitem__ = __getattr__


def _memcpy_async(dst, src, nbytes, kind, stream) -> int:
    """``cudaMemcpyAsync`` host to device (mstar's ``h2d`` stager, from its pinned arena), as the
    emulator's own H2D copy: priced on the stream named by its raw handle and on the H2D copy
    engine, and landing when launched, or at each replay of a graph that captured it. A value-less
    destination (an address of :func:`mstar_sim.memory.fake_base`'s) takes no bytes."""
    from mstar_sim import clock, cost, cuda, memory

    dst, src, nbytes, handle = (_value(x) for x in (dst, src, nbytes, stream))
    if _value(kind) != 1:  # cudaMemcpyHostToDevice: mstar makes no other
        return 1  # cudaErrorInvalidValue
    with clock.computing():
        target = cuda.ExternalStream(handle) if handle else cuda.current_stream()
        previous = clock.set_current_timeline(target.timeline)
        try:
            clock.charge(0.0 if clock.FAST else cost.copy_seconds(nbytes, "h2d"), engine=clock.ENGINES["h2d"])
        finally:
            clock.set_current_timeline(previous)
        if dst < memory.FAKE_START:
            write = functools.partial(ctypes.memmove, dst, src, nbytes)
            if clock.capturing() is None:
                write()
            else:
                clock.on_replay(write)
    return 0


class LibCudart:
    """What mstar calls of ``libcudart`` through ctypes: host registration (mstar's arena pins its
    segments: the range is pinned, :func:`mstar_sim.memory.register`) and ``cudaMemcpyAsync``
    (:func:`_memcpy_async`). The rest of the runtime is absent: a lookup of another symbol raises
    AttributeError, as ctypes does for a symbol a library lacks."""

    _name = "libcudart.so"

    def __init__(self):
        from mstar_sim import memory

        self._fns = {
            "cudaHostRegister": _Fn(lambda ptr, nbytes, flags: memory.register(_value(ptr), _value(nbytes))),
            "cudaHostUnregister": _Fn(lambda ptr: memory.unregister(_value(ptr))),
            "cudaGetLastError": _Fn(_ok),
            "cudaMemcpyAsync": _Fn(_memcpy_async),
        }

    def __getattr__(self, name):
        try:
            return self._fns[name]
        except KeyError:
            raise AttributeError(f"mstar_sim: {name}: no CUDA runtime on the emulated GPU") from None

    __getitem__ = __getattr__


# the native libraries mstar may load through ctypes, and what the emulator does
# (read by the optional compatibility audit in audits/test_compatibility.py)
LIBRARIES = {
    "libcuda": "this stand-in (graph introspection, contexts)",
    "libcudart": "this stand-in's host registration and cudaMemcpyAsync (LibCudart), priced as the emulator's copies",
    "libamdhip64": "ROCm only: mstar never loads it on a CUDA device",
}


def install() -> None:
    """``libcuda`` is this stand-in, and ``libcudart`` :class:`LibCudart`."""
    for name in ("CDLL", "PyDLL"):
        real = getattr(ctypes, name)

        class Loader(real):
            def __new__(cls, lib, *args, **kwargs):
                if isinstance(lib, str) and "libcuda.so" in lib:
                    return LibCuda()
                if isinstance(lib, str) and "libcudart" in lib:
                    return LibCudart()
                return super().__new__(cls)

        Loader.__name__ = Loader.__qualname__ = name
        setattr(ctypes, name, Loader)
