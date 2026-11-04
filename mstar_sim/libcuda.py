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
    _write(count, 0 if tape is None else max(1, len(tape.entries)))
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


def install() -> None:
    real = ctypes.CDLL

    class CDLL(real):
        def __new__(cls, name, *args, **kwargs):
            if isinstance(name, str) and "libcuda.so" in name:
                return LibCuda()
            return super().__new__(cls)

    ctypes.CDLL = CDLL
