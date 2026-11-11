"""A step that repeats, ended by the device: one launch of a WHILE node around
the step's capture, which stops once every row has stopped or the step's count
is reached.

A run ends short only when every row has stopped, so a row still running always
ran all of them, and what the host reserved and committed for the step stands.
The host learns how many iterations ran from the output's ``run_replays`` when
the step lands.
A row's stop stays decided by the graph runtime, from the tokens that run kept;
the device only decides when to end.

torch binds no WHILE node, so ``CudaWhile`` calls the CUDA runtime through
ctypes, and ``cudaGraphSetConditional`` needs a kernel of ours, compiled with
NVRTC to a cubin for the device's own arch and linked against ``libcudadevrt``
by nvJitLink, so the driver loads machine code and JIT-compiles none: a driver
older than the NVRTC torch ships can't read its PTX.
"""
from __future__ import annotations

import ctypes
import os
from collections.abc import Collection, Mapping, Sequence
from typing import NamedTuple

import torch

from mstar.utils.h2d import PinnedStager

_STOP_SOURCE = r"""
typedef unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle, unsigned int);

extern "C" __global__ void run_stop(
    const long long* tokens, int stride, int rows, const long long* stop_ids, int num_ids,
    const unsigned char* immune, unsigned char* marks,
    long long* index, const long long* limit, long long* continuation
) {
    int done = 0;
    for (int row = 0; row < rows; ++row) {
        if (!immune[row]) {
            for (int i = 0; i < num_ids; ++i) {
                if (tokens[row * stride] == stop_ids[i]) marks[row] = 1;
            }
        }
        done += marks[row];
    }
    long long next = *index + 1;
    *index = next;
    *continuation = done < rows && next < *limit ? 1 : 0;
}

extern "C" __global__ void set_graph_condition(
    const long long* predicate, cudaGraphConditionalHandle handle
) {
    cudaGraphSetConditional(handle, *predicate != 0 ? 1u : 0u);
}
"""


class WalkAsk(NamedTuple):
    """What the worker tells the engine of a walk asked to repeat, before capture."""
    # the signals one iteration hands the next on the device
    loop_back: frozenset[str]
    # the most iterations a device run takes; 0 replays it from the host
    device_limit: int = 0


class RunAsk(NamedTuple):
    """What a walk's capture needs to be the body of a device run."""
    # the most iterations a run takes
    limit: int
    # the output holding each row's token
    token_signal: str
    stop_ids: frozenset[int]
    # output -> the static input the next iteration reads
    carries: Mapping[str, str]


class RunState:
    """A capture slot's run, on the device. ``marks``: a row's stop in this
    run, padding rows set, so an earlier EOS doesn't end it. ``tokens``: one
    column an iteration, outside the capture's memory pool, where the next
    iteration's scratch would land on them."""

    def __init__(self, bs: int, limit: int, stop_ids: Collection[int], device: torch.device):
        self.index = torch.zeros(1, dtype=torch.long, device=device)
        self.continuation = torch.zeros_like(self.index)
        self.limit = torch.ones(1, dtype=torch.long, device=device)
        self.immune = torch.zeros(bs, dtype=torch.bool, device=device)
        self.marks = torch.zeros(bs, dtype=torch.bool, device=device)
        self.tokens = torch.empty(bs, limit, dtype=torch.long, device=device)
        self.stop_ids = torch.tensor(sorted(stop_ids), dtype=torch.long, device=device)
        self._rows = torch.arange(bs, device=device)
        self._immune_stager = PinnedStager(torch.bool, numel=bs)
        self._staged: tuple | None = None

    def stage(self, real: int, immune: Sequence[bool], limit: int) -> None:
        """Before the launch, on its stream. A staged value keeps its address:
        the graph reads it."""
        self.index.zero_()
        self.continuation.fill_(1)
        key = (real, tuple(immune), limit)
        if key != self._staged:
            self._staged = key
            self.limit.fill_(limit)
            self._immune_stager.copy_(self.immune, immune, pad_value=False)
        torch.ge(self._rows, real, out=self.marks)

    def keep_token(self, token: torch.Tensor) -> None:
        """Inside the body: this iteration's token column."""
        self.tokens.index_copy_(1, self.index, token.unsqueeze(1).to(self.tokens.dtype))

    def results(self, repeats: int, names: Collection[str]) -> dict[str, torch.Tensor]:
        """After the launch: the stop buffers, a column an iteration. ``index``
        holds how many ran."""
        return {name: self.tokens[:, :repeats] for name in names}


_KERNELS: dict[int, tuple[ctypes.c_void_p, ctypes.c_void_p]] = {}


class _DlInfo(ctypes.Structure):
    """``Dl_info``, which ``dladdr`` fills: the library a symbol was loaded from."""
    _fields_ = [
        ("path", ctypes.c_char_p), ("base", ctypes.c_void_p), ("name", ctypes.c_char_p), ("addr", ctypes.c_void_p),
    ]


class CudaWhile:
    """The CUDA runtime calls a WHILE run takes, through ctypes."""

    class _Conditional(ctypes.Structure):
        _fields_ = [
            ("handle", ctypes.c_ulonglong), ("type", ctypes.c_int), ("size", ctypes.c_uint),
            ("graphs", ctypes.POINTER(ctypes.c_void_p)), ("ctx", ctypes.c_void_p),
        ]

    class _KernelNode(ctypes.Structure):
        """``CUDA_KERNEL_NODE_PARAMS_v2``."""
        _fields_ = [
            ("func", ctypes.c_void_p), ("grid", ctypes.c_uint * 3), ("block", ctypes.c_uint * 3),
            ("shared", ctypes.c_uint), ("args", ctypes.POINTER(ctypes.c_void_p)), ("extra", ctypes.c_void_p),
            ("kern", ctypes.c_void_p), ("ctx", ctypes.c_void_p),
        ]

    _NODE_NAMES = {3: "host", 7: "event record", 10: "memory alloc", 11: "memory free"}
    _TYPE_CONDITIONAL = 13

    def __init__(self):
        self._rt = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")
        self._driver = ctypes.CDLL("libcuda.so.1")

    @staticmethod
    def nvrtc() -> tuple[ctypes.CDLL, str | None]:
        """The NVRTC torch loads, and the directory it was loaded from."""
        from torch.cuda._utils import _get_nvrtc_library

        nvrtc = _get_nvrtc_library()
        info = _DlInfo()
        found = ctypes.CDLL(None).dladdr(ctypes.cast(nvrtc.nvrtcCreateProgram, ctypes.c_void_p), ctypes.byref(info))
        return nvrtc, os.path.dirname(info.path.decode()) if found and info.path else None

    @staticmethod
    def link_libraries(nvrtc_dir: str | None) -> tuple[str | None, str | None]:
        """The paths of ``libnvJitLink`` and ``libcudadevrt.a`` (``cudaGraphSetConditional`` links against it), or
        ``(None, None)`` when no CUDA release has both. They have to come from one release, the NVRTC's own (a newer
        devrt or linker fails the link), so the places are tried as pairs: next to the NVRTC torch loaded (the one
        ``nvidia/cu13/lib`` directory, or the older wheels' sibling ``nvidia/nvjitlink/lib`` and
        ``nvidia/cuda_runtime/lib``), then each toolkit."""
        nvjitlink, devrt = f"libnvJitLink.so.{torch.version.cuda.split('.')[0]}", "libcudadevrt.a"
        places = []
        if nvrtc_dir:
            places += [
                (nvrtc_dir, nvrtc_dir),
                tuple(os.path.join(nvrtc_dir, "..", "..", package, "lib") for package in ("nvjitlink", "cuda_runtime")),
            ]
        toolkits = [os.path.join(root, "lib64") for root in (os.environ.get("CUDA_HOME"), "/usr/local/cuda") if root]
        places += [(lib64, lib64) for lib64 in toolkits]
        for link_dir, devrt_dir in places:
            if os.path.exists(os.path.join(link_dir, nvjitlink)) and os.path.exists(os.path.join(devrt_dir, devrt)):
                return os.path.join(link_dir, nvjitlink), os.path.join(devrt_dir, devrt)
        return None, None

    @staticmethod
    def _invoke(lib: ctypes.CDLL, name: str, *args, log=None) -> None:
        status = getattr(lib, name)(*args)
        if status != 0:
            raise RuntimeError(f"{name} failed with error {status}" + (f":\n{log()}" if log else ""))

    def call(self, name: str, *args) -> None:
        self._invoke(self._rt, name, *args)

    def _driver_call(self, name: str, *args) -> None:
        self._invoke(self._driver, name, *args)

    def _compile_stop(self, arch: str, nvrtc: ctypes.CDLL) -> bytes:
        """``_STOP_SOURCE`` as a relocatable cubin for ``arch`` (``sm_90``), by torch's NVRTC."""
        prog = ctypes.c_void_p()

        def log() -> str:
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            text = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(prog, text)
            return text.value.decode()

        self._invoke(
            nvrtc, "nvrtcCreateProgram", ctypes.byref(prog), _STOP_SOURCE.encode(), b"run_stop.cu", 0, None, None,
        )
        try:
            options = (ctypes.c_char_p * 2)(f"--gpu-architecture={arch}".encode(), b"-rdc=true")
            self._invoke(nvrtc, "nvrtcCompileProgram", prog, 2, options, log=log)
            size = ctypes.c_size_t()
            self._invoke(nvrtc, "nvrtcGetCUBINSize", prog, ctypes.byref(size))
            cubin = ctypes.create_string_buffer(size.value)
            self._invoke(nvrtc, "nvrtcGetCUBIN", prog, cubin)
        finally:
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        return cubin.raw

    def _link_stop(self, arch: str, cubin: bytes, nvrtc_dir: str | None) -> ctypes.Array:
        """``cubin`` and ``libcudadevrt`` linked to the device's SASS, by the nvJitLink of the NVRTC's CUDA."""
        path, devrt = self.link_libraries(nvrtc_dir)
        if devrt is None or path is None:
            raise RuntimeError(
                "linking the device stop kernel needs libnvJitLink and libcudadevrt.a from one CUDA release, "
                "and none was found; set CUDA_HOME to a toolkit that has them"
            )
        lib, handle = ctypes.CDLL(path), ctypes.c_void_p()

        def log() -> str:
            size = ctypes.c_size_t()
            lib.nvJitLinkGetErrorLogSize(handle, ctypes.byref(size))
            text = ctypes.create_string_buffer(size.value)
            lib.nvJitLinkGetErrorLog(handle, text)
            return text.value.decode()

        options = (ctypes.c_char_p * 1)(f"-arch={arch}".encode())
        self._invoke(lib, "nvJitLinkCreate", ctypes.byref(handle), 1, options)
        try:
            # input types: 1 is a cubin, 6 a library
            self._invoke(
                lib, "nvJitLinkAddData", handle, 1, ctypes.c_char_p(cubin), ctypes.c_size_t(len(cubin)),
                b"run_stop.cubin", log=log,
            )
            self._invoke(lib, "nvJitLinkAddFile", handle, 6, devrt.encode(), log=log)
            self._invoke(lib, "nvJitLinkComplete", handle, log=log)
            size = ctypes.c_size_t()
            self._invoke(lib, "nvJitLinkGetLinkedCubinSize", handle, ctypes.byref(size), log=log)
            linked = ctypes.create_string_buffer(size.value)
            self._invoke(lib, "nvJitLinkGetLinkedCubin", handle, linked, log=log)
        finally:
            lib.nvJitLinkDestroy(ctypes.byref(handle))
        return linked

    def load_kernels(self) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
        """The stop and condition kernels of the current device, compiled once a device (a module belongs to a
        context; the module loads on one thread, so no lock)."""
        device = torch.cuda.current_device()
        if device not in _KERNELS:
            arch = "sm_{}{}".format(*torch.cuda.get_device_capability())
            nvrtc, nvrtc_dir = self.nvrtc()
            linked = self._link_stop(arch, self._compile_stop(arch, nvrtc), nvrtc_dir)
            module, stop, condition = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
            self._driver_call("cuModuleLoadData", ctypes.byref(module), linked)
            self._driver_call("cuModuleGetFunction", ctypes.byref(stop), module, b"run_stop")
            self._driver_call("cuModuleGetFunction", ctypes.byref(condition), module, b"set_graph_condition")
            _KERNELS[device] = stop, condition
        return _KERNELS[device]

    def stop(self, state: RunState, tokens: torch.Tensor, stream: int) -> None:
        """Inside the step's capture: ``run_stop`` on ``stream``."""
        row = tokens.reshape(len(state.marks), -1)
        values = [
            ctypes.c_void_p(row.data_ptr()), ctypes.c_int(row.stride(0)), ctypes.c_int(len(state.marks)),
            ctypes.c_void_p(state.stop_ids.data_ptr()), ctypes.c_int(len(state.stop_ids)),
            ctypes.c_void_p(state.immune.data_ptr()), ctypes.c_void_p(state.marks.data_ptr()),
            ctypes.c_void_p(state.index.data_ptr()), ctypes.c_void_p(state.limit.data_ptr()),
            ctypes.c_void_p(state.continuation.data_ptr()),
        ]
        args = (ctypes.c_void_p * len(values))(*(ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values))
        self._driver_call(
            "cuLaunchKernel", self.load_kernels()[0], 1, 1, 1, 1, 1, 1, 0, ctypes.c_void_p(stream), args, None,
        )

    def node_refusal(self, step_graph: int) -> str | None:
        """Why a WHILE body can't hold ``step_graph``: what a child graph refuses."""
        count = ctypes.c_size_t()
        self.call("cudaGraphGetNodes", ctypes.c_void_p(step_graph), None, ctypes.byref(count))
        if count.value == 0:
            return None
        nodes = (ctypes.c_void_p * count.value)()
        self.call("cudaGraphGetNodes", ctypes.c_void_p(step_graph), nodes, ctypes.byref(count))
        for node in nodes:
            kind = ctypes.c_int()
            found = self._rt.cudaGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(kind))
            name = "conditional" if found != 0 else self._NODE_NAMES.get(kind.value)
            if name is not None:
                return f"its capture holds a {name} node, which a child graph of a WHILE body can't"
        return None

    def add_condition_setter(
        self, graph: int, previous: ctypes.c_void_p | None, predicate: torch.Tensor, handle: int,
    ) -> ctypes.c_void_p:
        """A graph node that sets ``handle`` from a device predicate."""
        values = [ctypes.c_void_p(predicate.data_ptr()), ctypes.c_ulonglong(handle)]
        args = (ctypes.c_void_p * len(values))(*(ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values))
        params = self._KernelNode(func=self.load_kernels()[1], grid=(1, 1, 1), block=(1, 1, 1), args=args)
        node = ctypes.c_void_p()
        self._driver_call(
            "cuGraphAddKernelNode_v2", ctypes.byref(node), ctypes.c_void_p(graph),
            None if previous is None else ctypes.byref(previous), ctypes.c_size_t(int(previous is not None)),
            ctypes.byref(params),
        )
        return node

    def add_while(self, graph: int, previous: ctypes.c_void_p, handle: int) -> int:
        """A WHILE node after ``previous``, and its body graph."""
        raw = (ctypes.c_longlong * 32)()  # cudaGraphNodeParams: type, reserved, union, reserved
        conditional = self._Conditional.from_address(ctypes.addressof(raw) + 16)
        ctypes.c_int.from_address(ctypes.addressof(raw)).value = self._TYPE_CONDITIONAL
        conditional.handle, conditional.type, conditional.size = handle, 1, 1  # 1: WHILE
        node = ctypes.c_void_p()
        # the form that takes edge data: CUDA 13 gave the base name this signature
        add_node = "cudaGraphAddNode" if int(torch.version.cuda.split(".")[0]) >= 13 else "cudaGraphAddNode_v2"
        self.call(
            add_node, ctypes.byref(node), ctypes.c_void_p(graph), ctypes.byref(previous), None, ctypes.c_size_t(1),
            ctypes.byref(raw),
        )
        return conditional.graphs[0]


class WhileGraph:
    """WHILE(``predicate``) { ``child`` }, launched in one call. Holds the child
    capture, whose graph it references."""

    def __init__(self, cuda: CudaWhile, child, predicate: torch.Tensor, stream):
        self._cuda, self._child, self._predicate, self._stream = cuda, child, predicate, stream
        self._graph, self._exec = ctypes.c_void_p(), ctypes.c_void_p()
        cuda.call("cudaGraphCreate", ctypes.byref(self._graph), ctypes.c_uint(0))
        handle = ctypes.c_ulonglong()
        cuda.call(
            "cudaGraphConditionalHandleCreate", ctypes.byref(handle), self._graph, ctypes.c_uint(0), ctypes.c_uint(0),
        )
        setter = cuda.add_condition_setter(self._graph.value, None, predicate, handle.value)
        body = cuda.add_while(self._graph.value, setter, handle.value)
        raw_child = int(child.raw_cuda_graph())
        if reason := cuda.node_refusal(raw_child):
            raise ValueError(reason)
        node = ctypes.c_void_p()
        cuda.call(
            "cudaGraphAddChildGraphNode", ctypes.byref(node), ctypes.c_void_p(body), None, ctypes.c_size_t(0),
            ctypes.c_void_p(raw_child),
        )
        cuda.add_condition_setter(body, node, predicate, handle.value)
        cuda.call("cudaGraphInstantiateWithFlags", ctypes.byref(self._exec), self._graph, ctypes.c_ulonglong(0))

    def replay(self) -> None:
        self._cuda.call("cudaGraphLaunch", self._exec, ctypes.c_void_p(self._stream()))

    def reset(self) -> None:
        return
