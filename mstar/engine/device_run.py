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
NVRTC and linked against ``libcudadevrt``.
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
        self._kernels: tuple[ctypes.c_void_p, ...] | None = None

    @staticmethod
    def devrt_path() -> str | None:
        """``libcudadevrt.a``: ``cudaGraphSetConditional`` links against it."""
        for root in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
            path = os.path.join(root, "lib64", "libcudadevrt.a") if root else None
            if path and os.path.exists(path):
                return path
        return None

    def call(self, name: str, *args) -> None:
        status = getattr(self._rt, name)(*args)
        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA error {status}")

    def _driver_call(self, name: str, *args) -> None:
        status = getattr(self._driver, name)(*args)
        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA error {status}")

    def load_kernels(self) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
        """The stop and condition kernels, compiled once a process."""
        if self._kernels is None:
            from torch.cuda._utils import _nvrtc_compile

            ptx = _nvrtc_compile(_STOP_SOURCE, "run_stop", nvcc_options=["-rdc=true"])
            ptx = ptx[0] if isinstance(ptx, tuple) else ptx
            link, cubin, size = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_size_t()
            self._driver_call("cuLinkCreate_v2", 0, None, None, ctypes.byref(link))
            self._driver_call(
                "cuLinkAddData_v2", link, 1, ctypes.c_char_p(ptx), len(ptx), b"run_stop.ptx", 0, None, None,
            )
            self._driver_call("cuLinkAddFile_v2", link, 4, self.devrt_path().encode(), 0, None, None)
            self._driver_call("cuLinkComplete", link, ctypes.byref(cubin), ctypes.byref(size))
            module, stop, condition = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
            self._driver_call("cuModuleLoadData", ctypes.byref(module), cubin)
            self._driver_call("cuModuleGetFunction", ctypes.byref(stop), module, b"run_stop")
            self._driver_call("cuModuleGetFunction", ctypes.byref(condition), module, b"set_graph_condition")
            self._driver_call("cuLinkDestroy", link)
            self._kernels = stop, condition
        return self._kernels

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
