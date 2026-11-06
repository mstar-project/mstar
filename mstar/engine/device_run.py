"""A run of a captured step on the device: one host launch, a WHILE node around
the step, until a row-by-row stop says it's over.

The body is the step's capture plus what a run needs in the graph: each
iteration input's column in, each kept output's column out, the loop-back
signals the rows don't hold carried, and a fused stop kernel that marks rows
and sets the node's condition. A run ends on the first of: every row done, the
run's ``limit``, the host's yield word (ready work or an abort), or
``refill_at`` of its rows done. A row that's still live hands on to the next
run where it stands: the host learns how many iterations ran (``RUN_REPLAYS``)
when the run lands.

torch binds no WHILE node, so ``CudaWhile`` calls the runtime through ctypes, as
``kv/page_copier.py`` does; ``cudaGraphSetConditional`` needs a kernel of ours,
compiled with NVRTC and linked against ``libcudadevrt``. ``run_stop`` is a
device run's stop rule; a host-launched step's is ``StopPolicy.stops``.
"""
from __future__ import annotations

import ctypes
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import nullcontext
from typing import Any, NamedTuple

import torch

from mstar.engine.resources.resident import TRASH_SLOT

# a stop buffer of a run that repeats: the iterations it ran, on every row
RUN_REPLAYS = "__run_replays__"

_STOP_SOURCE = r"""
typedef unsigned long long cudaGraphConditionalHandle;
extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle, unsigned int);

extern "C" __global__ void run_stop(
    const long long* tokens, int stride, int rows, const long long* stop_ids, int num_ids,
    const unsigned char* immune, unsigned char* marks, long long* index, const long long* limit,
    const long long* refill, const volatile int* yield_word, const unsigned long long* handle,
    long long* host_tokens, volatile int* landed, int host_width
) {
    int done = 0;
    long long at = *index;
    for (int row = 0; row < rows; ++row) {
        if (host_tokens) host_tokens[row * host_width + at] = tokens[row * stride];
        if (!marks[row] && !immune[row]) {
            for (int i = 0; i < num_ids; ++i) {
                if (tokens[row * stride] == stop_ids[i]) marks[row] = 1;
            }
        }
        done += marks[row];
    }
    long long next = at + 1;
    *index = next;
    if (host_tokens) {
        __threadfence_system();
        *landed = (int)next;
    }
    bool go = done < rows && next < *limit && *yield_word == 0 && (*refill == 0 || done < *refill);
    cudaGraphSetConditional(*handle, go ? 1u : 0u);
}
"""


class RunAsk(NamedTuple):
    """What a walk's WHILE run needs of its captures."""
    token_signal: str
    stop_ids: frozenset[int]
    # also send a stopped row's reads and writes to the trash slot
    drop_rows: bool
    # the most iterations a run takes
    limit: int
    # loop-back signals the node's rows don't hold, carried in the body
    carries: Mapping[str, Any]
    # the row outputs kept a column a replay
    columns: Collection[str]
    # the static inputs that take a column a replay
    iteration: Collection[str]
    refill_at: float | None = None
    stream_tokens: bool = False


class TokenStream:
    """A run's tokens in pinned memory as its replays finish
    (``step_repeats: stream_tokens``): the device writes a column, then
    ``landed``, and the host sends what has landed while it waits. What it
    sent, by request, the landing skips."""

    def __init__(self, rows: int, limit: int, device: torch.device):
        pinned = device.type != "cpu"
        self.tokens = torch.zeros(rows, limit, dtype=torch.long, pin_memory=pinned)
        self.landed = torch.zeros(1, dtype=torch.int32, pin_memory=pinned)
        # what ``landed`` takes after each replay of a host loop
        self.counts = torch.arange(1, limit + 1, dtype=torch.int32, device=device)
        self.rids: Sequence[int] = ()
        self.sent: dict[int, int] = {}
        self.done: set[int] = set()
        self.columns = 0

    def begin(self, rids: Sequence[int]) -> None:
        """Before a run, on the host."""
        self.landed.zero_()
        self.rids, self.sent, self.done, self.columns = rids, {}, set(), 0

    def take(self, decide: Callable[[int, int, int], tuple[bool, bool]]) -> tuple[list[int], list[int]]:
        """What landed since the last call, as rows and tokens in order.
        ``decide(rid, column, token)`` says whether the token goes and whether
        it is the row's last: nothing of the row goes after that."""
        landed = int(self.landed[0])
        rids: list[int] = []
        values: list[int] = []
        if landed == self.columns:
            return rids, values
        rows = self.tokens[:len(self.rids), self.columns:landed].tolist()
        for offset in range(landed - self.columns):
            for rid, row in zip(self.rids, rows, strict=True):
                if rid in self.done:
                    continue
                send, last = decide(rid, self.columns + offset, row[offset])
                if send:
                    rids.append(rid)
                    values.append(row[offset])
                    self.sent[rid] = self.sent.get(rid, 0) + 1
                if last:
                    self.done.add(rid)
        self.columns = landed
        return rids, values


class RunState:
    """What a slot's run holds on the device. ``marks``: a row's stop in this
    run, padding rows set; they last one run, so an EOS from before it doesn't
    end it."""

    def __init__(
        self, bs: int, token_signal: str, stop_ids: Collection[int], device: torch.device,
        dropping: Any = None, stream: TokenStream | None = None,
    ):
        self.token_signal = token_signal
        # where its tokens land as it goes, for a walk that streams them
        self.stream = stream
        # a capture that also sends a stopped row's reads and writes to the trash slot
        self.dropping = dropping
        self.index = torch.zeros(1, dtype=torch.long, device=device)
        self.limit = torch.ones(1, dtype=torch.long, device=device)
        # the marks a run ends at, ``refill_at`` of its rows; 0 is off
        self.refill = torch.zeros(1, dtype=torch.long, device=device)
        self.immune = torch.zeros(bs, dtype=torch.bool, device=device)
        self.marks = torch.zeros(bs, dtype=torch.bool, device=device)
        self.replays = torch.zeros(bs, dtype=torch.long, device=device)
        # set by the host to end the run; its own, since the step behind stages while this one runs
        self.yield_word = torch.zeros(1, dtype=torch.int32, pin_memory=torch.accelerator.is_available())
        self.stop_ids = torch.tensor(sorted(stop_ids), dtype=torch.long, device=device)
        self._rows = torch.arange(bs, device=device)
        self._staged: tuple | None = None

    def stage(self, real: int, immune: Sequence[bool], limit: int, refill_at: float | None = None) -> None:
        """Before a run, on the host. A staged value keeps its address: the graph reads it."""
        self.index.zero_()
        self.yield_word.zero_()
        key = (real, tuple(immune), limit, refill_at)
        if key != self._staged:
            self._staged = key
            self.limit.fill_(limit)
            pad = len(self.marks) - real
            self.refill.fill_(0 if refill_at is None else pad + max(1, int(refill_at * real + 0.999999)))
            self.immune.zero_()
            if any(immune):
                self.immune[:len(immune)].copy_(torch.tensor(immune, dtype=torch.bool))
        torch.ge(self._rows, real, out=self.marks)

    def head(self) -> None:
        """Before a replay's forward: a stopped row reads and writes the trash slot."""
        if self.dropping is not None:
            rows = self.dropping.rows
            torch.where(self.marks, torch.full_like(rows, TRASH_SLOT), rows, out=self.dropping.slots)


class CudaWhile:
    """The CUDA runtime calls a WHILE run takes, through ctypes."""

    class _Conditional(ctypes.Structure):
        _fields_ = [
            ("handle", ctypes.c_ulonglong), ("type", ctypes.c_int), ("size", ctypes.c_uint),
            ("graphs", ctypes.POINTER(ctypes.c_void_p)), ("ctx", ctypes.c_void_p),
        ]

    _NODE_NAMES = {3: "host", 7: "event record", 10: "memory alloc", 11: "memory free", 13: "conditional"}
    _TYPE_CONDITIONAL, _COND_WHILE, _ASSIGN_DEFAULT = 13, 1, 1

    def __init__(self):
        self._rt = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")
        self._driver = ctypes.CDLL("libcuda.so.1")
        self._kernel: ctypes.c_void_p | None = None

    @staticmethod
    def devrt_path() -> str | None:
        """``libcudadevrt.a``: ``cudaGraphSetConditional`` links against it."""
        for root in (os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
            path = os.path.join(root, "lib64", "libcudadevrt.a") if root else None
            if path and os.path.exists(path):
                return path
        return None

    def _call(self, name: str, *args) -> None:
        status = getattr(self._rt, name)(*args)
        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA error {status}")

    def _driver_call(self, name: str, *args) -> None:
        status = getattr(self._driver, name)(*args)
        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA error {status}")

    def _load_kernel(self) -> ctypes.c_void_p:
        """``run_stop``, compiled once a process."""
        if self._kernel is None:
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
            module, kernel = ctypes.c_void_p(), ctypes.c_void_p()
            self._driver_call("cuModuleLoadData", ctypes.byref(module), cubin)
            self._driver_call("cuModuleGetFunction", ctypes.byref(kernel), module, b"run_stop")
            self._driver_call("cuLinkDestroy", link)
            self._kernel = kernel
        return self._kernel

    def new_loop(self, device: torch.device) -> tuple[int, int, torch.Tensor]:
        """The outer graph, its condition's handle, and the handle on the device."""
        graph, handle = ctypes.c_void_p(), ctypes.c_ulonglong()
        self._call("cudaGraphCreate", ctypes.byref(graph), 0)
        self._call(
            "cudaGraphConditionalHandleCreate", ctypes.byref(handle), graph, 1, self._ASSIGN_DEFAULT,
        )
        return graph.value, handle.value, torch.tensor([handle.value], dtype=torch.int64, device=device)

    def stop(self, state: RunState, tokens: torch.Tensor, handle: torch.Tensor, stream: int) -> None:
        """Inside the step's capture: ``run_stop`` on ``stream``."""
        row = tokens.reshape(len(state.marks), -1)
        values = [
            ctypes.c_void_p(row.data_ptr()), ctypes.c_int(row.stride(0)), ctypes.c_int(len(state.marks)),
            ctypes.c_void_p(state.stop_ids.data_ptr()), ctypes.c_int(len(state.stop_ids)),
            ctypes.c_void_p(state.immune.data_ptr()), ctypes.c_void_p(state.marks.data_ptr()),
            ctypes.c_void_p(state.index.data_ptr()), ctypes.c_void_p(state.limit.data_ptr()),
            ctypes.c_void_p(state.refill.data_ptr()), ctypes.c_void_p(state.yield_word.data_ptr()),
            ctypes.c_void_p(handle.data_ptr()),
            ctypes.c_void_p(state.stream.tokens.data_ptr() if state.stream else None),
            ctypes.c_void_p(state.stream.landed.data_ptr() if state.stream else None),
            ctypes.c_int(state.stream.tokens.shape[1] if state.stream else 0),
        ]
        args = (ctypes.c_void_p * len(values))(*(ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values))
        self._driver_call(
            "cuLaunchKernel", self._load_kernel(), 1, 1, 1, 1, 1, 1, 0, ctypes.c_void_p(stream), args, None,
        )

    def node_refusal(self, step_graph: int) -> str | None:
        """Why a WHILE body can't hold ``step_graph``: what the child graph refuses."""
        count = ctypes.c_size_t()
        self._call("cudaGraphGetNodes", ctypes.c_void_p(step_graph), None, ctypes.byref(count))
        nodes = (ctypes.c_void_p * count.value)()
        self._call("cudaGraphGetNodes", ctypes.c_void_p(step_graph), nodes, ctypes.byref(count))
        for node in nodes:
            kind = ctypes.c_int()
            # the runtime fails to name a conditional node's type
            found = self._rt.cudaGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(kind))
            name = self._NODE_NAMES.get(kind.value if found == 0 else self._TYPE_CONDITIONAL)
            if name is not None:
                return f"its capture holds a {name} node, which a child graph of a WHILE body can't"
        return None

    def finish(self, outer: int, handle: int, step_graph: int) -> int:
        """The WHILE node, its body the step's graph; the outer graph's exec."""
        body = ctypes.c_void_p()
        raw = (ctypes.c_longlong * 32)()  # cudaGraphNodeParams: type, reserved, union, reserved
        conditional = self._Conditional.from_address(ctypes.addressof(raw) + 16)
        ctypes.c_int.from_address(ctypes.addressof(raw)).value = self._TYPE_CONDITIONAL
        conditional.handle, conditional.type, conditional.size = handle, self._COND_WHILE, 1
        node, child = ctypes.c_void_p(), ctypes.c_void_p()
        self._call(
            "cudaGraphAddNode", ctypes.byref(node), ctypes.c_void_p(outer), None, ctypes.c_size_t(0), ctypes.byref(raw),
        )
        body = conditional.graphs[0]
        self._call(
            "cudaGraphAddChildGraphNode", ctypes.byref(child), ctypes.c_void_p(body), None, ctypes.c_size_t(0),
            ctypes.c_void_p(step_graph),
        )
        graph_exec = ctypes.c_void_p()
        self._call(
            "cudaGraphInstantiateWithFlags", ctypes.byref(graph_exec), ctypes.c_void_p(outer), ctypes.c_ulonglong(0),
        )
        return graph_exec.value

    def launch(self, graph_exec: int, stream: int) -> None:
        self._call("cudaGraphLaunch", ctypes.c_void_p(graph_exec), ctypes.c_void_p(stream))


class DeviceRun:
    """A slot's run: its state, its static columns, and what launches it.
    ``inputs``: static input -> its iteration input's column, staged before
    the run; ``stops``, ``rows``, ``spans``: the outputs kept per replay,
    ``[rows, limit, ...]`` each. ``launch`` is the host's one call.
    ``handle``: the WHILE condition's handle on the device, which the graph
    reads by address, so the run holds it as long as the graph lives."""

    def __init__(
        self, state: RunState, inputs: Mapping[str, torch.Tensor], launch: Callable[[], None], limit: int,
        handle: torch.Tensor,
    ):
        self.state = state
        self.handle = handle
        # set once the columns are sized, before the capture, which only writes them
        self.sealed = False
        self.inputs = inputs
        self.limit = limit
        # a column index per iteration (``results``)
        self._columns = torch.arange(limit, device=handle.device)
        self.launch = launch
        self.stops: dict[str, torch.Tensor] = {}
        self.rows: dict[str, torch.Tensor] = {}
        self.spans: dict[str, torch.Tensor] = {}

    def stage_columns(self, iteration: Mapping[str, torch.Tensor] | None) -> None:
        for name, column in (iteration or {}).items():
            self.inputs[name][:, :column.shape[1]].copy_(column)

    def keep(self, name_of: Mapping[str, torch.Tensor], kept: dict[str, torch.Tensor]) -> None:
        """Inside the body: a column of each of ``name_of``, this replay's."""
        for name, tensor in name_of.items():
            if name not in kept:
                if self.sealed:
                    raise RuntimeError(f"a run's column {name!r} was not sized before its capture")
                kept[name] = tensor.new_empty(tensor.shape[0], int(self.limit), *tensor.shape[1:])
            column_out(self.state, tensor, kept[name])

    def results(self, repeats: int) -> tuple[dict, dict, dict]:
        """After the launch: what ``AcceleratorGraphRunner.repeat`` returns, and
        the iterations that ran under ``RUN_REPLAYS``. A run that ended short
        of ``repeats`` (a stop, a yield, ``refill_at``) never wrote its columns
        past them: each repeats its last written one there, so whatever reads
        a column's last (the loop-back of the step speculated meanwhile, a
        node's ``postprocess``) reads the last iteration that ran."""
        state = self.state
        state.replays.copy_(state.index.expand_as(state.replays))
        ran = torch.minimum(self._columns[:repeats], state.index - 1)
        for column in (*self.rows.values(), *self.stops.values()):
            column[:, :repeats] = column.index_select(1, ran)
        stops = {name: column[:, :repeats] for name, column in self.stops.items()}
        stops[RUN_REPLAYS] = state.replays
        return (
            stops, {name: column[:, :repeats] for name, column in self.rows.items()},
            {name: column[:, :repeats] for name, column in self.spans.items()},
        )


def column_in(state: RunState, column: torch.Tensor, target: torch.Tensor) -> None:
    """Inside a body: this replay's column of ``column`` into ``target``."""
    target.copy_(column.index_select(1, state.index).reshape(target.shape))


def column_out(state: RunState, tensor: torch.Tensor, column: torch.Tensor) -> None:
    """Inside a body: ``tensor`` into this replay's column of ``column``."""
    column.index_copy_(1, state.index, tensor.unsqueeze(1).to(column.dtype))


def run_item(column: torch.Tensor, iteration: int, copies: "torch.cuda.Stream | None" = None) -> torch.Tensor:
    """One iteration of a row's column of a step that repeated, as what leaves
    the step. A view of the row, which is the step's own clone, unless its
    last dim strides by the run's width (a one-element item): ``contiguous``
    keeps that stride and a byte view refuses it, so that one is copied, on
    ``copies`` if given (``Worker._run_copies``). A copy per item was most of
    a Talker step's host time at N=4."""
    item = column[:, iteration]
    if item.dim() and item.stride(-1) == 1:
        return item
    with torch.cuda.stream(copies) if copies is not None else nullcontext():
        return item.clone(memory_format=torch.contiguous_format)
