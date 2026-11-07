

import dataclasses
import logging
import math
import os
import threading
import time
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from itertools import repeat
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Collection, Iterator, Mapping, NamedTuple, Sequence

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo, merge_publish_info
from mstar.distributed.communication import JointGroups, WorkerParallelGroups
from mstar.engine.accelerator_graph_runner import (
    AcceleratorGraphRunner,
    CapturePlan,
    PiecewiseAcceleratorGraphRunner,
    RunPart,
    autocast_scope,
    fail_if_graphs_required,
    output_named,
    plan_captures,
)
from mstar.engine.chunking import DEFAULT_MIN_CHUNK_TOKENS, ChunkingManagement, ChunkRange
from mstar.engine.connected_inputs import ConnectedInputs, active_guard
from mstar.engine.device_run import (
    RUN_COUNTS,
    RUN_DESCRIPTORS,
    RUN_REPLAYS,
    IterationWindow,
    RunAsk,
    TokenStream,
    run_item,
)
from mstar.engine.fp32_precision import Fp32Precision, cudnn_benchmark, fp32_precision_scope, node_fp32_precision
from mstar.engine.node_dtype import resolve_node_dtype
from mstar.engine.resources import (
    AdmitFailedReason,
    FullAdmitOutcome,
    NodeResourceSpec,
    PublishedInfo,
    Resource,
    ResourceReqConfig,
    SlotLease,
    StepContext,
    StepRunner,
    SubmoduleStep,
)
from mstar.engine.resources.base import EngineResourceInfo, KeyedWalks, build_resource, reachable_resources
from mstar.engine.resources.iteration_tables import STREAM_COUNTS, STREAM_TRANSITION, IterationTablesSpec, tables_key
from mstar.engine.resources.kv.config import instance_size, opening_prefix_caches, unrestorable
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.loopback_rows import LoopbackRows, LoopbackRowsSpec, rows_key
from mstar.engine.resources.lora.config import LoraBaseOnly, node_capture_key
from mstar.engine.resources.resident import (
    LOOP_COUNTER,
    NOT_ASKED,
    ResidentPlanner,
    build_planner,
    declared_streams,
    resident_asked,
    step_refusal,
    watch_table_reads,
    watched_counts,
)
from mstar.engine.resources.run_rows import RunRow, RunRowsSpec, run_rows_key
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.engine.resources.step import (
    ADMIT_OK,
    FULL_ADMIT_NOT_READY,
    AdmitOutcome,
    PerReqSetting,
)
from mstar.graph.loop_indices import LoopLayout
from mstar.graph.runtime.base import ExecutionUnit, GraphRuntime
from mstar.model.base import keyed_labels
from mstar.model.submodule_base import (
    EMPTY_INPUT_METADATA,
    ARNodeInputs,
    BatchedModelOutput,
    CarryPair,
    ChunkingPolicy,
    HostRows,
    InputMetadata,
    InputSeqLenInfo,
    IterationStream,
    LazyRequestStates,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    StepOrder,
    StopPolicy,
    engine_step,
    iteration_columns,
)
from mstar.platforms.compile import with_inductor_overrides
from mstar.profile.worker import ExecTimings
from mstar.utils import knobs
from mstar.utils.client_errors import client_error_text
from mstar.utils.device import DeviceCapability
from mstar.utils.h2d import to_device_nonblocking
from mstar.utils.ipc_format import OffloadDelta, ScheduleTPNode
from mstar.utils.profiler import (
    PhaseLaps,
    mark,
    nvtx_enabled,
    phase_end,
    phase_start,
    range_pop,
    range_push,
)

if TYPE_CHECKING:
    from mstar.engine.resources.kv.manager import KVManager
    from mstar.engine.resources.lora.pool import LoraAdapterPool
    from mstar.model.base import Model
    from mstar.worker.micro_scheduler import ScheduledBatch

logger = logging.getLogger(__name__)

# Block the GPU thread on its step's outputs before returning (1-step launch
# bound vs the worker's default 2-step). Off by default: it kills the
# GPU(N+1)/postprocess(N) overlap. Enable only where 2 steps overflow the CUDA
# launch queue and block a launch (machine/driver dependent).
_ENGINE_STEP_SYNC = os.environ.get("MSTAR_ENGINE_STEP_SYNC", "0") == "1"

# Publish resource state lazily: finalize_batch records a cheap snapshot per
# request and the worker finishes it only for the requests whose frame or
# completion carries it. Eager (0) builds the full publication for every
# request on every step; in a decode loop nothing reads it until the loop
# stops, and it was a quarter of the gpu thread's host time at batch 32.
_LAZY_PUBLISH = os.environ.get("MSTAR_LAZY_PUBLISH", "1") == "1"



@contextmanager
def _search_cap(submodule_mgmt, fraction: float | None, device) -> Iterator[None]:
    """Hold the caching allocator to ``gpu_memory_fraction`` while a node that
    asks for cuDNN's algorithm search (``capture_cudnn_benchmark``) is sized.
    The search tries workspaces as large as the device has free, and the
    allocator keeps them until sizing's ``empty_cache``, past the share a
    co-located worker plans around. A try past the cap fails and cuDNN
    skips that algorithm. Any other allocation past it fails too, so the
    boot fails when the node's eager step does not fit in the share."""
    if fraction is None or not submodule_mgmt.submodule.capture_cudnn_benchmark:
        yield
        return
    previous = torch.cuda.get_per_process_memory_fraction(device)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction(previous, device)


@contextmanager
def _outside_steps(submodule_mgmt) -> Iterator[None]:
    """A node's forwards run outside any step (compile warm-ups), with each of its LoRA
    pools as in a base-only step (``LoraAdapterPool.base_only``)."""
    with ExitStack() as stack:
        for pool in submodule_mgmt.lora_pools:
            stack.enter_context(pool.base_only())
        yield


def _last_iteration_loops_back(outputs: dict, columns: Collection[str], loop_back: Collection[str]) -> None:
    """A step that repeated keeps each of ``columns`` a column an iteration; a
    loop-back signal its node named after one hands on that column's last
    iteration, the one the next step continues from (a short device run's
    columns repeat its last, ``DeviceRun.results``)."""
    kept = {id(t) for name in columns for t in outputs.get(name, ())}
    for signal in loop_back:
        values = outputs.get(signal)
        if values and all(id(t) in kept for t in values):
            # a one-element iteration as a view of the row: no kernel a row a step
            outputs[signal] = [t.reshape(-1)[-1:] if t.dim() == 2 else run_item(t, -1) for t in values]


def node_lanes_of(walk_lanes: Mapping[tuple[str, str], int]) -> dict[str, int]:
    """node -> the one lane off the chain every walk of it in ``walk_lanes``
    runs on; a node with any walk on the chain, or on two lanes, is left out."""
    lanes: dict[str, set[int]] = {}
    for (node, _), lane in walk_lanes.items():
        lanes.setdefault(node, set()).add(lane)
    return {node: next(iter(held)) for node, held in lanes.items() if len(held) == 1 and 0 not in held}


def checkpoint_identity(path: str | Path) -> bytes:
    """Name the weights a cached page was produced under, without reading them.

    A checkpoint with no safetensors index falls back to its shard names and
    sizes, which catches a replaced shard but not one edited in place at the
    same length.
    """
    root = Path(path)
    index = root / "model.safetensors.index.json"
    parts: list[object] = []
    if index.is_file():
        parts.append(index.read_bytes())
    else:
        for shard in sorted(root.glob("*.safetensors")):
            parts += [shard.name, shard.stat().st_size]
    config = root / "config.json"
    if config.is_file():
        parts.append(config.read_bytes())
    return fingerprint(*parts)


class PagedKVCache(NamedTuple):
    key: str
    # the nodes on this engine that use it
    nodes: frozenset[str]
    cache: "KVManager"


class _MeasuredRow(NamedTuple):
    """What a prefix probe reads of a row before its inputs are prepared."""
    input_seq_len: int
    custom_pos_ids: None = None


class SlotRotation:
    """One rotation for a local component of shared resource buffers.

    Plan and GPU threads lease under one lock. The plan thread waits for
    commit_done before reusing the next slot; the cursor itself is not a fence.
    """

    def __init__(self, resources: Collection[Resource]):
        self.next_slot = 0
        self.lock = threading.Lock()
        self.needs_fence = any(res.force_double_buffer for res in resources)
        buffered = self.needs_fence or (
            knobs.pre_plan_enabled() and any(res.supports_preplan for res in resources)
        )
        self.count = max(2, int(os.environ.get("MSTAR_NUM_SLOTS", "2"))) if buffered else 1

    def lease(self) -> int:
        with self.lock:
            slot = self.next_slot
            self.next_slot = (slot + 1) % self.count
            return slot


@dataclass
class SubmoduleManagement:
    submodule: NodeSubmodule
    forward: Callable
    forward_batched: Callable
    joint_comm_group: JointGroups
    resources: dict[str, Resource]
    # cast and autocast dtype, None if the node opted out; resolved once at load
    node_dtype: torch.dtype | None = None
    accelerator_graph_runner: AcceleratorGraphRunner | None = None
    # the runner capture ran with, graphs or none: a bucket it didn't capture
    # runs eager, and the floor covers eager steps only up to its largest
    capture_runner: AcceleratorGraphRunner | None = None
    # the submodule's declared float32 settings; None keeps the process default
    fp32_precision: Fp32Precision | None = None
    # the node's request slots and device tables, where it runs on them
    resident: ResidentPlanner | None = None
    # walk -> why it stays host-planned, for each walk that loops the node and isn't resident
    resident_refusals: dict[str, str] = field(default_factory=dict)
    # walk -> the row outputs a step that repeats keeps a column of per replay
    run_columns: dict[str, frozenset[str]] = field(default_factory=dict)
    # walk -> how a step that repeats carries each loop-back signal between replays
    carries: dict[str, dict[str, CarryPair]] = field(default_factory=dict)
    # walk -> why its preprocess can't be skipped between replays (checked at load)
    carry_refusals: dict[str, str] = field(default_factory=dict)
    # walk -> the step its widest capture declares, for each walk that may repeat
    template_steps: dict[str, SubmoduleStep | None] = field(default_factory=dict)
    # the node's loop-back values kept on the device by request, where it keeps any
    loopback_rows: LoopbackRows | None = None
    # walk -> loop-back signal -> its pair, kept in those rows beside a
    # ``DeviceLoopback``'s, for each walk whose config asks
    row_signals: dict[str, dict[str, CarryPair]] = field(default_factory=dict)
    # walk -> the token stop that ends its WHILE runs, where asked
    # None denotes an iteration-input Loop bounded solely by its graph cap.
    device_stops: dict[str, StopPolicy | None] = field(default_factory=dict)
    # walk -> a run's share of its rows done that ends it, where asked
    refills: dict[str, float | None] = field(default_factory=dict)
    # walk -> the stop buffer whose tokens go to the client as they land, and a run's limit
    device_stream_tokens: dict[str, tuple[str, int]] = field(default_factory=dict)
    # streamed signal -> the walks whose steps put it in the node's ring (``step_repeats: stream_queue``)
    rings: dict[str, frozenset[str]] = field(default_factory=dict)

    # Shared by local nodes whose resource buffers can alias.
    slot_rotation: SlotRotation = field(init=False)

    # label -> PiecewiseCudaGraphRunner for inner-loop capture; spread into
    # ModelInputsFromEngine so the submodule's forward can look them up
    piecewise_runners: dict[str, PiecewiseAcceleratorGraphRunner] = field(
        default_factory=dict
    )
    # the LoRA pools bound to the node's modules, set at bind
    lora_pools: tuple = ()

    def __post_init__(self):
        self.slot_rotation = SlotRotation(self.resources.values())

    @property
    def num_slots(self) -> int:
        return self.slot_rotation.count

    @property
    def needs_slot_fence(self) -> bool:
        return self.slot_rotation.needs_fence

    @property
    def next_slot(self) -> int:
        return self.slot_rotation.next_slot

    def lease_slot(self) -> int:
        return self.slot_rotation.lease()

    def set_piecewise_slot(self, slot: int) -> None:
        """Point the region runners at this batch's slot, on the GPU thread just
        before its forward. Set here rather than at lease time: leasing runs
        ahead of the forward (pre-plan reserves N+1's slot while N is in flight),
        so a lease-time write would race the replay reading ``_current_slot``."""
        for runner in self.piecewise_runners.values():
            runner.set_slot(slot)

def _window_tables(
    specs, inputs: Mapping[str, list] | None, metadata, *, start: int = 0,
) -> dict[str, IterationStream]:
    """A row's stream windows: each streamed iteration input's items this
    step, ``[items, ...]`` (one item is collated as itself)."""
    if not specs:
        return {}
    inputs = inputs or {}
    chunks = getattr(metadata, "stream_chunks", {}) or {}
    out = {}
    for spec in specs:
        tensors = inputs.get(spec.source)
        chunk = chunks.get(spec.source)
        if chunk is not None:
            if chunk.reservation is not None:
                transitions = chunk.reservation.transitions[start:]
                if not transitions:
                    raise ValueError("retained stream input has no frozen policy transitions")
                table = tensors[0].reshape(chunk.num_items, -1) if tensors and chunk.num_items else None
                values = []
                empty = False
                for transition in transitions:
                    part = transition.chunk
                    if not part.num_items:
                        empty = True
                        continue
                    if empty or table is None:
                        raise ValueError("retained stream inputs cannot have data after an empty tail")
                    offset = part.start_offset - chunk.start_offset
                    value = table[offset:offset + part.num_items]
                    if value.numel() != math.prod(spec.shape):
                        raise ValueError(f"stream input {spec.source!r} does not fit its declared iteration shape")
                    values.append(value.reshape(spec.shape))
                packed = torch.stack(values) if values else None
                tail = transitions[0].chunk.tail_offset if transitions else chunk.tail_offset
                out[spec.source] = IterationStream(packed, tail)
            else:
                table = tensors[0].reshape(chunk.num_items, *spec.shape) if tensors and chunk.num_items else None
                out[spec.source] = IterationStream(table, chunk.tail_offset)
    return out


def _refuse_counter_read(node_name: str, ctx: StepContext, reads: Collection[str], stage: str) -> None:
    """A step that repeats runs its ``prepare_inputs`` and ``preprocess``
    once, so a value they derive from the loop counter would hold for every
    replay; an iteration's value belongs in ``ITERATION_INPUTS``."""
    if ctx.repeats > 1 and LOOP_COUNTER in reads:
        if guard := active_guard():
            guard.refuse(f"{stage} reads a Loop counter that later iterations do not renew")
        # the step's rows fail, not the worker: a branch only a real row takes can land here
        raise ValueError(
            f"{node_name}/{ctx.graph_walk}: a repeating step's {stage} read "
            f"{LOOP_COUNTER}, which its later replays don't renew; declare the value in ITERATION_INPUTS"
        )


@dataclass
class ForwardPassInfoWrapper:
    """
    Stop-gap solution: the dynamic loop iters in CurrentForwardPassInfo may
    be used in two places at once: in _postprocess_batch (for batch N), and
    in the forward pass for batch N+1, which require different iterations.
    The InputMetadata is built per executing batch, so its value is guaranteed
    to be up-to-date.

    TODO: the long-term solution is to maybe move the dynamic loop iters off
    of CurrentForwardPassInfo entirely.
    """
    info: CurrentForwardPassInfo
    metadata: InputMetadata

    # The attributes every per-row step reads (prepare_inputs, the stop
    # check) resolve without the ``__getattr__`` fallback, which costs a
    # failed lookup plus a Python call per read.
    @property
    def dynamic_loop_iter_counts(self):
        return watched_counts(self.metadata.dynamic_loop_iter_counts)

    @property
    def resource_configs(self):
        return self.info.resource_configs

    @property
    def max_tokens(self):
        return self.info.max_tokens

    @property
    def graph_walk(self):
        return self.info.graph_walk

    @property
    def step_metadata(self):
        return self.info.step_metadata

    def __getattr__(self, name):
        if name == "dynamic_loop_iter_counts":
            return watched_counts(self.metadata.dynamic_loop_iter_counts)
        return getattr(self.info, name)


@dataclass
class ExecutingBatch:
    node_name: str

    # Keyed by the worker's integer rid handle. CUDA-graph capture pads with
    # its own rows, which carry negative handles from the same space (see
    # cuda_graph_runner.dummy_rid_handle), so a padded batch is still int-keyed
    # throughout.
    per_request_info: Mapping[int, CurrentForwardPassInfo]
    step_context: StepContext

    running_batched: bool = False

    # Enables double-buffering for resources with non-blocking H2D
    slot: int | None = None
    # how its run on the device began (``RunState.started``), for one that went on one
    run_started: torch.Tensor | None = None
    # Descriptor readers owned by this ordinary logical step, until collection.
    run_rows: dict[int, RunRow] = field(default_factory=dict)

    # Selects among a walk's capture buckets; matches SubmoduleStep.cg_key_info
    cg_key_info: Any | None = None

    # Populated on preplan
    preplan_event: Any | None = None
    # The rids the staged plan was built over. The plan is theirs exactly —
    # order included — so it is stale the moment this stops matching
    # ``request_ids`` (a request dropped while threading outputs or preparing).
    preplanned_rids: tuple[int, ...] | None = None

    # Declared once for the batch (pre-plan declares it first when it runs)
    # and driven from here on
    step: SubmoduleStep | None = None
    # the step the batch committed last: a batch run row by row commits once per row
    committed: SubmoduleStep | None = None

    # {request_id: {input_name: [tensor]}}, what prepare_inputs reads
    per_request_input_tensors: Mapping[str, NameToTensorList] = field(
        default_factory=dict
    )
    # rids for which this step ends every finite stream into the node; the
    # consumer flushes what it held back
    final_stream_rids: set[int] = field(default_factory=set)
    # rid -> this step's InputMetadata; rids without any are absent
    per_request_input_metadata: Mapping[str, InputMetadata] = field(default_factory=dict)

    # rids for which this step ends every finite stream into the partition, or
    # whose stop check returned END_PARTITION; this step reports the partition done
    partition_done_rids: set[int] = field(default_factory=set)

    # rid -> the part of a chunked row's input this step runs
    chunk_ranges: Mapping[int, ChunkRange] = field(default_factory=dict)
    # rows whose whole input this step prepared (`_prepare_chunk`), on their
    # first chunk: a refused admit drops it, so the retry checks it again
    first_chunk_rids: set[int] = field(default_factory=set)

    # Populated on batch preparation
    inputs: list[NodeInputs] | None = None
    # the rows ``inputs`` holds, in order, as prepared; ``seq_len_of`` maps
    # them to their tokens on first use
    prepared_rids: tuple[int, ...] = ()
    _seq_lens: dict[int, int] | None = field(default=None, repr=False)
    # the one object every row shares, when the step was prepared with the
    # submodule's ``uniform_row_inputs``
    uniform_inputs: NodeInputs | None = None
    # the step's iteration inputs (``NodeSubmodule.ITERATION_INPUTS``), a
    # column a replay: ``[rows, repeats, ...]``, staged after ``preprocess``
    iteration_columns: dict[str, torch.Tensor] | IterationWindow | None = None
    # rid -> the stream windows its row took (an item a replay), whose items
    # past the step's settled count go back to their streams
    stream_windows: dict[int, list] = field(default_factory=dict)
    # rids the submodule declined this step — e.g. a speculatively scheduled
    # flow step for a request already past its own max iters
    skipped_rids: set[int] = field(default_factory=set)
    # rids whose whole walk this node's resources already hold. They run no
    # forward, but their node still completes so the request reaches its next walk.
    cached_rids: set[str] = field(default_factory=set)
    # rid -> error, for per-rid stages that raised. The rid leaves the batch;
    # the rest of it runs.
    failed_requests: dict[int, str] = field(default_factory=dict)
    # A lockstep step whose rows' input reads every rank checks before
    # preparing it: the leader's rows in its order (see
    # ``Worker._drop_rows_whose_read_failed``); None for no check
    read_check_rids: tuple[int, ...] | None = None

    admit_error: AdmitFailedReason | None = None
    # the resource that ran out, so an eviction can be scoped to it
    failed_resource: str | None = None

    # Every output signal of this step stays on the device (the node's
    # ``device_loopback_signals``): collect builds no per-request output
    # dicts and the per-request ``postprocess`` is skipped; the stop check,
    # the inline emit and the speculation splice read the rows instead.
    rows_only: bool = False

    # A mixed step's rows whose own walk keeps every output signal on the
    # device: such a row loops back as it would in a step of its walk alone,
    # while the step's other rows route tensors.
    loopback_rids: frozenset = frozenset()
    # per row, whether its last step here kept its loop-back values on the device
    kept_rows: list[bool] | None = None
    # where a run streaming its tokens leaves them as they land
    token_stream: TokenStream | None = None
    # streamed signal -> rid -> the ring position its first item of the step took,
    # and the positions each took (``Engine._reserve_rings``)
    ring_bases: dict[str, dict[int, int]] = field(default_factory=dict)
    ring_width: int = 1

    # This step's outputs, published as soon as the forward has been
    # submitted — the tensors exist then, even though their values land later.
    outputs: BatchedModelOutput = field(default_factory=BatchedModelOutput)

    # Durable resource state produced by this worker in this batch. This is
    # deliberately narrower than per_request_info's inherited aggregate.
    resource_publish_info: dict[str, dict[str, PublishedInfo]] = field(
        default_factory=dict
    )
    # What finalize_batch snapshotted instead of publishing (MSTAR_LAZY_PUBLISH);
    # ``Engine.materialize_publish`` turns entries into resource_publish_info.
    publish_snapshots: dict[str, dict[str, Any]] | None = None

    # The next step reads N's outputs, and plans against N's committed state.
    # Two separate dependencies, so two events: whoever prepares N+1 can start
    # threading N's outputs while N is still committing.
    outputs_ready: threading.Event = field(default_factory=threading.Event)
    commit_done: threading.Event = field(default_factory=threading.Event)

    # Recorded on the default stream once this step's GPU work is submitted;
    # what a reader of the output values has to wait on.
    completion_event: Any | None = None

    # Set by exec right before the forward's CUDA launch (where torch drops the
    # GIL). A worker that submitted this batch to the GPU thread and then wants
    # to do its own Python work waits on this first, so its GIL grab doesn't
    # stall the GPU thread's path to graph.replay(). None => nobody is waiting.
    launch_started_event: threading.Event | None = None

    # Per-step wall-clock, for the worker's profiler
    exec_timings: ExecTimings = field(default_factory=ExecTimings)

    def __post_init__(self):
        # Sparse metadata: only rids with stream chunks or loop counters get
        # an entry, so default like the .get() at prepare_inputs does. Built
        # with C-level maps: this runs for every step, on every row.
        rids = self.per_request_info.keys()
        self.per_request_info_wrapped = dict(zip(rids, map(
            ForwardPassInfoWrapper, self.per_request_info.values(),
            map(self.per_request_input_metadata.get, rids, repeat(EMPTY_INPUT_METADATA)),
        ), strict=True))

    @property
    def request_ids(self):
        return self.step_context.request_ids

    @request_ids.setter
    def request_ids(self, rids):
        if tuple(rids) != tuple(self.step_context.request_ids):
            # any padded list was built over the old set, so it now names rids
            # this step no longer runs; exec rebuilds it under a lease, and
            # until then `padded_request_ids` falls back to these
            self.step_context.set_padded_rids(None)
        self.step_context.request_ids = rids

    @property
    def graph_walk(self) -> str:
        return self.step_context.graph_walk

    def register_prepare_batch(self, inputs: list[NodeInputs], uniform: NodeInputs | None = None):
        self.inputs = inputs
        self._seq_lens = None
        self.uniform_inputs = uniform

    def seq_len_of(self, rid: int) -> int:
        """The tokens ``rid`` runs this step; -1 before it is prepared."""
        if self._seq_lens is None:
            if self.inputs is None:
                return -1
            self._seq_lens = {
                r: inp.input_seq_len
                for r, inp in zip(self.prepared_rids, self.inputs, strict=True)
            }
        return self._seq_lens.get(rid, -1)

    def release_waiters(self):
        """Let anything waiting on this step proceed.

        Called on every exit from ``exec``, so a step that raised before
        publishing outputs or committing doesn't strand the thread preparing
        the next one.
        """
        self.outputs_ready.set()
        self.commit_done.set()

    def register_failure(self, rid: int, error: Exception, walk: str | None = None):
        self.failed_requests[rid], _ = client_error_text(
            error,
            f"request {rid} failed (node={self.node_name}, walk={walk or self.graph_walk})",
            log=logger,
        )

    def drop_rids(self, rids: set[int]):
        """Take rids out of the step, leaving the rest of the batch intact."""
        if not rids:
            return
        self.request_ids = [rid for rid in self.request_ids if rid not in rids]
        # A row that does not run this step must not end its streams on it:
        # the worker would stop its loop and finalize a row it no longer has.
        self.final_stream_rids.difference_update(rids)
        self.partition_done_rids.difference_update(rids)
        self.per_request_info = {
            rid: info for rid, info in self.per_request_info.items()
            if rid not in rids
        }
        self.per_request_info_wrapped = {
            rid: info for rid, info in self.per_request_info_wrapped.items()
            if rid not in rids
        }

    def select_rows(self, rids: Sequence[int]) -> "ExecutingBatch":
        """A logical collection view; the original admitted receipt keeps its owners."""
        wanted = set(rids)
        inputs = dict(zip(self.request_ids, self.inputs, strict=True))
        view = dataclasses.replace(
            self, step_context=dataclasses.replace(self.step_context, request_ids=list(rids),
                                                  _padded_request_ids=None, _real_rids=None),
            inputs=[inputs[rid] for rid in rids], prepared_rids=tuple(rids), _seq_lens=None,
            per_request_info={rid: self.per_request_info[rid] for rid in rids},
            per_request_input_tensors={rid: value for rid, value in self.per_request_input_tensors.items()
                                       if rid in wanted},
            per_request_input_metadata={rid: value for rid, value in self.per_request_input_metadata.items()
                                        if rid in wanted},
            final_stream_rids=self.final_stream_rids & wanted,
            partition_done_rids=self.partition_done_rids & wanted,
            cached_rids=self.cached_rids & wanted,
            failed_requests={rid: error for rid, error in self.failed_requests.items() if rid in wanted},
            run_rows={rid: row for rid, row in self.run_rows.items() if rid in wanted},
            stream_windows={rid: edges for rid, edges in self.stream_windows.items() if rid in wanted},
            ring_bases={name: {rid: base for rid, base in bases.items() if rid in wanted}
                        for name, bases in self.ring_bases.items()},
            loopback_rids=self.loopback_rids & wanted,
            kept_rows=None if self.kept_rows is None else [
                dict(zip(self.request_ids, self.kept_rows, strict=True))[rid] for rid in rids
            ],
        )
        view.step_context.run_counts = self.step_context.run_counts
        view.per_request_info_wrapped = {rid: self.per_request_info_wrapped[rid] for rid in rids}
        return view

    def register_admit_error(
        self, reason: AdmitFailedReason, failed_resource: str | None = None,
    ):
        self.admit_error = reason
        self.failed_resource = failed_resource

    def set_slot(self, slot: int):
        self.slot = slot
        if self.step_context is not None:
            self.step_context.slot = slot

    def lease_slot(self, slot_lease: SlotLease):
        self.step_context.slot_lease = slot_lease
        if self.slot is None:
            self.set_slot(slot_lease.slot)


@dataclass
class ExecutionGroup:
    """One graph claim, its ordinary members, and their held launch bindings."""
    unit: ExecutionUnit
    claim: int
    members: tuple[tuple["ScheduledBatch", ExecutingBatch], ...]
    # Ordinary ready member that triggered this attempt; members stay in graph order.
    trigger: int = 0
    submitted: bool = False
    acknowledged: bool = False
    # Immutable graph facts captured with the claim, in canonical member order.
    finished_rids: tuple[frozenset[int], ...] = ()
    refusal: str | None = None
    parts: list[RunPart] = field(default_factory=list)
    initial_copies: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    capacity: int = 1


class Engine:
    _walk_runs: frozenset[str] = frozenset()
    # node -> its lane, for nodes off the chain, and each such lane's stream;
    # set at construction and load, empty for an engine with no lanes
    _walk_lanes: Mapping[tuple[str, str], int] = MappingProxyType({})
    _node_lanes: Mapping[str, int] = MappingProxyType({})
    _stage_lock: "threading.Lock | None" = None
    # (node, walk) -> the Loop it runs in; set at warmup
    _loops: Mapping[tuple[str, str], LoopLayout] = MappingProxyType({})
    _lane_streams: Mapping[int, Any] = MappingProxyType({})

    @property
    def device_capability(self) -> DeviceCapability:
        capability = getattr(self, "_device_capability", None)
        if capability is None:
            capability = self._device_capability = DeviceCapability.of(self._device or "cpu")
        return capability

    def __init__(
        self, graph_runtime: GraphRuntime,
        autocast_dtype=torch.bfloat16,
        enable_nvtx: bool = False,
        enable_profile: bool=False,
        gpu_memory_fraction: float | None = None,
        device_capability: DeviceCapability | None = None,
        walk_lanes: Mapping[tuple[str, str], int] | None = None,
    ):
        assert gpu_memory_fraction is None or 0 < gpu_memory_fraction <= 1, (
            f"gpu_memory_fraction is a fraction of the device, got {gpu_memory_fraction}"
        )
        self._device = None
        self._device_capability = device_capability
        # (node, walk) -> its lane, as ``EngineManager.build`` placed them
        self._walk_lanes = dict(walk_lanes or {})
        # node -> the one lane off the chain every walk of it runs on; a node
        # with walks on the chain is the chain's, whatever its other walks
        self._node_lanes = node_lanes_of(self._walk_lanes)
        # the plan thread's stage of the chain's next step, against a step off the
        # chain that plans the same resources on the GPU thread (``_beside_chain``)
        self._stage_lock = threading.Lock() if any(self._walk_lanes.values()) else None
        self._autocast_dtype = autocast_dtype
        # the share of the device this worker's allocations may hold once
        # capture is done, None for whatever is free
        self._gpu_memory_fraction = gpu_memory_fraction
        self._resources: dict[str, Resource] = {}
        # resource key -> the nodes on this engine that use it
        self._resource_nodes: dict[str, frozenset[str]] = {}
        # (node, combined walk) -> the walks whose rows a step of it runs last
        self._split_prompt_walks: dict[tuple[str, str], frozenset[str]] = {}
        self._submodules: dict[str, SubmoduleManagement] = {}
        self._runner: StepRunner = None
        # node -> the event after its plan stream's latest pre-plan; see
        # ``_order_after_pre_plans``
        self._pre_plan_done: dict[str, Any] = {}
        self._graph_runtime = graph_runtime
        # chunked prefill: per-walk policies and budgets, and rows mid-prefill
        self._chunking = ChunkingManagement(self._submodules)

        # node -> (offloaded, reloaded) since the last ``take_resident_delta``.
        # Only a TP leader drains this, to replicate its resident set onto the
        # rest of the instance; see ``ScheduleTPNode``.
        self._resident_delta: dict[str, OffloadDelta] = {}
        # Nodes this rank leads (journals its moves for the others) and nodes it
        # follows (may not move pages at all — see ``offload_request``). A node
        # that is not sharded is in neither: it owns its own eviction.
        self._tp_leader_nodes: set[str] = set()
        self._tp_follower_nodes: set[str] = set()

        self._enable_nvtx = enable_nvtx
        self._enable_profile = enable_profile

    def load_model(
        self,
        submodules: dict[str, NodeSubmodule],
        specs: list[NodeResourceSpec],
        parallel_groups: WorkerParallelGroups,
        device: torch.device,
        transfer_engine_info: TransferEngineInfo,
        kv_cache_type=None,
        model: "Model | None" = None,
        row_nodes: Collection[str] = (),
        joining: Mapping[str, int] | None = None,
    ):
        """``row_nodes``: the nodes whose config asks to keep their loop-back
        signals in device rows, beside those that declare a ``DeviceLoopback``."""
        self._device = device
        if self._device_capability is None or self._device_capability.device != device:
            self._device_capability = DeviceCapability.of(device)
        self._enable_nvtx = nvtx_enabled(self._enable_nvtx, device)
        if kv_cache_type is None:
            kv_cache_type = self._autocast_dtype

        node_names = set(submodules.keys())
        node_to_resources = {}
        # resource -> every node declaring it, on this worker or another
        declared_by: dict[str, frozenset[str]] = {}
        streamed_consumers = {
            consumer for walk in self._walk_runs for unit in self._graph_runtime.execution_units(walk)
            for _, consumer, _ in unit.streaming_edges
        }
        specs = [*specs, *self._row_specs(submodules, row_nodes),
                 *self._iteration_specs(submodules, streamed_consumers)]
        for node, limit in (joining or {}).items():
            if node not in submodules:
                continue
            if device.type != "cuda" or parallel_groups.get_joint_group_for_node(node).world_size != 1:
                raise ValueError(f"{node}: join_rows needs one CUDA rank")
            configs = submodules[node].get_accelerator_graph_configs(device, 1)
            rows = max((max(c.capture_batch_sizes or AcceleratorGraphRunner.CAPTURE_BATCH_SIZES)
                        for c in configs), default=0)
            if not rows:
                raise ValueError(f"{node}: join_rows needs whole-forward captures")
            specs.append(RunRowsSpec(resource_key=run_rows_key(node), nodes={node}, rows=rows,
                                     capacity=rows * (limit + 1), binding_width=2))
        specs_by_key = resolve_spec_dependencies(specs)
        instance = {node: parallel_groups.get_joint_group_for_node(node).world_size for node in node_names}
        checkpoint = model.checkpoint_path() if model is not None else None
        # a cache only another engine holds counts here at one rank: its own engine sizes it
        prefix_caches = opening_prefix_caches(specs, instance, checkpoint_named=checkpoint is not None)

        self._classify_tp_roles(node_names, parallel_groups)

        for spec in specs:
            relevant_nodes = spec.nodes & node_names
            if len(relevant_nodes) == 0:
                continue # resource not needed

            # A spec is a logical resource identity and may span replicas on
            # different workers (for example BAGEL's three CFG branches).
            # This Engine constructs only the local instance, whose consumers
            # validate_deployment checked share one parallel group.
            joint_comm_group = parallel_groups.get_joint_group_for_node(min(relevant_nodes))
            self._resources[spec.resource_key] = build_resource(
                spec,
                EngineResourceInfo(
                    device=device,
                    joint_comm_group=joint_comm_group,
                    transfer_engine_info=transfer_engine_info,
                    kv_dtype=kv_cache_type,
                    compute_dtype=self._compute_dtype(submodules, relevant_nodes),
                    needs_remote_transfer=(
                        parallel_groups.resource_needs_remote_transfer(
                            spec.nodes, relevant_nodes
                        )
                    ),
                    nodes=frozenset(relevant_nodes),
                    dependencies={
                        key: specs_by_key[key] for key in spec.depends_on()
                    },
                    max_rows_in_combined=self._max_rows_in_combined(
                        spec.resource_key, relevant_nodes, submodules, model,
                    ),
                    prefix_caches=prefix_caches,
                ),
            )

            for node in relevant_nodes:
                node_to_resources.setdefault(node, []).append(spec.resource_key)
            self._resource_nodes[spec.resource_key] = frozenset(relevant_nodes)
            self._add_split_row_order(spec, relevant_nodes)
            declared_by[spec.resource_key] = frozenset(spec.nodes)

        # rid -> (node, walk) of its last step here when that step was rows-only,
        # i.e. the sampler's slot master holds the token its next step reads
        self._loopback_steps: dict[int, tuple[str, str]] = {}
        # one stream per lane off the chain, made once; everything a lane issues goes on it (``on_lane``)
        self._lane_streams = {
            lane: self.device_capability.graph_backend.new_stream()
            for lane in sorted(set(self._walk_lanes.values()) - {0})
        }
        self._runner = StepRunner(
            self._resources,
            # every node, including one that owns nothing (Code2Wav) — an
            # absent node would fall back to the full sweep
            node_resources={n: node_to_resources.get(n, []) for n in node_names},
            enable_nvtx=self._enable_nvtx,
            declared_by=declared_by,
            on_lane=self._lane_contexts(node_names, declared_by, prefix_caches),
        )
        # node -> the nodes whose pre-plans its steps wait for (``_order_after_pre_plans``)
        self._plan_peers = self._plan_peers_of({n: node_to_resources.get(n, ()) for n in node_names})

        self._open_prefix_caches(specs_by_key, model, instance, prefix_caches, checkpoint)

        for node_name, submodule in submodules.items():
            # Inference only. `exec` is under no_grad, but `prepare_inputs` and
            # `postprocess_batch` are not, so anything they derive from a
            # parameter would build a graph and pin its intermediates — for the
            # whole request, once stashed in a `PerRequestState`.
            submodule.requires_grad_(False)
            resources = {
                label: self._resources[label] for label in node_to_resources.get(node_name, [])
            }
            self._submodules[node_name] = SubmoduleManagement(
                submodule=submodule,
                forward=submodule.forward,
                forward_batched=submodule.forward_batched,
                joint_comm_group=parallel_groups.get_joint_group_for_node(node_name),
                resources=resources,
                # the dtype the engine manager cast this node to
                node_dtype=resolve_node_dtype(submodule, self._autocast_dtype),
                fp32_precision=node_fp32_precision(node_name, submodule),
            )
            submodule.bind_node_resources(resources)
            self._submodules[node_name].lora_pools = tuple(
                resources[key] for key in getattr(submodule, "lora_keys", ())
            )
            if (key := rows_key(node_name)) in resources:
                self._submodules[node_name].loopback_rows = resources[key]
                submodule.loopback_rows_key = key
            if (key := tables_key(node_name)) in resources:
                submodule.iteration_tables = resources[key]

        self._share_slot_rotations()

    @staticmethod
    def _row_specs(submodules: Mapping[str, NodeSubmodule], row_nodes: Collection[str]) -> list[LoopbackRowsSpec]:
        """The rows of each node that reads its token back on the device
        (see DeviceLoopback) or whose config asks; a model declares none."""
        if not knobs.device_loopback_enabled():
            return []
        return [
            LoopbackRowsSpec(resource_key=rows_key(name), nodes={name}) for name, submodule in submodules.items()
            if name in row_nodes or getattr(submodule, "device_loopback", None) is not None
        ]

    @staticmethod
    def _iteration_specs(
        submodules: Mapping[str, NodeSubmodule], streamed_consumers: Collection[str] = (),
    ) -> list[IterationTablesSpec]:
        """Input/policy tables only for nodes whose resolved execution reads them."""
        return [
            IterationTablesSpec(resource_key=tables_key(name), nodes={name}) for name, submodule in submodules.items()
            if getattr(submodule, "ITERATION_INPUTS", None) or name in streamed_consumers
        ]

    def _build_resident_planner(self, node_name: str, runner: AcceleratorGraphRunner, looped: Collection[str]) -> None:
        """Put each walk in ``looped`` (those that loop ``node_name``) on
        resident tables if every owner of a fact it needs agrees, and keep why
        each other one stays host-planned (`resident_refusal`). After every
        runner has claimed its capture buffers, which a template's plan reads,
        and before any capture, which holds the tables' addresses."""
        if not looped:
            return
        submodule_mgmt = self._submodules[node_name]
        refusals = submodule_mgmt.resident_refusals
        if not resident_asked(submodule_mgmt.submodule):
            refusals.update(dict.fromkeys(looped, NOT_ASKED))
            return
        steps = {}
        for walk in sorted(looped):
            refusal = runner.resident_refusal(walk)
            if refusal is None:
                steps[walk] = runner.template_step(walk)
                refusal = self._keyed_resident_refusal(submodule_mgmt, runner, walk, steps[walk])
                if refusal is not None:
                    del steps[walk]
            if refusal is not None:
                refusals[walk] = f"runner: {refusal}"

        def probe(walk: str) -> str | None:
            # a resident step's tables fill these in its graph, after its preprocess
            with watch_table_reads() as reads:
                runner.prepare_template(walk)
            read = next((r for r in reads if r != LOOP_COUNTER), None)
            return f"node: its captured preprocess reads {read}, which the graph fills after it" if read else None

        if not steps:
            return
        resources = submodule_mgmt.resources
        submodule_mgmt.resident, refused = build_planner(
            node_name, steps,
            {key: resources[key] for key in self._runner.order if key in resources},
            self._resource_nodes, self._device, runner.max_capture_bs, probe,
        )
        refusals.update(refused)
        runner.bind_resident(submodule_mgmt.resident)

    def _keyed_resident_refusal(
        self, mgmt: SubmoduleManagement, runner: AcceleratorGraphRunner, walk: str, widest: SubmoduleStep | None,
    ) -> str | None:
        """Why a capture key of ``walk`` other than its widest bucket's can't run
        resident, or None: each key's captures declare their own step, and every
        one has to be a resident step over the same streams (a capture that
        isn't stops the worker when it is made)."""
        keys = [key for key in self._runner.order if key in mgmt.resources]
        if widest is None or step_refusal(keys, widest) is not None:
            return None  # the planner refuses the walk on its widest step itself
        want = declared_streams(keys, widest)
        for cg_key, step in runner.keyed_template_steps(walk).items():
            if (why := step_refusal(keys, step)) is not None:
                return f"its capture keyed {cg_key!r}: {why}"
            if declared_streams(keys, step) != want:
                return f"its capture keyed {cg_key!r} declares other streams than its widest"
        return None

    def resident_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """Why steps of ``graph_walk`` on ``node_name`` plan on the host, or
        None if they run on resident tables. Settled at load."""
        mgmt = self._submodules[node_name]
        if mgmt.resident is not None and graph_walk in mgmt.resident.walks:
            return None
        return mgmt.resident_refusals.get(graph_walk, "its node isn't in a Loop of it")

    def paged_kv_caches(self) -> list[PagedKVCache]:
        """Every paged KV cache this engine built, with the nodes that use it."""
        from mstar.engine.resources.kv.manager import KVManager

        return [
            PagedKVCache(key, self._resource_nodes[key], resource)
            for key, resource in self._resources.items()
            if isinstance(resource, KVManager)
        ]

    def lora_pools(self) -> list["LoraAdapterPool"]:
        """Every LoRA adapter pool this engine built, for runtime adapter loads."""
        from mstar.engine.resources.lora.pool import LoraAdapterPool

        return [r for r in self._resources.values() if isinstance(r, LoraAdapterPool)]

    def _compile_submodules(self) -> None:
        """Apply torch.compile to submodule forward paths.

        Compiles each submodule's ``forward`` and ``forward_batched`` with the
        default mode (fullgraph=False, dynamic=None), which in general provides
        performance gains without frequent slow recompiles.

        A node in ``_item_encoders`` is compiled with dynamic=True instead. Under
        the default it recompiles on its second input shape and the new graph
        rounds differently, so the KV cached from its earlier outputs would no
        longer match what it computes. A node in ``_item_encoders`` whose submodule
        sets ``marks_dynamic_dims`` keeps the default, since the submodule's
        `prepare_inputs` marks the dimensions that vary.
        """
        if not self.device_capability.is_available:
            return

        for node_name, submodule_mgmt in self._submodules.items():
            submodule = submodule_mgmt.submodule
            item_encoder = node_name in self._item_encoders

            if getattr(submodule, "disable_torch_compile", False):
                logger.info("Engine: torch.compile disabled for %s (submodule opt-out)", node_name)
                continue

            dynamic = True if item_encoder and not getattr(submodule, "marks_dynamic_dims", False) else None
            try:
                # no token count, these serve eager batches of any size
                submodule_mgmt.forward = with_inductor_overrides(torch.compile(
                    submodule.forward,
                    fullgraph=False,
                    dynamic=dynamic,
                ))
                submodule_mgmt.forward_batched = with_inductor_overrides(torch.compile(
                    submodule.forward_batched,
                    fullgraph=False,
                    dynamic=dynamic,
                ))
                logger.info("Engine: torch.compile applied to %s", node_name)
            except Exception:
                logger.warning(
                    "Engine: torch.compile failed for %s, using eager mode",
                    node_name, exc_info=True
                )

    def _compute_dtype(
        self, submodules: Mapping[str, NodeSubmodule], nodes: Collection[str],
    ) -> torch.dtype | None:
        """The dtype ``nodes`` were cast to and run in, or None if they differ or one keeps
        its own (``disable_autocast``)."""
        dtypes = {resolve_node_dtype(submodules[n], self._autocast_dtype) for n in nodes}
        return dtypes.pop() if len(dtypes) == 1 else None

    def _compile_regions(self) -> None:
        """Compile and warm each node's ``compile_regions`` with dynamic=True
        (a whole-forward compile uses dynamic=None) in torch.compile's default
        mode: the compile level (``MSTAR_COMPILE_LEVEL``) tunes captured
        forwards, whose shapes are fixed per bucket."""
        if not self.device_capability.is_available:
            return
        for node_name, submodule_mgmt in self._submodules.items():
            submodule = submodule_mgmt.submodule
            regions = submodule.compile_regions()
            if regions and not submodule.disable_torch_compile:
                raise ValueError(
                    f"{node_name} declares compile_regions but its forward is compiled whole; "
                    "set disable_torch_compile"
                )
            for path, examples in regions.items():
                owner_path, _, name = path.rpartition(".")
                owner = submodule
                for part in filter(None, owner_path.split(".")):
                    owner = getattr(owner, part)
                target = getattr(owner, name)
                if isinstance(target, torch.nn.Module):
                    owner, name, target = target, "forward", target.forward
                # an instance attribute shadows the method for every caller
                setattr(owner, name, with_inductor_overrides(torch.compile(target, dynamic=True)))
                with torch.no_grad(), fp32_precision_scope(submodule_mgmt.fp32_precision), autocast_scope(
                    self._autocast_dtype_for(node_name), self._device.type, node_name,
                ), _outside_steps(submodule_mgmt):
                    for args in examples:
                        getattr(owner, name)(*args)
                logger.info("Engine: torch.compile applied to %s.%s", node_name, path)

    def _autocast_dtype_for(self, node_name: str) -> torch.dtype | None:
        """This node's resolved dtype, or None for one that opted out."""
        return self._submodules[node_name].node_dtype

    def warmup(
        self, looped: Collection[tuple[str, str]] = (),
        carried: Mapping[tuple[str, str], Collection[str]] = MappingProxyType({}),
        device_stops: Mapping[tuple[str, str], Any] = MappingProxyType({}),
        kept_rows: Mapping[tuple[str, str], Collection[str]] = MappingProxyType({}),
    ) -> None:
        """Capture every node's graphs. ``looped``: the (node, walk) pairs whose
        walk runs the node inside a Loop, which may run on resident tables, by
        the Loop (a mapping) when its iteration inputs read the Loop's counter;
        ``carried``: the loop-back signals of each such pair whose steps may
        repeat, whose ``preprocess`` is checked here (``probe_carries``);
        ``device_stops``: the pairs asked to run on the device, by their
        node's ``step_repeats`` settings;
        ``kept_rows``: the loop-back signals of the pairs whose config asks to
        keep them in device rows."""
        if self._walk_runs and self._device.type != "cuda":
            raise ValueError("walk_runs currently requires CUDA graph composition")
        # the Loop each (node, walk) runs in, whose counter its iteration inputs read
        self._loops = dict(looped) if isinstance(looped, Mapping) else {}
        requests = set(carried) | set(device_stops)
        owners = {}
        for node_name, walk in sorted(requests):
            if node_name in self._submodules:
                for resource in self._submodules[node_name].resources.values():
                    key = id(resource), walk
                    config = device_stops.get((node_name, walk))
                    joining = bool(config is not None and config.join_rows)
                    owners[key] = resource, walk, joining or (key in owners and owners[key][2])
        for resource, walk, joining in owners.values():
            if joining:
                resource.ask_repeats(walk, join_rows=True)
            else:
                resource.ask_repeats(walk)
        # before capture: a region inside a captured graph compiles in its warm-up
        self._compile_regions()
        ag_runners: dict[str, AcceleratorGraphRunner] = {}
        piecewise: dict[str, dict[str, PiecewiseAcceleratorGraphRunner]] = {}
        for node_name, submodule_mgmt in self._submodules.items():
            submodule = submodule_mgmt.submodule
            capture_stream = self._capture_stream_for(node_name)
            runner = ag_runners[node_name] = AcceleratorGraphRunner(
                submodule_name=node_name,
                submodule=submodule,
                resources=submodule_mgmt.resources,
                step_runner=self._runner,
                device=self._device,
                autocast_dtype=self._autocast_dtype_for(node_name),
                joint_comm_group=submodule_mgmt.joint_comm_group,
                num_slots=submodule_mgmt.num_slots,
                enable_nvtx=self._enable_nvtx,
                device_capability=self.device_capability,
                capture_stream=capture_stream,
                separate_member=bool(self._walk_runs),
            )
            runner._connected_walks = self._walk_runs
            piecewise[node_name] = self._build_piecewise_runners(
                node_name, submodule_mgmt, capture_stream,
            )
            self._refuse_regions_beside_chain(node_name, piecewise[node_name])

        for (node_name, walk), signals in kept_rows.items():
            if node_name in self._submodules:
                self._ask_rows(node_name, walk, signals, ag_runners.get(node_name))
        for node_name, submodule_mgmt in self._submodules.items():
            if submodule_mgmt.loopback_rows is not None:
                ag_runners[node_name].bind_rows(
                    submodule_mgmt.loopback_rows, partial(self._row_pairs, submodule_mgmt),
                )

        # Every runner claims its static buffers before any of them captures:
        # nodes share resources, so a build driven by a later node would move
        # buffers an earlier node's graphs already recorded the address of.
        for node_name in self._submodules:
            ag_runners[node_name].prepare_for_capture()
            for runner in piecewise[node_name].values():
                runner.prepare_for_capture()
        for node_name in self._submodules:
            self._build_resident_planner(
                node_name, ag_runners[node_name], {walk for node, walk in looped if node == node_name},
            )
        for node_name, walk in sorted(requests):
            if node_name in self._submodules:
                self._probe_carries(node_name, ag_runners[node_name], walk, carried.get((node_name, walk), ()))
        # after the planners bind: dropping rows writes the row their captures read
        for (node_name, walk), config in device_stops.items():
            if node_name in self._submodules:
                self._ask_run(node_name, ag_runners[node_name], walk, config, carried.get((node_name, walk), ()))

        # every bucket sized before any is captured, so one plan weighs them all
        # against the largest eager step on the device
        plan = self._plan_captures(ag_runners, piecewise)
        for resource in self._resources.values():
            resource.seal_capture_buffers()

        for node_name, submodule_mgmt in self._submodules.items():
            runner = ag_runners[node_name]
            # the kernels a capture (and the compile inside it) picks are what
            # every replay runs, so it happens under the node's precision
            with self._capture_into(plan, node_name) as planned, \
                    fp32_precision_scope(submodule_mgmt.fp32_precision):
                runner.warmup_and_capture(planned)
            if plan is not None or runner.any_graphs:
                submodule_mgmt.capture_runner = runner
            if runner.any_graphs:
                submodule_mgmt.accelerator_graph_runner = runner

            captured: dict[str, PiecewiseAcceleratorGraphRunner] = {}
            with self._capture_into(plan, f"{node_name} piecewise") as planned:
                for label, pw_runner in piecewise[node_name].items():
                    with fp32_precision_scope(submodule_mgmt.fp32_precision):
                        pw_runner.warmup_and_capture(
                            None if planned is None else {bucket for region, bucket in planned if region == label}
                        )
                    if pw_runner.any_graphs:
                        captured[label] = pw_runner
            submodule_mgmt.piecewise_runners = captured
        if plan is not None:
            torch.cuda.empty_cache()
            free = torch.cuda.mem_get_info(self._device)[0]
            logger.log(
                logging.WARNING if free < plan.floor else logging.INFO,
                "CapturePlan[%s]: %.0f MiB free after capture, %.0f for the largest eager step",
                self._device, free / 2**20, plan.floor / 2**20,
            )

        fail_if_graphs_required([
            f"{node_name} {key}"
            for node_name, runner in ag_runners.items()
            for key in runner.dropped_buckets
        ] + [
            f"{node_name} {label} bs={bs} total_tokens={tokens}"
            for node_name, runners in piecewise.items()
            for label, runner in runners.items()
            for bs, tokens in runner.dropped_shapes
        ])

        # torch.compile applied after CUDA graph capture because the cuda
        # graph runner compiles internally
        self._compile_submodules()
        for submodule_mgmt in self._submodules.values():
            with fp32_precision_scope(submodule_mgmt.fp32_precision), _outside_steps(submodule_mgmt):
                submodule_mgmt.submodule.warmup_compiles()

        for resource in self._resources.values():
            resource.post_warmup_validate()

    def _plan_captures(
        self, ag_runners: dict[str, AcceleratorGraphRunner],
        piecewise: dict[str, dict[str, PiecewiseAcceleratorGraphRunner]],
    ) -> CapturePlan | None:
        """Size every bucket on the device, then plan which to capture.

        None off CUDA: sizing reads CUDA's allocator, and asking the driver for
        memory would fail a CPU worker at start. With no plan, an XPU worker
        captures every bucket.
        """
        if self._device is None or self._device.type != "cuda":
            return None
        def planned(size: Callable[[Any], dict]) -> tuple[dict[str, dict], int, CapturePlan]:
            """A plan over what ``size`` says each runner's buckets cost."""
            pools: dict[str, dict] = {}
            for node_name, submodule_mgmt in self._submodules.items():
                # sizing runs the forwards and captures, so under the node's precision too,
                # and it runs a node's only warm-up forwards, where cuDNN can time its convs
                with fp32_precision_scope(submodule_mgmt.fp32_precision), \
                        _search_cap(submodule_mgmt, self._gpu_memory_fraction, self._device), \
                        cudnn_benchmark(submodule_mgmt.submodule.capture_cudnn_benchmark):
                    pools[node_name] = size(ag_runners[node_name])
                    # one pool for all of a node's regions (see `_build_piecewise_runners`)
                    pools[f"{node_name} piecewise"] = {
                        (label, bucket): cost
                        for label, runner in piecewise[node_name].items()
                        for bucket, cost in size(runner).items()
                    }
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(self._device)
            reusable = None
            if self._gpu_memory_fraction is not None:
                # shortcut: counts this worker's PyTorch allocations, not its CUDA
                # context or NCCL buffers; add those if a capped worker still crowds another
                capped = int(self._gpu_memory_fraction * total) - torch.cuda.memory_reserved(self._device)
                # Free memory already excludes what sizing holds outside the allocator, and `capped`
                # doesn't, so the plan counts graphs as reusing that memory only up to the gap between them.
                reusable = max(capped - free, 0)
                free = min(free, capped)
            return pools, free, plan_captures(pools, free, reusable)

        pools, free, plan = planned(lambda runner: runner.size_captures())
        # a walk's smaller buckets were sized by its largest, which is enough
        # while every graph fits. If one doesn't, here or on another rank of a
        # runner's group, each is sized by its own capture, and the plan made again
        short = any(
            cost.graph is not None and bucket not in plan.buckets[pool]
            for pool, costs in pools.items() for bucket, cost in costs.items()
        )
        pools, free, plan = planned(lambda runner: runner.size_lent_captures(short))
        logger.info(
            "CapturePlan[%s]: %.0f MiB free, %.0f for the largest eager step",
            self._device, free / 2**20, plan.floor / 2**20,
        )
        if free < plan.floor:
            logger.warning(
                "CapturePlan[%s]: no room for graphs beside the largest eager step "
                "(gpu_memory_fraction %s)", self._device, self._gpu_memory_fraction,
            )
        for pool, costs in pools.items():
            if costs:
                logger.info(
                    "CapturePlan[%s] %s: %d of %d buckets, %.0f MiB predicted, %.0f outside the allocator",
                    self._device, pool, len(plan.buckets[pool]), len(costs), plan.predicted[pool] / 2**20,
                    plan.outside[pool] / 2**20,
                )
        return plan

    @contextmanager
    def _capture_into(self, plan: CapturePlan | None, pool: str) -> Iterator[set | None]:
        """Yield what ``plan`` captures into ``pool``, then log what capturing it took."""
        if plan is None:
            yield None
            return
        torch.cuda.empty_cache()
        before = torch.cuda.memory_reserved(self._device)
        yield plan.buckets[pool]
        torch.cuda.empty_cache()
        took = torch.cuda.memory_reserved(self._device) - before
        logger.log(
            logging.WARNING if took > plan.predicted[pool] else logging.INFO,
            "CapturePlan[%s] %s: capture took %.0f MiB, %.0f predicted", self._device, pool,
            took / 2**20, plan.predicted[pool] / 2**20,
        )

    def on_lane(self, lane: int) -> AbstractContextManager:
        """Make ``lane``'s stream this thread's current one, for everything
        issued on the lane: its steps, the collate of their inputs, and the
        teardown of what its nodes hold (a slot zeroed on another stream could
        land after the next request's first write to it). The device runtime's
        per-call context, never the stream object's own, so two threads can be
        on one lane at once. The chain (lane 0) stays on the current stream."""
        stream = self._lane_streams.get(lane)
        if stream is None:
            return nullcontext()
        return self.device_capability.graph_backend.stream_context(stream)

    def _lane_contexts(
        self, node_names: Collection[str], declared_by: Mapping[str, frozenset[str]],
        prefix_caches: Collection[str],
    ) -> dict[str, Callable[[], AbstractContextManager]]:
        """Give every resource a lane off the chain holds the context its work
        runs under there: its lane's stream (``on_lane``).

        A lane's steps admit, plan and commit inline on the GPU thread, between
        two chain steps, while the plan thread stages the chain's next step and
        the main thread ingests, readies and tears down other rows. So a
        resource on a lane off the chain is that lane's alone: declared only by
        this worker's nodes on it, with every resource it depends on there too.
        The runner puts each call into it from outside the step on the lane's
        stream too (``StepRunner._on_lane_of``). What only the chain does stays
        on the chain: a prefix cache, state the main thread moves outside the
        step (``Resource.journals_resident_moves``: eviction, prefix leases,
        page trails). Host rows stage through a ring of the lane's own
        (``Worker._stage_host_rows``).

        Nothing fences a lane's admit and plan against the main thread, so a
        resource a lane holds must keep them safe against its ingest, readiness
        and teardown of other rows, as the recurrent pool's lock does."""
        if not self._lane_streams:
            return {}
        backend = self.device_capability.graph_backend

        def lane_of(key: str) -> set[int]:
            return {self._node_lanes.get(node, 0) for node in declared_by[key]}

        contexts = {}
        for key, resource in self._resources.items():
            # one not built here is the runner's ``topo_sort`` to refuse, by name
            deps = [dep for dep in resource.depends_on() if dep in declared_by]
            holds = lane_of(key)
            if holds == {0} and all(lane_of(dep) == {0} for dep in deps):
                continue

            def refuse(reason: str, key=key, holds=holds) -> ValueError:
                return ValueError(f"resource {key!r}, declared by {sorted(declared_by[key])} on lanes "
                                  f"{sorted(holds)}, can't be placed there: {reason}")

            # first for one off the chain: a node on another worker is on no lane here, so it reads as the
            # chain's; one on the chain may be declared there too (BAGEL's CFG replicas), so its deps decide
            if holds != {0} and not declared_by[key] <= set(node_names):
                raise refuse("a node on another worker declares it too")
            if len(holds) > 1:
                raise refuse("nodes on two lanes declare it, and a step on one could change it while "
                             "another's reads it")
            crossed = sorted(dep for dep in deps if lane_of(dep) != holds)
            if crossed:
                raise refuse(f"it depends on {crossed}, which another lane holds")
            if key in prefix_caches:
                raise refuse("it keeps a prefix cache")
            if type(resource).journals_resident_moves():
                raise refuse("the main thread moves its state outside the step (eviction, prefix leases or "
                             "page trails)")
            (lane,) = holds
            contexts[key] = partial(backend.stream_context, self._lane_streams[lane])
        return contexts

    def _capture_stream_for(self, node_name: str) -> Any:
        """The stream a node captures on: its lane's, or None for the
        backend's shared one on the chain.

        Graphs captured on one stream share what the capture ties to it, so two
        of them must never run at once. On one lane they can't, with one step
        in flight each, but a step off the chain can overlap any graph of any
        other lane, and with a shared capture stream that silently changes the
        other graph's result.
        """
        backend = self.device_capability.graph_backend
        if backend is None or not backend.is_available():
            return None
        return self._lane_streams.get(self._node_lanes.get(node_name))

    def _build_piecewise_runners(
        self, node_name: str, submodule_mgmt: SubmoduleManagement, capture_stream: Any = None,
    ) -> dict[str, PiecewiseAcceleratorGraphRunner]:
        """One runner per region the submodule declares, not yet captured.

        The caller captures them, and drops a region whose capture failed so
        its forward takes the eager path for that label.
        """
        node_dtype = self._autocast_dtype_for(node_name)
        configs = submodule_mgmt.submodule.get_piecewise_accelerator_graph_configs(
            self._device,
            # the dtype the region runs in; an opted-out node keeps its params'
            node_dtype or torch.float32,
            submodule_mgmt.joint_comm_group.world_size,
        )
        runners: dict[str, PiecewiseAcceleratorGraphRunner] = {}
        # One graph memory pool for all of the node's regions (see
        # ``PiecewiseCudaGraphRunner``): its captured-memory footprint becomes
        # the largest region's rather than the sum over regions.
        memory_pool = None
        if configs and self.device_capability.graph_backend is not None:
            backend = self.device_capability.graph_backend
            if backend.is_available():
                memory_pool = backend.graph_pool_handle()
        for label, config in configs.items():
            runner = PiecewiseAcceleratorGraphRunner(
                label=f"{node_name}_{label}",
                config=config,
                resources=submodule_mgmt.resources,
                step_runner=self._runner,
                device=self._device,
                autocast_dtype=node_dtype,
                joint_comm_group=submodule_mgmt.joint_comm_group,
                num_slots=submodule_mgmt.num_slots,
                node_name=node_name,
                memory_pool=memory_pool,
                device_capability=self.device_capability,
                capture_stream=capture_stream,
                lora_keys=getattr(submodule_mgmt.submodule, "lora_keys", ()),
            )
            runners[label] = runner
        return runners

    def _open_prefix_caches(
        self, specs_by_key, model, instance: Mapping[str, int], opened: frozenset[str], checkpoint: str | None,
    ) -> None:
        """Open the caches built here that ``opened`` names, and close the rest.
        ``opened`` is `opening_prefix_caches` over ``instance``, the lockstep
        instance of each node here, for ``model``'s ``checkpoint``. Each cache is
        rooted in the weights, the preprocessing, and the config of every
        resource its pages were planned against."""
        shared = [
            checkpoint_identity(checkpoint) if checkpoint is not None else b"",
            model.preprocess_fingerprint() if model is not None else "",
        ]
        declared = keyed_labels(model) if model is not None else {}
        self._prefix_model = type(model).__name__
        opened_keys = opened & declared.keys()
        self._log_shut_caches(specs_by_key, declared.keys() - opened, opened, instance, checkpoint is not None)
        # node -> the walks a probe may run on, from the keys of every label on an open cache the node uses
        self._keyed_walks: dict[str, set[str]] = {}
        for key in opened_keys:
            for node in specs_by_key[key].nodes:
                self._keyed_walks.setdefault(node, set()).update(
                    walk for label in declared[key].values() for walk in label.keys.walks
                )
        for key, spec in specs_by_key.items():
            if key in self._resources and spec.depends_on() <= self._resources.keys():
                self._resources[key].bind_dependencies({dep: self._resources[dep] for dep in spec.depends_on()})
        for key, resource in self._resources.items():
            if key not in opened:
                resource.close_prefix_cache()
                continue
            # its own, then its dependents' in key order, so the root does not
            # move with the order the engine happened to build them in
            parts = [resource.fingerprint()]
            for other in sorted(specs_by_key):
                if key in specs_by_key[other].depends_on():
                    built = self._resources.get(other)
                    mark = built.fingerprint() if built is not None else None
                    if mark is not None:
                        parts.append(mark)
            resource.enable_prefix_cache(fingerprint(*shared, *parts), {
                label: KeyedWalks(
                    kv_label.keys.walks, tuple(walk for walk, _ in kv_label.growth if kv_label.keys.growth),
                )
                for label, kv_label in declared.get(key, {}).items()
            })
        for key in sorted(opened_keys & self._resources.keys()):
            resource = self._resources[key]
            nodes = ", ".join(sorted(specs_by_key[key].nodes))
            for label in declared[key]:
                logger.info("KV %s: prefix cache open for %s on %s", key, label, nodes)
            if not resource.supports_eviction:
                # a hit allocates only the prompt's tail, so admission no longer
                # bounds how many requests decode at once
                logger.warning(
                    "KV %s: prefix cache on with cpu_offload_pages 0: a decode "
                    "step that finds nothing to evict holds its requests until "
                    "they time out. Set max_concurrent_requests to the pool's "
                    "pages over the pages one request needs. cpu_offload_pages "
                    "gives the worker a victim, but one that shares most of its "
                    "prompt frees little",
                    key,
                )
        # the nodes of a keyed walk that don't use the label's cache, such as an image
        # encoder. The KV computed from their output is cached under their input's digest.
        self._item_encoders: set[str] = set()
        for key in opened_keys:
            for label in declared[key].values():
                for walk in label.keys.walks:
                    nodes = model.get_graph_walk_graphs()[walk].get_nodes()
                    self._item_encoders.update(set(nodes) - specs_by_key[key].nodes)

    def _log_shut_caches(
        self, specs_by_key: dict[str, NodeResourceSpec], shut: set[str], opened: frozenset[str],
        instance: Mapping[str, int], checkpoint_named: bool,
    ) -> None:
        """Say why each keyed cache here that asks for a prefix cache stays shut,
        from the pieces of `opening_prefix_caches` that shut it."""
        for key in sorted(shut & self._resources.keys()):
            spec = specs_by_key[key]
            # the cache is on at all: it opens at one rank for a model that names its checkpoint
            if not spec.keeps_prefix_cache(1, checkpoint_named=True):
                continue
            size = instance_size(spec, instance)
            nodes = ", ".join(sorted(spec.nodes))
            if not spec.keeps_prefix_cache(size, checkpoint_named=True):
                logger.info("KV %s: prefix cache off at world size %d (prefix_cache_tp opts in)", key, size)
            elif not spec.keeps_prefix_cache(size, checkpoint_named):
                logger.info(
                    "KV %s: prefix cache off: %s names no checkpoint, so "
                    "builds that differ only in their weights cannot be "
                    "told apart; set prefix_cache_salt to enable it",
                    key, self._prefix_model,
                )
            else:
                restoring = sorted(
                    over.resource_key for over in unrestorable(specs_by_key, opened)
                    if not over.nodes.isdisjoint(spec.nodes)
                )
                logger.info(
                    "KV %s: prefix cache off: %s on %s can't restore a cached prefix",
                    key, ", ".join(restoring) or "a resource", nodes,
                )

    def prepare_inputs(self, batch: ExecutingBatch) -> None:
        """Per-rid ``submodule.prepare_inputs``, onto ``batch.inputs``.

        Per-rid, so a raise is attributable to one request: record it and take
        that rid out rather than losing the batch. A rid the submodule declines
        (returns None) leaves the same way, without being an error.
        """
        if self._enable_nvtx:
            range_push(f"engine.prepare_inputs.bs{len(batch.request_ids)}")
        try:
            if batch.step_context.repeats > 1:
                with watch_table_reads() as reads:
                    self._prepare_inputs(batch)
                _refuse_counter_read(batch.node_name, batch.step_context, reads, "prepare_inputs")
            else:
                self._prepare_inputs(batch)
        finally:
            if self._enable_nvtx:
                range_pop()

    def _prepare_inputs(self, batch: ExecutingBatch) -> None:
        mgmt = self._submodules[batch.node_name]
        submodule = mgmt.submodule
        node_inputs: list[NodeInputs] = []
        # hoisted out of the per-request loop: the same for every row
        prepare = submodule.prepare_inputs
        resources = mgmt.resources
        wrapped = batch.per_request_info_wrapped
        tensors = batch.per_request_input_tensors
        metadata = batch.per_request_input_metadata
        final_rids = batch.final_stream_rids
        # only a keyed walk probes the prefix cache (see _skip_cached_prefix);
        # read leniently, as the probe itself was only reached per prepared row.
        # Under a combined walk each row has its own walk, so a mixed step
        # probes, prepares and stamps per row, and takes no uniform path.
        walk = batch.step_context.graph_walk
        mixed = bool(batch.step_context.request_walks)
        walk_of = batch.step_context.walk_of
        keyed_walks = (getattr(self, "_keyed_walks", None) or {}).get(
            batch.node_name, (),
        )
        loopback = batch.kept_rows = self._loopback_rows(batch)
        # a kept signal no model hook reads: its row, as if routed
        kept = mgmt.row_signals.get(walk, {}) if loopback is not None and not mixed else {}
        # a chunked row is cut from its whole input, so no uniform path
        if not mixed and walk not in keyed_walks and not batch.chunk_ranges:
            uniform = submodule.uniform_row_inputs(
                walk, has_loopback=loopback is not None and all(loopback),
            )
            if uniform is not None:
                # one object for all the step's rows: the per-row call would
                # return an equal one each time (128 calls at a full decode
                # batch, on the gpu thread before the launch)
                node_inputs = [uniform] * len(batch.request_ids)
                batch.register_prepare_batch(node_inputs, uniform=uniform)
                batch.prepared_rids = tuple(batch.request_ids)
                batch.running_batched = submodule.can_batch(
                    batch=batch, model_inputs=node_inputs
                )
                return
        # a step of one walk probes (or not) for every row alike
        step_probes = not mixed and walk in keyed_walks
        for i, rid in enumerate(batch.request_ids):
            row_walk = walk_of(rid) if mixed else walk
            probe_prefix = (row_walk in keyed_walks) if mixed else step_probes
            chunk = batch.chunk_ranges.get(rid)
            try:
                if chunk is not None:
                    req_inputs = self._prepare_chunk(batch, rid, row_walk, chunk)
                    if req_inputs is None:
                        batch.skipped_rids.add(rid)
                    else:
                        node_inputs.append(req_inputs)
                    continue
                row_inputs = tensors.get(rid, {})
                if kept and loopback[i]:
                    row_inputs = {**row_inputs, **{s: [mgmt.loopback_rows.row(s, rid)] for s in kept}}
                req_inputs = prepare(
                    graph_walk=row_walk,
                    fwd_info=wrapped[rid],
                    inputs=row_inputs,
                    resources=resources,
                    # the step that ends the last of the node's streams: it must
                    # flush whatever it held back (a vocoder's crossfade tail, the
                    # look-ahead frames a token encoder withholds)
                    is_final_stream_chunk=rid in final_rids,
                    input_metadata=metadata.get(rid, EMPTY_INPUT_METADATA),
                    has_loopback=loopback is not None and loopback[i],
                )
                if req_inputs is not None and probe_prefix:
                    req_inputs = self._skip_cached_prefix(batch, rid, req_inputs)
                if req_inputs is not None and (probe_prefix or mixed):
                    # the submodule reads a mixed step's rows by their own walk
                    req_inputs.graph_walk = row_walk
            except Exception as error:
                guard = active_guard()
                if guard is not None and guard.reason is not None:
                    raise
                batch.register_failure(rid, error, walk=row_walk)
                continue
            if req_inputs is None:
                batch.skipped_rids.add(rid)
            else:
                node_inputs.append(req_inputs)

        batch.register_prepare_batch(node_inputs)
        self._chunking.check_budget(batch, node_inputs)
        batch.drop_rids(batch.skipped_rids | batch.failed_requests.keys())
        batch.prepared_rids = tuple(batch.request_ids)
        batch.running_batched = submodule.can_batch(
            batch=batch, model_inputs=node_inputs
        )

    def _loopback_rows(self, batch: ExecutingBatch) -> list[bool] | None:
        """Per row, whether its input token is on the device (see
        ``_loopback_steps``); None when no row's is."""
        steps = self._loopback_steps
        if not steps:
            return None
        get = steps.get
        ctx = batch.step_context
        if ctx.request_walks:
            # a mixed step: a row's token is on the device if its own walk's
            # last step kept it there, whatever this step is labelled
            node, walk_of = batch.node_name, ctx.walk_of
            return [get(rid) == (node, walk_of(rid)) for rid in batch.request_ids]
        key = (batch.node_name, ctx.graph_walk)
        return [get(rid) == key for rid in batch.request_ids]

    def _record_loopback(self, batch: ExecutingBatch) -> None:
        steps = self._loopback_steps
        if batch.rows_only:
            steps.update(dict.fromkeys(
                batch.request_ids, (batch.node_name, batch.step_context.graph_walk),
            ))
        elif batch.loopback_rids:
            # a mixed step: a row looped back under its own walk, so the next
            # step of that walk finds its token on the device
            keep, node, walk_of = (
                batch.loopback_rids, batch.node_name, batch.step_context.walk_of,
            )
            for rid in batch.request_ids:
                if rid in keep:
                    steps[rid] = (node, walk_of(rid))
                else:
                    steps.pop(rid, None)
        elif steps:
            for rid in batch.request_ids:
                steps.pop(rid, None)

    def inline_client_signals(
        self, node_name: str, graph_walk: str,
    ) -> dict[str, str]:
        """The node's client signals that can ride inline this walk."""
        mgmt = self._submodules.get(node_name)
        if mgmt is None:
            return {}
        return mgmt.submodule.inline_client_signals(graph_walk)

    def _prepare_chunk(
        self, batch: ExecutingBatch, rid: int, walk: str, chunk: ChunkRange,
    ) -> NodeInputs | None:
        """One chunk of a row's input, cut from the whole input prepared once
        on its first chunk and kept until the last one lands
        (``release_chunk_inputs``). A chunked row's first chunk starts past its cached prefix, if any.

        Raises when the prepared input is not the length the scheduler cut it
        by (``get_input_sequence_len`` disagreeing with ``prepare_inputs``):
        its chunks would miss tokens or run past the end, so the row fails."""
        submodule = self._submodules[batch.node_name].submodule
        key = (rid, batch.node_name)
        full = self._chunking.prepared.get(key)
        start, end, total = chunk.start, chunk.end, chunk.total
        if full is None:
            full = submodule.prepare_inputs(
                graph_walk=walk,
                fwd_info=batch.per_request_info_wrapped[rid],
                inputs=batch.per_request_input_tensors.get(rid, {}),
                resources=self._submodules[batch.node_name].resources,
                is_final_stream_chunk=rid in batch.final_stream_rids,
                input_metadata=batch.per_request_input_metadata.get(rid, EMPTY_INPUT_METADATA),
            )
            if full is None:
                return None
            if full.input_seq_len != total:
                raise RuntimeError(
                    f"{batch.node_name}/{walk}: the row was measured at {total} tokens "
                    f"and chunked by that, but its prepared inputs hold {full.input_seq_len}"
                )
            if walk in self._keyed_walks.get(batch.node_name, ()):
                # a first chunk starts past what `input_sequence_len` matched;
                # the probe answers the same again, now checked against the prepared inputs
                prefix = None
                policy = self._chunking.policy(batch.node_name, walk)
                declines = (
                    policy.reuses_cached_prefix is not None
                    and not policy.reuses_cached_prefix(batch.per_request_info_wrapped[rid])
                )
                if not declines and self._prefix_cuttable(full, batch.node_name, walk):
                    prefix = self._runner.resolve_cached_prefix(rid, batch.node_name, walk, full)
                matched = prefix.tokens if prefix is not None else 0
                if matched != start:
                    if start:
                        raise RuntimeError(
                            f"{batch.node_name}/{walk}: the row's chunks start past a "
                            f"{start}-token cached prefix, but its prepared inputs match {matched}"
                        )
                    # Measured with no hit, and another request's pages landed
                    # while it waited. Its chunks are cut from 0, so decline the
                    # hit, keeping the chain so the row still keys its own pages.
                    prefix = prefix._replace(tokens=0, position=prefix.position - prefix.tokens)
                # a declined or uncuttable walk writes over the lease (prefix None)
                self._runner.apply_cached_prefix(rid, batch.node_name, walk, full, prefix)
            self._chunking.prepared[key] = full
            batch.first_chunk_rids.add(rid)
        part = submodule.split_inputs(
            walk, batch.per_request_info_wrapped[rid], full, start, end,
        )
        part.graph_walk = walk
        part.chunk_start = start
        part.chunk_total = full.input_seq_len
        return part

    def release_chunk_inputs(self, rid: int, node_name: str) -> None:
        self._chunking.release(rid, node_name)

    def device_loopback_signals(
        self, node_name: str, graph_walk: str,
    ) -> frozenset[str]:
        """Loop-back signals of this walk whose per-row value the node keeps in
        its loop-back rows, so the worker routes them with no tensor: the next
        step reads them there and every fallback batch arrives with the signal
        empty. ``prepare_inputs`` is told per row (``has_loopback``) whether
        its last step here kept them; a row without it (e.g. its previous
        step ran on another rank) brings the tensor. On a walk whose output
        signals all stay on the device the engine builds no per-request
        output dicts and does not call ``postprocess``: the stop check, the
        inline emit and the next step's speculation read the step's rows
        instead. Only a captured node keeps rows; ``MSTAR_DEVICE_LOOPBACK=0``
        keeps every node on the tensor path."""
        mgmt = self._submodules.get(node_name)
        if mgmt is None or mgmt.loopback_rows is None or mgmt.accelerator_graph_runner is None:
            return frozenset()
        return frozenset(self._row_outputs(mgmt, graph_walk))

    @staticmethod
    def _row_outputs(mgmt: SubmoduleManagement, graph_walk: str) -> dict[str, str]:
        """Each signal of ``graph_walk`` the node's rows keep -> its output."""
        loopback = getattr(mgmt.submodule, "device_loopback", None)
        out = dict(loopback.outputs()) if loopback is not None and loopback.walk == graph_walk else {}
        out.update((signal, pair.output) for signal, pair in mgmt.row_signals.get(graph_walk, {}).items())
        return out

    @staticmethod
    def _row_pairs(mgmt: SubmoduleManagement, graph_walk: str) -> tuple[dict[str, str], dict[str, str]]:
        """What a capture of ``graph_walk`` reads at its head (signal -> static
        input) and writes at its tail (signal -> output). A ``DeviceLoopback``'s
        token at the tail of every capture: a mixed step's decode rows read it next."""
        heads: dict[str, str] = {}
        tails: dict[str, str] = {}
        loopback = getattr(mgmt.submodule, "device_loopback", None)
        if loopback is not None:
            tails.update(loopback.outputs())
            if loopback.walk == graph_walk:
                heads[loopback.signal] = loopback.input
                heads.update((name, name) for name in loopback.states)
        for signal, pair in mgmt.row_signals.get(graph_walk, {}).items():
            heads[signal] = pair.input
            tails[signal] = pair.output
        return heads, tails

    def _ask_run(
        self, node_name: str, runner: AcceleratorGraphRunner, graph_walk: str, config, signals: Collection[str],
    ) -> None:
        """Before capture: have ``graph_walk``'s captures run on the device,
        where its stop is declared or its actual Loop cap bounds it, on one rank."""
        mgmt = self._submodules[node_name]
        if mgmt.joint_comm_group.world_size > 1:
            return  # the stop must match on every rank, and NCCL in a WHILE body is untested
        iteration = bool(getattr(mgmt.submodule, "ITERATION_INPUTS", None))
        loop = self._loops.get((node_name, graph_walk)) if iteration else None
        policies = self.get_stop_policies(node_name, graph_walk)
        if policies is None:
            return  # a custom host stop cannot be skipped
        policy = next((p for p in policies.values() if p.reads_token), next(iter(policies.values()), None))
        connected = any(
            unit.streaming_edges and (node_name, graph_walk) in zip(unit.nodes, unit.graph_walks, strict=True)
            for walk in self._walk_runs for unit in self._graph_runtime.execution_units(walk)
        )
        if policy is None and (not iteration or loop is None) and not connected:
            return
        mgmt.device_stops[graph_walk] = policy
        mgmt.refills[graph_walk] = config.refill_at
        if connected and self._runner.prefix_chain_keys(node_name):
            runner.refuse_run(graph_walk, "resource: independently ready rows cannot key a prefix chain")
            return
        if config.join_rows and self._runner.prefix_chain_keys(node_name):
            runner.refuse_run(graph_walk, "resource: joining rows cannot key a prefix chain")
            return
        if iteration:
            refusal = None
            if loop is None or (policies and set(policies) != {loop.name}):
                refusal = "graph: iteration inputs need stops belonging to their actual Loop"
            elif config.past_cap:
                refusal = "graph: iteration input windows cannot run past their Loop cap"
            elif self._runner.prefix_chain_keys(node_name):
                refusal = "resource: actual iteration rows cannot key a prefix chain"
            elif mgmt.loopback_rows is None:
                refusal = "resource: actual iteration progress needs loop-back rows"
            elif (why := self.resident_refusal(node_name, graph_walk)) is not None:
                refusal = f"planner: actual iteration progress needs resident state: {why}"
            if refusal is not None:
                runner.refuse_run(graph_walk, refusal)
                return
        if config.device_stop == "drop_rows" and self._runner.prefix_chain_keys(node_name):
            # its tokens past a stop hold no KV, and the chain keys every token that lands
            runner.refuse_run(graph_walk, "resource: a dropped row's pages past its stop would key a prefix chain")
            return
        heads = self._row_pairs(mgmt, graph_walk)[0] if mgmt.loopback_rows is not None else {}
        carries = mgmt.submodule.repeat_carry(graph_walk, signals)
        runner.ask_run(graph_walk, RunAsk(
            None if policy is None else policy.token_signal, () if policy is None else policy.eos_tokens,
            config.device_stop == "drop_rows", config.max,
            {signal: pair for signal, pair in carries.items() if signal not in heads},
            mgmt.run_columns.get(graph_walk, frozenset()),
            frozenset(getattr(mgmt.submodule, "ITERATION_INPUTS", None) or ()), config.refill_at,
            config.stream_tokens, loop=loop,
            membership=mgmt.resources[run_rows_key(node_name)] if config.join_rows else None,
            independent_rows=connected and not config.join_rows,
        ))

    def ask_ring(self, node_name: str, signal: str, depth: int, walks: Collection[str]) -> LoopbackRows | str:
        """After capture: keep ``signal``'s items, which ``walks`` stream to a
        consumer on this GPU, in a ring of the node's rows ``depth`` deep,
        allocated now from a capture's output; the rows, or why not."""
        mgmt = self._submodules[node_name]
        rows, runner = mgmt.loopback_rows, mgmt.accelerator_graph_runner
        if rows is None:
            return "node: it keeps no loop-back rows (asked for by repeats or resident planning)"
        item = None if runner is None else runner.captured_output(signal)
        if item is None:
            return f"runner: no capture of the node outputs {signal!r}, to size its ring by"
        rows.ask_ring(signal, depth)
        if (refusal := rows.ring_table(signal, item)) is not None:
            return refusal
        mgmt.rings[signal] = frozenset(walks)
        return rows

    def ring_rows(self, node_name: str) -> LoopbackRows | None:
        """The rows that hold the node's rings."""
        return self._submodules[node_name].loopback_rows

    def ring_room(self, node_name: str, rids: Collection[int]) -> int | None:
        """The fewest items any of ``rids``' rings takes now; None for a node with none."""
        mgmt = self._submodules[node_name]
        if not mgmt.rings:
            return None
        rows = mgmt.loopback_rows
        return min((rows.ring_room(signal, rid) for signal in mgmt.rings for rid in rids), default=None)

    def _reserve_rings(self, mgmt: SubmoduleManagement, batch: ExecutingBatch) -> None:
        """Before the launch: the ring positions each row's items of each ringed
        signal take. A row whose walk streams none, or whose prompt is still in
        chunks, takes none."""
        ctx = batch.step_context
        chunks = batch.chunk_ranges
        rids = [
            rid for rid in batch.request_ids
            if rid not in chunks or chunks[rid].is_final
        ]
        batch.ring_width = ctx.repeats
        for signal, walks in mgmt.rings.items():
            mine = [rid for rid in rids if ctx.walk_of(rid) in walks]
            if mine:
                bases = mgmt.loopback_rows.ring_reserve(signal, mine, ctx.repeats)
                batch.ring_bases[signal] = dict(zip(mine, bases, strict=True))

    def _write_rings(self, mgmt: SubmoduleManagement, batch: ExecutingBatch, raw: BatchedModelOutput) -> None:
        """Right behind the step, on its stream: each row's items of each ringed
        signal into the positions reserved for it, one copy a signal."""
        index_of = {rid: i for i, rid in enumerate(batch.request_ids)}
        for signal, bases in batch.ring_bases.items():
            items = output_named(raw, signal)
            if items is None:
                continue
            if batch.ring_width == 1:
                items = items.reshape(items.shape[0], 1, *items.shape[1:])
            rids = list(bases)
            rows = [index_of[rid] for rid in rids]
            picked = items if rows == list(range(len(rows))) else items[
                to_device_nonblocking(rows, items.device, dtype=torch.long)
            ]
            mgmt.loopback_rows.ring_write(signal, rids, picked[:, :batch.ring_width], [bases[rid] for rid in rids])

    def stream_tokens(self, node_name: str, graph_walk: str, signal: str, limit: int) -> str | None:
        """A walk's runs, up to ``limit`` long, leave ``signal``, a stop
        buffer, where the host can send it as it lands; why they can't, or None.
        A slot's landing area is the slot's: with one slot, the next step would
        reset it while this one's tokens are still being sent."""
        mgmt = self._submodules[node_name]
        runner = mgmt.accelerator_graph_runner
        if runner is None or runner.num_slots < 2:
            return "runner: one capture slot, which the next step's tokens would land in while this one's go out"
        mgmt.device_stream_tokens[graph_walk] = (signal, limit)
        return None

    def yield_runs(self, node_name: str) -> None:
        """Other work is ready: the node's run in flight ends at its next iteration."""
        runner = self._submodules[node_name].accelerator_graph_runner
        if runner is not None:
            runner.yield_runs()

    def settle_run(
        self, batch: ExecutingBatch, width: int, counts: Mapping[int, int] | None = None,
    ) -> None:
        """Settle row executions separately from the run's output width."""
        ctx = batch.step_context
        if type(width) is not int or not 0 <= width <= ctx.repeats:
            raise ValueError(f"invalid run width {width!r}")
        if counts is None:
            counts = dict.fromkeys(ctx.counted or ctx.request_ids, width)
        if any(type(count) is not int or not 0 <= count <= width for count in counts.values()):
            raise ValueError("row execution count exceeds the run width")
        self._runner.settle_run(batch.step, counts)
        ctx.repeats = width

    def device_stop_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """After capture: why ``graph_walk``'s captures don't skip a replay
        whose rows have all stopped, or None."""
        mgmt = self._submodules[node_name]
        if mgmt.joint_comm_group.world_size > 1:
            return "deployment: it runs on more than one rank"
        if graph_walk not in mgmt.device_stops:
            return "graph: it has no declared stop policy"
        if mgmt.accelerator_graph_runner is None:
            return "runner: the node captures no graphs"
        return mgmt.accelerator_graph_runner.run_refusal(graph_walk)

    def launch_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """After capture: why a run of this walk stays on the host's launch
        loop, or None."""
        return self._submodules[node_name].accelerator_graph_runner.launch_refusal(graph_walk)

    def _ask_rows(
        self, node_name: str, graph_walk: str, signals: Collection[str], runner: "AcceleratorGraphRunner | None" = None,
    ) -> None:
        """Before capture: keep each loop-back signal of this walk in the node's
        rows, beside a ``DeviceLoopback``'s, where it runs on one rank."""
        mgmt = self._submodules[node_name]
        if mgmt.loopback_rows is None or mgmt.joint_comm_group.world_size > 1:
            return
        if runner is not None:
            if not any(spec.bucket.graph_walk == graph_walk for spec in runner.prepare_for_capture()):
                return  # no capture to keep rows for (``rows_refusal`` says so)
            if runner.template_step(graph_walk) is None:
                return  # nothing would place its rows for the captures to read (``rows_refusal`` says so)
        loopback = getattr(mgmt.submodule, "device_loopback", None)
        own = set(loopback.outputs()) if loopback is not None and loopback.walk == graph_walk else set()
        mgmt.row_signals[graph_walk] = mgmt.submodule.repeat_carry(graph_walk, set(signals) - own)

    def rows_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """After capture: why this walk's loop-back signals don't stay in the
        node's rows between steps, or None: each must be a buffer of every
        capture of the walk, one fixed-shape row per request. One that isn't
        goes back to being routed, which its captures' heads read just the same."""
        mgmt = self._submodules[node_name]
        if mgmt.joint_comm_group.world_size > 1:
            return "deployment: it runs on more than one rank"
        runner = mgmt.accelerator_graph_runner
        if runner is None:
            return "runner: the node captures no graphs"
        if not any(spec.bucket.graph_walk == graph_walk for spec in runner.prepare_for_capture()):
            return "runner: it has no capture"
        steps = mgmt.template_steps
        step = steps[graph_walk] if graph_walk in steps else runner.template_step(graph_walk)
        if step is None:
            # the rows' plan places each row's index for the capture's head and tail
            mgmt.row_signals.pop(graph_walk, None)
            return "node: it declares no step, so nothing places a row in the rows its capture reads"
        pairs = mgmt.row_signals.get(graph_walk, {})
        refusal = runner.row_refusal(graph_walk, {signal: pair.output for signal, pair in pairs.items()})
        if refusal is not None:
            mgmt.row_signals.pop(graph_walk, None)
            return f"runner: {refusal}"
        return None

    @staticmethod
    def _write_routed_rows(
        batch: ExecutingBatch, mgmt: SubmoduleManagement, slot, engine_inputs: ModelInputsFromEngine,
        preprocessed: Mapping[str, Any],
        *, stager=None,
    ) -> None:
        """A capture reads its rows at its head, so what a row brought routed
        (its first step here) goes into them first: a row's one host write."""
        kept = batch.kept_rows
        real = len(batch.request_ids)
        routed = list(range(real)) if kept is None else [i for i in range(real) if not kept[i]]
        if not routed:
            return
        rids = [engine_inputs.request_ids[i] for i in routed]
        for signal, name in slot.row_heads.items():
            value = preprocessed.get(name)
            if isinstance(value, torch.Tensor):
                if stager is None:
                    mgmt.loopback_rows.write_rows(signal, rids, value, routed)
                else:
                    mgmt.loopback_rows.write_rows(signal, rids, value, routed, stager=stager)

    def resident_asked(self, node_name: str) -> bool:
        return resident_asked(self._submodules[node_name].submodule)

    def resident_nodes(self) -> frozenset[str]:
        """The nodes the serving config asks for resident planning."""
        return frozenset(name for name in self._submodules if self.resident_asked(name))

    def _keep_rows(
        self, mgmt: SubmoduleManagement, graph_walk: str, rids: Sequence, outputs,
    ) -> None:
        """An eager step's tail, which a capture does itself: each loop-back
        value the node keeps on the device, out of ``outputs``, into the rows
        of ``rids``."""
        rows = mgmt.loopback_rows
        if rows is None or mgmt.accelerator_graph_runner is None:
            return  # an eager-only node reads none back (``device_loopback_signals``)
        for signal, output in self._row_pairs(mgmt, graph_walk)[1].items():
            value = output_named(outputs, output)
            if value is not None:
                rows.write(signal, rids, value)

    def after_output_copy(self, batch: ExecutingBatch) -> None:
        if batch.committed is not None:
            self._runner.after_output_copy(batch.committed)

    def step_finished(
        self, batch: ExecutingBatch, outputs: dict[str, NameToTensorList],
        host_rows: HostRows | None = None, skip_rids: Collection[int] = (),
    ) -> None:
        """A step's outputs are on the host (the stop check's copy) and its
        stops not yet decided: so a request ending here has its state up to
        date, e.g. its pages indexed for prefix reuse. ``skip_rids`` as for
        ``extend_prefix_chains``."""
        self.extend_prefix_chains(batch, outputs, host_rows=host_rows, skip_rids=skip_rids)

    def extend_prefix_chains(
        self, batch: ExecutingBatch, outputs: Mapping[str, Mapping[str, list[torch.Tensor]]],
        host_rows: HostRows | None = None,
        skip_rids: Collection[int] = (),
    ) -> None:
        """Key what this step generated, from the stop check's host copy.

        ``skip_rids`` names rows whose sampled token is not a real sample -- a
        prompt's non-final chunk -- so keying the chain on it would key a
        token the request never generated.
        """
        keys = self._runner.prefix_chain_keys(batch.node_name)
        if not keys:
            return
        rids = batch.request_ids
        if skip_rids:
            rids = [rid for rid in rids if rid not in skip_rids]
            if not rids:
                return
        # The batch call keys each row under its own walk, so a combined-walk
        # step takes it too; only the absence of host rows falls through.
        if host_rows is not None:
            ctx = batch.step_context
            self._runner.extend_prefix_chains_batch(
                rids, batch.node_name,
                # a step of one walk passes the walk: no call per row
                ctx.walk_of if ctx.request_walks else ctx.graph_walk,
                host_rows, outputs, keys=keys,
            )
            return
        for rid in rids:
            per_rid = outputs.get(rid)
            if isinstance(per_rid, Mapping):
                self._runner.extend_prefix_chains(
                    rid, batch.node_name, batch.step_context.walk_of(rid), per_rid,
                    keys=keys,
                )

    def decide_cached_prefix(self, batch: ExecutingBatch) -> dict[str, dict[str, list[int]]]:
        """Rank 0's cached-prefix leases for the step it is sending, by wire id, taken now so the
        followers can take the same pages before they build it."""
        keyed = self._keyed_walks.get(batch.node_name)
        ctx = batch.step_context
        if (
            not keyed or (not ctx.request_walks and ctx.graph_walk not in keyed)
            or not self._runner.prefix_chain_keys(batch.node_name)
        ):
            return {}
        hits = {}
        for rid in batch.request_ids:
            walk = ctx.walk_of(rid)
            if walk not in keyed:
                continue
            leases = self._runner.lease_cached_prefix(rid, batch.node_name, walk)
            if leases:
                hits[self._graph_runtime.get_rid_string(rid)] = leases
        return hits

    def prefix_cache_open(self, node_name: str) -> bool:
        return bool(self._runner.prefix_chain_keys(node_name))

    def adopt_cached_prefix(self, head: ScheduleTPNode, caught_up: bool) -> bool:
        """On a TP follower, take rank 0's leases before building the step. False until this rank holds them all."""
        ready = True
        for rid, walk, leases in self._resolve_hits(head):
            ready &= self._runner.adopt_cached_prefix(rid, head.node_name, walk, leases, caught_up)
        return ready

    def release_cached_prefix(self, batch: ExecutingBatch) -> None:
        """Give back the leases of a step this rank built and won't run."""
        ctx = batch.step_context
        for rid in batch.request_ids:
            self._runner.release_cached_prefix(rid, batch.node_name, ctx.walk_of(rid))

    def drop_cached_prefix(self, head: ScheduleTPNode) -> None:
        """On a TP follower, give back the leases of a head it drops without building."""
        for rid, walk, _ in self._resolve_hits(head):
            self._runner.release_cached_prefix(rid, head.node_name, walk)

    def _resolve_hits(self, head: ScheduleTPNode):
        """Yield (handle, walk, leases) per request in ``head``'s hits, skipping a request this rank
        hasn't ingested or has removed."""
        walk_idx = dict(zip(head.request_ids, head.walk_idx, strict=True)) if head.walks else None
        for wire_rid, leases in head.prefix_hits.items():
            rid = self._graph_runtime.get_rid_handle(wire_rid)
            if rid is not None:
                walk = head.walks[walk_idx[wire_rid]] if walk_idx else head.graph_walk
                yield rid, walk, leases

    def page_trails(self, node_name: str) -> dict[str, tuple[int, int]]:
        return self._runner.page_trails(node_name)

    def check_page_trails(
        self, node_name: str, graph_walk: str, tp_seq: int, trails: dict[str, tuple[int, int]],
    ) -> None:
        """On a TP follower, raise if a resource's page allocations differ from rank 0's when it
        sent this step."""
        for pool, mine in self._runner.page_trails(node_name).items():
            if trails.get(pool, mine) != mine:
                raise RuntimeError(
                    f"KV {pool}: this rank's page allocations {mine} (count, hash) differ from "
                    f"rank 0's {trails[pool]} before {node_name}/{graph_walk}, tp seq {tp_seq}"
                )

    def _skip_cached_prefix(
        self, batch: ExecutingBatch, rid: str, inputs: NodeInputs,
    ) -> NodeInputs | None:
        """Cut the leading tokens this node's resources already hold, or return
        None when they hold the whole walk.

        Only the walks a keyed stream names are probed, and only when
        `split_inputs` can cut their inputs; a guided walk writes two labels
        from one input, so it is skipped. A skipped walk is still passed to
        `apply_cached_prefix` with no prefix, because it writes over whatever
        the resources hold for it.
        """
        walk = batch.step_context.walk_of(rid)
        if walk not in self._keyed_walks.get(batch.node_name, ()):
            return inputs
        if not self._prefix_cuttable(inputs, batch.node_name, walk):
            self._runner.apply_cached_prefix(rid, batch.node_name, walk, inputs, None)
            return inputs
        prefix = self._runner.resolve_cached_prefix(rid, batch.node_name, walk, inputs)
        self._runner.apply_cached_prefix(
            rid, batch.node_name, walk, inputs, prefix,
        )
        matched = prefix.tokens if prefix is not None else 0
        if matched <= 0:
            return inputs
        if matched >= inputs.input_seq_len:
            batch.cached_rids.add(rid)
            return None
        return self._submodules[batch.node_name].submodule.split_inputs(
            walk, batch.per_request_info_wrapped[rid], inputs,
            matched, inputs.input_seq_len,
        )

    def _prefix_cuttable(self, inputs: NodeInputs, node_name: str, walk: str) -> bool:
        assert isinstance(inputs, ARNodeInputs), (
            f"{self._prefix_model} keys the {walk!r} walk of {node_name} "
            "for prefix reuse, but that walk's inputs are not a sequence to cut"
        )
        submodule = self._submodules[node_name].submodule
        return not (
            inputs.kwargs or inputs.resource_step_info or (
                inputs.tensor_inputs
                and not getattr(submodule, "cuts_tensor_inputs", False)
            )
        )

    def exec(
        self, batch: ExecutingBatch
    ) -> BatchedModelOutput:
        """Run one step: declare → admit → plan → forward → commit.

        Captured replay and the eager forward are the same path. Under a lease
        the batch is padded to the slot's shape, preprocess output is staged
        into the static buffers, and the launch is a replay; everything above
        the launch is identical.
        """
        nvtx = self._enable_nvtx
        if self._enable_profile and batch.exec_timings.start is None:
            batch.exec_timings.start = time.perf_counter()
        # Every request vetoed this step (`prepare_inputs` returned None) or
        # failed preparing, so there is no forward to run. Reaching one anyway
        # dies on an empty `inputs`; the walk's next step is scheduled as usual.
        if not batch.request_ids:
            batch.outputs = BatchedModelOutput(
                per_rid_outputs={rid: {} for rid in batch.request_ids}
            )
            batch.outputs_ready.set()
            batch.release_waiters()
            return batch.outputs
        if nvtx:
            range_push(
                f"engine.{batch.node_name}.{batch.step_context.graph_walk}"
                f".bs{len(batch.request_ids)}"
            )
        try:
            # inference-only, under the same scope the capture path used
            with torch.no_grad(), autocast_scope(
                self._autocast_dtype_for(batch.node_name),
                device_type=self._device.type,
                node_name=batch.node_name,
            ):
                laps = PhaseLaps("engine.exec.")
                # a caller that pre-planned already reserved one; otherwise take it here
                lease = batch.step_context.slot_lease or self.reserve_replay_slot(batch)
                batch.step_context.check_tp_repeats()
                laps.lap("reserve_slot")
                # admit/plan/commit assume the whole batch reaches the forward
                # in order; an unbatchable walk with >1 request can't, so each
                # request runs its own full cycle and merges.
                if not batch.running_batched and lease is None:
                    # the unbatched forward has per-request outputs: no rows to stay on
                    batch.rows_only = False
                    batch.loopback_rids = frozenset()
                if not batch.running_batched and len(batch.request_ids) > 1 and lease is None:
                    batch.outputs = self._exec_per_request(batch)
                else:
                    batch.outputs = self._exec_single(batch)
            laps = PhaseLaps("engine.exec.")
            self._finish_outputs(batch)
            laps.lap("record_loopback")
            return batch.outputs
        except Exception:
            # A raise before the plan was promoted (declare, admit) would leave
            # the stage for the next step's admit to find. Drop it here. The
            # lease is released by _exec_single's own finally.
            if batch.preplanned_rids is not None and self._runner.staged:
                self._runner.clear_preplan()
                batch.preplanned_rids = None
            raise
        finally:
            batch.preplan_event = None
            batch.release_waiters()
            if nvtx:
                range_pop()

    def _finish_outputs(self, batch: ExecutingBatch) -> None:
        """Publish ordinary completion metadata before the outputs become ready."""
        if batch.admit_error is None:
            if RUN_DESCRIPTORS not in (batch.outputs.check_stop_buffers or {}):
                self._record_loopback(batch)
            batch.outputs.loopback_rids = batch.loopback_rids
            if batch.step_context.repeats > 1:
                batch.outputs.run_columns = self.run_columns(batch.node_name, batch.step_context.graph_walk)
        batch.outputs_ready.set()

    @staticmethod
    def _pad_batch(
        batch: ExecutingBatch, submodule_mgmt: SubmoduleManagement, lease: SlotLease | None,
    ) -> tuple[list[NodeInputs], Mapping[int, CurrentForwardPassInfo]]:
        """The admitted rows of an ordinary step, padded to its leased capture."""
        cg_runner = submodule_mgmt.accelerator_graph_runner
        inputs = batch.inputs
        req_info = batch.per_request_info_wrapped
        if lease is not None:
            inputs = cg_runner.pad_inputs(lease, inputs, uniform=batch.uniform_inputs)
            req_info = cg_runner.step_metadata(
                lease, batch.request_ids, req_info
            )
            batch.step_context.set_padded_rids(
                cg_runner.step_ids(lease, batch.request_ids)
            )
        return inputs, req_info

    def _exec_single(
        self, batch: ExecutingBatch
    ) -> BatchedModelOutput:
        """The one-forward path: batched (a lease replay or ``forward_batched``)
        or a single eager request."""
        laps = PhaseLaps("engine.exec.")
        submodule_mgmt = self._submodules[batch.node_name]
        # On the GPU thread, right before the forward: point the region runners
        # at this batch's slot so the piecewise lease and replay agree with it.
        submodule_mgmt.set_piecewise_slot(batch.slot or 0)
        cg_runner = submodule_mgmt.accelerator_graph_runner
        lease = batch.step_context.slot_lease
        real_bs = len(batch.request_ids)

        inputs, req_info = self._pad_batch(batch, submodule_mgmt, lease)
        laps.lap("pad")
        retained = False

        try:
            admit, batch.step = self._declare_and_admit(
                # padded: `inputs` was padded to the bucket above, and the
                # model declares one segment per row it will run
                batch, rids=batch.step_context.padded_request_ids, inputs=inputs,
                submodule=submodule_mgmt.submodule,
                ctx=batch.step_context,
                nvtx=self._enable_nvtx, step=batch.step
            )
            if not admit.ok:
                return BatchedModelOutput(
                    per_rid_outputs={rid: {} for rid in batch.request_ids}
                )
            if submodule_mgmt.rings:
                # before the launch, which lets the next step be built against this one's room
                self._reserve_rings(submodule_mgmt, batch)

            raw, batch.step = self._drive_step(
                batch, submodule_mgmt, batch.request_ids, inputs, req_info,
                batch.step_context, lease, batch.running_batched,
                step=batch.step, set_launch=True,
            )
            if raw is None:
                return BatchedModelOutput(
                    per_rid_outputs={rid: {} for rid in batch.request_ids}
                )
            # Commit first: releasing `commit_done` here is what lets a pre-plan
            # of N+1 overlap this step's per-request tail.
            batch.commit_done.set()
            if RUN_DESCRIPTORS in (raw.check_stop_buffers or {}):
                # The worker closes publication and collects each descriptor
                # under its own logical receipt before releasing this lease.
                # Enqueue-time fixed-rid collection/ring writes would attribute
                # a later tenant's values to the capture's original row.
                batch.outputs = raw
                retained = True
                return raw
            if submodule_mgmt.rings:
                self._write_rings(submodule_mgmt, batch, raw)
            out = self._collect_step(batch, submodule_mgmt, lease, raw, inputs, req_info)
            # Optional 1-step launch throttle (see MSTAR_ENGINE_STEP_SYNC).
            # Not for correctness — downstream reads already gate on the
            # completion event, so the token exists before it is read. This just
            # holds the GPU thread until the step drains, tightening the 2-step
            # launch bound to 1. Must stay after `commit_done`, which the plan
            # thread gates on. Skipped during capture: you can't sync mid-capture.
            if (
                _ENGINE_STEP_SYNC
                and self.device_capability.is_accelerator
                and not self.device_capability.is_capturing()
            ):
                if self._enable_nvtx:
                    range_push("engine.await_outputs")
                try:
                    torch.accelerator.current_stream(self._device).synchronize()
                finally:
                    if self._enable_nvtx:
                        range_pop()
            return out
        finally:
            laps.skip()
            if lease is not None and not retained:
                cg_runner.release(lease, real_bs)
            laps.lap("release")

    def _prepare_stream_choices(self, batch: ExecutingBatch, maker: str, signal: str,
                                prototype: torch.Tensor, guard, rid: int):
        """Certify each ordinary policy pop before selecting one on the device.

        Only structural input changes are allowed. Compute choices must have
        identical prepared inputs; an empty final pop
        may complete without a forward. The stream owner keeps the decisions.
        """
        edges = [edge for edge in batch.stream_windows.get(rid, ()) if edge.name == signal]
        if len(edges) != 1 or edges[0].chunk.reservation is None or not edges[0].chunk.reservation.decisions:
            guard.refuse("a connected stream has no retained policy decisions")
        reservation = edges[0].chunk.reservation
        choices = []
        reference = None
        for transition in reservation.transitions:
            chunk = transition.chunk._replace(reservation=None)
            if chunk.num_items > prototype.shape[0]:
                guard.refuse("a policy chunk exceeds its retained producer coverage")
            key = (maker, batch.node_name, signal, rid)
            # The prototype supplies geometry only. No unproduced ring value
            # is read or uploaded while ordinary preparation is certified.
            if chunk.num_items:
                guard.bindings[key] = prototype[0] if chunk.num_items == 1 else prototype[:chunk.num_items]
            tensors = {**batch.per_request_input_tensors.get(rid, {}),
                       signal: [guard.unknown(key)] if chunk.num_items else []}
            metadata = batch.per_request_input_metadata.get(rid, EMPTY_INPUT_METADATA)
            metadata = metadata._replace(stream_chunks={**metadata.stream_chunks, signal: chunk})
            candidate = dataclasses.replace(
                batch, step_context=dataclasses.replace(batch.step_context, request_ids=[rid],
                                                        _padded_request_ids=None, _real_rids=None),
                per_request_info={rid: batch.per_request_info[rid]},
                per_request_input_tensors={rid: tensors}, per_request_input_metadata={rid: metadata},
                stream_windows={rid: batch.stream_windows[rid]},
                inputs=None, prepared_rids=(), skipped_rids=set(), cached_rids=set(), failed_requests={},
                final_stream_rids={rid} if chunk.is_final else set(),
                partition_done_rids=batch.partition_done_rids & {rid},
            )
            with guard.isolated_info(candidate):
                self.prepare_inputs(candidate)
                if candidate.failed_requests:
                    guard.refuse("a future policy chunk failed ordinary preparation")
                if not candidate.request_ids:
                    if not chunk.is_final or chunk.num_items:
                        guard.refuse("only an empty final policy chunk may finish without computation")
                    choices.append(None)
                    continue
                if reference is not None and not guard.same(reference, candidate.inputs):
                    guard.refuse("policy alternatives change prepared host inputs")
                reference = candidate.inputs
            choices.append(candidate)
        return choices

    def _compose_stream(self, group, leases, window, payload, guard):
        """Lower the claimed stream through the existing capture entries/bodies.

        Policy selection and payload access belong to the retained owners;
        this only connects their operations to the backend's IF/WHILE nodes.
        No model forward is captured again.
        """
        from mstar.engine.accelerator_graph_backend import GraphConditional
        from mstar.engine.accelerator_graph_runner import capture_into_graph
        from mstar.engine.resources.loopback_rows import LoopbackRowsStep

        maker, consumer, signal = group.unit.streaming_edges[0]
        batches = {batch.node_name: batch for _, batch in group.members}
        source_batch, batch = batches[maker], batches[consumer]
        source_mgmt, mgmt = self._submodules[maker], self._submodules[consumer]
        ring = source_mgmt.loopback_rows
        runner = mgmt.accelerator_graph_runner
        slot = runner.slot_for(leases[consumer])
        source = None if maker not in leases else source_mgmt.accelerator_graph_runner.slot_for(leases[maker])
        # Each member owns its compact layout. Finished source rows are absent
        # from its admission, while their consumer can still drain held items.
        target_rows = tuple(batch.request_ids)
        source_rows = () if source is None else tuple(source_batch.request_ids)
        if not set(source_rows) <= set(target_rows):
            guard.refuse("every live producer needs a claimed consumer row")
        correspondence = tuple(target_rows.index(rid) for rid in source_rows)
        # Retired producers have no live member slot. Their existing ring
        # readers retain a fixed address; live entries are refreshed below.
        ring.plan(LoopbackRowsStep(), dataclasses.replace(batch.step_context, resident=None))
        lease = batch.step_context.slot_lease
        indices = ring.indices(lease.bucket, lease.buffer_slot)
        source_lease = None if source is None else source_batch.step_context.slot_lease
        source_indices = (None if source is None else
                          ring.indices(source_lease.bucket, source_lease.buffer_slot))
        capacity = slot.run.state.counts.numel()
        source_capacity = 0 if source is None else source.run.state.counts.numel()
        if indices.numel() != capacity or (source is not None and source_indices.numel() != source_capacity):
            guard.refuse("streamed member addresses need their own captured row capacity")
        if ((capacity > 1 and slot.run.state.active is None)
                or (source_capacity > 1 and source.run.state.active is None)):
            guard.refuse("independently ready streamed rows need captured owner masks")
        if source is not None and source.run.state.dropping is None:
            guard.refuse("a streamed producer needs resident ring-owner bindings")
        input_name, owner_signal, payload = payload
        if len(payload) != len(batch.request_ids):
            guard.refuse("each streamed row needs its own captured destination")
        destination_owner = mgmt.loopback_rows if owner_signal is not None else None
        destination_input = slot.static_inputs[input_name]
        destination_view = None
        if destination_owner is not None:
            try:
                destination_view = destination_owner.input_view(owner_signal, destination_input)
            except ValueError as error:
                guard.refuse(str(error))
        item = ring.row(signal, batch.request_ids[0])
        targets = []
        for prototype, target in payload:
            if prototype.numel() % item[0].numel():
                guard.refuse("a streamed destination does not hold whole producer items")
            width = prototype.numel() // item[0].numel()
            shape = (width, *item.shape[1:])
            if tuple(prototype.shape) not in (shape, tuple(item.shape[1:]) if width == 1 else shape):
                guard.refuse("a streamed payload changed its owner's item geometry")
            try:
                targets.append(target.view(1, *shape))
            except RuntimeError:
                guard.refuse("a streamed destination cannot preserve its owner's item coordinates")
        policy = source_mgmt.device_stops.get(source_batch.graph_walk)
        withheld = policy is not None and signal in policy.withhold
        if source is not None and withheld and any(
            policy.limits(source_batch.per_request_info_wrapped[rid]).ignore_eos for rid in source_rows
        ):
            guard.refuse("withheld items inside an EOS-immune run need explicit item identities")
        token = None if not withheld or source is None else output_named(source.static_outputs, policy.token_signal)
        direct = signal in source_mgmt.submodule.DIRECT_OUTPUTS.get(source_batch.graph_walk, ())
        if source is not None and direct:
            raw = source.static_outputs
            if (raw.per_rid_outputs or raw.packed_outputs or raw.output_spans
                    or source_mgmt.submodule.repeat_carry(source_batch.graph_walk, {signal})[signal].output != signal):
                guard.refuse("a direct output needs its same-named unspanned captured row tensor")
        produced = (None if source is None else (source.static_outputs.row_outputs or {}).get(signal) if direct
                    else output_named(source.static_outputs, signal))
        if source is not None and (produced is None or produced.shape != (source_capacity, *item.shape[1:])):
            guard.refuse("a streamed producer must emit one captured item per iteration")
        key = ("stream", id(slot.graph), None if source is None else id(source.graph), id(window),
               input_name, owner_signal,
               tuple((target.data_ptr(), tuple(target.shape), target.stride(), target.storage_offset())
                     for target in targets), indices.data_ptr(), item.untyped_storage().data_ptr(),
               withheld, correspondence, None if source_indices is None else source_indices.data_ptr())
        if slot.composed_graph is not None and slot.composed_key == key:
            return slot.composed_graph
        state = slot.run.state

        retained_token = None if source is None or not withheld else source.run.stops.get(policy.token_signal)
        if source is not None and withheld and source_capacity > 1 and retained_token is None:
            guard.refuse("withheld streamed rows need their retained actual stop token")

        def eos(values):
            stopped = torch.zeros_like(source.run.state.counts, dtype=torch.bool)
            for value in policy.eos_tokens:
                stopped.logical_or_(values.reshape(source_capacity) == value)
            return stopped

        def kept():
            executed = (source.run.state.counts > 0 if source.run.state.active is None
                        else source.run.state.active)
            return executed if not withheld else executed & ~eos(token)

        def project(values):
            if source_capacity == capacity and correspondence == tuple(range(len(target_rows))):
                return values
            result = torch.zeros_like(window.counts)
            for row, target in enumerate(correspondence):
                result[target:target + 1].copy_(values[row:row + 1])
            return result

        def choose():
            window.gather()
            if source is None:
                emitted = torch.zeros_like(window.counts)
            else:
                count = source.run.state.counts
                if withheld:
                    last = (token if retained_token is None else
                            retained_token.gather(1, (count - 1).clamp(min=0)[:, None]))
                    count = count - ((count > 0) & eos(last)).long()
                emitted = project(count)
            window.select(window.origins[:, 0] + window.origins[:, 1] + emitted,
                          window.origins[:, 2].bool(), ~state.marks)
            # A semantic empty final flush needs no compute budget. A chunk
            # that does compute is acknowledged only if its body will execute.
            allowed = ((window.selected >= 0)
                       & ((window.metadata[:, 5] == 0)
                          | ((state.continuation != 0) & (state.counts < state.limit))))
            window.selected.copy_(torch.where(allowed, window.selected, -1))
            window.ready.copy_(allowed.any().reshape(1))
            computing = allowed & (window.metadata[:, 5] != 0)
            window.compute_ready.copy_(computing.any().reshape(1))
            if state.eligible is not None:
                state.eligible.copy_(computing)
            running = torch.zeros_like(window.continuation) if source is None else source.run.state.continuation
            window.continuation.copy_((running != 0) | (window.ready != 0))

        def source_addresses():
            if source is not None:
                ring.gather_rows(source_lease.bucket, source_lease.buffer_slot, source.run.state.dropping.rows)
                for row, target in enumerate(correspondence):
                    indices[target:target + 1].copy_(source_indices[row:row + 1])

        def publish():
            source_addresses()
            positions = window.produced_positions(project(source.run.state.counts))
            active = kept()
            if source_capacity == capacity and correspondence == tuple(range(len(target_rows))):
                ring.ring_scatter(signal, source_indices, positions, produced, active)
            else:
                for row, target in enumerate(correspondence):
                    ring.ring_scatter(signal, source_indices[row:row + 1], positions[target:target + 1],
                                      produced[row:row + 1], active[row:row + 1])

        def gather():
            source_addresses()
            destination_indices = None
            if destination_owner is not None:
                destination_indices = destination_owner.gather_rows(
                    lease.bucket, lease.buffer_slot, state.dropping.rows,
                )
                destination_owner.head(lease.bucket, lease.buffer_slot, owner_signal, destination_input)
            for row, target in enumerate(targets):
                positions = window.payload_positions(target.shape[1])
                ring.ring_gather(signal, indices[row:row + 1], positions[row:row + 1], target,
                                 None if state.eligible is None else state.eligible[row:row + 1])
            if destination_owner is not None:
                computing = (window.selected >= 0) & (window.metadata[:, 5] != 0)
                destination_indices.masked_fill_(~computing, 0)
                destination_owner.tail(lease.bucket, lease.buffer_slot, owner_signal, destination_view)

        backend = runner._graph_backend
        def capture(fn):
            return capture_into_graph(
                fn, backend.graph_pool_handle(), self._device, None, keep_graph=True,
            )[0]
        select, read, advance = capture(choose), capture(gather), capture(window.advance)
        body = []
        entry = []
        if source is not None:
            entry.append(source.run.entry)
            body.append(GraphConditional("if", source.run.state.continuation, (source.graph, capture(publish))))
        entry.extend((slot.run.entry, select))
        body.extend((select, GraphConditional("if", window.compute_ready, (read, slot.graph)),
                     GraphConditional("if", window.ready, (advance,)), select))
        graph = backend.compose((*entry, GraphConditional("while", window.continuation, tuple(body))))
        slot.composed_graph, slot.composed_key = graph, key
        return graph

    def exec_group(self, group: ExecutionGroup) -> list[BatchedModelOutput] | None:
        """Prepare all ordinary members, then submit their captures together.

        None is an unsubmitted refusal: the worker runs the original head.
        Once submission starts, errors use ordinary failure settlement instead
        of replaying any member. No graph-runtime call runs on this thread.
        """
        bindings: dict[Any, torch.Tensor] = {}
        guard = ConnectedInputs(bindings)
        prepared = []
        admitted = []
        rows = tuple(group.members[group.trigger][1].request_ids)
        common_rows = rows
        member_walks = dict(zip(group.unit.nodes, group.unit.graph_walks, strict=True))
        finished = dict(zip(group.unit.nodes, group.finished_rids or [frozenset()] * len(group.members), strict=True))
        active = [(scheduled, batch) for scheduled, batch in group.members
                  if batch.request_ids and not set(batch.request_ids) <= finished[batch.node_name]]
        leases = {}
        slots = {}
        owners = {}
        back_keys, boundary_fields = set(), []
        boundary = []
        streamed = bool(group.unit.streaming_edges and any(
            edge.chunk.reservation is not None and edge.chunk.reservation.decisions
            for _, batch in active for edges in batch.stream_windows.values() for edge in edges
        ))
        idle_sources = ({maker for maker, _, _ in group.unit.streaming_edges
                         for _, batch in group.members if batch.node_name == maker and batch.ring_width == 0}
                        if streamed else set())
        active = [(scheduled, batch) for scheduled, batch in active if batch.node_name not in idle_sources]
        stream_choices, stream_target, stream_window = None, None, None
        composed = None
        try:
            with torch.no_grad(), guard.preparing(), ExitStack() as preparation:
                metadata_preparation = ExitStack()
                if not active or any(set(batch.request_ids) & finished[batch.node_name] for _, batch in active):
                    guard.refuse("finished member rows must be removed before preparation")
                if streamed and (len(group.members) != 2 or len(group.unit.streaming_edges) != 1
                                 or group.unit.edges != group.unit.streaming_edges):
                    guard.refuse("streamed composition needs two members and one internal edge")
                if streamed:
                    maker, consumer, _ = group.unit.streaming_edges[0]
                    layouts = {batch.node_name: set(batch.request_ids) for _, batch in group.members}
                    if not layouts[maker] <= layouts[consumer]:
                        guard.refuse("every live producer needs a claimed consumer row")
                for _, batch in active:
                    preparation.enter_context(guard.isolated_info(batch))
                preparation.enter_context(metadata_preparation)
                repeats = active[0][1].step_context.repeats
                if not streamed and any(batch.step_context.repeats != repeats for _, batch in active):
                    guard.refuse("members requested different iteration counts")
                if not streamed and repeats > 1 and group.unit.loop_name is None:
                    guard.refuse("the graph did not certify a complete connected Loop")
                if not streamed and repeats > 1 and any(
                    len(batch.per_request_input_tensors.get(rid, {}).get(signal, ())) != 1
                    for _, batch in group.members for _, consumer, signal in group.unit.loop_back
                    if consumer == batch.node_name for rid in rows
                ):
                    repeats = 1  # the first ordinary iteration produces an empty seed
                    for _, batch in group.members:
                        batch.step_context.repeats = 1
                # Streamed members already carry and count their own Loops
                # inside DeviceRun. Only a shared fixed-count Loop needs
                # these inter-member boundary copies and seed checks.
                loop_back = group.unit.loop_back if not streamed and repeats > 1 else ()
                direct_producers = set()
                for maker in {a for a, _, _ in (*group.unit.edges, *loop_back)}:
                    if not any(batch.node_name == maker for _, batch in active):
                        continue
                    producer = self._submodules[maker].submodule
                    kind = type(producer)
                    walk = member_walks[maker]
                    if (kind.batch_postprocess is not NodeSubmodule.batch_postprocess
                            or kind.filter_batched_output is not NodeSubmodule.filter_batched_output
                            or kind.unpack_packed_outputs is not NodeSubmodule.unpack_packed_outputs
                            or producer.spanned_outputs(walk) or producer.get_stop_policies(walk) is None):
                        guard.refuse("a producer changes its routed outputs outside capture")
                    if kind.postprocess is not NodeSubmodule.postprocess:
                        signals = {signal for a, _, signal in group.unit.edges if a == maker}
                        if not streamed or not signals <= producer.DIRECT_OUTPUTS.get(walk, frozenset()):
                            guard.refuse("a custom producer needs declared direct outputs")
                        direct_producers.add(maker)
                for member, (_, batch) in enumerate(group.members):
                    if not any(batch is live for _, live in active):
                        continue
                    rows = tuple(batch.request_ids)
                    if not streamed and rows != common_rows:
                        guard.refuse("fixed-count connected members need matching logical rows")
                    mgmt = self._submodules[batch.node_name]
                    runner, submodule = mgmt.accelerator_graph_runner, mgmt.submodule
                    if runner is None or batch.chunk_ranges or batch.step_context.request_walks:
                        guard.refuse("a member has no whole capture or has chunked/mixed rows")
                    if (batch.graph_walk in self._keyed_walks.get(batch.node_name, ())
                            or (mgmt.rings and not streamed)):
                        guard.refuse("a member prepares cached prefixes or stream windows")
                    if streamed and mgmt.joint_comm_group.world_size != 1:
                        guard.refuse("streamed composition is single-rank")
                    if streamed:
                        if why := mgmt.carry_refusals.get(batch.graph_walk):
                            guard.refuse(f"{batch.node_name}: {why}")
                        internal = {signal for maker, _, signal in group.unit.streaming_edges
                                    if maker == batch.node_name}
                        if set(mgmt.rings) - internal:
                            guard.refuse("a streamed member has output rings outside the composed edge")
                    incoming = [(a, signal) for a, b, signal in group.unit.edges
                                if b == batch.node_name and (not finished[a] or streamed)]
                    tensors = {rid: dict(batch.per_request_input_tensors.get(rid, {})) for rid in rows}
                    stream_incoming = incoming if streamed else ()
                    if stream_incoming:
                        maker, signal = stream_incoming[0]
                        source = self._submodules[maker].loopback_rows
                        stream_choices = {
                            rid: self._prepare_stream_choices(batch, maker, signal, source.row(signal, rid), guard, rid)
                            for rid in rows
                        }
                        references = {
                            rid: next((choice for choice in choices if choice is not None), None)
                            for rid, choices in stream_choices.items()
                        }
                        if any(reference is None for reference in references.values()):
                            guard.refuse("a final stream tail with no computation completes on its ordinary path")
                        batch.per_request_input_tensors = {
                            rid: ref.per_request_input_tensors[rid] for rid, ref in references.items()
                        }
                        batch.per_request_input_metadata = {
                            rid: ref.per_request_input_metadata[rid] for rid, ref in references.items()
                        }
                        batch.per_request_info_wrapped = {
                            rid: ref.per_request_info_wrapped[rid] for rid, ref in references.items()
                        }
                        batch.register_prepare_batch([references[rid].inputs[0] for rid in rows])
                        batch.prepared_rids = rows
                        batch.kept_rows = ([references[rid].kept_rows[0] for rid in rows]
                                           if all(ref.kept_rows is not None for ref in references.values()) else None)
                        batch.final_stream_rids = set().union(*(ref.final_stream_rids for ref in references.values()))
                        metadata_preparation.enter_context(guard.stream_metadata(batch, {
                            rid: {signal: tuple(choice.per_request_input_metadata[rid].stream_chunks[signal]
                                                for choice in choices if choice is not None)}
                            for rid, choices in stream_choices.items()
                        }))
                        batch.running_batched = submodule.can_batch(batch=batch, model_inputs=batch.inputs)
                    for maker, signal in (() if stream_incoming else incoming):
                        maker_mgmt = self._submodules[maker]
                        maker_slot = maker_mgmt.accelerator_graph_runner.slot_for(leases[maker])
                        if maker_slot.static_outputs.per_rid_outputs:
                            guard.refuse("a producer changes its routed outputs outside capture")
                        output = maker_mgmt.submodule.repeat_carry(member_walks[maker], {signal})[signal].output
                        source = (maker_slot.static_outputs.row_outputs or {}).get(output)
                        if source is None or source.ndim == 0 or source.shape[0] < len(rows):
                            guard.refuse(f"{maker}/{signal} has no captured row output")
                        for i, rid in enumerate(rows):
                            key = maker, batch.node_name, signal, rid
                            # Ordinary collect gives the consumer split(1)
                            # views, retaining the row axis.
                            bindings[key] = source[i:i + 1]
                            tensors[rid][signal] = [guard.unknown(key)]
                    for maker, consumer, signal in loop_back:
                        if consumer != batch.node_name:
                            continue
                        incoming.append((maker, signal))
                        for rid in rows:
                            key = maker, consumer, signal, rid
                            bindings[key] = tensors[rid][signal][0]
                            back_keys.add(key)
                            tensors[rid][signal] = [guard.unknown(key)]
                    if not stream_incoming:
                        batch.per_request_input_tensors = tensors
                        with autocast_scope(self._autocast_dtype_for(batch.node_name), device_type=self._device.type,
                                            node_name=batch.node_name):
                            self.prepare_inputs(batch)
                    if tuple(batch.request_ids) != rows:
                        guard.refuse("a member changed the common row set")
                    for inp in batch.inputs:
                        for f in dataclasses.fields(inp):
                            value = getattr(inp, f.name)
                            # ARNodeInputs also exposes input_ids/input_embeds
                            # as tensor fields. Only host/declaration fields
                            # must be entirely real at this stage; final
                            # staging still certifies every future tensor.
                            if f.name != "tensor_inputs" and not isinstance(value, torch.Tensor):
                                guard.require_real(value, f"NodeInputs.{f.name}")
                    # The same rotation can serve several node captures.
                    # Lease it once so a large group cannot wrap back onto a
                    # still-live component slot from the preceding step.
                    if mgmt.slot_rotation in slots:
                        batch.set_slot(slots[mgmt.slot_rotation])
                    preview = None
                    if stream_incoming:
                        if mgmt.piecewise_runners:
                            guard.refuse("streamed capture selection needs a whole-step declaration")
                        # The existing declaration derives the capture key
                        # from certified inputs, without reading an arbitrary
                        # policy alternative in cg_key_info. This unpadded
                        # preview is never admitted or planned.
                        with watch_table_reads() as reads:
                            preview = engine_step(
                                submodule, graph_walk=batch.graph_walk, request_ids=batch.request_ids,
                                inputs=batch.inputs, slot_lease=None, piecewise_leases={},
                                per_request_info=batch.per_request_info,
                            )
                        _refuse_counter_read(batch.node_name, batch.step_context, reads, "declare_step")
                        guard.require_real(preview, "streamed capture declaration")
                    lease = (self.reserve_replay_slot(batch, capture_step=preview) if preview is not None
                             else self.reserve_replay_slot(batch))
                    if lease is None:
                        guard.refuse("a member has no capture for the common rows")
                    slots.setdefault(mgmt.slot_rotation, lease.slot)
                    leases[batch.node_name] = lease
                    slot = runner.slot_for(lease)
                    if not streamed and (member + 1 < len(group.members) or repeats > 1):
                        policies = submodule.get_stop_policies(batch.graph_walk)
                        if (policies is None or (repeats > 1 and not set(policies) <= {group.unit.loop_name})
                                or (policies and (member + 1 < len(group.members)
                                                  or any(p.reads_token for p in policies.values())))):
                            guard.refuse("a member can end its Loop before later members execute")
                    if lease.bucket.graph_walk in runner._generator_walks:
                        guard.refuse("a composed member draws from a torch generator")
                    if streamed:
                        if slot.run is None or slot.run.state.membership is not None:
                            guard.refuse("streamed composition needs captured fixed-membership bodies")
                        if lease.bucket.bs > 1 and slot.run.state.active is None:
                            guard.refuse("batched streamed bodies need captured owner masks")
                        if self._custom_output_tail(submodule) and batch.node_name not in direct_producers and (
                            not stream_incoming or batch.step_context.repeats != 1
                        ):
                            guard.refuse("a custom streamed tail needs one consumer compute and no producer transform")
                    elif slot.run is not None or batch.step_context.resident is not None:
                        guard.refuse("connected resident runs need their shared run state staged before launch")
                    batch.step_context.unit_id = group.claim
                    batch.step_context.unit_member = member
                    inputs, info = self._pad_batch(batch, mgmt, lease)
                    prepared.append((batch, mgmt, lease, inputs, info, incoming))
                    with watch_table_reads() as reads:
                        batch.step = self._declare_step(batch, batch.step_context.padded_request_ids, inputs,
                                                        submodule, batch.step_context, batch.step)
                    _refuse_counter_read(batch.node_name, batch.step_context, reads, "declare_step")
                    if batch.step is not None:
                        batch.step.set_ctx(batch.step_context)
                        if preview is not None and batch.step.cg_key_info != preview.cg_key_info:
                            guard.refuse("padded streamed declaration changes its capture key")
                        guard.require_real(batch.step.steps, "resource declarations")
                        for resource_step in batch.step.steps.values():
                            if getattr(resource_step, "pre_forks", ()) or getattr(resource_step, "post_forks", ()):
                                guard.refuse("a member forks resource state during planning")
                    for key, resource in reachable_resources(self._runner.resources, mgmt.resources).items():
                        owners.setdefault(id(resource), []).append((key, resource, batch))
                # A declaration may make an otherwise compatible owner unsafe
                # to share. Check every occurrence, including a dependency
                # with no explicit step, before the first member admits.
                for occurrences in owners.values():
                    if len({batch.node_name for _, _, batch in occurrences}) < 2:
                        continue
                    if any(batch.node_name in direct_producers for _, _, batch in occurrences):
                        guard.refuse("a custom direct producer cannot share resource owners")
                    for key, resource, batch in occurrences:
                        if self._submodules[batch.node_name].piecewise_runners:
                            guard.refuse("a shared owner also has nested piecewise capture buffers")
                        resource_step = None if batch.step is None else batch.step.get(key)
                        if why := resource.connected_refusal(resource_step, batch.step_context):
                            guard.refuse(f"{batch.node_name}/{key}: {why}")
                if streamed:
                    # Every member may end early or run zero iterations, even
                    # when its compute ask is one. Its actual declaration and
                    # dependencies must promise reversible accounting first.
                    for occurrences in owners.values():
                        for key, resource, batch in occurrences:
                            part = (None if batch.step is None or batch.step_context.resident is not None
                                    else batch.step.get(key))
                            if why := resource.repeat_refusal(batch.graph_walk, part):
                                guard.refuse(f"{batch.node_name}/{key}: {why}")
                if not streamed and repeats > 1:
                    for occurrences in owners.values():
                        for key, resource, batch in occurrences:
                            step = None if batch.step is None else batch.step.get(key)
                            if resource.repeat_refusal(batch.graph_walk, step) is not None:
                                repeats = 1
                            else:
                                repeats = min(repeats, resource.max_repeats(step, batch.step_context))
                    repeats = max(1, repeats)
                    for _, batch in group.members:
                        batch.step_context.repeats = repeats
                for batch, _, _, inputs, _, _ in prepared:
                    admit = self._admit_step(batch, batch.step_context.padded_request_ids, inputs,
                                             batch.step_context, batch.step)
                    if not admit.ok:
                        guard.refuse("a member refused admission")
                    if batch.step is not None:
                        admitted.append(batch.step)
                    if not streamed and batch.step_context.repeats != repeats:
                        guard.refuse("a member changed the common iteration count during admission")
                for batch, mgmt, lease, inputs, info, incoming in prepared:
                    rows = tuple(batch.request_ids)
                    with autocast_scope(self._autocast_dtype_for(batch.node_name), device_type=self._device.type,
                                        node_name=batch.node_name):
                        engine_inputs, values = self._plan_step(batch, mgmt, batch.request_ids, inputs, info,
                                                    batch.step_context, lease, batch.step)
                    guard.require_real(batch.step_context.plan_results, "resource plans")
                    runner = mgmt.accelerator_graph_runner
                    slot = runner.slot_for(lease)
                    row_inputs = {name: signal for signal, name in slot.row_heads.items()}

                    def destinations(name, copies, mgmt=mgmt, slot=slot, batch=batch, row_inputs=row_inputs):
                        signal = row_inputs.get(name)
                        if signal is None:
                            return copies
                        try:
                            return mgmt.loopback_rows.input_copies(
                                signal, batch.step_context.padded_request_ids, slot.static_inputs[name], copies,
                            )
                        except ValueError as error:
                            guard.refuse(str(error))

                    carries = mgmt.submodule.repeat_carry(batch.graph_walk, {signal for _, signal in incoming})
                    expected = {}
                    for maker, signal in incoming:
                        expected.setdefault(carries[signal].input, []).extend(
                            (maker, batch.node_name, signal, rid) for rid in rows
                        )
                    copies, real = [], {}
                    for name, value in values.items():
                        if guard.contains(value):
                            if name not in expected or not isinstance(value, torch.Tensor):
                                guard.refuse(f"future values escaped into preprocess field {name}")
                            target = slot.static_inputs.get(name)
                            if not isinstance(target, torch.Tensor) or name not in slot.static_input_keys:
                                guard.refuse(f"no captured destination for {name}")
                            dim = runner._static_buffer_seq_dims.get((slot.config_idx, name), 0)
                            if value.ndim != target.ndim or value.shape[dim] > target.shape[dim]:
                                guard.refuse(f"future field {name} exceeds its captured destination")
                            target = target.narrow(dim, 0, value.shape[dim])
                            keys = tuple(expected.pop(name))
                            static_copies = guard.copies(value, target, keys, set(keys) - back_keys)
                            # Certification still checks row bounds/strides even
                            # when capture keeps static rather than owner views.
                            routed = destinations(name, static_copies)
                            if streamed and incoming:
                                if stream_target is not None or len(routed) != len(rows):
                                    guard.refuse("each streamed row needs one contiguous owner destination")
                                stream_target = (name, row_inputs.get(name), static_copies)
                            else:
                                copies.extend(routed)
                            group.initial_copies.extend(destinations(
                                name, guard.copies(value, target, keys, bindings.keys() - set(keys)),
                            ))
                            if back_keys.intersection(keys):
                                group.initial_copies.extend(destinations(
                                    name, guard.copies(value, target, keys, back_keys),
                                ))
                                boundary_fields.append((value, target, keys, name, destinations))
                        else:
                            real[name] = value
                            if name in row_inputs and isinstance(value, torch.Tensor):
                                kept = batch.kept_rows
                                target = slot.static_inputs[name]
                                routed = [(value[i], target[i]) for i in range(len(rows))
                                          if kept is None or not kept[i]]
                                group.initial_copies.extend(destinations(name, routed))
                    if expected:
                        guard.refuse("preprocess discarded a future input")
                    iteration = (batch.iteration_columns.columns if isinstance(batch.iteration_columns, IterationWindow)
                                 else batch.iteration_columns or {})
                    if iteration.keys() & {pair.input for pair in carries.values()}:
                        guard.refuse("an iteration column overwrites a connected input")
                    if iteration.keys() & row_inputs.keys():
                        guard.refuse("an iteration column needs its captured row owner's destination")
                    runner._stage(lease, real)
                    if streamed:
                        self._stage_capture(batch, mgmt, runner, engine_inputs, real, lease, batch.request_ids)
                    columns = [] if streamed else [
                        *((column, slot.static_inputs[name]) for name, column in iteration.items()),
                        *self._runner.replay_columns(batch.step, repeats),
                    ]
                    group.parts.append(RunPart(slot, copies, columns))
                # Model preparation is over. Owner staging below reads the
                # retained policy receipts themselves, not a model-visible
                # representative of an unresolved decision.
                metadata_preparation.close()
                if streamed:
                    maker, consumer, signal = group.unit.streaming_edges[0]
                    batches = {batch.node_name: batch for _, batch in group.members}
                    consumer_batch, producer_batch = batches[consumer], batches[maker]
                    rows = tuple(consumer_batch.request_ids)
                    reservations = [next(edge.chunk.reservation for edge in consumer_batch.stream_windows[rid]
                                         if edge.name == signal) for rid in rows]
                    tables = self._submodules[consumer].submodule.iteration_tables
                    consumer_state = self._submodules[consumer].accelerator_graph_runner.slot_for(
                        leases[consumer],
                    ).run.state
                    planner = self._submodules[consumer].resident
                    if planner is None or consumer_state.dropping is None:
                        guard.refuse("streamed policy needs resident owner bindings")
                    coverage = max(len(r.positions) for r in reservations)
                    if coverage > len(self._submodules[maker].loopback_rows.row(signal, rows[0])):
                        guard.refuse("streamed policy coverage exceeds its retained ring capacity")
                    try:
                        stream_window = tables.prepare_stream_window(
                            leases[consumer], max(len(r.decisions) for r in reservations),
                            max(len(r.transitions) for r in reservations), coverage,
                            rows=consumer_state.dropping.rows, slots=planner.max_slots,
                        )
                    except ValueError as error:
                        guard.refuse(str(error))  # a larger policy snapshot takes the ordinary path
                    if stream_target is None:
                        guard.refuse("a streamed consumer discarded its future payload")
                    composed = self._compose_stream(group, leases, stream_window, stream_target, guard)
                    tables.stage_stream_window(
                        leases[consumer], [planner.slot_of(rid) for rid in rows], reservations,
                        [[choice is not None for choice in stream_choices[rid]] for rid in rows],
                        producer_limits=[
                            producer_batch.ring_width if maker in leases and rid in producer_batch.request_ids else 0
                            for rid in rows
                        ],
                    )
                elif repeats > 1:
                    row_of = {rid: i for i, rid in enumerate(rows)}
                    for maker, consumer, signal, rid in back_keys:
                        mgmt = self._submodules[maker]
                        slot = mgmt.accelerator_graph_runner.slot_for(leases[maker])
                        name = mgmt.submodule.repeat_carry(member_walks[maker], {signal})[signal].output
                        source = (slot.static_outputs.row_outputs or {}).get(name)
                        if (slot.static_outputs.per_rid_outputs or source is None or source.ndim == 0
                                or source.shape[0] < len(rows)):
                            guard.refuse(f"{maker}/{signal} has no captured Loop carry")
                        row = row_of[rid]
                        guard.rebind((maker, consumer, signal, rid), source[row:row + 1])
                    boundary = [copy for value, target, keys, name, destinations in boundary_fields
                                for copy in destinations(name, guard.copies(value, target, keys, back_keys))]
            if not streamed and self._device.type != "cpu":
                from mstar.engine.accelerator_graph_backend import copy_words

                # Native composition only accepts its existing memcpy layouts.
                # Refuse before any seed staging or irreversible submission.
                pairs = [*group.initial_copies, *boundary,
                         *(pair for part in group.parts for pair in part.copies)]
                for source, target in pairs:
                    try:
                        source = source.view(target.shape)
                    except RuntimeError:
                        guard.refuse("connected graph copies cannot reshape a strided source")
                    if copy_words(target, source) is None:
                        guard.refuse("connected graph copies need contiguous rows without casts")
            # Every binding has been checked. This is the irreversible point;
            # the main thread publishes it to Rust before landing members.
            for scheduled, batch in group.members:
                scheduled.repeats = batch.step_context.repeats
            group.submitted = True
            head = group.members[group.trigger][1]
            if head.launch_started_event is not None:
                head.launch_started_event.set()
            if streamed:
                for source, target in group.initial_copies:
                    target.copy_(source.reshape_as(target), non_blocking=True)
                composed.replay()
                retained = []
                for batch, mgmt, lease, _, _, _ in prepared:
                    run = mgmt.accelerator_graph_runner.slot_for(lease).run
                    stops, values, spans = run.results(batch.step_context.repeats)
                    if batch.node_name == group.unit.streaming_edges[0][1]:
                        stops = {**stops, STREAM_COUNTS: stream_window.counts,
                                 STREAM_TRANSITION: stream_window.history}
                    raw = mgmt.accelerator_graph_runner.slot_for(lease).static_outputs
                    retained.append(BatchedModelOutput(
                        check_stop_buffers=stops, row_outputs={**(raw.row_outputs or {}), **values},
                        output_spans={**(raw.output_spans or {}), **spans} or None,
                    ))
            else:
                retained = prepared[0][1].accelerator_graph_runner.launch_parts(
                    group.parts, repeats,
                    [mgmt.run_columns.get(batch.graph_walk, ()) for batch, mgmt, _, _, _, _ in prepared],
                    capacity=max(repeats, group.capacity), boundary=boundary, initial=group.initial_copies,
                )
            for batch, mgmt, _, _, _, _ in prepared:
                self._commit_step(batch, mgmt, batch.step, batch.step_context)
                # A later member can advance the same owner. Preserve what
                # this member publishes now; the worker still decides which
                # non-aborting rows may materialize/send it.
                batch.publish_snapshots = self._runner.publish_snapshot(
                    batch.request_ids, node_name=batch.node_name,
                    graph_walk=batch.graph_walk, request_walks=batch.step_context.request_walks,
                )
                batch.commit_done.set()
            for (batch, mgmt, lease, inputs, info, _), columns in zip(prepared, retained, strict=True):
                if streamed:
                    # Independent counts and policy receipts resolve before a
                    # model tail or graph completion can see these columns.
                    batch.outputs = columns
                    batch.outputs.row_request_ids = tuple(batch.request_ids)
                    batch.outputs.run_columns = self.run_columns(batch.node_name, batch.graph_walk)
                    batch.outputs_ready.set()
                    continue
                raw = mgmt.accelerator_graph_runner.slot_for(lease).static_outputs
                if repeats > 1:
                    raw = dataclasses.replace(
                        raw, check_stop_buffers=columns.check_stop_buffers,
                        row_outputs={**(raw.row_outputs or {}), **(columns.row_outputs or {})},
                        output_spans={**(raw.output_spans or {}), **(columns.output_spans or {})} or None,
                    )
                output = self._collect_step(batch, mgmt, lease, raw, inputs, info)
                if repeats > 1:
                    output.run_columns = self.run_columns(batch.node_name, batch.graph_walk)
                batch.outputs = output
                self._finish_outputs(batch)
                batch.outputs = self._postprocess_outputs(batch, output)
            return [batch.outputs for _, batch in group.members]
        except Exception:
            if group.submitted or guard.reason is None:
                raise
            group.refusal = guard.reason
            return None
        finally:
            if not group.submitted:
                # Preparation may have queued plan/staging writes. Keep pages
                # and their buffers owned until those writes have drained.
                if self.device_capability.is_accelerator:
                    torch.accelerator.current_stream(self._device).synchronize()
                for step in reversed(admitted):
                    self._runner.rollback_admit(step)
                self.release_group(group)
            for _, batch in reversed(group.members):
                batch.release_waiters()

    @staticmethod
    def _custom_output_tail(submodule: NodeSubmodule) -> bool:
        kind = type(submodule)
        return (kind.postprocess is not NodeSubmodule.postprocess
                or kind.batch_postprocess is not NodeSubmodule.batch_postprocess
                or kind.filter_batched_output is not NodeSubmodule.filter_batched_output
                or kind.unpack_packed_outputs is not NodeSubmodule.unpack_packed_outputs)

    def materialize_group_inputs(self, batch: ExecutingBatch, producer: str, signal: str,
                                 selected: Mapping[int, int], width: int) -> None:
        """Give each ordinary tail its actual, still-retained stream input."""
        if not width or not self._custom_output_tail(self._submodules[batch.node_name].submodule):
            return
        if width != 1 or batch.step_context.repeats != 1:
            raise ValueError("a custom connected tail needs exactly one ordinary consumer compute")
        bindings = {}
        tensors = dict(batch.per_request_input_tensors)
        metadata = dict(batch.per_request_input_metadata)
        infos = dict(batch.per_request_info_wrapped)
        for rid, transition in selected.items():
            if rid in batch.failed_requests:
                continue
            edge, = batch.stream_windows[rid]
            receipt = edge.chunk.reservation
            chunks = [receipt.transitions[index].chunk for index in receipt.selected_indices(transition)
                      if receipt.transitions[index].chunk.num_items]
            if len(chunks) != 1 or edge.name != signal:
                raise ValueError("a landed consumer does not name its one actual input chunk")
            chunk, = chunks
            start = chunk.start_offset - receipt.origin
            positions = receipt.positions[start:start + chunk.num_items]
            if start < 0 or len(positions) != chunk.num_items:
                raise ValueError("a landed consumer input is outside its retained source coverage")
            items = self._submodules[producer].loopback_rows.ring_read(signal, rid, positions)
            value = items[0] if chunk.num_items == 1 else items
            bindings[producer, batch.node_name, signal, rid] = value
            tensors[rid] = {**tensors.get(rid, {}), signal: [value]}
            info = metadata.get(rid, EMPTY_INPUT_METADATA)
            metadata[rid] = info._replace(
                stream_chunks={**info.stream_chunks, signal: chunk._replace(reservation=None)},
            )
            infos[rid] = dataclasses.replace(infos[rid], metadata=metadata[rid])
        batch.inputs = [ConnectedInputs.materialize(inp, bindings)
                        if rid in selected and rid not in batch.failed_requests else inp
                        for rid, inp in zip(batch.request_ids, batch.inputs, strict=True)]
        batch.uniform_inputs = None
        batch.per_request_input_tensors, batch.per_request_input_metadata = tensors, metadata
        batch.per_request_info_wrapped = infos

    def collect_group_outputs(self, batch: ExecutingBatch, width: int, counts: Mapping[int, int]):
        """Close one independently executed member after its actual count lands.

        The worker closes members in graph order, publishing the producer's
        real arrivals before settling the consumer's semantic receipt.
        """
        raw, reserved = batch.outputs, batch.step_context.repeats
        if RUN_REPLAYS not in (raw.check_stop_buffers or {}):
            raise ValueError("independent member collection needs uncollected run outputs")
        mgmt, lease = self._submodules[batch.node_name], batch.step_context.slot_lease
        custom = self._custom_output_tail(mgmt.submodule)
        live = [rid for rid in batch.request_ids if counts[rid] and rid not in batch.failed_requests]
        empty = not live
        # A producing member owns the internal output ring. Its tail sees the
        # ordinary standalone representation: requested-one has no iteration
        # axis; requested-N keeps N columns (last-valid padding included), even
        # when fewer ran. The worker later routes only actual kept columns.
        producer_repeat = custom and bool(batch.ring_bases) and reserved > 1 and not empty

        def settle():
            self.settle_run(batch, width, counts)
            if batch.step is None:
                batch.step_context.run_counts = MappingProxyType(dict(counts))
            batch.publish_snapshots = self._runner.publish_snapshot(
                batch.request_ids, node_name=batch.node_name,
                graph_walk=batch.graph_walk, request_walks=batch.step_context.request_walks,
            )

        if not producer_repeat:
            settle()
        semantic = {name: value for name, value in raw.check_stop_buffers.items()
                    if name in (STREAM_COUNTS, STREAM_TRANSITION)}
        if empty:
            batch.outputs = BatchedModelOutput(check_stop_buffers=semantic, row_request_ids=tuple(batch.request_ids))
            return batch.outputs
        ordinary = custom and not producer_repeat
        if ordinary and (width != 1 or reserved != 1):
            raise ValueError("a custom connected consumer tail cannot collect multiple computations")
        columns = self.run_columns(batch.node_name, batch.graph_walk)
        visible = reserved if producer_repeat else width
        buffers = {name: (value[:, :visible] if value.ndim > 1 and value.shape[1] == reserved else value)
                   for name, value in raw.check_stop_buffers.items() if name not in (RUN_COUNTS, RUN_REPLAYS)}
        values = {name: value[:, :visible] if name in columns else value
                  for name, value in (raw.row_outputs or {}).items()}
        spans = {name: value[:, :visible] if name in columns else value
                 for name, value in (raw.output_spans or {}).items()}
        if ordinary:
            values = {name: value[:, 0] if name in columns else value for name, value in values.items()}
            spans = {name: value[:, 0] if name in columns else value for name, value in spans.items()}
            buffers = {name: value[:, 0] if name not in semantic and value.ndim > 1
                       and value.shape[1] == reserved else value for name, value in buffers.items()}
        raw = dataclasses.replace(raw, check_stop_buffers=buffers, row_outputs=values, output_spans=spans or None)
        selected = batch if len(live) == len(batch.request_ids) else batch.select_rows(live)
        if selected is not batch:
            selected.rows_only = False  # only actual logical rows are routed from this view
        try:
            inputs, info = self._pad_batch(selected, mgmt, lease)
            output = self._collect_step(selected, mgmt, lease, raw, inputs, info)
            output.run_columns = frozenset() if ordinary else columns
            selected.outputs = output
            self._finish_outputs(selected)
            output = self._postprocess_outputs(selected, output)
            if selected is not batch:
                batch.failed_requests.update(selected.failed_requests)
                positions = [batch.request_ids.index(rid) for rid in live]

                def restore(values, original):
                    if not values:
                        return original
                    restored = dict(original or {})
                    for name, value in values.items():
                        old = restored.get(name)
                        full = (old.clone() if old is not None else
                                value.new_zeros((len(batch.request_ids), *value.shape[1:])))
                        indices = to_device_nonblocking(positions, value.device, dtype=torch.long)
                        full.index_copy_(0, indices, value[:len(live)])
                        restored[name] = full
                    return restored

                output = dataclasses.replace(
                    output, check_stop_buffers=restore(output.check_stop_buffers, buffers),
                    output_spans=restore(output.output_spans, spans) or None,
                    row_request_ids=tuple(batch.request_ids), row_views=None,
                    span_request_ids=tuple(batch.request_ids) if spans or output.output_spans else None,
                )
            batch.outputs = output
        finally:
            if producer_repeat:
                settle()  # ordinary producer tail precedes resource rewind
        return batch.outputs

    def release_group(self, group: ExecutionGroup) -> None:
        """After all member tails (or failure): return their existing capture leases."""
        event = group.members[group.trigger][1].completion_event
        if event is not None:
            event.synchronize()
        for _, batch in reversed(group.members):
            lease = batch.step_context.slot_lease
            if lease is not None:
                self._submodules[batch.node_name].accelerator_graph_runner.release(lease, len(batch.request_ids))
                batch.step_context.slot_lease = None

    def _collect_step(
        self, batch: ExecutingBatch, submodule_mgmt: SubmoduleManagement,
        lease: SlotLease | None, raw: BatchedModelOutput,
        inputs: list[NodeInputs], req_info: Mapping[int, CurrentForwardPassInfo],
    ) -> BatchedModelOutput:
        """Collect one ordinary member's outputs after its resource commit."""
        if RUN_DESCRIPTORS in (raw.check_stop_buffers or {}):
            return raw  # its output owners are fixed only when publication closes
        if self._enable_nvtx:
            range_push("engine.collect_outputs")
        t0 = phase_start()
        selecting = raw.row_request_ids is not None and tuple(batch.request_ids) != raw.row_request_ids
        if selecting:
            raw = dataclasses.replace(raw, check_stop_buffers=dict(raw.check_stop_buffers or {}))
            self._add_resource_host_rows(submodule_mgmt, raw, batch.step_context.repeats)
        try:
            out = self._collect_outputs(
                submodule_mgmt, lease, raw, inputs, req_info,
                request_ids=batch.request_ids,
                step_request_ids=batch.step_context.padded_request_ids,
                graph_walk=batch.step_context.graph_walk,
                rows_only=batch.rows_only,
            )
        finally:
            phase_end("engine.collect_outputs", t0)
            if self._enable_nvtx:
                range_pop()
        if not selecting:
            self._add_resource_host_rows(submodule_mgmt, out, batch.step_context.repeats)
        return out

    def _seed_run_rows(self, batch: ExecutingBatch, slot, immune: Sequence[bool]) -> None:
        """Pin and publish incumbents before launching an empty membership window."""
        membership = slot.run.state.membership
        if not membership.idle or batch.run_rows:
            raise ValueError("a joining capture must close its previous descriptor readers before reuse")
        ctx = batch.step_context
        if ctx.resident is None or batch.step is None or len(immune) != len(batch.request_ids):
            raise ValueError("joining incumbents need their admitted resident step")
        for key, binding in ctx.resident.bindings.items():
            binding.check_horizon(batch.step.get(key), ctx)
        pins = ctx.resident.pin(batch.request_ids)
        execution = torch.cuda.current_stream(self._device) if membership.publication_stream is not None else None
        if execution is not None:
            membership.publication_stream.wait_stream(execution)
        publication = torch.cuda.stream(membership.publication_stream) if execution is not None else nullcontext()
        try:
            with publication:
                for row, (rid, resident, ignore) in enumerate(zip(batch.request_ids, pins, immune, strict=True)):
                    descriptor = membership.reserve_row(
                        rid, self._runner.request_generation(rid), row, (resident, int(ignore)),
                    )
                    if descriptor is None:
                        raise ValueError("a fresh membership window lacks descriptor credits")
                    batch.run_rows[rid] = descriptor
                for descriptor in batch.run_rows.values():
                    membership.publish_row(descriptor)
            if execution is not None:
                execution.wait_stream(membership.publication_stream)
        except BaseException:
            # No WHILE was submitted. Drain only this previously empty window,
            # and finish queued copies before making owner rows reusable.
            with torch.cuda.stream(membership.publication_stream) if execution is not None else nullcontext():
                for descriptor in batch.run_rows.values():
                    membership.cancel(descriptor)
                membership.drain()
            if execution is not None:
                membership.publication_stream.synchronize()
            membership.acknowledge(membership.control.cpu().tolist())
            for descriptor in batch.run_rows.values():
                membership.release(descriptor)
            batch.run_rows.clear()
            ctx.resident.unpin(pins)
            raise

    def admit_run_row(self, batch: ExecutingBatch, active: ExecutingBatch, row: int) -> RunRow | None:
        """Prepare one ordinary logical step beside an active capture.

        Called on its membership publication stream, after the worker takes
        the ordinary graph in-flight flag/request hold. No host plan, capture
        staging, or forward runs here. The worker attaches the returned receipt
        before publishing its descriptor; an unpublished receipt can settle at
        zero using the same actual-count path as an executed one.
        """
        if (batch.node_name != active.node_name or batch.graph_walk != active.graph_walk
                or len(batch.request_ids) != 1 or batch.run_rows or batch.committed is not None
                or batch.chunk_ranges or batch.step_context.request_walks):
            raise ValueError("joining needs one fresh ordinary row of the active walk")
        mgmt = self._submodules[batch.node_name]
        lease = active.step_context.slot_lease
        if lease is None or mgmt.resident is None:
            raise ValueError("joining needs an active resident capture")
        slot = mgmt.accelerator_graph_runner.slot_for(lease)
        membership = None if slot.run is None else slot.run.state.membership
        if membership is None or not 0 <= row < lease.bucket.bs:
            raise ValueError("joining row falls outside the active membership capture")
        membership._check_publication_stream()
        if set(active.run_rows) != set(active.request_ids):
            raise ValueError("joining needs pinned incumbent descriptor readers")
        ctx = batch.step_context
        if ctx.repeats != active.step_context.repeats or ctx.repeats < 1:
            raise ValueError("joining rows must reserve the complete active capture horizon")
        if batch.inputs is None or batch.failed_requests or batch.skipped_rids or batch.cached_rids:
            raise ValueError("joining needs a successfully prepared ordinary row")
        rid = batch.request_ids[0]
        ctx.slot, ctx.slot_lease, ctx.resident = lease.slot, lease, mgmt.resident
        step = batch.step = self._declare_step(batch, [rid], batch.inputs, mgmt.submodule, ctx, batch.step)
        if step is None or step.cg_key_info != node_capture_key(lease.bucket.cg_key_info):
            raise ValueError("joining step does not match the active capture declaration")
        step.set_ctx(ctx)
        outcome = self._runner.admit_rows(step)
        if not outcome.ok:
            return None  # admit_rows already unwound this candidate alone
        plan = descriptor = None
        try:
            if not outcome.ready:
                return None
            plan = ctx.resident.prepare_rows(step, ctx)
            inputs, infos = self._pad_batch(batch, mgmt, lease)
            engine_inputs, preprocessed = self._preprocess_step(
                batch, mgmt, [rid], inputs, infos, ctx, lease, step,
            )
            for name, captured in slot.static_inputs.items():
                if not isinstance(captured, torch.Tensor) and preprocessed.get(name) != captured:
                    raise ValueError("joining inputs change a captured non-tensor specialization")
            if slot.row_heads:
                if batch.kept_rows != [True] and any(
                    not isinstance(preprocessed.get(name), torch.Tensor) for name in slot.row_heads.values()
                ):
                    raise ValueError("joining row has no routed value for a captured input owner")
                self._write_routed_rows(batch, mgmt, slot, engine_inputs, preprocessed, stager=membership.stager)
            if slot.run.window is not None:
                snapshot = batch.iteration_columns
                if not isinstance(snapshot, IterationWindow):
                    raise ValueError("joining iteration inputs need their prepared owner window")
                mgmt.submodule.iteration_tables.stage_window(
                    lease, plan.slots.tolist(), snapshot, stager=membership.stager,
                )
                batch.iteration_columns = snapshot
            policy = mgmt.device_stops[batch.graph_walk]
            immune = policy is not None and policy.limits(batch.per_request_info_wrapped[rid]).ignore_eos
            descriptor = membership.reserve_row(
                rid, self._runner.request_generation(rid), row, (int(plan.slots[0]), int(immune)),
            )
            if descriptor is None:
                return None
            if mgmt.rings:
                self._reserve_rings(mgmt, batch)
            ctx.resident.flush(plan, stager=membership.stager)
            self._runner.commit(step, track_rows=True)
            batch.committed = step
            batch.run_started = active.run_started
            ctx.resident_plan = plan
            batch.run_rows[rid] = descriptor
            ctx.resident.advance(plan)
            return descriptor
        finally:
            if batch.committed is None:
                # Failed preparation may have queued writes into the candidate's
                # private rows. Finish those before releasing/reusing owners.
                if membership.publication_stream is not None:
                    membership.publication_stream.synchronize()
                if descriptor is not None:
                    membership.cancel(descriptor)
                    membership.release(descriptor)
                if plan is not None:
                    ctx.resident.release(rid)
                    ctx.resident.unpin(tuple(plan.slots.tolist()))
                self._runner.rollback_admit(step)

    def finish_run_outputs(self, batch: ExecutingBatch, outputs: BatchedModelOutput) -> None:
        """Finish a retained capture after all descriptor copies are enqueued.

        Runs on the same execution stream as collection. The worker keeps the
        capture out of scheduling until this transition and retains owner/
        descriptor readers until the collection completion event lands.
        """
        if (RUN_DESCRIPTORS not in (batch.outputs.check_stop_buffers or {})
                or RUN_DESCRIPTORS in (outputs.check_stop_buffers or {})):
            raise ValueError("a retained run must finish once with identity-resolved outputs")
        lease = batch.step_context.slot_lease
        if lease is None:
            raise ValueError("a retained run has no capture lease")
        self._submodules[batch.node_name].accelerator_graph_runner.release(lease, len(batch.request_ids))
        batch.outputs = outputs
        # Each positive logical receipt already published its own loop-back
        # metadata. This physical capture may also have zero-execution rows.
        batch.outputs_ready.set()

    def run_membership(self, batch: ExecutingBatch):
        """The declared membership owner of this retained capture."""
        lease = batch.step_context.slot_lease
        if lease is None or not batch.run_rows:
            return None
        runner = self._submodules[batch.node_name].accelerator_graph_runner
        return runner.slot_for(lease).run.state.membership

    def collect_run_outputs(
        self, batch: ExecutingBatch,
        records: Sequence[tuple[BatchedModelOutput, Sequence[Sequence[int]]]],
    ) -> BatchedModelOutput:
        """Collect a settled logical receipt under its own inputs and metadata.

        The worker settles the complete admitted membership first and returns
        proved-zero rows separately. This tail only sees rows that executed;
        descriptor identities, never the physical capture rows, select values.
        Capture storage and owner pins remain held until these copies finish.
        """
        ctx = batch.step_context
        if (ctx.run_counts is None or not batch.request_ids
                or any(ctx.run_counts.get(rid, 0) <= 0 for rid in batch.request_ids)):
            raise ValueError("run output collection needs settled, nonempty logical rows")
        width = max(ctx.run_counts[rid] for rid in batch.request_ids)
        if ctx.repeats != width:
            raise ValueError("run output width must match its settled logical receipt")
        columns = self.run_columns(batch.node_name, batch.graph_walk)
        rows = []
        for rid in batch.request_ids:
            descriptor = batch.run_rows[rid]
            actual = sum(ids.count(descriptor.handle) for _, identities in records for ids in identities)
            output = self._run_row_output(records, descriptor.handle, width)
            if output is None or actual != ctx.run_counts[rid]:
                raise ValueError("descriptor output count disagrees with its settled receipt")
            for values in (output.row_outputs, output.output_spans):
                for name, value in (values or {}).items():
                    if name not in columns:
                        values[name] = value[:, actual - 1]
            rows.append(output)

        def join(attribute: str, excluded=frozenset()):
            values = [getattr(row, attribute) or {} for row in rows]
            names = values[0].keys() - excluded
            if any(value.keys() - excluded != names for value in values):
                raise ValueError("logical rows have incompatible retained outputs")
            return {name: torch.cat([value[name] for value in values]) for name in names}

        raw = BatchedModelOutput(
            check_stop_buffers=join("check_stop_buffers", {RUN_COUNTS, RUN_REPLAYS}),
            row_outputs=join("row_outputs") or None, output_spans=join("output_spans") or None,
        )
        mgmt = self._submodules[batch.node_name]
        if mgmt.rings:
            # A width-one physical capture still stores an iteration axis.
            ring_raw = raw
            if batch.ring_width == 1:
                ring_raw = BatchedModelOutput(
                    check_stop_buffers={k: v[:, 0] for k, v in (raw.check_stop_buffers or {}).items()},
                    row_outputs={k: v[:, 0] if k in columns else v for k, v in (raw.row_outputs or {}).items()},
                )
            self._write_rings(mgmt, batch, ring_raw)
        outputs = self._collect_outputs(
            mgmt, None, raw, batch.inputs, batch.per_request_info_wrapped,
            batch.request_ids, tuple(batch.request_ids), batch.graph_walk, rows_only=batch.rows_only,
        )
        outputs.run_columns = self.run_columns(batch.node_name, batch.graph_walk)
        outputs.loopback_rids = batch.loopback_rids
        self._record_loopback(batch)
        return self._postprocess_outputs(batch, outputs)

    @staticmethod
    def _run_row_output(
        records: Sequence[tuple[BatchedModelOutput, Sequence[Sequence[int]]]], descriptor: int, width: int,
    ) -> BatchedModelOutput | None:
        """Compact one logical row from completed, descriptor-stamped runs.

        The caller retains each capture until this copy finishes and supplies
        identity columns from the same completed publication as the outputs.
        Physical row/launch boundaries do not define this request's ordinal.
        Zero execution has no output to collect or pass to a model tail.
        """
        if descriptor <= 0 or width < 1:
            raise ValueError("a run row needs a positive descriptor and logical width")
        selected = []
        shapes = []
        metadata = {RUN_COUNTS, RUN_REPLAYS, RUN_DESCRIPTORS}
        for index, (raw, identities) in enumerate(records):
            if raw.per_rid_outputs or raw.packed_outputs:
                raise ValueError("joining outputs must retain row-addressed iteration columns")
            columns = len(identities[0]) if identities else 0
            if any(len(row) != columns for row in identities):
                raise ValueError("run identity rows must have one common width")
            shapes.append((len(identities), columns))
            for column in range(columns):
                rows = [row for row, ids in enumerate(identities) if ids[column] == descriptor]
                if len(rows) > 1:
                    raise ValueError("a descriptor executed on two physical rows in one iteration")
                selected.extend((index, row, column) for row in rows)
        if not selected:
            return None
        if len(selected) > width:
            raise ValueError("a run row executed beyond its admitted logical horizon")

        def gather(attribute: str, excluded=frozenset()) -> dict[str, torch.Tensor]:
            sources = [getattr(raw, attribute) or {} for raw, _ in records]
            names = sources[0].keys() - excluded
            if any(source.keys() - excluded != names for source in sources):
                raise ValueError("queued runs changed their retained output names")
            output = {}
            for name in names:
                values = [source[name] for source in sources]
                if any(not isinstance(value, torch.Tensor) or value.ndim < 2
                       or tuple(value.shape[:2]) != shape for value, shape in zip(values, shapes, strict=True)):
                    raise ValueError(f"run output {name!r} has no matching row/iteration columns")
                pieces = [values[index][row:row + 1, column:column + 1] for index, row, column in selected]
                # Preserve ordinary repeated-step tail semantics; actual counts
                # below distinguish these padding values from executed work.
                pieces.extend([pieces[-1]] * (width - len(pieces)))
                output[name] = torch.cat(pieces, dim=1)
            return output

        stops = gather("check_stop_buffers", metadata)
        rows = gather("row_outputs")
        spans = gather("output_spans")
        anchor = next(iter(stops.values()), next(iter(rows.values()), next(iter(spans.values()), None)))
        if anchor is None:
            raise ValueError("a joining run retained no row-addressed outputs")
        count = anchor.new_full((1,), len(selected), dtype=torch.long)
        stops.update({RUN_COUNTS: count, RUN_REPLAYS: count})
        return BatchedModelOutput(check_stop_buffers=stops, row_outputs=rows or None, output_spans=spans or None)

    @staticmethod
    def _add_resource_host_rows(
        submodule_mgmt: SubmoduleManagement, out: BatchedModelOutput, repeats: int,
    ) -> None:
        """Adds ``Resource.step_host_rows`` to the step's stop buffers."""
        for key, resource in submodule_mgmt.resources.items():
            if not (rows := resource.step_host_rows(repeats)):
                continue
            if out.check_stop_buffers is None:
                raise RuntimeError(
                    f"resource {key!r} has rows for the host (e.g. log-probs), but this node's "
                    "forward hands over no row-addressed stop buffers",
                )
            if clash := rows.keys() & out.check_stop_buffers.keys():
                raise RuntimeError(f"resource {key!r} adds host rows {sorted(clash)} the step already has")
            out.check_stop_buffers = {**out.check_stop_buffers, **rows}

    def _exec_per_request(
        self, batch: ExecutingBatch
    ) -> dict[str, NameToTensorList]:
        """Full step cycle per request, without a whole-forward lease, outputs merged.

        The unbatchable fallback: no whole-forward capture applies (a lease
        would have set ``running_batched``), so each request declares/admits/
        plans/commits on its own, keeping the plan matched to its
        single-request forward. Each request still takes its own
        piecewise-region leases in ``_declare_and_admit``, so a captured inner
        loop can replay per request.
        """
        nvtx = self._enable_nvtx
        submodule_mgmt = self._submodules[batch.node_name]
        merged = BatchedModelOutput(
            per_rid_outputs={rid: {} for rid in batch.request_ids}
        )
        launched = False

        # Step 1: loop through all of the requests for admit errors.
        # Each request is its own cycle, so each takes its own slot off the
        # shared counter — to the rotation these sub-steps ARE steps. The last
        # one doesn't advance it, so the batch leaves the counter one past the
        # slot it last used, exactly as a single-forward step would.
        slot = batch.slot or 0
        steps: dict[str, SubmoduleStep] = {}
        ctxs: dict[str, StepContext] = {}
        for i, (rid, inp) in enumerate(
            zip(batch.request_ids, batch.inputs, strict=True)
        ):
            ctxs[rid] = StepContext(
                request_ids=(rid,),
                graph_walk=batch.step_context.graph_walk,
                slot=slot, capture=False,
            )
            if i != len(batch.request_ids) - 1:
                slot = submodule_mgmt.lease_slot()
            # declare (here) and drive (below) are separate loops, so the
            # region runners are pointed at the slot in both
            submodule_mgmt.set_piecewise_slot(ctxs[rid].slot)
            admit_outcome, steps[rid] = self._declare_and_admit(
                batch, rids=[rid], inputs=[inp],
                submodule=submodule_mgmt.submodule,
                ctx=ctxs[rid], nvtx=nvtx
            )
            if not admit_outcome.ok:
                # all or nothing, as a batched step's admit is: the requests
                # admitted ahead of this one will not run either
                for earlier in reversed(batch.request_ids[:i]):
                    if steps[earlier] is not None:
                        self._runner.rollback_admit(steps[earlier])
                return merged

        # Step 2: drive step, plan -> forward -> commit loop.
        # Nothing stages before here, so only this loop fences, in two places:
        # on reuse inside the loop (the rotation wraps after `num_slots`
        # requests), and on the way out — see below.
        fence = (
            submodule_mgmt.needs_slot_fence
            and self.device_capability.is_accelerator
        )
        slot_events: dict[int, Any] = {}

        # a raise outside the per-rid try fails the batch; the drain and
        # `commit_done` still run
        try:
            for rid, inp in zip(batch.request_ids, batch.inputs, strict=True):
                req_info = {rid: batch.per_request_info_wrapped[rid]}
                slot = ctxs[rid].slot
                submodule_mgmt.set_piecewise_slot(slot)
                in_flight = slot_events.get(slot)
                if in_flight is not None:
                    in_flight.synchronize()

                if nvtx:
                    range_push(f"engine.per_request.{rid}")
                try:
                    raw, _ = self._drive_step(
                        batch, submodule_mgmt, [rid], [inp], req_info, ctxs[rid],
                        lease=None, running_batched=False,
                        step=steps[rid], set_launch=not launched,
                    )
                    if raw is None:
                        continue
                    launched = True

                    merged.update(self._collect_outputs(
                        submodule_mgmt, None, raw, [inp], req_info,
                        request_ids=[rid], step_request_ids=(rid,),
                        graph_walk=batch.step_context.graph_walk,
                    ))
                except Exception as error:
                    # this rid fails alone, as in prepare_inputs, and keeps its
                    # admitted state (which queued kernels may still write) until
                    # its REMOVE. Under TP/SP the ranks stay in step only if every
                    # rank raises here.
                    batch.register_failure(rid, error)
                finally:
                    # on every exit: a raise after plan can leave this slot's
                    # staging copy queued
                    if fence:
                        slot_events[slot] = torch.Event(device=self._device)
                        slot_events[slot].record()
                    if nvtx:
                        range_pop()
        finally:
            # Releasing `commit_done` lets the plan thread pre-plan the next
            # batch, which stages `next_slot` while this batch's work may still
            # be queued. Only that slot needs draining: every other one the loop
            # touched is re-consumed a batch or more later, by which point the
            # worker has synced on this step.
            event = slot_events.get(submodule_mgmt.next_slot)
            if event is not None:
                event.synchronize()
            batch.commit_done.set()
        # Same optional 1-step launch throttle as _exec_single. This path is
        # always eager (never capturing), but the guard is kept for parity.
        if (
            _ENGINE_STEP_SYNC
            and self.device_capability.is_accelerator
            and not self.device_capability.is_capturing()
        ):
            if nvtx:
                range_push("engine.await_outputs")
            try:
                torch.accelerator.current_stream(self._device).synchronize()
            finally:
                if nvtx:
                    range_pop()
        return merged

    def _declare_and_admit(
        self,
        batch: ExecutingBatch,
        rids: list[str],
        inputs: list[NodeInputs],
        submodule: NodeSubmodule,
        nvtx: bool,
        ctx: StepContext,
        step: SubmoduleStep | None=None,
    ) -> tuple[AdmitOutcome, SubmoduleStep]:
        step = self._declare_step(batch, rids, inputs, submodule, ctx, step)
        return self._admit_step(batch, rids, inputs, ctx, step), step

    def _admit_step(
        self, batch: ExecutingBatch, rids: Sequence[int], inputs: list[NodeInputs],
        ctx: StepContext, step: SubmoduleStep | None,
    ) -> AdmitOutcome:
        if step is None:
            return ADMIT_OK
        nvtx = self._enable_nvtx
        if batch.chunk_ranges:
            ctx.chunk_tokens_after = {
                rid: inp.chunk_total - inp.chunk_start - inp.input_seq_len
                for rid, inp in zip(rids, inputs, strict=True)
                if not inp.is_final_chunk
            }
            ctx.reserve_whole_prefill = PerReqSetting.over(
                ctx.chunk_tokens_after,
                lambda rid: self._chunking.policy(
                    batch.node_name, batch.step_context.walk_of(rid),
                ).reserve_whole_prefill,
            )
        step.set_ctx(ctx)
        if nvtx:
            range_push("engine.admit")
        t0 = phase_start()
        try:
            admit_outcome = self._runner.admit(step)
        finally:
            phase_end("engine.admit", t0)
            if nvtx:
                range_pop()
        if not admit_outcome.ok:
            batch.register_admit_error(admit_outcome.reason, admit_outcome.failed_resource)
            for rid in batch.first_chunk_rids:
                self._chunking.release(rid, batch.node_name)
        return admit_outcome

    def _declare_step(
        self, batch: ExecutingBatch, rids: Sequence[int], inputs: list[NodeInputs],
        submodule: NodeSubmodule, ctx: StepContext, step: SubmoduleStep | None,
    ) -> SubmoduleStep | None:
        """Declare once, separately from admission so a unit can preflight it."""
        if step is None:
            self._maybe_lease_piecewise_regions(batch.node_name, ctx, inputs)
            if self._enable_nvtx:
                range_push("engine.declare_step")
            t0 = phase_start()
            try:
                step = engine_step(
                    submodule,
                    graph_walk=batch.graph_walk, request_ids=rids, inputs=inputs,
                    slot_lease=ctx.slot_lease,
                    piecewise_leases=ctx.piecewise_leases,
                    per_request_info=batch.per_request_info,
                )
            finally:
                phase_end("engine.declare_step", t0)
                if self._enable_nvtx:
                    range_pop()
        return step

    def _maybe_lease_piecewise_regions(
        self, node_name: str, ctx: StepContext, inputs: list[NodeInputs],
    ) -> None:
        """Take each opted-in region's slot, and report it on the context.

        A region still declares, plans and commits its own step — its shape is
        its own, and several regions can share a resource — but the outer
        ``declare_step`` has to know which resources are already spoken for so
        it doesn't declare them a second time. The lease is settled from the
        same batch size and token count the region will resolve at replay, so
        the two agree by construction.

        Nothing to take when the whole forward replays: the region is baked
        inside that graph and runs no Python, so the outer step owns its work.
        """
        runners = self._submodules[node_name].piecewise_runners
        if not runners or ctx.slot_lease is not None:
            return
        bs = len(inputs)
        total_tokens = sum(inp.input_seq_len for inp in inputs)
        leases = {}
        for label, runner in runners.items():
            # a batch the region refuses runs it eagerly, with no slot
            if not runner.lease_before_step or not runner.can_lease(inputs, ctx.request_ids):
                continue
            lease = runner.lease_slot(bs, total_tokens)
            if lease is not None:
                leases[label] = lease
        if leases:
            ctx.set_piecewise_leases(leases)

    def _plan_step(
        self,
        batch: ExecutingBatch,
        submodule_mgmt: SubmoduleManagement,
        request_ids: list[int],
        inputs: list[NodeInputs],
        req_info: Mapping[int, CurrentForwardPassInfo],
        ctx: StepContext,
        lease: SlotLease | None,
        step: SubmoduleStep | None,
    ) -> tuple[ModelInputsFromEngine, dict[str, Any]]:
        """Plan and preprocess an admitted ordinary step, without launching it."""
        nvtx = self._enable_nvtx

        if step is not None:
            if lease is not None and step.cg_key_info != node_capture_key(lease.bucket.cg_key_info):
                # The slot was leased from `cg_key_info` before the step was
                # declared. If the two disagree the replay runs a graph that
                # was planned for a different declaration — silently wrong
                # output, so fail here instead. Derive both from one place.
                raise RuntimeError(
                    f"{batch.node_name}: leased {lease.bucket} but the step "
                    f"declares cg_key_info={step.cg_key_info!r}; "
                    "cg_key_info() and declare_step disagree"
                )
            step.set_ctx(ctx)
            # before any copy this plan issues, the resident stage's included
            self._order_after_pre_plans(batch.node_name)
            # promoted and inline plans per resource, apart
            plan_phase = "res.plan" if batch.preplan_event is not None else "res.plan_fresh"
            if ctx.resident is not None:
                ctx.plan_phase = plan_phase
                t0 = phase_start()
                ctx.resident.plan(step, ctx)
                phase_end(f"{plan_phase}.resident", t0)

            # Admit was already done, can move on straight to plan
            if nvtx:
                # promoted = a pre-plan was consumed; fresh = planned inline
                range_push(
                    "engine.plan.promoted" if batch.preplan_event is not None
                    else "engine.plan.fresh"
                )
            t0 = phase_start()
            try:
                self._runner.plan(step, plan_phase)
            finally:
                phase_end(
                    "engine.plan.promoted" if batch.preplan_event is not None
                    else "engine.plan.fresh",
                    t0,
                )
                if nvtx:
                    range_pop()

        return self._preprocess_step(batch, submodule_mgmt, request_ids, inputs, req_info, ctx, lease, step)

    def _preprocess_step(
        self, batch: ExecutingBatch, submodule_mgmt: SubmoduleManagement,
        request_ids: list[int], inputs: list[NodeInputs], req_info: Mapping[int, CurrentForwardPassInfo],
        ctx: StepContext, lease: SlotLease | None, step: SubmoduleStep | None,
    ) -> tuple[ModelInputsFromEngine, dict[str, Any]]:
        """Prepare declared inputs without planning or writing capture buffers."""
        nvtx = self._enable_nvtx
        cg_runner, submodule = submodule_mgmt.accelerator_graph_runner, submodule_mgmt.submodule
        rids = list(ctx.padded_request_ids)
        laps = PhaseLaps("engine.")
        engine_inputs = ModelInputsFromEngine(
            request_ids=rids,
            per_request_info=req_info,
            resources=submodule_mgmt.resources,
            piecewise_runners=submodule_mgmt.piecewise_runners,
            piecewise_leases=ctx.piecewise_leases,
            # padding rows get their own states, like their cache streams:
            # the submodule indexes this by step id, not by real rid
            per_request_states=LazyRequestStates(submodule, rids),
            captured=lease is not None,
            # `inputs` is padded after the real rows under a lease
            real_seq_len=sum(inp.input_seq_len for inp in inputs[:len(request_ids)]),
            step=step,
            per_request_input_metadata=batch.per_request_input_metadata,
            loopback_rows=submodule_mgmt.loopback_rows,
            rows_in_graph=lease is not None and bool(cg_runner.slot_for(lease).row_heads),
        )
        laps.lap("engine_inputs")
        if nvtx:
            range_push("engine.preprocess")
        t0 = phase_start()
        try:
            columns = None
            if getattr(submodule, "ITERATION_INPUTS", None):
                run = None if lease is None else cg_runner.slot_for(lease).run
                loop = (
                    self._loops[(batch.node_name, ctx.graph_walk)]
                    if run is not None and run.window is not None else None
                )
                batch.iteration_columns = self._iteration_columns(
                    batch.node_name, submodule, ctx, rids, req_info, batch.per_request_input_tensors,
                    actual_loop=loop,
                )
                columns = batch.iteration_columns
                if isinstance(columns, IterationWindow):
                    columns = columns.columns
                engine_inputs.iteration_inputs = columns
            if ctx.resident is None and ctx.repeats == 1:
                preprocessed = submodule.preprocess(ctx.graph_walk, engine_inputs=engine_inputs, inputs=inputs)
            else:
                with watch_table_reads() as reads:
                    preprocessed = submodule.preprocess(ctx.graph_walk, engine_inputs=engine_inputs, inputs=inputs)
                table = next((r for r in reads if r != LOOP_COUNTER), None)
                if table is not None and ctx.resident is not None:
                    # a branch the capture template didn't take at load: the step's rows fail
                    raise ValueError(
                        f"{batch.node_name}/{ctx.graph_walk}: a resident step's preprocess read {table}, "
                        "which its graph fills after it; serve it with resident_planning: off"
                    )
                _refuse_counter_read(batch.node_name, ctx, reads, "preprocess")
            for name, column in (columns or {}).items():
                preprocessed.setdefault(name, column[:, 0])
        finally:
            phase_end("engine.preprocess", t0)
            if nvtx:
                range_pop()

        return engine_inputs, preprocessed

    def _drive_step(
        self,
        batch: ExecutingBatch,
        submodule_mgmt: SubmoduleManagement,
        request_ids: list[str],
        inputs: list[NodeInputs],
        req_info: Mapping[str, CurrentForwardPassInfo],
        ctx: StepContext,
        lease: SlotLease | None,
        running_batched: bool,
        step: SubmoduleStep | None,
        set_launch: bool,
    ) -> tuple[dict | None, SubmoduleStep | None]:
        """Plan, preprocess, forward and commit one admitted ordinary step.

        ``request_ids`` are the real rids (for admit-fail / collect); the model
        runs over ``ctx.request_ids`` (padded under a lease). Returns
        ``(raw_outputs, step)``; ``raw_outputs`` is None when admit failed.
        """
        engine_inputs, preprocessed = self._plan_step(
            batch, submodule_mgmt, request_ids, inputs, req_info, ctx, lease, step,
        )
        nvtx = self._enable_nvtx
        cg_runner = submodule_mgmt.accelerator_graph_runner
        if self._enable_profile and batch.exec_timings.fwd_start is None:
            batch.exec_timings.fwd_start = time.perf_counter()
        # The waiter is released inside the forward, immediately before the
        # launch that drops the GIL — not here, which is still several GIL-held
        # staging copies away from it.
        release_event = batch.launch_started_event if set_launch else None
        if nvtx:
            # the launch/enqueue span, not the GPU work: `synchronize=True`
            # here would drain the stream and destroy the overlap
            range_push("engine.forward")
        t0 = phase_start()
        try:
            # eager, compiled (its first call traces under it too) or replayed
            with fp32_precision_scope(submodule_mgmt.fp32_precision):
                raw = self._forward(
                    batch, submodule_mgmt, cg_runner, engine_inputs, preprocessed,
                    lease, running_batched, request_ids, step, release_event,
                )
        finally:
            phase_end("engine.forward", t0)
            if nvtx:
                range_pop()

        if batch.stream_windows and batch.run_started is None:
            # Physical execution is known before output collection or a model
            # tail can fail. Per-request forwards aggregate into the batch's
            # existing execution facts, not their temporary resource contexts.
            actual = dict(batch.step_context.run_counts or {})
            actual.update(dict.fromkeys(request_ids, ctx.repeats))
            batch.step_context.run_counts = MappingProxyType(actual)

        self._commit_step(batch, submodule_mgmt, step, ctx)
        return raw, step

    def _commit_step(
        self, batch: ExecutingBatch, submodule_mgmt: SubmoduleManagement,
        step: SubmoduleStep | None, ctx: StepContext,
    ) -> None:
        """Settle one submitted step through the ordinary resource owners."""
        nvtx = self._enable_nvtx
        if step is not None:
            if nvtx:
                range_push("engine.commit")
            t0 = phase_start()
            try:
                self._runner.commit(step, track_rows=batch.run_started is not None)
                batch.committed = step
                if submodule_mgmt.resident is not None:
                    submodule_mgmt.resident.commit(step, ctx)
            finally:
                phase_end("engine.commit", t0)
                if nvtx:
                    range_pop()

    def _stage_capture(
        self, batch: ExecutingBatch, submodule_mgmt: SubmoduleManagement,
        cg_runner, engine_inputs: ModelInputsFromEngine, preprocessed: dict[str, Any],
        lease: SlotLease, request_ids: Sequence[int],
    ):
        """Stage an admitted capture's existing owners without launching it."""
        graph_walk = batch.step_context.graph_walk
        ctx = batch.step_context
        if ctx.resident is not None:
            ctx.resident.flush(ctx.resident_plan)
        slot = cg_runner.slot_for(lease)
        if slot.row_heads:
            self._write_routed_rows(batch, submodule_mgmt, slot, engine_inputs, preprocessed)
        if slot.run is not None:
            policy = submodule_mgmt.device_stops[graph_walk]
            infos = engine_inputs.per_request_info
            immune = [policy is not None and policy.limits(infos[rid]).ignore_eos for rid in request_ids]
            slot.run.state.stage(len(request_ids), immune, ctx.repeats, submodule_mgmt.refills[graph_walk])
            if slot.run.state.membership is not None:
                self._seed_run_rows(batch, slot, immune)
            if slot.run.window is not None:
                submodule_mgmt.submodule.iteration_tables.stage_window(
                    lease, ctx.resident_plan.slots.tolist(), batch.iteration_columns,
                )
            else:
                slot.run.stage_inputs(batch.iteration_columns)
            batch.run_started = slot.run.state.started
        streamed = submodule_mgmt.device_stream_tokens.get(graph_walk)
        if streamed is not None and ctx.repeats > 1:
            batch.token_stream = cg_runner.token_stream(lease, *streamed)
            batch.token_stream.begin(request_ids)
        return slot

    def _forward(
        self,
        batch: ExecutingBatch,
        submodule_mgmt: SubmoduleManagement,
        cg_runner,
        engine_inputs: ModelInputsFromEngine,
        preprocessed: dict[str, Any],
        lease: SlotLease | None,
        running_batched: bool,
        request_ids: list[str],
        step: SubmoduleStep,
        release_event: "threading.Event | None" = None,
    ) -> BatchedModelOutput:
        """Replay the leased slot, or run the eager forward.

        ``release_event`` is set as late as possible before the call that drops
        the GIL — the replay, or the submodule forward on the eager path."""
        graph_walk = batch.step_context.graph_walk
        if lease is not None:
            ctx = batch.step_context
            slot = self._stage_capture(
                batch, submodule_mgmt, cg_runner, engine_inputs, preprocessed, lease, request_ids,
            )
            streamed = submodule_mgmt.device_stream_tokens.get(graph_walk)
            out = cg_runner.run_forward(
                lease, preprocessed,
                plan_done_event=batch.preplan_event,
                launch_started_event=release_event,
            )
            if slot.run is not None and (ctx.repeats > 1 or slot.run.state.membership is not None):
                stops, rows, spans = slot.run.results(ctx.repeats)
                out = dataclasses.replace(
                    out, check_stop_buffers=stops, row_outputs={**(out.row_outputs or {}), **rows},
                    output_spans={**(out.output_spans or {}), **spans} or None,
                )
            elif slot.run is not None:
                # A width-one launch can consist entirely of rows stopped by
                # its predecessor. Preserve ordinary output shapes, but settle
                # those rows from the device's counts as for a longer run.
                out = dataclasses.replace(
                    out, check_stop_buffers={**(out.check_stop_buffers or {}), **slot.run.execution_counts()},
                )
            elif ctx.repeats > 1:
                columns = submodule_mgmt.run_columns.get(graph_walk, frozenset())
                # a signal in the rows moves through them, in the graph
                carries = {s: pair for s, pair in submodule_mgmt.carries[graph_walk].items() if s not in slot.row_heads}
                part = cg_runner.run_part(
                    lease, carries, self._runner.replay_columns(step, ctx.repeats), batch.iteration_columns,
                )
                stops, rows, spans = cg_runner.repeat(
                    [part], ctx.repeats, columns,
                    None if batch.token_stream is None else (batch.token_stream, streamed[0]),
                )
                out = dataclasses.replace(
                    out, check_stop_buffers=stops, row_outputs={**(out.row_outputs or {}), **rows},
                    output_spans={**(out.output_spans or {}), **spans} or None,
                )
            return out
        if release_event is not None:
            release_event.set()
        if running_batched:
            out = BatchedModelOutput.coerce(
                submodule_mgmt.forward_batched(
                    graph_walk, engine_inputs=engine_inputs, **preprocessed
                )
            )
            self._keep_rows(submodule_mgmt, graph_walk, engine_inputs.request_ids, out)
        else:
            assert len(request_ids) == 1, (
                "the unbatched forward takes one request; batch of "
                f"{len(request_ids)} needs running_batched"
            )
            raw = submodule_mgmt.forward(graph_walk, engine_inputs=engine_inputs, **preprocessed)
            out = raw if isinstance(raw, BatchedModelOutput) else BatchedModelOutput.coerce({request_ids[0]: raw})
        return out

    def _finish_ordinary_iterations(self, batch, mgmt, output: BatchedModelOutput) -> None:
        """Publish an ordinary fallback after its complete node computation.

        This includes plain captures and output-name normalization in the tail.
        Dependent fallback steps are rebuilt after their predecessor settles.
        """
        layouts = self.iteration_layouts(batch.node_name) or {}
        rids = output.row_request_ids or tuple(batch.request_ids)
        infos_by_rid = batch.per_request_info_wrapped
        for walk, loop in layouts.items():
            positions = [i for i, rid in enumerate(rids) if rid in infos_by_rid
                         and rid not in batch.cached_rids and rid not in batch.failed_requests
                         and batch.step_context.walk_of(rid) == walk]
            if not positions:
                continue
            selected = [rids[i] for i in positions]
            infos = [infos_by_rid[rid] for rid in selected]
            policy = mgmt.device_stops[walk]
            # A cap-only Loop never schedules beyond its remaining room. It
            # needs no persistent EOS marker; a new Loop entry resets position.
            limits = [] if policy is None else [policy.limits(info) for info in infos]
            stopped = to_device_nonblocking([
                limit.max_iters is not None and info.metadata.dynamic_loop_iter_counts.get(loop.name, 0) + 1
                >= limit.max_iters for info, limit in zip(infos, limits, strict=True)
            ] if policy is not None else [False] * len(infos), self._device, dtype=torch.bool)
            if policy is not None and policy.reads_token:
                tokens = output_named(output, policy.token_signal)
                if tokens is None:
                    tokens = torch.stack([
                        output.per_rid_outputs[rid][policy.token_signal][0].reshape(-1)[0] for rid in selected
                    ])
                else:
                    pick = to_device_nonblocking(positions, self._device, dtype=torch.long)
                    tokens = tokens.reshape(tokens.shape[0], -1)[:, 0].index_select(0, pick)
                ids = to_device_nonblocking(list(policy.eos_tokens), self._device, dtype=tokens.dtype)
                immune = to_device_nonblocking([limit.ignore_eos for limit in limits], self._device, dtype=torch.bool)
                stopped.logical_or_(torch.isin(tokens, ids) & ~immune)
            entries = to_device_nonblocking([
                self._iteration_entry(rid, info.metadata, loop) for rid, info in zip(selected, infos, strict=True)
            ], self._device, dtype=torch.long)
            mgmt.loopback_rows.finish_iteration(walk, loop, selected, entries, stopped)

    def postprocess_batch(
        self, batch: ExecutingBatch, outputs: BatchedModelOutput,
    ) -> None:
        """Per-rid ``submodule.postprocess``, e.g. the non-capturable tail of a
        walk whose graph covered only the forward.

        Per-rid like ``prepare_inputs``: a raise fails that request and leaves
        the rest of the batch to route normally.
        """
        if self._enable_nvtx:
            range_push("engine.postprocess")
        t0 = phase_start()
        try:
            self._postprocess_batch(batch, outputs)
        finally:
            phase_end("engine.postprocess", t0)
            if self._enable_nvtx:
                range_pop()

    def _postprocess_batch(
        self, batch: ExecutingBatch, outputs: BatchedModelOutput,
    ) -> None:
        submodule_mgmt = self._submodules[batch.node_name]
        submodule = submodule_mgmt.submodule
        failed = batch.failed_requests
        columns = loop_back = ()
        if batch.step_context.repeats > 1 or outputs.run_columns:
            columns = submodule_mgmt.run_columns.get(batch.graph_walk, frozenset())
            # loop-back signals a node may name after a run column (``text_inputs = new_token``)
            loop_back = [s for s in submodule_mgmt.carries.get(batch.graph_walk, {}) if s not in columns]
        for rid, node_inputs in zip(batch.request_ids, batch.inputs, strict=True):
            if failed and rid in failed:
                # its per-request forward raised: nothing to finish, and a
                # second raise here would replace the error the client gets
                continue
            rid_outputs = outputs.get(rid, {})
            try:
                submodule.postprocess(
                    request_id=rid,
                    request_info=batch.per_request_info_wrapped[rid],
                    outputs=rid_outputs,
                    inputs=node_inputs,
                )
            except Exception as error:
                batch.register_failure(rid, error)
                continue
            if loop_back:
                _last_iteration_loops_back(rid_outputs, columns, loop_back)
        if type(submodule).batch_postprocess is not NodeSubmodule.batch_postprocess:
            live = [
                (rid, node_inputs)
                for rid, node_inputs in zip(batch.request_ids, batch.inputs, strict=True)
                if rid not in batch.failed_requests
            ]
            if not live:
                return
            rids = [rid for rid, _ in live]
            try:
                submodule.batch_postprocess(
                    outputs,
                    {rid: batch.per_request_info_wrapped[rid] for rid in rids},
                    [node_inputs for _, node_inputs in live],
                    piecewise_runners=submodule_mgmt.piecewise_runners,
                )
            except Exception as error:
                logger.exception(
                    "batch_postprocess failed for requests %s (node=%s, walk=%s)",
                    rids, batch.node_name, batch.step_context.graph_walk,
                )
                for rid in rids:
                    batch.register_failure(rid, error)

    def exec_and_postprocess(
        self, batch: ExecutingBatch
    ) -> BatchedModelOutput:
        """The forward and its per-rid tail, which belong to the same step: a
        walk that captured only its forward finishes in ``postprocess``.

        ``prepare_inputs`` and the stop check stay outside — the worker has to
        place those itself.
        """
        beside = self._stage_lock is not None and self._beside_chain(batch.node_name, batch.graph_walk)
        with self._stage_lock if beside else nullcontext():
            outputs = self.exec(batch)
        return self._postprocess_outputs(batch, outputs)

    def _postprocess_outputs(self, batch: ExecutingBatch, outputs: BatchedModelOutput) -> BatchedModelOutput:
        """The ordinary host tail, shared by independent and grouped steps."""
        if RUN_DESCRIPTORS in (outputs.check_stop_buffers or {}):
            return outputs  # descriptor grouping precedes every identity-dependent tail
        laps = PhaseLaps("engine.")
        # A failed admit means no forward ran and every rid's outputs are empty,
        # so the tail has nothing to consume; the worker re-drives the step.
        # A rows-only step has no per-request outputs; only a batch tail runs.
        if batch.admit_error is None:
            if not batch.rows_only:
                self.postprocess_batch(batch, outputs)
            else:
                submodule_mgmt = self._submodules[batch.node_name]
                submodule = submodule_mgmt.submodule
                if type(submodule).batch_postprocess is not NodeSubmodule.batch_postprocess:
                    submodule.batch_postprocess(
                        outputs, batch.per_request_info_wrapped, batch.inputs,
                        piecewise_runners=submodule_mgmt.piecewise_runners,
                    )
            mgmt = self._submodules[batch.node_name]
            if (batch.run_started is None and mgmt.device_stops
                    and getattr(mgmt.submodule, "ITERATION_INPUTS", None)):
                self._finish_ordinary_iterations(batch, mgmt, outputs)
            for rid in batch.cached_rids:
                self._runner.complete_cached_walk(rid, batch.node_name, batch.step_context.walk_of(rid))
        laps.lap("tail")
        return outputs

    def reads_tokens(self, node_name: str) -> bool:
        """Whether the node keeps its landed tokens (``NodeSubmodule.record_tokens``)."""
        mgmt = self._submodules.get(node_name)
        return mgmt is not None and type(mgmt.submodule).record_tokens is not NodeSubmodule.record_tokens

    def record_tokens(self, batch: ExecutingBatch, walk_rids: Mapping[str, Sequence[int]], host_rows: HostRows) -> None:
        """Hand the node the host tokens of the rows it reads them for."""
        submodule = self._submodules[batch.node_name].submodule
        buffer = host_rows.buffers[submodule.token_stop.buffer]
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        tokens = None
        for walk, rids in walk_rids.items():
            wanted = submodule.token_rows(walk, [rid for rid in rids if rid in row_of])
            if wanted:
                if tokens is None:
                    tokens = buffer.reshape(buffer.shape[0], -1).tolist()
                submodule.record_tokens(
                    walk, {rid: tokens[row_of[rid]] for rid in wanted}, batch.per_request_info_wrapped,
                )

    def ask_stop_odds(self, node_name: str, graph_walk: str) -> str | None:
        """Before capture: have the sampler of ``graph_walk``'s token stop
        score its stop ids each sample; why it can't, or None."""
        policy = next(
            (p for p in (self.get_stop_policies(node_name, graph_walk) or {}).values() if p.reads_token), None,
        )
        if policy is None or not policy.eos_tokens:
            return "node: its stop reads no token"
        resource = self._submodules[node_name].resources.get(policy.sampler)
        if resource is None:
            return "node: its stop names no sampler"
        return resource.ask_stop_odds(graph_walk, policy.eos_tokens)

    def get_stop_policies(self, node_name: str, graph_walk: str) -> dict[str, StopPolicy] | None:
        """The node's declared stop policies on ``graph_walk``
        (``NodeSubmodule.get_stop_policies``); None when ``check_stop``
        decides, or the node has no submodule here."""
        mgmt = self._submodules.get(node_name)
        return None if mgmt is None else mgmt.submodule.get_stop_policies(graph_walk)

    def check_stop_for_batch(
        self, batch: ExecutingBatch, outputs: Mapping[str, Mapping[str, list[torch.Tensor]]],
        host_rows: HostRows | None = None,
    ) -> dict[str, set[str]]:
        """Each rid's ``submodule.check_stop``, as rid -> its stops (loop names,
        or ``END_PARTITION``).

        Reads tensor values, so it belongs on the caller's slow-postprocess
        path rather than in ``exec``, where the ``.item()`` / ``.cpu()`` would
        block the GPU thread. Per-rid like the other stages: a raise fails that
        request onto the batch, because letting it escape would abandon the
        check for the rest of the batch and leave their loops running past
        their stop condition.

        With ``host_rows``, ``check_stop_batched`` checks the batch in one call;
        if it raises, the per-request path runs so a failure lands on its own
        request.
        """
        submodule = self._submodules[batch.node_name].submodule
        if host_rows is not None:
            try:
                batched = submodule.check_stop_batched(
                    batch.request_ids, batch.per_request_info_wrapped, host_rows,
                )
            except Exception:
                logger.debug(
                    "batched check_stop failed (node=%s); checking per request",
                    batch.node_name, exc_info=True,
                )
                batched = None
            if batched is not None:
                return batched
        stops: dict[str, set[str]] = {}
        for rid in batch.request_ids:
            rid_outputs = outputs.get(rid)
            if not rid_outputs:
                continue
            try:
                rid_stops = submodule.check_stop(
                    rid, batch.per_request_info_wrapped[rid], rid_outputs
                )
            except Exception as error:
                batch.register_failure(rid, error)
                continue
            if rid_stops:
                stops[rid] = rid_stops
        return stops

    def finalize_batch(
        self, batch: ExecutingBatch,
        publish_request_ids: list[int] | None = None,
    ) -> dict[str, dict[str, PublishedInfo]]:
        if self._enable_nvtx:
            range_push("engine.finalize_batch")
        try:
            rids = (
                batch.request_ids if publish_request_ids is None
                else publish_request_ids
            )
            if batch.publish_snapshots is not None:
                allowed = set(rids)
                batch.publish_snapshots = {
                    rid: snapshot for rid, snapshot in batch.publish_snapshots.items() if rid in allowed
                }
                if not _LAZY_PUBLISH:
                    self.materialize_publish(batch, rids)
                return batch.resource_publish_info
            if _LAZY_PUBLISH:
                # Snapshot now, on the gpu thread, so the lengths are this
                # step's; finish in ``materialize_publish`` for the rids whose
                # frame or completion carries the publication.
                batch.publish_snapshots = self._runner.publish_snapshot(
                    rids,
                    node_name=batch.node_name,
                    graph_walk=batch.step_context.graph_walk,
                    request_walks=batch.step_context.request_walks,
                )
                return batch.resource_publish_info
            # Returns rid -> {resource label -> published info}
            published = self._runner.publish(
                rids,
                node_name=batch.node_name,
                graph_walk=batch.step_context.graph_walk,
                request_walks=batch.step_context.request_walks,
            )
            for rid, info in batch.per_request_info.items():
                if rid not in published:
                    continue
                info.update_publish_info(published[rid])
            return published
        finally:
            if self._enable_nvtx:
                range_pop()

    def materialize_publish(
        self, batch: ExecutingBatch, request_ids: list[int],
    ) -> None:
        """Finish the publication ``finalize_batch`` snapshotted, for the rids
        whose frame or completion carries it. Idempotent per rid."""
        snaps = batch.publish_snapshots
        if not snaps:
            return
        todo = [
            rid for rid in request_ids
            if rid in snaps and rid not in batch.resource_publish_info
        ]
        if not todo:
            return
        published = self._runner.publish_from_snapshots(
            snaps, todo,
            node_name=batch.node_name,
            graph_walk=batch.step_context.graph_walk,
            request_walks=batch.step_context.request_walks,
        )
        for rid in todo:
            per_key = published.get(rid) or {}
            batch.resource_publish_info[rid] = per_key
            info = batch.per_request_info.get(rid)
            if info is not None and per_key:
                info.update_publish_info(per_key)

    def finalize_stopped_requests(
        self,
        batch: ExecutingBatch,
        request_ids: list[int],
    ) -> None:
        """Publish state that must include the last iteration of a stopped loop."""
        # A row that ran nothing this step -- one the prefix cache served whole
        # -- has no iteration here to publish, though its loop still stops.
        request_ids = [rid for rid in request_ids if rid in batch.per_request_info]
        if not request_ids:
            return
        published = self._runner.publish_after_stop(
            request_ids,
            node_name=batch.node_name,
            graph_walk=batch.step_context.graph_walk,
            request_walks=batch.step_context.request_walks,
        )
        for rid in request_ids:
            rid_published = published.get(rid, {})
            batch.per_request_info[rid].update_publish_info(rid_published)
            merge_publish_info(
                batch.resource_publish_info.setdefault(rid, {}),
                rid_published,
            )

    def _collect_outputs(
        self,
        submodule_mgmt: SubmoduleManagement,
        lease: SlotLease | None,
        raw_outputs: BatchedModelOutput,
        inputs: list[NodeInputs],
        req_info: Mapping[str, CurrentForwardPassInfo],
        request_ids: list[str],
        step_request_ids: tuple[str, ...],
        graph_walk: str,
        rows_only: bool = False,
    ) -> BatchedModelOutput:
        """Per-rid outputs for the real requests: drop the padding rows and map
        a captured graph's keys back to real ids.

        ``rows_only`` (see ``ExecutingBatch.rows_only``) keeps the row views
        and the stop buffers but builds no per-rid dicts: nothing downstream
        reads them on such a step, and at 128 rows they were a few hundred
        allocations on the gpu thread.

        A captured forward emits its per-rid entries under the slot's padding
        ids (those were the batch at capture time), so entry ``i`` belongs to
        ``request_ids[i]`` on either path.
        """
        submodule = submodule_mgmt.submodule
        out_ids = None
        original = raw_outputs.row_request_ids
        if original is not None and tuple(request_ids) != original:
            if len(set(original)) != len(original) or len(set(request_ids)) != len(request_ids):
                raise ValueError("logical output rows must have unique request identities")
            positions = {rid: index for index, rid in enumerate(original)}
            if any(rid not in positions for rid in request_ids) or raw_outputs.packed_outputs:
                raise ValueError("logical output selection needs known row-addressed values")
            selected = [positions[rid] for rid in request_ids]
            capture_ids = (original if lease is None
                           else submodule_mgmt.accelerator_graph_runner.slot_for(lease).dummy_rids)
            out_ids = [capture_ids[index] for index in selected]
            indices = {}

            def project(values):
                if values is None:
                    return None
                result = {}
                for name, value in values.items():
                    if not isinstance(value, torch.Tensor) or not value.ndim or value.shape[0] < len(original):
                        raise ValueError(f"output {name!r} has no original request-row dimension")
                    if value.device not in indices:
                        indices[value.device] = to_device_nonblocking(selected, value.device, dtype=torch.long)
                    result[name] = value.index_select(0, indices[value.device])
                return result

            raw_outputs = dataclasses.replace(
                raw_outputs, row_outputs=project(raw_outputs.row_outputs),
                check_stop_buffers=project(raw_outputs.check_stop_buffers),
                output_spans=project(raw_outputs.output_spans), row_request_ids=tuple(request_ids),
            )
        if (
            rows_only
            and raw_outputs.check_stop_buffers is not None
            and not raw_outputs.per_rid_outputs
            and not raw_outputs.packed_outputs
            and type(submodule).filter_batched_output
            is NodeSubmodule.filter_batched_output
        ):
            # No clone either: the worker takes the host copy on this thread,
            # on this stream, right behind the step (``_stage_host_rows``), so
            # a later replay cannot overwrite the rows first, and nothing else
            # reads them on the device. Padded rows past the real ones ride
            # along and are never read.
            return BatchedModelOutput(
                check_stop_buffers=raw_outputs.check_stop_buffers,
                row_request_ids=tuple(request_ids),
                rows_only=True,
            )
        if out_ids is None:
            out_ids = (step_request_ids if lease is None
                       else submodule_mgmt.accelerator_graph_runner.slot_for(lease).dummy_rids)
        outputs: dict[str, NameToTensorList] = {}

        row_clones, row_views = self._merge_per_rid(
            outputs, raw_outputs, request_ids, out_ids, submodule, req_info,
            rows_only=rows_only,
        )
        self._merge_unpacked(
            outputs, raw_outputs, request_ids, submodule,
            inputs[:len(request_ids)], req_info,
        )
        spans = raw_outputs.output_spans
        if spans:
            spans = self._real_spans(submodule, graph_walk, spans, len(request_ids))
        return BatchedModelOutput(
            per_rid_outputs=outputs,
            rows_only=rows_only and not outputs,
            # the stop buffer that is a row output shares its clone: one copy
            # of the sampled tokens per step, not two
            check_stop_buffers=raw_outputs.clone_check_stop_buffers(
                reuse=row_clones,
            ),
            # row i is request_ids[i] as the forward ran them, not the
            # worker's later-rewritten request list
            row_request_ids=tuple(request_ids),
            row_views=row_views,
            output_spans=spans or None,
            span_request_ids=tuple(request_ids) if spans else None,
        )

    @staticmethod
    def _real_spans(
        submodule: NodeSubmodule, graph_walk: str,
        spans: dict[str, torch.Tensor], num_rows: int,
    ) -> dict[str, torch.Tensor]:
        """The step's spans narrowed to its real rows (a replay pads past
        them), refusing a span on an output the walk does not declare and a
        step that spans some declared outputs but not all: the per-request
        path merges one forward's spans into another's by name, so every
        forward on a walk has to return the same names."""
        declared = submodule.spanned_outputs(graph_walk)
        undeclared = spans.keys() - declared
        if undeclared:
            raise ValueError(
                f"{type(submodule).__name__} returned spans for {sorted(undeclared)} "
                f"on walk {graph_walk!r}, which spanned_outputs does not declare"
            )
        missing = declared - spans.keys()
        if missing:
            raise ValueError(
                f"{type(submodule).__name__} returned spans for {sorted(spans)} on walk "
                f"{graph_walk!r} but none for {sorted(missing)}: a step that returns spans "
                f"returns one for every output spanned_outputs declares"
            )
        for name, span in spans.items():
            # a step that repeated holds a span a replay: [rows, repeats, 2]
            if span.dim() not in (2, 3) or span.shape[-1] != 2 or span.shape[0] < num_rows:
                raise ValueError(
                    f"{type(submodule).__name__}: the span of {name!r} is "
                    f"{tuple(span.shape)}, not [{num_rows}, 2]"
                )
        # A view, not a clone: the gpu thread copies the spans to the host on
        # this stream right behind the step, before a later replay can write them.
        return {name: span[:num_rows] for name, span in spans.items()}

    def _merge_per_rid(
        self,
        outputs: dict[str, NameToTensorList],
        raw_outputs: BatchedModelOutput,
        request_ids: list[str],
        out_ids: list[str],
        submodule: NodeSubmodule,
        req_info: Mapping[str, CurrentForwardPassInfo],
        rows_only: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, ...]] | None]:
        """Fold the forward's per-rid entries into ``outputs``.

        ``row_outputs`` are cloned once for the batch and handed out as row
        views; per-rid entries are cloned individually after
        ``filter_batched_output``. A name in both resolves to the per-rid one.
        Returns the row clones so the caller can share them, and on the
        rows-only path the views themselves (name -> views, row i for
        ``request_ids[i]``), which the store describes without reading each.
        """
        row_clones = raw_outputs.clone_row_outputs(len(request_ids))
        # The common decode case: only row outputs, and a submodule that does
        # not filter per request. One ``split`` per output makes every row
        # view in a single call, the same ``tensor[i:i+1]`` views the loop
        # below would slice one at a time, and the per-rid dicts are built
        # without the candidate/from_rows bookkeeping.
        if (
            not raw_outputs.per_rid_outputs
            and type(submodule).filter_batched_output
            is NodeSubmodule.filter_batched_output
        ):
            row_views = {
                name: tensor.split(1)
                for name, tensor in row_clones.items()
                if isinstance(tensor, torch.Tensor) and tensor.dim()
            }
            if row_views and not rows_only:
                for i, rid in enumerate(request_ids):
                    merged = outputs.setdefault(rid, {})
                    for name, views in row_views.items():
                        if i < len(views):
                            merged[name] = [views[i]]
            return row_clones, (row_views or None)
        for i, (rid, out_id) in enumerate(
            zip(request_ids, out_ids, strict=False)
        ):
            rid_out = raw_outputs.get(out_id)
            # names served by a view of the batch clone: don't clone again
            from_rows: set[str] = set()
            candidates: dict[str, Any] = {}
            for name, tensor in row_clones.items():
                if isinstance(tensor, torch.Tensor) and tensor.dim() and (
                    i < tensor.shape[0]
                ):
                    candidates[name] = [tensor[i : i + 1]]
                    from_rows.add(name)
            if isinstance(rid_out, dict):
                for name, value in rid_out.items():
                    candidates[name] = value
                    from_rows.discard(name)
            if not candidates:
                continue
            # captured output keys are fixed for graph compat; the submodule
            # decides which of them this real request should receive
            rid_out = submodule.filter_batched_output(req_info.get(rid), candidates)
            merged = outputs.setdefault(rid, {})
            for key, value in rid_out.items():
                if key in from_rows:
                    merged[key] = value
                elif isinstance(value, list):
                    merged[key] = [t.clone() for t in value]
                elif isinstance(value, torch.Tensor):
                    merged[key] = [value.clone()]
                else:
                    merged[key] = value
        return row_clones, None

    def _merge_unpacked(
        self,
        outputs: dict[str, NameToTensorList],
        raw_outputs: BatchedModelOutput,
        request_ids: list[str],
        submodule: NodeSubmodule,
        real_inputs: list[NodeInputs],
        req_info: Mapping[str, CurrentForwardPassInfo],
    ) -> None:
        """Let the submodule slice any batch-wide packed sentinels.

        A captured region can't do this itself: the per-request slice ends
        depend on the real seq_lens, which only reach it through the plan.
        """
        if (
            not raw_outputs.packed_outputs
            and type(submodule).unpack_packed_outputs
            is NodeSubmodule.unpack_packed_outputs
        ):
            # nothing packed and the base no-op: skip the per-rid lengths
            return
        unpacked = submodule.unpack_packed_outputs(
            static_output=raw_outputs.packed_outputs,
            request_ids=request_ids,
            real_seq_lens=[inp.input_seq_len for inp in real_inputs],
            inputs=real_inputs,
            per_request_info=req_info,
        )
        for rid, rid_out in unpacked.items():
            outputs.setdefault(rid, {}).update(rid_out)


    def splits_by_capture_key(self, node_name: str, graph_walk: str) -> bool:
        """Whether ``capture_group`` can tell this walk's requests apart: the
        submodule opts in and the node captures, the walk's whole forward or
        any piecewise region (a region is not tied to a walk)."""
        submodule_mgmt = self._submodules[node_name]
        if not submodule_mgmt.submodule.split_batches_by_capture_key(graph_walk):
            return False
        if submodule_mgmt.piecewise_runners:
            return True
        cg_runner = submodule_mgmt.accelerator_graph_runner
        return cg_runner is not None and cg_runner.captures_walk(graph_walk)

    def capture_group(
        self, node_name: str, graph_walk: str, request_id: int,
        request_info: CurrentForwardPassInfo,
        input_metadata: InputMetadata | None = None,
    ) -> Any | None:
        """The capture bucket this one request would replay, by
        ``cg_key_info``; None when the walk has no capture or the submodule
        does not split by it. On a node whose only capture is a piecewise
        region it is a scheduler group, not a bucket.

        A batch spanning two groups matches no capture and runs eager, so the
        scheduler batches each group on its own when the submodule opts in
        (``split_batches_by_capture_key``). ``input_metadata`` is the step's
        input facts the lease will see, e.g. the stream chunk it consumes.
        """
        if not self.splits_by_capture_key(node_name, graph_walk):
            return None
        submodule_mgmt = self._submodules[node_name]
        if input_metadata is None:
            return submodule_mgmt.submodule.cg_key_info(graph_walk, {request_id: request_info})
        return submodule_mgmt.submodule.cg_key_info(
            graph_walk, {request_id: request_info},
            per_request_input_metadata={request_id: input_metadata},
        )

    def _add_split_row_order(self, spec: NodeResourceSpec, nodes: set[str]) -> None:
        """Record the row order ``spec``'s split kernels read (``CombinedSplit``)."""
        for split in spec.combined_splits():
            last = frozenset(split.prefill_walks)
            for node in nodes:
                key = (node, split.combined_walk)
                if self._split_prompt_walks.setdefault(key, last) != last:
                    raise ValueError(
                        f"resource {spec.resource_key!r}: {node!r} splits combined walk "
                        f"{split.combined_walk!r} with prefill walks {sorted(last)}, "
                        f"another resource with {sorted(self._split_prompt_walks[key])}"
                    )

    def order_split_rows(
        self, node_name: str, graph_walk: str, request_ids: list[int],
        request_walks: Mapping[int, str],
    ) -> list[int]:
        """``request_ids`` as a step of ``graph_walk`` runs them: a walk run on
        split kernels takes its prefill walks' rows last, in their order."""
        last = self._split_prompt_walks.get((node_name, graph_walk))
        if last is None:
            return request_ids
        first = [rid for rid in request_ids if request_walks.get(rid) not in last]
        if len(first) == len(request_ids):
            return request_ids
        return first + [rid for rid in request_ids if request_walks.get(rid) in last]

    @staticmethod
    def _max_rows_in_combined(
        resource_key: str,
        nodes: set[str],
        submodules: dict[str, NodeSubmodule],
        model: "Model | None",
    ) -> dict[str, int | None]:
        """Each walk of ``nodes`` to its ``max_rows_in_combined``, one value a
        walk across the nodes a resource serves."""
        if model is None:
            return {}
        caps: dict[str, int | None] = {}
        for walk, graph in model.get_graph_walk_graphs().items():
            for node in sorted(nodes & set(graph.get_nodes())):
                cap = submodules[node].max_rows_in_combined(walk)
                if walk in caps and caps[walk] != cap:
                    raise ValueError(
                        f"resource {resource_key!r}: the nodes it serves cap "
                        f"walk {walk!r} differently in a combined step "
                        f"({caps[walk]} and {cap})"
                    )
                caps[walk] = cap
        return caps

    def get_max_rows_in_combined(self, node_name: str, graph_walk: str) -> int | None:
        """Rows of ``graph_walk`` a step mixing it with another walk may hold."""
        return self._submodules[node_name].submodule.max_rows_in_combined(graph_walk)

    def get_step_order(self, node_name: str, graph_walk: str) -> StepOrder:
        """The submodule's ``step_order`` for ``graph_walk``."""
        return self._submodules[node_name].submodule.step_order(graph_walk)

    def get_max_batch_size(self, node_name: str, graph_walk: str) -> int | None:
        """Most requests this node will take in one step, or None for no cap.

        Two sources: what the submodule says it can batch, and the largest
        bucket this walk has a capture config for, captured or not — past it a
        step runs eager at a size the capture budget never measured, so the
        scheduler splits instead. Splitting is the scheduler's job, not the
        engine's: the pieces then pipeline like any other batch instead of
        running back to back.
        """
        submodule_mgmt = self._submodules[node_name]
        caps = [submodule_mgmt.submodule.max_batch_size(graph_walk)]
        if submodule_mgmt.capture_runner is not None:
            caps.append(
                submodule_mgmt.capture_runner.max_batch_size_for(graph_walk)
            )
        capped = [cap for cap in caps if cap is not None]
        return min(capped) if capped else None

    def iteration_layouts(self, node_name: str) -> Mapping[str, LoopLayout] | None:
        """Graph-owned contexts for walks whose device runs need actual progress."""
        mgmt = self._submodules[node_name]
        if not mgmt.device_stops or not getattr(mgmt.submodule, "ITERATION_INPUTS", None):
            return None
        return {walk: self._loops[(node_name, walk)] for walk in mgmt.device_stops
                if (node_name, walk) in self._loops}

    def streamed_iteration_inputs(self, node_name: str) -> frozenset[str]:
        """The streamed inputs ``node_name`` reads an item a replay of
        (``IterationInput.streamed``)."""
        submodule = self._submodules[node_name].submodule
        specs = getattr(submodule, "ITERATION_INPUTS", {})
        return frozenset(spec.source for spec in specs.values() if spec.streamed)

    def run_columns(self, node_name: str, graph_walk: str) -> frozenset[str]:
        """The per-iteration outputs a step of this walk that repeats keeps a
        column of per replay (``keep_run_columns``)."""
        return self._submodules[node_name].run_columns.get(graph_walk, frozenset())

    def _iteration_entry(self, rid: int, metadata: InputMetadata, loop: LoopLayout) -> list[int]:
        if metadata.graph_pass is None:
            raise ValueError("actual Loop progress needs a snapshotted graph pass")
        return [self._runner.request_generation(rid), metadata.graph_pass,
                *(metadata.dynamic_loop_iter_counts.get(name, 0) for name in loop.ancestors)]

    def _iteration_columns(
        self, node_name: str, submodule: NodeSubmodule, ctx: StepContext, rids: Sequence[int],
        req_info: Mapping[int, Any], inputs: Mapping[int, Mapping[str, list]] = MappingProxyType({}),
        *, actual_loop: LoopLayout | None = None,
    ) -> dict[str, torch.Tensor] | IterationWindow:
        """The step's iteration inputs, a column a replay: each row at its own
        iteration of its walk's Loop (0 outside one), a padding row at its
        ``capture_fill``."""
        streamed = [spec for spec in submodule.ITERATION_INPUTS.values() if spec.streamed]
        rows, origins, entries, horizons = [], [], [], []
        real = set(ctx.request_ids) if actual_loop is not None else ()
        for rid in rids:
            if actual_loop is not None and rid not in real:
                rows.append(None)
                origins.append(0)
                entries.append([-1] * (2 + len(actual_loop.ancestors)))
                horizons.append(0)
                continue
            info = req_info.get(rid)
            loop = self._loops.get((node_name, ctx.walk_of(rid)))
            # the counter as the engine reads it, not watched
            counts = getattr(info, "metadata", info)
            k = 0 if loop is None or counts is None else counts.dynamic_loop_iter_counts.get(loop.name, 0)
            if actual_loop is not None:
                if loop != actual_loop or counts is None or counts.iteration_origin is None:
                    raise ValueError("a device input window needs its row's snapshotted Loop origin")
                origin = counts.iteration_origin
                if origin < 0 or origin > k:
                    raise ValueError("an input window origin must precede its projected Loop position")
                origins.append(origin)
                entries.append(self._iteration_entry(rid, counts, actual_loop))
                horizons.append(k - origin + ctx.repeats)
                k = origin
            # a stream window is items of the node's Loop; a row outside it has none
            if streamed:
                # A skipped retained reader can retry eagerly after its
                # predecessor settled. It still holds the original snapshot;
                # without the device selector, trim its consumed prefix here.
                origin = getattr(counts, "iteration_origin", None)
                start = k - origin if actual_loop is None and origin is not None else 0
                window = {} if loop is None else _window_tables(
                    streamed, inputs.get(rid), getattr(info, "metadata", None), start=start,
                )
                rows.append((rid, k, window))
            else:
                rows.append((rid, k))
        if actual_loop is None:
            return iteration_columns(submodule, rows, ctx.repeats, self._device)
        columns = iteration_columns(
            submodule, rows, max(horizons, default=ctx.repeats), self._device, horizons=horizons,
        )
        if len(rids) == 1 and horizons == [1]:
            # The ordinary single-item fast path is an arena view. A queued
            # window must survive a later source assignment in that arena.
            columns = {name: value.clone() for name, value in columns.items()}
        return IterationWindow(
            columns, to_device_nonblocking(origins, self._device, dtype=torch.long),
            to_device_nonblocking(entries, self._device, dtype=torch.long),
        )

    def keep_run_columns(self, node_name: str, graph_walk: str, signals: Collection[str]) -> str | None:
        """Have a step of this walk that repeats keep a column per replay of
        each of ``signals`` (its per-iteration outputs); why it can't, or None."""
        mgmt = self._submodules[node_name]
        runner = mgmt.accelerator_graph_runner
        if signals and runner is not None:
            if (refusal := runner.run_column_refusal(graph_walk, signals)) is not None:
                return f"runner: {refusal}"
        mgmt.run_columns[graph_walk] = frozenset(signals)
        return None

    def _probe_carries(
        self, node_name: str, runner: AcceleratorGraphRunner, walk: str, signals: Collection[str],
    ) -> None:
        """Before capture: whether ``walk``'s ``preprocess`` hands each
        loop-back signal to the forward unchanged, which a run skips between
        replays (``AcceleratorGraphRunner.prepare_template``)."""
        mgmt = self._submodules[node_name]
        if runner.batched_refusal(walk) is not None:
            return  # no batched capture of it: its steps never repeat
        mgmt.template_steps[walk] = runner.template_step(walk)
        with watch_table_reads() as reads:
            refusal = runner.prepare_template(walk, mgmt.submodule.repeat_carry(walk, signals))
        if refusal is None and LOOP_COUNTER in reads:
            refusal = f"its captured preprocess reads {LOOP_COUNTER}, which a run's later replays don't renew"
        if refusal is not None:
            mgmt.carry_refusals[walk] = f"node: {refusal}"

    def repeat_refusal(self, node_name: str, graph_walk: str, signals: Collection[str] = ()) -> str | None:
        """Why a step of this walk can't replay with no host work between, or
        None if it may: the deployment, the walk's step, its capture and every
        resource of the node each answer. ``signals``: the walk's loop-back
        signals, each carried between replays (``NodeSubmodule.repeat_carry``).
        After capture, which some resources decide at."""
        mgmt = self._submodules[node_name]
        runner = mgmt.accelerator_graph_runner
        # a host-planned step replays its one plan: each resource answers for its part
        host_planned, host_step = False, None
        if self.resident_refusal(node_name, graph_walk) is not None:
            if graph_walk not in mgmt.template_steps:
                # no batched capture of it was probed
                why = "it has no capture" if runner is None else runner.batched_refusal(graph_walk)
                return f"runner: {why or 'its captures were not probed'}"
            # None: the step touches no resource, so there is nothing to plan
            host_planned, host_step = True, mgmt.template_steps[graph_walk]
        if (refusal := mgmt.carry_refusals.get(graph_walk)) is not None:
            return refusal
        carries = mgmt.submodule.repeat_carry(graph_walk, signals)
        if runner is not None and (refusal := runner.carry_refusal(graph_walk, carries)) is not None:
            return f"runner: {refusal}"
        for key, resource in mgmt.resources.items():
            part = None if host_step is None else host_step.get(key)
            if host_planned and part is None:
                continue  # the step doesn't touch it
            if (refusal := resource.repeat_refusal(graph_walk, part)) is not None:
                return f"resource {key!r}: {refusal}"
        mgmt.carries[graph_walk] = carries
        return None

    def repeat_note(self, node_name: str, graph_walk: str) -> str | None:
        """What the node's resources changed so this walk can repeat, or None."""
        notes = (r.repeat_note(graph_walk) for r in self._submodules[node_name].resources.values())
        return "; ".join(note for note in notes if note) or None

    def get_max_batch_tokens(self, node_name: str, graph_walk: str) -> int | None:
        return self._chunking.max_batch_tokens(node_name, graph_walk)

    def set_token_budgets(
        self, overrides: Mapping[str, Any], walks: set[str],
        combined_walk_of: Mapping[tuple[str, str], str] | None = None,
        override_mixed: Collection[str] = (),
        no_chunking: Collection[str] = (),
        min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    ) -> None:
        self._chunking.set_budgets(
            overrides, walks, combined_walk_of, override_mixed, no_chunking,
            min_chunk_tokens,
        )

    def min_chunk_tokens(self, budget: int) -> int:
        return self._chunking.min_chunk_tokens(budget)

    def set_chunking_policies(
        self, policies: Mapping[tuple[str, str], ChunkingPolicy],
    ) -> None:
        self._chunking.set_policies(policies)

    def walk_token_caps(self, node_name: str, graph_walk: str) -> dict[str, int] | None:
        return self._chunking.walk_caps(node_name, graph_walk)

    def supports_chunked_prefill(self, node_name: str, graph_walk: str) -> bool:
        return self._chunking.chunkable(node_name, graph_walk)

    def declares_input_sequence_len(self, node_name: str) -> bool:
        """Whether the node's submodule reports row lengths before prepare."""
        impl = type(self._submodules[node_name].submodule).get_input_sequence_len
        return impl is not NodeSubmodule.get_input_sequence_len

    def input_sequence_len(
        self, node_name: str, graph_walk: str,
        request_info: CurrentForwardPassInfo, inputs: NameToTensorList,
        rid: int | None = None,
    ) -> InputSeqLenInfo | None:
        """A row's length; a chunkable keyed row's leaves out the prefix the
        cache holds, which ``rid``'s lease keeps until its first chunk applies it.
        A row the cache holds whole measures 0, so the budget never cuts it and
        prepare serves it (`_skip_cached_prefix`): a chunk cut from 0 would
        write over the lease the walks before it were served from."""
        submodule = self._submodules[node_name].submodule
        info = submodule.get_input_sequence_len(graph_walk, request_info, inputs)
        policy = self._chunking.policy(node_name, graph_walk)
        if (
            info is None or rid is None
            or graph_walk not in self._keyed_walks.get(node_name, ())
            # the manager's, not the policy's: a walk the deployment does not
            # chunk is never cut, so it starts at its first token
            or not self._chunking.chunkable(node_name, graph_walk)
            or (policy.reuses_cached_prefix is not None
                and not policy.reuses_cached_prefix(request_info))
        ):
            return info
        prefix = self._runner.resolve_cached_prefix(
            rid, node_name, graph_walk, _MeasuredRow(info.seq_len),
        )
        matched = min(prefix.tokens if prefix is not None else 0, info.seq_len)
        if matched <= 0:
            return info
        return InputSeqLenInfo(info.seq_len - matched, cached_prefix=matched)

    def check_ready(
        self, node_name: str, request_id: str,
        request_info: CurrentForwardPassInfo,
        allow_reload: bool = True,
    ) -> FullAdmitOutcome:
        """Whether this node can run the request now.

        An offloaded request is brought back first; it stays not-ready until
        that fits, which is the scheduler's cue to run something else (and,
        on OOM, to evict). Then each resource takes in whatever the request
        published elsewhere — a KV transfer from a prefill worker, say — and
        reports whether that has landed.

        The outcome, not a bool: a resource can report the request
        unservable (``AdmitRuntimeError``), and the caller has to fail it
        rather than scan it again forever.
        """
        if self.is_offloaded(node_name, request_id):
            if not allow_reload:
                return FULL_ADMIT_NOT_READY
            if not self.reload_request(node_name, request_id):
                return FULL_ADMIT_NOT_READY
        return self._runner.admit_retrieve(
            rid=request_id, node_name=node_name,
            graph_walk=request_info.graph_walk,
            published=request_info.resource_publish_info,
        )

    def admit_waiting(
        self, published_of: Callable[[str, str], Mapping[str, PublishedInfo] | None],
    ) -> list[tuple[str, str]]:
        """The (node, rid) pairs admitted since `check_ready` answered them ``queued``; see
        `Admission.admit_waiting`."""
        return self._runner.admit_waiting(published_of)

    def reserve_replay_slot(
        self, batch: ExecutingBatch, *, capture_step: SubmoduleStep | None = None,
    ) -> SlotLease | None:
        """Lease the slot this batch will replay on, before it is dispatched.

        Reserving up front is what puts pre-plan(N+1) and replay(N) on
        different slots. No captured graph for the batch's shape leaves the
        lease unset, i.e. the step runs eager.

        A batch that hasn't been through ``prepare_inputs`` has no token count
        yet, so only a batched capture can serve it — see
        ``CudaGraphRunner.select_batched_bucket``.
        """
        if batch.step_context.slot_lease is not None:
            return batch.step_context.slot_lease
        submodule_mgmt = self._submodules[batch.node_name]

        if self._walk_lanes and self._beside_chain(batch.node_name, batch.graph_walk):
            # The chain holds its rotation of slots (a run in flight, the next
            # step staged), and every per-step buffer is keyed by slot: a step
            # beside it takes a slot of its lane's, past the rotation, and runs
            # eager, never on a slot's captures.
            batch.set_slot(submodule_mgmt.num_slots + self._walk_lanes[batch.node_name, batch.graph_walk] - 1)
            batch.step_context.repeats = 1
            batch.step_context.check_tp_repeats()
            return None
        if batch.slot is None:
            batch.set_slot(submodule_mgmt.lease_slot())

        cg_runner = submodule_mgmt.accelerator_graph_runner
        if cg_runner is None:
            batch.step_context.repeats = 1  # an eager step runs one iteration
            batch.step_context.check_tp_repeats()
            return None
        # Which of the walk's captures this batch belongs to. The engine can't
        # derive it — what separates two captures of one walk is the model's
        # business (bagel: guidance on/off) — and the lease is taken before the
        # step is declared, so the submodule is asked directly. It answers from
        # the same per-request facts its `declare_step` stamps on the step.
        key = capture_step.cg_key_info if capture_step is not None else submodule_mgmt.submodule.cg_key_info(
            batch.step_context.graph_walk, batch.per_request_info_wrapped,
            per_request_input_metadata=batch.per_request_input_metadata,
        )
        num_tokens = (
            None if batch.inputs is None
            else sum(inp.input_seq_len for inp in batch.inputs)
        )
        lease = None
        pools = submodule_mgmt.lora_pools
        if pools and not any(pool.runs_adapters(batch.request_ids) for pool in pools):
            # no row names an adapter: the graph captured without LoRA kernels, when
            # there is one (the pool checks the rows again at plan)
            batch.cg_key_info = LoraBaseOnly(key)
            lease = cg_runner.lease_slot(
                graph_walk=batch.step_context.graph_walk, bs=len(batch.request_ids),
                slot=batch.slot, num_tokens=num_tokens, cg_key_info=batch.cg_key_info,
            )
        if lease is None:
            batch.cg_key_info = key
            lease = cg_runner.lease_slot(
                graph_walk=batch.step_context.graph_walk, bs=len(batch.request_ids),
                slot=batch.slot, num_tokens=num_tokens, cg_key_info=key,
            )
        planner = submodule_mgmt.resident
        ctx = batch.step_context
        ctx.resident = planner if lease is not None and planner is not None and planner.covers(lease.bucket) else None
        ctx.resident_plan = None
        if lease is not None:
            batch.lease_slot(lease)
            ctx.force_prefill = cg_runner.is_packed(lease)
        else:
            # an eager step runs one iteration: only a replay repeats
            ctx.repeats = 1
            ctx.check_tp_repeats()
        return lease

    def reserve_iteration_run(self, batch: ExecutingBatch) -> bool:
        """Prove a queued successor uses resident addressing and actual progress.

        A capture miss can execute eagerly only after its predecessor settles;
        its host-built addresses cannot be repaired by the device row mask.
        """
        lease = self.reserve_replay_slot(batch)
        if lease is None or batch.step_context.resident is None:
            return False
        runner = self._submodules[batch.node_name].accelerator_graph_runner
        run = runner.slot_for(lease).run
        return run is not None and run.window is not None

    def can_pre_plan(self, node_name: str) -> bool:
        """Whether pre-planning this node could stage anything at all.

        A node with no pre-planning resource has nothing to stage: its step
        declaration is empty, so `pre_plan_for_batch` is guaranteed to bail.
        Asked BEFORE the plan thread waits on the previous batch's commit, so
        a stateless node does not pay that wait to reach a foregone conclusion.
        """
        mgmt = self._submodules.get(node_name)
        if mgmt is None or mgmt.accelerator_graph_runner is None:
            return False
        return any(r.supports_preplan for r in mgmt.resources.values())

    def speculative_input_seq_len(
        self, node_name: str, graph_walk: str,
        fwd_info: CurrentForwardPassInfo | None, prev_seq_len: int,
    ) -> int | None:
        return self._submodules[node_name].submodule.speculative_input_seq_len(
            graph_walk, fwd_info, prev_seq_len,
        )

    def pre_plan_for_batch(self, batch: ExecutingBatch) -> bool:
        """Admit and plan the pre-planning resources a step ahead.

        Worth doing only under a lease: the reserved slot is not the one the
        in-flight replay reads, so this can write its plan buffers on the plan
        stream while that replay runs. ``exec`` then re-drives the full sweep,
        where each pre-planned resource promotes what was staged here, and the
        replay waits on the event recorded here.

        An unprepared batch declares its step over the capture config's inputs
        for the leased bucket, which is why the lease had to come from a
        batched capture. That step is not cached: ``exec`` re-declares it over
        the real inputs.

        TODO: pre-planning a packed capture (and chunked prefill) needs the
        real token counts here, i.e. ``prepare_inputs`` run ahead of the
        forward. That needs every submodule's ``prepare_inputs`` to be
        async-safe — no ``.item()`` on a tensor the in-flight step is still
        producing — which some are not (e.g. qwen3_tts).
        ``ExecutingBatch.outputs_ready`` is the hook for it.

        Returns False when nothing was planned ahead, in which case ``exec``
        plans inline.
        """
        submodule_mgmt = self._submodules[batch.node_name]
        cg_runner = submodule_mgmt.accelerator_graph_runner
        lease = batch.step_context.slot_lease
        if cg_runner is None or lease is None:
            return False
        with self._stage_lock or nullcontext():
            return self._pre_plan_staged(batch, submodule_mgmt, cg_runner, lease)

    def _refuse_regions_beside_chain(self, node_name: str, regions: Collection[str]) -> None:
        """A step beside the chain takes a slot past the node's rotation, which
        a piecewise region wraps back onto a slot of the chain's: the two would
        replay one graph on one set of buffers."""
        beside = sorted(
            walk for (node, walk), lane in self._walk_lanes.items()
            if node == node_name and lane and self._node_lanes.get(node) != lane
        )
        if regions and beside:
            raise ValueError(
                f"{node_name!r} walks {beside} are placed on a lane beside the node's walks on the chain, "
                f"but the node has piecewise regions {sorted(regions)}, which a lane has no slot of"
            )

    def _beside_chain(self, node_name: str, graph_walk: str) -> bool:
        """Whether a step of this walk runs off the chain on what the chain
        uses too: resources the chain plans, whose admit, plan and commit then
        must not overlap the plan thread staging the chain's next step on them
        (``_stage_lock``), or a node whose captures are the chain's. Its
        arrival drops that stage, as any step the stage wasn't for does."""
        lane = self._walk_lanes.get((node_name, graph_walk), 0)
        if not lane:
            return False
        held = self._runner.lane_held
        return self._node_lanes.get(node_name) != lane or any(
            key not in held for key in self._submodules[node_name].resources
        )

    def _pre_plan_staged(self, batch: ExecutingBatch, submodule_mgmt, cg_runner, lease) -> bool:
        if self._enable_nvtx:
            range_push(f"engine.pre_plan.bs{len(batch.request_ids)}")
        try:
            return self._pre_plan_for_batch(batch, submodule_mgmt, cg_runner, lease)
        finally:
            if self._enable_nvtx:
                range_pop()

    def settle_tp_repeats(self, batch: ExecutingBatch) -> None:
        """Freeze a parallel leader's count before broadcasting it. This only
        declares from capture templates; input preparation stays after N.
        The worker has already waited for N's host commit, outside our lock."""
        ctx = batch.step_context
        if ctx.repeats > 1:
            with self._stage_lock or nullcontext():
                lease = self.reserve_replay_slot(batch)
                if lease is not None:
                    mgmt = self._submodules[batch.node_name]
                    step = self._declare_preplan_step(batch, mgmt, mgmt.accelerator_graph_runner, lease)
                    if step is not None:
                        step.set_ctx(ctx)
                        self._runner._settle_repeats(step)
        ctx.tp_repeats = ctx.repeats

    def _declare_preplan_step(self, batch, submodule_mgmt, cg_runner, lease):
        # Declare-only, so the unprepared case takes the bucket's shared
        # template rows rather than cloning a fresh set per pre-plan.
        inputs = (
            cg_runner.declare_inputs_for(lease)
            if batch.inputs is None
            else cg_runner.pad_inputs(lease, batch.inputs, uniform=batch.uniform_inputs)
        )
        batch.step_context.set_padded_rids(
            cg_runner.step_ids(lease, batch.request_ids)
        )
        step = engine_step(
            submodule_mgmt.submodule,
            graph_walk=batch.step_context.graph_walk,
            request_ids=batch.step_context.padded_request_ids,
            inputs=inputs,
            slot_lease=lease,
            per_request_info=batch.per_request_info,
        )
        return step

    def _pre_plan_for_batch(
        self, batch: ExecutingBatch, submodule_mgmt: SubmoduleManagement,
        cg_runner, lease: SlotLease,
    ) -> bool:

        laps = PhaseLaps("engine.pre_plan.")
        t0 = phase_start()
        step = self._declare_preplan_step(batch, submodule_mgmt, cg_runner, lease)
        phase_end("engine.pre_declare", t0)
        if step is None:
            return False
        if batch.inputs is not None or (
            not submodule_mgmt.piecewise_runners
            and submodule_mgmt.submodule.step_is_input_free(
                batch.step_context.graph_walk,
            )
        ):
            # exec drives this step as declared here, and its admit skips
            # the resources pre_admit covered
            batch.step = step

        batch.step_context.is_preplan = True
        step.set_ctx(batch.step_context)
        try:
            t0 = phase_start()
            admit_outcome = self._runner.pre_admit(step)
            phase_end("engine.pre_admit", t0)
            if not admit_outcome.ok:
                # exec re-drives the step and reports the failure from there
                self.reset_pre_plan_for_batch(batch)
                return False
            stream = cg_runner.plan_stream()
            ctx = batch.step_context
            event = torch.Event(device=self._device)
            try:
                with stream if stream is not None else nullcontext():
                    self._order_after_pre_plans(batch.node_name)
                    # A raise here takes what a raise from `pre_plan` below takes:
                    # `Worker._preplan_spec` resets the pre-plan, whose reservation
                    # stays with its streams for exec's admit to find.
                    if ctx.resident is not None:
                        ctx.plan_phase = "res.pre_plan"
                        t0 = phase_start()
                        ctx.resident.plan(step, ctx)
                        phase_end("res.pre_plan.resident", t0)
                    self._runner.pre_plan(step)
            finally:
                # recorded even when the plan raised: its copies are in flight
                event.record(stream)
                self._pre_plan_done[batch.node_name] = event
            laps.skip()
            batch.preplan_event = event
            batch.preplanned_rids = tuple(batch.request_ids)
            laps.lap("event")
        finally:
            batch.step_context.is_preplan = False
        return True

    def _plan_peers_of(self, node_to_resources: Mapping[str, Collection[str]]) -> dict[str, tuple[str, ...]]:
        """Each node, and the nodes that share a resource with it, directly or
        through what their resources depend on: the pre-plans whose copies a
        step of the node can skip or overwrite (``_order_after_pre_plans``)."""
        held = {
            node: {id(res) for res in reachable_resources(self._resources, keys).values()}
            for node, keys in node_to_resources.items()
        }
        return {
            node: tuple(sorted(other for other, theirs in held.items() if other == node or keys & theirs))
            for node, keys in held.items()
        }

    def _share_slot_rotations(self) -> None:
        """Resolve components before any runner allocates slot-keyed buffers."""
        remaining = set(self._submodules)
        while remaining:
            component = set()
            todo = [min(remaining)]
            while todo:
                node = todo.pop()
                if node not in component:
                    component.add(node)
                    todo.extend(set(self._plan_peers[node]) - component)
            remaining.difference_update(component)
            resources = reachable_resources(
                self._resources,
                {key for node in component for key in self._submodules[node].resources},
            )
            rotation = SlotRotation(resources.values())
            for node in component:
                self._submodules[node].slot_rotation = rotation

    def _order_after_pre_plans(self, node_name: str) -> None:
        """Order the current stream after every copy a pre-plan that
        ``node_name``'s step could depend on has issued.

        Resources skip a copy into a slot's buffer that already holds the
        bytes (``H2DMirror``, ``_SlotBuffer.zero_prefix``), recorded when the
        copy is issued. A pre-plan's copy runs on its node's plan stream, and
        the pre-plan may be dropped before its batch runs, so nothing else
        orders a later step that skips its own copy, or overwrites the buffer,
        after it. Every buffer a plan copies into belongs to one resource, or
        to the node's resident tables, so the step waits on its own node's
        plan stream and on those of the nodes sharing a resource with it
        (``_plan_peers``), the last writers of what it touches; a step on a
        lane, whose resources no other node holds and none pre-plans, waits on
        nothing."""
        done = self._pre_plan_done
        events = tuple(done[n] for n in self._plan_peers[node_name] if n in done)
        if events:
            stream = torch.accelerator.current_stream(self._device)
            for event in events:
                stream.wait_event(event)

    def preplan_is_stale(self, batch: ExecutingBatch) -> bool:
        """Whether the staged plan no longer describes this batch."""
        return (
            batch.preplanned_rids is not None
            and batch.preplanned_rids != tuple(batch.request_ids)
        )

    def reset_pre_plan_for_batch(self, batch: ExecutingBatch | None = None) -> None:
        """Drop planned-ahead state, e.g. when its batch never dispatched.

        The resources hold the plan itself. The batch keeps whatever
        ``prepare_inputs`` produced — only the plan is redone — but the leased
        slot's padding rows are returned so their pages aren't left attributed
        to a step that never ran.
        """
        if batch is not None:
            if batch.preplanned_rids is not None and self._enable_nvtx:
                # a discarded pre-plan falls back to inline plan; mark it so the
                # cost is visible next to the nvtx ranges
                mark(f"engine.preplan_discarded.{batch.node_name}")
            batch.preplan_event = None
            batch.preplanned_rids = None
            # The step was declared over the rows the pre-plan saw: template
            # rows for an unprepared batch, or a request set that has since
            # lost rows to prepare_inputs (preplan_is_stale). Either way exec
            # declares and admits again over the batch as it stands now; a
            # kept step would go to admit unchecked.
            batch.step = None
            lease = batch.step_context.slot_lease
            cg_runner = self._submodules[batch.node_name].accelerator_graph_runner
            if lease is not None and cg_runner is not None:
                cg_runner.release(lease, len(batch.request_ids))
        # through the runner, so its record of the staged step goes too
        self._runner.clear_preplan()

    # ── Eviction ────────────────────────────────────────────────────────
    #
    # Which requests exist, how recently each ran, and when to reclaim are the
    # worker's; the resources only move their own state. A node's resources
    # are reclaimed together, so a request is either resident or not.

    def evictable(self, node_name: str, resources: "Collection[str] | None" = None) -> bool:
        """Whether offloading a request can free anything on ``node_name``, or, given
        ``resources``, anything of those (a LoRA pool, say, never evicts)."""
        owned = self._submodules[node_name].resources
        return any(
            resource.supports_eviction
            for key, resource in owned.items()
            if resources is None or key in resources
        )

    def is_offloaded(self, node_name: str, request_id: str) -> bool:
        return any(
            resource.is_offloaded(request_id)
            for resource in self._submodules[node_name].resources.values()
        )

    def offload_request(self, node_name: str, request_id: int) -> int:
        """Move the request off-device across this node's resources.

        Returns what was reclaimed, in whatever each resource counts (pages,
        today); 0 means nothing moved and the caller should pick another
        victim.
        """
        if node_name in self._tp_follower_nodes:
            # Rank 0 decides what this instance evicts, and its decision arrives
            # as a resident-set delta. TP followers can deadlock if they are not
            # given a single source of truth for eviction.
            return 0
        freed = sum(
            resource.offload(request_id)
            for resource in self._submodules[node_name].resources.values()
            if resource.supports_eviction
        )
        if freed > 0:
            self._release_resident_slot(node_name, request_id)
            if node_name in self._resident_delta:
                self._resident_delta[node_name].add_offloaded(
                    self._graph_runtime.get_rid_string(request_id)
                )
        return freed

    def _release_resident_slot(self, node_name: str, request_id: int) -> None:
        """An offloaded request needs no slot: a reload is written whole anyway."""
        planner = self._submodules[node_name].resident
        if planner is not None:
            planner.release(request_id)

    def reload_request(self, node_name: str, request_id: int) -> bool:
        """Bring it back. False when any resource can't fit it yet, in which
        case the request stays offloaded and the caller retries later."""
        if node_name in self._tp_follower_nodes:
            # Same rule the other way round: a reload spends the pages an
            # eviction freed, so it is just as much rank 0's call. Not ready here
            # until the delta says so.
            return False
        offloaded = [
            resource
            for resource in self._submodules[node_name].resources.values()
            if resource.supports_eviction and resource.is_offloaded(request_id)
        ]
        # check every resource fits first, so a refusal doesn't reload and re-offload whatever did
        if not all(resource.can_reload(request_id) for resource in offloaded):
            return False
        reloaded: list[Resource] = []
        for resource in offloaded:
            if not resource.reload(request_id):
                # undone, not kept: every page move rank 0 makes is journalled or undone, and a partial one can't be
                for done in reloaded:
                    done.offload(request_id)
                return False
            reloaded.append(resource)
        # Logged at the same level as the eviction in
        # ``Worker._try_offload_cold_request`` and as a follower's replay, so
        # the two ranks' resident-set moves can be lined up side by side.
        # A reload only happens after an eviction, so this is not chatty.
        logger.info(
            "Reloaded request %s on %s",
            self._graph_runtime.get_rid_string(request_id), node_name,
        )
        if node_name in self._resident_delta:
            self._resident_delta[node_name].add_reloaded(
                self._graph_runtime.get_rid_string(request_id)
            )
        return True

    def _classify_tp_roles(
        self, node_names: set[str], parallel_groups: WorkerParallelGroups,
    ) -> None:
        """Sort this rank's nodes into the ones it leads, follows, or owns alone.

        ``world_size > 1``, not ``> 0``: every node has a world size of at least
        one, so ``> 0`` makes an un-sharded node a "leader" and leaves it
        journalling resident-set moves that nothing ever drains —
        ``take_resident_delta`` is only reached through the TP broadcast, which
        skips un-sharded nodes, so the journal grows for the life of the process.
        """
        sharded = {
            node for node in node_names
            if parallel_groups.get_instance_world_size_for_node(node) > 1
        }
        self._tp_leader_nodes = {
            node for node in sharded
            if parallel_groups.get_instance_rank_for_node(node) == 0
        }
        self._tp_follower_nodes = sharded - self._tp_leader_nodes
        for node in self._tp_leader_nodes:
            self._resident_delta[node] = OffloadDelta.new()

    def take_resident_delta(
        self, node_name: str,
    ) -> OffloadDelta:
        """The moves on this node since the last call, ready for the wire.

        Verbatim. Every move in it is one this rank executed, so a rank in the
        same state can always apply it — which is what makes a refused replay
        mean the states differ, and nothing else.
        """
        if node_name not in self._resident_delta:
            return OffloadDelta.new()
        return self._resident_delta[node_name].take()

    def apply_resident_delta(
        self, node_name: str, delta: OffloadDelta,
    ) -> bool:
        """Match the leader's resident set on this node. Mutates OffloadDelta
        by popping off the steps that have already landed. False means not all
        has landed and the caller must retry before trying the next step.
        """

        resources = self._submodules[node_name].resources.values()
        for _ in range(len(delta)):
            wire_rid, is_offload = delta.peek_left()
            # Wire strings, not handles: a handle is minted per worker, so rank
            # 0's is meaningless here. None means this rank has no live request
            # by that name -- it removed it already, on the other side of the
            # step rank 0 recorded the move. Its pages are gone, so the move is
            # moot and the entry is spent; logged because reaching this for a
            # rid the rank has NOT YET admitted would silently lose a move and
            # diverge the resident set for good.
            rid = self._graph_runtime.get_rid_handle(wire_rid)
            if rid is None:
                logger.debug(
                    "Replay: no live request named %s on %s; move is moot",
                    wire_rid, node_name,
                )
                delta.pop_left()
                continue
            if is_offload:
                if self.is_offloaded(node_name, rid):
                    logger.debug(
                        "Replay: %s already off %s", wire_rid, node_name,
                    )
                elif self.reclaimable(node_name, rid) <= 0:
                    # Nothing resident to move, so this rank already matches the
                    # leader. Distinct from the refusal below, and the difference
                    # matters: ``offload`` answers 0 for both, and treating
                    # "nothing to free" as retryable wedges the caller forever.
                    logger.debug(
                        "Replay: %s holds nothing on %s to offload",
                        wire_rid, node_name,
                    )
                else:
                    freed = sum(
                        resource.offload(rid)
                        for resource in resources
                        if resource.supports_eviction
                    )
                    if freed <= 0:
                        # Refused for now: a step still holds these pages and
                        # will release them when it commits.
                        logger.warning(
                            "Replay of rank 0's offload of %s on %s refused "
                            "(a step still holds its pages); %d moves still "
                            "owed. This rank cannot run rank 0's next step "
                            "until it catches up.",
                            wire_rid, node_name, len(delta),
                        )
                        return False
                    self._release_resident_slot(node_name, rid)
                    logger.info(
                        "Replayed rank 0's offload of request %s from %s "
                        "(%d reclaimed)", wire_rid, node_name, freed,
                    )
            elif not self.is_offloaded(node_name, rid):
                logger.debug(
                    "Replay: %s already resident on %s", wire_rid, node_name,
                )
            elif not all(
                resource.reload(rid)
                for resource in resources
                if resource.supports_eviction and resource.is_offloaded(rid)
            ):
                    # No room yet. Silent until now, and this is the shape of a
                    # follower that stops mid-delta and never finishes: rank 0
                    # then waits on a collective this rank will never join.
                logger.warning(
                    "Replay of rank 0's reload of %s on %s does not fit; "
                    "%d moves still owed. This rank cannot run rank 0's "
                    "next step until it catches up.",
                    wire_rid, node_name, len(delta),
                )
                return False
            else:
                logger.info(
                    "Replayed rank 0's reload of request %s on %s",
                    wire_rid, node_name,
                )
            delta.pop_left()
        return True

    def reclaimable(self, node_name: str, request_id: str, affected_resources: set[str] | None=None) -> int:
        """What this node's resources could reclaim from the request. 0 means
        offloading it would free nothing, so it is not a candidate."""
        submod_mgmt = self._submodules[node_name]
        if affected_resources is None:
            affected_resources = submod_mgmt.resources.keys()
        return sum(
            self._resources[res].reclaimable(request_id)
            for res in affected_resources if res in submod_mgmt.resources \
                and self._resources[res].supports_eviction
        )

    def offload_priority(
        self, node_name: str, request_id: str, resource_label: str,
    ) -> float:
        """How much one named resource wants this request gone.

        Only the PRIORITY eviction policy asks; it names the resource to
        consult, since "most worth reclaiming" means something different per
        resource.
        """
        resource = self._submodules[node_name].resources.get(resource_label)
        return 0.0 if resource is None else resource.get_offload_priority(request_id)

    def add_request(
        self, request_id: str,
        overrides: Mapping[str, ResourceReqConfig] | None = None,
        request_info: CurrentForwardPassInfo | None = None,
    ) -> None:
        """Open the request on every resource, then tell each submodule. All or
        nothing: a raise leaves the request on none of them."""
        self._runner.ingest_request(request_id, overrides)
        if request_info is None:
            return
        try:
            for submodule_mgmt in self._submodules.values():
                submodule_mgmt.submodule.on_request_added(request_id, request_info)
        except Exception:
            self.remove_request(request_id)
            raise

    @property
    def publisher_nodes(self) -> Mapping[str, frozenset[str]]:
        """Each publishing resource's declaring nodes, across the deployment."""
        return self._runner.publisher_nodes

    def skip_publish(self, request_id, keys) -> None:
        """Never publish ``keys`` for this request (see ``StepRunner.skip_publish``)."""
        self._runner.skip_publish(request_id, keys)

    def never_publish(self, keys) -> None:
        """Never publish ``keys`` for any request (see ``StepRunner.never_publish``)."""
        self._runner.never_publish(keys)

    def remove_request(self, request_id: str) -> set[str]:
        """Tear the request down on every resource. Returns the resources it
        held pages of that the teardown hands back, which `reclaimable` counts:
        a KV cache whose pages reach the free list or are left to the prefix
        index alone. An offloaded request, or one that never ran, frees none."""
        self._loopback_steps.pop(request_id, None)
        freed = {
            key for key, resource in self._resources.items()
            if resource.reclaimable(request_id)
        }
        self._runner.remove_request(request_id)
        self._chunking.forget_request(request_id)
        for node_name, submodule_mgmt in self._submodules.items():
            with self.on_lane(self._node_lanes.get(node_name, 0)):
                submodule_mgmt.submodule.cleanup_request(request_id)
            if submodule_mgmt.resident is not None:
                submodule_mgmt.resident.release(request_id)
        return freed

    def shutdown(self):
        for resource in self._resources.values():
            resource.cleanup()
