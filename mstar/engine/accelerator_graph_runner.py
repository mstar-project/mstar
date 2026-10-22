import ctypes
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, is_dataclass, replace
from functools import partial
from typing import Any, Callable, Collection, Iterator, Mapping, NamedTuple

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import JointGroups
from mstar.engine.accelerator_graph_backend import (
    AcceleratorGraphPool,
    CapturedGraph,
)
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    AcceleratorGraphConfigType,
    PiecewiseAcceleratorGraphConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewiseConfigType,
)
from mstar.engine.resources import BucketKey, CGSlotSpec, Resource, SlotLease, StepContext, StepRunner
from mstar.engine.resources.lora.config import LoraBaseOnly
from mstar.engine.resources.step import PIECEWISE_WALK
from mstar.model.submodule_base import (
    BatchedModelOutput,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
    declared_step,
)
from mstar.platforms import current_platform
from mstar.platforms.compile import with_inductor_overrides
from mstar.utils import profiler
from mstar.utils.device import DeviceCapability

logger = logging.getLogger(__name__)


DEFAULT_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


# torch.compile mode for a captured forward whose config's ``compile_mode`` is None
_DEFAULT_COMPILE_MODE = "max-autotune-no-cudagraphs"


def compile_kwargs(config_mode: str | None) -> dict[str, Any]:
    """``torch.compile``'s arguments for a capture config whose ``compile_mode``
    is ``config_mode``, which the config checked when it was built."""
    kwargs: dict[str, Any] = {"fullgraph": False, "dynamic": False}
    mode = _DEFAULT_COMPILE_MODE if config_mode is None else config_mode
    # "default" is torch.compile's own: Inductor fusion, cuBLAS GEMMs, no autotuning
    if mode != "default":
        kwargs["mode"] = mode
    return kwargs


def blocked_capture_resources(step, resources: Mapping[str, Resource]) -> list[str]:
    """Return resource keys that prevent capture of the declared step."""
    if step is None:
        return []
    return sorted(key for key in step.steps if not resources[key].can_capture())


def autocast_scope(dtype: torch.dtype | None, device_type: str):
    """A forward's autocast scope; ``None`` (``disable_autocast``) runs the
    submodule in its own dtype and shuts out any ambient autocast."""
    return torch.amp.autocast(
        device_type, enabled=dtype is not None, dtype=dtype or torch.bfloat16
    )


def agree_across_ranks(
    comm_group: JointGroups | None, values: list[int], device: torch.device,
    reduce: Callable[..., torch.Tensor],
) -> list[int]:
    """Reduce each of ``values`` over every rank of the joint group with
    ``reduce``. `torch.amin` ANDs flags and `torch.amax` takes the largest.

    Capture failure is per-rank: a rank that keeps a graph another rank
    dropped replays it while the other runs eager, which hangs as soon as the
    captured region holds a collective. Callers pass an order derived from the
    configs, which are identical on every rank.
    """
    groups = [] if comm_group is None else [
        comm_group.tp_group, comm_group.sp_group
    ]
    if not values or all(group.world_size == 1 for group in groups):
        return values
    gathered = torch.tensor(values, dtype=torch.int64, device=device)
    for group in groups:
        if group.world_size > 1:
            # all_gather is rank-major, so dim 0 runs over the ranks
            gathered = reduce(group.all_gather(gathered, dim=0).reshape(
                group.world_size, -1
            ), dim=0)
    return gathered.tolist()


#: Handles for padding rows. Real handles are >= 0 (they index the worker's rid
#: table) and -1 is CurrentForwardPassInfo's unstamped default, so counting down
#: from -2 collides with neither. A padding row IS its handle everywhere the
#: engine keys by rid; the name it was minted from is kept only so logs and
#: CurrentForwardPassInfo.request_id stay readable. Memoized per name because a
#: submodule keys its per-request state by the handle: one shared value would
#: collapse every padding row onto the same state.
_DUMMY_HANDLE_BASE = -2
_dummy_handles: dict[str, int] = {}
_dummy_names: dict[int, str] = {}


def dummy_rid_handle(name: str) -> int:
    handle = _dummy_handles.get(name)
    if handle is None:
        handle = _dummy_handles[name] = _DUMMY_HANDLE_BASE - len(_dummy_handles)
        _dummy_names[handle] = name
    return handle


def dummy_rid_name(handle: int) -> str:
    """The name a padding-row handle was minted from. Debug/logging only."""
    return _dummy_names.get(handle, f"__cg_unknown_{handle}__")


def dummy_metadata(
    rids: list[int], graph_walk: str,
) -> dict[int, CurrentForwardPassInfo]:
    """Stand-in request info for padding rows, which have no real request."""
    return {
        rid: CurrentForwardPassInfo(
            request_id=dummy_rid_name(rid),
            rid_handle=rid,
            graph_walk=graph_walk,
            fwd_index=0,
            random_seed=0,
            max_tokens=1,
        ) for rid in rids
    }


def capture_into_graph(run, pool: AcceleratorGraphPool, device, autocast_dtype, keep_graph: bool = False):
    """Capture ``run`` into a new CUDA graph allocated from ``pool``.

    Returns the graph and what ``run`` returned. A capture that fails part
    way leaves two things behind that would sink every later capture on the
    same pool: the caching allocator keeps routing this thread's allocations
    into the pool (the next capture_begin fails with "already recording to
    mempool_id"), and torch.cuda.graph never switches the thread back off its
    capture stream. Undo both before re-raising, so the buckets after a
    failed one still capture and the thread keeps running on the stream it
    came in on.

    A torch.compile'd forward that would recompile inside the capture is
    failed up front with the guard that broke: dynamo saves the CUDA RNG state
    before it compiles, which CUDA refuses during capture, so such a capture
    cannot succeed and would otherwise die with an unrelated error.
    """
    backend = DeviceCapability.of(device).graph_backend
    if backend is None:
        raise ValueError(f"Accelerator graphs are unsupported on device {device!r}")
    prev_stream = backend.current_stream()
    graph = backend.create_graph(keep_graph)
    # benchmark off, so cuDNN searches in the warm-up and not in here: a conv
    # whose cached plan can't get its workspace would search again, timing
    # candidates and emptying the allocator's cache, which fails the capture
    benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        with torch.compiler.set_stance("fail_on_recompile"), \
                autocast_scope(autocast_dtype, backend.device_type):
            with backend.capture(graph, pool=pool):
                output = run()
    except BaseException:
        pool.on_failure(graph)
        backend.recover_failed_capture(pool, prev_stream)
        raise
    finally:
        torch.backends.cudnn.benchmark = benchmark
    backend.synchronize()
    pool.on_success()
    return graph, output


def capture_with_static_outputs(
    run, warm_outputs, pool: AcceleratorGraphPool, device, autocast_dtype, keep_graph: bool = False,
):
    """Capture ``run`` so its outputs land in buffers allocated OUTSIDE ``pool``.

    ``warm_outputs`` is what ``run`` returned on a warm-up call (``{name:
    Tensor}``); it fixes the shapes and dtypes. The captured graph ends with a
    device-to-device copy of each output into a buffer allocated here, before
    the capture, from the ordinary allocator.

    Why not hand out the graph's own output tensors: blocks a graph frees
    while it is being captured go back to the pool and are handed to the
    graphs captured after it, so an output that lived in the pool could sit
    exactly where an earlier-captured graph keeps its scratch and be
    overwritten by that graph's next replay. That is harmless while a pool
    belongs to one region whose graphs replay one at a time and whose
    outputs are consumed before the next replay, but not once a node's
    regions share one pool and replay in data-dependent order. Buffers outside
    the pool stay valid until this graph's own next replay, whatever else the
    node runs in between.
    """
    static_outputs = {
        name: torch.empty_like(value) for name, value in warm_outputs.items()
    }

    def captured():
        for name, value in run().items():
            static_outputs[name].copy_(value)
        return static_outputs

    graph, _ = capture_into_graph(captured, pool, device, autocast_dtype, keep_graph)
    return graph, static_outputs


def fail_if_graphs_required(missing: list[str]) -> None:
    """MSTAR_REQUIRE_CUDA_GRAPHS=1 turns a dropped bucket into a startup
    failure, for deployments that would rather not come up than serve the
    eager path at 10-20x the latency."""
    required = os.environ.get("MSTAR_REQUIRE_ACCELERATOR_GRAPHS", "0")
    if not missing or required != "1":
        return
    raise RuntimeError(
        "accelerator graph capture failed for " + ", ".join(missing)
        + " and accelerator graphs are required"
    )


def _is_packed(config: AcceleratorGraphConfig) -> bool:
    return config.get_config_type() == AcceleratorGraphConfigType.FLASH_INFER_PACKED


class CaptureCost(NamedTuple):
    """What one bucket costs on the device: its warm-up, and a capture thrown away, its own or its walk's largest."""
    # bytes the eager forward allocated at its peak, above what was live
    peak: int
    # bytes one slot's graph holds at most, in its pool and out; None if it can't be captured
    graph: int | None
    # what of ``graph`` no later graph in the pool reuses: outputs, and a region's inputs
    kept: int
    # graphs the bucket captures, one per slot
    slots: int
    # bytes each slot's graph holds outside the allocator: its executable memory
    outside: int = 0
    # the walk's measurement, whose copies hold memory the graphs reuse
    executable: "ExecutableMemory | None" = None


class GraphMemory(NamedTuple):
    """What a captured graph holds: in its pool, and outside the allocator."""
    pool: int
    outside: int


class ExecutableMemory:
    """The executable memory a walk's graphs take per node, outside any pool,
    measured on copies of its first throwaway graph: executables instantiated
    from it through libcuda. The cost per node changes with a graph's size,
    even for the same kernels: a chain of adds takes 3.1 KiB a node at 300
    nodes and 2.4 at 2222. A walk's graphs are about the same size, so the
    first graph's cost per node holds for all of them.

    The copies are held until the runner captures. The driver keeps a destroyed
    graph's executable memory to reuse, and maps new memory beside it, so after
    any destroy, free memory drops by less than a copy takes.
    """

    def __init__(self):
        self.per_node: float | None = None
        # a lower bound on the bytes the copies hold
        self.held = 0
        self._copies: list[ctypes.c_void_p] = []

    def measure(self, graph, nodes: int) -> None:
        """Set ``per_node`` and ``held`` from copies of ``graph``, held until `release`.

        The count starts after the first copy that takes free memory, since
        copies until then fill pieces the driver mapped earlier. It adds one
        2 MiB piece, since the first counted copy may start partway into a
        piece. Free memory is the device's, so another process allocating or
        freeing moves it: a count is kept only if free memory never rose and no
        copy took more than two pieces above the mean. If no count is kept,
        ``per_node`` is the largest rate a count gave. None if no copy could be
        counted.
        """
        cuda = ctypes.CDLL("libcuda.so.1")
        largest: float | None = None
        drops: list[int] | None = None
        free = torch.cuda.mem_get_info()[0]
        tries = 0
        first = len(self._copies)
        while tries < _MEASURE_TRIES and len(self._copies) < _MEASURE_COPIES:
            copy = ctypes.c_void_p()
            status = cuda.cuGraphInstantiateWithFlags(
                ctypes.byref(copy), ctypes.c_void_p(graph.raw_cuda_graph()), ctypes.c_ulonglong(0),
            )
            if status != 0:
                logger.warning("Stopped measuring graph executable memory: CUresult %d", status)
                break
            self._copies.append(copy)
            drop = free - torch.cuda.mem_get_info()[0]
            free -= drop
            if drop < 0:
                tries += 1
                drops = None
            elif drops is None:
                if drop > 0:
                    drops = []
            else:
                drops.append(drop)
                if sum(drops) >= _MEASURE_BYTES:
                    rate = (sum(drops) + _EXEC_PIECE) / (len(drops) * nodes)
                    if max(drops) <= sum(drops) / len(drops) + 2 * _EXEC_PIECE:
                        self.per_node = rate
                        # Subtract two pieces, since the count may start partway into one and leave the last
                        # one partly unused.
                        each = (sum(drops) - 2 * _EXEC_PIECE) / len(drops)
                        self.held += int((len(self._copies) - first) * each)
                        return
                    largest = max(rate, largest or 0)
                    tries += 1
                    drops = None
        if drops:
            largest = max((sum(drops) + _EXEC_PIECE) / (len(drops) * nodes), largest or 0)
        elif drops is None and not tries and self._copies:
            # every copy fit in a piece mapped before
            largest = _EXEC_PIECE / (len(self._copies) * nodes)
        self.per_node = largest

    def release(self) -> None:
        # copies are made only on NVIDIA GPUs, since others may have no libcuda
        if self._copies:
            cuda = ctypes.CDLL("libcuda.so.1")
            for copy in self._copies:
                cuda.cuGraphExecDestroy(copy)
            self._copies.clear()


#: the driver maps executable memory in pieces of 2 MiB
_EXEC_PIECE = 2 * 2**20
#: copies are counted until they take 16 MiB, so the one piece added to a count is an eighth of it at most
_MEASURE_BYTES = 16 * 2**20
#: a count spoiled by another process is tried again
_MEASURE_TRIES = 4
_MEASURE_COPIES = 512


def _tensor_bytes(value: Any) -> int:
    """Bytes of the distinct storages behind every tensor in ``value``, however nested."""
    storages: dict[int, int] = {}
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, torch.Tensor):
            storages[item.untyped_storage().data_ptr()] = item.untyped_storage().nbytes()
        elif isinstance(item, Mapping):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif is_dataclass(item):
            pending.extend(vars(item).values())
    return sum(storages.values())


def _capture_thrown_away(
    capture: Callable[..., tuple[Any, Any]], what: str, executable: ExecutableMemory,
) -> GraphMemory | None:
    """Run ``capture`` into a pool of its own and hand it all back: what the
    graph held, None if it failed.

    A capture records kernels and collectives without running them, so a rank
    short of memory fails its own capture here and no other rank waits on it.
    """
    torch.cuda.empty_cache()
    pool = DeviceCapability.of("cuda").graph_backend.graph_pool_handle()
    # Keep the graph so its nodes can be counted. It is never instantiated, so no executable is
    # destroyed before `ExecutableMemory` measures. The measurement calls libcuda, so a ROCm
    # graph's executable memory is counted as zero.
    counted = current_platform.is_cuda()
    nodes = 0
    try:
        graph, outputs = capture(pool=pool, keep_graph=counted)
        if counted:
            nodes = _graph_nodes(graph)
            if nodes and executable.per_node is None:
                executable.measure(graph, nodes)
                if executable.per_node is None:
                    # Leave the graph unsized. Counted as zero, its executable memory would go unplanned.
                    logger.warning("Could not measure the executable memory of %s", what)
                    graph = outputs = None
                else:
                    logger.info(
                        "%s: graph executable memory, %.0f bytes a node, %.0f MiB held for the walk's graphs",
                        what, executable.per_node, executable.held / 2**20,
                    )
    except RuntimeError:
        logger.warning("Could not capture %s to size it", what, exc_info=True)
        graph = outputs = None
    taken = None if graph is None else sum(
        segment["total_size"] for segment in torch.cuda.memory_snapshot(pool.handle)
    )
    del graph, outputs
    # A capture that never ended, or any capture before torch 2.12, keeps its stream's cuBLAS
    # workspace in the pool. No real graph uses it yet.
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    return None if taken is None else GraphMemory(taken, round(nodes * (executable.per_node or 0)))


def _graph_nodes(graph) -> int:
    """``graph`` must be captured with ``keep_graph=True``, so the driver still holds it to count."""
    count = ctypes.c_size_t(0)
    status = ctypes.CDLL("libcuda.so.1").cuGraphGetNodes(
        ctypes.c_void_p(graph.raw_cuda_graph()), None, ctypes.byref(count),
    )
    assert status == 0, f"cuGraphGetNodes failed with CUresult {status}"
    return count.value


def _size_in_walk(
    captured: list[CaptureCost], cost: CaptureCost, kept_in_pool: bool,
    capture: Callable[..., tuple[Any, Any]], what: str, executable: ExecutableMemory,
) -> tuple[CaptureCost, bool]:
    """Return ``cost`` with its graph's size, and whether that size came from
    another bucket's capture: the bucket is captured, thrown away, unless two
    of its walk already were and they show it fits in the larger's scratch.

    The lender is the bucket with the largest eager peak of those ``captured``
    in the walk so far. A bucket that peaks no higher takes the lender's
    scratch and adds its own kept bytes: a walk runs one forward at every
    shape, and a smaller shape allocates no more at any step, so its graph
    fits in the scratch a larger one leaves free. A smaller bucket runs the same
    kernels, so its executable memory is taken as the lender's.

    That holds only while no captured bucket took more scratch than the
    lender. Where one did, a lender tells nothing of this walk, and every
    bucket of it is captured: Chatterbox's CFG decode takes 24 MiB at 32 rows
    and 56 MiB at 16.
    """
    def scratch(bucket: CaptureCost) -> int:
        return -(-(bucket.graph - bucket.kept) // _SEGMENT)

    cost = cost._replace(executable=executable)
    lender = max(captured, key=lambda bucket: bucket.peak, default=None)
    if (
        len(captured) > 1 and cost.peak <= lender.peak
        and all(scratch(bucket) <= scratch(lender) for bucket in captured)
    ):
        return cost._replace(graph=lender.graph - lender.kept + cost.kept, outside=lender.outside), True
    memory = _capture_thrown_away(capture, what, executable)
    if memory is not None:
        cost = cost._replace(
            graph=memory.pool if kept_in_pool else memory.pool + cost.kept, outside=memory.outside,
        )
        captured.append(cost)
    return cost, False


#: the caching allocator's smallest segment: what a graph's kept tensors are counted in
_SEGMENT = 2 * 2**20


class CapturePlan(NamedTuple):
    """Which buckets each graph pool on a device captures, chosen before any is."""
    # pool -> the buckets it captures
    buckets: dict[str, set]
    # pool -> bytes it is predicted to hold
    predicted: dict[str, int]
    # pool -> bytes its graphs hold outside the allocator
    outside: dict[str, int]
    # bytes the largest eager step on the device takes
    floor: int


def plan_captures(
    pools: Mapping[str, Mapping[Any, CaptureCost]], free: int, reusable: int | None = None,
) -> CapturePlan:
    """The cheapest buckets across every pool on a device, added while the
    largest eager step still fits in ``free`` beside them.

    A pool's graphs, captured largest first, reuse its memory: it holds what
    every graph keeps, plus the most any one graph needs beyond that. So a
    bucket adds its slots' kept bytes, and whatever it needs past the pool's
    most so far. That is a model; the boot log compares it with what capture took.
    A bucket also adds its slots' executable memory, less the memory its
    graphs reuse from the walk's copies. A walk's copies pay for its planned
    graphs' executable memory once, across all its buckets. ``reusable`` bounds
    the reuse summed over all buckets, None for no bound.
    """
    floor = max((cost.peak for costs in pools.values() for cost in costs.values()), default=0)
    room = free - floor
    plan = CapturePlan({pool: set() for pool in pools}, dict.fromkeys(pools, 0), dict.fromkeys(pools, 0), floor)
    scratch = dict.fromkeys(pools, 0)
    # walk's measurement -> what its copies held that no planned graph reuses yet
    unused = {
        cost.executable: cost.executable.held
        for costs in pools.values() for cost in costs.values() if cost.executable is not None
    }
    bound = float("inf") if reusable is None else reusable

    def reused(cost: CaptureCost) -> int:
        return min(unused.get(cost.executable, 0), bound, cost.slots * cost.outside)

    def added(pool: str, cost: CaptureCost) -> int:
        # whole segments: the costliest graph's scratch may have no room left for a small tensor
        kept = -(-cost.kept // _SEGMENT) * _SEGMENT
        return cost.slots * kept + max(cost.graph - cost.kept - scratch[pool], 0)

    def on_device(candidate) -> int:
        pool, _, _, cost = candidate
        return added(pool, cost) + cost.slots * cost.outside - reused(cost)

    left = [
        (pool, index, bucket, cost) for pool, costs in pools.items()
        for index, (bucket, cost) in enumerate(costs.items()) if cost.graph is not None
    ]
    while left:
        # a tie goes to the pool named first, which leads with its node, then to capture order
        best = min(left, key=lambda candidate: (on_device(candidate), *candidate[:2]))
        pool, _, bucket, cost = best
        if on_device(best) > room:
            break
        room -= on_device(best)
        used = reused(cost)
        if used:
            unused[cost.executable] -= used
            bound -= used
        plan.predicted[pool] += added(pool, cost)
        plan.outside[pool] += cost.slots * cost.outside
        scratch[pool] = max(scratch[pool], cost.graph - cost.kept)
        plan.buckets[pool].add(bucket)
        left.remove(best)
    return plan


class DummyRowPool:
    """The padding rows captured replays pad onto.

    Ingested once and kept for the runner's lifetime: a re-ingest would hand
    the resources a fresh stream and orphan the pages the last capture left
    resident, and keeping them resident is what lets a step's padding tail
    allocate nothing.
    """

    def __init__(
        self, prefix: str, step_runner: StepRunner,
        resources: Mapping[str, Resource],
    ):
        self._prefix = prefix
        self._step_runner = step_runner
        self._resources = resources
        self._held: dict[str, list[int]] = {}

    def handles(self, key: str, bs: int) -> list[int]:
        """``bs`` padding-row handles for ``key``. Negative and stable per
        (prefix, key, index), so they never collide with a real request's
        handle nor with another pool's rows."""
        return [
            dummy_rid_handle(f"__cg_{self._prefix}_{key}_{i}__")
            for i in range(bs)
        ]

    def ensure(self, key: str, bs: int) -> list[int]:
        """``bs`` rows for ``key``, ingesting any this pool hasn't opened yet."""
        held = self._held.setdefault(key, [])
        rids = self.handles(key, bs)
        for rid in rids[len(held):]:
            self._step_runner.ingest_capture_row(rid)
            held.append(rid)
        return rids

    def reset(self, rids: list[int], free: bool=False) -> None:
        for rid in rids:
            for resource in self._resources.values():
                resource.reset_request(rid, free=free)

    def release_all(self) -> None:
        """Hand back every padding row's storage once capture is done: a
        replay pads with zero-length rows, so it is capture-time residue."""
        for rids in self._held.values():
            self.reset(rids, free=True)


@dataclass
class AcceleratorGraphSlot:
    """One captured graph + the buffers its replay reads and writes.

    Double-buffer: a bucket holds one of these per slot. Replay alternates
    between slots so plan(N+1) on the inactive slot's resources can run
    concurrently with replay(N) on the active slot — so a node with nothing
    to pre-plan keeps a single slot.
    """
    graph: CapturedGraph
    # preprocess output as captured; tensor entries are the static buffers
    static_inputs: dict[str, Any]
    static_input_keys: tuple[str, ...]
    static_outputs: BatchedModelOutput
    # padding rows address these; their streams stay resident between steps
    dummy_rids: list[int]
    dummy_metadata: dict[int, CurrentForwardPassInfo]
    config_idx: int


@dataclass
class AcceleratorGraphBucket:
    """The captured slots for one (walk, cg_key_info, bs, num_tokens)."""
    config: AcceleratorGraphConfig
    config_idx: int
    slots: list[AcceleratorGraphSlot] = field(default_factory=list)


class WarmedSpec(NamedTuple):
    """A spec prepared and run eagerly: what its capture reads."""
    run: Callable[[], Any]
    static_inputs: dict[str, Any]
    static_input_keys: tuple[str, ...]
    dummy_rids: list[int]
    dummy_metadata: dict[int, CurrentForwardPassInfo]
    # bytes the last warm-up forward allocated at its peak, above what was live
    peak: int
    # bytes of what it returned
    returned: int


class AcceleratorGraphRunner:
    CAPTURE_BATCH_SIZES = DEFAULT_CAPTURE_BATCH_SIZES
    NUM_WARMUP = 2

    def __init__(
        self,
        submodule_name: str,
        submodule: NodeSubmodule,
        resources: Mapping[str, Resource],
        step_runner: StepRunner,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
        joint_comm_group: JointGroups,
        num_slots: int,
        enable_nvtx: bool=False,
        device_capability: DeviceCapability | None = None,
    ):
        self._submodule_name = submodule_name
        self._submodule = submodule
        self._comm_group = joint_comm_group

        self._capture_configs: list[AcceleratorGraphConfig] = (
            submodule.get_accelerator_graph_configs(
                device, joint_comm_group.world_size
            )
        )
        self._resources = resources
        self._step_runner = step_runner
        self._device = device
        self._device_capability = device_capability or DeviceCapability.of(device or "cpu")
        self._graph_backend = self._device_capability.graph_backend
        self._autocast_dtype = autocast_dtype
        self._enable_nvtx = enable_nvtx

        # How many slots to double-buffer; the caller decides (pre-plan or a
        # force_double_buffer resource), since without either the slots would
        # hold identical graphs and double the capture time and memory.
        self._num_slots = num_slots

        self._buckets: dict[BucketKey, AcceleratorGraphBucket] = {}

        self._memory_pool: AcceleratorGraphPool | None = None
        # buckets that failed to capture (here or on a peer rank) and run eagerly
        self.dropped_buckets: list[BucketKey] = []
        self._blocked_buckets: list[BucketKey] = []
        # set by prepare_for_capture; capture reuses it rather than re-deriving
        self._prepared_slot_specs: list[CGSlotSpec] | None = None
        # bucket -> what it costs, from size_captures
        self._capture_costs: dict[BucketKey, CaptureCost] = {}
        # the buckets of them sized by a lender's capture, not their own
        self._lent: set[BucketKey] = set()
        # walk -> its graphs' executable memory
        self._executables: dict[tuple, ExecutableMemory] = {}

        # (config_idx, tensor_key) → max-bucket static buffer. Lazily populated
        # by _intern_static_buffer on the first capture. Smaller-bucket captures
        # slice the leading dim of the same buffer
        self._shared_static_buffers: dict[tuple[int, str], torch.Tensor] = {}

        # (config_idx, tensor_key) → the dim that carries the (bucket-varying)
        # seq length in that tensor's original layout. _intern_static_buffer
        # brings this dim to the front for storage (so smaller buckets reslice
        # along dim 0) and inverts the move on return. Written once, by the
        # first (largest) capture, and read by every later one — see there.
        self._static_buffer_seq_dims: dict[tuple[int, str], int] = {}

        # padding rows, keyed "<config_idx>_slot<slot>". Sized by the largest
        # bucket (captures run largest-first); smaller buckets take a prefix.
        self._dummy_rows = DummyRowPool(
            prefix=submodule_name, step_runner=step_runner, resources=resources,
        )

        # Sum of bytes that WOULD have been allocated by per-capture clones
        # (one full tensor per call). For logging purposes.
        self._capture_clone_bytes_naive = 0

        # config_idx -> its compiled forward, shared by every bucket of that
        # config so shapes captured twice (once per slot) compile once
        self._compiled_forwards: dict[int, Any] = {}

        # Plan-overlap stream. Lazily created the first time pre_plan
        # is called from Worker.plan_executor.
        self._plan_stream: Any | None = None

        # Per-bucket template rows for declare-only calls; see
        # `declare_inputs_for`.
        self._declare_inputs: dict[BucketKey, list[NodeInputs]] = {}

        self.max_bs = max(
            (max(self._batch_sizes(config)) for config in self._capture_configs),
            default=1,
        )

    @property
    def any_graphs(self):
        return bool(self._buckets)

    def _batch_sizes(self, config: AcceleratorGraphConfig) -> list[int]:
        return config.capture_batch_sizes or self.CAPTURE_BATCH_SIZES

    # ── Capture ─────────────────────────────────────────────────────────

    def _get_slot_specs(self) -> list[CGSlotSpec]:
        specs = []
        for config_idx, config in enumerate(self._capture_configs):
            keys = [config.additional_key_info]
            if getattr(self._submodule, "lora_keys", ()):
                # a node serving adapters also captures each bucket for steps with no
                # adapter row, recording no LoRA kernel (see `LoraBaseOnly`); the two
                # share the config's compiled forward, static buffers and padding rows
                keys.append(LoraBaseOnly(config.additional_key_info))
            for bs in self._batch_sizes(config):
                for num_tokens in config.get_total_tokens(bs):
                    for key in keys:
                        bucket = BucketKey(
                            graph_walk=config.capture_graph_walk,
                            cg_key_info=key,
                            bs=bs,
                            num_tokens=num_tokens,
                        )
                        for slot in range(self._num_slots):
                            specs.append(CGSlotSpec(
                                bucket=bucket,
                                slot=slot,
                                config=config,
                                config_idx=config_idx,
                            ))
        return specs

    def _get_addtl_slot_specs(self, spec: CGSlotSpec) -> list[CGSlotSpec]:
        """The same capture, keyed under the config's other replay walks."""
        return [
            replace(spec, bucket=replace(spec.bucket, graph_walk=walk))
            for walk in spec.config.replay_graph_walks
            if walk != spec.config.capture_graph_walk
        ]

    def prepare_for_capture(self) -> list[CGSlotSpec]:
        """Claim the static buffers this runner's captures will read.

        Split from the capture so every runner claims first: nodes share
        resources, and a build driven by a later node would move buffers an
        earlier node's graphs already baked in.
        """
        if self._prepared_slot_specs is None:
            slot_specs = self._get_slot_specs()
            # reverse sort based on batch size, then total tokens (largest first)
            slot_specs.sort(key=lambda s: (s.bs, s.num_tokens), reverse=True)
            blocked_configs = set()
            checked_configs = set()
            for spec in slot_specs:
                if spec.config_idx in checked_configs:
                    continue
                checked_configs.add(spec.config_idx)
                rids = self._dummy_rows.handles(f"{spec.config_idx}_slot0", spec.bs)
                step = declared_step(
                    self._submodule,
                    graph_walk=spec.bucket.graph_walk,
                    request_ids=rids,
                    inputs=spec.config.get_node_inputs(spec.bs, spec.num_tokens),
                    slot_lease=SlotLease(slot=0, bucket=spec.bucket),
                    per_request_info=dummy_metadata(rids, spec.bucket.graph_walk),
                )
                blocked = blocked_capture_resources(step, self._resources)
                if blocked:
                    logger.error(
                        "Dropping accelerator graph config %s for %s: "
                        "resources cannot capture: %s",
                        spec.config_idx, self._submodule_name, blocked,
                    )
                    blocked_configs.add(spec.config_idx)
            self._blocked_buckets = list(dict.fromkeys(
                spec.bucket for spec in slot_specs if spec.config_idx in blocked_configs
            ))
            self.dropped_buckets = list(self._blocked_buckets)
            slot_specs = [
                spec for spec in slot_specs if spec.config_idx not in blocked_configs
            ]
            max_seq_len = max((
                spec.bucket.num_tokens for spec in slot_specs
            ), default=1)
            if slot_specs:
                self._step_runner.build_accelerator_graph_buffers(
                    slot_specs, max_bs=self.max_bs, max_seq_len=max_seq_len,
                    node_name=self._submodule_name,
                )
            self._prepared_slot_specs = slot_specs
        return self._prepared_slot_specs

    def size_captures(self) -> dict[BucketKey, CaptureCost]:
        """Warm every spec up, and capture each walk's largest buckets once,
        thrown away, before any runner captures for real.

        Every bucket is warmed rather than the first in capture order: that
        order leads with batch size, so it can open on a decode bucket while a
        one-row prefill of thousands of tokens peaks far higher. And every slot: a
        slot's first plan builds state of its own (FlashInfer's graph wrappers
        and their workspace, 750 MiB for the Talker's second slot), which has
        to be spent before free memory is read.
        """
        # walk -> the buckets of it captured so far
        captured: dict[tuple, list[CaptureCost]] = {}
        for spec in self.prepare_for_capture():
            # the warm-up may hold collectives; every rank sizes these specs, in
            # config order, as it does the captures
            self._comm_group.tp_group.barrier()
            self._comm_group.sp_group.barrier()
            free = torch.cuda.mem_get_info(self._device)[0]
            try:
                with self._warmed(spec, self.NUM_WARMUP) as warmed:
                    if spec.slot == 0:
                        walk = (spec.bucket.graph_walk, spec.bucket.cg_key_info)
                        self._capture_costs[spec.bucket], lent = _size_in_walk(
                            captured.setdefault(walk, []),
                            CaptureCost(warmed.peak, None, warmed.returned, self._num_slots), True,
                            partial(
                                capture_into_graph, warmed.run, device=self._device,
                                autocast_dtype=self._autocast_dtype,
                            ),
                            f"{self._submodule_name} {spec}", self._executables.setdefault(walk, ExecutableMemory()),
                        )
                        if lent:
                            self._lent.add(spec.bucket)
            except Exception as e:
                # Fail the boot. A skipped step would leave its peak out of the plan's floor.
                # A step that cannot run here cannot run once graphs hold memory either.
                raise RuntimeError(
                    f"AcceleratorGraphRunner[{self._submodule_name}]: could not run {spec} eagerly to size "
                    f"its capture, with {free / 2**20:.0f} MiB free before it: {e}"
                ) from e
        return self._capture_costs

    def size_lent_captures(self, wanted: bool) -> dict[BucketKey, CaptureCost]:
        """Capture every bucket a lender sized, thrown away, if ``wanted`` here
        or on another rank: a plan that leaves graphs out needs what each
        takes, not what its walk's largest does.

        Every rank asks, so the ranks of a group size again together or not at
        all, and plan on the same kind of cost. With no forward and no barrier:
        sizing ran the forwards, and a capture runs no collective.
        """
        if not agree_across_ranks(self._comm_group, [int(wanted)], self._device, torch.amax)[0]:
            return self._capture_costs
        for spec in self.prepare_for_capture():
            if spec.slot != 0 or spec.bucket not in self._lent:
                continue
            self._lent.discard(spec.bucket)
            cost = self._capture_costs[spec.bucket]._replace(graph=None)
            try:
                with self._warmed(spec, forwards=0) as warmed:
                    memory = _capture_thrown_away(
                        partial(
                            capture_into_graph, warmed.run, device=self._device,
                            autocast_dtype=self._autocast_dtype,
                        ),
                        f"{self._submodule_name} {spec}",
                        self._executables[(spec.bucket.graph_walk, spec.bucket.cg_key_info)],
                    )
                if memory is not None:
                    cost = cost._replace(graph=memory.pool, outside=memory.outside)
            except Exception:
                logger.warning(
                    "AcceleratorGraphRunner[%s]: could not prepare %s to size its capture",
                    self._submodule_name, spec, exc_info=True,
                )
            self._capture_costs[spec.bucket] = cost
        return self._capture_costs

    def warmup_and_capture(self, planned: Collection[BucketKey] | None):
        """Capture the ``planned`` buckets, of every config and batch size.
        With no plan, every bucket, each warmed up first as sizing would have."""
        # the plan has read free memory, and the graphs captured here reuse what the copies held
        for executable in self._executables.values():
            executable.release()
        if self._graph_backend is None or not self._graph_backend.is_available():
            logger.warning(
                "%s is not available, skipping graph capture for %s",
                getattr(self._device, "type", "accelerator").upper(),
                self._submodule_name,
            )
            return

        self._memory_pool = self._graph_backend.graph_pool_handle()
        mem_before = self._graph_backend.memory_allocated()

        slot_specs = self.prepare_for_capture()
        sized = planned is not None
        # what every rank planned, as though captured in full (see `_buckets_captured_everywhere`)
        planned = self._buckets_captured_everywhere(slot_specs, {
            bucket: range(self._num_slots)
            for bucket in (planned if sized else {spec.bucket for spec in slot_specs})
        })
        # costliest first, so the graphs after it fit in what it leaves free; by
        # the most each costs on any rank, so every rank warms up in one order
        buckets = list(dict.fromkeys(spec.bucket for spec in slot_specs))
        costs = dict(zip(buckets, agree_across_ranks(self._comm_group, [
            self._capture_costs[bucket].graph if sized and bucket in planned else 0 for bucket in buckets
        ], self._device, torch.amax), strict=True))

        # bucket -> slot index -> (spec, captured slot). Registration is
        # deferred to a whole bucket at a time: its slots are double buffers of
        # one shape, so a bucket holding only some of them would silently hand
        # pre-plan and replay the same buffers.
        captured: dict[BucketKey, dict[int, tuple[CGSlotSpec, AcceleratorGraphSlot]]] = {}
        for spec in sorted(slot_specs, key=lambda spec: -costs[spec.bucket]):
            if spec.bucket not in planned:
                continue
            # every rank barriers once per spec, pass or fail — a rank that
            # stopped early here would leave the others waiting
            self._comm_group.tp_group.barrier()
            self._comm_group.sp_group.barrier()

            try:
                slot = self._capture_one(spec, forwards=0 if sized else self.NUM_WARMUP)
            except Exception:
                logger.error(
                    "Failed to capture accelerator graph for %s: %s",
                    self._submodule_name, spec, exc_info=True
                )
                continue

            captured.setdefault(spec.bucket, {})[spec.slot] = (spec, slot)
            if  len(captured[spec.bucket]) == self._num_slots:
                logger.info(
                    "Captured accelerator graph for %s: %s (%d slots)",
                    self._submodule_name, spec.bucket, self._num_slots
                )

        agreed = self._buckets_captured_everywhere(slot_specs, captured)
        for bucket_key, slots in captured.items():
            if bucket_key not in agreed:
                logger.error(
                    "Dropping accelerator graph bucket %s for %s: captured %d of %d "
                    "slots here, or fewer on another rank",
                    bucket_key, self._submodule_name, len(slots), self._num_slots,
                )
                continue
            # by slot index, so a bucket's list position is its slot index
            for slot_idx in sorted(slots):
                spec, slot = slots[slot_idx]
                for key_spec in [spec, *self._get_addtl_slot_specs(spec)]:
                    self._register_slot(key_spec, slot)
            config = self._buckets[bucket_key].config
            if self._num_slots > 1 and config.get_config_type() == AcceleratorGraphConfigType.BASIC_BATCHED:
                # pre-seed preplan inputs; cached per bucket, so once is enough.
                # A packed bucket's rows would never be read: a batch with no inputs
                # yet can only lease a batched one (see `select_batched_bucket`).
                self.declare_inputs_for(SlotLease(slot=0, bucket=bucket_key))

        self._dummy_rows.release_all()
        self._report_dropped(
            [spec.bucket for spec in slot_specs if spec.slot == 0], agreed
        )

        mem_after = self._graph_backend.memory_allocated()
        self._log_memory(mem_before, mem_after)

    def _report_dropped(self, wanted: list, kept) -> None:
        """Say loudly which buckets run eagerly. A dropped bucket is a silent
        10-20x latency cliff otherwise, and one failure used to take every
        later bucket with it."""
        wanted = list(dict.fromkeys([*self._blocked_buckets, *wanted]))
        self.dropped_buckets = [key for key in wanted if key not in kept]
        if self.dropped_buckets:
            logger.error(
                "AcceleratorGraphRunner[%s]: captured %d of %d buckets; these run "
                "eagerly: %s",
                self._submodule_name, len(wanted) - len(self.dropped_buckets),
                len(wanted), self.dropped_buckets,
            )

    def _buckets_captured_everywhere(
        self,
        slot_specs: list[CGSlotSpec],
        captured: Mapping[BucketKey, Mapping[int, Any]],
    ) -> set[BucketKey]:
        """The buckets every rank captured in full.

        Capture failure is a per-rank event, so a rank that keeps a bucket
        another rank dropped will lease and replay it while the other runs
        eager — and a captured region holding a collective then hangs. The
        candidate list comes from the configs, which are identical on every
        rank, so agreeing is one flag per bucket reduced in that order.
        """
        order: list[BucketKey] = []
        seen: set[BucketKey] = set()
        for spec in slot_specs:
            if spec.bucket not in seen:
                seen.add(spec.bucket)
                order.append(spec.bucket)

        agreed = agree_across_ranks(
            self._comm_group,
            [len(captured.get(key, ())) == self._num_slots for key in order],
            self._device, torch.amin,
        )
        return {key for key, ok in zip(order, agreed, strict=True) if ok}

    def _register_slot(self, spec: CGSlotSpec, slot: AcceleratorGraphSlot) -> None:
        bucket = self._buckets.get(spec.bucket)
        if bucket is None:
            bucket = self._buckets[spec.bucket] = AcceleratorGraphBucket(
                config=spec.config, config_idx=spec.config_idx,
            )
        # the caller registers a bucket's slots in index order, and only once
        # all of them captured, so append keeps list position == slot index
        bucket.slots.append(slot)

    def _log_memory(self, mem_before: int, mem_after: int):
        shared_bytes = sum(
            t.numel() * t.element_size() for t in self._shared_static_buffers.values()
        )
        # Report both: the deterministic synthetic counter (clean before/after
        # for the buffer-reuse change in isolation) and the actual GPU delta
        # (covers FlashInfer wrappers + dummy KV state too, but is noisier).
        logger.info(
            "AcceleratorGraphRunner[%s]: warmup_and_capture done. "
            "shared_static_buffers: %d entries, %.2f MB resident "
            "(would have been %.2f MB with per-capture clones — saved %.2f MB). "
            "Total accelerator alloc delta during warmup: %.2f MB.",
            self._submodule_name,
            len(self._shared_static_buffers),
            shared_bytes / (1024 ** 2),
            self._capture_clone_bytes_naive / (1024 ** 2),
            (self._capture_clone_bytes_naive - shared_bytes) / (1024 ** 2),
            (mem_after - mem_before) / (1024 ** 2),
        )

    def _capture_one(self, spec: CGSlotSpec, forwards: int) -> AcceleratorGraphSlot:
        # no forward after sizing, which already ran this spec's
        with self._warmed(spec, forwards) as warmed:
            # what preparing it left cached in the allocator is out of the
            # capture pool's reach
            torch.accelerator.empty_cache()
            graph, output = capture_into_graph(
                warmed.run, self._memory_pool, self._device, self._autocast_dtype,
            )
            return self._build_slot_from_capture(
                output=BatchedModelOutput.coerce(output),
                graph=graph,
                static_inputs=warmed.static_inputs,
                static_input_keys=warmed.static_input_keys,
                dummy_rids=warmed.dummy_rids,
                dummy_metadata=warmed.dummy_metadata,
                config_idx=spec.config_idx,
            )

    @contextmanager
    def _warmed(self, spec: CGSlotSpec, forwards: int) -> Iterator[WarmedSpec]:
        """Prepare ``spec`` and run its forward eagerly ``forwards`` times, as
        its first capture needs."""
        walk = spec.bucket.graph_walk
        config = spec.config
        dummy_rids = self._dummy_rows.ensure(
            f"{spec.config_idx}_slot{spec.slot}", spec.bs
        )
        dummy_inputs = config.get_node_inputs(spec.bs, spec.num_tokens)
        engine_inputs = self._dummy_engine_inputs(dummy_rids, walk)

        lease = SlotLease(slot=spec.slot, bucket=spec.bucket)
        step = declared_step(
            self._submodule,
            graph_walk=walk, request_ids=dummy_rids, inputs=dummy_inputs,
            slot_lease=lease, per_request_info=engine_inputs.per_request_info,
        )
        if step is not None:
            step = replace(step, _ctx=StepContext(
                request_ids=tuple(dummy_rids),
                graph_walk=walk,
                slot=spec.slot,
                capture=True,
                slot_lease=lease,
                force_prefill=_is_packed(config),
            ))

        engine_inputs.step = step

        def prepare() -> dict[str, Any]:
            if step is not None:
                outcome = self._step_runner.admit(step)
                if not outcome.ok:
                    raise RuntimeError(
                        f"capture admit failed for {spec.bucket}: {outcome.reason}"
                    )
                self._step_runner.plan(step)
            with self._step_runner.compute_scope(step):
                return self._submodule.preprocess(
                    graph_walk=walk,
                    engine_inputs=engine_inputs,
                    inputs=dummy_inputs,
                )

        try:
            static_inputs = prepare()
            # Intern before capture so every bucket of this config records the
            # same GPU addresses; replay writes real values into them.
            for key, value in list(static_inputs.items()):
                if isinstance(value, torch.Tensor):
                    seq_dim_override = (
                        config.input_seq_dims.get(key) if config.input_seq_dims else None
                    )
                    static_inputs[key] = self._intern_static_buffer(
                        spec.config_idx, key, value,
                        seq_len=spec.num_tokens, seq_dim_override=seq_dim_override,
                    )
            static_input_keys = tuple(
                key for key, value in static_inputs.items()
                if isinstance(value, torch.Tensor)
            )

            forward = self._forward_for(spec)

            def run_forward():
                with self._step_runner.compute_scope(step):
                    return forward(
                        graph_walk=walk,
                        engine_inputs=engine_inputs,
                        **static_inputs,
                    )

            self._graph_backend.set_device()
            self._graph_backend.synchronize()
            peak = returned = 0
            for _ in range(forwards):
                # the last forward's peak, not the first's: that one may compile and autotune
                live = torch.accelerator.memory_allocated(self._device)
                torch.accelerator.reset_peak_memory_stats(self._device)
                with autocast_scope(self._autocast_dtype, self._device.type):
                    returned = _tensor_bytes(run_forward())
                peak = torch.accelerator.max_memory_allocated(self._device) - live
                # back to a clean stream state so the re-plan below (and the
                # capture after it) sees the same shapes the first prepare did
                self._dummy_rows.reset(dummy_rids)
                prepare()
            self._graph_backend.synchronize()

            yield WarmedSpec(
                run=run_forward,
                static_inputs=static_inputs,
                static_input_keys=static_input_keys,
                dummy_rids=dummy_rids,
                dummy_metadata=engine_inputs.per_request_info,
                peak=peak,
                returned=returned,
            )
        finally:
            # Capture ran the dummy rows as real requests; free their storage
            # now, since replay's padding rows address the sink page and slot.
            self._dummy_rows.reset(dummy_rids, free=True)

    def _forward_for(self, spec: CGSlotSpec):
        """The callable this bucket captures, compiled once per config.

        One wrapper per config rather than per capture: ``dynamic=False`` keys
        compiled code by shape, so every bucket of a config shares a cache and
        the second slot's identical shapes are free. A wrapper per capture
        would recompile — and re-autotune — all of them.

        The platform's inductor overrides wrap that shared callable per bucket,
        so each bucket compiles on its first call under the overrides for its
        own token count.
        """
        forward = getattr(self._submodule, spec.config.capture_forward_method)
        if not spec.config.compile:
            return forward
        if spec.config_idx not in self._compiled_forwards:
            self._compiled_forwards[spec.config_idx] = torch.compile(
                forward, **compile_kwargs(spec.config.compile_mode),
            )
        return with_inductor_overrides(
            self._compiled_forwards[spec.config_idx], spec.num_tokens,
        )

    def _dummy_engine_inputs(
        self, dummy_rids: list[int], graph_walk: str,
    ) -> ModelInputsFromEngine:
        return ModelInputsFromEngine(
            request_ids=list(dummy_rids),
            per_request_info=dummy_metadata(dummy_rids, graph_walk),
            resources=dict(self._resources),
            captured=True,
        )

    @staticmethod
    def _seq_dim(value: torch.Tensor, seq_len: int) -> int:
        """Index of the first dim whose size matches ``seq_len``, else 0.

        Used to bring the (bucket-varying) seq dim to the front for shared-buffer
        interning: most inputs are seq-leading (returns 0), but mrope-style ids
        carry seq in a later dim (e.g. ``[3, seq]`` → 1)."""
        for dim, size in enumerate(value.shape):
            if size == seq_len:
                return dim
        return 0

    def _intern_static_buffer(
        self, config_idx: int, key: str, value: torch.Tensor,
        seq_len: int | None = None, seq_dim_override: int | None = None,
    ) -> torch.Tensor:
        """Return a slice view into the shared buffer for (config_idx, key).

        The buffer is allocated at the first (largest, since captures are sorted
        largest-first) bucket's shape; smaller buckets reslice its leading dim.
        If ``seq_len`` is given, the seq dim is moved to the front for storage and
        back on return, so the captured forward sees the original layout.
        ``seq_dim_override``, from the config's ``input_seq_dims``, is used
        instead of `_seq_dim`'s size-based guess when a caller already knows
        which dim varies with the bucket (`_seq_dim` can pick the wrong dim
        when an unrelated axis happens to match ``seq_len``).
        """
        buf_key = (config_idx, key)
        if seq_len is None:
            seq_dim = 0
        elif buf_key in self._static_buffer_seq_dims:
            # settled by the first (largest) capture. `_seq_dim` matches on
            # size, so a smaller bucket's seq_len can collide with another axis
            # — and the buffer is shared, so its layout cannot vary anyway
            seq_dim = self._static_buffer_seq_dims[buf_key]
        else:
            seq_dim = (
                seq_dim_override if seq_dim_override is not None
                else self._seq_dim(value, seq_len)
            )
            self._static_buffer_seq_dims[buf_key] = seq_dim
        stored = value.movedim(seq_dim, 0) if seq_dim != 0 else value
        shared = self._shared_static_buffers.get(buf_key)
        if shared is None:
            shared = torch.empty(stored.shape, dtype=stored.dtype, device=stored.device)
            self._shared_static_buffers[buf_key] = shared
        self._capture_clone_bytes_naive += stored.numel() * stored.element_size()
        leading = stored.shape[0]
        if leading > shared.shape[0] or stored.shape[1:] != shared.shape[1:]:
            raise RuntimeError(
                f"_intern_static_buffer: key={key!r} (config_idx={config_idx}) needs "
                f"{tuple(stored.shape)}, shared buffer is {tuple(shared.shape)}; "
                "captures must be largest-first with matching trailing dims"
            )
        sliced = shared[:leading]
        sliced.copy_(stored)
        return sliced.movedim(0, seq_dim) if seq_dim != 0 else sliced

    def _build_slot_from_capture(
        self, output, graph, static_inputs, static_input_keys,
        dummy_rids, dummy_metadata, config_idx,
    ) -> AcceleratorGraphSlot:
        """Wrap one slot's capture artifacts into a CudaGraphSlot."""
        return AcceleratorGraphSlot(
            graph=graph,
            static_inputs=static_inputs,
            static_input_keys=static_input_keys,
            static_outputs=output,
            dummy_rids=list(dummy_rids),
            dummy_metadata=dict(dummy_metadata),
            config_idx=config_idx,
        )

    # ── Replay ──────────────────────────────────────────────────────────

    def select_bucket(
        self, graph_walk: str, bs: int, num_tokens: int,
        cg_key_info: Any | None = None,
    ) -> BucketKey | None:
        """Tightest captured bucket that fits this batch, or None for eager.

        A walk may have several captures (e.g. one per image resolution, each a
        fixed shape with its own token count), so every matching bucket is
        considered rather than the first config declared. ``num_tokens`` counts
        the batch's input tokens; a bucket is keyed by the tokens its plan
        carries, so the request is scaled by that config's multiplier.
        """
        best: BucketKey | None = None
        for key, bucket in self._buckets.items():
            if key.graph_walk != graph_walk or key.cg_key_info != cg_key_info:
                continue
            needed = num_tokens * bucket.config.total_tokens_multiplier
            if key.bs < bs or key.num_tokens < needed or not bucket.slots:
                continue
            if best is None or (key.num_tokens, key.bs) < (best.num_tokens, best.bs):
                best = key
        return best

    def select_batched_bucket(
        self, graph_walk: str, bs: int, cg_key_info: Any | None = None,
    ) -> BucketKey | None:
        """A bucket for a batch whose inputs aren't built yet.

        Only a batched capture can answer this: its token count is a property
        of the config (``bs`` rows of a fixed per-request length), so the
        bucket follows from the batch size alone. A packed capture's token
        count is a property of the *requests*, so it has to wait for
        ``prepare_inputs`` and go through ``select_bucket``.

        This is what lets pre-plan run before inputs exist. Lifting the
        restriction means making ``prepare_inputs`` safe to run ahead of the
        forward — see ``Engine.pre_plan_for_batch``.
        """
        best: BucketKey | None = None
        for key, bucket in self._buckets.items():
            if key.graph_walk != graph_walk or key.cg_key_info != cg_key_info:
                continue
            if key.bs < bs or not bucket.slots:
                continue
            if bucket.config.get_config_type() != AcceleratorGraphConfigType.BASIC_BATCHED:
                continue
            # the config's own token count for this padded batch; a bucket
            # whose num_tokens says otherwise belongs to a different capture
            if key.num_tokens not in bucket.config.get_total_tokens(key.bs):
                continue
            if best is None or (key.bs, key.num_tokens) < (best.bs, best.num_tokens):
                best = key
        return best

    def captures_walk(self, graph_walk: str) -> bool:
        """Whether any capture replays this walk."""
        return any(
            graph_walk in config.replay_graph_walks
            for config in self._capture_configs
        )

    def max_batch_size_for(self, graph_walk: str) -> int | None:
        """Largest batch this walk was captured for, or None for no cap.

        A config that opts out of capping captures only an acceleration subset
        of batch sizes — larger batches run eager rather than being split — so
        it does not constrain the batch. With every config for the walk opting
        out, nothing here caps it.
        """
        capped = [
            max(self._batch_sizes(config)) for config in self._capture_configs
            if graph_walk in config.replay_graph_walks
            and config.caps_eager_batch_size
        ]
        return max(capped) if capped else None

    def can_run(
        self, graph_walk: str, bs: int, num_tokens: int,
        cg_key_info: Any | None = None,
    ) -> bool:
        return self.select_bucket(
            graph_walk, bs, num_tokens, cg_key_info
        ) is not None

    def is_packed(self, lease: SlotLease) -> bool:
        """Whether ``lease`` replays a packed capture."""
        return _is_packed(self._buckets[lease.bucket].config)

    def lease_slot(
        self, graph_walk: str, bs: int,
        slot: int,
        num_tokens: int | None = None,
        cg_key_info: Any | None = None,
    ) -> SlotLease | None:
        """Pick the bucket and double-buffer slot an upcoming step replays on.

        ``num_tokens=None`` means the batch's inputs aren't built yet, so the
        bucket comes from the batched-capture search instead.

        ``slot=None`` advances the bucket's counter so the next lease lands on
        the other slot; a caller that already reserved one (pre-plan) passes it
        back so both submissions target the same slot.
        """
        key = (
            self.select_batched_bucket(graph_walk, bs, cg_key_info)
            if num_tokens is None
            else self.select_bucket(graph_walk, bs, num_tokens, cg_key_info)
        )
        if key is None:
            return None
        bucket = self._buckets[key]

        # Hand back the bucket the capture ran under, not the alias this walk
        # found it by. Resources key their per-slot state (attention wrappers,
        # position buffers) on `lease.bucket`, so a lease naming an aliased
        # walk would build them fresh at replay instead of reusing the ones the
        # graph recorded addresses for — see `_get_addtl_slot_specs`.
        return SlotLease(
            # a registered bucket always holds `_num_slots` slots — a partial
            # capture is dropped whole — so this agrees with `_next_slot`'s wrap
            slot=slot % len(bucket.slots),
            bucket=replace(key, graph_walk=bucket.config.capture_graph_walk),
        )

    def slot_for(self, lease: SlotLease) -> AcceleratorGraphSlot:
        return self._buckets[lease.bucket].slots[lease.slot]

    def config_for(self, lease: SlotLease) -> AcceleratorGraphConfig:
        return self._buckets[lease.bucket].config

    def declare_inputs_for(self, lease: SlotLease) -> list[NodeInputs]:
        """Template rows for a declare-only call, cached per bucket.

        A bucket's shape is fixed, so these rows are constant; building them
        fresh cost ~130us of clones per pre-plan at bs=16, plus that many D2D
        copies. Only for callers that hand the rows to ``declare_step`` and
        nothing else — the list is shared, so a caller that mutates it or
        passes it to ``preprocess`` corrupts every later step on this bucket.
        """
        cached = self._declare_inputs.get(lease.bucket)
        if cached is None:
            cached = self._declare_inputs[lease.bucket] = (
                self._buckets[lease.bucket].config.get_node_inputs(
                    lease.bucket.bs, lease.bucket.num_tokens
                )
            )
        return cached

    def pad_inputs(
        self, lease: SlotLease, inputs: list[NodeInputs],
    ) -> list[NodeInputs]:
        """Real inputs plus the rows that bring the batch to capture batch size.
        Set the length to zero so no new pages are allocated for dummy requests.
        """
        num_pad = lease.bucket.bs - len(inputs)
        if num_pad <= 0:
            return list(inputs)
        padding = self._buckets[lease.bucket].config.get_node_inputs(
            num_pad, 0
        )
        return [*inputs, *padding]

    def step_ids(self, lease: SlotLease, request_ids: list[int]) -> list[int]:
        """The padded addressing for one step: real ids first, then the slot's
        padding ids. Plans, commits, and advances address real request state by
        its own id; only the padding rows run against the slot's own state."""
        dummy_rids = self.slot_for(lease).dummy_rids
        return [*request_ids, *dummy_rids[len(request_ids):lease.bucket.bs]]

    def step_metadata(
        self, lease: SlotLease, request_ids: list[int],
        per_request_info: Mapping[int, CurrentForwardPassInfo],
    ) -> dict[int, CurrentForwardPassInfo]:
        slot = self.slot_for(lease)
        meta = {rid: per_request_info[rid] for rid in request_ids}
        for rid in slot.dummy_rids[len(request_ids):lease.bucket.bs]:
            meta[rid] = slot.dummy_metadata[rid]
        return meta

    def _stage(self, lease: SlotLease, preprocessed: dict[str, Any]) -> None:
        """Copy this step's preprocess output into the captured static buffers.

        Each key is sliced along the dim ``_intern_static_buffer`` recorded as
        its seq dim, so mrope-style ``[.., seq, ..]`` inputs land on the right
        axis. Trailing slots beyond the real length keep capture-time contents;
        the plan's indptrs keep attention off them.
        """
        slot = self.slot_for(lease)
        for key in slot.static_input_keys:
            value = preprocessed.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            static_buf = slot.static_inputs[key]
            seq_dim = self._static_buffer_seq_dims.get((slot.config_idx, key), 0)
            static_buf.narrow(seq_dim, 0, value.shape[seq_dim]).copy_(value)

    def _replay(self, lease: SlotLease) -> dict:
        slot = self.slot_for(lease)
        # The launch, not the GPU work: replay is async, so this range measures
        # enqueue cost only. GPU-side duration comes from --cuda-graph-trace.
        if self._enable_nvtx:
            profiler.range_push(f"cg.replay.slot[{lease.slot}]")
        slot.graph.replay()
        if self._enable_nvtx:
            profiler.range_pop()
        return slot.static_outputs

    def run_forward(
        self, lease: SlotLease, preprocessed: dict[str, Any],
        plan_done_event: Any | None = None,
        launch_started_event: "threading.Event | None" = None,
    ) -> dict:
        """Stage this step's inputs into the slot and replay it.

        ``plan_done_event`` is the pre-plan's event: its plan wrote this slot's
        buffers on another stream, so the replay has to wait for those writes
        before it reads them.

        ``launch_started_event`` releases the submitting thread. It is set HERE
        rather than before the forward: staging copies each preprocessed tensor
        into its static buffer and holds the GIL while doing so, so releasing
        earlier hands the main thread a GIL the GPU thread still needs. Only
        `graph.replay()` below drops it in C++.
        """
        self._stage(lease, preprocessed)
        if plan_done_event is not None:
            self._graph_backend.current_stream().wait_event(plan_done_event)
        if launch_started_event is not None:
            launch_started_event.set()
        return self._replay(lease)

    def release(self, lease: SlotLease, real_bs: int) -> None:
        """Return the padding rows to their at-rest state after a step.

        Padding rows (`is_padding_row`) run against SINK_PAGE and the recurrent
        pool's sink slot, so nothing stays resident: padding must never compete
        with real requests for storage.
        """
        dummy_rids = self.slot_for(lease).dummy_rids
        self._dummy_rows.reset(dummy_rids[real_bs:lease.bucket.bs], free=True)

    def plan_stream(self) -> Any | None:
        """Dedicated stream for pre-planning.

        Pre-plan must not submit onto the default stream: whether its memcpys
        land before or after the GPU thread records the previous batch's
        completion event is timing-dependent, and landing after delays that
        event past pre-plan's own kernels. Its own stream keeps the two
        independent; a plan-done event gates the replay that reads the buffers.
        """
        if self._graph_backend is None or not self._graph_backend.is_available():
            return None
        if self._plan_stream is None:
            self._plan_stream = self._graph_backend.new_stream()
        return self._plan_stream


# Piecewise buckets share the resources' (bucket, slot, label) buffer space
# with the full-forward captures. They never coexist on one node — a forward
# either replays whole or runs eager around captured regions — so the walk name
# (`PIECEWISE_WALK`, defined beside `StepContext`) plus the region's label is
# enough to keep the keys apart.


@dataclass
class PiecewiseGraphData:
    """One captured region, for one (bs, total_tokens) bucket."""
    graph: CapturedGraph
    static_inputs: dict[str, torch.Tensor]
    static_outputs: dict[str, torch.Tensor]
    dummy_rids: list[int]
    shape: PiecewiseCaptureShape
    bucket: BucketKey


class PiecewiseOutput:
    """Dict-like view over a captured region's output buffers.

    The runner replays into persistent buffers sized for the padded bucket;
    only the leading ``real_len`` rows are meaningful. Indexing and ``get``
    return an owned CLONE of that leading slice — safe to keep past the next
    replay. ``get_view`` returns the same slice WITHOUT copying.
    """
    __slots__ = ("_outputs", "_real_len")

    def __init__(self, outputs: dict[str, torch.Tensor], real_len: int):
        self._outputs = outputs
        self._real_len = real_len

    def __contains__(self, key: str) -> bool:
        return key in self._outputs

    def keys(self):
        return self._outputs.keys()

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._outputs[key][:self._real_len].clone()

    def get(self, key: str, default=None):
        value = self._outputs.get(key)
        if value is None:
            return default
        return value[:self._real_len].clone()

    def get_view(self, key: str, default=None):
        """The leading ``real_len`` slice WITHOUT copying.

        The result aliases the runner-owned static output buffer and is
        OVERWRITTEN by the next ``run`` of this runner. Nothing else writes
        there: the buffers live outside the graph memory pool, so replays of
        other regions sharing the node's pool leave them alone. Read it within
        the same step; use ``get`` when you need something that outlives it.
        """
        value = self._outputs.get(key)
        if value is None:
            return default
        return value[:self._real_len]


class PiecewiseGraphKey(NamedTuple):
    bs: int
    seq_len: int
    slot: int


class WarmedRegion(NamedTuple):
    """A region's shape planned and run eagerly: what its capture reads."""
    run: Callable[[], Any]
    static_inputs: dict[str, torch.Tensor]
    dummy_rids: list[int]
    warm_outputs: dict[str, torch.Tensor]
    peak: int


class PiecewiseAcceleratorGraphRunner:
    """Captures one inner callable of a submodule's forward as a CUDA graph.

    Where ``CudaGraphRunner`` replays a whole ``forward_batched`` under engine
    control, this captures a SUB-REGION — a transformer block loop, say —
    while the surrounding preamble stays eager and the submodule invokes
    ``run`` itself. The config supplies the callable, its static buffers, and
    the region's own step declaration; the runner drives that step through the
    same admit → plan → commit cycle the engine uses, so the region's
    attention plan lands in the resources' per-(bucket, slot, label) buffers
    and the captured graph reads them at fixed addresses.
    """

    CAPTURE_BATCH_SIZES = DEFAULT_CAPTURE_BATCH_SIZES
    NUM_WARMUP = 2

    def __init__(
        self,
        label: str,
        config: PiecewiseAcceleratorGraphConfig,
        resources: Mapping[str, Resource],
        step_runner: StepRunner,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
        num_slots: int,
        joint_comm_group: JointGroups | None = None,
        node_name: str | None = None,
        memory_pool: AcceleratorGraphPool | None = None,
        device_capability: DeviceCapability | None = None,
    ):
        self._label = label
        self._config = config
        self._resources = resources
        self._step_runner = step_runner
        self._device = device
        self._device_capability = device_capability or DeviceCapability.of(device or "cpu")
        self._graph_backend = self._device_capability.graph_backend
        self._autocast_dtype = autocast_dtype
        self._comm_group = joint_comm_group
        # scopes buffer allocation; `label` carries it but mangled with the region
        self._node_name = node_name
        self._prepared_shapes: list | None = None

        self._capture_batch_sizes = sorted(
            config.capture_batch_sizes or self.CAPTURE_BATCH_SIZES
        )

        self._graphs: dict[PiecewiseGraphKey, PiecewiseGraphData] = {}
        self._memory_pool = memory_pool
        # (bs, total_tokens) shapes that failed to capture and run eagerly
        self.dropped_shapes: list[tuple[int, int]] = []
        # bucket -> what it costs, from size_captures
        self._capture_costs: dict[BucketKey, CaptureCost] = {}
        # the buckets of them sized by a lender's capture, not their own
        self._lent: set[BucketKey] = set()
        # executable memory of the region's graphs, which are all one walk
        self._executable = ExecutableMemory()
        # bucket -> the shape, dtype and device of each output of its warm-up
        self._output_likes: dict[BucketKey, dict[str, tuple]] = {}
        self._dummy_rows = DummyRowPool(
            prefix=f"pw_{label}", step_runner=step_runner, resources=resources,
        )

        # how many slots the caller allows; `prepare_for_capture` narrows it to
        # what this region actually needs
        self._max_slots = num_slots
        self._num_slots = 1
        self._current_slot = 0

    def _size_slots(self) -> None:
        """A region has no pre-plan path, so a second slot buys nothing unless a
        resource its step touches stages into a reused host buffer behind an
        async H2D (see ``Resource.force_double_buffer``)."""
        if self._max_slots <= 1 or self._config.declare_step is None:
            return
        for shape in self._config.get_capture_shapes(self._capture_batch_sizes):
            # handles, not `ensure`: the declaration only labels its rows, and
            # ingesting here would run before the buffers are built
            step = self._config.declare_step(
                self._dummy_rows.handles(self._dummy_key(shape, 0), shape.bs),
                list(shape.seq_lens),
            )
            if step is not None and any(
                self._resources[key].force_double_buffer for key in step.steps
            ):
                self._num_slots = self._max_slots
                return

    def set_slot(self, slot: int):
        self._current_slot = slot % self._num_slots

    @property
    def any_graphs(self) -> bool:
        return bool(self._graphs)

    def _dummy_key(self, shape: PiecewiseCaptureShape, slot: int) -> str:
        """Padding rows are per slot, as in ``CudaGraphRunner``: a slot's plan
        and commit run against its own tail, so sharing rows between slots puts
        two graphs on one set of per-request state."""
        return f"{shape.bs}_{shape.total_tokens}_slot{slot}"

    def _bucket(self, shape: PiecewiseCaptureShape) -> BucketKey:
        return BucketKey(
            graph_walk=PIECEWISE_WALK,
            bs=shape.bs,
            num_tokens=shape.total_tokens,
            cg_key_info=self._label,
        )

    # ── Capture ─────────────────────────────────────────────────────────

    def prepare_for_capture(self) -> list:
        """Claim this region's static buffers; see ``CudaGraphRunner``."""
        if self._prepared_shapes is None:
            shapes = self._config.get_capture_shapes(self._capture_batch_sizes)
            if shapes and self._config.declare_step is not None:
                shape = shapes[0]
                step = self._config.declare_step(
                    self._dummy_rows.handles(self._dummy_key(shape, 0), shape.bs),
                    list(shape.seq_lens),
                )
                blocked = blocked_capture_resources(step, self._resources)
                if blocked:
                    logger.error(
                        "Dropping piecewise accelerator graph config %s: "
                        "resources cannot capture: %s", self._label, blocked,
                    )
                    self.dropped_shapes = [(s.bs, s.total_tokens) for s in shapes]
                    shapes = []
            if shapes:
                self._size_slots()
                self._step_runner.build_accelerator_graph_buffers(
                    [
                        CGSlotSpec(
                            bucket=self._bucket(shape), slot=slot,
                            config=self._config,
                        )
                        for shape in shapes
                        for slot in range(self._num_slots)
                    ],
                    max_bs=max(shape.bs for shape in shapes),
                    max_seq_len=max(shape.total_tokens for shape in shapes),
                    node_name=self._node_name,
                )
            self._prepared_shapes = shapes
        return self._prepared_shapes

    def size_captures(self) -> dict[BucketKey, CaptureCost]:
        """Warm every shape up and capture the largest once, thrown away; see
        `AcceleratorGraphRunner.size_captures`."""
        self._graph_backend.set_device()
        shapes = sorted(
            self.prepare_for_capture(), key=lambda s: (s.bs, s.total_tokens), reverse=True,
        )
        # the region is one walk
        captured: list[CaptureCost] = []
        for shape in shapes:
            for slot in range(self._num_slots):
                if self._comm_group is not None:
                    self._comm_group.tp_group.barrier()
                    self._comm_group.sp_group.barrier()
                free = torch.cuda.mem_get_info(self._device)[0]
                try:
                    with self._warmed(shape, slot, self.NUM_WARMUP) as warmed:
                        self._output_likes[self._bucket(shape)] = {
                            name: (value.shape, value.dtype, value.device)
                            for name, value in warmed.warm_outputs.items()
                        }
                        if slot == 0:
                            # a region's outputs and inputs live outside the pool, a set per graph
                            kept = _tensor_bytes(warmed.warm_outputs) + _tensor_bytes(warmed.static_inputs)
                            self._capture_costs[self._bucket(shape)], lent = _size_in_walk(
                                captured, CaptureCost(warmed.peak, None, kept, self._num_slots), False,
                                partial(
                                    capture_with_static_outputs, warmed.run, warmed.warm_outputs,
                                    device=self._device, autocast_dtype=self._autocast_dtype,
                                ),
                                f"{self._label} bs={shape.bs} total_tokens={shape.total_tokens}", self._executable,
                            )
                            if lent:
                                self._lent.add(self._bucket(shape))
                except Exception as e:
                    # as in `AcceleratorGraphRunner.size_captures`
                    raise RuntimeError(
                        f"PiecewiseAcceleratorGraphRunner[{self._label}]: could not run bs={shape.bs} "
                        f"total_tokens={shape.total_tokens} slot={slot} eagerly to size its capture, "
                        f"with {free / 2**20:.0f} MiB free before it: {e}"
                    ) from e
        # hand the pages back: held through the captures, they would shift every
        # admit after them on a node that shares the resource
        self._dummy_rows.release_all()
        return self._capture_costs

    def size_lent_captures(self, wanted: bool) -> dict[BucketKey, CaptureCost]:
        """Capture every shape a lender sized, thrown away, if ``wanted`` here
        or on another rank; see `AcceleratorGraphRunner.size_lent_captures`."""
        if not agree_across_ranks(self._comm_group, [int(wanted)], self._device, torch.amax)[0]:
            return self._capture_costs
        for shape in self.prepare_for_capture():
            bucket = self._bucket(shape)
            if bucket not in self._lent:
                continue
            self._lent.discard(bucket)
            cost = self._capture_costs[bucket]._replace(graph=None)
            try:
                with self._warmed(shape, 0, forwards=0) as warmed:
                    memory = _capture_thrown_away(
                        partial(
                            capture_with_static_outputs, warmed.run, self._outputs_like(bucket),
                            device=self._device, autocast_dtype=self._autocast_dtype,
                        ),
                        f"{self._label} bs={shape.bs} total_tokens={shape.total_tokens}", self._executable,
                    )
                if memory is not None:
                    cost = cost._replace(graph=memory.pool + cost.kept, outside=memory.outside)
            except Exception:
                logger.warning(
                    "PiecewiseAcceleratorGraphRunner[%s]: could not prepare bs=%d total_tokens=%d "
                    "to size its capture", self._label, shape.bs, shape.total_tokens, exc_info=True,
                )
            self._capture_costs[bucket] = cost
        self._dummy_rows.release_all()
        return self._capture_costs

    def _outputs_like(self, bucket: BucketKey) -> dict[str, torch.Tensor]:
        """Tensors shaped as the outputs of sizing's warm-up of ``bucket``, one
        element each, expanded: a capture reads their shape and dtype."""
        return {
            name: torch.empty((), dtype=dtype, device=device).expand(size)
            for name, (size, dtype, device) in self._output_likes[bucket].items()
        }

    def warmup_and_capture(self, planned: Collection[BucketKey] | None) -> None:
        self._executable.release()
        if self._graph_backend is None or not self._graph_backend.is_available():
            logger.warning(
                "%s is not available, skipping piecewise capture for %s",
                getattr(self._device, "type", "accelerator").upper(),
                self._label,
            )
            return

        self._graph_backend.set_device()
        if self._memory_pool is None:
            self._memory_pool = self._graph_backend.graph_pool_handle()

        shapes = self.prepare_for_capture()
        if not shapes:
            return

        # largest bucket first, matching the full-forward runner
        ordered = sorted(
            shapes, key=lambda s: (s.bs, s.total_tokens), reverse=True
        )
        # as in `AcceleratorGraphRunner.warmup_and_capture`: what every rank planned, costliest first
        sized = planned is not None
        planned_everywhere = agree_across_ranks(
            self._comm_group, [not sized or self._bucket(shape) in planned for shape in ordered],
            self._device, torch.amin,
        )
        costs = agree_across_ranks(self._comm_group, [
            self._capture_costs[self._bucket(shape)].graph if sized and ok else 0
            for shape, ok in zip(ordered, planned_everywhere, strict=True)
        ], self._device, torch.amax)
        captured = [False] * len(ordered)
        for i in sorted(range(len(ordered)), key=lambda i: -costs[i]):
            shape = ordered[i]
            if not planned_everywhere[i]:
                continue
            shape_captured = True
            for slot in range(self._num_slots):
                # keep ranks in lockstep: the region may hold collectives, and a
                # rank still in pre-capture setup would mismatch one already in the
                # warmup forward. Every rank walks all the slots even after a
                # failure here, or the barrier counts diverge and they hang.
                if self._comm_group is not None:
                    self._comm_group.tp_group.barrier()
                    self._comm_group.sp_group.barrier()
                try:
                    self._capture_one(shape, slot, forwards=0 if sized else self.NUM_WARMUP)
                except Exception:
                    shape_captured = False
                    logger.error(
                        "PiecewiseAcceleratorGraphRunner[%s]: failed to capture bs=%d "
                        "total_tokens=%d slot=%d", self._label, shape.bs,
                        shape.total_tokens, slot, exc_info=True,
                    )
            captured[i] = shape_captured
            if shape_captured:
                logger.info(
                    "PiecewiseAcceleratorGraphRunner[%s]: captured bs=%d total_tokens=%d "
                    "(%d slots)", self._label, shape.bs, shape.total_tokens,
                    self._num_slots,
                )

        # `ordered` comes from the config, so every rank agrees on the order
        for shape, ok in zip(
            ordered,
            agree_across_ranks(self._comm_group, captured, self._device, torch.amin),
            strict=True,
        ):
            if ok:
                continue
            # all-or-nothing per shape: a shape missing a slot can't rotate
            dropped = [
                slot for slot in range(self._num_slots)
                if self._graphs.pop(PiecewiseGraphKey(
                    bs=shape.bs, seq_len=shape.total_tokens, slot=slot
                ), None) is not None
            ]
            if dropped:
                logger.error(
                    "PiecewiseAcceleratorGraphRunner[%s]: dropping bs=%d total_tokens=%d "
                    "slots=%s, not captured on every rank / for every slot",
                    self._label, shape.bs, shape.total_tokens, dropped,
                )
            self.dropped_shapes.append((shape.bs, shape.total_tokens))
        if self.dropped_shapes:
            logger.error(
                "PiecewiseAcceleratorGraphRunner[%s]: captured %d of %d shapes; these "
                "run eagerly: %s", self._label,
                len(ordered) - len(self.dropped_shapes), len(ordered),
                self.dropped_shapes,
            )

    def _capture_one(self, shape: PiecewiseCaptureShape, slot: int, forwards: int) -> None:
        # as in `AcceleratorGraphRunner._capture_one`
        with self._warmed(shape, slot, forwards) as warmed:
            torch.accelerator.empty_cache()
            graph, static_outputs = capture_with_static_outputs(
                warmed.run, warmed.warm_outputs if forwards else self._outputs_like(self._bucket(shape)),
                self._memory_pool, self._device, self._autocast_dtype,
            )

        self._graphs[PiecewiseGraphKey(
            bs=shape.bs, seq_len=shape.total_tokens, slot=slot
        )] = PiecewiseGraphData(
            graph=graph,
            static_inputs=warmed.static_inputs,
            static_outputs=static_outputs,
            dummy_rids=list(warmed.dummy_rids),
            shape=shape,
            bucket=self._bucket(shape),
        )

    @contextmanager
    def _warmed(self, shape: PiecewiseCaptureShape, slot: int, forwards: int) -> Iterator[WarmedRegion]:
        """Plan ``shape`` and run the region eagerly ``forwards`` times; see
        ``AcceleratorGraphRunner._warmed``."""
        dummy_rids = self._dummy_rows.ensure(
            self._dummy_key(shape, slot), shape.bs
        )
        static_inputs = self._config.make_static_inputs(shape)
        step = self._declare(
            dummy_rids, shape.seq_lens, self._bucket(shape), capture=True,
            slot=slot
        )
        call = self._call_inputs(static_inputs, dummy_rids, step)

        fn = self._config.capture_fn
        if self._config.compile:
            fn = with_inductor_overrides(
                torch.compile(fn, **compile_kwargs(self._config.compile_mode)),
                shape.total_tokens,
            )

        def run_fn():
            with self._step_runner.compute_scope(step):
                return self._normalize_output(fn(call))

        try:
            self._plan(step, shape)
            self._graph_backend.synchronize()
            warm_outputs, peak = None, 0
            for _ in range(forwards):
                # as in `AcceleratorGraphRunner._warmed`
                live = torch.accelerator.memory_allocated(self._device)
                torch.accelerator.reset_peak_memory_stats(self._device)
                with autocast_scope(self._autocast_dtype, self._device.type):
                    warm_outputs = run_fn()
                peak = torch.accelerator.max_memory_allocated(self._device) - live
                # back to a clean stream state so the re-plan below (and the
                # capture after it) sees the shapes the first plan did
                self._dummy_rows.reset(dummy_rids)
                self._plan(step, shape)
            self._graph_backend.synchronize()

            yield WarmedRegion(
                run=run_fn, static_inputs=static_inputs,
                dummy_rids=dummy_rids, warm_outputs=warm_outputs, peak=peak,
            )
        finally:
            # pages stay with the dummy streams: replay's padding rows address
            # the same ids, so their plan finds the storage already resident
            self._dummy_rows.reset(dummy_rids)

    def _call_inputs(
        self,
        static_inputs: dict[str, torch.Tensor],
        step_ids: list[int],
        step=None,
    ) -> PiecewiseCallInputs:
        """What the region is handed, over the padded capture batch.

        Built once, for the capture: the region's Python runs only there, so
        the rows are the runner's padding ids and the only thing a replay
        changes is the contents of the static buffers. ``step`` is the
        capture's declaration, so the region can tell it runs on a leased
        slot (its context names the lease) and record the in-graph reads of
        the slot's buffers a replay relies on.
        """
        return PiecewiseCallInputs(
            static_inputs=static_inputs,
            engine_inputs=ModelInputsFromEngine(
                request_ids=list(step_ids),
                per_request_info=dummy_metadata(step_ids, PIECEWISE_WALK),
                resources=dict(self._resources),
                step=step,
            ),
            kwargs=self._config.forward_kwargs,
        )

    def _declare(
        self, request_ids: list[int], seq_lens: list[int],
        bucket: BucketKey, capture: bool, slot: int,
        real_bs: int | None = None,
    ):
        """The region's step over the padded batch, addressed at its slot.

        The declaration covers every row, but the context reports only the real
        head as ``request_ids`` and the padding tail on `set_padded_rids`, as
        the outer path does. A resource sizes its per-request writeback off
        ``ctx.request_ids``: counting the tail there makes it scatter padding
        rows back over real state.
        """
        if self._config.declare_step is None:
            return None
        step = self._config.declare_step(list(request_ids), list(seq_lens))
        if step is None:
            return None
        n = len(request_ids) if real_bs is None else real_bs
        ctx = StepContext(
            request_ids=tuple(request_ids[:n]),
            graph_walk=PIECEWISE_WALK,
            slot=slot,
            capture=capture,
            slot_lease=SlotLease(slot=slot, bucket=bucket),
            force_prefill=self._config.get_config_type() == PiecewiseConfigType.PACKED,
        )
        ctx.set_padded_rids(tuple(request_ids))
        step.set_ctx(ctx)
        return step

    def _plan(self, step, shape: PiecewiseCaptureShape) -> None:
        if step is None:
            return
        outcome = self._step_runner.admit(step)
        if not outcome.ok:
            raise RuntimeError(
                f"piecewise {self._label!r} admit failed for bs={shape.bs}, "
                f"total_tokens={shape.total_tokens}: {outcome.reason}"
            )
        self._step_runner.plan(step)

    @staticmethod
    def _normalize_output(out) -> dict[str, torch.Tensor]:
        """Coerce the captured callable's return into ``{name: Tensor}``.

        The contract is a dict; a bare tensor is accepted (under ``"x"``) so a
        single-output block loop can ``return x`` directly.
        """
        if isinstance(out, torch.Tensor):
            return {"x": out}
        if isinstance(out, dict):
            return out
        raise TypeError(
            f"piecewise capture_fn must return a Tensor or dict[str, Tensor], "
            f"got {type(out).__name__}"
        )

    # ── Replay ──────────────────────────────────────────────────────────

    def _padded_bs(self, batch_size: int) -> int | None:
        return next(
            (bs for bs in self._capture_batch_sizes if bs >= batch_size), None
        )

    def _resolve(
        self, batch_size: int, total_tokens: int | None,
    ) -> PiecewiseGraphData | None:
        """The captured bucket serving this call, or None for eager.

        BATCHED buckets are fixed by the padded batch size; PACKED ones take
        the smallest captured token bucket that fits.
        """
        padded_bs = self._padded_bs(batch_size)
        if padded_bs is None:
            return None
        if self._config.get_config_type() == PiecewiseConfigType.BATCHED:
            return self._graphs.get(PiecewiseGraphKey(
                bs=padded_bs, seq_len=self._config.seq_len * padded_bs,
                slot=self._current_slot
            ))
        if total_tokens is None:
            return None
        fits = [
            key for key in self._graphs
            if key.bs == padded_bs
            and key.slot == self._current_slot
            and key.seq_len >= total_tokens
        ]
        if not fits:
            return None
        return self._graphs[min(fits, key=lambda key: key.seq_len)]

    def can_run(self, batch_size: int, total_tokens: int | None = None) -> bool:
        return self._resolve(batch_size, total_tokens) is not None

    @property
    def lease_before_step(self) -> bool:
        """Whether this region takes its slot ahead of the declaration."""
        return self._config.lease_before_step

    def can_lease(self, inputs: list[NodeInputs]) -> bool:
        """The config's ``can_lease`` for this batch; True when it sets none."""
        return self._config.can_lease is None or self._config.can_lease(inputs)

    def lease_slot(
        self, batch_size: int, total_tokens: int | None = None,
    ) -> SlotLease | None:
        """The bucket this batch would replay on, asked before the forward.

        Same question ``run`` settles per call, answered early so the outer
        ``declare_step`` can plan the region's resources against it.
        """
        data = self._resolve(batch_size, total_tokens)
        return None if data is None else SlotLease(
            slot=self._current_slot, bucket=data.bucket
        )

    def run(
        self,
        static_inputs: dict[str, torch.Tensor],
        request_ids: list[int] | None = None,
        seq_lens: list[int] | None = None,
        real_bs: int | None = None,
    ) -> PiecewiseOutput:
        """Replay the captured region for these real inputs.

        Copies each real input into the runner-owned buffer of the same name,
        declares and plans the region's step over the padded batch, replays,
        then commits and returns the padded buffers behind a real-length view.

        Only the static buffers carry data into a replay: the region's Python
        ran once, at capture, so whatever it read off ``PiecewiseCallInputs``
        is baked into the graph.
        """
        if real_bs is None:
            if request_ids is not None:
                real_bs = len(request_ids)
            elif seq_lens is not None:
                real_bs = len(seq_lens)
            else:
                raise ValueError(
                    "piecewise run: pass real_bs, request_ids, or seq_lens to "
                    "determine the batch size"
                )

        is_packed = self._config.get_config_type() == PiecewiseConfigType.PACKED
        real_total_tokens = sum(seq_lens) if seq_lens is not None else None
        data = self._resolve(real_bs, real_total_tokens if is_packed else None)
        if data is None:
            raise RuntimeError(
                f"piecewise {self._label!r}: no captured graph for bs={real_bs}, "
                f"total_tokens={real_total_tokens}"
            )

        for name, value in static_inputs.items():
            buffer = data.static_inputs.get(name)
            if buffer is None or not isinstance(value, torch.Tensor):
                continue
            n = value.shape[0]
            buffer[:n].copy_(value)
            if n < buffer.shape[0]:
                # the padded tail is real compute for a BATCHED capture, so it
                # reads whatever is here; zero rather than last step's values
                buffer[n:].zero_()

        step = None
        if request_ids is not None:
            step_ids = [
                *request_ids, *data.dummy_rids[real_bs:data.shape.bs]
            ]
            step = self._declare(
                step_ids,
                self._config.replay_seq_lens(data.shape, seq_lens, real_bs),
                data.bucket,
                capture=False,
                slot=self._current_slot,
                real_bs=real_bs,
            )
        try:
            self._plan(step, data.shape)
            data.graph.replay()
            if step is not None:
                self._step_runner.commit(step)
        finally:
            self._dummy_rows.reset(data.dummy_rids[real_bs:data.shape.bs])

        return PiecewiseOutput(
            data.static_outputs, real_total_tokens if is_packed else real_bs
        )
