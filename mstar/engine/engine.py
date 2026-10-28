

import logging
import os
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo, merge_publish_info
from mstar.distributed.communication import JointGroups, WorkerParallelGroups
from mstar.engine.accelerator_graph_runner import (
    AcceleratorGraphRunner,
    CapturePlan,
    PiecewiseAcceleratorGraphRunner,
    autocast_scope,
    fail_if_graphs_required,
    plan_captures,
)
from mstar.engine.fp32_precision import Fp32Precision, fp32_precision_scope, node_fp32_precision
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
from mstar.engine.resources.base import EngineResourceInfo, KeyedWalks, build_resource
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.engine.resources.step import (
    ADMIT_OK,
    FULL_ADMIT_NOT_READY,
    AdmitOutcome,
)
from mstar.graph.runtime.base import GraphRuntime
from mstar.model.base import keyed_labels
from mstar.model.submodule_base import (
    EMPTY_INPUT_METADATA,
    ARNodeInputs,
    BatchedModelOutput,
    HostRows,
    InputMetadata,
    LazyRequestStates,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.platforms.compile import with_inductor_overrides
from mstar.profile.worker import ExecTimings
from mstar.utils import knobs
from mstar.utils.client_errors import client_error_text
from mstar.utils.device import DeviceCapability
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
    from mstar.model.base import Model

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


@dataclass
class SubmoduleManagement:
    submodule: NodeSubmodule
    forward: Callable
    forward_batched: Callable
    joint_comm_group: JointGroups
    resources: dict[str, Resource]
    accelerator_graph_runner: AcceleratorGraphRunner | None = None
    # the runner capture ran with, graphs or none: a bucket it didn't capture
    # runs eager, and the floor covers eager steps only up to its largest
    capture_runner: AcceleratorGraphRunner | None = None
    # the submodule's declared float32 settings; None keeps the process default
    fp32_precision: Fp32Precision | None = None

    # Rotated globally, not per runner: slot-keyed buffers are shared across
    # buckets and regions, so per-runner counters let consecutive steps collide
    # on a slot. Leased from the plan thread and the GPU thread both, hence the
    # lock.
    _next_slot: int = field(init=False, default=0)
    _num_slots: int = field(init=False, default=1)
    _forced_double_buffer: bool = field(init=False, default=False)
    _slot_lock: threading.Lock = field(
        init=False, default_factory=threading.Lock, repr=False
    )

    # label -> PiecewiseCudaGraphRunner for inner-loop capture; spread into
    # ModelInputsFromEngine so the submodule's forward can look them up
    piecewise_runners: dict[str, PiecewiseAcceleratorGraphRunner] = field(
        default_factory=dict
    )

    def __post_init__(self):
        # Two hazards want a second slot, both of them the host running ahead of
        # the GPU: pre-plan needs plan(N+1) to write buffers replay(N) isn't
        # reading, and ``force_double_buffer`` is the same for a resource
        # staging into a reused host buffer. The latter is not pre-plan
        # specific — the main runner has it too (see `Resource`).
        # TODO: submodule-wide, so a capture touching neither kind still pays
        # double the buffers. Each runner could size its own count from the
        # resources its captures touch, as PiecewiseCudaGraphRunner does.
        preplan_enabled = knobs.pre_plan_enabled()
        self._forced_double_buffer = any(
            res.force_double_buffer for res in self.resources.values()
        )
        double_buffered = self._forced_double_buffer or (
            preplan_enabled
            and any(res.supports_preplan for res in self.resources.values())
        )
        if not double_buffered:
            self._num_slots = 1
            return
        self._num_slots = int(os.environ.get("MSTAR_NUM_SLOTS", "2"))
        if self._num_slots < 2:
            # MSTAR_NUM_SLOTS=1 is the knob for turning double-buffering off,
            # but neither hazard has a single-buffered form: one slot puts the
            # next step's staging on top of a DMA that may not have retired.
            # Turn pre-planning off (`knobs: {pre_plan: false}`, or
            # MSTAR_PRE_PLAN_SPEC=0) to drop to a single slot.
            logger.warning(
                "MSTAR_NUM_SLOTS=%d, but a resource on this node needs "
                "double-buffering; using 2 slots.", self._num_slots,
            )
            self._num_slots = 2

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def needs_slot_fence(self) -> bool:
        """Whether reusing a slot has to wait out the GPU work that last held
        it — true exactly when a resource stages into a reused host buffer."""
        return self._forced_double_buffer

    @property
    def next_slot(self) -> int:
        """The slot the next lease hands out. Read unlocked: an int read is
        atomic under the GIL, and the only other leaser is the plan thread,
        which is gated on the previous batch's ``commit_done``."""
        return self._next_slot

    def lease_slot(self) -> int:
        with self._slot_lock:
            slot = self._next_slot
            self._next_slot = (self._next_slot + 1) % self._num_slots
            return slot

    def set_piecewise_slot(self, slot: int) -> None:
        """Point the region runners at this batch's slot, on the GPU thread just
        before its forward. Set here rather than at lease time: leasing runs
        ahead of the forward (pre-plan reserves N+1's slot while N is in flight),
        so a lease-time write would race the replay reading ``_current_slot``."""
        for runner in self.piecewise_runners.values():
            runner.set_slot(slot)

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
        return self.metadata.dynamic_loop_iter_counts

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
            return self.metadata.dynamic_loop_iter_counts
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

    # Populated on batch preparation
    inputs: list[NodeInputs] | None = None
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
        self.per_request_info_wrapped = {
            rid: ForwardPassInfoWrapper(
                # Sparse: only rids with stream chunks or loop counters get an
                # entry, so default like the .get() at prepare_inputs does.
                info=info, metadata=self.per_request_input_metadata.get(
                    rid, EMPTY_INPUT_METADATA,
                ),
            ) for rid, info in self.per_request_info.items()
        }

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

    def register_prepare_batch(self, inputs: list[NodeInputs]):
        self.inputs = inputs

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


class Engine:
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
    ):
        assert gpu_memory_fraction is None or 0 < gpu_memory_fraction <= 1, (
            f"gpu_memory_fraction is a fraction of the device, got {gpu_memory_fraction}"
        )
        self._device = None
        self._device_capability = device_capability
        self._autocast_dtype = autocast_dtype
        # the share of the device this worker's allocations may hold once
        # capture is done, None for whatever is free
        self._gpu_memory_fraction = gpu_memory_fraction
        self._resources: dict[str, Resource] = {}
        # resource key -> the nodes on this engine that use it
        self._resource_nodes: dict[str, frozenset[str]] = {}
        self._submodules: dict[str, SubmoduleManagement] = {}
        self._runner: StepRunner = None
        self._graph_runtime = graph_runtime

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
    ):
        self._device = device
        if self._device_capability is None or self._device_capability.device != device:
            self._device_capability = DeviceCapability.of(device)
        self._enable_nvtx = nvtx_enabled(self._enable_nvtx, device)
        if kv_cache_type is None:
            kv_cache_type = self._autocast_dtype

        node_names = set(submodules.keys())
        node_to_resources = {}
        specs_by_key = resolve_spec_dependencies(specs)

        self._classify_tp_roles(node_names, parallel_groups)

        for spec in specs:
            relevant_nodes = spec.nodes & node_names
            if len(relevant_nodes) == 0:
                continue # resource not needed

            # A spec is a logical resource identity and may span replicas on
            # different workers (for example BAGEL's three CFG branches).
            # This Engine constructs only the local instance, whose consumers
            # validate_deployment checked share one parallel group.
            joint_comm_group = parallel_groups.get_joint_group_for_node(
                next(iter(relevant_nodes))
            )
            self._resources[spec.resource_key] = build_resource(
                spec,
                EngineResourceInfo(
                    device=device,
                    joint_comm_group=joint_comm_group,
                    transfer_engine_info=transfer_engine_info,
                    kv_dtype=kv_cache_type,
                    needs_remote_transfer=(
                        parallel_groups.resource_needs_remote_transfer(
                            spec.nodes, relevant_nodes
                        )
                    ),
                    dependencies={
                        key: specs_by_key[key] for key in spec.depends_on()
                    },
                    max_rows_in_combined=self._max_rows_in_combined(
                        spec.resource_key, relevant_nodes, submodules, model,
                    ),
                ),
            )

            for node in relevant_nodes:
                node_to_resources.setdefault(node, []).append(spec.resource_key)
            self._resource_nodes[spec.resource_key] = frozenset(relevant_nodes)

        # rid -> (node, walk) of its last step here when that step was rows-only,
        # i.e. the sampler's slot master holds the token its next step reads
        self._loopback_steps: dict[int, tuple[str, str]] = {}
        self._runner = StepRunner(
            self._resources,
            # every node, including one that owns nothing (Code2Wav) — an
            # absent node would fall back to the full sweep
            node_resources={n: node_to_resources.get(n, []) for n in node_names},
            enable_nvtx=self._enable_nvtx,
        )

        self._open_prefix_caches(specs_by_key, model)

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
                fp32_precision=node_fp32_precision(node_name, submodule),
            )
            submodule.bind_node_resources(resources)
            if getattr(submodule, "reads_device_loopback", False):
                # the sampler keeps its slot masters only for a node that
                # reads them back on the device (see NodeSubmodule)
                for resource in resources.values():
                    enable = getattr(resource, "enable_device_loopback_reader", None)
                    if enable is not None:
                        enable()

    def paged_kv_caches(self) -> list[PagedKVCache]:
        """Every paged KV cache this engine built, with the nodes that use it."""
        from mstar.engine.resources.kv.manager import KVManager

        return [
            PagedKVCache(key, self._resource_nodes[key], resource)
            for key, resource in self._resources.items()
            if isinstance(resource, KVManager)
        ]

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

    def _autocast_dtype_for(self, submodule: NodeSubmodule) -> torch.dtype | None:
        """This node's autocast dtype, or None for one that opted out."""
        return None if submodule.disable_autocast else self._autocast_dtype

    def warmup(self) -> None:
        ag_runners: dict[str, AcceleratorGraphRunner] = {}
        piecewise: dict[str, dict[str, PiecewiseAcceleratorGraphRunner]] = {}
        for node_name, submodule_mgmt in self._submodules.items():
            submodule = submodule_mgmt.submodule
            ag_runners[node_name] = AcceleratorGraphRunner(
                submodule_name=node_name,
                submodule=submodule,
                resources=submodule_mgmt.resources,
                step_runner=self._runner,
                device=self._device,
                autocast_dtype=self._autocast_dtype_for(submodule),
                joint_comm_group=submodule_mgmt.joint_comm_group,
                num_slots=submodule_mgmt.num_slots,
                enable_nvtx=self._enable_nvtx,
                device_capability=self.device_capability,
            )
            piecewise[node_name] = self._build_piecewise_runners(
                node_name, submodule_mgmt
            )

        # Every runner claims its static buffers before any of them captures:
        # nodes share resources, so a build driven by a later node would move
        # buffers an earlier node's graphs already recorded the address of.
        for node_name in self._submodules:
            ag_runners[node_name].prepare_for_capture()
            for runner in piecewise[node_name].values():
                runner.prepare_for_capture()

        # every bucket sized before any is captured, so one plan weighs them all
        # against the largest eager step on the device
        plan = self._plan_captures(ag_runners, piecewise)

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
                # sizing runs the forwards and captures, so under the node's precision too
                with fp32_precision_scope(submodule_mgmt.fp32_precision):
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

    def _build_piecewise_runners(
        self, node_name: str, submodule_mgmt: SubmoduleManagement,
    ) -> dict[str, PiecewiseAcceleratorGraphRunner]:
        """One runner per region the submodule declares, not yet captured.

        The caller captures them, and drops a region whose capture failed so
        its forward takes the eager path for that label.
        """
        node_dtype = self._autocast_dtype_for(submodule_mgmt.submodule)
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
            )
            runners[label] = runner
        return runners

    def _open_prefix_caches(self, specs_by_key, model) -> None:
        """Root each resource's cache in the weights, the preprocessing, and the
        config of every resource its pages were planned against."""
        checkpoint = model.checkpoint_path() if model is not None else None
        shared = [
            checkpoint_identity(checkpoint) if checkpoint is not None else b"",
            model.preprocess_fingerprint() if model is not None else "",
        ]
        declared = keyed_labels(model) if model is not None else {}
        self._prefix_model = type(model).__name__
        # node -> the walks a probe may run on, from the keys of every label on a cache the node uses
        self._keyed_walks: dict[str, set[str]] = {}
        for key, by_label in declared.items():
            for node in specs_by_key[key].nodes:
                self._keyed_walks.setdefault(node, set()).update(
                    walk for label in by_label.values() for walk in label.keys.walks
                )
        # a cache another engine holds is taken as open when its config says so
        opened_keys = {
            key for key in declared
            if key not in self._resources and specs_by_key[key].config.prefix_cache
        }
        for key, resource in self._resources.items():
            if checkpoint is None and declared.get(key):
                config = specs_by_key[key].config
                if config.prefix_cache and not config.prefix_cache_salt:
                    logger.info(
                        "KV %s: prefix cache off: %s names no checkpoint, so "
                        "builds that differ only in their weights cannot be "
                        "told apart; set prefix_cache_salt to enable it",
                        key, self._prefix_model,
                    )
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
            opened = resource.enable_prefix_cache(fingerprint(*shared, *parts), {
                label: KeyedWalks(
                    kv_label.keys.walks, tuple(walk for walk, _ in kv_label.growth if kv_label.keys.growth),
                )
                for label, kv_label in declared.get(key, {}).items()
            })
            if not opened or not declared.get(key):
                continue
            opened_keys.add(key)
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

    def prepare_inputs(self, batch: ExecutingBatch) -> None:
        """Per-rid ``submodule.prepare_inputs``, onto ``batch.inputs``.

        Per-rid, so a raise is attributable to one request: record it and take
        that rid out rather than losing the batch. A rid the submodule declines
        (returns None) leaves the same way, without being an error.
        """
        if self._enable_nvtx:
            range_push(f"engine.prepare_inputs.bs{len(batch.request_ids)}")
        try:
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
        loopback = self._loopback_rows(batch)
        if not mixed and walk not in keyed_walks:
            uniform = submodule.uniform_row_inputs(
                walk, has_loopback=loopback is not None and all(loopback),
            )
            if uniform is not None:
                # one object for all the step's rows: the per-row call would
                # return an equal one each time (128 calls at a full decode
                # batch, on the gpu thread before the launch)
                node_inputs = [uniform] * len(batch.request_ids)
                batch.register_prepare_batch(node_inputs)
                batch.running_batched = submodule.can_batch(
                    batch=batch, model_inputs=node_inputs
                )
                return
        # a step of one walk probes (or not) for every row alike
        step_probes = not mixed and walk in keyed_walks
        for i, rid in enumerate(batch.request_ids):
            row_walk = walk_of(rid) if mixed else walk
            probe_prefix = (row_walk in keyed_walks) if mixed else step_probes
            try:
                req_inputs = prepare(
                    graph_walk=row_walk,
                    fwd_info=wrapped[rid],
                    inputs=tensors.get(rid, {}),
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
                batch.register_failure(rid, error, walk=row_walk)
                continue
            if req_inputs is None:
                batch.skipped_rids.add(rid)
            else:
                node_inputs.append(req_inputs)

        batch.register_prepare_batch(node_inputs)
        batch.drop_rids(batch.skipped_rids | batch.failed_requests.keys())
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

    def device_loopback_signals(
        self, node_name: str, graph_walk: str,
    ) -> frozenset[str]:
        """The node's loop-back signals it keeps on the device this walk, so
        the worker routes them without a tensor (see the submodule hook)."""
        mgmt = self._submodules.get(node_name)
        if mgmt is None:
            return frozenset()
        return mgmt.submodule.device_loopback_signals(graph_walk)

    def extend_prefix_chains(
        self, batch: ExecutingBatch, outputs: dict[str, NameToTensorList],
        host_rows: HostRows | None = None,
    ) -> None:
        """Key what this step generated, from the stop check's host copy."""
        keys = self._runner.prefix_chain_keys(batch.node_name)
        if not keys:
            return
        # The batch call keys each row under its own walk, so a combined-walk
        # step takes it too; only the absence of host rows falls through.
        if host_rows is not None:
            ctx = batch.step_context
            self._runner.extend_prefix_chains_batch(
                batch.request_ids, batch.node_name,
                # a step of one walk passes the walk: no call per row
                ctx.walk_of if ctx.request_walks else ctx.graph_walk,
                host_rows, outputs, keys=keys,
            )
            return
        for rid in batch.request_ids:
            per_rid = outputs.get(rid)
            if isinstance(per_rid, dict):
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
        assert isinstance(inputs, ARNodeInputs), (
            f"{self._prefix_model} keys the {walk!r} walk of {batch.node_name} "
            "for prefix reuse, but that walk's inputs are not a sequence to cut"
        )
        submodule = self._submodules[batch.node_name].submodule
        if inputs.kwargs or inputs.resource_step_info or (
            inputs.tensor_inputs and not getattr(submodule, "cuts_tensor_inputs", False)
        ):
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
        return submodule.split_inputs(
            walk, batch.per_request_info_wrapped[rid], inputs,
            matched, inputs.input_seq_len,
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
                self._autocast_dtype_for(self._submodules[batch.node_name].submodule),
                device_type=self._device.type,
            ):
                laps = PhaseLaps("engine.exec.")
                # a caller that pre-planned already reserved one; otherwise take it here
                lease = batch.step_context.slot_lease or self.reserve_replay_slot(batch)
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
            if batch.admit_error is None:
                self._record_loopback(batch)
                batch.outputs.loopback_rids = batch.loopback_rids
            batch.outputs_ready.set()
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

        inputs = batch.inputs
        req_info = batch.per_request_info_wrapped
        if lease is not None:
            inputs = cg_runner.pad_inputs(lease, inputs)
            req_info = cg_runner.step_metadata(
                lease, batch.request_ids, req_info
            )
            batch.step_context.set_padded_rids(
                cg_runner.step_ids(lease, batch.request_ids)
            )
        laps.lap("pad")

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
            if self._enable_nvtx:
                range_push("engine.collect_outputs")
            t0 = phase_start()
            try:
                out = self._collect_outputs(
                    submodule_mgmt, lease, raw, inputs, req_info,
                    request_ids=batch.request_ids,
                    step_request_ids=batch.step_context.padded_request_ids,
                    rows_only=batch.rows_only,
                )
            finally:
                phase_end("engine.collect_outputs", t0)
                if self._enable_nvtx:
                    range_pop()
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
            if lease is not None:
                cg_runner.release(lease, real_bs)
            laps.lap("release")

    def _exec_per_request(
        self, batch: ExecutingBatch
    ) -> dict[str, NameToTensorList]:
        """Full step cycle per request, eager (no lease), outputs merged.

        The unbatchable fallback: no capture applies (a lease would have set
        ``running_batched``), so each request declares/admits/plans/commits on
        its own, keeping the plan matched to its single-request forward.
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
                if fence:
                    slot_events[slot] = torch.Event(device=self._device)
                    slot_events[slot].record()

                merged.update(self._collect_outputs(
                    submodule_mgmt, None, raw, [inp], req_info,
                    request_ids=[rid], step_request_ids=(rid,),
                ))
            finally:
                if nvtx:
                    range_pop()

        # Releasing `commit_done` lets the plan thread pre-plan the next batch,
        # which stages `next_slot` while this batch's work may still be queued.
        # Only that slot needs draining: every other one the loop touched is
        # re-consumed a batch or more later, by which point the worker has
        # synced on this step.
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
        if step is None:
            self._maybe_lease_piecewise_regions(batch.node_name, ctx, inputs)
            if nvtx:
                range_push("engine.declare_step")
            t0 = phase_start()
            try:
                step = submodule.declare_step(
                    graph_walk=batch.graph_walk, request_ids=rids, inputs=inputs,
                    slot_lease=ctx.slot_lease,
                    piecewise_leases=ctx.piecewise_leases,
                    per_request_info=batch.per_request_info,
                )
            finally:
                phase_end("engine.declare_step", t0)
                if nvtx:
                    range_pop()
        if step is None:
            # the submodule declared no step (a node owning no resources);
            # there is nothing to admit, and the forward still runs
            return ADMIT_OK, None
        # admit reads the step's ctx, so bind it before the sweep rather than
        # in `_drive_step`
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
            batch.register_admit_error(
                admit_outcome.reason, admit_outcome.failed_resource,
            )
        return admit_outcome, step

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
            if not runner.lease_before_step or not runner.can_lease(inputs):
                continue
            lease = runner.lease_slot(bs, total_tokens)
            if lease is not None:
                leases[label] = lease
        if leases:
            ctx.set_piecewise_leases(leases)

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
        """declare → admit → plan → preprocess → forward → commit for one forward.

        ``request_ids`` are the real rids (for admit-fail / collect); the model
        runs over ``ctx.request_ids`` (padded under a lease). Returns
        ``(raw_outputs, step)``; ``raw_outputs`` is None when admit failed.
        """
        nvtx = self._enable_nvtx
        cg_runner = submodule_mgmt.accelerator_graph_runner
        submodule = submodule_mgmt.submodule
        rids = list(ctx.padded_request_ids)

        if step is not None:
            if lease is not None and step.cg_key_info != lease.bucket.cg_key_info:
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

            # Admit was already done, can move on straight to plan
            if nvtx:
                # promoted = a pre-plan was consumed; fresh = planned inline
                range_push(
                    "engine.plan.promoted" if batch.preplan_event is not None
                    else "engine.plan.fresh"
                )
            t0 = phase_start()
            try:
                self._runner.plan(step)
            finally:
                phase_end(
                    "engine.plan.promoted" if batch.preplan_event is not None
                    else "engine.plan.fresh",
                    t0,
                )
                if nvtx:
                    range_pop()

        laps = PhaseLaps("engine.")
        engine_inputs = ModelInputsFromEngine(
            request_ids=rids,
            per_request_info=req_info,
            resources=submodule_mgmt.resources,
            piecewise_runners=submodule_mgmt.piecewise_runners,
            # padding rows get their own states, like their cache streams:
            # the submodule indexes this by step id, not by real rid
            per_request_states=LazyRequestStates(submodule, rids),
            captured=lease is not None,
            # `inputs` is padded after the real rows under a lease
            real_seq_len=sum(inp.input_seq_len for inp in inputs[:len(request_ids)]),
            step=step,
            per_request_input_metadata=batch.per_request_input_metadata,
        )
        laps.lap("engine_inputs")
        if nvtx:
            range_push("engine.preprocess")
        t0 = phase_start()
        try:
            preprocessed = submodule.preprocess(
                ctx.graph_walk, engine_inputs=engine_inputs, inputs=inputs,
            )
        finally:
            phase_end("engine.preprocess", t0)
            if nvtx:
                range_pop()

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
                    lease, running_batched, request_ids, release_event,
                )
        finally:
            phase_end("engine.forward", t0)
            if nvtx:
                range_pop()

        if step is not None:
            if nvtx:
                range_push("engine.commit")
            t0 = phase_start()
            try:
                self._runner.commit(step)
            finally:
                phase_end("engine.commit", t0)
                if nvtx:
                    range_pop()
        return raw, step

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
        release_event: "threading.Event | None" = None,
    ) -> BatchedModelOutput:
        """Replay the leased slot, or run the eager forward.

        ``release_event`` is set as late as possible before the call that drops
        the GIL — the replay, or the submodule forward on the eager path."""
        graph_walk = batch.step_context.graph_walk
        if lease is not None:
            return cg_runner.run_forward(
                lease, preprocessed, plan_done_event=batch.preplan_event,
                launch_started_event=release_event,
            )
        if release_event is not None:
            release_event.set()
        if running_batched:
            return BatchedModelOutput.coerce(
                submodule_mgmt.forward_batched(
                    graph_walk, engine_inputs=engine_inputs, **preprocessed
                )
            )
        assert len(request_ids) == 1, (
            "the unbatched forward takes one request; batch of "
            f"{len(request_ids)} needs running_batched"
        )
        return BatchedModelOutput.coerce(
            {request_ids[0]: submodule_mgmt.forward(
                graph_walk, engine_inputs=engine_inputs, **preprocessed
            )}
        )

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
        for rid, node_inputs in zip(batch.request_ids, batch.inputs, strict=True):
            try:
                submodule.postprocess(
                    request_id=rid,
                    request_info=batch.per_request_info_wrapped[rid],
                    outputs=outputs.get(rid, {}),
                    inputs=node_inputs,
                )
            except Exception as error:
                batch.register_failure(rid, error)
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
        outputs = self.exec(batch)
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
            for rid in batch.cached_rids:
                self._runner.complete_cached_walk(rid, batch.node_name, batch.step_context.walk_of(rid))
        laps.lap("tail")
        return outputs

    def check_stop_for_batch(
        self, batch: ExecutingBatch, outputs: dict[str, NameToTensorList],
        host_rows: HostRows | None = None,
    ) -> dict[str, set[str]]:
        """Each rid's ``submodule.check_stop``, as rid -> loops that should stop.

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
                ctx = batch.step_context
                batched = submodule.check_stop_batched(
                    batch.request_ids, batch.per_request_info_wrapped, host_rows,
                    step_walk=None if ctx.request_walks else ctx.graph_walk,
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
        out_ids = (
            step_request_ids if lease is None
            else submodule_mgmt.accelerator_graph_runner.slot_for(lease).dummy_rids
        )
        outputs: dict[str, NameToTensorList] = {}

        row_clones, row_views = self._merge_per_rid(
            outputs, raw_outputs, request_ids, out_ids, submodule, req_info,
            rows_only=rows_only,
        )
        self._merge_unpacked(
            outputs, raw_outputs, request_ids, submodule,
            inputs[:len(request_ids)], req_info,
        )
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
        )

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
        submodule opts in and the walk has a capture."""
        submodule_mgmt = self._submodules[node_name]
        if not submodule_mgmt.submodule.split_batches_by_capture_key(graph_walk):
            return False
        cg_runner = submodule_mgmt.accelerator_graph_runner
        return cg_runner is not None and cg_runner.captures_walk(graph_walk)

    def capture_group(
        self, node_name: str, graph_walk: str, request_id: int,
        request_info: CurrentForwardPassInfo,
        input_metadata: InputMetadata | None = None,
    ) -> Any | None:
        """The capture bucket this one request would replay, by
        ``cg_key_info``; None when the walk has no capture or the submodule
        does not split by it.

        A batch spanning two groups matches no capture and runs eager, so the
        scheduler batches each group on its own when the submodule opts in
        (``split_batches_by_capture_key``). ``input_metadata`` is the step's
        input facts the lease will see, e.g. the stream chunk it consumes.
        """
        submodule_mgmt = self._submodules[node_name]
        if not submodule_mgmt.submodule.split_batches_by_capture_key(graph_walk):
            return None
        cg_runner = submodule_mgmt.accelerator_graph_runner
        if cg_runner is None or not cg_runner.captures_walk(graph_walk):
            return None
        if input_metadata is None:
            return submodule_mgmt.submodule.cg_key_info(graph_walk, {request_id: request_info})
        return submodule_mgmt.submodule.cg_key_info(
            graph_walk, {request_id: request_info},
            per_request_input_metadata={request_id: input_metadata},
        )

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

    def reserve_replay_slot(self, batch: ExecutingBatch) -> SlotLease | None:
        """Lease the slot this batch will replay on, before it is dispatched.

        Reserving up front is what puts pre-plan(N+1) and replay(N) on
        different slots. No captured graph for the batch's shape leaves the
        lease unset, i.e. the step runs eager.

        A batch that hasn't been through ``prepare_inputs`` has no token count
        yet, so only a batched capture can serve it — see
        ``CudaGraphRunner.select_batched_bucket``.
        """
        submodule_mgmt = self._submodules[batch.node_name]

        if batch.slot is None:
            batch.set_slot(submodule_mgmt.lease_slot())

        cg_runner = submodule_mgmt.accelerator_graph_runner
        if cg_runner is None:
            return None
        # Which of the walk's captures this batch belongs to. The engine can't
        # derive it — what separates two captures of one walk is the model's
        # business (bagel: guidance on/off) — and the lease is taken before the
        # step is declared, so the submodule is asked directly. It answers from
        # the same per-request facts its `declare_step` stamps on the step.
        batch.cg_key_info = submodule_mgmt.submodule.cg_key_info(
            batch.step_context.graph_walk, batch.per_request_info_wrapped,
            per_request_input_metadata=batch.per_request_input_metadata,
        )
        lease = cg_runner.lease_slot(
            graph_walk=batch.step_context.graph_walk,
            bs=len(batch.request_ids),
            slot=batch.slot,
            num_tokens=(
                None if batch.inputs is None
                else sum(inp.input_seq_len for inp in batch.inputs)
            ),
            cg_key_info=batch.cg_key_info,
        )
        if lease is not None:
            batch.lease_slot(lease)
            batch.step_context.force_prefill = cg_runner.is_packed(lease)
        return lease

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
        if self._enable_nvtx:
            range_push(f"engine.pre_plan.bs{len(batch.request_ids)}")
        try:
            return self._pre_plan_for_batch(batch, submodule_mgmt, cg_runner, lease)
        finally:
            if self._enable_nvtx:
                range_pop()

    def _pre_plan_for_batch(
        self, batch: ExecutingBatch, submodule_mgmt: SubmoduleManagement,
        cg_runner, lease: SlotLease,
    ) -> bool:

        laps = PhaseLaps("engine.pre_plan.")
        # Declare-only, so the unprepared case takes the bucket's shared
        # template rows rather than cloning a fresh set per pre-plan.
        inputs = (
            cg_runner.declare_inputs_for(lease)
            if batch.inputs is None
            else cg_runner.pad_inputs(lease, batch.inputs)
        )
        batch.step_context.set_padded_rids(
            cg_runner.step_ids(lease, batch.request_ids)
        )
        laps.lap("pad")
        t0 = phase_start()
        step = submodule_mgmt.submodule.declare_step(
            graph_walk=batch.step_context.graph_walk,
            request_ids=batch.step_context.padded_request_ids,
            inputs=inputs,
            slot_lease=lease,
            per_request_info=batch.per_request_info,
        )
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
            with stream if stream is not None else nullcontext():
                self._runner.pre_plan(step)
            laps.skip()
            batch.preplan_event = torch.Event(device=self._device)
            batch.preplan_event.record(stream)
            batch.preplanned_rids = tuple(batch.request_ids)
            laps.lap("event")
        finally:
            batch.step_context.is_preplan = False
        return True

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

    def evictable(self, node_name: str) -> bool:
        return any(
            resource.supports_eviction
            for resource in self._submodules[node_name].resources.values()
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
        if freed > 0 and node_name in self._resident_delta:
            self._resident_delta[node_name].add_offloaded(
                self._graph_runtime.get_rid_string(request_id)
            )
        return freed

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
    ) -> None:
        self._runner.ingest_request(request_id, overrides)

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
        for submodule_mgmt in self._submodules.values():
            submodule_mgmt.submodule.cleanup_request(request_id)
        return freed

    def shutdown(self):
        for resource in self._resources.values():
            resource.cleanup()
