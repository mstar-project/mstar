import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerTPGroups
from mstar.engine.kv_store import KVCacheConfig, StoreWritePolicy
from mstar.profile.worker import ExecTimings


class EngineType(Enum):
    KV_CACHE = "kv_cache"
    STATELESS = "stateless"


@dataclass(frozen=True)
class EngineCapabilities:
    """Static declaration of optional surfaces an engine implements.

    The worker and ``EngineManager`` consult these flags instead of
    ``isinstance`` / ``hasattr`` probes to decide whether to iterate or
    dispatch into engine-specific code paths (e.g. CPU-offload victim
    selection, KV-cache LRU tracking, store write-policy push). The
    default ``EngineCapabilities()`` declares an engine that needs none
    of the optional surfaces — stateless engines leave it untouched.
    """
    requires_kv_cache: bool = False
    supports_cpu_offload: bool = False


@dataclass
class NodeBatch:
    """Input to an engine's execute_batch()."""
    node_name: str
    graph_walk: str
    request_ids: list[str]

    # {request_id: {input_name: list[tensor]}}
    per_request_input_tensors: dict[str, NameToTensorList]
    per_request_info: dict[str, CurrentForwardPassInfo] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    # rids whose consumed streaming input was the final chunk — this pass
    # reports the partition done (see worker._postprocess_batch)
    final_stream_rids: set[str] = field(default_factory=set)
    exec_timings: ExecTimings = field(default_factory=ExecTimings)


@dataclass
class PreparedBatch:
    """CPU-side preparation result; carries the batch plus per-request input
    objects produced by ``submodule.prepare_inputs``.

    ``skipped_rids`` lets a submodule veto specific requests (e.g. audio codec
    returns None for streaming inputs that don't yet have enough frames). The
    template loop emits empty output slots for skipped rids so the worker's
    per-rid bookkeeping stays consistent.

    ``metadata`` is the per-engine extension point — KV-cache engines stash
    cache-manager references and label sets here; stateless engines leave
    it empty.
    """
    batch: NodeBatch
    submodule: Any | None = None
    node_inputs: list = field(default_factory=list)
    skipped_rids: set[str] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)

    @property
    def active_request_ids(self) -> list[str]:
        return [rid for rid in self.batch.request_ids if rid not in self.skipped_rids]

    @property
    def graph_walk(self) -> str:
        return self.batch.graph_walk

    @property
    def node_name(self) -> str:
        return self.batch.node_name


@dataclass
class PlannedBatch:
    """Plan-state result; carries the prepared batch plus any planning artifacts
    (FlashInfer attention plan handles, dispatch-mode decisions, etc.).

    Engines without a separate planning step (stateless) leave ``metadata`` empty.
    """
    prepared: PreparedBatch
    metadata: dict = field(default_factory=dict)

    @property
    def batch(self) -> NodeBatch:
        return self.prepared.batch

    @property
    def submodule(self):
        return self.prepared.submodule

    @property
    def node_inputs(self):
        return self.prepared.node_inputs

    @property
    def active_request_ids(self) -> list[str]:
        return self.prepared.active_request_ids


@dataclass
class NodeOutput:
    """Output from an engine's execute_batch()."""
    # {request_id: {output_name: [tensor]}}
    per_request_output_tensors: dict[str, NameToTensorList]
    # Set to True when page allocation failed; worker should hold and retry.
    allocation_failed: bool = False
    # When allocation_failed=True, details about the failure:
    alloc_pages_short: int = 0
    alloc_failed_request_id: str | None = None
    # CUDA event recorded on the default stream after this step's GPU work
    # was submitted, used by the worker to (a) sync only on GPU(N) (not
    # GPU(N+1) which is queued behind it after speculation), and (b) gate a
    # side-stream D→H copy of the produced tokens. Set by the worker in
    # _execute_on_gpu_thread; engines don't populate it themselves.
    completion_event: "torch.cuda.Event | None" = None


class BaseEngine(ABC):
    def __init__(
        self, enable_nvtx: bool = False,
        enable_profile: bool=False,
        **kwargs
    ):
        self.enable_nvtx = enable_nvtx
        self.enable_profile = enable_profile

    def has_autocast(self):
        return True

    def get_max_batch_size(self, node_name: str, graph_walk: str):
        return None

    @abstractmethod
    def engine_type(self) -> EngineType:
        ...

    @abstractmethod
    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        tp_groups: WorkerTPGroups,
        kv_cache_config: list[KVCacheConfig],
        device: torch.device,
        **kwargs
    ) -> None:
        """
        Receive the submodules this engine is responsible for
        (keyed by node name) and perform engine-specific initialization
        (KV cache allocation, FlashInfer workspace, etc.).
        """
        ...

    # ── Named execution hooks ────────────────────────────────────────────
    #
    # Default ``execute_batch`` is a template method that calls these four
    # hooks in order. Subclasses with custom error handling (e.g. KV-cache
    # allocation-failure recovery) may override ``execute_batch`` entirely
    # and call the hooks themselves; subclasses with the standard flow
    # override only the per-phase hooks.
    #
    # Splitting these out lets the worker overlap CPU prep with the GPU
    # forward (run ``prepare_batch`` for batch N+1 while ``execute_forward``
    # for batch N is queued on the GPU), and lets new engines opt into a
    # planning stage without changing the worker.

    def prepare_batch(self, batch: NodeBatch) -> PreparedBatch:
        """CPU-side preparation. Override to fetch submodules, run
        ``prepare_inputs``, and report any rids that should be skipped this
        step. The returned ``PreparedBatch`` flows into ``plan_batch``.

        Default: return an empty PreparedBatch with all requests active.
        """
        return PreparedBatch(batch=batch)

    def plan_batch(self, prepared: PreparedBatch) -> PlannedBatch:
        """Optional planning step (e.g. FlashInfer attention plan). Wraps the
        prepared batch with any plan-state references the forward needs.

        Default: no-op pass-through. Engines with attention planning override.
        """
        return PlannedBatch(prepared=prepared)

    @abstractmethod
    def execute_forward(self, planned: PlannedBatch) -> NodeOutput:
        """GPU forward + sampling. Every concrete engine implements this —
        the default ``execute_batch`` template calls it after
        ``prepare_batch`` and ``plan_batch``.
        """
        ...

    def postprocess_batch(self, planned: PlannedBatch, output: NodeOutput) -> None:
        """CPU-side per-rid postprocess. Default: no-op."""
        return

    def finalize_batch(self, batch: NodeBatch) -> None:
        """Worker-driven cleanup hook called once per batch in a ``finally``,
        regardless of whether the forward succeeded, returned an
        ``allocation_failed`` NodeOutput, or raised. Subclasses use it to
        mirror engine-internal state (e.g. KV-cache seq_info) back onto
        ``batch.per_request_info`` so the next iter sees the updated values.

        Must be safe to call on any batch — including one where
        ``execute_batch`` was never reached.
        """
        return

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        """Template method: prepare → plan → forward → postprocess.

        Subclasses with custom error/cleanup envelopes (e.g. allocation-failure
        recovery, finally-block state writebacks) may override this entirely
        and call the hooks themselves to keep the cleanup invariant intact.
        """
        if self.enable_profile and batch.exec_timings.start is None:
            batch.exec_timings.start = time.perf_counter()
        prepared = self.prepare_batch(batch)
        if not prepared.active_request_ids:
            return NodeOutput(
                per_request_output_tensors={rid: {} for rid in batch.request_ids}
            )
        planned = self.plan_batch(prepared)
        output = self.execute_forward(planned)
        self.postprocess_batch(planned, output)
        # Empty output slots for skipped rids keep worker bookkeeping consistent.
        output.per_request_output_tensors.update(
            {rid: {} for rid in prepared.skipped_rids}
        )
        return output

    # ── Async pre-execution hooks ────────────────────────────────────────
    #
    # The worker uses these to coordinate a double-buffered CUDA-graph
    # runner: ``reserve_replay_slot`` picks the slot that the upcoming
    # batch will replay into; ``pre_plan_for_batch`` warms the plan-state
    # cache on that slot ahead of GPU submission so the GPU thread can
    # skip a GIL-contended plan() call; ``reset_pre_plan_for_batch``
    # rolls back the warmed state when the dispatched batch is dropped
    # before it reaches the GPU.
    #
    # Engines without an async pre-plan surface inherit the defaults
    # below — all three are safe no-ops, so the worker calls them
    # unconditionally.

    def reserve_replay_slot(self, batch: NodeBatch) -> int | None:
        """Reserve a CUDA-graph replay slot for ``batch`` and stash it on
        ``batch.metadata['cuda_graph_slot']``. Returns the slot index,
        or ``None`` when no captured graph matches.
        """
        return None

    def pre_plan_for_batch(
        self,
        batch: NodeBatch,
        prev_completion_event: "torch.cuda.Event | None" = None,
    ) -> bool:
        """Off-thread CPU planning: warm plan-state caches (e.g. FlashInfer
        attention plan) on the slot reserved by ``reserve_replay_slot``
        so the GPU thread can skip an inline plan() call.

        Called by the worker on its ``plan_executor`` thread, ahead of
        GPU submission. Returns ``True`` when planning ran (the caller
        should await its future before running this batch); ``False``
        when no planning was performed (the GPU thread plans inline).
        """
        return False

    def reset_pre_plan_for_batch(self, batch: NodeBatch) -> None:
        """Clear any pre-plan state that ``pre_plan_for_batch`` set on the
        slot for ``batch``. Called when the dispatched batch is dropped
        before reaching the GPU thread, so the next plan() call on that
        slot recomputes from scratch instead of trusting stale state.
        """
        return

    # ── Capabilities + optional surfaces ────────────────────────────────
    #
    # ``capabilities`` is a class-level declaration of which optional
    # surfaces this engine class implements. Worker / EngineManager check
    # it instead of ``isinstance`` / ``hasattr`` probes. The methods below
    # are the corresponding surfaces — all safe no-op defaults so engines
    # that don't opt in can still be called uniformly. KVCacheEngine
    # overrides both the capability flags and the methods; stateless
    # engines leave them at default.

    capabilities = EngineCapabilities()

    def lru_tracked_nodes(self) -> list[str]:
        """Nodes for which the worker should LRU-track per-request activity
        (used to pick CPU-offload victims). Default: no nodes — stateless
        engines have no KV state to age out.
        """
        return []

    def set_alloc_write_policy(self, policy: StoreWritePolicy) -> None:
        """Apply a store write policy. Default: no-op — engines without an
        alloc manager have nothing to set.
        """
        return

    def offload_candidates(self, node_name: str) -> list[tuple[str, int]]:
        """Return ``(request_id, gpu_pages_held)`` for every request with
        GPU pages on ``node_name``. The worker partitions the result into
        in-batch vs external candidates and picks an eviction victim.
        Default: empty list — no offloadable state.
        """
        return []

    def offload_request(self, node_name: str, request_id: str) -> int:
        """Offload ``request_id``'s KV pages on ``node_name`` to CPU.
        Returns the number of GPU pages freed (0 if nothing was freed or
        the engine doesn't support offload).
        """
        return 0

    def reload_request(self, node_name: str, request_id: str) -> bool:
        """Reload an offloaded request back to GPU on ``node_name``.
        Returns True on success; False if the request isn't offloaded, GPU
        pages are insufficient, or the engine doesn't support offload.
        """
        return False

    def is_offloaded(self, node_name: str, request_id: str) -> bool:
        """Whether ``request_id`` is currently CPU-offloaded on ``node_name``.
        Default: False.
        """
        return False

    def execute_with_max_batch_size(self, batch: NodeBatch) -> NodeOutput:
        if self.enable_profile:
            batch.exec_timings.start = time.perf_counter()
        bs = self.get_max_batch_size(batch.node_name, batch.graph_walk)
        n = len(batch.request_ids)
        if bs is None or n <= bs:
            return self.execute_batch(batch)

        output = NodeOutput(
            per_request_output_tensors={}
        )

        for i in range(0, n, bs):
            rids = batch.request_ids[i:min(i+bs, n)]
            minibatch = NodeBatch(
                node_name=batch.node_name,
                graph_walk=batch.graph_walk,
                request_ids=rids,
                per_request_input_tensors={
                    rid: batch.per_request_input_tensors[rid] for rid in rids \
                        if rid in batch.per_request_input_tensors
                },
                per_request_info={
                    rid: batch.per_request_info[rid] for rid in rids \
                        if rid in batch.per_request_info
                },
                metadata=batch.metadata
            )
            minibatch_out = self.execute_batch(minibatch)
            # Fold each sub-batch's forward window back into the parent batch's
            # exec_timings (earliest launch wins); the worker sets fwd_end on the
            # parent once the whole split forward has been submitted.
            if self.enable_profile:
                batch.exec_timings.update(minibatch.exec_timings)
            output.per_request_output_tensors.update(
                minibatch_out.per_request_output_tensors
            )
            if minibatch_out.allocation_failed:
                output.allocation_failed = True
                output.alloc_pages_short = minibatch_out.alloc_pages_short
                output.alloc_failed_request_id = minibatch_out.alloc_failed_request_id
                return output
        return output

    @abstractmethod
    def add_request(self, request_id: str, **kwargs) -> None:
        ...

    @abstractmethod
    def remove_request(self, request_id: str) -> None:
        ...

    def check_ready(
        self, node_name: str, request_id: str,
        request_info: CurrentForwardPassInfo,
    ):
        """
        Check if the engine is ready to execute.
        """
        return True

    def check_stop_for_batch(
        self, batch: NodeBatch, output: NodeOutput
    ) -> dict[str, set[str]]:
        """
        Per-rid stop-condition check for a finished batch.

        Called by the worker on its slow-postprocess path *after*
        ``execute_batch`` returns. May read tensor values. Returns
        ``{request_id: {loop_name, ...}}`` for rids whose loops should stop.

        Default: no stops. Stateless engines override this to delegate the
        check to the submodule; the AR engine has its own value-driven logic.
        """
        return {}

    def warmup(self) -> None:
        """Optional CUDA graph capture. Override in subclasses."""
        return

    def shutdown(self) -> None:
        return
