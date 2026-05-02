import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import EngineType
from mminf.graph.base import GraphNode
from mminf.worker.engine_manager import EngineManager
from mminf.worker.node_manager_utils import WorkerGraphsManager

logger = logging.getLogger(__name__)


@dataclass
class ReadyNodeEntry:
    """A ready node entry for a single request."""
    request_id: str
    worker_graph_id: str
    graph_walk: str


@dataclass
class ScheduledBatch:
    """A batch of nodes ready to be executed."""
    node_name: str
    graph_walk: str
    node_objects: dict[str,GraphNode]
    # request_id -> worker_graph_id (for push-back on OOM)
    request_to_worker_graph: dict[str, str] = None

    # Phase 2 chunked-prefill: per-request flag indicating whether this
    # request's slice should produce sampled output this step. Populated
    # by `MicroScheduler._get_chunked_step_batch` for thinker_step batches;
    # propagated to ``NodeBatch.is_terminal_per_request`` at build time.
    # Empty dict (default) means "all terminal" — Phase 1 behavior.
    is_terminal_per_request: dict[str, bool] = None

    # Phase 2 chunked-prefill: per-request chunk size for prefill chunks.
    # Populated alongside ``is_terminal_per_request`` for thinker_step
    # batches. Used by the worker to (a) slice prompt token tensors and
    # (b) advance ``prefill_tokens_consumed`` after the step. None /
    # empty means "no chunked-prefill in this batch".
    prefill_chunk_sizes: dict[str, int] = None


# ----------------------------------------------------------------------
# Phase 2: chunked-prefill mixed-batch packing.
#
# Decode-first packing under a per-step token budget. Each decode is 1
# token; prefill chunks fill remaining budget. If a prefill's remaining
# tokens fit in budget, that chunk is "terminal" — the request transitions
# to decode after this step, so we sample its output. Non-terminal chunks
# skip lm_head + sampling.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeReadyRequest:
    """A request that has 1 token to decode this step."""

    rid: str


@dataclass(frozen=True)
class PrefillReadyRequest:
    """A request with chunked prefill in progress."""

    rid: str
    tokens_remaining: int


@dataclass
class ChunkedStepPlan:
    """The scheduler's verdict for one mixed-batch step.

    decode_rids: requests that should each contribute 1 token (decode).
    prefill_allocations: rid → number of tokens to feed this step.
    terminal_prefills: rids whose prefill completes this step (last chunk).
        These need lm_head + sampling to produce the first decode token.
    """

    decode_rids: list[str] = field(default_factory=list)
    prefill_allocations: dict[str, int] = field(default_factory=dict)
    terminal_prefills: set[str] = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        return len(self.decode_rids) + sum(self.prefill_allocations.values())


def plan_chunked_step(
    ready_decodes: list[DecodeReadyRequest],
    ready_prefills: list[PrefillReadyRequest],
    max_step_tokens: int,
) -> ChunkedStepPlan:
    """Pack one step under the token budget.

    Decode-first because each decode is 1 token; running them keeps tail
    latency stable. Prefill fills remaining budget. If a prefill request's
    remaining tokens fit in the budget, the chunk is terminal (transitions
    the request to decode after this step).
    """
    if max_step_tokens <= 0:
        raise ValueError(f"max_step_tokens must be positive, got {max_step_tokens}")

    plan = ChunkedStepPlan()
    budget = max_step_tokens

    # Decodes first.
    for req in ready_decodes:
        if budget <= 0:
            break
        plan.decode_rids.append(req.rid)
        budget -= 1

    # Prefill fills remaining budget.
    for req in ready_prefills:
        if budget <= 0:
            break
        if req.tokens_remaining <= 0:
            continue
        chunk = min(req.tokens_remaining, budget)
        plan.prefill_allocations[req.rid] = chunk
        if chunk == req.tokens_remaining:
            plan.terminal_prefills.add(req.rid)
        budget -= chunk

    return plan


# Priority: lower value = higher priority
# AR decode is most latency-sensitive
PRIORITY = {
    EngineType.AR: 0,
    EngineType.CODE_PREDICTOR: 0.5,
    EngineType.FLOW: 1,
    EngineType.ENC_DEC: 2,
    EngineType.AUDIO_CODEC: 3,
}

class SchedulingType(Enum):
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"


class MicroScheduler:
    """
    Simple MVP scheduler: scans all worker graph queues for ready nodes,
    groups by node name, returns the highest-priority group.
    """

    # Seconds to wait before retrying a held request after OOM
    HOLD_BACKOFF_SECONDS = 0.05

    def __init__(
        self, engine_manager: EngineManager,
        sched_type=SchedulingType.ROUND_ROBIN,
        max_step_tokens: int = 2048,
    ):
        self.engine_manager = engine_manager
        self.batch_number = 0
        self.sched_type = sched_type
        self.node_and_walk_to_last_batch_num = {}
        # request_id -> monotonic time until which the request is held
        self.held_until: dict[str, float] = {}

        # Phase 2 chunked-prefill: max tokens per step (decode + prefill).
        # Only consulted when an AR engine has scheduler_owns_chunking=True;
        # otherwise the existing single-walk batching path is used.
        # TODO(Phase 2 Task 8): surface this in YAML model_config; for now
        # the worker passes it through from model_config["max_step_tokens"]
        # if set, else this default.
        self.max_step_tokens = max_step_tokens

    def _select_node_priority(
        self, node_name_to_requests: dict[str, list[ReadyNodeEntry]]
    ):
        # Pick the node name with highest priority (lowest PRIORITY value)
        best_node_name = None
        best_priority = float("inf")

        for node_name in node_name_to_requests:
            if node_name not in self.engine_manager.node_to_engine:
                continue
            engine = self.engine_manager.get_engine(node_name)
            prio = PRIORITY.get(engine.engine_type(), 99)
            if prio < best_priority:
                best_priority = prio
                best_node_name = node_name
        if best_node_name is None:
            return None, None
        entries = node_name_to_requests[best_node_name]

        # Enforce same graph_walk for the entire batch.
        # Pick the most common graph_walk to maximize batch size;
        # remaining requests stay in the queue for the next cycle.
        walk_counts: dict[str, int] = {}
        for e in entries:
            walk_counts[e.graph_walk] = walk_counts.get(e.graph_walk, 0) + 1
        graph_walk = max(walk_counts, key=walk_counts.get)

        return node_name, graph_walk

    def _select_node_rr(
        self, node_name_to_requests: dict[str, list[ReadyNodeEntry]]
    ):
        best_node_name = None
        best_graph_walk = None
        least_recent_step = float('inf')

        for node_name, reqs in node_name_to_requests.items():
            for req in reqs:
                step = self.node_and_walk_to_last_batch_num.get((
                    node_name, req.graph_walk
                ), 0)
                if step < least_recent_step:
                    least_recent_step = step
                    best_node_name = node_name
                    best_graph_walk = req.graph_walk
        return best_node_name, best_graph_walk

    def hold_requests(self, request_ids: list[str]) -> None:
        """Put requests on hold for a brief backoff period after OOM."""
        deadline = time.monotonic() + self.HOLD_BACKOFF_SECONDS
        for rid in request_ids:
            self.held_until[rid] = deadline

    # ------------------------------------------------------------------
    # Phase 2 chunked-prefill: mixed batch packing.
    # ------------------------------------------------------------------

    def _ar_engine_owns_chunking(self) -> bool:
        """True iff this scheduler should pack mixed thinker_step batches.

        The flag lives on the AREngine. We only consult it when an AR
        engine is present on this worker; non-AR-only workers (e.g.,
        Talker / Code2Wav) preserve Phase 1 behavior.
        """
        ar_engine = self.engine_manager.get_ar_engine()
        if ar_engine is None:
            return False
        return getattr(ar_engine, "scheduler_owns_chunking", False)

    def _get_chunked_step_batch(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        target_node_name: str | None = None,
        exclude_target: tuple[str, str] | None = None,
    ) -> ScheduledBatch | None:
        """Pack a single ``thinker_step`` batch from ready AR-engine requests.

        Walks every ready AR node, classifying each request as decode-ready
        (``is_prefill_complete=True``) or prefill-ready (mid-chunked-prefill).
        Calls ``plan_chunked_step`` with the worker's max-step budget, then
        pops the popped nodes' GraphNodes and returns a single ``ScheduledBatch``
        whose ``graph_walk`` is ``thinker_step`` and whose
        ``is_terminal_per_request`` map encodes the plan.

        Returns None when no AR requests are ready (caller falls back to the
        non-chunked scheduling path).

        Caveat (Phase 2 Task 5 scope): the per-request prompt-token slicing
        for prefill chunks and the post-step ``prefill_tokens_consumed``
        advance are wired separately on the worker side — this method only
        produces the batch + metadata. Behavioral coverage of the full
        round-trip lives in Task 6 (qwen3_omni weights).
        """
        now = time.monotonic()
        # Expire stale hold entries (mirrors get_next_batch).
        self.held_until = {
            rid: t for rid, t in self.held_until.items() if t > now
        }

        # rid -> (worker_graph_id, node_name, graph_walk, fwd_info)
        ready: dict[str, tuple[str, str, str, CurrentForwardPassInfo]] = {}

        for worker_graph_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue
                if request_id in self.held_until:
                    continue
                for sname in node_names:
                    if target_node_name is not None and sname != target_node_name:
                        continue
                    if sname not in self.engine_manager.node_to_engine:
                        continue
                    engine = self.engine_manager.get_engine(sname)
                    if engine.engine_type() != EngineType.AR:
                        continue
                    node_partition = worker_graphs_manager.get_partition_for_node(sname)
                    graph_walk = worker_graphs_manager.get_graph_walk(
                        request_id, node_partition
                    )
                    if exclude_target is not None and (sname, graph_walk) == exclude_target:
                        continue
                    fwd_info = worker_graphs_manager.get_fwd_info(request_id, node_partition)
                    if not engine.check_ready(sname, request_id, fwd_info):
                        continue
                    # Take the first eligible (rid, node_name) pair per request.
                    if request_id not in ready:
                        ready[request_id] = (worker_graph_id, sname, graph_walk, fwd_info)

        if not ready:
            return None

        # Classify each ready request.
        decode_ready: list[DecodeReadyRequest] = []
        prefill_ready: list[PrefillReadyRequest] = []
        for rid, (_wg_id, _sname, _walk, fwd_info) in ready.items():
            if fwd_info.is_prefill_complete:
                decode_ready.append(DecodeReadyRequest(rid=rid))
            else:
                tokens_remaining = max(
                    0,
                    fwd_info.prefill_tokens_total - fwd_info.prefill_tokens_consumed,
                )
                prefill_ready.append(
                    PrefillReadyRequest(rid=rid, tokens_remaining=tokens_remaining)
                )

        plan = plan_chunked_step(decode_ready, prefill_ready, self.max_step_tokens)
        if plan.total_tokens == 0:
            return None

        # Build the unified batch. Order: decodes first, then prefills.
        batch_rids = list(plan.decode_rids) + list(plan.prefill_allocations.keys())
        node_objects: dict[str, GraphNode] = {}
        request_to_worker_graph: dict[str, str] = {}
        is_terminal_per_request: dict[str, bool] = {}
        prefill_chunk_sizes: dict[str, int] = {}

        # Pop ready nodes for each rid; choose the same node name across rids
        # (the scheduler's _select_node helpers normally enforce this; here we
        # accept whatever node was ready since all are AR. In practice on a
        # qwen3-omni-style worker the AR node is "Thinker" for all rids.)
        node_name_for_batch: str | None = None
        for rid in batch_rids:
            wg_id, sname, _walk, _fwd = ready[rid]
            queue = worker_graphs_manager.queues[wg_id]
            popped = queue.pop_ready_nodes(rid, [sname])
            if not popped:
                continue
            assert len(popped) == 1
            node_objects[rid] = popped[0]
            request_to_worker_graph[rid] = wg_id
            if node_name_for_batch is None:
                node_name_for_batch = sname

            if rid in plan.decode_rids:
                is_terminal_per_request[rid] = True
            else:
                # prefill chunk: terminal iff this is the last chunk
                is_terminal_per_request[rid] = rid in plan.terminal_prefills
                prefill_chunk_sizes[rid] = plan.prefill_allocations[rid]

        if not node_objects or node_name_for_batch is None:
            return None

        logger.debug(
            "MicroScheduler chunked-step: node=%s rids=%d decodes=%d prefills=%d budget=%d",
            node_name_for_batch, len(node_objects),
            len(plan.decode_rids), len(plan.prefill_allocations),
            self.max_step_tokens,
        )
        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(
            node_name_for_batch, "thinker_step"
        )] = self.batch_number

        return ScheduledBatch(
            node_name=node_name_for_batch,
            graph_walk="thinker_step",
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
            is_terminal_per_request=is_terminal_per_request,
            prefill_chunk_sizes=prefill_chunk_sizes,
        )

    def get_next_batch(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        max_batch_size: int | None = None,
        target_node_name: str | None = None,
        target_graph_walk: str | None = None,
        exclude_target: tuple[str, str] | None = None,
    ) -> ScheduledBatch | None:
        """
        Scans all worker graph queues for ready nodes.
        Groups by node name. Returns highest-priority group.

        Args:
            max_batch_size: If set, limit the number of requests in the batch.
                Useful for CUDA graph compatibility (must match captured sizes).
            target_node_name: If set, only schedule this node name.
            target_graph_walk: If set, only schedule this graph walk.
            exclude_target: If set, skip this (node_name, graph_walk) pair.
        """
        # Phase 2 chunked-prefill: when the AR engine on this worker has
        # opted into scheduler-driven chunking, dispatch through the
        # mixed-batch packer first. If it produces a batch, return it; if
        # no AR requests are ready (None), fall through to the existing
        # path so non-AR engines continue to schedule normally. The flag
        # defaults to False so Phase 1 behavior is preserved.
        # ``target_graph_walk`` overrides this path so callers explicitly
        # asking for a specific walk (e.g., a non-thinker walk on a
        # multi-engine worker) still get the legacy semantics.
        if (
            target_graph_walk is None
            and self._ar_engine_owns_chunking()
        ):
            chunked = self._get_chunked_step_batch(
                worker_graphs_manager,
                target_node_name=target_node_name,
                exclude_target=exclude_target,
            )
            if chunked is not None:
                return chunked
            # Fall through: AR queue empty this tick, but other engines
            # (e.g., Talker) may still have ready work.

        # Collect all ready (node_name, request_id, graph_walk) tuples
        # grouped by node name
        node_name_to_requests: dict[str, list[ReadyNodeEntry]] = {}
        now = time.monotonic()

        # Expire stale hold entries
        self.held_until = {
            rid: t for rid, t in self.held_until.items() if t > now
        }

        for worker_graph_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue  # request was removed between scheduling cycles
                # Skip requests in OOM backoff
                if request_id in self.held_until:
                    continue
                for sname in node_names:
                    if target_node_name is not None and sname != target_node_name:
                        continue
                    node_partition = worker_graphs_manager.get_partition_for_node(sname)
                    graph_walk = worker_graphs_manager.get_graph_walk(request_id, node_partition)
                    if target_graph_walk is not None and graph_walk != target_graph_walk:
                        continue
                    if exclude_target is not None and (sname, graph_walk) == exclude_target:
                        continue
                    fwd_info = worker_graphs_manager.get_fwd_info(request_id, node_partition)
                    # check if the node is ready on the engine level
                    # (e.g., for AR, whether the kv cache is read in)
                    engine = self.engine_manager.get_engine(sname)
                    if not engine.check_ready(sname, request_id, fwd_info):
                        continue
                    node_name_to_requests.setdefault(sname, []).append(
                        ReadyNodeEntry(request_id, worker_graph_id, graph_walk)
                    )

        if not node_name_to_requests:
            return None

        if self.sched_type == SchedulingType.PRIORITY:
            best_node_name, graph_walk = self._select_node_priority(node_name_to_requests)
        elif self.sched_type == SchedulingType.ROUND_ROBIN:
            best_node_name, graph_walk = self._select_node_rr(node_name_to_requests)
        else:
            raise NotImplementedError(f"Unkown scheduling type {self.sched_type}")

        if best_node_name is None:
            return None

        # Pop ready nodes for all requests of this node name
        entries = [e for e in node_name_to_requests[best_node_name] \
                   if e.graph_walk == graph_walk]

        # Limit batch size if requested (e.g., for CUDA graph compatibility)
        if max_batch_size is not None and len(entries) > max_batch_size:
            entries = entries[:max_batch_size]

        node_objects = {}
        request_to_worker_graph = {}

        for entry in entries:
            queue = worker_graphs_manager.queues[entry.worker_graph_id]
            popped = queue.pop_ready_nodes(entry.request_id, [best_node_name])
            if popped:
                assert len(popped) == 1
                node_objects[entry.request_id] = popped[0]
                request_to_worker_graph[entry.request_id] = entry.worker_graph_id

        if not node_objects:
            return None

        logger.debug(
            "MicroScheduler scheduling node %s with graph walk %s for %d requests",
            best_node_name, graph_walk, len(node_objects)
        )
        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(
            best_node_name, graph_walk
        )] = self.batch_number

        return ScheduledBatch(
            node_name=best_node_name,
            graph_walk=graph_walk,
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
        )
