import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from mstar.engine.base import EngineType
from mstar.graph.base import GraphNode
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker.engine_manager import EngineManager
from mstar.worker.node_manager_utils import WorkerGraphsManager

logger = logging.getLogger(__name__)

# Mixed-batch chunk grid (MSTAR_MIXED_CHUNK_SIZES). Kept as a
# scheduler-local duplicate of ThinkerSubmodule.MIXED_BATCH_CHUNK_SIZES (same
# pattern as qwen3_omni_model._PREFILL_CHUNK_BUCKETS mirroring
# ThinkerSubmodule.PREFILL_TOKEN_BUCKETS) so the scheduler does not import the
# model submodule. If the submodule default changes, change this too.
#
# Boot-time only (the grid IS the CUDA-graph capture buckets — see
# ThinkerSubmodule.get_cuda_graph_configs / MIXED_BATCH_CHUNK_SIZES): resolved
# once and cached for the process lifetime, not re-resolved at runtime.
_MIXED_CHUNK_SIZES_DEFAULT = (256, 288, 512)
_mixed_chunk_sizes_cache: tuple[int, ...] | None = None


def _resolve_mixed_chunk_sizes() -> tuple[int, ...]:
    """Parse MSTAR_MIXED_CHUNK_SIZES the same way ThinkerSubmodule does
    (union with the default, never shrink) so the scheduler's gate agrees
    with whatever grid was actually captured at boot."""
    global _mixed_chunk_sizes_cache
    if _mixed_chunk_sizes_cache is not None:
        return _mixed_chunk_sizes_cache
    raw = os.environ.get("MSTAR_MIXED_CHUNK_SIZES", "").strip()
    if not raw:
        _mixed_chunk_sizes_cache = _MIXED_CHUNK_SIZES_DEFAULT
        return _mixed_chunk_sizes_cache
    try:
        vals = {int(x) for x in raw.split(",") if x.strip()}
    except ValueError:
        _mixed_chunk_sizes_cache = _MIXED_CHUNK_SIZES_DEFAULT
        return _mixed_chunk_sizes_cache
    vals = {v for v in vals if v > 0}
    if not vals:
        _mixed_chunk_sizes_cache = _MIXED_CHUNK_SIZES_DEFAULT
    else:
        _mixed_chunk_sizes_cache = tuple(
            sorted(vals | set(_MIXED_CHUNK_SIZES_DEFAULT))
        )
    return _mixed_chunk_sizes_cache

# ---------------------------------------------------------------------------
# MSTAR_ENCODER_ASYNC: pipeline the vision/audio encoder ahead of the Thinker.
#
# Default OFF. When ON, the micro-scheduler treats ``vision_encoder`` and
# ``audio_encoder`` (both STATELESS) as higher-priority than the Thinker
# decode/prefill — but only until ``MSTAR_ENCODER_ASYNC_DEPTH`` encoded
# buffers are in flight (i.e. produced by the encoder but not yet consumed
# by the Thinker's matching ``prefill_vision`` / ``prefill_audio`` step).
#
# Without this flag, the encoder for request N+1 only runs after request N
# has been dispatched to the Thinker. The encoder GPU sits idle during the
# Thinker decode of N. With this flag, the encoder for N+1 starts while N
# is in ``thinker_decode``, so when the Thinker is ready for N+1's prefill
# the encoded buffer is already populated — zero encoder wait.
#
# Bounding the in-flight depth prevents the encoder from monopolizing the
# GPU under heavy admission; the default depth of 4 is a conservative
# ceiling chosen for that reason.
# ---------------------------------------------------------------------------
def _encoder_async_enabled() -> bool:
    return os.environ.get("MSTAR_ENCODER_ASYNC", "0") in ("1", "true", "True")


def _encoder_async_depth() -> int:
    raw = os.environ.get("MSTAR_ENCODER_ASYNC_DEPTH", "4")
    try:
        v = int(raw)
        return v if v > 0 else 4
    except ValueError:
        return 4


# Node names the encoder-async path treats as "encoder" walks.
_ENCODER_NODE_NAMES = frozenset({"vision_encoder", "audio_encoder"})
# Graph walks whose first node consumes an encoder output on the Thinker.
_ENCODER_CONSUMING_WALKS = frozenset({"prefill_vision", "prefill_audio"})


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


# Priority: lower value = higher priority
# KV-cache decode is most latency-sensitive
PRIORITY = {
    EngineType.KV_CACHE: 0,
    EngineType.STATELESS: 2,
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
        parallel_leader_nodes: set[str] | None = None,
        max_consec_tp_follower_batches: int = 1,
        tp_nodes: set[str] | None = None,
    ):
        self.engine_manager = engine_manager
        self.batch_number = 0
        self.sched_type = sched_type

        # lockstep-parallel (TP / SP instance) scheduling. Upstream (#154/#176)
        # renamed tp_rank_zero_nodes -> parallel_leader_nodes; the mixed-batch
        # code below still references self.tp_rank_zero_nodes, so alias
        # both names to the same set (identical concept: the rank-0 / leader
        # node of each lockstep-parallel instance).
        self.parallel_leader_nodes = parallel_leader_nodes
        self.tp_rank_zero_nodes = parallel_leader_nodes
        # Nodes with TP world_size > 1. Used to keep mixed-batch assembly
        # off TP nodes: a mixed batch's per-request walks are heterogeneous and
        # TP fan-out (ScheduleTPNode / sharding group lookup) is keyed by a
        # single batch walk, so a "thinker_mixed" TP batch has no sharding
        # group. TP mixed batching is not yet supported.
        self.tp_nodes = tp_nodes or set()
        self.tp_batches_pending_schedule = deque()
        self.num_consec_tp_follower_batches = 0
        self.max_consec_tp_follower_batches = max_consec_tp_follower_batches

        self.node_and_walk_to_last_batch_num = {}
        # Mixed-batch: count of thinker_mixed batches assembled by this
        # scheduler. Surfaced in the per-assembly INFO log; a monotonic counter
        # gives runtime evidence the mixed path is firing without DEBUG.
        self.mixed_batches_assembled = 0
        # EAGER FOLD (MSTAR_EAGER_FOLD): one-shot arm set by the worker
        # right before the get_next_batch that should assemble a LARGE-chunk
        # (C > _max_chunk_tokens()) mixed step to run EAGER. Read-and-cleared
        # by _try_assemble_mixed so exactly one assembly relaxes the chunk cap;
        # every other assembly (and the whole flag-off path) keeps the captured
        # cap, so a large chunk can never reach the CAPTURED spec-fold pop.
        # Default False -> byte-identical.
        self._eager_fold_armed = False
        # request_id -> monotonic time until which the request is held
        self.held_until: dict[str, float] = {}
        # Rids with a deferred remove; stop initiating new work for them.
        # Shared by reference with Worker._pending_removes.
        self.pending_removes: set[str] = set()

        # --- MSTAR_ENCODER_ASYNC bookkeeping ---------------------------------
        # ``encoder_async_enabled``: cached at construction so a single feature-
        # flag check governs every dispatch decision. Reading the env var once
        # also keeps the hot path branch cheap.
        # ``encoder_async_depth``: max number of encoded-but-not-yet-Thinker-
        # consumed buffers we'll let pile up. Each batched encoder dispatch
        # counts as one "in flight" credit regardless of how many requests
        # were coalesced into the batch — batching is independent of
        # pipelining depth, and the depth bound is about wall-clock head-room
        # (how far ahead of Thinker the encoder is allowed to run), not
        # about KV cache memory.
        # ``encoder_async_in_flight``: incremented when the scheduler returns
        # an encoder batch; decremented when the matching ``prefill_vision``
        # / ``prefill_audio`` step is scheduled on the Thinker (the
        # downstream consumer of the buffered embeddings).
        self.encoder_async_enabled = _encoder_async_enabled()
        self.encoder_async_depth = _encoder_async_depth()
        self.encoder_async_in_flight = 0
        # --- MSTAR_ENC_STEP_BUDGET bookkeeping -------------------------------
        # Per-encoder-node spatial-merge factor cache. Used to convert the raw
        # ViT patch-row count on an encoder node's input edge into post-merge
        # embed tokens for the step-budget accounting. Resolved lazily from the
        # submodule the first time an encoder wave is budgeted, then cached
        # (the factor is a fixed model property). See ``_encoder_merge_factor``.
        self._enc_merge_factor: dict[str, int] = {}
        if self.encoder_async_enabled:
            logger.info(
                "MicroScheduler: MSTAR_ENCODER_ASYNC=1 (depth=%d). "
                "Encoder walks pipeline ahead of Thinker decode.",
                self.encoder_async_depth,
            )

    def _maybe_pick_async_encoder(
        self, node_name_to_requests: dict[str, list[ReadyNodeEntry]]
    ) -> tuple[str | None, str | None]:
        """If MSTAR_ENCODER_ASYNC is enabled, prefer a ready encoder node.

        Returns ``(node_name, graph_walk)`` for the encoder pick, or
        ``(None, None)`` if no preemption applies (flag off, no encoder
        ready, depth saturated, or no encoder node has a ready request).

        Depth budget: when ``encoder_async_in_flight`` reaches
        ``encoder_async_depth`` we fall back to normal scheduling so the
        Thinker can catch up. This prevents the encoder from running an
        unbounded number of buffers ahead — bad both for GPU memory
        (each buffer pins vision/audio embeds) and for first-token latency
        of the requests already in the Thinker.
        """
        if not self.encoder_async_enabled:
            return None, None
        if self.encoder_async_in_flight >= self.encoder_async_depth:
            return None, None
        for node_name in _ENCODER_NODE_NAMES:
            entries = node_name_to_requests.get(node_name)
            if not entries:
                continue
            # Bias toward whichever walk is least-recently scheduled, mirroring
            # the round-robin tie-breaker for non-encoder nodes. In practice
            # both ``audio_encoder`` and ``vision_encoder`` only emit a single
            # walk (``prefill_audio`` / ``prefill_vision``) so this collapses
            # to the first entry's walk, but we keep the RR semantics for
            # robustness if a future walk reuses these encoder nodes.
            walk_counts: dict[str, int] = {}
            for e in entries:
                walk_counts[e.graph_walk] = walk_counts.get(e.graph_walk, 0) + 1
            graph_walk = max(walk_counts, key=walk_counts.get)
            return node_name, graph_walk
        return None, None

    def _select_node_priority(
        self, node_name_to_requests: dict[str, list[ReadyNodeEntry]]
    ):
        # MSTAR_ENCODER_ASYNC: pipeline the encoder ahead of the Thinker when
        # there's budget. See ``_maybe_pick_async_encoder`` for the depth
        # bound and rationale.
        async_node, async_walk = self._maybe_pick_async_encoder(
            node_name_to_requests
        )
        if async_node is not None:
            return async_node, async_walk

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
        # MSTAR_ENCODER_ASYNC: short-circuit the RR sweep to an encoder
        # node when there's pipeline budget; otherwise fall through to
        # the regular least-recent-step tie-breaker.
        async_node, async_walk = self._maybe_pick_async_encoder(
            node_name_to_requests
        )
        if async_node is not None:
            return async_node, async_walk

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

    def register_tp_follow(
        self, message: ScheduleTPNode
    ):
        self.tp_batches_pending_schedule.append(message)

    def _try_schedule_tp_follow(
        self, worker_graphs_manager: WorkerGraphsManager,
    ) -> ScheduledBatch | None:
        if len(self.tp_batches_pending_schedule) == 0:
            return
        first_tp_node: ScheduleTPNode = self.tp_batches_pending_schedule[0]
        if self.num_consec_tp_follower_batches >= self.max_consec_tp_follower_batches and \
                self.has_ready_excluding(
                    worker_graphs_manager,
                    (first_tp_node.node_name, first_tp_node.graph_walk)
                ):
            return
        # check if batch is ready
        node_partition = worker_graphs_manager.get_partition_for_node(first_tp_node.node_name)
        wgid = worker_graphs_manager.get_worker_graph_id_for_node(
            first_tp_node.request_ids[0], first_tp_node.node_name
        )
        queue = worker_graphs_manager.queues[wgid]
        for rid in first_tp_node.request_ids:
            wg = queue.per_request_queues[rid]
            if first_tp_node.node_name not in wg.ready_node_names:
                return
            fwd_info = worker_graphs_manager.get_fwd_info(rid, node_partition)
            # check if the node is ready on the engine level
            # (e.g., for AR, whether the kv cache is read in)
            engine = self.engine_manager.get_engine(first_tp_node.node_name)
            if not engine.check_ready(first_tp_node.node_name, rid, fwd_info):
                return

        node_objects = {}
        request_to_worker_graph = {}

        # TODO: this code is also repeated below, should pull into a helper fn
        for rid in first_tp_node.request_ids:
            popped = queue.pop_ready_nodes(rid, [first_tp_node.node_name])
            if popped:
                assert len(popped) == 1
                node_objects[rid] = popped[0]
                request_to_worker_graph[rid] = wgid

        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(
            first_tp_node.node_name, first_tp_node.graph_walk
        )] = self.batch_number

        self.tp_batches_pending_schedule.popleft()

        return ScheduledBatch(
            node_name=first_tp_node.node_name,
            graph_walk=first_tp_node.graph_walk,
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
        )


    # Mixed-batch walk names + capacity. Kept as module-adjacent
    # constants (mirroring ThinkerSubmodule.MIXED_BATCH_*) so the scheduler
    # does not import the model submodule. If those change, change these.
    _MIXED_DECODE_WALK = "thinker_decode"
    _MIXED_CHUNK_WALK = "prefill_text"
    # A VISION prefill chunk may also serve as the mixed step's
    # single chunk row when MSTAR_MIXED_BATCH_VISION is on. Same C caps; the
    # Thinker's mixed capture then carries deepstack statics + the MRoPE
    # side-channel (see ThinkerSubmodule.preprocess / get_cuda_graph_configs).
    _MIXED_VISION_CHUNK_WALK = "prefill_vision"
    _MIXED_MAX_DECODE = 31          # padded_bs 32 = up to 31 decode + 1 chunk row

    @classmethod
    def _max_chunk_tokens(cls) -> int:
        """Largest captured mixed-step chunk bucket (C). Was a hardcoded 512
        constant; now derives from the same MSTAR_MIXED_CHUNK_SIZES-resolved
        grid ThinkerSubmodule captured at boot (_resolve_mixed_chunk_sizes),
        so the chunk-size gate below — and MSTAR_COADMIT's budget clamp in
        worker.py, which reads this via the class, not an instance — track any
        grid growth automatically instead of silently going stale. Callable as
        both ``self._max_chunk_tokens()`` and ``MicroScheduler._max_chunk_tokens()``
        (worker.py needs the latter, from __init__, before a scheduler exists)."""
        return max(_resolve_mixed_chunk_sizes())

    def _mixed_min_decode(self) -> int:
        """Occupancy floor for chain-folding (see has_mixed_opportunity).

        Default 0 (no floor — preserves the previously-validated default
        behavior, including small-batch s2t folds) UNLESS an EAGER (every-step)
        fold policy is on — MSTAR_MIXED_SINGLE_CHUNK or MSTAR_MIXED_BUDGET_TOKENS
        — where every admission depends on fold slots and a small decode side
        means folding throttles admission: default 24 there. The occupancy floor
        exists because unconditional single-chunk folding was found to starve
        decode occupancy, so the eager budget policy inherits it.

        Overrides (precedence): MSTAR_MIXED_BUDGET_MIN_DECODE (the V2 knob) wins,
        else MSTAR_MIXED_MIN_DECODE (the general one), else the default above.
        Cached after first read (read once at init)."""
        v = getattr(self, "_mixed_min_decode_cached", None)
        if v is None:
            import os

            from mstar.model.qwen3_omni.qwen3_omni_model import (
                mixed_budget_tokens,
                mixed_single_chunk_enabled,
            )
            eager = mixed_single_chunk_enabled() or mixed_budget_tokens() > 0
            default = 24 if eager else 0
            raw = os.environ.get("MSTAR_MIXED_BUDGET_MIN_DECODE")
            if raw is None:
                raw = os.environ.get("MSTAR_MIXED_MIN_DECODE")
            try:
                v = int(raw) if raw is not None else default
            except ValueError:
                v = default
            self._mixed_min_decode_cached = v
        return v

    def _mixed_chunk_walks(self) -> set[str]:
        """Walks that may serve as a mixed step's single chunk row. prefill_text
        always; prefill_vision only when the vision-capable mixed capture exists.

        A prefill_vision chunk carries per-layer deepstack + a custom MRoPE
        advance, replayable ONLY by a thinker_mixed graph that baked the deepstack
        static buffers at capture time. The hard gate is therefore
        ``mixed_vision_capture_provisioned()`` (the boot-recorded capture truth),
        NOT the live env flag: a runtime MSTAR_MIXED_BATCH_VISION ON-flip after a
        vision-off boot must never add prefill_vision here — routing it to the
        unprovisioned (text-signature) graph is the UNCAP-IMA failure. The live
        ``mixed_batch_vision_enabled()`` is ANDed only to allow a safe-direction
        runtime OFF-flip (stop routing even though the graph could still replay
        it), which one-server dyn_ab A/Bs rely on. Shared by the peek, the
        assembler, and the mid-chain pop, so all three agree."""
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            mixed_batch_vision_enabled,
            mixed_vision_capture_provisioned,
        )
        walks = {self._MIXED_CHUNK_WALK}
        if mixed_vision_capture_provisioned() and mixed_batch_vision_enabled():
            walks.add(self._MIXED_VISION_CHUNK_WALK)
        return walks

    def _chunk_entry_passes_gates(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        node_name: str,
        node_partition,
        entry: "ReadyNodeEntry",
        eager_ok: bool = False,
    ) -> bool:
        """Per-request gates a chunk row must pass to join a mixed batch.

        Shared by the assembler (_try_assemble_mixed) and the read-only peek
        (has_mixed_opportunity) so the two never diverge on what counts as a
        mixable chunk. Gates (see _try_assemble_mixed docstring): chunk metadata
        present (chunked, not a full unchunked prefill), C within the largest
        captured bucket, and repetition_penalty == 1.0 (a discarded chunk sample
        must not perturb penalty state).

        ``eager_ok`` (MSTAR_EAGER_FOLD): raise the chunk-size ceiling
        from the largest CAPTURED bucket (``_max_chunk_tokens()``) to
        MSTAR_EAGER_FOLD_MAX_CHUNK, because an eager-fold mixed step is run
        UNCAPTURED (a dynamic varlen forward), so it is not bound by the capture
        grid. Passed True ONLY by _try_assemble_mixed when the worker armed an
        eager fold; NEVER by pop_mixed_chunk_for_spec (the spec-fold pop targets a
        CAPTURED replay, so its chunk must stay within the captured cap or it
        would route to an uncaptured graph = IMA). Default False keeps the
        captured cap."""
        fwd_info = worker_graphs_manager.get_fwd_info(
            entry.request_id, node_partition,
        )
        clen = fwd_info.step_metadata.get("prefill_chunk_len")
        if clen is None:
            return False  # unchunked full prefill — don't mix (bucket blow)
        cap = self._max_chunk_tokens()
        if eager_ok:
            from mstar.model.qwen3_omni.qwen3_omni_model import eager_fold_max_chunk
            cap = max(cap, eager_fold_max_chunk())
        if int(clen) > cap:
            return False
        sc = fwd_info.sampling_config.get(node_name)
        if sc is not None and getattr(sc, "repetition_penalty", 1.0) != 1.0:
            return False  # penalty state corruption on discarded chunk sample
        return True

    def _chunk_over_budget(
        self, clen, n_decode: int | None, budget_tokens: int,
    ) -> bool:
        """V2 budget gate (MSTAR_MIXED_BUDGET_TOKENS): a chunk is over budget
        when the policy is on (budget > 0) and folding it would push the mixed
        step past ``budget`` total tokens (``n_decode`` 1-token decode rows + the
        ``C``-token chunk). budget <= 0 (off) or an unknown decode size / chunk
        length never binds. Shared by the peek (has_mixed_opportunity) and the
        pop (pop_mixed_chunk_for_spec) so the two agree on which chunk folds."""
        if budget_tokens <= 0 or n_decode is None or clen is None:
            return False
        return n_decode + int(clen) > budget_tokens

    def has_mixed_opportunity(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        decode_target: tuple[str, str],
        n_decode: int | None = None,
        budget_tokens: int = 0,
        bypass_floor: bool = False,
    ) -> bool:
        """Read-only peek: would a mixed batch assemble RIGHT NOW if the decode
        group named by ``decode_target`` were back in the ready queue?

        The worker calls this while a decode chain is in flight — the decode
        rids are ``_speculatively_scheduled`` and thus absent from the ready
        scan (base.py register_ingested_input gate) — to decide whether to break
        the chain into the NON-speculative path so ``get_next_batch`` can
        assemble decode + chunk into a ``thinker_mixed`` batch there. Because the
        decode rids are absent, this peek only needs to confirm a mixable CHUNK
        is ready on the decode's node; the decode side is guaranteed to re-enter
        the queue once the in-flight step completes and un-flags.

        Mirrors the gates in ``_try_assemble_mixed`` without popping or mutating
        queue state. Returns False when the flag is off (so the default
        yield-away path is byte-identical when mixed batching is disabled).

        ``n_decode``: size of the in-flight decode chain, when the caller knows
        it. Folding admits at most ONE chunk per step, so during ramp-up (small
        decode side, many requests still prefilling) folding THROTTLES admission
        and starves decode occupancy — measured to substantially reduce req/s
        when every short span became foldable (MSTAR_MIXED_SINGLE_CHUNK).
        Standalone prefill fills the batch faster there. Gate: fold only when
        n_decode >= MSTAR_MIXED_MIN_DECODE (default 24); None skips the gate
        (non-spec assembler paths size the decode side themselves).

        ``budget_tokens``: V2 per-step token budget (MSTAR_MIXED_BUDGET_TOKENS).
        When > 0, a chunk only counts as an opportunity if n_decode + C fits the
        budget; the scan keeps looking for a smaller chunk otherwise. 0 = off
        (no cap), so the default yield-boundary behavior is byte-identical.

        ``bypass_floor``: MSTAR_COADMIT sets this when a BRAND-NEW
        request is ready, to skip the occupancy floor for that arrival — vLLM
        co-admits a new prefill into the decode step regardless of how many
        decodes are in flight. Only the fold-vs-standalone timing changes: the
        spec-fold pop (``pop_mixed_chunk_for_spec``) carries no floor, so a True
        here maps to a real fold. Default False keeps the default budget-policy
        behavior byte-identical.
        """
        from mstar.model.qwen3_omni.qwen3_omni_model import mixed_batch_enabled
        if not mixed_batch_enabled():
            return False
        if (
            not bypass_floor
            and n_decode is not None
            and n_decode < self._mixed_min_decode()
        ):
            return False

        decode_node_name, decode_walk = decode_target
        if decode_walk != self._MIXED_DECODE_WALK:
            return False
        if decode_node_name in self.tp_nodes:
            return False  # TP mixed batching is not yet supported

        chunk_walks = self._mixed_chunk_walks()
        node_partition = worker_graphs_manager.get_partition_for_node(
            decode_node_name
        )
        now = time.monotonic()
        for _wg_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue
                if request_id in self.pending_removes:
                    continue
                if request_id in self.held_until and self.held_until[request_id] > now:
                    continue
                if decode_node_name not in node_names:
                    continue
                if decode_node_name not in self.tp_rank_zero_nodes:
                    continue
                walk = worker_graphs_manager.get_graph_walk(
                    request_id, node_partition,
                )
                if walk not in chunk_walks:
                    continue
                fwd_info = worker_graphs_manager.get_fwd_info(
                    request_id, node_partition,
                )
                engine = self.engine_manager.get_engine(decode_node_name)
                if not engine.check_ready(decode_node_name, request_id, fwd_info):
                    continue
                entry = ReadyNodeEntry(request_id, _wg_id, walk)
                if not self._chunk_entry_passes_gates(
                    worker_graphs_manager, decode_node_name, node_partition, entry,
                ):
                    continue
                clen = fwd_info.step_metadata.get("prefill_chunk_len")
                if self._chunk_over_budget(clen, n_decode, budget_tokens):
                    continue  # over budget with this decode side; keep scanning
                return True
        return False

    def has_eager_fold_opportunity(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        decode_target: tuple[str, str],
    ) -> bool:
        """Read-only peek (MSTAR_EAGER_FOLD): is a BRAND-NEW request's
        first prefill chunk ready that is TOO LARGE for the captured mixed fold
        (C > ``_max_chunk_tokens()``) but within the eager cap
        (<= MSTAR_EAGER_FOLD_MAX_CHUNK), on the in-flight decode's node?

        The worker calls this while a decode spec chain is live (decode rids
        _speculatively_scheduled, hence absent from the ready scan) to decide
        whether to BREAK the chain into the non-speculative path, where
        get_next_batch -> _try_assemble_mixed (armed) assembles decode + this
        large chunk into a thinker_mixed batch that runs EAGER (no captured graph
        matches its token count). Mirrors has_mixed_opportunity's ready scan +
        readiness/rank-0/hold filters, narrowed to:
          * a brand-new request (fwd_index == 0 — first prefill, no KV state), so
            the eager step (slower than a replay) fires at arrival, not mid-decode;
          * a chunk walk allowed by _mixed_chunk_walks() (prefill_text always;
            prefill_vision only when MSTAR_MIXED_BATCH_VISION booted — a vision
            chunk otherwise lacks the deepstack the mixed preprocess needs);
          * capture_cap < C <= eager cap (a chunk that already fits the
            captured fold is left to the normal captured/spec/coadmit path,
            not made eager).

        Returns False when the flag is off (so the worker never breaks a chain for
        it), keeping flag-off byte-identical. Does not pop or mutate queue state;
        the actual assembly + its own gates run later in _try_assemble_mixed."""
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            eager_fold_enabled,
            eager_fold_max_chunk,
        )
        if not eager_fold_enabled():
            return False

        decode_node_name, decode_walk = decode_target
        if decode_walk != self._MIXED_DECODE_WALK:
            return False
        if decode_node_name in self.tp_nodes:
            return False  # TP mixed batching is not yet supported

        eager_cap = eager_fold_max_chunk()
        capture_cap = self._max_chunk_tokens()
        chunk_walks = self._mixed_chunk_walks()
        node_partition = worker_graphs_manager.get_partition_for_node(
            decode_node_name
        )
        now = time.monotonic()
        for _wg_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue
                if request_id in self.pending_removes:
                    continue
                if request_id in self.held_until and self.held_until[request_id] > now:
                    continue
                if decode_node_name not in node_names:
                    continue
                if decode_node_name not in self.tp_rank_zero_nodes:
                    continue
                walk = worker_graphs_manager.get_graph_walk(
                    request_id, node_partition,
                )
                if walk not in chunk_walks:
                    continue
                fwd_info = worker_graphs_manager.get_fwd_info(
                    request_id, node_partition,
                )
                if fwd_info.fwd_index != 0:
                    continue  # not a brand-new request's first prefill
                clen = fwd_info.step_metadata.get("prefill_chunk_len")
                if clen is None:
                    continue  # unchunked — not a foldable chunk row
                clen = int(clen)
                if clen <= capture_cap or clen > eager_cap:
                    continue  # fits captured fold, or beyond the eager bound
                sc = fwd_info.sampling_config.get(decode_node_name)
                if sc is not None and getattr(sc, "repetition_penalty", 1.0) != 1.0:
                    continue
                engine = self.engine_manager.get_engine(decode_node_name)
                if not engine.check_ready(decode_node_name, request_id, fwd_info):
                    continue
                return True
        return False

    def _try_assemble_mixed(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        node_name_to_requests: dict[str, list["ReadyNodeEntry"]],
        max_batch_size: int | None,
    ) -> "ScheduledBatch | None":
        """Assemble a mixed thinker_decode + prefill_text-chunk batch.

        Returns a ScheduledBatch with graph_walk="thinker_mixed" covering all
        ready decode rows on a node plus ONE ready prefill_text chunk row, or
        None when the flag is off / no node has both / a gate fails (in which
        case ``get_next_batch`` falls through to the normal single-walk path).

        Per-request CurrentForwardPassInfo.graph_walk is NOT touched here — each
        popped node keeps its own walk (thinker_decode / prefill_text), which is
        what the submodule's prepare_inputs / postprocess dispatch on. Only the
        batch-level graph_walk is "thinker_mixed" (drives config + runner).

        Gates (any failing → None, fall back to plain alternation):
          * MSTAR_MIXED_BATCH on.
          * A node with BOTH a decode group and >=1 prefill_text chunk row.
          * The chunk row carries chunk metadata (prefill_chunk_len set): a
            full unchunked prefill is not mixed (it would blow the token bucket).
          * Chunk C <= _max_chunk_tokens() so (n + C) fits a captured bucket.
          * Chunk request repetition_penalty == 1.0: a non-last chunk row's
            sampled token is discarded (postprocess drops it), but Sampler.sample
            adds every sampled token to the seen-token mask + advances the RNG
            when any rep-penalty is active, which would corrupt the chunk
            request's penalty state. Decode rows are unaffected.
        """
        from mstar.model.qwen3_omni.qwen3_omni_model import mixed_batch_enabled
        if not mixed_batch_enabled():
            self._eager_fold_armed = False  # one-shot arm can't outlive a no-op
            return None

        # EAGER FOLD (MSTAR_EAGER_FOLD): read-and-CLEAR the one-shot arm
        # the worker set before this get_next_batch. When armed, the chunk gate
        # accepts a chunk up to MSTAR_EAGER_FOLD_MAX_CHUNK (> the captured
        # cap) and we prefer such a large chunk, so the assembled thinker_mixed
        # step's token count matches no captured graph and execute_forward runs it
        # EAGER. Clearing it here guarantees exactly ONE assembly relaxes the cap;
        # every other assembly keeps the captured cap (byte-identical).
        eager_ok = self._eager_fold_armed
        self._eager_fold_armed = False

        # Allow a prefill_vision chunk row alongside prefill_text
        # when the vision flag is on. A vision chunk carries deepstack + the
        # MRoPE side-channel, which only the vision-capable mixed capture can
        # replay; with the flag off the chunk row stays prefill_text.
        chunk_walks = self._mixed_chunk_walks()

        for node_name, entries in node_name_to_requests.items():
            if node_name in self.tp_nodes:
                continue  # TP mixed batching is not yet supported (see __init__ note)
            decode_entries = [
                e for e in entries if e.graph_walk == self._MIXED_DECODE_WALK
            ]
            chunk_entries = [
                e for e in entries if e.graph_walk in chunk_walks
            ]
            if not decode_entries or not chunk_entries:
                continue
            # Occupancy floor (see _mixed_min_decode): a small decode side
            # makes the mixed step poor value AND throttles admission (one
            # chunk per step). Let prefills run standalone instead. Floor is
            # 0 unless MSTAR_MIXED_SINGLE_CHUNK, so default behavior is unchanged.
            if len(decode_entries) < self._mixed_min_decode():
                continue

            node_partition = worker_graphs_manager.get_partition_for_node(node_name)

            # Pick the first chunk entry that passes the per-request gates. When
            # an eager fold is armed, prefer a LARGE chunk (C > the captured cap)
            # so the arm targets the chunk the peek confirmed — a small chunk
            # that also fits the captured fold is left to the normal path. Fall
            # back to first-fit if no large chunk is present (arm then behaves
            # like a normal assembly).
            def _passes(e, node_name=node_name, node_partition=node_partition):
                return self._chunk_entry_passes_gates(
                    worker_graphs_manager, node_name, node_partition, e,
                    eager_ok=eager_ok,
                )

            chunk_entry = None
            if eager_ok:
                for e in chunk_entries:
                    if not _passes(e):
                        continue
                    fwd_info = worker_graphs_manager.get_fwd_info(
                        e.request_id, node_partition,
                    )
                    clen = fwd_info.step_metadata.get("prefill_chunk_len")
                    if clen is not None and int(clen) > self._max_chunk_tokens():
                        chunk_entry = e
                        break
            if chunk_entry is None:
                for e in chunk_entries:
                    if _passes(e):
                        chunk_entry = e
                        break
            if chunk_entry is None:
                continue

            # Cap decode rows so total rows (decode + 1 chunk) fit the padded_bs
            # bucket, honoring any caller max_batch_size too.
            cap = self._MIXED_MAX_DECODE
            if max_batch_size is not None:
                cap = min(cap, max_batch_size - 1)
            if cap < 1:
                continue
            decode_entries = decode_entries[:cap]

            batch = self._pop_mixed_batch(
                worker_graphs_manager, node_name, decode_entries, chunk_entry,
            )
            if batch is not None:
                return batch
        return None

    def _pop_mixed_batch(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        node_name: str,
        decode_entries: list["ReadyNodeEntry"],
        chunk_entry: "ReadyNodeEntry",
    ) -> "ScheduledBatch | None":
        """Pop the selected decode + chunk nodes and build the mixed batch.

        Mirrors the pop loop in get_next_batch. If nothing pops (races with a
        removal), returns None so the caller falls back to the normal path.
        """
        node_partition = worker_graphs_manager.get_partition_for_node(node_name)
        chunk_fwd_info = worker_graphs_manager.get_fwd_info(
            chunk_entry.request_id, node_partition,
        )
        chunk_len = chunk_fwd_info.step_metadata.get("prefill_chunk_len")

        node_objects = {}
        request_to_worker_graph = {}
        for entry in [*decode_entries, chunk_entry]:
            queue = worker_graphs_manager.queues[entry.worker_graph_id]
            popped = queue.pop_ready_nodes(entry.request_id, [node_name])
            if popped:
                assert len(popped) == 1
                node_objects[entry.request_id] = popped[0]
                request_to_worker_graph[entry.request_id] = entry.worker_graph_id

        # Require at least one decode row AND the chunk row to have popped;
        # a lone chunk is just a normal prefill and should go the normal path.
        if chunk_entry.request_id not in node_objects or len(node_objects) < 2:
            # Push back anything already popped so those requests are not
            # stranded off the ready queue (mirrors the requeue in
            # _apply_encoder_step_budget).
            for rid, node in node_objects.items():
                worker_graphs_manager.queues[
                    request_to_worker_graph[rid]
                ].push_back_node(rid, node)
            return None

        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(node_name, "thinker_mixed")] = \
            self.batch_number
        n_decode = len(node_objects) - 1
        self.mixed_batches_assembled += 1
        # INFO (not DEBUG): one line per assembly so runtime evidence that mixed
        # batching is actually firing doesn't require a DEBUG flood. n_decode/C
        # are the batch shape; total = the assembled count so far this worker.
        logger.info(
            "mixed batch: n_decode=%d C=%s node=%s chunk_rid=%s total=%d",
            n_decode, chunk_len, node_name,
            chunk_entry.request_id, self.mixed_batches_assembled,
        )
        return ScheduledBatch(
            node_name=node_name,
            graph_walk="thinker_mixed",
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
        )

    def pop_mixed_chunk_for_spec(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        decode_target: tuple[str, str],
        n_decode: int | None = None,
        budget_tokens: int = 0,
    ) -> "tuple[GraphNode, str, str, int | None] | None":
        """Pop ONLY the mixable chunk node for a mid-chain mixed speculation
        (MSTAR_MIXED_SPEC).

        Unlike ``_pop_mixed_batch``, this does NOT touch the decode rows: during
        a live decode spec chain the decode rids are ``_speculatively_scheduled``
        and absent from the ready queue — they continue via the worker's
        speculation machinery (registry nodes made ready by
        ``ingest_for_speculation``), NOT via the ready queue. The worker builds
        the decode continuation itself and calls this to obtain the single chunk
        row to fold in.

        Scans exactly like ``has_mixed_opportunity`` (same gates, via
        ``_chunk_entry_passes_gates``, and the same ``n_decode`` /
        ``budget_tokens`` V2 budget filter) and pops the FIRST passing chunk node
        on ``decode_target``'s node. Returns
        ``(chunk_node, request_id, worker_graph_id, prefill_chunk_len)`` or None
        when the flag is off / no mixable chunk is ready / the pop races a
        removal. The caller injects the returned node into the speculative
        ScheduledBatch under ``graph_walk="thinker_mixed"``.

        Guarded by ``mixed_batch_spec_enabled`` so the whole path is unreachable
        (and thus byte-identical) unless MSTAR_MIXED_SPEC is on.
        """
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            mixed_batch_spec_enabled,
        )
        if not mixed_batch_spec_enabled():
            return None

        decode_node_name, decode_walk = decode_target
        if decode_walk != self._MIXED_DECODE_WALK:
            return None
        if decode_node_name in self.tp_nodes:
            return None  # TP mixed batching is not yet supported

        chunk_walks = self._mixed_chunk_walks()
        node_partition = worker_graphs_manager.get_partition_for_node(
            decode_node_name
        )
        now = time.monotonic()
        for wg_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            # Snapshot the rids so popping mid-iteration can't mutate the map we
            # are scanning.
            for request_id, node_names in list(ready_map.items()):
                if request_id not in worker_graphs_manager.per_request_info:
                    continue
                if request_id in self.pending_removes:
                    continue
                if request_id in self.held_until and self.held_until[request_id] > now:
                    continue
                if decode_node_name not in node_names:
                    continue
                if decode_node_name not in self.tp_rank_zero_nodes:
                    continue
                walk = worker_graphs_manager.get_graph_walk(
                    request_id, node_partition,
                )
                if walk not in chunk_walks:
                    continue
                fwd_info = worker_graphs_manager.get_fwd_info(
                    request_id, node_partition,
                )
                engine = self.engine_manager.get_engine(decode_node_name)
                if not engine.check_ready(decode_node_name, request_id, fwd_info):
                    continue
                entry = ReadyNodeEntry(request_id, wg_id, walk)
                if not self._chunk_entry_passes_gates(
                    worker_graphs_manager, decode_node_name, node_partition, entry,
                ):
                    continue
                chunk_len = fwd_info.step_metadata.get("prefill_chunk_len")
                # V2 budget gate: mirror has_mixed_opportunity so pop selects the
                # SAME first budget-fitting chunk the peek approved (else a bigger
                # chunk could pop and blow the budget the peek respected).
                if self._chunk_over_budget(chunk_len, n_decode, budget_tokens):
                    continue
                popped = queue.pop_ready_nodes(request_id, [decode_node_name])
                if not popped:
                    continue  # raced a removal; keep scanning
                assert len(popped) == 1
                return popped[0], request_id, wg_id, chunk_len
        return None

    # -----------------------------------------------------------------------
    # MSTAR_ENC_STEP_BUDGET — encoder step budget
    #
    # vLLM-Omni packs ALL images scheduled in a step into ONE varlen eager ViT
    # forward, bounded by a per-step embed-token budget (32768) rather than a
    # request-count grid; over-budget images defer to a later step. M*'s eager
    # encoder already runs one varlen forward over every request that happens
    # to be ready at the ``vision_encoder`` node (the ``MSTAR_VIS_BATCH_SIZES``
    # grid only bounds the *Thinker* ``prefill_vision`` CUDA-graph capture, not
    # the eager encoder node — eager has no capture-shape constraint), but that
    # wave is otherwise unbounded, so a burst of large images could blow encoder
    # memory. This budget makes the eager wave explicit and safe: gather pending
    # image (or audio) encodes in arrival order up to a summed embed-token
    # budget into the single varlen forward, and push the remainder back onto
    # their ready queues so they form the next wave.
    #
    # Read per-call (not cached at import) so the flag can be toggled at
    # runtime without a reboot. Default unset -> ``None`` -> the whole path is a
    # no-op, byte-identical to today. Also skipped when ``MSTAR_MERGED_PREFILL``
    # is on (the merged walk owns encode+prefill as one bs=1 walk; there is no
    # standalone encoder wave to budget).
    # -----------------------------------------------------------------------
    @staticmethod
    def _encoder_step_budget() -> int | None:
        """Return the embed-token budget for the encoder wave, or ``None`` when
        the feature is off (unset / non-positive / merged-prefill active)."""
        raw = os.environ.get("MSTAR_ENC_STEP_BUDGET")
        if not raw:
            return None
        if os.environ.get("MSTAR_MERGED_PREFILL", "0") in ("1", "true", "True"):
            # No standalone encoder wave under merged prefill — fall back to the
            # existing (unbudgeted) behavior so the two flags don't interact.
            return None
        try:
            budget = int(raw)
        except ValueError:
            return None
        return budget if budget > 0 else None

    def _encoder_merge_factor(self, node_name: str) -> int:
        """Spatial-merge factor (patch rows per post-merge embed token) for an
        encoder node, resolved from its submodule and cached. Vision uses
        ``spatial_merge_size**2`` (exposed as ``merge_sq`` on
        ``NativeVisionEncoderSubmodule``); nodes without a merge factor (audio)
        return 1, so the budget then counts raw encoder input rows for them."""
        cached = self._enc_merge_factor.get(node_name)
        if cached is not None:
            return cached
        factor = 1
        try:
            engine = self.engine_manager.get_engine(node_name)
            submodule = getattr(engine, "submodules", {}).get(node_name)
            factor = int(getattr(submodule, "merge_sq", 1)) or 1
        except Exception:  # pragma: no cover - defensive; fall back to raw rows
            factor = 1
        self._enc_merge_factor[node_name] = factor
        return factor

    @staticmethod
    def _node_embed_tokens(node: GraphNode, merge_factor: int) -> int:
        """Embed-token contribution of one encoder request's ready inputs.

        The encoder's varlen sequence length is the row count (dim 0) of its
        largest input edge — ``pixel_values`` (ViT patch rows) for vision,
        ``audio_features`` for audio; the tiny ``image_grid_thw`` /
        ``audio_seqlens`` edges never dominate. Post-merge embed tokens =
        rows // merge_factor. Returns at least 1 so an item always makes
        progress and an edge we can't measure never silently costs 0."""
        rows = 0
        ready_inputs = getattr(node.ready_signals, "ready_inputs", {})
        for edge in ready_inputs.values():
            tinfo = getattr(edge, "tensor_info", None)
            if tinfo and getattr(tinfo[0], "dims", None):
                rows = max(rows, int(tinfo[0].dims[0]))
        return max(1, rows // max(1, merge_factor))

    def _apply_encoder_step_budget(
        self,
        node_name: str,
        entries: list[ReadyNodeEntry],
        node_objects: dict[str, GraphNode],
        request_to_worker_graph: dict[str, str],
        worker_graphs_manager: WorkerGraphsManager,
        budget: int,
    ) -> None:
        """Trim an already-popped encoder wave to ``budget`` embed tokens.

        Walks ``entries`` in order (== arrival order, the order requests were
        registered in the per-request queues), keeping the prefix whose summed
        embed tokens fit the budget (always keeping at least one so the wave
        makes progress even if a single item exceeds the budget), and pushing
        the over-budget tail's nodes back onto their ready queues so they are
        re-formed into the next wave. The deferred set is a contiguous tail, so
        no request ever jumps ahead of an earlier one. Mutates ``node_objects``
        / ``request_to_worker_graph`` in place to drop the deferred requests."""
        merge_factor = self._encoder_merge_factor(node_name)
        used = 0
        cutoff: int | None = None
        for i, entry in enumerate(entries):
            node = node_objects.get(entry.request_id)
            if node is None:
                continue  # raced removal in the pop loop above
            cost = self._node_embed_tokens(node, merge_factor)
            if used > 0 and used + cost > budget:
                cutoff = i  # first item that no longer fits -> defer this + rest
                break
            used += cost
        if cutoff is None:
            return  # whole wave fits the budget; nothing to defer
        for entry in entries[cutoff:]:
            rid = entry.request_id
            if rid not in node_objects:
                continue
            wg_id = request_to_worker_graph.pop(rid, None)
            node = node_objects.pop(rid, None)
            if wg_id is not None and node is not None:
                worker_graphs_manager.queues[wg_id].push_back_node(rid, node)

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
        # Collect all ready (node_name, request_id, graph_walk) tuples
        # grouped by node name
        node_name_to_requests: dict[str, list[ReadyNodeEntry]] = {}
        now = time.monotonic()

        # Expire stale hold entries
        self.held_until = {
            rid: t for rid, t in self.held_until.items() if t > now
        }

        tp_follow_batch = self._try_schedule_tp_follow(worker_graphs_manager)
        if tp_follow_batch is None:
            self.num_consec_tp_follower_batches = 0
        else:
            self.num_consec_tp_follower_batches += 1
            return tp_follow_batch

        for worker_graph_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue  # request was removed between scheduling cycles
                if request_id in self.pending_removes:
                    continue  # remove deferred for in-flight safety; don't start new work
                # Skip requests in OOM backoff
                if request_id in self.held_until:
                    continue
                for sname in node_names:
                    if sname not in self.parallel_leader_nodes:
                        continue # only rank 0 can initiate scheduling!
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

        # Mixed prefill+decode batch (MSTAR_MIXED_BATCH). Before the
        # normal one-walk-per-batch selection, try to assemble a mixed batch
        # (N thinker_decode rows + 1 prefill_text chunk row) so the captured
        # thinker_mixed graph runs both kinds in one forward. Returns None
        # (flag off, no opportunity, or gates fail) → fall through to the
        # normal path, byte-identical to today.
        mixed = self._try_assemble_mixed(
            worker_graphs_manager, node_name_to_requests, max_batch_size,
        )
        if mixed is not None:
            return mixed

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

        # MSTAR_ENC_STEP_BUDGET: cap the eager encoder wave by summed
        # embed tokens and defer the over-budget tail to the next wave. Read the
        # flag per-call so it can be toggled live; no-op / byte-
        # identical when unset (or under MSTAR_MERGED_PREFILL), and only ever
        # touches the standalone encoder nodes.
        if best_node_name in _ENCODER_NODE_NAMES:
            budget = self._encoder_step_budget()
            if budget is not None:
                self._apply_encoder_step_budget(
                    best_node_name, entries, node_objects,
                    request_to_worker_graph, worker_graphs_manager, budget,
                )
                if not node_objects:
                    return None  # defensive; the prefix always keeps >=1

        logger.debug(
            "MicroScheduler scheduling node %s with graph walk %s for %d requests",
            best_node_name, graph_walk, len(node_objects)
        )
        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(
            best_node_name, graph_walk
        )] = self.batch_number

        # MSTAR_ENCODER_ASYNC depth bookkeeping. We count a credit each time
        # an encoder batch is *scheduled* (regardless of batch size, since the
        # depth bound is about pipeline lead, not memory footprint per batch),
        # and release a credit when the matching Thinker prefill walk runs.
        # The Sequential[encoder, Thinker] structure of ``prefill_audio`` /
        # ``prefill_vision`` guarantees the encoder fires exactly once per
        # walk on the Thinker side, so this counter stays bounded as long as
        # the Thinker actually consumes the buffered embeddings. If a request
        # is removed mid-flight (RemoveRequest before the Thinker step ran),
        # the credit is reclaimed by ``release_encoder_async_credit`` from
        # the worker's remove path so the counter cannot drift upward.
        if self.encoder_async_enabled:
            if best_node_name in _ENCODER_NODE_NAMES:
                self.encoder_async_in_flight += 1
            elif (
                best_node_name == "Thinker"
                and graph_walk in _ENCODER_CONSUMING_WALKS
                and self.encoder_async_in_flight > 0
            ):
                # One Thinker prefill_vision/prefill_audio step consumes one
                # buffered batch of encoder outputs. The encoder may have
                # produced N requests' worth of embeds in a single batched
                # call (MSTAR_BATCH_VISION_PREFILL), but the Thinker side
                # consumes them sequentially — one walk per request. The
                # accounting here is per encoder *batch*, so as long as a
                # single Thinker walk releases a credit we'll always have
                # capacity to keep the encoder one batch ahead.
                self.encoder_async_in_flight -= 1

        return ScheduledBatch(
            node_name=best_node_name,
            graph_walk=graph_walk,
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
        )

    def release_encoder_async_credit(self, count: int = 1) -> None:
        """Release ``count`` credits from the encoder-async in-flight counter.

        Called when a request is torn down before its buffered encoder
        output is consumed by the Thinker (e.g. RemoveRequest mid-flight,
        or a hard failure that drops the Thinker prefill step). Without
        this, the counter would drift up and eventually saturate the
        depth budget, silently disabling the async pipeline path.

        No-op when the flag is off or the counter is already at zero.
        """
        if not self.encoder_async_enabled:
            return
        self.encoder_async_in_flight = max(
            0, self.encoder_async_in_flight - count,
        )

    def has_ready_excluding(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        exclude_target: tuple[str, str] | None,
    ) -> bool:
        """Cheap peek: any worker-graph queue ready with a (node, walk) other
        than `exclude_target`? Used by the speculation path to decide whether
        breaking the spec chain for fairness is actually warranted on this
        worker — on single-walk workers (e.g. Orpheus LLM) the answer is
        always False, so speculation can run every iter.

        Does NOT pop or modify queue state. Mirrors the ready-scan in
        get_next_batch but stops at the first match.
        """
        now = time.monotonic()
        # Don't bother expiring held_until here — we only read it; the next
        # get_next_batch call will refresh.
        for _worker_graph_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue
                if request_id in self.held_until and self.held_until[request_id] > now:
                    continue
                for sname in node_names:
                    node_partition = worker_graphs_manager.get_partition_for_node(sname)
                    graph_walk = worker_graphs_manager.get_graph_walk(
                        request_id, node_partition,
                    )
                    if exclude_target is not None and (sname, graph_walk) == exclude_target:
                        continue
                    fwd_info = worker_graphs_manager.get_fwd_info(request_id, node_partition)
                    engine = self.engine_manager.get_engine(sname)
                    if not engine.check_ready(sname, request_id, fwd_info):
                        continue
                    return True
        return False

    def has_new_request_ready(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        exclude_target: tuple[str, str] | None,
    ) -> bool:
        """Cheap peek: is a BRAND-NEW request ready on its FIRST prefill walk?

        Used by MSTAR_ADMIT_FASTPATH: a request that has never run a
        forward pass (``fwd_index == 0`` -> no KV state) and is waiting on a
        prefill walk should become eligible for scheduling at the NEXT decision
        point instead of sitting behind the spec-chain yield gate (which today
        batches admissions at fairness peeks / the consecutive-spec ceiling).
        Returns True iff some ready ``(node, walk)`` other than
        ``exclude_target`` belongs to such a request.

        Same ready-scan + engine readiness / rank-0 / pending-remove filters as
        ``get_next_batch`` (so a True here means ``get_next_batch`` would
        actually schedule that node), narrowed to first-prefill nodes. Does NOT
        pop or modify queue state. The caller only changes WHEN a yield-away
        happens; the admission itself still flows through ``get_next_batch`` (or
        the mixed fold), so all budget/bucket gates stay intact.
        """
        now = time.monotonic()
        for _worker_graph_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue
                if request_id in self.pending_removes:
                    continue  # remove deferred; get_next_batch won't start it
                if request_id in self.held_until and self.held_until[request_id] > now:
                    continue
                for sname in node_names:
                    if sname not in self.tp_rank_zero_nodes:
                        continue  # only rank 0 initiates scheduling
                    node_partition = worker_graphs_manager.get_partition_for_node(sname)
                    graph_walk = worker_graphs_manager.get_graph_walk(
                        request_id, node_partition,
                    )
                    if exclude_target is not None and (sname, graph_walk) == exclude_target:
                        continue
                    # Brand-new request = first prefill walk, no KV state yet.
                    if not graph_walk.startswith("prefill"):
                        continue
                    fwd_info = worker_graphs_manager.get_fwd_info(
                        request_id, node_partition,
                    )
                    if fwd_info.fwd_index != 0:
                        continue
                    engine = self.engine_manager.get_engine(sname)
                    if not engine.check_ready(sname, request_id, fwd_info):
                        continue
                    return True
        return False
