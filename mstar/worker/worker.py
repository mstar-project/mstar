import logging
import os
import sys
import threading
import time
import time as _time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from time import sleep

import torch

from mstar.api_server.request_types import (
    APIServerMessage,
    ResultTensors,
    ResultTensorsBatch,
    SlimResultTokens,
)
from mstar.communication.communicator import CommProtocol, make_communicator
from mstar.communication.event import EventWakeup
from mstar.communication.tensors import NameToTensorList, create_tensor_communication_manager
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.base import EngineType, NodeBatch, NodeOutput
from mstar.engine.kv_store import KVCacheConfig, StoreWritePolicy, TransferEngineInfo
from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.graph_io import format_graph_edge_list
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.model.base import Model, WorkerGraph
from mstar.profile.worker import WorkerProfileInfo
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.utils.ipc_format import (
    AbortRequest,
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    MessageSource,
    NewRequest,
    PackedConductorMessage,
    RemoveRequest,
    ScheduleTPNode,
    SetupDone,
    StopLoops,
    TensorReceived,
    UnpersistTensors,
    WorkerGraphsDone,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.utils.mega_cache import (
    boot_phase,
    load_mega_cache,
    save_mega_cache,
)
from mstar.utils.profiler import range_pop, range_push
from mstar.worker.emit_sidecar import (
    ITEM_INLINE,
    SIDECAR_WALKS,
    SIDECAR_WALKS_I2T_EXTRA,
    SidecarClient,
)
from mstar.worker.engine_manager import EngineManager
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.node_manager_utils import (
    NodeOutputRouting,
    WorkerGraphQueues,
    WorkerGraphsManager,
)

logger = logging.getLogger(__name__)


@dataclass
class PendingBatch:
    batch: ScheduledBatch
    node_batch: NodeBatch
    node_name: str
    partition: str
    graph_walk: str
    future: Future
    speculative_new_iter: bool = False
    loop_name: str = None

@dataclass
class PendingSide:
    """A prefill/encoder batch executing on the side stream + side executor,
    concurrent with the decode chain. Distinct from PendingBatch: it never
    speculates, never loops back, and its postprocess runs opportunistically
    on the main thread. See Worker.run() (MSTAR_SIDE_PREFILL)."""
    batch: ScheduledBatch
    node_batch: NodeBatch
    node_name: str
    partition: str
    graph_walk: str
    future: Future

@dataclass
class Speculation:
    scheduled_batch: ScheduledBatch
    node_batch: NodeBatch
    # ``(name, next_node)`` pairs the spec batch consumed from batch_N's
    # outputs. Two cases:
    #   * Same-node loop-back (AR decode iter K → iter K+1): pairs are
    #     ``{(name, batch_N.node_name) for name in loop_back_outputs}``.
    #   * Forward node A -> node B transition: pairs are
    #     ``{(edge.name, edge.next_node) for edge in batch_N.outputs if
    #     edge.next_node == spec_target.node_name}``.
    # Consumed in ``_thread_outputs_to_speculative`` to splice batch_N's
    # outputs into the spec batch's per-rid input tensors.
    consumed_edges: set[tuple[str, str]]
    continuing_rids: set[str]
    partition: str
    is_new_iter: bool
    is_same_node: bool
    # rid -> edges
    consumed_streaming_edges: dict[str, list[GraphEdge]] = field(default_factory=dict)
    is_yield_away: bool = False
    loop_name: str | None = None
    dropped: set[str] = field(default_factory=set)

    plan_future: Future | None = None


@dataclass(frozen=True)
class PendingLoopStop:
    rid: str
    graph_walk: str
    loop_name: str


class EvictionPolicy(Enum):
    """Strategy for choosing which request to offload to CPU on OOM."""
    LRU = "lru"              # least-recently-used (by execution time)
    MOST_PAGES = "most_pages"  # request holding the most GPU pages


class Worker:
    """
    Real worker that integrates WorkerGraphsManager, EngineManager,
    MicroScheduler, and MooncakeCommunicationManager to execute
    computation via engines.
    """

    def __init__(
        self,
        worker_id: str,
        worker_ids: list[str],
        model: Model,
        my_worker_graphs: list[WorkerGraph],
        kv_config: dict[str, KVCacheConfig],
        model_config: dict,
        all_worker_graph_ids_to_graph_walks: dict[str, set[str]],
        all_worker_graph_ids_to_nodes: dict[str, set[str]],
        all_worker_graph_ids_to_dyn_loops: dict[str, set[str]],
        sharding_config: ShardingConfig,
        parallel_groups: WorkerParallelGroups,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mstar",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: torch.device = torch.device("cuda"),
        enable_nvtx: bool = False,
        enable_prof: bool = False,
        tcp_transfer_device="",
        dist_init_method=None
    ):
        boot_phase("process_start")
        self.worker_id = worker_id
        self.device = device
        self.enable_nvtx = enable_nvtx

        self.enable_prof = enable_prof
        self.profile_info = WorkerProfileInfo()

        # Fast path: send tiny integer new-token emit_to_client tensors inline
        # in the result_tensors message instead of via the SHM tensor
        # transport. Default OFF. See _inline_emit_uuids / _send_outputs.
        self._inline_emit = os.environ.get("MSTAR_INLINE_EMIT", "0") == "1"

        # Fast path: coalesce all qualifying inline emit_to_client messages of
        # one decode step (across every rid in the batch) into ONE
        # result_tensors_batch APIServerMessage, fanned out on the api_server
        # side. Default OFF. Batch implies inline: it is the single flag ruling
        # emission for qualifying edges, so enabling it turns on inline-emit
        # semantics for those edges even if MSTAR_INLINE_EMIT is not set.
        # Non-qualifying edges/messages are unaffected. See _send_outputs /
        # _postprocess_batch.
        self._batch_emit = os.environ.get("MSTAR_BATCH_EMIT", "0") == "1"
        if self._batch_emit:
            self._inline_emit = True

        # MSTAR_WGD_PACK (board #11): coalesce this step's per-rid
        # conductor-bound ConductorMessage sends (WORKER_GRAPHS_DONE) into
        # ONE packed send instead of one send_pyobj/ZMQ frame per rid. A
        # batch step with N concurrently-completing rids (e.g. i2t B32) pays
        # N conductor hops today; this collapses them to 1. Off (default):
        # byte-identical per-rid immediate sends. On: only the WIRE framing
        # changes (N frames -> 1 PackedConductorMessage), never message
        # content or inter-message order — see _send_outputs. The legacy
        # (non-sidecar) send path only; MSTAR_EMIT_SIDECAR-scoped rids keep
        # their own per-rid conductor sends (out of scope here, same
        # technique applies there as a follow-up).
        self._wgd_pack = os.environ.get("MSTAR_WGD_PACK", "0") == "1"

        # Fast path: memoize the per-rid store_and_populate_graph_edges work in
        # _postprocess_batch so a continuing steady-state decode step replays a
        # cached routing-metadata plan (uuid + tensor payload swapped) instead of
        # re-deriving sharding/tp/TensorPointerInfo every step. Default OFF. The
        # replay is invalidated on ANY structural change; the slow path stays
        # byte-identical when off. See _postprocess_batch and
        # TensorCommunicationManager.store_and_populate_graph_edges_fast.
        self._fast_postproc = os.environ.get("MSTAR_FAST_POSTPROC", "0") == "1"

        # W5 mixed-batch DEBUG validation (MSTAR_MIXED_BATCH_ASSERT): assert the
        # spec chain survives a folded mixed step and flag lost fold races. Read
        # once; static for the process.
        self.mixed_batch_assert = (
            os.environ.get("MSTAR_MIXED_BATCH_ASSERT", "").strip().lower()
            in ("1", "true", "yes", "on")
        )

        # W5 fold-rate experiment (MSTAR_MIXED_SINGLE_CHUNK): short spans are
        # single-chunk-mixable AND folds are attempted at every chain step
        # (not just yield boundaries). Read once; static for the process.
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            mixed_budget_tokens as _mbt,
        )
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            mixed_single_chunk_enabled as _msce,
        )
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            mixed_split_attn_enabled as _msae,
        )
        self.mixed_single_chunk = _msce()
        # Static for the process — capture bakes the split layout, so this
        # is read once at init.
        self.mixed_split_attn = _msae()
        # V2 budgeted admission (MSTAR_MIXED_BUDGET_TOKENS, 0=off): fold a ready
        # chunk into the spec chain on EVERY step (subject to the budget + the
        # occupancy floor) instead of only at yield boundaries. Unlike the
        # single-chunk / split flags this bakes nothing into capture (it only
        # changes fold timing).
        self._mixed_budget_tokens = _mbt()
        # MSTAR_COADMIT (fix #1): one-step co-admission of a brand-new request's
        # first chunk into the running decode step (vLLM's unified per-step
        # admission). The effective fold budget under COADMIT
        # (MSTAR_COADMIT_BUDGET_TOKENS, default 32768 = vLLM's ~32k) is HARD-
        # CLAMPED to the largest captured mixed step so a fold can never route to
        # an uncaptured bucket (the UNCAP IMA lesson). Cached here (read once at init).
        self._coadmit_budget_tokens = self._compute_coadmit_budget()
        # Eager-fold peek backoff state (see the fold_probe site).
        self._peek_backoff = 0
        self._peek_skip = 0

        # Diagnostics (MSTAR_WALK_STATS): count executed steps per
        # (node, graph_walk) and log every 200 steps at WARNING (visible under
        # --log-level WARNING). Measures the mixed-batch fold rate on real runs
        # without nsys. Default OFF; a dict lookup + int increment per step when
        # on.
        self._walk_stats = (
            {}
            if os.environ.get("MSTAR_WALK_STATS", "").strip().lower()
            in ("1", "true", "yes", "on")
            else None
        )
        self._walk_stats_step = 0

        # W5-P2 residual (MSTAR_MIXED_PREPLAN): pre-plan a chain-folded
        # thinker_mixed step's packed attention on the plan_executor thread
        # (implies MSTAR_MIXED_SPEC). Read once via the model flag helper so
        # the implication is enforced in one place. Static for the process.
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            mixed_batch_preplan_enabled as _mixed_batch_preplan_enabled,
        )
        self.mixed_batch_preplan = _mixed_batch_preplan_enabled()
        # Counters for the INFO summary of preplanned-vs-inline mixed steps.
        self._mixed_preplan_count = 0
        self._mixed_inline_count = 0

        # MSTAR_DIRECT_FEED: on uniform AR decode speculation (same-walk,
        # same-node loop-back), splice the spec batch's loop-back text_inputs
        # straight from batch_N's batched sampled-tokens tensor
        # (NodeOutput.batched_sampled_tokens) instead of the per-rid tensors
        # threaded out of the registry-backed output map. Default OFF; when
        # off _thread_outputs_to_speculative is byte-identical. The registry
        # store / route path (_postprocess_batch) is UNCHANGED either way — it
        # still populates the loop-back edge's ready-state + tensor_info that
        # the NEXT spec placeholder build (_get_input_tensors) and any fall
        # back to non-speculative decode depend on. See the block in
        # _thread_outputs_to_speculative for the exact safety conditions.
        self._direct_feed = os.environ.get("MSTAR_DIRECT_FEED", "0") == "1"

        # MSTAR_SLIM_EMIT (implies/requires MSTAR_BATCH_EMIT's collector): after
        # the first full ResultTensors per (rid, edge-name) — the api server's
        # template — steady-state steps append SlimResultTokens (values only)
        # to the batch, skipping the per-rid GraphEdge pickle. Default OFF.
        self._slim_emit = (
            os.environ.get("MSTAR_SLIM_EMIT", "0") == "1" and self._batch_emit
        )
        self._slim_emit_sent: set[tuple[str, str]] = set()

        # MSTAR_SLIM_EMIT2 (requires MSTAR_SLIM_EMIT): slim items carry
        # loop_key (plain ints) instead of a pickled NestedLoopIndices, and
        # skip building the unused full ResultTensors on the slim hit path.
        # loop_key is sent ONLY while the step's loop layout still matches the
        # template step's (checked per step below); otherwise the item falls
        # back to the full loop_indices object. Default OFF.
        self._slim_emit2 = (
            os.environ.get("MSTAR_SLIM_EMIT2", "0") == "1" and self._slim_emit
        )
        # (rid, edge_name) -> (loop_name_order list, loop_indices key tuple)
        # captured from the template step's NestedLoopIndices — the same
        # object the api server caches, so key order matches through pickle.
        self._slim_emit_loop_layout: dict[
            tuple[str, str], tuple[list, tuple]
        ] = {}

        # MSTAR_FAST_SEND: trim the per-rid Python around the emit path —
        # compute _inline_emit_uuids once per rid per step (stashed on the
        # routing object by _register_outputs, reused by _send_outputs), skip
        # the empty-set register_for_send call (SHM impl enters a CUDA
        # side-stream context even for a no-op), and write the manager
        # bookkeeping (buffer_new_tokens / buffer_output_signals /
        # register_output_loop_indices) inline against one hoisted
        # per_request_info reference — same effects, no per-call lookups.
        # Default OFF; the off path is byte-identical.
        self._fast_send = os.environ.get("MSTAR_FAST_SEND", "0") == "1"

        # MSTAR_SCHED_PACK: scheduler/loop-shell micro-cuts bundle —
        # (a) the two sum(routing.*.values(), start=[]) flattens per rid per
        # step are computed once in _inline_emit_uuids and stashed on the
        # routing object for _register_outputs (same step, same object;
        # recomputed if absent, never a wrong set); (b) the per-chain-step fairness
        # peek (has_ready_excluding — a full ready-scan) backs off
        # exponentially after consecutive negative peeks (cap
        # MSTAR_SCHED_PACK_PEEK_CAP steps, default 8) — same bounded-
        # staleness argument as the fold-peek backoff: a fairness yield (and
        # therefore a fold boundary) is delayed by at most the cap. Default
        # OFF; both sub-cuts byte-identical when off.
        self._sched_pack = os.environ.get("MSTAR_SCHED_PACK", "0") == "1"
        self._sched_pack_peek_cap = int(
            os.environ.get("MSTAR_SCHED_PACK_PEEK_CAP", "8")
        )

        # N1 (MSTAR_FAST_CHECKSTOP): batched int-compare stop check for
        # uniform thinker_decode steps (one tolist over the pinned buffer
        # instead of per-rid .item()). Default OFF.
        self._fast_checkstop = (
            os.environ.get("MSTAR_FAST_CHECKSTOP", "0") == "1"
        )
        self._thinker_eos_id: int | None = None

        # N1-Talker (MSTAR_FAST_CHECKSTOP_TALKER): the speech-walk analogue of
        # N1. TalkerSubmodule.check_stop does a per-request layer0_codes.item()
        # host read every AR frame; this batches the talker stop condition the
        # same way the thinker one is batched — one flat D→H of just the layer0
        # code per rid + int compares against codec_eos_token_id. Separate flag
        # from MSTAR_FAST_CHECKSTOP and WALK-GATED to talker_decode so thinker
        # paths are untouched: unconditional worker fast paths were measured to tax
        # Talker steps ~17%, so this stays OFF for every non-talker
        # walk regardless of the flag. Default OFF.
        self._fast_checkstop_talker = (
            os.environ.get("MSTAR_FAST_CHECKSTOP_TALKER", "0") == "1"
        )
        self._talker_codec_eos_id: int | None = None

        # (c) MSTAR_CODEC_CHUNK_EMIT: the Talker emits one [num_codes] codec
        # frame per AR step onto the codec_tokens StreamingGraphEdge, so the
        # colocated Code2Wav StreamBuffer takes ~chunk (25) individual puts +
        # id->tensor dict churn per LeftContextChunkPolicy window. When on,
        # local-route codec frames are STAGED and written in one batched put per
        # chunk boundary (StreamBuffer.stage/flush_pending), leaving the buffered
        # item sequence — and every popped window — byte-identical (coalesce at
        # the policy's chunk granularity, so a chunk becomes ready at the same
        # frame count; timing preserved). Edge-gated to policies that opt in via
        # coalesce_size()>1 (only the Talker->Code2Wav codec edge does), so
        # thinker_states/thinker_mask (chunk=1) and other streams are untouched.
        # Default OFF. Only the local (colocated) streaming route is coalesced;
        # remote streaming is unchanged.
        self._codec_chunk_emit = (
            os.environ.get("MSTAR_CODEC_CHUNK_EMIT", "0") == "1"
        )

        # MSTAR_EMIT_SIDECAR: exile emit
        # message construction, the api_server transport, the WGD-feeding
        # accumulators (pending_new_tokens / current_output_chunks /
        # output_loop_indices), and WGD assembly to a per-worker pure-CPU
        # sidecar PROCESS, for requests whose worker graphs on this worker
        # all live on the text walks (SIDECAR_WALKS). Default OFF; when off
        # every code path below is untouched.
        #
        # Read ONCE — static for the process. The sidecar is a spawned
        # process, so this flag is read once at init; A/B via two-server
        # alternation.
        #
        # Requires the winning emit stack: sidecar-side construction is
        # pinned to BATCH+SLIM+SLIM2 semantics, so enabling it without those
        # flags would make the flag-on byte stream diverge from the baseline
        # it must match — refuse loudly instead of diverging quietly.
        # (_slim_emit2 already implies _slim_emit implies _batch_emit.)
        self._emit_sidecar = os.environ.get("MSTAR_EMIT_SIDECAR", "0") == "1"
        if self._emit_sidecar and not self._slim_emit2:
            logger.critical(
                "MSTAR_EMIT_SIDECAR=1 requires MSTAR_BATCH_EMIT, "
                "MSTAR_SLIM_EMIT and MSTAR_SLIM_EMIT2 all on; disabling the "
                "sidecar — worker %s stays on the legacy emit path.",
                worker_id,
            )
            self._emit_sidecar = False

        # MSTAR_SIDECAR_I2T (default off): widen the admission walk-gate to
        # also scope i2t rids (prefill_vision / prefill_multimodal — see
        # emit_sidecar.SIDECAR_WALKS_I2T_EXTRA for why these were excluded
        # initially and why admitting them is safe). Read once, same as
        # MSTAR_EMIT_SIDECAR — the walk set below feeds the ONE admission
        # decision point (_add_new_request) and must not change mid-life for
        # an already-admitted rid. Requires MSTAR_EMIT_SIDECAR itself; with
        # it off there is no sidecar client to scope rids into, so the wider
        # set would be dead weight — refuse loudly instead of silently
        # no-op'ing.
        self._sidecar_i2t = os.environ.get("MSTAR_SIDECAR_I2T", "0") == "1"
        if self._sidecar_i2t and not self._emit_sidecar:
            logger.critical(
                "MSTAR_SIDECAR_I2T=1 requires MSTAR_EMIT_SIDECAR=1; "
                "disabling the i2t walk-gate widening — worker %s scopes "
                "only the base SIDECAR_WALKS set.",
                worker_id,
            )
            self._sidecar_i2t = False
        # The set _add_new_request checks my_walks against. Equal to
        # SIDECAR_WALKS (by identity of contents) when the flag is off, so
        # flag-off admission decisions — and therefore every downstream byte
        # on the wire — are unchanged from before this flag existed.
        self._sidecar_walks = (
            SIDECAR_WALKS | SIDECAR_WALKS_I2T_EXTRA
            if self._sidecar_i2t else SIDECAR_WALKS
        )
        # rids whose emit/WGD path the sidecar owns (decided ONCE at
        # admission, see _add_new_request), and rids stranded by a sidecar
        # failure (client-bound output dropped while the conductor processes
        # our ABORT_REQUEST — their stream cannot be resumed).
        self._sidecar_rids: set[str] = set()
        self._sidecar_condemned: set[str] = set()
        self._sidecar_client: SidecarClient | None = None

        if self._emit_sidecar:
            # Spawned here so the child's import cost (~seconds) hides
            # behind weight load / CUDA-graph capture (~minutes).
            self._sidecar_client = SidecarClient(
                worker_id=worker_id,
                socket_path_prefix=socket_path_prefix,
                log_level=logging.getLevelName(
                    logging.getLogger().getEffectiveLevel()
                ),
            )

        # MSTAR_SIDECAR_CHECKSTOP — deferred-consume of the check_stop D→H.
        # Instead of blocking the main
        # thread on ``side.synchronize()`` (the ~1.1-2.1 ms graph-tail wait),
        # record an event after the side-stream copy, run the
        # cheap per-rid Python that follows, then POLL ``event.query()`` at the
        # consumption point. Ready => consume this step with no wait; not ready
        # => fall back to a blocking wait (counted) so the stop DECISION is
        # always made this step from this step's tokens. The V1 lesson holds:
        # stop-state computation + application stay synchronous and worker-side;
        # only the WAIT moves. max_tokens enforcement is a pure counter and
        # never touches this D→H, so a stalled copy can never cause runaway.
        #
        # This is a GIL-valve removal (Law 2): it converts only if
        # the main thread is still the wall after the emit sidecar. The
        # checkstop_deferred_consume / checkstop_sync_fallback counters make the
        # conversion observable; if fallbacks dominate, the wait was
        # load-bearing graph-tail and this is correctly a no-op, not a
        # regression (the fallback path is byte-identical to flag-off).
        #
        # SCOPE (this build): the deferred-consume + same-step decision above,
        # ONLY. The fuller offload (EOS decided in the sidecar and
        # returned via a StopFeedback message, stops landing 2-3 steps late, a
        # persistent multi-step overstay set) is deliberately NOT built here:
        # it requires a sidecar->worker reverse channel that breaks the one-way
        # data-flow invariant that is the sidecar's central safety property
        # (it produces nothing the worker reads), and under the
        # same-step-decision rule the worker still decides
        # authoritatively so that feedback would be redundant. See the report /
        # DESIGN notes; that path needs GPU shadow validation before it can
        # safely replace the worker's stop authority.
        #
        # Read ONCE (static — it changes the postprocess control
        # flow, not a tunable). Requires CUDA; on a CPU worker it is a no-op
        # because there is no completion event to defer on.
        self._sidecar_checkstop = (
            os.environ.get("MSTAR_SIDECAR_CHECKSTOP", "0") == "1"
        )
        # MSTAR_SKIP_REDUNDANT_SYNC: the blanket completion_event.synchronize()
        # in postprocess (before check_stop) is redundant when SIDECAR_CHECKSTOP
        # is on — the deferred check_stop copy self-gates on completion_event via
        # its own side stream, _await_checkstop polls that copy before the stop
        # decision, and client emit reuses the same gated copy. Worse, the blanket
        # sync BLOCKS the main thread before the per-rid dynamic-loop-iter Python
        # can overlap the D→H copy, defeating the sidecar overlap. Skipping it
        # (only when sidecar_checkstop is on, so a gate exists) lets that overlap
        # happen. Default off = the blanket sync stays (byte-identical).
        self._skip_redundant_sync = (
            self._sidecar_checkstop
            and os.environ.get("MSTAR_SKIP_REDUNDANT_SYNC", "0") == "1"
        )
        # Shadow mode (mandatory before any perf cell): when on, the
        # legacy SYNCHRONOUS check_stop is recomputed alongside the deferred
        # path and the two stop sets are asserted equal, mismatches logged at
        # WARNING with a checkstop_shadow_mismatch counter. Legacy stays
        # authoritative while shadowing, so a bug surfaces as a logged mismatch,
        # never a corrupted stream.
        self._sidecar_checkstop_shadow = (
            self._sidecar_checkstop
            and os.environ.get("MSTAR_SIDECAR_CHECKSTOP_SHADOW", "0") == "1"
        )
        # Reusable side-stream event for the deferred check_stop copy. One event
        # suffices: each step consumes (queries or waits on) it before the next
        # step records it again, so only one copy is ever in flight.
        self._checkstop_event: "torch.cuda.Event | None" = None

        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)

        # ``dist_init_method`` is normally provided by the conductor — it
        # picks a free TCP port at startup so multiple ``mstar`` runs on
        # the same host don't collide. The ``tcp://{hostname}:29500``
        # fallback is for standalone Worker construction (e.g. tests);
        # production paths always pass a value.
        if dist_init_method is None:
            dist_init_method = f"tcp://{hostname}:29500"

        self.parallel_groups = parallel_groups
        self.parallel_groups.init_dist(init_method=dist_init_method)

        # Build node_to_partition mapping from model's partitions and graph walks
        node_to_partition: dict[str, str] = {}
        if model is not None:
            partitions = model.get_partitions()
            walks = model.get_graph_walk_graphs()
            for pdef in partitions:
                for walk_name in pdef.graph_walks:
                    section = walks.get(walk_name)
                    if section:
                        for node_name in section.get_nodes():
                            node_to_partition[node_name] = pdef.name

        self.communicator = make_communicator(
            my_id=worker_id,
            push_ids=worker_ids + ["conductor", "api_server", "api_server_preprocess_worker"],
            ipc_socket_path_prefix=socket_path_prefix,
        )
        self.wakeup_event = EventWakeup()
        self.communicator.register_event_for_poll(self.wakeup_event)

        self.tensor_manager = create_tensor_communication_manager(
            protocol=tensor_comm_protocol,
            my_entity_id=worker_id,
            hostname=hostname,
            device=self.device,
            communicator=self.communicator,
            tcp_transfer_device=tcp_transfer_device,
            enable_prof=enable_prof
        )

        node_names = set()
        for wg in my_worker_graphs:
            node_names.update(wg.section.get_nodes())

        self.engine_manager = EngineManager.build(
            node_names,
            device=device,
            kv_config=kv_config,
            model_config=model_config,
            parallel_groups=self.parallel_groups,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id=worker_id,
                my_session_id=self.tensor_manager.my_session_id,
                transfer_engine=self.tensor_manager.transfer_engine
            ),
            model=model,
            enable_nvtx=self.enable_nvtx,
            enable_prof=self.enable_prof
        )
        # EngineManager.build() has allocated + loaded all submodule weights
        # onto the device; the heavy compile/capture is deferred to warmup_all()
        # in run(). This is the weight-load boundary for boot-phase timing.
        boot_phase("weights_loaded")

        self.worker_graphs_manager = WorkerGraphsManager(
            queues={
                worker_graph.worker_graph_id: WorkerGraphQueues(
                    worker_graph_id=worker_graph.worker_graph_id,
                    graph_walks=worker_graph.graph_walks,
                    worker_graph=worker_graph,
                    per_request_queues={},
                    tensor_manager=self.tensor_manager
                )
                for worker_graph in my_worker_graphs
            },
            per_request_info={},
            all_worker_graph_ids_to_graph_walks=all_worker_graph_ids_to_graph_walks,
            all_worker_graph_ids_to_dyn_loops=all_worker_graph_ids_to_dyn_loops,
            all_worker_graph_ids_to_nodes=all_worker_graph_ids_to_nodes,
            node_to_partition=node_to_partition,
            base_sharding_config=sharding_config,
            worker_id=self.worker_id
        )

        # The lockstep unit for a node is its whole instance: the tensor-parallel
        # row composed with the sequence-parallel column. Exactly one rank per
        # instance — instance rank 0, i.e. rank 0 in BOTH its TP and SP comm
        # groups — leads scheduling and broadcasts ScheduleTPNode to the rest;
        # every other instance rank follows. Keying the leader off the TP rank
        # alone would elect one leader per TP row (e.g. ranks 0 and 2 of a
        # tp2*sp2 instance), racing the followers and desyncing the per-step
        # graph walk.
        self.parallel_leader_nodes = set([
            node for node in node_names
            if self.parallel_groups.get_instance_rank_for_node(node) == 0
        ])

        # v1: disallow multiple lockstep-scheduled nodes in the same worker.
        # A node is lockstep-scheduled when its instance spans more than one rank,
        # i.e. tp_size * sp_size > 1. Pure sequence-parallel nodes (tp_size 1,
        # sp_size > 1) need this too: their attention all-to-all requires the
        # whole instance to step together. Without SP this is just tp_size > 1.
        self.parallel_nodes = set([
            node for node in node_names
            if self.parallel_groups.get_instance_world_size_for_node(node) > 1
        ])
        if len(self.parallel_nodes) > 1:
            raise NotImplementedError(
                f"Multiple parallel nodes {self.parallel_nodes} found in worker "
                f"{worker_id}; current implementation requires at most one "
                "lockstep-parallel node per worker."
            )

        self.is_tp_follower = len(self.parallel_nodes - self.parallel_leader_nodes) > 0

        # Aliases for our MSTAR_MIXED_BATCH / mixed-step scheduling code, which
        # predates upstream's TP->parallel (#154/#176) rename. Same concept:
        # tp_rank_zero_nodes == leader (instance rank 0) node set;
        # tp_nodes == nodes whose parallel instance world_size > 1. In the
        # non-TP/non-SP shipping config both are empty sets.
        self.tp_rank_zero_nodes = self.parallel_leader_nodes
        self.tp_nodes = self.parallel_nodes

        self.scheduler = MicroScheduler(
            self.engine_manager,
            tp_nodes=self.tp_nodes,
            parallel_leader_nodes=self.parallel_leader_nodes,
        )

        # Determine store write policy based on worker graph topology
        node_engine_types = model.get_node_engine_types() if model is not None else {}
        write_policy = self._compute_store_write_policy(
            my_worker_graphs, all_worker_graph_ids_to_graph_walks,
            all_worker_graph_ids_to_nodes,
            node_engine_types=node_engine_types,
        )
        self.engine_manager.set_alloc_write_policies(write_policy)
        logger.info(
            "Worker %s: store write policy = %s", worker_id, write_policy.value
        )

        self._unprocessed_messages = {} # req_id -> messages for requests that are not in the queue

        # CPU offloading: LRU tracking and eviction policy
        self._last_active: dict[tuple[str, str], float] = {}  # (request_id, node_name) -> monotonic timestamp
        self.eviction_policy = EvictionPolicy.LRU

        # Async-scheduling cross-iter state. Initialized here (rather than in
        # run()) because _remove_request — which can be invoked indirectly
        # from _process_messages on any iter — reads/writes them.
        # _in_flight_rids: rids referenced by an in-flight GPU step or its
        #   speculation; REMOVE_REQUEST for these is deferred.
        # _pending_removes: deferred REMOVE_REQUESTs.
        # _pending_loop_stops: loop-stops produced by check_stop in this iter's
        #   postprocess, consumed by next iter's speculation to drop rids whose
        #   loop has ended. Keyed by (rid, graph_walk, loop_name) — see
        #   PendingLoopStop.
        self._in_flight_rids: set[str] = set()
        self._pending_removes: set[str] = set()
        self._pending_loop_stops: set[PendingLoopStop] = set()
        # Let the scheduler see deferred removes so it stops initiating new work
        # for those rids (shared by reference — mutations are visible to both).
        self.scheduler.pending_removes = self._pending_removes

        # Side stream for D→H copies in postprocess (check_stop pre-materialize).
        # The default stream has GPU(N+1) queued behind GPU(N)'s outputs after
        # speculation, so syncing on default would also drain GPU(N+1) and
        # erase the overlap. The side stream waits on
        # ``output.completion_event`` (recorded after GPU(N)) and then runs
        # an isolated D→H, so the main thread only blocks on the copy.
        # Lazy-initialized — workers without CUDA never touch it.
        self._d2h_stream: "torch.cuda.Stream | None" = None
        self._pinned_d2h_buffers: dict[
            tuple[str, torch.dtype, tuple[int, ...]], list[torch.Tensor]
        ] = defaultdict(list)

        # MSTAR_SIDE_PREFILL: run a thinker-prefill (or stateless encoder)
        # batch CONCURRENTLY with the decode chain on a second CUDA stream +
        # second executor thread, instead of yielding the decode chain to run
        # it inline. Default OFF. See run() for the loop restructure and
        # _execute_on_gpu_thread for the stream plumbing.
        #
        # Correctness note on KV: the thinker-prefill hands its first token to
        # decode ASYNCHRONOUSLY through the conductor (persist → conductor
        # rebuilds text_inputs → back as a fresh decode batch a later iter),
        # NOT via a local graph edge. The side batch's postprocess runs on the
        # main thread and fully drains the side stream (the completion_event is
        # recorded ON the side stream, and _prematerialize_for_check_stop
        # side.wait_event(completion_event)+synchronize before the token is
        # even routed). So the prefill's KV pages are durably written long
        # before the decode step for that rid arrives — no cross-stream
        # wait_event is needed on the decode replay path. The single invariant
        # is: record the side batch's completion_event on the side stream (see
        # _execute_on_gpu_thread), so the existing token-materialization wait
        # gates on the right stream.
        # MSTAR_ENC_OVERLAP_V2 (default OFF): the next increment beyond
        # MSTAR_ENCODER_ASYNC. In the encoff/PD split topology the encoders run
        # on rank 0 and the Thinker on rank 1, so ENCODER_ASYNC already stops the
        # encoder from contending with decode. What it does NOT fix: when the
        # encoder's embeds arrive cross-rank at the Thinker, the freshly-ready
        # vision/audio prefill still breaks the decode spec chain (a fairness
        # yield) and runs STANDALONE on the default stream, freezing the
        # in-flight decodes for that step (the residual "prefill freezes decode"
        # serialization). V2 keeps the decode chain alive on encoder completion
        # and routes the just-arrived prefill onto the SIDE stream so it overlaps
        # decode instead of freezing it. It is built entirely on the
        # MSTAR_SIDE_PREFILL substrate (side executor + side stream + reap/drain
        # correctness gate), which V2 therefore activates. Read once here for the
        # substrate; the behavior branch in run() re-reads MSTAR_ENC_OVERLAP_V2
        # per-call so MSTAR_DYNFLAGS can A/B it. Byte-identical when off (the
        # substrate is only built if MSTAR_SIDE_PREFILL was already set).
        self._enc_overlap_v2 = os.environ.get("MSTAR_ENC_OVERLAP_V2", "0") == "1"
        self._side_prefill = (
            os.environ.get("MSTAR_SIDE_PREFILL", "0") == "1"
            or self._enc_overlap_v2
        )
        self._side_stream: "torch.cuda.Stream | None" = None
        # rids currently executing on the side stream — treated as in-flight
        # for deferred-remove safety (see _apply_pending_removes_safe_to_drop).
        self._side_in_flight_rids: set[str] = set()
        # Belt-and-suspenders: the scheduler is single-caller by construction
        # (main thread owns all get_next_batch / has_ready_excluding calls; the
        # side executor only EXECUTES pre-built batches). This lock guards the
        # scan+pop critical section so the invariant is defended even if a
        # future change slips a scheduler call onto another thread.
        self._scheduler_lock = threading.Lock()

        # MSTAR_ENCODER_ASYNC: low-priority side stream for speculative encoder
        # forwards. Lazy-init via ``_get_encoder_async_stream`` (so the flag
        # can be flipped between init and run() without a restart for tests,
        # and so workers without CUDA never allocate a stream they can't
        # back). See ``_execute_on_gpu_thread`` for the dispatch site.
        self._encoder_async_stream: "torch.cuda.Stream | None" = None

        # Streaming buffers: request_id -> edge_name -> list of tensors
        # (Legacy path — kept for models without PartitionTopology)
        self.streaming_buffers: dict[str, dict[str, list[torch.Tensor]]] = {}

        # New streaming path: PartitionTopology + StreamBuffer on consumer worker
        self.partition_topology = model.get_partition_topology() if model else None

        # Determine which partition this worker serves (by checking which node names
        # appear in my_worker_graphs vs the topology connections)
        self._my_consumer_connections = []
        if self.partition_topology:
            my_node_names = set()
            for wg in my_worker_graphs:
                my_node_names.update(wg.section.get_nodes())
            for conn in self.partition_topology.connections:
                # Check if any graph walk graph node for the consumer partition is on this worker
                # by checking if the streaming edge's next_node is in my nodes
                if any(n in my_node_names for n in self._get_node_names_for_partition(conn.to_partition, model)):
                    self._my_consumer_connections.append(conn)

        # Set of edge names that arrive via streaming (used to distinguish
        # streaming inputs from conductor-triggered non-streaming inputs
        # when checking whether a target node is ready for ingestion).
        self._streaming_edge_names: set[str] = {
            conn.edge_name for conn in self._my_consumer_connections
        }

        # Build consumer node cache: edge_name -> next_node name
        self._consumer_node_cache: dict[str, str] = {}
        if self._my_consumer_connections and model:
            walks = model.get_graph_walk_graphs()
            for conn in self._my_consumer_connections:
                for section in walks.values():
                    if hasattr(section, 'input_names') and conn.edge_name in section.input_names:
                        self._consumer_node_cache[conn.edge_name] = section.name

    def _get_node_names_for_partition(self, partition_name: str, model: Model) -> list[str]:
        """Get the node names that belong to a partition."""
        walks = model.get_graph_walk_graphs()
        partitions = model.get_partitions()
        for pdef in partitions:
            if pdef.name == partition_name:
                nodes = set()
                for walk_name in pdef.graph_walks:
                    section = walks.get(walk_name)
                    if section and hasattr(section, 'name'):
                        nodes.add(section.name)
                return list(nodes)
        return []

    def _compute_store_write_policy(
        self,
        my_worker_graphs: list[WorkerGraph],
        all_worker_graph_ids_to_graph_walks: dict[str, set[str]],
        all_worker_graph_ids_to_nodes: dict[str, set[str]],
        node_engine_types: dict[str, EngineType] | None = None,
    ) -> StoreWritePolicy:
        """Determine whether this worker needs to write KV to the mooncake store.

        If this worker handles ALL AR engine graph walks, no other worker
        needs its KV cache — return NEVER. Otherwise return ALWAYS.
        """
        my_ar_walks_nodes: set[str] = set()
        all_ar_walks_nodes: set[str] = set()

        def _uses_kv_cache(node_name: str) -> bool:
            # Local engine instance wins via declared capability; remote
            # nodes fall back to the model's static type map.
            engine = self.engine_manager.node_to_engine.get(node_name)
            if engine is not None:
                return engine.capabilities.requires_kv_cache
            if node_engine_types and node_name in node_engine_types:
                return node_engine_types[node_name] == EngineType.KV_CACHE
            return False

        # Collect this worker's AR graph walks
        for wg in my_worker_graphs:
            for node_name in wg.section.get_nodes():
                if _uses_kv_cache(node_name):
                    my_ar_walks_nodes.update([(walk, node_name) for walk in wg.graph_walks])

        # Collect all workers' AR graph walks
        for wg_id, walks in all_worker_graph_ids_to_graph_walks.items():
            nodes = all_worker_graph_ids_to_nodes.get(wg_id, set())
            for node_name in nodes:
                if _uses_kv_cache(node_name):
                    all_ar_walks_nodes.update([(walk, node_name) for walk in walks])

        if not all_ar_walks_nodes:
            return StoreWritePolicy.NEVER  # no AR engines at all

        if my_ar_walks_nodes == all_ar_walks_nodes:
            logger.info(
                "No LLM disaggregation detected; my_ar_walks_nodes == all_ar_walks_nodes: %s",
                str(my_ar_walks_nodes)
            )
            return StoreWritePolicy.NEVER  # all AR walks on this worker

        return StoreWritePolicy.ALWAYS

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _add_new_request(self, body: NewRequest) -> None:
        logger.debug("Worker %s received request %s", self.worker_id, body.request_id)
        now = _time.monotonic()
        for node_name in self.engine_manager.lru_tracked_nodes():
            self._last_active[(body.request_id, node_name)] = now

        self.worker_graphs_manager.add_request(
            request_id=body.request_id,
            partition_worker_graph_ids=body.partition_worker_graph_ids,
            worker_graph_to_workers=body.worker_graph_to_workers,
            current_fwd_info=body.request_info
        )
        self.engine_manager.add_request(body.request_id)
        self.tensor_manager.register_request(
            body.request_id,
            self.worker_graphs_manager.per_request_info[body.request_id].sharding_config
        )

        # MSTAR_EMIT_SIDECAR: decide sidecar scope ONCE per rid, from the
        # FULL worker_graph_to_workers map (the conductor sends the same map
        # on every partition's NewRequest), so a later partition's add can
        # never flip a rid between owners mid-flight — the split-brain trap
        # this prevents. Scoped ⇔ every walk this rid can EVER run on
        # THIS worker is in self._sidecar_walks (base text walks, plus the
        # i2t vision walks when MSTAR_SIDECAR_I2T=1 — audio/Talker/Code2Wav
        # walks stay exactly flat either way: one set-membership test per
        # admission is the whole tax).
        if (
            self._sidecar_client is not None
            and body.request_id not in self._sidecar_rids
        ):
            my_walks: set[str] = set()
            wg_to_walks = (
                self.worker_graphs_manager.all_worker_graph_ids_to_graph_walks
            )
            for wg_id, wg_workers in body.worker_graph_to_workers.items():
                if self.worker_id in wg_workers:
                    my_walks |= wg_to_walks.get(wg_id, set())
            if my_walks and my_walks <= self._sidecar_walks:
                reg = self._sidecar_client.register_rid(body.request_id)
                if self._sidecar_client.send(reg):
                    self._sidecar_rids.add(body.request_id)
                else:
                    self._disable_sidecar("rid registration send failed")

        # Create StreamBuffers for consumer connections on this worker
        for conn in self._my_consumer_connections:
            req_info = self.worker_graphs_manager.per_request_info[body.request_id]
            req_info.stream_buffers[conn.edge_name] = StreamBuffer(
                request_id=body.request_id,
                edge_name=conn.edge_name,
                from_partition=conn.from_partition,
                policy=conn.chunk_policy_factory(),
            )

        # Start RDMA reads for tensors that have tensor_info
        futures = self.tensor_manager.start_read_tensors(
            body.request_id, body.initial_inputs,
            graph_walk=body.request_info.graph_walk
        )
        self.wakeup_event.register_futures(futures)

        # Signal-only edges (tensor_info is None) can be processed immediately
        signal_only = [
            edge for edge in body.initial_inputs if len(edge.tensor_info) == 0
        ]
        if signal_only:
            self.worker_graphs_manager.process_new_inputs(
                request_id=body.request_id, inputs=signal_only,
                can_buffer=True
            )
        # process messages that may have came in out-of-order
        if body.request_id in self._unprocessed_messages:
            self._process_message_list(self._unprocessed_messages[body.request_id])
            del self._unprocessed_messages[body.request_id]


    def _remove_request(self, body: RemoveRequest) -> None:
        if self.is_tp_follower and body.source not in (MessageSource.TP_RANK_0, MessageSource.SELF):
            return # wait for removal message from TP rank 0 to avoid race conditions

        # Async-scheduling deferral: if this rid is currently held by an
        # in-flight GPU step (or its speculation), tearing down engine /
        # tensor state now would race the GPU thread reading those tensors
        # / KV pages. Queue the remove and apply it once no in-flight step
        # references the rid (see _apply_pending_removes_safe_to_drop in
        # the run loop).
        if body.request_id in getattr(self, "_in_flight_rids", set()):
            self._pending_removes.add(body.request_id)
            return

        # If we are the TP leader for this request, signal the followers to
        # remove it too. Followers defer removal until they get this message
        # (see the guard at the top of this method) so they can't tear down
        # state we're still reading from an in-flight step/speculation.
        cfg = self.worker_graphs_manager.per_request_info.get(body.request_id)
        if cfg is not None:
            followers: set[str] = set()
            for group in cfg.sharding_config.groups:
                # _workers is rank-ordered; index 0 is this worker when we are
                # rank 0. Only real TP groups (tp_size > 1) have followers.
                if group.tp_size > 1 and group._tp_rank == 0:
                    followers.update(group._workers[1:])
            for worker in followers:
                self.communicator.send(
                    worker, msg=WorkerMessage(
                        message_type=WorkerMessageType.REMOVE_REQUEST,
                        body=RemoveRequest(
                            request_id=body.request_id,
                            source=MessageSource.TP_RANK_0,
                        )
                    )
                )

        self.engine_manager.remove_request(body.request_id)
        self.worker_graphs_manager.remove_request(body.request_id)
        self.tensor_manager.cleanup_request(body.request_id)
        self.profile_info.pop_request(body.request_id)
        self.streaming_buffers.pop(body.request_id, None)

        # MSTAR_EMIT_SIDECAR: rid teardown drops the sidecar's per-rid state
        # (accumulators, slim protocol entries, cached edge templates).
        if body.request_id in self._sidecar_rids:
            self._sidecar_rids.discard(body.request_id)
            if self._sidecar_client is not None:
                rec = self._sidecar_client.remove_rid(body.request_id)
                if not self._sidecar_client.send(rec):
                    self._disable_sidecar("rid removal send failed")
        self._sidecar_condemned.discard(body.request_id)

        for node_name in self.engine_manager.lru_tracked_nodes():
            self._last_active.pop((body.request_id, node_name), None)

        # If the removed request had an encoder forward dispatched but the
        # Thinker prefill step that would consume that buffer never ran, the
        # encoder-async depth counter would otherwise leak. Conservatively
        # release one credit on every remove — the helper no-ops when the
        # flag is off, or when the counter is already at zero (so a remove
        # for a request that had no encoder step is harmless).
        self.scheduler.release_encoder_async_credit()

    def _handle_tensor_received(self, body: TensorReceived) -> None:
        """Sender-side cleanup: receiver confirmed RDMA read, free source buffers."""
        for (uuid, ref_cnt) in body.successful_tensors.items():
            self.tensor_manager.dereference(
                body.request_id, uuid, n=ref_cnt
            )

    def _process_new_inputs(self, body: InputSignals) -> None:
        logger.debug(
            "Received new signals %s at worker %s for request %s",
            format_graph_edge_list(body.inputs), self.worker_id, body.request_id
        )
        req_info = self.worker_graphs_manager.per_request_info.get(body.request_id)

        if self.enable_nvtx:
            range_push("process_new_inputs.routing_update")
        # Handle producer_done signal: mark all StreamBuffers for this request as done
        if body.producer_done:
            if req_info:
                for sbuf in req_info.stream_buffers.values():
                    if sbuf.from_partition in body.producer_done:
                        # If we have multiple consumer partitions colocated, we need to signal
                        # the right one
                        sbuf.signal_done()

        # Separate streaming edges — they'll be handled when tensors are ready
        # (streaming edges with tensor_info go through RDMA, handled in _check_ready_tensors)
        non_streaming = [edge for edge in body.inputs if not edge.is_streaming]
        streaming_with_tensors = [edge for edge in body.inputs if edge.is_streaming and edge.tensor_info]

        # Only update fwd_info when there are non-streaming edges (i.e., this is
        # a conductor-triggered forward pass, not just streaming data from another
        # partition). Streaming-only InputSignals must not overwrite the current
        # partition's fwd_info.
        if non_streaming:
            self.worker_graphs_manager.update_request_info(
                body.request_id, current_fwd_info=body.request_info,
                partition_name=body.partition_name
            )

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("process_new_inputs.start_read")
        # Start RDMA reads for non-streaming edges with tensor_info
        futures = self.tensor_manager.start_read_tensors(
            body.request_id, non_streaming,
            graph_walk=body.request_info.graph_walk
        )
        self.wakeup_event.register_futures(futures)
        # Start RDMA reads for streaming edges with tensor_info (will be routed to buffer in _check_ready_tensors)
        if streaming_with_tensors:
            futures = self.tensor_manager.start_read_tensors(
                body.request_id, streaming_with_tensors,
            )
            self.wakeup_event.register_futures(futures)
            for edge in streaming_with_tensors:
                stream_buf = req_info.stream_buffers[edge.name]
                for info in edge.tensor_info:
                    stream_buf.pre_read_register(info.uuid)
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("process_new_inputs.process_inputs")

        # Streaming signal-only edges: nothing to buffer (no tensor data)
        # This shouldn't normally happen for streaming edges

        # Signal-only non-streaming edges can be processed immediately
        signal_only = [edge for edge in non_streaming if len(edge.tensor_info) == 0]
        if signal_only:
            self.worker_graphs_manager.process_new_inputs(
                request_id=body.request_id, inputs=signal_only,
                can_buffer=True
            )
        if self.enable_nvtx:
            range_pop()

    def _unpersist_tensors(self, body: UnpersistTensors):
        for (uuid, ref_cnt) in body.uuid_to_ref_count.items():
            self.tensor_manager.increment_ref(
                body.request_id, uuid, n=ref_cnt
            )
            self.tensor_manager.set_persist(
                body.request_id, uuid, persist=False
            )

    def _stop_loops(self, body: StopLoops):
        if not self.worker_graphs_manager.has_partition(
            body.request_id, body.partition_name
        ):
            return
        fwd_info = self.worker_graphs_manager.get_fwd_info(
            body.request_id, body.partition_name
        )
        loop_names = set()
        for name, stop_time in body.loop_stop_times.items():
            if name not in fwd_info.loop_stop_times or stop_time.label_context_gt(
                fwd_info.loop_stop_times[name], name
            ):
                loop_names.add(name)
            fwd_info.loop_stop_times[name] = stop_time
        if loop_names:
            self.worker_graphs_manager.stop_loops(
                body.request_id, body.partition_name, loop_names
            )

    def _process_message_list(self, messages: list[WorkerMessage]):
        msg_types_needing_active_request = [
            WorkerMessageType.REMOVE_REQUEST,
            WorkerMessageType.INPUT_SIGNALS,
            WorkerMessageType.STOP_LOOPS
        ]
        # Snapshot: a REMOVE handled mid-iteration can re-buffer trailing
        # signals onto this same list, and mutating it while iterating it would
        # never terminate.
        for message in list(messages):
            if (
                message.message_type in msg_types_needing_active_request and \
                message.body.request_id not in self.worker_graphs_manager.per_request_info
            ):
                # got an out-of-order request
                self._unprocessed_messages.setdefault(
                    message.body.request_id, []
                ).append(message)
                continue
            if message.message_type == WorkerMessageType.NEW_REQUEST:
                self._add_new_request(message.body)
            elif message.message_type == WorkerMessageType.REMOVE_REQUEST:
                self._remove_request(message.body)
            elif message.message_type == WorkerMessageType.INPUT_SIGNALS:
                self._process_new_inputs(message.body)
            elif message.message_type == WorkerMessageType.TENSOR_RECEIVED:
                self._handle_tensor_received(message.body)
            elif message.message_type == WorkerMessageType.UNPERSIST_TENSORS:
                self._unpersist_tensors(message.body)
            elif message.message_type == WorkerMessageType.STOP_LOOPS:
                self._stop_loops(message.body)
            elif message.message_type == WorkerMessageType.SCHEDULE_TP:
                self.scheduler.register_tp_follow(message.body)
            elif message.message_type == WorkerMessageType.PACKED:
                # MSTAR_WGD_PACK: unpack and dispatch each contained message
                # through this SAME handler, in order — recursion also
                # replays the out-of-order-request buffering above per inner
                # message, exactly as if each had arrived as its own
                # top-level message. Understood unconditionally regardless
                # of this worker's own flag state (see PackedWorkerMessage).
                self._process_message_list(message.body.messages)

    def _process_messages(self) -> None:
        self._process_message_list(self.communicator.get_all_new_messages())

    # ------------------------------------------------------------------
    # Tensor readiness
    # ------------------------------------------------------------------

    def _route_streaming_tensor(self, request_id: str, edge: GraphEdge) -> None:
        """Route a streaming tensor to its request's StreamBuffer for this edge."""
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
        stream_buf = req_info.stream_buffers[edge.name]

        for info in edge.tensor_info:
            tensor = self.tensor_manager.get_tensor(
                request_id=request_id, uuid=info.uuid,
            )

            stream_buf.put(info.uuid, tensor.clone())
            self.tensor_manager.dereference(request_id, info.uuid)

    def _route_streaming_local_edge(
        self, request_id: str, edge: GraphEdge, stream_buf: StreamBuffer,
    ) -> None:
        """Register + route one local (colocated) streaming edge to its
        StreamBuffer. Shared by the legacy and sidecar send paths.

        Registration order (pre_read_register) is always per-frame, so the
        buffered item order is unchanged. When MSTAR_CODEC_CHUNK_EMIT is on and
        the edge's policy opts into coalescing (coalesce_size>1 — the codec
        edge), frames are STAGED and written in one batched put per chunk
        boundary; otherwise each frame is written immediately (and any staged
        remainder from a just-flipped-off flag is drained first, so nothing is
        stranded)."""
        for info in edge.tensor_info:
            stream_buf.pre_read_register(info.uuid)
        if self._codec_chunk_emit and stream_buf.policy.coalesce_size() > 1:
            self._route_streaming_tensor_coalesced(request_id, edge, stream_buf)
        else:
            if stream_buf.num_pending():
                stream_buf.flush_pending()
            self._route_streaming_tensor(request_id, edge)

    def _route_streaming_tensor_coalesced(
        self, request_id: str, edge: GraphEdge, stream_buf: StreamBuffer,
    ) -> None:
        """(c) MSTAR_CODEC_CHUNK_EMIT: stage arrived frames and write them to
        the buffer in one batched put per ``coalesce_size`` boundary.

        The D->H/get + clone + producer-side dereference still happen per frame
        at arrival (frame lifetime ends here); only the buffer WRITE is
        batched. flush_pending is byte-identical to per-frame puts, so the
        consumer's windows are unchanged. Bumps WALK_STATS codec_chunk_emits
        once per batched flush."""
        size = stream_buf.policy.coalesce_size()
        for info in edge.tensor_info:
            tensor = self.tensor_manager.get_tensor(
                request_id=request_id, uuid=info.uuid,
            )
            stream_buf.stage(info.uuid, tensor.clone())
            self.tensor_manager.dereference(request_id, info.uuid)
            if stream_buf.num_pending() >= size:
                stream_buf.flush_pending()
                self._ws_inc("codec_chunk_emits")

    def _pop_streaming_edge(
        self, sbuf: StreamBuffer, edge_name: str, request_id: str
    ) -> GraphEdge | None:
        consumer_node = self._consumer_node_cache.get(edge_name, "")
        synthetic_edge = sbuf.pop_waiting_edge()
        if synthetic_edge is None and sbuf.has_chunk_ready():
            chunk = sbuf.pop_chunk()
            chunk_tensor = chunk.data.get("data")
            if chunk_tensor is None:
                # Empty chunk — producer done, no more data.
                # Create edge with empty tensor_info.
                synthetic_edge = GraphEdge(
                    next_node=consumer_node,
                    name=edge_name,
                    tensor_info=[],
                    _final_stream_chunk=chunk.is_final,
                )
            else:
                # Normal chunk — store tensor and create edge with tensor_info.
                # Local streaming tensors are routed from outputs that were
                # already gated on the producer completion event before being
                # stored, so avoid a default-stream sync here. If future
                # streaming producers bypass that path, StreamChunk should
                # carry producer events and this call site should wait on
                # those events before storing with skip_cuda_sync=True.
                tensor_infos = self.tensor_manager.store_and_return_tensor_info(
                    request_id, {edge_name: [chunk_tensor]},
                    skip_cuda_sync=True,
                )
                synthetic_edge = GraphEdge(
                    next_node=consumer_node,
                    name=edge_name,
                    tensor_info=tensor_infos.get(edge_name, []),
                    _final_stream_chunk=chunk.is_final,
                )
        return synthetic_edge

    def _poll_stream_buffers_for_speculation(
        self, request_id: str, node_name: str
    ) -> list[GraphEdge]:
        result = []
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
        if req_info is None:
            return []
        for edge_name, sbuf in req_info.stream_buffers.items():
            consumer_node = self._consumer_node_cache.get(edge_name, "")
            if consumer_node != node_name:
                continue
            edge = self._pop_streaming_edge(sbuf, edge_name, request_id)
            if edge is not None:
                result.append(edge)
        return result

    def _return_speculative_streaming_edge(
        self, request_id: str, edge: GraphEdge
    ):
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
        if req_info is None:
            return
        sbuf = req_info.stream_buffers.get(edge.name)
        if sbuf is not None:
            sbuf.store_uningested_edge(edge)

    def _poll_stream_buffers(self) -> None:
        """Check all active StreamBuffers; when a chunk is ready, feed it as a normal input."""
        for request_id, req_info in list(self.worker_graphs_manager.per_request_info.items()):
            for edge_name, sbuf in req_info.stream_buffers.items():
                synthetic_edge = self._pop_streaming_edge(sbuf, edge_name, request_id)

                if synthetic_edge is not None:
                    # Streaming edges go through the same path as regular ones —
                    # ReadySignals.is_ready_for_streaming flips on as soon as
                    # the streaming inputs are the only ones missing. Empty
                    # leftover list means the edge was claimed. The final-chunk
                    # signal rides the synthetic edge to the consuming pass,
                    # which reports the partition done in _postprocess_batch —
                    # NOT here, where an earlier in-flight pass's WGD could read
                    # it before the final output chunk is emitted.
                    leftovers = self.worker_graphs_manager.process_new_streaming_inputs(
                        request_id=request_id, inputs=[synthetic_edge],
                        can_buffer=False # important: only ingest for this loop iter only!
                    )
                    if leftovers:
                        sbuf.store_uningested_edge(synthetic_edge)


    def _check_ready_tensors(self) -> None:
        """Poll for completed RDMA transfers, feed ready graph edges to worker graph queues."""
        self.wakeup_event.drain()
        ready = self.tensor_manager.get_ready_tensors()
        for request_id, edges in ready.items():
            # Separate streaming edges from normal edges
            streaming = [e for e in edges if e.is_streaming]
            normal = [e for e in edges if not e.is_streaming]

            if self.enable_nvtx:
                range_push("check_ready-tensors.route_streaming")
            for edge in streaming:
                self._route_streaming_tensor(request_id, edge)

            if self.enable_nvtx:
                range_pop(synchronize=False)
                range_push("process_new_inputs.process_inputs")

            if normal:
                self.worker_graphs_manager.process_new_inputs(
                    request_id=request_id, inputs=normal,
                    can_buffer=True
                )
            if self.enable_nvtx:
                range_pop(synchronize=False)

    # ------------------------------------------------------------------
    # CPU offloading
    # ------------------------------------------------------------------

    def _try_offload_cold_request(
        self, node_name: str, batch_ids: set[str]
    ) -> str | None:
        """Offload one request's KV pages to CPU using the configured eviction policy.

        Prefers requests outside *batch_ids*. If none exist, falls back to
        picking a victim *within* the batch (the caller should then exclude
        it from execution).

        Returns the victim request_id, or None if offloading wasn't possible.
        """
        engine = self.engine_manager.get_engine(node_name)
        if not engine.capabilities.supports_cpu_offload:
            return None

        candidates_raw = engine.offload_candidates(node_name)
        if not candidates_raw:
            return None

        # Split candidates by whether they belong to the in-flight batch;
        # we prefer evicting requests not currently being executed.
        external: list[tuple[str, int]] = []
        in_batch: list[tuple[str, int]] = []
        for rid, total_pages in candidates_raw:
            if rid in batch_ids:
                in_batch.append((rid, total_pages))
            else:
                external.append((rid, total_pages))

        candidates = external or in_batch
        if not candidates:
            return None

        victim_id = self._select_eviction_victim(node_name, candidates)
        freed = engine.offload_request(node_name, victim_id)
        logger.info(
            "Offloaded request %s to CPU (%d GPU pages freed, "
            "policy=%s, in_batch=%s)",
            victim_id, freed, self.eviction_policy.value,
            victim_id in batch_ids,
        )
        return victim_id if freed > 0 else None

    def _select_eviction_victim(
        self, node_name: str, candidates: list[tuple[str, int]]
    ) -> str:
        """Pick a victim from *candidates* based on ``self.eviction_policy``.

        Each candidate is ``(request_id, total_gpu_pages)``.
        """
        if self.eviction_policy == EvictionPolicy.MOST_PAGES:
            return max(candidates, key=lambda x: x[1])[0]

        # LRU: pick the request with the oldest last_active timestamp.
        # Ties (or missing entries) broken by most pages.
        return min(
            candidates,
            key=lambda x: (
                self._last_active.get((x[0], node_name), 0.0),  # oldest first
                -x[1],                               # then most pages
            ),
        )[0]

    def _try_reload_request(self, node_name: str, request_id: str) -> bool:
        """Reload an offloaded request back to GPU. Returns True if reloaded."""
        engine = self.engine_manager.get_engine(node_name)
        if not engine.is_offloaded(node_name, request_id):
            return False
        if engine.reload_request(node_name, request_id):
            logger.info("Reloaded request %s from CPU to GPU", request_id)
            return True
        logger.debug(
            "Cannot reload request %s yet (insufficient GPU pages)", request_id,
        )
        return False

    # ------------------------------------------------------------------
    # Batch building
    # ------------------------------------------------------------------

    def _build_node_batch(self, batch: ScheduledBatch) -> NodeBatch:
        """Gather input tensors from tensor_manager for all requests in the batch."""
        per_request_inputs: dict[str, NameToTensorList] = {}
        per_request_info: dict[CurrentForwardPassInfo] = {}
        final_stream_rids: set[str] = set()
        batch_partition = self.worker_graphs_manager.get_partition_for_node(batch.node_name)

        for request_id, node in batch.node_objects.items():
            tensors = {}
            ready_inputs = node.ready_signals.ready_inputs
            for input_name, edge in ready_inputs.items():
                tensors[input_name] = [
                    self.tensor_manager.get_tensor(
                        request_id=request_id, uuid=info.uuid
                    ) for info in edge.tensor_info
                ]
                if edge._final_stream_chunk:
                    final_stream_rids.add(request_id)
            per_request_inputs[request_id] = tensors
            per_request_info[request_id] = self.worker_graphs_manager.get_fwd_info(request_id, batch_partition)

        return NodeBatch(
            node_name=batch.node_name,
            graph_walk=batch.graph_walk,
            request_ids=list(batch.node_objects.keys()),
            per_request_input_tensors=per_request_inputs,
            per_request_info=per_request_info,
            final_stream_rids=final_stream_rids,
        )

    def maybe_send_zmq_to_tp_followers(
        self, node_batch: NodeBatch
    ):
        if node_batch.node_name not in self.parallel_nodes or \
                node_batch.node_name not in self.parallel_leader_nodes:
            return
        # this worker is only a part of one TP group for this node,
        # so, we can just look at the sharding_config for the first
        # request to get the relevant workers
        sample_rid = node_batch.request_ids[0]
        cfg = self.worker_graphs_manager.per_request_info[sample_rid]
        workers = cfg.sharding_config.get_sharding_group(
            node_batch.node_name, node_batch.graph_walk
        )._workers[1:]
        for worker in workers:
            self.communicator.send(
                worker, msg=WorkerMessage(
                    message_type=WorkerMessageType.SCHEDULE_TP,
                    body=ScheduleTPNode(
                        node_name=node_batch.node_name,
                        graph_walk=node_batch.graph_walk,
                        request_ids=node_batch.request_ids
                    )
                )
            )

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------
    def _inline_emit_uuids(
        self,
        routing: NodeOutputRouting,
        prematerialized_new_tokens: dict[str, list[int]] | None,
    ) -> set[str]:
        """UUIDs of emit_to_client tensors eligible for the inline fast path.

        A uuid qualifies only when (a) the inline-emit flag is on, (b) its
        edge is a tiny integer new-token tensor already prematerialized to
        CPU ints (present in ``prematerialized_new_tokens``), and (c) the
        uuid is used ONLY by qualifying emit_to_client edges. Condition (c)
        is essential: the same produced tensor's uuid can also feed a
        persist / loop-back (to_workers) / streaming edge, which still need
        the SHM transport — skipping their SHM write would break the
        consumer's read. So we exclude any uuid that appears on a
        non-inline edge.
        """
        if not self._inline_emit or not prematerialized_new_tokens:
            return set()

        inline_candidates: set[str] = set()
        for edge in routing.emit_to_client:
            if edge.name not in prematerialized_new_tokens:
                continue
            # Only integer new-token edges are prematerialized; audio /
            # multimodal edges are never in the prem dict, so they never
            # reach here.
            inline_candidates.update(info.uuid for info in edge.tensor_info)

        if not inline_candidates:
            return set()

        # Any uuid also referenced by a non-inline consumer must keep its
        # SHM write; drop it from the inline set.
        tw_flat = sum(routing.to_workers.values(), start=[])
        stw_flat = sum(routing.streaming_to_workers.values(), start=[])
        if self._sched_pack:
            # MSTAR_SCHED_PACK (a): _register_outputs walks the same two
            # flattened lists for the same routing object later this step —
            # stash them so it doesn't rebuild the concatenations per rid.
            routing.sched_pack_flats = (tw_flat, stw_flat)
        non_inline_uuids: set[str] = set()
        for edge in (
            routing.persist +
            tw_flat +
            stw_flat +
            routing.streaming_local +
            routing.routed_to_this_worker_graph
        ):
            non_inline_uuids.update(info.uuid for info in edge.tensor_info)

        pure_inline = inline_candidates - non_inline_uuids
        return pure_inline

    def _register_outputs(
        self,
        batch: ScheduledBatch,
        routing_per_request: dict[str, NodeOutputRouting],
        prematerialized_per_request: dict[str, dict[str, list[int]] | None] | None = None,
    ):
        """
        For outputs going to other workers: register tensors for RDMA send
        and populate tensor_info on the GraphEdges.
        For outputs staying local: store tensors in tensor_manager.
        Returns the output edges per request (with tensor_info filled in).

        ``prematerialized_per_request`` (optional): per-rid prematerialized
        new-token ints, used to identify emit_to_client uuids that will be
        sent inline (see ``_inline_emit_uuids``). Inline uuids are NOT
        registered for send — no SHM file is written for them.
        """
        for request_id, _node in batch.node_objects.items():
            routing = routing_per_request[request_id]
            prem = (
                (prematerialized_per_request or {}).get(request_id)
            )
            inline_uuids = self._inline_emit_uuids(routing, prem)
            if self._fast_send:
                # MSTAR_FAST_SEND: stash for _send_outputs — it sees the same
                # routing object and the same prem dict later this step, so
                # the set is identical there. Re-deriving it per rid was pure
                # waste, and sharing one set pins the send-side inline
                # decision to the SHM-skip decision made here.
                routing.inline_emit_uuids = inline_uuids
            # MSTAR_SCHED_PACK (a): reuse the flattens stashed by
            # _inline_emit_uuids above (same step, same object). POP, don't
            # get: FAST_ROUTE replays clone the routing object per step and a
            # copied stash could go stale if a later step early-returns from
            # _inline_emit_uuids before re-stashing — consuming it here makes
            # cross-step reuse impossible.
            flats = routing.__dict__.pop("sched_pack_flats", None)
            if self._sched_pack and flats is not None:
                tw_flat, stw_flat = flats
            else:
                tw_flat = sum(routing.to_workers.values(), start=[])
                stw_flat = sum(routing.streaming_to_workers.values(), start=[])
            # upstream (#177) changed register_for_send to take tensor_infos
            # (a list of TensorPointerInfo) instead of a set of uuids. Build
            # the info-by-uuid dict; our inline-skip / fast-send opts below
            # operate on it exactly as they did on the uuid set.
            infos_by_uuid = {}
            for edge in (
                routing.persist +
                tw_flat +
                routing.emit_to_client +
                stw_flat
            ):
                for info in edge.tensor_info:
                    infos_by_uuid[info.uuid] = info
            # Inline-emit uuids skip SHM registration entirely: no file
            # write, no remote fetch, no ack. Their producer-side ref is
            # released locally in _send_outputs instead.
            skip = inline_uuids
            for _u in skip:
                infos_by_uuid.pop(_u, None)
            # MSTAR_FAST_SEND: an empty registration is a no-op (the loop
            # body never runs), but the SHM implementation still enters its
            # CUDA side-stream context per call — and on the steady inline
            # decode path the set is empty for every rid, every step. Skip
            # the call outright.
            if infos_by_uuid or not self._fast_send:
                self.tensor_manager.register_for_send(
                    request_id=request_id,
                    tensor_infos=list(infos_by_uuid.values()),
                    skip_cuda_sync=True,
                )


    def _send_outputs(
        self, request_id: str, outputs: NodeOutputRouting,
        nested_loop_indices: NestedLoopIndices,
        graph_walk: str | None = None,
        partition_name: str | None = None,
        prematerialized_new_tokens: dict[str, list[int]] | None = None,
        node_speculatively_scheduled: bool=False,
        batch_collector: list["ResultTensors"] | None = None,
        wgd_pack_buffer: list[ConductorMessage] | None = None,
    ) -> None:
        """
        Send outputs to other workers and to the conductor.
        Persist signals and new-token counts are buffered and sent together
        with the WORKER_GRAPHS_DONE message to avoid race conditions.

        ``prematerialized_new_tokens`` (optional): `{signal_name: [int, ...]}`
        for this request, where the caller has already done the D→H copy
        for the new-token tensors. When provided, this function skips the
        per-tensor ``.cpu()`` call — meaningful when the caller batched
        multiple requests' new-token transfers into a single D→H to avoid
        N serialized ``cudaMemcpyAsync`` + ``cudaStreamSynchronize`` per
        step.

        ``batch_collector`` (optional): when supplied (MSTAR_BATCH_EMIT), a
        qualifying inline emit_to_client edge's ``ResultTensors`` is appended
        to this list instead of being sent as its own result_tensors message;
        the caller coalesces the whole step's collector into a single
        result_tensors_batch message. ONLY inline-qualifying edges are
        collected — every non-inline emit edge (and every other message here)
        is sent immediately exactly as without the flag. The producer-side
        ref release for inline uuids is unchanged: it happens here per rid.

        ``wgd_pack_buffer`` (optional): when supplied (MSTAR_WGD_PACK), this
        rid's WORKER_GRAPHS_DONE ``ConductorMessage`` is appended to the list
        instead of being sent immediately; the caller flushes the whole
        step's buffer as one packed send to "conductor" after the per-rid
        loop. Peer-worker INPUT_SIGNALS sends below are unaffected.
        """
        if graph_walk is None:
            graph_walk = self.worker_graphs_manager.get_graph_walk(request_id, partition_name)
        # MSTAR_FAST_SEND: hoist the per-request info once. The manager
        # bookkeeping below (buffer_new_tokens / buffer_output_signals /
        # register_output_loop_indices) each re-does the per_request_info
        # lookup behind a method call, per rid per step; on this path their
        # effects are written inline, verbatim, against this one reference.
        # A missing rid leaves fast_info None, so the slow-path manager call
        # runs and raises exactly as without the flag.
        fast_info = (
            self.worker_graphs_manager.per_request_info.get(request_id)
            if self._fast_send else None
        )
        for worker_id, edges in outputs.to_workers.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=edges,
                    request_info=self.worker_graphs_manager.get_fwd_info(request_id, partition_name),
                    partition_name=partition_name
                ),
            )
            self.communicator.send(worker_id, message)

        # Buffer persist signals for this request
        if outputs.persist:
            self.worker_graphs_manager.buffer_persist_signals(
                request_id, outputs.persist
            )

        if outputs.new_token_outputs:
            name_to_count: dict[str, int] = {}
            for signal in outputs.new_token_outputs:
                # upstream (#149) removed the conductor_new_token path: the
                # conductor now needs only per-signal token COUNTS (numel, no
                # D->H sync), not the materialized values. This supersedes our
                # prematerialized_new_tokens D->H-avoidance for THIS path (numel
                # needs no copy at all). Actual token VALUES still reach the
                # client via the emit_to_client / emit-sidecar path below. The
                # prematerialized_new_tokens param is retained: _inline_emit_uuids
                # still consumes it further down.
                if signal.name in name_to_count:
                    continue  # don't double-count new tokens
                count = 0
                for tensor_info in signal.tensor_info:
                    tensor = self.tensor_manager.get_tensor(
                        request_id=request_id,
                        uuid=tensor_info.uuid,
                    )
                    count += tensor.numel()
                name_to_count[signal.name] = count
            self.worker_graphs_manager.buffer_new_token_counts(
                request_id, name_to_count
            )

        if outputs.emit_to_client:
            if fast_info is not None:
                # Inline of worker_graphs_manager.buffer_output_signals
                # (load-bearing per-step accumulation, flushed on WGD).
                fast_info.current_output_chunks += [
                    signal.name for signal in outputs.emit_to_client
                ]
            else:
                self.worker_graphs_manager.buffer_output_signals(
                    request_id, outputs.emit_to_client
                )
            # MSTAR_FAST_SEND: _register_outputs already derived this set
            # from the same (routing, prem) pair this step and stashed it on
            # the routing object — reuse it. Recompute only when the stash is
            # missing (flag off, or flipped between register and send).
            inline_uuids = outputs.inline_emit_uuids
            if not self._fast_send or inline_uuids is None:
                inline_uuids = self._inline_emit_uuids(
                    outputs, prematerialized_new_tokens
                )
            # uuids we release locally, weighted by how many emit tensor_info
            # entries reference each (mirrors the per-tensor_info ack count
            # the data worker would have sent via TENSOR_RECEIVED).
            local_release: dict[str, int] = {}
            for graph_edge in outputs.emit_to_client:
                if fast_info is not None:
                    # Inline of
                    # worker_graphs_manager.register_output_loop_indices.
                    fast_info.output_loop_indices[graph_edge.name] = (
                        nested_loop_indices
                    )
                else:
                    self.worker_graphs_manager.register_output_loop_indices(
                        request_id=request_id, loop_indices=nested_loop_indices,
                        output_name=graph_edge.name
                    )
                edge_inline = self._inline_emit and bool(graph_edge.tensor_info) and all(
                    info.uuid in inline_uuids for info in graph_edge.tensor_info
                )
                tkey = (request_id, graph_edge.name)
                slim_hit = (
                    self._slim_emit and batch_collector is not None
                    and edge_inline and tkey in self._slim_emit_sent
                )
                metadata: dict = {}
                inline_vals: list | None = None
                if edge_inline:
                    # Carry the token values inline; the consumer skips the
                    # SHM fetch entirely. dtype/shape come from tensor_info
                    # on the (still-attached) graph_edge, so the consumer
                    # reconstructs a byte-identical tensor for postprocess.
                    inline_vals = prematerialized_new_tokens[graph_edge.name]
                    # MSTAR_FAST_SEND: on the slim steady path the metadata
                    # dict's only consumer is the full ResultTensors, which
                    # SLIM_EMIT2 skips below — the slim item carries
                    # inline_vals directly. Don't build the two dicts.
                    if not (self._fast_send and self._slim_emit2 and slim_hit):
                        metadata = {
                            "inline_values": {graph_edge.name: inline_vals}
                        }
                    for info in graph_edge.tensor_info:
                        local_release[info.uuid] = local_release.get(info.uuid, 0) + 1
                # MSTAR_SLIM_EMIT2: on the slim steady path the full
                # ResultTensors below is provably unused (the hit branch
                # appends a SlimResultTokens and the immediate-send else is
                # unreachable when batch_collector/edge_inline hold) — skip
                # building it.
                if self._slim_emit2 and slim_hit:
                    result_tensors = None
                else:
                    result_tensors = ResultTensors(
                        request_id=request_id,
                        modality=graph_edge.output_modality,
                        graph_edge=graph_edge,
                        loop_indices=nested_loop_indices,
                        metadata=metadata,
                    )
                if batch_collector is not None and edge_inline:
                    # Coalesced path: defer to a single result_tensors_batch
                    # message built by the caller after the rid loop. Only
                    # inline edges are collected — non-inline edges below still
                    # send their own message, byte-identical to the flag-off
                    # path.
                    #
                    # MSTAR_SLIM_EMIT: after the first full item for this
                    # (rid, name) — the api server's template — send only the
                    # token values. Skips pickling a GraphEdge per rid per
                    # step (the bulk of send_outputs' main-thread cost).
                    if self._slim_emit:
                        if slim_hit:
                            # MSTAR_SLIM_EMIT2: carry the loop state as plain
                            # ints when the step's layout (loop_name_order
                            # content + loop_indices key ORDER) still matches
                            # the template step's — the consumer's rebuild
                            # from its cached template is then value-identical
                            # (verified round-trip incl. max /
                            # label_context_gt). Any drift: full object.
                            loop_key = None
                            if self._slim_emit2:
                                layout = self._slim_emit_loop_layout.get(tkey)
                                if (
                                    layout is not None
                                    and nested_loop_indices.loop_name_order
                                    == layout[0]
                                    and tuple(
                                        nested_loop_indices.loop_indices.keys()
                                    ) == layout[1]
                                ):
                                    loop_key = (
                                        nested_loop_indices.wg_fwd_pass_idx,
                                        *nested_loop_indices.loop_indices.values(),
                                    )
                            # inline_vals is always set here: slim_hit
                            # implies edge_inline. Same list object the
                            # metadata dict carried before the FAST_SEND
                            # skip, so the pickled payload is unchanged.
                            batch_collector.append(SlimResultTokens(
                                request_id=request_id,
                                name=graph_edge.name,
                                values=inline_vals,
                                loop_indices=(
                                    None if loop_key is not None
                                    else nested_loop_indices
                                ),
                                loop_key=loop_key,
                            ))
                        else:
                            self._slim_emit_sent.add(tkey)
                            if self._slim_emit2:
                                # Capture the template's loop layout (copies:
                                # the NLI is fresh per step but not owned).
                                self._slim_emit_loop_layout[tkey] = (
                                    list(nested_loop_indices.loop_name_order),
                                    tuple(nested_loop_indices.loop_indices.keys()),
                                )
                            batch_collector.append(result_tensors)
                    else:
                        batch_collector.append(result_tensors)
                else:
                    message = APIServerMessage(
                        message_type="result_tensors",
                        body=result_tensors,
                    )
                    self.communicator.send("api_server", message)

            # Release the producer-side ref for inline uuids now: no
            # TENSOR_RECEIVED ack will ever arrive for them (they were never
            # registered for send in _register_outputs), so this stands in
            # for the ack's dereference and prevents a tensor_store leak.
            for uuid, n in local_release.items():
                self.tensor_manager.dereference(request_id, uuid, n=n)

        # Handle streaming edges
        # Local streaming: route to StreamBuffer
        # (fast_info is this same object when MSTAR_FAST_SEND found the rid;
        # the [] lookup keeps the missing-rid KeyError behavior otherwise.)
        req_info = (
            fast_info if fast_info is not None
            else self.worker_graphs_manager.per_request_info[request_id]
        )
        for edge in outputs.streaming_local:
            stream_buf = req_info.stream_buffers[edge.name]
            self._route_streaming_local_edge(request_id, edge, stream_buf)

        # Remote streaming: send to destination workers
        for worker_id, edges in outputs.streaming_to_workers.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=edges,
                    request_info=self.worker_graphs_manager.get_fwd_info(request_id, partition_name),
                    partition_name=partition_name
                ),
            )
            self.communicator.send(worker_id, message)
        if outputs.completed_worker_graph_ids:
            fwd_info = self.worker_graphs_manager.get_fwd_info(request_id, partition_name)
            if partition_name is None:
                partition_name = getattr(fwd_info, 'partition_name', 'default')
            req_info = self.worker_graphs_manager.per_request_info.get(request_id)
            p_done = (
                req_info.per_partition_info[partition_name].stream_partition_done \
                    and not node_speculatively_scheduled
            ) if req_info else False

            # Collect stream consumption info
            stream_consumed = {}
            if req_info:
                for edge_name, sbuf in req_info.stream_buffers.items():
                    stream_consumed[edge_name] = sbuf._consumed

            message = ConductorMessage(
                message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
                body=WorkerGraphsDone(
                    request_id=request_id,
                    worker_graph_ids=outputs.completed_worker_graph_ids,
                    is_first_tp_rank=outputs.is_first_tp_rank,
                    persist_signals=self.worker_graphs_manager.flush_persist_signals(request_id),
                    new_token_counts=self.worker_graphs_manager.flush_new_token_counts(request_id),
                    output_signal_names=self.worker_graphs_manager.flush_output_signals(request_id),
                    per_label_seq_info=self.worker_graphs_manager.get_seq_info(request_id, partition_name),
                    partition_name=partition_name,
                    partition_done=p_done,
                    stream_tokens_consumed=stream_consumed,
                    output_loop_indices=self.worker_graphs_manager.get_output_loop_indices(request_id),
                    graph_timings=self.profile_info.per_rid_graph_timings.get(request_id, {}),
                    rx_info=self.tensor_manager.get_rx_info(request_id),
                    tx_info=self.tensor_manager.get_tx_info(request_id),
                ),
            )
            if wgd_pack_buffer is not None:
                wgd_pack_buffer.append(message)
            else:
                self.communicator.send("conductor", message)

    def _send_outputs_sidecar(
        self, request_id: str, outputs: NodeOutputRouting,
        nested_loop_indices: NestedLoopIndices,
        partition_name: str | None = None,
        prematerialized_new_tokens: dict[str, list[int]] | None = None,
        node_speculatively_scheduled: bool = False,
        build_record: bool = True,
    ) -> tuple | None:
        """MSTAR_EMIT_SIDECAR twin of ``_send_outputs`` for sidecar-scoped
        rids. Returns this rid's entry for the step record (or None).

        Every worker-side effect of ``_send_outputs`` is kept byte-identical
        — peer-worker INPUT_SIGNALS sends, persist buffering, streaming
        routing, the inline-emit producer-ref release — and every
        client-bound effect becomes a record field:

        - ``buffer_new_tokens``            -> entry new-token field
        - ``buffer_output_signals``        -> derived by the sidecar from
                                              item order
        - ``register_output_loop_indices`` -> derived from each item's
                                              loop ints
        - emit construction + api_server send -> sidecar (items)
        - WGD flush/assembly + conductor send -> sidecar (boundary field)

        The worker NEVER writes pending_new_tokens / current_output_chunks /
        output_loop_indices for a scoped rid — steady, boundary and
        non-inline paths alike ride the record, so WGD is assembled from a
        single owner (the hardest invariant here: WORKER_GRAPHS_DONE must
        have exactly one assembler, or two producers race the same rid's
        accumulator state).

        ``build_record=False`` (rid condemned by a sidecar failure): perform
        only the worker-side effects and return None — the client-bound
        stream is intentionally dropped while the rid is failed fast via
        ABORT_REQUEST (a resumed stream would need the dead sidecar's
        accumulator state).

        Record cheapness: items are tuples of ints and interned
        indices; the only non-scalar payloads are the token lists (the SAME
        list objects as the new-token field, so the record pickle memoizes
        them) and a GraphEdge at boundary rate (first inline template per
        (rid, name), or a non-inline edge whose fresh tensor_info the
        consumer must fetch via SHM — the record just moves that pickle one
        hop).
        """
        client = self._sidecar_client
        # Peer-worker routing stays on the worker: it feeds next-step
        # readiness on other workers (scheduler contract).
        for worker_id, edges in outputs.to_workers.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=edges,
                    request_info=self.worker_graphs_manager.get_fwd_info(request_id, partition_name),
                    partition_name=partition_name
                ),
            )
            self.communicator.send(worker_id, message)

        # Persist accumulation stays worker-side (the sidecar owns only the
        # three WGD accumulators); the flushed dict ships in the boundary field
        # below so the sidecar's WGD carries it exactly as legacy's did.
        if outputs.persist:
            self.worker_graphs_manager.buffer_persist_signals(
                request_id, outputs.persist
            )

        new_tokens_field: list | None = None
        if outputs.new_token_outputs:
            # Same name-dedup + D2H fallback as the legacy buffer_new_tokens
            # feeding — the fallback ``.cpu()`` is a CUDA read and can only
            # live on the worker.
            name_to_new_token: dict = {}
            for signal in outputs.new_token_outputs:
                if signal.name in name_to_new_token:
                    continue # don't double-count new tokens
                if (
                    prematerialized_new_tokens is not None
                    and signal.name in prematerialized_new_tokens
                ):
                    new_tokens = prematerialized_new_tokens[signal.name]
                else:
                    new_tokens = []  # list[int]
                    for tensor_info in signal.tensor_info:
                        tensor = self.tensor_manager.get_tensor(
                            request_id=request_id,
                            uuid=tensor_info.uuid
                        )
                        new_tokens.extend(tensor.cpu().numpy().tolist())
                name_to_new_token[signal.name] = new_tokens
            if build_record and name_to_new_token:
                new_tokens_field = [
                    (client.name_idx(name), toks)
                    for name, toks in name_to_new_token.items()
                ]

        items: list | None = None
        if outputs.emit_to_client:
            # Same inline set as _register_outputs' SHM-skip decision (the
            # FAST_SEND stash when present) — the record's inline flag is
            # DERIVED from the worker's tensor-lifecycle decision.
            inline_uuids = outputs.inline_emit_uuids
            if not self._fast_send or inline_uuids is None:
                inline_uuids = self._inline_emit_uuids(
                    outputs, prematerialized_new_tokens
                )
            if build_record:
                items = []
                rid_idx = client.rid_idx(request_id)
                layout_idx = client.layout_idx(nested_loop_indices)
                wg_fwd = nested_loop_indices.wg_fwd_pass_idx
                loop_vals = tuple(nested_loop_indices.loop_indices.values())
            local_release: dict[str, int] = {}
            for graph_edge in outputs.emit_to_client:
                edge_inline = self._inline_emit and bool(graph_edge.tensor_info) and all(
                    info.uuid in inline_uuids for info in graph_edge.tensor_info
                )
                inline_vals: list | None = None
                if edge_inline:
                    inline_vals = prematerialized_new_tokens[graph_edge.name]
                    for info in graph_edge.tensor_info:
                        local_release[info.uuid] = local_release.get(info.uuid, 0) + 1
                if build_record:
                    name_idx = client.name_idx(graph_edge.name)
                    items.append((
                        name_idx,
                        ITEM_INLINE if edge_inline else 0,
                        inline_vals,
                        layout_idx, wg_fwd, loop_vals,
                        graph_edge if client.ship_edge(
                            rid_idx, name_idx, edge_inline
                        ) else None,
                    ))
            # Producer-side ref release for inline uuids, identical to
            # legacy — tensor lifecycle never leaves the worker.
            for uuid, n in local_release.items():
                self.tensor_manager.dereference(request_id, uuid, n=n)

        # Streaming stays worker-side (audio rides these; on the scoped text
        # walks both are empty in steady state).
        req_info = self.worker_graphs_manager.per_request_info[request_id]
        for edge in outputs.streaming_local:
            stream_buf = req_info.stream_buffers[edge.name]
            self._route_streaming_local_edge(request_id, edge, stream_buf)
        for worker_id, edges in outputs.streaming_to_workers.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=edges,
                    request_info=self.worker_graphs_manager.get_fwd_info(request_id, partition_name),
                    partition_name=partition_name
                ),
            )
            self.communicator.send(worker_id, message)

        boundary: tuple | None = None
        if outputs.completed_worker_graph_ids:
            fwd_info = self.worker_graphs_manager.get_fwd_info(request_id, partition_name)
            if partition_name is None:
                partition_name = getattr(fwd_info, 'partition_name', 'default')
            p_done = (
                req_info.per_partition_info[partition_name].stream_partition_done
                and not node_speculatively_scheduled
            )
            stream_consumed = {}
            for edge_name, sbuf in req_info.stream_buffers.items():
                stream_consumed[edge_name] = sbuf._consumed
            if build_record:
                # The worker-only WGD fields (boundary record).
                # The sidecar merges them with ITS accumulators and sends
                # the WORKER_GRAPHS_DONE (the conductor tolerates late WGD:
                # conductor.py's unknown-rid guard).
                boundary = (
                    outputs.completed_worker_graph_ids,
                    outputs.is_first_tp_rank,
                    self.worker_graphs_manager.flush_persist_signals(request_id),
                    self.worker_graphs_manager.get_seq_info(request_id, partition_name),
                    partition_name,
                    p_done,
                    stream_consumed,
                    self.profile_info.per_rid_graph_timings.get(request_id, {}),
                    self.tensor_manager.get_rx_info(request_id),
                    self.tensor_manager.get_tx_info(request_id),
                )

        if not build_record or (
            new_tokens_field is None and not items and boundary is None
        ):
            return None
        return (client.rid_idx(request_id), new_tokens_field, items, boundary)

    def _disable_sidecar(self, reason: str) -> None:
        """Permanent fallback: the sidecar died or its
        queue hit HWM. A dead sidecar must fail fast, never hang the client.
        Legacy path for all new work; in-flight sidecar-owned
        rids have emit/WGD state stranded in the dead process and cannot be
        reconstructed — fail them fast via the conductor's abort path rather
        than letting them ride the api_server's 15 s TTL. No
        restart-and-resume: resuming means replaying accumulator state,
        which is the split-brain trap again."""
        client = self._sidecar_client
        if client is None:
            return
        self._sidecar_client = None
        logger.critical(
            "Worker %s: emit sidecar disabled (%s; hwm_trips=%d, "
            "records_sent=%d) — failing %d in-flight sidecar-owned "
            "request(s) fast and falling back to the legacy emit path",
            self.worker_id, reason, client.hwm_trips, client.records_sent,
            len(self._sidecar_rids),
        )
        if client.hwm_trips:
            # Mechanism counter rides WALK_STATS — no counter, no verdict.
            self._ws_inc("_sidecar_hwm_trips")
        for rid in self._sidecar_rids:
            self.communicator.send("conductor", ConductorMessage(
                message_type=ConductorMessageType.ABORT_REQUEST,
                body=AbortRequest(request_id=rid),
            ))
        self._sidecar_condemned |= self._sidecar_rids
        self._sidecar_rids.clear()
        client.shutdown()

    # ------------------------------------------------------------------
    # Main loop — async scheduling
    #
    # Pipeline shape:
    #   iter K (main thread):                          GPU thread
    #     CPU preamble  ───────────────► overlaps with execute_batch(N)
    #     speculate + build N+1
    #     await GPU(N).future Python return
    #     thread N's outputs → N+1's loop-back inputs
    #     submit GPU(N+1) ───────────────► execute_batch(N+1)
    #     _postprocess_batch(N) ─────────► overlap with GPU(N+1)
    #
    # Speculation scope (currently): AR engine only, intra-worker, 1-deep,
    # for rids whose loop is still continuing.
    # ------------------------------------------------------------------

    def _compute_coadmit_budget(self) -> int:
        """MSTAR_COADMIT_BUDGET_TOKENS clamped to the largest captured mixed step.

        The requested budget (default 32768, matching vLLM's ~32k unified per-
        step token budget) is HARD-CLAMPED to ``bs + max_captured_chunk`` — the
        largest mixed step the CUDA-graph grid actually captured. The chunk-size
        cap (``MicroScheduler._max_chunk_tokens()``, 512 by default, wider under
        MSTAR_MIXED_CHUNK_SIZES — see the s4 grid-growth report) is the real
        ceiling on a foldable chunk; this clamp lands at 32 + that cap, so the
        budget is never the binding constraint for any chunk the capture already
        allows AND never admits a fold whose bucket wasn't captured (the UNCAP
        IMA lesson). It is thus never MORE restrictive than the V2 budget for a
        valid fold.

        Previously this read a hardcoded ``_MIXED_MAX_CHUNK_TOKENS = 512`` class
        CONSTANT, which did NOT track grid growth despite the docstring's claim
        (the constant just sat there at 512 regardless of what got captured) —
        fixed by calling ``_max_chunk_tokens()``, which resolves the same
        MSTAR_MIXED_CHUNK_SIZES grid ThinkerSubmodule captured at boot, so this
        now actually auto-widens the day the capture grid grows.
        """
        raw = os.environ.get("MSTAR_COADMIT_BUDGET_TOKENS")
        try:
            want = int(raw.strip()) if raw is not None else 32768
        except (ValueError, AttributeError):
            want = 32768
        # bs = _MIXED_MAX_DECODE + 1 (31 decode rows + 1 chunk row = padded 32).
        # Reference the class (not self.scheduler) so this is safe to call from
        # __init__ before the scheduler instance is built; _max_chunk_tokens()
        # is a classmethod for exactly this reason.
        max_captured = (
            MicroScheduler._MIXED_MAX_DECODE + 1
            + MicroScheduler._max_chunk_tokens()
        )
        return min(want, max_captured)


    def _ws_inc(self, key: str) -> None:
        """MSTAR_WALK_STATS: bump a named diagnostic counter (no-op when off).
        Counters ride the same dict as the per-walk step counts and are logged
        by the same every-200-steps WARNING line."""
        if self._walk_stats is not None:
            self._walk_stats[key] = self._walk_stats.get(key, 0) + 1

    def _pre_plan_for_speculative_batch(
        self,
        engine,
        spec_node_batch: NodeBatch,
        prev_advance_event: "threading.Event | None",
    ) -> bool:
        """Dispatch entry point on the plan_executor for the speculative batch.

        Waits on ``prev_advance_event`` — set by the GPU thread RIGHT AFTER
        ``advance_seq_lens(prev)`` runs (~tens of µs into prev replay) —
        rather than on the full prev_future. This is the key to overlap:
        plan(N+1) starts as soon as alloc_manager state is post-(N), which
        is well before replay(N)'s GPU work finishes. plan(N+1) runs
        concurrent with the rest of replay(N)'s GPU kernels on the disjoint
        slot's wrapper buffers. await_plan on the GPU thread should drop
        to ~0 because plan(N+1) has finished long before replay(N+1)
        begins.

        Returns True if pre-planning was applied; False otherwise — the
        caller submits the spec batch with plan_future regardless, so a
        False return means the GPU thread will plan inline (no skip).
        """
        try:
            if prev_advance_event is not None:
                # Safety timeout — should fire well within 100ms in normal
                # operation. If it doesn't (e.g., GPU thread crashed),
                # bail out rather than block plan_executor forever.
                if not prev_advance_event.wait(timeout=10.0):
                    logger.warning(
                        "Worker %s: plan_executor timed out waiting for "
                        "prev advance_event; skipping pre-plan",
                        self.worker_id,
                    )
                    self._reset_skip_plan_flags(spec_node_batch)
                    return False
            return engine.pre_plan_for_batch(
                spec_node_batch,
                prev_completion_event=None,
            )
        except Exception:
            logger.exception("Worker %s: plan_executor pre-plan failed", self.worker_id)
            self._reset_skip_plan_flags(spec_node_batch)
            return False

    def _reset_skip_plan_flags(self, spec_node_batch: NodeBatch) -> None:
        """Clear pre-plan state on the SPECIFIC slot that
        ``pre_plan_for_batch`` targeted for ``spec_node_batch``.

        Used to recover from speculation drops / failures where the pre-
        plan was dispatched but the spec batch never reached the GPU
        thread — leaving entries in the slot's ``_pre_planned_labels``
        would cause the next real plan_attention call on that slot to
        short-circuit incorrectly.

        Slot-targeted (not worker-global) so that any other slot's
        valid in-flight pre-plan whose flags have not yet been consumed
        by the matching replay isn't stomped. The engine's
        ``reset_pre_plan_for_batch`` looks up the same (key, slot) the
        pre-plan path used; engines without a pre-plan surface inherit
        ``BaseEngine``'s no-op default.
        """
        engine = self.engine_manager.get_engine(spec_node_batch.node_name)
        engine.reset_pre_plan_for_batch(spec_node_batch)

    def _get_encoder_async_stream(self) -> "torch.cuda.Stream | None":
        """Lazily allocate the low-priority CUDA stream used for the encoder
        forward when ``MSTAR_ENCODER_ASYNC=1``.

        Using a non-default stream with the lowest priority lets the encoder
        forward overlap with concurrent Thinker decode kernels (which keep
        running on the default, higher-priority stream). The driver still
        time-slices SM occupancy, but the lower-priority stream is preferred
        when the queue is contended, so the encoder is a "good citizen"
        relative to latency-sensitive decode steps.

        Returns ``None`` when CUDA is unavailable (e.g. tests on CPU), in
        which case we fall through to default-stream execution — the
        speculative dispatch still helps by being scheduled earlier even if
        it can't physically overlap.
        """
        if not torch.cuda.is_available():
            return None
        if getattr(self, "_encoder_async_stream", None) is None:
            # priority=0 is the lowest priority (numerically larger = lower
            # priority in CUDA's API). We deliberately don't pick the most
            # extreme priority via ``get_stream_priority_range`` because the
            # range can be empty on non-Tesla devices; the default low value
            # is universally supported.
            try:
                self._encoder_async_stream = torch.cuda.Stream(
                    device=self.device, priority=0,
                )
            except (TypeError, RuntimeError):
                # Some builds may not accept the priority kwarg or device kw.
                # Fallback to a plain side stream — still gets us the
                # non-default-stream benefit even if priority isn't honored.
                self._encoder_async_stream = torch.cuda.Stream()
        return self._encoder_async_stream

    def _init_cuda_executor_thread(self) -> None:
        """Pin this executor thread to the worker's CUDA device.

        The CUDA current device is per-thread and defaults to 0. PyTorch
        ops carry per-tensor device guards, but raw Triton launches and
        bare ``torch.cuda.current_stream()`` / ``synchronize()`` calls
        resolve against the THREAD's device — on a worker whose model
        lives on a non-zero device, work issued from an unpinned thread
        lands on device 0's stream, unordered with the real compute.
        """
        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)

    def _execute_on_gpu_thread(
        self,
        batch: ScheduledBatch,
        node_batch: NodeBatch,
        plan_future: Future | None = None,
        advance_event: "threading.Event | None" = None,
        stream: "torch.cuda.Stream | None" = None,
    ) -> NodeOutput:
        """Run the engine on a GPU executor thread.

        The NVTX range bracketing this call is ``synchronize=False`` —
        adding a ``cudaDeviceSynchronize`` at the marker boundary would
        drain the GPU on every iter and hide the overlap between
        post-processing and the next step's kernel execution.

        After ``execute_with_max_batch_size`` returns we record a CUDA event
        and stash it on the output. Downstream sync points on the main thread
        wait on this event.

        ``stream``: when None (the decode / normal path), execute and record
        the completion event on the DEFAULT stream — unchanged behavior. When
        a side stream is passed (MSTAR_SIDE_PREFILL), execute the batch and
        record the completion event on THAT stream, so the batch overlaps
        decode replays on the default stream and downstream token-
        materialization waits gate on the correct stream. The captured graphs
        carry no stream affinity (torch.cuda.graph captures on its own
        internal stream), so replay on a side stream is valid; prefill and
        decode use disjoint static I/O buffers and FlashInfer workspaces, so
        concurrent execution does not corrupt.

        When MSTAR_ENCODER_ASYNC=1 and the batch is an encoder node
        (``vision_encoder`` / ``audio_encoder``), the forward runs on a
        dedicated low-priority side stream (input-fenced against the
        default stream, completion event recorded on the side stream, and
        the default stream fenced on it afterwards) so encoder kernels
        overlap Thinker work on the default stream.
        """
        from mstar.utils.profiler import range_pop, range_push

        engine = self.engine_manager.get_engine(batch.node_name)
        logger.debug(
            "Executing batch for node %s on engine %s",
            node_batch.node_name, str(type(engine))
        )
        if self.enable_nvtx:
            range_push("worker.gpu_thread_start", synchronize=False)
            range_pop(synchronize=False)
        # Wait for the plan_executor's pre-planned wrapper.plan() call to
        # finish before running this batch — its results land on the captured
        # graph's persistent wrappers, and the next plan_attention call(s)
        # will see the matching label in _pre_planned_labels only because
        # plan_executor populated it. Wait releases the GIL.
        if plan_future is not None:
            if self.enable_nvtx:
                range_push("worker.gpu_thread.await_plan", synchronize=False)
            try:
                plan_future.result()
            finally:
                if self.enable_nvtx:
                    range_pop(synchronize=False)
        if self.enable_nvtx:
            range_push(
                f"worker[{self.worker_id}].node[{batch.node_name}].graph_walk[{batch.graph_walk}]",
                synchronize=False,
            )
        if self._walk_stats is not None:
            key = (batch.node_name, batch.graph_walk)
            self._walk_stats[key] = self._walk_stats.get(key, 0) + 1
            self._walk_stats_step += 1
            # Merged multimodal prefill folds (MSTAR_MERGED_PREFILL): explicit
            # count of text+vision walks that ran as one prefill_multimodal step
            # instead of a separate prefill_text + prefill_vision pair.
            if batch.graph_walk == "prefill_multimodal":
                self._walk_stats["merged_prefill_walks"] = (
                    self._walk_stats.get("merged_prefill_walks", 0) + 1
                )
            # Merged text+audio prefill folds (MSTAR_MERGED_PREFILL_AUDIO):
            # explicit count of text+audio walks that ran as one
            # prefill_multimodal_audio step instead of a separate prefill_text +
            # prefill_audio pair. Proves the audio-merge mechanism is alive
            # (Law 4) — dumped at WARNING level with the rest of WALK_STATS below.
            if batch.graph_walk == "prefill_multimodal_audio":
                self._walk_stats["merged_prefill_audio_walks"] = (
                    self._walk_stats.get("merged_prefill_audio_walks", 0) + 1
                )
            # Classify standalone prefill steps: chunked (clen bucket) vs
            # unchunked (raw span bucket from the widest input tensor) — sizes
            # the two mixable-gate misses (no chunk metadata / C too big).
            if batch.graph_walk.startswith("prefill_"):
                for rid in node_batch.request_ids:
                    fi = node_batch.per_request_info.get(rid)
                    clen = None
                    if fi is not None:
                        md = getattr(fi, "step_metadata", None)
                        if md:
                            clen = md.get("prefill_chunk_len")
                    if clen is None:
                        span = 0
                        for tl in node_batch.per_request_input_tensors.get(
                            rid, {}
                        ).values():
                            for t in tl:
                                if hasattr(t, "shape") and len(t.shape) >= 1:
                                    span = max(span, int(t.shape[0]))
                        ck = f"_pf_unchunked_{batch.graph_walk[8:]}_" + (
                            "le256" if span <= 256 else
                            "le512" if span <= 512 else "gt512"
                        )
                    else:
                        ck = f"_pf_chunk_{batch.graph_walk[8:]}_" + (
                            "le256" if int(clen) <= 256 else
                            "le512" if int(clen) <= 512 else "gt512"
                        )
                    self._walk_stats[ck] = self._walk_stats.get(ck, 0) + 1
            if self._walk_stats_step % 200 == 0:
                logger.warning(
                    "WALK_STATS step=%d %s",
                    self._walk_stats_step,
                    sorted(self._walk_stats.items(), key=lambda kv: -kv[1]),
                )

        # MSTAR_ENCODER_ASYNC: encoder nodes run on a dedicated low-priority
        # side stream (see docstring). Distinct from the MSTAR_SIDE_PREFILL
        # ``stream`` parameter: this path adds input/downstream fences the
        # side-prefill contract does not need.
        use_side_stream = (
            stream is None
            and
            getattr(self.scheduler, "encoder_async_enabled", False)
            and batch.node_name in ("vision_encoder", "audio_encoder")
            and torch.cuda.is_available()
        )
        _ws_t0 = _time.perf_counter() if self._walk_stats is not None else 0.0
        try:
            if use_side_stream:
                side_stream = self._get_encoder_async_stream()
                if side_stream is None:
                    # CUDA disappeared between the flag check and stream
                    # allocation — degrade gracefully to default-stream
                    # execution. This is the same fallback the rest of the
                    # worker uses when ``torch.cuda.is_available()`` flips.
                    output = engine.execute_with_max_batch_size(node_batch)
                    if torch.cuda.is_available():
                        event = torch.cuda.Event()
                        event.record(torch.cuda.default_stream(self.device))
                        output.completion_event = event
                    return output
            if stream is not None:
                # Side-stream execution (MSTAR_SIDE_PREFILL): run the whole
                # batch on the side stream so it overlaps default-stream decode
                # replays, and record the completion event on the SAME stream.
                with torch.cuda.stream(stream):
                    output = engine.execute_with_max_batch_size(node_batch)
                    if torch.cuda.is_available():
                        event = torch.cuda.Event()
                        event.record(stream)
                        output.completion_event = event
            else:
                output = engine.execute_with_max_batch_size(node_batch)
                if torch.cuda.is_available():
                    event = torch.cuda.Event()
                    event.record(torch.cuda.default_stream(self.device))
                    output.completion_event = event
            if self._walk_stats is not None:
                # Wall ms per walk (CPU submit side; GPU async tail not
                # included — comparable across configs, not absolute).
                mk = f"_ms_{batch.graph_walk}"
                self._walk_stats[mk] = self._walk_stats.get(mk, 0) + int(
                    (_time.perf_counter() - _ws_t0) * 1000
                )
            return output
        finally:
            # Safety net: ensure advance_event fires even if the engine
            # raised before reaching ``advance_seq_lens`` inside
            # ``_run_basic_batched``. Without this, a plan_executor waiting
            # on prev_advance_event would block forever on the failure path.
            if advance_event is not None:
                advance_event.set()
            # Same idea for launch_started_event: if the engine raised
            # before reaching the deep set site, release the main-thread
            # waiter early instead of making it eat the full timeout.
            launch_started_event = node_batch.metadata.get("launch_started_event")
            if launch_started_event is not None:
                launch_started_event.set()
            # Mirror engine-internal state (e.g. KV-cache seq_info) back
            # onto node_batch.per_request_info so the next iter's prep /
            # routing sees the updated values. Runs regardless of success,
            # allocation_failed, or an uncaught raise — finalize_batch
            # reads whatever state the engine actually reached.
            engine.finalize_batch(node_batch)
            if self.enable_nvtx:
                range_pop(synchronize=False)

    def _handle_allocation_failure(
        self, batch: ScheduledBatch, node_batch: NodeBatch
    ) -> None:
        """Push back nodes and hold the rids for backoff after KV OOM.

        Under TP, this runs on every rank of the TP group independently:
        admission decisions (``add_request`` / ``alloc`` / ``free``) are
        all driven by rank 0's scheduler and replicated via the
        ``ScheduleTPNode`` ZMQ broadcast, so the page allocator state is
        symmetric across ranks. Both rank 0 and followers raise
        ``AllocationFailedError`` on the same batch and both reach this
        function with the same ``batch_ids``; their local actions
        (push-back, hold) produce identical follower state.

        ``KVCacheEngine._verify_tp_kv_symmetry`` fails fast at startup if
        that invariant ever breaks.

        v2 caveat: this function does not yet coordinate ``_last_active``
        / eviction-victim selection across TP ranks. Wall-clock LRU can
        pick different victims per rank under contention, leading to
        request-id ↔ page-index drift and (eventually) asymmetric OOM on
        future reloads. Today's TP configs don't enable CPU offload, so
        the path isn't exercised; revisit when we light up offload + TP.
        """
        batch_ids = set(batch.node_objects.keys())
        victim_id = self._try_offload_cold_request(node_batch.node_name, batch_ids)

        # Push all batch nodes back to their queues
        for request_id, node in batch.node_objects.items():
            wg_id = batch.request_to_worker_graph[request_id]
            self.worker_graphs_manager.queues[wg_id].push_back_node(
                request_id, node
            )

        if victim_id is not None:
            self.scheduler.hold_requests([victim_id])
            logger.warning(
                "OOM on node=%s walk=%s: offloaded victim=%s, "
                "retrying %d remaining requests",
                batch.node_name, batch.graph_walk, victim_id,
                len(batch_ids) - (1 if victim_id in batch_ids else 0),
            )
        else:
            self.scheduler.hold_requests(list(batch_ids))
            logger.warning(
                "OOM on node=%s walk=%s: no offload possible, "
                "holding %d requests",
                batch.node_name, batch.graph_walk, len(batch_ids),
            )

    # ------------------------------------------------------------------
    # Speculation
    # ------------------------------------------------------------------

    def _can_speculate(self, batch: ScheduledBatch) -> bool:
        if any(
            not node.enable_async_scheduling for node in batch.node_objects.values()
        ) or batch.node_name in self.parallel_nodes:
            # disable speculation for lockstep-parallel nodes for now
            return False
        # Mixed batch: whether the chain may CONTINUE from a thinker_mixed step.
        #
        # * MSTAR_MIXED_SPEC off (0cc7c71): never speculate FROM a mixed step.
        #   The mixed batch ran on the non-spec path and the chain restarts on
        #   the following uniform decode step.
        #
        # * MSTAR_MIXED_SPEC on: DO speculate the next uniform decode step from
        #   the mixed batch. The decode rids' new-token outputs exist, so
        #   ``_try_speculate_next`` threads them into the continuation exactly as
        #   a pure-decode step; the chunk rid is excluded from the continuation
        #   there (it emits no continuing decode token, or its first token is
        #   admitted via the normal ready path) so the guess is uniform decode,
        #   not heterogeneous. This is what keeps the folded mixed step INSIDE
        #   the chain instead of breaking it.
        if batch.graph_walk == "thinker_mixed":
            from mstar.model.qwen3_omni.qwen3_omni_model import (
                mixed_batch_spec_enabled,
            )
            return mixed_batch_spec_enabled()
        return True

    def _is_side_eligible(self, batch: ScheduledBatch) -> bool:
        """Whether ``batch`` may run on the side stream concurrently with the
        decode chain (MSTAR_SIDE_PREFILL).

        Eligible batches are the ones whose GPU work we want to hide behind
        decode: thinker-PREFILL walks (KV_CACHE engine, graph_walk starting
        with ``prefill``) and STATELESS encoder nodes (which route into a
        prefill; they may or may not be co-located depending on the 2-GPU
        config variant). Decode (``thinker_decode``) is never side-eligible —
        it is the chain we overlap AGAINST, not a batch to overlap.

        TP nodes are excluded: TP scheduling is driven by rank-0 broadcast
        (ScheduleTPNode) and the follower ranks execute on their own GPU
        threads with default-stream ordering assumptions; running a TP batch on
        a side stream on rank 0 only would desynchronize the group. Keep the
        side path single-rank (non-TP) prefill/encoder work.
        """
        if not self._side_prefill:
            return False
        if not batch.node_objects:
            return False
        if batch.node_name in self.tp_nodes:
            return False
        engine = self.engine_manager.get_engine(batch.node_name)
        etype = engine.engine_type()
        if etype == EngineType.STATELESS:
            return True
        if etype == EngineType.KV_CACHE and batch.graph_walk.startswith("prefill"):
            return True
        return False

    def _reap_side_if_done(self, pending_side: "PendingSide | None") -> "PendingSide | None":
        """If a side batch has finished on the side stream, postprocess it on
        the MAIN thread and return None; otherwise return it unchanged.

        Opportunistic: never blocks on the future. Postprocess reuses the
        normal routing path (_postprocess_batch), whose completion_event
        handling drains the side stream before routing — see the correctness
        note in __init__."""
        if pending_side is None:
            return None
        if not pending_side.future.done():
            return pending_side
        try:
            output: NodeOutput = pending_side.future.result()
            for node in pending_side.batch.node_objects.values():
                node._speculatively_scheduled = False
            if output.allocation_failed:
                # KV OOM on the side prefill: rehabilitate exactly like the
                # decode path (push nodes back + hold rids). Does not touch the
                # decode chain — the failed rids are disjoint from live decode.
                self._handle_allocation_failure(
                    pending_side.batch, pending_side.node_batch
                )
            else:
                # _postprocess_batch unconditionally clears
                # self._pending_loop_stops at its tail (they are a one-iter
                # decode-speculation mechanism, consumed only by the NEXT
                # decode speculative_new_iter postprocess). This side reap runs
                # at the TOP of the loop, BEFORE the current iter's decode
                # speculation consumes the stops the previous decode iter set —
                # so letting the side postprocess clear them would drop a
                # legitimate decode loop-stop. Prefill batches never set
                # speculative_new_iter and any prefill-walk stops they add are
                # never consumed, so the decode chain's pending stops must pass
                # through the side postprocess untouched: snapshot and restore.
                saved_loop_stops = set(self._pending_loop_stops)
                self._postprocess_batch(
                    PendingBatch(
                        batch=pending_side.batch,
                        node_batch=pending_side.node_batch,
                        node_name=pending_side.node_name,
                        partition=pending_side.partition,
                        graph_walk=pending_side.graph_walk,
                        future=pending_side.future,
                    ),
                    output,
                )
                self._pending_loop_stops = saved_loop_stops
        except Exception:
            logger.exception(
                "Worker %s: side prefill batch failed", self.worker_id
            )
        finally:
            self._side_in_flight_rids = set()
        return None

    def _drain_side(self, pending_side: "PendingSide | None") -> None:
        """BLOCK until an in-flight side prefill finishes, then postprocess it.

        Called before a NON-speculative scheduling round (the decode chain has
        broken). The next get_next_batch may hand out a decode batch that reads
        KV pages a still-running side prefill is writing; those two would land
        on the default stream and the side stream with no ordering between
        them. Draining here forces the side prefill (and its KV writes) to
        complete and be routed before any new default-stream work is scheduled,
        so the drain is the chain-break correctness gate. No-op when the flag
        is off or nothing is in flight."""
        if not self._side_prefill or pending_side is None:
            return
        # Block on the future so _reap_side_if_done's done() check passes and
        # it runs the full postprocess (which itself drains the side stream via
        # the completion_event before routing).
        try:
            pending_side.future.result()
        except Exception:
            logger.exception(
                "Worker %s: side prefill drain failed", self.worker_id
            )
        self._reap_side_if_done(pending_side)

    def _maybe_dispatch_side(
        self,
        pending_side: "PendingSide | None",
        side_executor: "ThreadPoolExecutor | None",
        exclude_target: "tuple[str, str] | None",
    ) -> "PendingSide | None":
        """When a decode chain is active and no side batch is in flight, try to
        schedule a prefill/encoder batch and run it on the side stream.

        ``exclude_target`` is the active decode (node, walk) so the scheduler
        never hands back the decode group. Because get_next_batch POPS the
        chosen nodes out of the ready queue, the subsequent speculation
        has_ready_excluding won't re-see them — so dispatching here does not
        disturb the decode speculation chain. Returns the new pending_side (or
        the unchanged one if nothing was dispatched).

        All scheduling stays on the main thread (this method is only called
        from run()); the side executor solely EXECUTES the built batch."""
        if not self._side_prefill or side_executor is None:
            return pending_side
        if pending_side is not None:
            return pending_side  # one side batch in flight at a time
        with self._scheduler_lock:
            batch = self.scheduler.get_next_batch(
                self.worker_graphs_manager,
                exclude_target=exclude_target,
            )
        if batch is None:
            return pending_side
        if not self._is_side_eligible(batch):
            # Not a prefill/encoder we want on the side stream. We already
            # popped it from the queue; push its nodes back so the normal
            # (main-stream) path schedules it next iter.
            self._pushback_scheduled_batch(batch)
            return pending_side

        node_batch = self._build_node_batch(batch)
        # Force this batch down the eager / batched path, NEVER the captured
        # decode CUDA graph. KVCacheEngine._can_use_cuda_graph returns False
        # when this flag is set: the captured graph shares interned static I/O
        # buffers across slots and mutates a single-writer next_slot counter,
        # both of which a concurrent side replay would race. The eager path
        # uses FlashInfer workspace label "main", disjoint from decode's
        # per-slot labels, so it is safe concurrent with decode. The flag rides
        # NodeBatch.metadata, which execute_with_max_batch_size propagates to
        # every minibatch, so the gate holds even under max-batch-size splits.
        node_batch.metadata["side_stream"] = True
        batch_partition = self.worker_graphs_manager.get_partition_for_node(
            batch.node_name
        )
        for request_id, req_info in node_batch.per_request_info.items():
            req_info.dynamic_loop_iter_counts.update(
                self.worker_graphs_manager.get_dynamic_loop_iters(
                    request_id, partition=batch_partition,
                )
            )
        # In-flight bookkeeping: defer removes for these rids until the side
        # batch is postprocessed, and keep the nodes off the ready queue while
        # executing (same guard decode uses).
        for node in batch.node_objects.values():
            node._speculatively_scheduled = True
        self._side_in_flight_rids = set(batch.node_objects.keys())
        self.maybe_send_zmq_to_tp_followers(node_batch)
        future = side_executor.submit(
            self._execute_on_gpu_thread,
            batch, node_batch,
            None,            # plan_future — side prefill plans inline (eager)
            None,            # advance_event — not part of the spec chain
            self._side_stream,
        )
        self.wakeup_event.register_future(future)
        logger.debug(
            "Side-dispatch: %s %s", batch.node_name, node_batch.request_ids
        )
        return PendingSide(
            batch=batch,
            node_batch=node_batch,
            node_name=batch.node_name,
            partition=batch_partition,
            graph_walk=batch.graph_walk,
            future=future,
        )

    def _pushback_scheduled_batch(self, batch: ScheduledBatch) -> None:
        """Return a popped-but-not-run batch's nodes to their ready queues so a
        later schedule can pick them up. Used when a side-dispatch candidate
        turns out not to be side-eligible."""
        for rid, node in batch.node_objects.items():
            wg_id = batch.request_to_worker_graph.get(rid)
            if wg_id is None:
                continue
            queue = self.worker_graphs_manager.queues.get(wg_id)
            if queue is None:
                continue
            queue.push_back_node(rid, node)

    def _get_wgio_for_rid(self, batch: ScheduledBatch, rid: str):
        """Per-rid WorkerGraphIO for the wg that owns this rid in this batch.
        """
        wg_id = batch.request_to_worker_graph[rid]
        return self.worker_graphs_manager.queues[wg_id].per_request_queues[rid]

    def _get_input_tensors(
        self, rid: str, node: GraphNode, check_next_iter: bool
    ) -> NameToTensorList:
        inputs = node.ready_next_iter.ready_inputs if check_next_iter \
            else node.ready_signals.ready_inputs
        tensors = {}
        for input_name, edge in inputs.items():
            tensors[input_name] = [
                self.tensor_manager.get_tensor(
                    request_id=rid, uuid=info.uuid,
                )
                for info in edge.tensor_info
            ]
        return tensors

    def _try_speculate_next(
        self,
        pending: PendingBatch
    ) -> Speculation | None:
        """Build a speculative N+1 batch + node_batch, by checking which nodes
        will become ready after the current batch's outputs are ingested.

        The speculated batch is a merge of:
          * **continuing** rids (subset of batch_N still alive, not
            pending-stop / pending-remove) — placeholder inputs are gathered
            from the registry now (``_get_input_tensors``) and the entries
            tied to ``consumed_edges`` are overwritten with batch_N's outputs
            after await by ``_thread_outputs_to_speculative``.
          * **fresh** rids — newly-arrived requests whose spec-target node
            is ready in the queue right now. Their inputs come from the
            usual tensor_manager path (same as ``_build_node_batch``).
            Without this merge, new rids have to wait for the entire
            current speculation chain to drain before they can be scheduled.
        """
        batch_N = pending.batch
        partition_N = pending.partition

        # The walk the CONTINUATION runs under. Normally identical to
        # batch_N.graph_walk. When speculating FROM a folded thinker_mixed step
        # (MSTAR_MIXED_SPEC), the continuation is a UNIFORM decode batch —
        # thinker_decode — so every downstream use of the walk here (loop-stop
        # dedup match, fresh-rid target_graph_walk, spec batch / node batch walk)
        # must be thinker_decode, NOT thinker_mixed (which matches no ready rid
        # and no recorded stop). Any further chunk fold onto this continuation is
        # decided back in run() and re-tags the batch there.
        graph_walk = pending.graph_walk
        if graph_walk == "thinker_mixed":
            graph_walk = "thinker_decode"

        # MSTAR_MIXED_SPEC: when speculating FROM a folded thinker_mixed step,
        # the chunk row is NOT part of the continuing decode chain. A non-last
        # chunk emits no decode token (its next step is the following prefill
        # chunk, a different spec target), and even a last chunk's first decode
        # token is cleaner to admit via the normal ready path than to special-
        # case here. So exclude the chunk rid(s) from BOTH the spec-target sample
        # and the continuation membership: sample from a decode rid, skip the
        # chunk in the continuation loop. The chunk rid re-enters the ready queue
        # when the mixed step's postprocess routes its output under its own walk,
        # and rejoins the chain via the usual fresh-rid merge / normal schedule.
        # For every non-mixed batch ``chunk_rids`` is empty → byte-identical.
        chunk_rids: set[str] = set()
        if batch_N.graph_walk == "thinker_mixed":
            for r in batch_N.node_objects:
                info = pending.node_batch.per_request_info.get(r)
                if info is not None and info.graph_walk != "thinker_decode":
                    chunk_rids.add(r)

        # sample node and RID to see which node we will be speculating
        # (TODO: refine this to be, e.g., a majority vote). Sample a DECODE rid
        # so the spec target is the decode loop-back, never the chunk's next
        # (prefill) node.
        rid, sample_node = next(
            (
                (r, n) for r, n in batch_N.node_objects.items()
                if r not in chunk_rids
            ),
            (None, None),
        )
        if sample_node is None:
            return  # mixed step was chunk-only (no decode rows) — nothing to continue
        wgio = self._get_wgio_for_rid(batch_N, rid)

        # If sample_node has no outputs at all, it can't feed any spec target.
        if not sample_node.outputs:
            return
        ready_for_spec = wgio.ingest_for_speculation(
            sample_node.outputs, sample_node.name
        )
        wgio.clear_speculative_inputs()

        # Filter out destinations that aren't speculation candidates.
        #
        # * ``info.node_name in self.parallel_nodes`` — lockstep-parallel nodes
        #   don't support speculation in v1 (the instance leader schedules;
        #   followers can't initiate).
        # * ``not wgio.nodes[info.node_name].enable_async_scheduling`` — the
        #   destination node opts out of async scheduling. Mirrors the
        #   source-side check in ``_can_speculate``; without this, a
        #   destination that's structurally ineligible (e.g. a node whose
        #   downstream graph isn't speculation-safe) could still be picked,
        #   then dropped per-rid further down.
        ready_for_spec = [
            info for info in ready_for_spec
            if info.node_name not in self.parallel_nodes
            and wgio.nodes[info.node_name].enable_async_scheduling
        ]

        if not ready_for_spec:
            return # no nodes can be speculated

        # TODO: use the microscheduler to break ties when ready_for_spec
        # contains multiple ready nodes
        spec_node_info = ready_for_spec[0]
        speculating_same_node = spec_node_info.node_name == batch_N.node_name

        continuing = []
        new_node_objects: dict[str, GraphNode] = {}
        new_request_to_worker_graph: dict[str, str] = {}
        per_request_inputs: dict[str, NameToTensorList] = {}
        consumed_streaming_edges: dict[str, GraphEdge] = {}
        for rid, batch_N_node in batch_N.node_objects.items():
            if rid in chunk_rids:
                # Folded chunk row (MSTAR_MIXED_SPEC): not a continuing decode
                # rid — it advances to its own next node via postprocess routing
                # and rejoins the chain through the normal ready path.
                continue
            wgio = self._get_wgio_for_rid(batch_N, rid)
            loop = wgio.loops.get(spec_node_info.loop_name)

            # check conditions where the rid cannot be furtuer speculated
            already_removed = rid in self._pending_removes
            already_stopped = spec_node_info.is_new_loop_iter and PendingLoopStop(
                rid, graph_walk, spec_node_info.loop_name
            ) in self._pending_loop_stops
            is_stopping = spec_node_info.is_new_loop_iter and loop is not None and (
                loop.curr_iter + 1 >= loop.max_iters or loop._finish_signal
            )
            if already_removed or already_stopped or is_stopping:
                # Loop/request has already finished, don't speculate further work
                continue

            # If the speculation is contingent on streaming edges, ingest the
            # appropriate streaming edges
            node = wgio.nodes[spec_node_info.node_name]

            # temporarily set to prevent ingesting streaming inputs from re-adding the node to
            # the ready queue
            node._speculatively_scheduled = True
            streaming_edges = self._poll_stream_buffers_for_speculation(
                rid, spec_node_info.node_name
            )
            # Track which slot each ingest landed in so we can roll back if
            # the readiness check below fails. ``ingest_input`` returns success
            # without telling us which slot it used, so peek the slot state
            # before the call.
            ingested_into_ready_signals: list[GraphEdge] = []
            ingested_into_ready_next_iter: list[GraphEdge] = []
            for edge in streaming_edges:
                already_in_ready_signals = (
                    edge.name in node.ready_signals.ready_names
                )
                if node.ingest_input(
                    edge, can_buffer=speculating_same_node
                ):
                    if already_in_ready_signals:
                        ingested_into_ready_next_iter.append(edge)
                    else:
                        ingested_into_ready_signals.append(edge)
                else:
                    self._return_speculative_streaming_edge(rid, edge)

            # Check if the node is ready after ingesting the streaming edges
            wgio.ingest_for_speculation(
                batch_N_node.outputs, batch_N_node.name
            )
            fully_ready = node.is_ready_for_speculation(
                check_next_iter=speculating_same_node,
                allow_streaming=False
            )
            wgio.clear_speculative_inputs()
            node._speculatively_scheduled = False # reset in case this rid gets dropped
            if not fully_ready:
                # Roll back the streaming edges we just ingested so the
                # chunks don't sit in the spec target's ready_signals /
                # ready_next_iter unused. Return each chunk to its
                # StreamBuffer's uningested cache so a future scheduling
                # of this node consumes it normally. The registry's
                # ``ready_names`` / ``ready_next_iter`` sets weren't
                # touched (gated by ``_speculatively_scheduled=True``
                # above), so no registry-state rollback is needed.
                for edge in ingested_into_ready_signals:
                    node.ready_signals.remove(edge.name)
                    self._return_speculative_streaming_edge(rid, edge)
                for edge in ingested_into_ready_next_iter:
                    node.ready_next_iter.remove(edge.name)
                    self._return_speculative_streaming_edge(rid, edge)
                continue

            # prepare speculative batch
            continuing.append(rid)
            new_node_objects[rid] = node
            new_request_to_worker_graph[rid] = wgio.wg_id
            per_request_inputs[rid] = self._get_input_tensors(
                rid, node, check_next_iter=speculating_same_node,
            )
            consumed_streaming_edges[rid] = ingested_into_ready_next_iter + ingested_into_ready_signals

        if not continuing:
            return None

        # Edges that the spec batch effectively "consumed" from batch_N's
        # outputs: every output of sample_node whose destination is the spec node
        consumed_edges: set[tuple[str, str]] = {
            (edge.name, edge.next_node)
            for edge in sample_node.outputs
            if edge.next_node == spec_node_info.node_name
        }

        # Merge in fresh rids whose spec-target node is ready right now
        # Speculation only consumes work compatible with the spec target. In
        # partitioned models, unrelated ready work stays queued for
        # the normal scheduler path.
        fresh_batch = self.scheduler.get_next_batch(
            self.worker_graphs_manager,
            target_node_name=spec_node_info.node_name,
            target_graph_walk=graph_walk,
        )

        if fresh_batch is not None:
            # The merge below relabels these node objects with the spec
            # target's name/walk, so a batch for any other node must not be
            # merged in.
            assert fresh_batch.node_name == spec_node_info.node_name, (
                f"Speculation asked for {spec_node_info.node_name!r} but the "
                f"scheduler returned {fresh_batch.node_name!r}"
            )
            for rid, node in fresh_batch.node_objects.items():
                if rid in new_node_objects:
                    # Shouldn't happen — continuing rids are held by the
                    # in-flight step and shouldn't be in ready queues —
                    # but if it does, the in-flight rid wins.
                    #
                    # KNOWN GAP (latent): if fresh_batch ever came from the
                    # TP-follow path, ``_try_schedule_tp_follow`` has already
                    # popped the leader's ScheduleTPNode. Pushing the node
                    # back onto the ready queue does not undo that, and this
                    # worker is a follower for that node so nothing will
                    # reschedule it — the TP group stalls. Unreachable today
                    # (spec targets exclude ``parallel_nodes``, so the
                    # target_node_name filter never matches a TP batch), but
                    # any future overlap needs the scheduler to re-queue the
                    # message, or to defer the popleft until the caller
                    # commits to the batch.
                    wg_id = fresh_batch.request_to_worker_graph[rid]
                    self.worker_graphs_manager.queues[wg_id].push_back_node(rid, node)
                    continue

                per_request_inputs[rid] = self._get_input_tensors(
                    rid, node, check_next_iter=False
                )
                new_node_objects[rid] = node
                new_request_to_worker_graph[rid] = (
                    fresh_batch.request_to_worker_graph[rid]
                )

        spec_batch = ScheduledBatch(
            node_name=spec_node_info.node_name,
            graph_walk=graph_walk,
            node_objects=new_node_objects,
            request_to_worker_graph=new_request_to_worker_graph,
        )

        request_ids = list(new_node_objects.keys())
        spec_node_batch = NodeBatch(
            node_name=spec_node_info.node_name,
            graph_walk=graph_walk,
            request_ids=request_ids,
            per_request_input_tensors=per_request_inputs,
            per_request_info={
                rid: self.worker_graphs_manager.get_fwd_info(
                    rid, partition_N
                ) for rid in request_ids
            },
            final_stream_rids={
                rid for rid, edges in consumed_streaming_edges.items()
                if any(e._final_stream_chunk for e in edges)
            },
        )

        logger.debug(f"Speculating: {spec_node_info.node_name} {spec_node_batch.request_ids}")

        return Speculation(
            scheduled_batch=spec_batch,
            node_batch=spec_node_batch,
            consumed_edges=consumed_edges,
            continuing_rids=set(continuing),
            partition=pending.partition,
            is_new_iter=spec_node_info.is_new_loop_iter,
            is_same_node=speculating_same_node,
            loop_name=spec_node_info.loop_name,
            consumed_streaming_edges=consumed_streaming_edges
        )

    def _try_fold_mixed_chunk_into_spec(
        self, speculation: Speculation, budget_tokens: int | None = None,
    ) -> int | None:
        """MSTAR_MIXED_SPEC: fold ONE ready mixable prefill chunk row into an
        already-built decode continuation ``speculation``, turning the next
        speculative batch into a MIXED (thinker_mixed) step that rides inside the
        chain instead of breaking it (0cc7c71).

        The decode side of ``speculation`` is exactly the normal continuation
        built by ``_try_speculate_next`` — same membership, same loop-back
        threading. This only ADDS the chunk row:

          * pops the chunk node from the ready queue via the scheduler's
            ``pop_mixed_chunk_for_spec`` (the decode rids are mid-chain and NOT
            in the ready queue, so only the chunk is popped);
          * gathers the chunk's inputs from its ``ready_signals`` — the same
            source ``_build_node_batch`` uses for the non-spec mixed path — NOT
            from N's outputs (the chunk rid's inputs were already delivered by
            the conductor when its chunk node became ready, exactly like a
            "fresh" rid in ``_try_speculate_next``);
          * flips the batch-level ``graph_walk`` to ``thinker_mixed`` (per-rid
            walks are untouched — postprocess routes by per-request
            effective_walk, 7c09d10).

        The chunk rid is deliberately NOT added to ``continuing_rids``, so
        ``_thread_outputs_to_speculative`` skips it (no loop-back from N) — same
        as a fresh rid. Returns the folded chunk's length C (the chunk row's
        token count) if a chunk was folded in (batch is now mixed), or None if
        none was ready (leave ``speculation`` as the uniform decode
        continuation). Only called when the flag is on and a mixed opportunity
        was peeked, so the common case (chunk still ready) folds.
        """
        spec_batch = speculation.scheduled_batch
        spec_node_batch = speculation.node_batch
        decode_node_name = spec_batch.node_name

        # n_decode = the continuation size BEFORE the chunk row is appended; feed
        # it + the V2 budget to the pop so it selects the same budget-fitting
        # chunk has_mixed_opportunity approved (both scan first-fit identically).
        # ``budget_tokens`` defaults to the V2 budget but MSTAR_COADMIT (fix #1)
        # passes its clamped unified budget so the pop's over-budget filter
        # matches the budget the peek used — else a chunk the peek approved could
        # fail to pop under a different ceiling.
        if budget_tokens is None:
            budget_tokens = self._mixed_budget_tokens
        popped = self.scheduler.pop_mixed_chunk_for_spec(
            self.worker_graphs_manager,
            (decode_node_name, spec_batch.graph_walk),
            n_decode=len(spec_batch.node_objects),
            budget_tokens=budget_tokens,
        )
        if popped is None:
            return None
        chunk_node, chunk_rid, chunk_wg_id, chunk_len = popped

        # Guard: the chunk rid must be distinct from the continuing decode rids.
        # Decode rids are mid-chain (speculatively scheduled, off the queue) so
        # this should always hold; if it somehow doesn't, push the chunk back and
        # keep the pure-decode continuation rather than corrupt membership.
        if chunk_rid in spec_batch.node_objects:
            self.worker_graphs_manager.queues[chunk_wg_id].push_back_node(
                chunk_rid, chunk_node
            )
            return None

        # Keep the chunk node off the ready queue while it executes in the spec
        # step (same guard the decode nodes carry), so a concurrent schedule
        # can't re-pick it.
        chunk_node._speculatively_scheduled = True

        # Chunk inputs come from its own ready_signals (conductor-delivered),
        # exactly like _build_node_batch / a fresh rid — never threaded from N.
        chunk_inputs = self._get_input_tensors(
            chunk_rid, chunk_node, check_next_iter=False,
        )
        chunk_final_stream = any(
            edge._final_stream_chunk
            for edge in chunk_node.ready_signals.ready_inputs.values()
        )

        spec_batch.node_objects[chunk_rid] = chunk_node
        spec_batch.request_to_worker_graph[chunk_rid] = chunk_wg_id
        spec_batch.graph_walk = "thinker_mixed"

        # n_decode = the continuation size BEFORE the chunk row is appended.
        # Every continuation row is a thinker_decode row (1 new token), so the
        # mixed batch's per-row plan seq_lens are [1]*n_decode + [C] in exactly
        # the request_ids order below (decode rows first, chunk row last) —
        # matching what the packed preprocess builds from the ARNodeInputs.
        n_decode = len(spec_node_batch.request_ids)

        spec_node_batch.graph_walk = "thinker_mixed"
        spec_node_batch.request_ids = list(spec_node_batch.request_ids) + [chunk_rid]
        spec_node_batch.per_request_input_tensors[chunk_rid] = chunk_inputs
        spec_node_batch.per_request_info[chunk_rid] = (
            self.worker_graphs_manager.get_fwd_info(chunk_rid, speculation.partition)
        )
        if chunk_final_stream:
            spec_node_batch.final_stream_rids = set(
                spec_node_batch.final_stream_rids
            ) | {chunk_rid}

        # MSTAR_MIXED_PREPLAN: stash the packed pre-plan params so the engine's
        # reserve / pre-plan / reset trio routes to the packed runner surface.
        # num_tokens picks the token bucket (n_decode 1-token rows + the C-token
        # chunk); seq_lens is the per-row plan list the packed pre-plan pads and
        # feeds to plan_attention. Set ONLY under the flag, so flag-off (and the
        # non-preplan mixed path) never carries this key and the trio stays on
        # its BASIC_BATCHED-only behavior. chunk_len can be None if the chunk
        # carried no prefill_chunk_len metadata (shouldn't happen for a mixable
        # chunk) — guard so we don't stash a broken bucket.
        if self.mixed_batch_preplan and chunk_len is not None:
            # Under MSTAR_MIXED_SPLIT_ATTN the engine pads this real-row shape
            # to the fixed-region layout inside pre_plan_packed_batch (and
            # _get_key_for adds the dummy-row tokens), so the stash stays in
            # real-row terms either way. With split, the pre-plan is SIZED:
            # each folded step otherwise plans TWO wrappers inline on the
            # gpu thread (~3-5ms x ~500 folds/cell at i2t B32).
            spec_node_batch.metadata["mixed_preplan"] = {
                "num_tokens": n_decode + int(chunk_len),
                "seq_lens": [1] * n_decode + [int(chunk_len)],
            }

        # Validation hook: distinguish a chain-RIDING mixed assembly from a
        # chain-BREAK assembly (micro_scheduler's "mixed batch:" log). n_decode
        # is the continuation size; C is the chunk bucket.
        logger.info(
            "mixed-in-chain: n_decode=%d C=%s node=%s chunk_rid=%s",
            len(spec_batch.node_objects) - 1, chunk_len,
            decode_node_name, chunk_rid,
        )
        # Return the folded chunk length (C) for WALK_STATS budget accounting.
        # chunk_len is guaranteed non-None here — the mixable gate requires
        # prefill_chunk_len — but coerce defensively so the caller's None-check
        # (fold happened vs not) never trips on a stray missing metadata.
        return int(chunk_len) if chunk_len is not None else 0

    def _thread_outputs_to_speculative(
        self, speculation: Speculation, output_N: NodeOutput
    ):
        threaded_continuing: set[str] = set()
        dropped: set[str] = set()

        # MSTAR_DIRECT_FEED fast path. Precondition for using the batched
        # tensor at all: the flag is on, this is a same-node loop-back
        # (uniform thinker_decode) step, and batch_N's engine exposed the
        # [bs] sampled-tokens tensor + its rid order. Any consumed edge NOT
        # covered by that tensor (a rid missing from batched_sampled_rids, or
        # a consumed edge that isn't the loop-back token) falls through to the
        # per-rid output-map copy below — so a partial/absent tensor never
        # drops a rid, it just uses the slower (but identical-valued) route.
        # The batched tensor's rows are the SAME clone the per-rid new_token /
        # text_inputs views point at (cuda_graph_runner._sample_and_remap), so
        # sampled[i:i+1] is byte-identical to rid_outputs["text_inputs"][0];
        # it's a fresh per-step clone (no aliasing with FlashInfer's reused
        # sampling buffer — see the clone rationale there), so the row views
        # stay valid until the spec batch consumes them.
        row_for_rid: dict[str, "torch.Tensor"] = {}
        direct_feed_active = (
            self._direct_feed
            and speculation.is_same_node
            and output_N.batched_sampled_tokens is not None
            and output_N.batched_sampled_rids is not None
        )
        if direct_feed_active:
            sampled = output_N.batched_sampled_tokens
            for i, r in enumerate(output_N.batched_sampled_rids):
                row_for_rid[r] = sampled[i:i + 1]

        for rid in list(speculation.node_batch.request_ids):
            if rid not in speculation.continuing_rids:
                continue  # fresh rid — inputs already gathered.
            rid_outputs = output_N.per_request_output_tensors.get(rid, {})
            row_view = row_for_rid.get(rid) if direct_feed_active else None
            ok = True
            for input_name, _ in speculation.consumed_edges:
                tensors = rid_outputs.get(input_name, [])
                # Substitute the batched row ONLY when it is provably the same
                # value as this edge's per-rid output: a single 1-element token
                # view (the loop-back sampled token). Any other loop-back edge
                # (multi-tensor / non-token) takes the per-rid copy path so a
                # future same-node loop with a non-token loop-back stays correct.
                if (
                    row_view is not None
                    and len(tensors) == 1
                    and torch.is_tensor(tensors[0])
                    and tensors[0].numel() == 1
                ):
                    speculation.node_batch.per_request_input_tensors[rid][input_name] \
                        = [row_view]
                    continue
                if not tensors:
                    ok = False
                    break
                speculation.node_batch.per_request_input_tensors[rid][input_name] \
                    = list(tensors)
            if ok:
                threaded_continuing.add(rid)
            else:
                dropped.add(rid)

        if dropped:
            logger.warning(
                "Speculation: dropped rids %s (no loop-back output from N)",
                sorted(dropped),
            )
            speculation.node_batch.request_ids = [
                r for r in speculation.node_batch.request_ids if r not in dropped
            ]
            for r in dropped:
                speculation.node_batch.per_request_input_tensors.pop(r, None)
                speculation.node_batch.per_request_info.pop(r, None)
                speculation.scheduled_batch.request_to_worker_graph.pop(r, None)
                speculation.scheduled_batch.node_objects.pop(r, None)
                for edge in speculation.consumed_streaming_edges.get(rid, []):
                    self._return_speculative_streaming_edge(rid, edge)
                speculation.consumed_streaming_edges.pop(rid, None)
        speculation.continuing_rids = threaded_continuing
        speculation.dropped = dropped

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------
    def _cleanup_consumed_inputs(self, batch: ScheduledBatch) -> None:
        """Free input tensors that were consumed by the just-executed node."""
        for node in batch.node_objects.values():
            node.ready_signals.clear()


    def _prematerialized_new_tokens(
        self, cpu_output, rid: str,
    ) -> dict[str, list[int]] | None:
        """Extract prematerialized new-token ints for ``rid`` from check_stop's
        CPU output.

        Reuses check_stop's side-stream D→H copies for the new-token ints.
        Without this, buffer_new_tokens does a per-rid ``get_tensor().cpu()``
        — a default-stream sync per request per step (32/step at B32) that
        also serializes against the in-flight speculative step's kernels on
        the default stream. Only integer, non-CUDA tensors qualify; audio /
        multimodal (float / large) outputs are never included.
        """
        rid_cpu = cpu_output.per_request_output_tensors.get(rid)
        if not isinstance(rid_cpu, dict):
            return None
        prem: dict[str, list[int]] = {}
        for name, tensors in rid_cpu.items():
            if (
                isinstance(tensors, list)
                and tensors
                and all(
                    torch.is_tensor(t)
                    and not t.is_cuda
                    and not t.is_floating_point()
                    for t in tensors
                )
            ):
                prem[name] = [
                    int(v) for t in tensors for v in t.flatten().tolist()
                ]
        return prem

    def _postprocess_batch(
        self, batch_N: PendingBatch,
        output: NodeOutput,
    ):
        if self.enable_nvtx:
            range_push("worker.postprocess.cleanup_inputs", synchronize=False)
        self._cleanup_consumed_inputs(batch_N.batch)
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.pending_loop_stops", synchronize=False)
        # If any nodes in the batch have "overstayed" their loop stop, then make
        # sure to not route their outputs
        valid_rids = set(batch_N.node_batch.request_ids)
        if batch_N.speculative_new_iter:
            # MSTAR_MIXED_SPEC: a mixed spec batch carries batch-level walk
            # "thinker_mixed", but loop stops are recorded under the rid's OWN
            # walk (thinker_decode, see the stop-recording below / 7c09d10). The
            # overstay dedup here only concerns the CONTINUING decode rids (the
            # chunk row has no pending loop stop), so match pending stops against
            # the decode walk, not "thinker_mixed" (which no stop is keyed under
            # and would skip the dedup entirely). Non-mixed batches are
            # unchanged: overstay_walk == batch_N.graph_walk.
            overstay_walk = batch_N.graph_walk
            if overstay_walk == "thinker_mixed":
                overstay_walk = "thinker_decode"
            for pending_stop in self._pending_loop_stops:
                if pending_stop.loop_name != batch_N.loop_name \
                        or pending_stop.graph_walk != overstay_walk \
                        or pending_stop.rid not in batch_N.node_batch.request_ids:
                    continue
                stopped_rid = pending_stop.rid
                output.per_request_output_tensors.pop(stopped_rid, None)
                valid_rids.discard(stopped_rid)
                batch_N.batch.node_objects.pop(stopped_rid, None)
                batch_N.batch.request_to_worker_graph.pop(stopped_rid, None)
                batch_N.node_batch.per_request_info.pop(stopped_rid, None)
                # Structural change (rid dropped mid-step): drop any replay plan.
                if self._fast_postproc:
                    self.tensor_manager.invalidate_populate_plan(stopped_rid)
                # MSTAR_FAST_ROUTE2: same trigger, route-plan analogue (no-op
                # when the plan cache is empty / flag off).
                self.worker_graphs_manager.invalidate_route_plan(stopped_rid)
        batch_N.node_batch.request_ids = list(valid_rids)
        if not valid_rids:
            range_pop(synchronize=False)
            return

        # pending stops are only needed for one iteration, so can be cleared now
        self._pending_loop_stops.clear()

        # An engine can drop rids that were skipped during execution (a
        # submodule's prepare_inputs returned None) from node_batch.request_ids,
        # but it cannot reach the worker-side ScheduledBatch. Reconcile it here so
        # the routing/output loops below only touch rids that produced outputs.
        for rid in list(batch_N.batch.request_to_worker_graph):
            if rid not in valid_rids:
                batch_N.batch.request_to_worker_graph.pop(rid, None)
                batch_N.batch.node_objects.pop(rid, None)

        per_req_nested_idxs = {
            rid: self.worker_graphs_manager.get_nested_loop_idxs_for_node(
                rid, batch_N.partition, batch_N.node_name
            ) for rid in batch_N.node_batch.request_ids
        }

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.update_lru", synchronize=False)

        # Update LRU
        t = _time.monotonic()
        for rid in batch_N.node_batch.request_ids:
            self._last_active[(rid, batch_N.node_name)] = t

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.synchronize_completion_event", synchronize=False)

        # Wait for batch N's completion event before proceeding
        # TODO: may need to refine this based on how it affects performance?
        # MSTAR_SKIP_REDUNDANT_SYNC: skip this blanket sync — the deferred
        # check_stop copy self-gates on completion_event and _await_checkstop
        # polls it before the stop decision; emit reuses that gated copy. Only
        # taken when SIDECAR_CHECKSTOP is on (so the gate exists).
        if (
            torch.cuda.is_available()
            and batch_N.batch.node_objects
            and not self._skip_redundant_sync
        ):
            if output.completion_event is not None:
                if self.enable_nvtx:
                    range_push("worker.postprocess.completion_event_sync", synchronize=False)
                output.completion_event.synchronize()
                if self.enable_nvtx:
                    range_pop(synchronize=False)
            else:
                torch.cuda.default_stream().synchronize()

        if self.enable_prof:
            batch_N.node_batch.exec_timings.fwd_end = time.perf_counter()

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.check_stop", synchronize=False)

        # Check for stops. MSTAR_SIDECAR_CHECKSTOP: enqueue the
        # side-stream check_stop D→H WITHOUT blocking, run the cheap per-rid
        # dynamic-loop-iter Python (overlapping the in-flight copy), then poll
        # the copy event and decide THIS step. Flag off: prematerialize blocks
        # inline and the two independent loops below are output-identical
        # regardless of order.
        engine = self.engine_manager.get_engine(batch_N.node_name)
        cpu_output = self._prematerialize_for_check_stop(
            output,
            batch_fast=(batch_N.graph_walk == "thinker_decode"),
            talker_fast=(
                self._fast_checkstop_talker
                and batch_N.graph_walk == "talker_decode"
            ),
            defer=self._sidecar_checkstop,
        )

        for rid, req_info in batch_N.node_batch.per_request_info.items():
            new_iters = self.worker_graphs_manager.get_dynamic_loop_iters(
                rid, partition=batch_N.partition,
            )
            req_info.dynamic_loop_iter_counts.update(new_iters)

        # Same-step barrier: the stop DECISION must read this step's tokens, so
        # poll the deferred copy (or fall back to a counted blocking wait) before
        # computing stops. No deferred/late decision — that is the V1 identity
        # failure (a late stop decision reads the wrong step's tokens).
        if self._sidecar_checkstop:
            self._await_checkstop(cpu_output)

        new_stops = self._compute_new_stops(batch_N, engine, cpu_output)

        # Shadow (mandatory pre-perf): recompute the stop set from a
        # forced-synchronous D→H of the SAME GPU outputs and assert agreement.
        # Legacy stays authoritative — a bug surfaces as a logged mismatch +
        # counter, never a corrupted stream. A mismatch here means the deferred
        # copy event reported ready before the copy truly landed.
        if self._sidecar_checkstop_shadow:
            ref_output = self._prematerialize_for_check_stop(
                output,
                batch_fast=(batch_N.graph_walk == "thinker_decode"),
                talker_fast=(
                    self._fast_checkstop_talker
                    and batch_N.graph_walk == "talker_decode"
                ),
                defer=False,
            )
            ref_stops = self._compute_new_stops(batch_N, engine, ref_output)
            if ref_stops != new_stops:
                self._ws_inc("checkstop_shadow_mismatch")
                logger.warning(
                    "MSTAR_SIDECAR_CHECKSTOP shadow mismatch (walk=%s): "
                    "deferred=%s reference=%s",
                    batch_N.graph_walk, new_stops, ref_stops,
                )
                new_stops = ref_stops

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.stop_loops", synchronize=False)

        # Stop loops, if applicable
        for rid, loop_names in new_stops.items():
            # Only stop dyn loops that are actually active for this rid; if
            # none remain after filtering, there is nothing to stop.
            loop_names = {
                ln for ln in loop_names
                if self.worker_graphs_manager.check_dyn_loop(
                    rid, batch_N.partition, ln
                )
            }
            if not loop_names:
                continue
            # A loop stop makes this rid's next completion terminal (declared
            # loop outputs + filtered loop-back signals) rather than the
            # steady-state loop-back re-injection the plan was captured for.
            # Drop the plan so the terminal step (and any subsequent walk)
            # rebuilds from the slow path.
            if self._fast_postproc:
                self.tensor_manager.invalidate_populate_plan(rid)
            # MSTAR_FAST_ROUTE2: a loop stop makes the next completion
            # terminal — drop the route plan alongside the populate plan.
            self.worker_graphs_manager.invalidate_route_plan(rid)
            self.worker_graphs_manager.stop_loops(
                rid, partition=batch_N.partition,
                loop_names=loop_names,
                req_info=batch_N.node_batch.per_request_info[rid],
                last_node_run=batch_N.node_name
            )
            # W5-P2 mixed batch: record the stop under the rid's OWN walk, not
            # the batch-level "thinker_mixed", so the speculative overstay
            # dedup (which matches PendingLoopStop.graph_walk against a later
            # batch's real walk) can find it. Non-mixed batches are unchanged
            # (per-request walk == batch walk). Mixed batches are themselves
            # non-speculative, so this only matters for a later spec iter.
            stop_walk = batch_N.graph_walk
            if stop_walk == "thinker_mixed":
                stop_walk = batch_N.node_batch.per_request_info[rid].graph_walk
            self._pending_loop_stops.update([
                PendingLoopStop(
                    rid=rid,
                    graph_walk=stop_walk,
                    loop_name=name
                ) for name in loop_names
            ])

            # Send "loop done" messages to peer workers (small ZMQ msgs)
            stop_loop_workers: dict[str, set[str]] = {}
            for loop_name in loop_names:
                for worker in self.worker_graphs_manager.get_dyn_loop_workers(
                    rid, batch_N.partition, loop_name
                ):
                    stop_loop_workers.setdefault(worker, set()).add(loop_name)
            for worker, loop_names in stop_loop_workers.items():
                if worker == self.worker_id:
                    continue
                self.communicator.send(
                    entity_id=worker,
                    msg=WorkerMessage(
                        message_type=WorkerMessageType.STOP_LOOPS,
                        body=StopLoops(
                            request_id=rid,
                            loop_names=loop_names,
                            loop_stop_times=batch_N.node_batch.per_request_info[rid].loop_stop_times,
                            partition_name=batch_N.partition
                        )
                    )
                )

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.route_outputs", synchronize=False)
        # Mark nodes complete and route
        routing_per_request: dict[str, NodeOutputRouting] = {}
        per_request_uuids: dict[str, set[str]] = {}
        for rid, wg_id in batch_N.batch.request_to_worker_graph.items():
            # W5-P2 mixed batch: outputs must be stored + routed under each
            # request's OWN walk (thinker_decode / prefill_text), not the
            # batch-level "thinker_mixed". Tensor storage keys graph edges by
            # walk, and process_node_outputs looks up the next node's worker
            # graph via ``walk_node_to_worker_graph_id[(walk, next_node)]`` — a
            # "thinker_mixed" key doesn't exist, so routing would silently
            # misfire (edges treated as external, the decode/chunk loop never
            # advances). For every non-mixed batch this is byte-identical:
            # per_request_info[rid].graph_walk == batch_N.graph_walk.
            effective_walk = batch_N.graph_walk
            if effective_walk == "thinker_mixed":
                effective_walk = batch_N.node_batch.per_request_info[rid].graph_walk
            # Store output tensors before marking the node as complete so that
            # loop outputs can be buffered properly.
            req_output_tensors = output.per_request_output_tensors.get(rid)
            node = batch_N.batch.node_objects[rid]
            node.reset_outputs() # reset stale outputs
            if req_output_tensors:
                if self._fast_postproc:
                    # Memoized replay of the (rid, node, walk) store/populate
                    # derivation; falls back internally to the exact slow-path
                    # call below on any miss/mismatch and rebuilds the plan.
                    graph_node_info = (
                        self.tensor_manager.store_and_populate_graph_edges_fast(
                            request_id=rid,
                            tensors=req_output_tensors,
                            graph_edges=node.outputs,
                            node_name=node.name,
                            graph_walk=effective_walk,
                        )
                    )
                else:
                    graph_node_info = self.tensor_manager.store_and_populate_graph_edges(
                        request_id=rid,
                        tensors=req_output_tensors,
                        graph_edges=node.outputs,
                        node_name=node.name,
                        graph_walk=effective_walk,
                        skip_cuda_sync=True,
                        skip_ref_count=True,
                    )
                per_request_uuids[rid] = {
                    info.uuid for infos in graph_node_info.values() for info in infos
                }

            completion_output = self.worker_graphs_manager.mark_node_complete(
                rid, wg_id, batch_N.node_name
            )
            real_outputs = [edge.clone() for edge in completion_output.output_edges]

            routing_per_request[rid] = self.worker_graphs_manager.process_node_outputs(
                rid, node_name=batch_N.node_name,
                outputs=real_outputs, graph_walk=effective_walk
            )

            if rid in per_request_uuids:
                routing = routing_per_request[rid]

                for edge in routing.persist:
                    for info in edge.tensor_info:
                        self.tensor_manager.set_persist(
                            request_id=rid, uuid=info.uuid, persist=True
                        )

                # NOTE: routing.persist is not included here because the tensors are
                # (1) kept alive by the persist marker, and (2) would otherwise be
                #  double-counted (e.g., we should not be incrementing the refcount of
                # a persist signal that has EMPTY_DESTINATION; that's the conductor's
                # job to properly compute the reference when unpersisting the signal)
                routed_edges = (
                    routing.routed_to_this_worker_graph
                    + routing.emit_to_client
                    + routing.streaming_local
                    + sum(routing.to_workers.values(), start=[])
                    + sum(routing.streaming_to_workers.values(), start=[])
                )
                self.tensor_manager.set_output_ref_counts(
                    rid, per_request_uuids[rid], routed_edges
                )

        # Build the prematerialized new-token ints once, before register_outputs,
        # so both the SHM-skip decision (register_outputs) and the inline send
        # (_send_outputs) see the same per-rid prem dict. Restricted to the
        # Thinker text-decode walk: on Talker/Code2Wav steps this dict is pure
        # per-step overhead (their outputs route via streaming edges, never
        # via buffer_new_tokens/inline emit) and measurably regressed the
        # audio paths at batch (i2s B32 −17% flag-off, qb_queue1.log probes).
        # NOTE (W5-P2): a thinker_mixed batch does NOT take this fast inline
        # new-token emit — its decode rows still emit correctly via the normal
        # buffer path, just without the SHM-skip optimization. Extending prem to
        # the mixed batch's decode rows (chunk row has no new token unless last)
        # is a P3 perf follow-up; correctness is unaffected.
        _prem_walks = ("thinker_decode",)
        if batch_N.graph_walk in _prem_walks:
            prem_per_request: dict[str, dict[str, list[int]] | None] = {
                rid: self._prematerialized_new_tokens(cpu_output, rid)
                for rid in routing_per_request
            }
        else:
            prem_per_request = {rid: None for rid in routing_per_request}

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.register_outputs", synchronize=False)
        self._register_outputs(
            batch_N.batch, routing_per_request,
            prematerialized_per_request=prem_per_request,
        )

        # send outputs
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.send_outputs", synchronize=False)

        # The consuming pass (not the earlier ingest) reports the partition
        # done, so it rides this pass's WGD with the final output loop index.
        for rid in batch_N.node_batch.final_stream_rids:
            req_info = self.worker_graphs_manager.per_request_info.get(rid)
            if req_info is not None:
                req_info.per_partition_info[batch_N.partition].stream_partition_done = True

        # set this before send_outputs so that we can send updated profiling info to the conductor
        if self.enable_prof:
            self.profile_info.register_end(
                batch_N.node_batch.node_name,
                batch_N.node_batch.graph_walk,
                batch_N.node_batch.request_ids,
                batch_N.node_batch.exec_timings,
            )
        # MSTAR_BATCH_EMIT: collect every rid's qualifying inline emit results
        # into one list and send them as a single result_tensors_batch message
        # after the loop, instead of one result_tensors message per rid. When
        # off, batch_collector stays None and each rid sends its own messages
        # exactly as before (byte-identical).
        batch_collector: list[ResultTensors] | None = (
            [] if self._batch_emit else None
        )
        # MSTAR_EMIT_SIDECAR: rid entries for this step's record. Scoped
        # rids' emit/WGD work is diverted to _send_outputs_sidecar (record
        # fields); unscoped rids take the legacy path below untouched. A rid
        # is in exactly one population for its whole life (decided at
        # admission), so each (rid, name) stream stays on ONE FIFO and the
        # slim-template protocol holds on both.
        sidecar_entries: list | None = (
            [] if (self._sidecar_rids or self._sidecar_condemned) else None
        )
        # MSTAR_WGD_PACK: this step's conductor-bound WORKER_GRAPHS_DONE
        # messages, collected across every (non-sidecar-scoped) rid and
        # flushed as one packed send below instead of one send per rid.
        # Read once per step (a natural iteration boundary) so one step's
        # sends never split across the two wire
        # formats.
        wgd_pack_buffer: list[ConductorMessage] | None = (
            [] if self._wgd_pack else None
        )
        for rid, routing in routing_per_request.items():
            if sidecar_entries is not None and (
                rid in self._sidecar_rids or rid in self._sidecar_condemned
            ):
                # A condemned rid (sidecar died mid-flight) keeps its
                # worker-side effects (build_record=False) but its
                # client-bound record is dropped — it is being failed fast
                # via ABORT_REQUEST and its stream cannot be resumed.
                entry = self._send_outputs_sidecar(
                    rid, routing,
                    nested_loop_indices=per_req_nested_idxs[rid],
                    partition_name=batch_N.partition,
                    prematerialized_new_tokens=prem_per_request[rid],
                    node_speculatively_scheduled=batch_N.batch.node_objects[rid]._speculatively_scheduled,
                    build_record=(rid in self._sidecar_rids),
                )
                if entry is not None:
                    sidecar_entries.append(entry)
                continue
            self._send_outputs(
                rid, routing,
                nested_loop_indices=per_req_nested_idxs[rid],
                graph_walk=batch_N.graph_walk,
                partition_name=batch_N.partition,
                prematerialized_new_tokens=prem_per_request[rid],
                node_speculatively_scheduled=batch_N.batch.node_objects[rid]._speculatively_scheduled,
                batch_collector=batch_collector,
                wgd_pack_buffer=wgd_pack_buffer,
            )
        if wgd_pack_buffer:
            # A single-message buffer is sent unpacked (no PACKED envelope):
            # it is already 1 frame, so wrapping it would only add overhead.
            # PACKED is used exactly when it saves a frame (>=2 messages).
            if len(wgd_pack_buffer) == 1:
                self.communicator.send("conductor", wgd_pack_buffer[0])
            else:
                self.communicator.send(
                    "conductor",
                    ConductorMessage(
                        message_type=ConductorMessageType.PACKED,
                        body=PackedConductorMessage(messages=wgd_pack_buffer),
                    ),
                )
        if batch_collector:
            self.communicator.send(
                "api_server",
                APIServerMessage(
                    message_type="result_tensors_batch",
                    body=ResultTensorsBatch(items=batch_collector),
                ),
            )
        if sidecar_entries:
            # One compact record per step. A failed NOBLOCK
            # send is a sidecar failure: permanent fallback, never a block.
            if self._sidecar_client.send(
                self._sidecar_client.build_step(sidecar_entries)
            ):
                self._ws_inc("_sidecar_step_records")
            else:
                self._disable_sidecar("step record send failed")

        if self.enable_nvtx:
            range_pop(synchronize=False)

        return routing_per_request

    def _get_pinned_d2h_buffer(
        self,
        purpose: str,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        index: int = 0,
    ) -> torch.Tensor:
        key = (purpose, dtype, tuple(shape))
        buffers = self._pinned_d2h_buffers[key]
        while len(buffers) <= index:
            buffers.append(
                torch.empty(key[2], dtype=dtype, device="cpu", pin_memory=True)
            )
        return buffers[index]

    def _compute_new_stops(
        self, batch_N: PendingBatch, engine, cpu_output: NodeOutput,
    ) -> dict:
        """Stop-state COMPUTATION: pure int/counter compares
        over the prematerialized CPU tokens. Extracted verbatim from
        ``_postprocess_batch`` so MSTAR_SIDECAR_CHECKSTOP_SHADOW can recompute it
        against a forced-synchronous D→H. No CUDA reads here beyond ``.tolist()``
        on the already-copied pinned buffers — the copy must be complete before
        this is called (the caller's barrier guarantees it)."""
        flat = getattr(cpu_output, "_checkstop_flat", None)
        if (
            self._fast_checkstop_talker
            and batch_N.graph_walk == "talker_decode"
            and flat is not None
        ):
            # N1-Talker fast path: uniform talker_decode layer0_codes batch.
            # Semantics identical to TalkerSubmodule.check_stop (layer0 code ==
            # codec_eos, or iter+1 >= talker_max_tokens), but one tolist() covers
            # the batch and the compares are pure ints. No ignore_eos for the
            # talker — codec_eos is the only valid stop signal.
            self._ws_inc("talker_fast_checkstop_steps")
            tokens = flat.tolist()
            eos_id = self._talker_codec_eos_id
            if eos_id is None:
                submod = engine.submodule_management[
                    batch_N.node_name
                ].submodule
                eos_id = self._talker_codec_eos_id = (
                    submod.config.talker.codec_eos_token_id
                )
            new_stops = {}
            per_info = batch_N.node_batch.per_request_info
            for i, rid in enumerate(cpu_output._checkstop_rids):
                info = per_info.get(rid)
                if info is None:
                    continue
                max_tokens = info.step_metadata.get(
                    "talker_max_tokens", info.max_tokens
                )
                if (
                    (eos_id is not None and int(tokens[i]) == eos_id)
                    or info.dynamic_loop_iter_counts.get(
                        "talker_decode_loop", 0
                    ) + 1 >= max_tokens
                ):
                    new_stops[rid] = {"talker_decode_loop"}
            return new_stops
        if self._fast_checkstop and flat is not None:
            # N1 fast path: uniform thinker_decode new-token batch. Semantics
            # identical to ThinkerSubmodule.check_stop (token == im_end and
            # not ignore_eos, or iter+1 >= max_tokens), but one tolist()
            # covers the whole batch and the compares are pure ints.
            tokens = flat.tolist()
            eos_id = self._thinker_eos_id
            if eos_id is None:
                submod = engine.submodule_management[
                    batch_N.node_name
                ].submodule
                eos_id = self._thinker_eos_id = submod.config.im_end_token_id
            new_stops = {}
            per_info = batch_N.node_batch.per_request_info
            for i, rid in enumerate(cpu_output._checkstop_rids):
                info = per_info.get(rid)
                if info is None:
                    continue
                if (
                    (
                        int(tokens[i]) == eos_id
                        and not info.sampling_config["Thinker"].ignore_eos
                    )
                    or info.dynamic_loop_iter_counts.get(
                        "thinker_decode_loop", 0
                    ) + 1 >= info.max_tokens
                ):
                    new_stops[rid] = {"thinker_decode_loop"}
            return new_stops
        return engine.check_stop_for_batch(batch_N.node_batch, cpu_output)

    def _checkstop_barrier(
        self, side: "torch.cuda.Stream", defer: bool,
    ) -> "torch.cuda.Event | None":
        """Terminate the side-stream check_stop D→H (MSTAR_SIDECAR_CHECKSTOP).

        Legacy (``defer=False``): block the main thread on the copy exactly as
        before and return None — byte-identical to the flag-off path.

        Deferred (``defer=True``): record a reusable event on the side stream
        and return it WITHOUT blocking. The caller runs the cheap per-rid Python
        that follows (overlapping the in-flight copy) and then polls the event
        at ``_await_checkstop`` — ready => consume with no wait, else a counted
        blocking fallback. One event suffices: the copy is consumed before the
        next step records it again."""
        if not defer:
            side.synchronize()
            return None
        ev = self._checkstop_event
        if ev is None:
            ev = self._checkstop_event = torch.cuda.Event()
        ev.record(side)
        return ev

    def _await_checkstop(self, cpu_output: NodeOutput) -> None:
        """Barrier before the first read of a deferred check_stop copy
        (MSTAR_SIDECAR_CHECKSTOP). Poll the copy event: ready => consume this
        step with no wait (checkstop_deferred_consume); not ready => block on it
        (checkstop_sync_fallback) so the stop DECISION is still made this step
        from this step's tokens (the same-step rule — no deferred decision, the
        V1 identity trap). No-op when there was no deferred copy (non-CUDA path,
        or an early return in _prematerialize_for_check_stop)."""
        ev = getattr(cpu_output, "_checkstop_event", None)
        if ev is None:
            return
        if ev.query():
            self._ws_inc("checkstop_deferred_consume")
        else:
            ev.synchronize()
            self._ws_inc("checkstop_sync_fallback")

    def _prematerialize_for_check_stop(
        self,
        output: NodeOutput,
        batch_fast: bool = True,
        talker_fast: bool = False,
        defer: bool = False,
    ) -> NodeOutput:
        """Side-stream D→H of every CUDA tensor in
        ``output.per_request_output_tensors`` so the subsequent
        ``check_stop`` reads (typically ``.item()`` on the sampled token)
        don't trigger a default-stream sync. With same-thread async,
        GPU(N+1)'s kernels are already queued on default stream behind
        N's outputs by the time we get here — a default-stream sync would
        block waiting for N+1 to finish, defeating the overlap.

        Returns a fresh ``NodeOutput`` with CPU tensors for the per-rid
        outputs, sharing the original's allocation_failed / event fields.
        Skipped (returns ``output`` unchanged) when there's no completion
        event (CPU execution) or when CUDA is unavailable.

        AR engines emit small per-rid output dicts (sampled token + maybe
        a code) so the cost is negligible. If a future engine emits large
        tensors here (e.g. activations), revisit.
        """
        if not torch.cuda.is_available() or output.completion_event is None:
            return output
        if not output.per_request_output_tensors:
            return output

        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream(device=self.device)
        side = self._d2h_stream
        side.wait_event(output.completion_event)

        # Fast path: the common AR-decode shape is exactly one small
        # same-shaped tensor per rid under one key (new_token). Batch the
        # whole step into a single cat + one pinned D2H instead of a
        # per-rid copy loop (32 tiny copies/step at B32).
        per_rid = output.per_request_output_tensors
        rids = list(per_rid.keys())
        uniform_key: str | None = None
        # batch_fast gates the uniform-shape probe to the Thinker text-decode
        # walk: on Talker steps the per-step probe cost outweighs the copy
        # savings (audio-path regression, see qb_queue1.log attribution).
        if batch_fast and rids and all(
            isinstance(per_rid[r], dict)
            and len(per_rid[r]) == 1
            and isinstance(next(iter(per_rid[r].values())), list)
            and len(next(iter(per_rid[r].values()))) == 1
            and torch.is_tensor(next(iter(per_rid[r].values()))[0])
            and next(iter(per_rid[r].values()))[0].is_cuda
            and next(iter(per_rid[r].values()))[0].numel() == 1
            for r in rids
        ):
            keys = {next(iter(per_rid[r].keys())) for r in rids}
            dtypes = {next(iter(per_rid[r].values()))[0].dtype for r in rids}
            if len(keys) == 1 and len(dtypes) == 1:
                uniform_key = next(iter(keys))
        if uniform_key is not None:
            with torch.cuda.stream(side):
                flat_gpu = torch.cat(
                    [next(iter(per_rid[r].values()))[0].reshape(1) for r in rids]
                )
                flat_cpu = self._get_pinned_d2h_buffer(
                    "check_stop_flat", flat_gpu.shape, flat_gpu.dtype,
                )
                flat_cpu.copy_(flat_gpu, non_blocking=True)
            ev = self._checkstop_barrier(side, defer)
            cpu_fast: dict = {
                r: {uniform_key: [flat_cpu[i:i + 1]]}
                for i, r in enumerate(rids)
            }
            out = NodeOutput(
                per_request_output_tensors=cpu_fast,
                allocation_failed=output.allocation_failed,
                alloc_pages_short=output.alloc_pages_short,
                alloc_failed_request_id=output.alloc_failed_request_id,
                completion_event=output.completion_event,
            )
            out._checkstop_event = ev
            # N1 (MSTAR_FAST_CHECKSTOP): stash the flat pinned buffer + rid
            # order so check_stop can do ONE tolist() + int compares instead
            # of a per-rid .item() (+ attr chains) x bs.
            if uniform_key == "new_token":
                out._checkstop_flat = flat_cpu
                out._checkstop_rids = rids
            return out

        # N1-Talker (MSTAR_FAST_CHECKSTOP_TALKER): talker_decode analogue of the
        # thinker uniform probe above. The talker's per-rid output is multi-key
        # (talker_input_embeds + codec_tokens + layer0_codes), so the single-key
        # probe never matches it; check_stop only needs layer0_codes, so batch
        # JUST that scalar into one cat + one pinned D→H (the embeds/codec_tokens
        # are routed to Code2Wav from the GPU ``output`` and are never read off
        # this CPU copy, so we skip their D→H entirely). talker_fast is already
        # walk-gated (talker_decode) AND flag-gated by the caller.
        if talker_fast and rids and all(
            isinstance(per_rid[r], dict)
            and isinstance(per_rid[r].get("layer0_codes"), list)
            and len(per_rid[r]["layer0_codes"]) == 1
            and torch.is_tensor(per_rid[r]["layer0_codes"][0])
            and per_rid[r]["layer0_codes"][0].is_cuda
            and per_rid[r]["layer0_codes"][0].numel() == 1
            for r in rids
        ):
            dtypes = {per_rid[r]["layer0_codes"][0].dtype for r in rids}
            if len(dtypes) == 1:
                with torch.cuda.stream(side):
                    flat_gpu = torch.cat(
                        [per_rid[r]["layer0_codes"][0].reshape(1) for r in rids]
                    )
                    flat_cpu = self._get_pinned_d2h_buffer(
                        "check_stop_talker_flat", flat_gpu.shape, flat_gpu.dtype,
                    )
                    flat_cpu.copy_(flat_gpu, non_blocking=True)
                ev = self._checkstop_barrier(side, defer)
                # cpu_output only feeds check_stop; carry layer0_codes per rid so
                # a fall-through to engine.check_stop_for_batch (never taken on
                # the fast path) would still read a valid CPU token.
                cpu_fast_t: dict = {
                    r: {"layer0_codes": [flat_cpu[i:i + 1]]}
                    for i, r in enumerate(rids)
                }
                out = NodeOutput(
                    per_request_output_tensors=cpu_fast_t,
                    allocation_failed=output.allocation_failed,
                    alloc_pages_short=output.alloc_pages_short,
                    alloc_failed_request_id=output.alloc_failed_request_id,
                    completion_event=output.completion_event,
                )
                out._checkstop_event = ev
                out._checkstop_flat = flat_cpu
                out._checkstop_rids = rids
                return out

        cpu_per_rid: dict = {}
        buffer_indices: dict[tuple[str, torch.dtype, tuple[int, ...]], int] = defaultdict(int)
        with torch.cuda.stream(side):
            for rid, name_to_list in output.per_request_output_tensors.items():
                if not isinstance(name_to_list, dict):
                    cpu_per_rid[rid] = name_to_list
                    continue
                cpu_per_rid[rid] = {}
                for name, tensors in name_to_list.items():
                    if not isinstance(tensors, list):
                        cpu_per_rid[rid][name] = tensors
                        continue
                    new_list = []
                    for t in tensors:
                        if torch.is_tensor(t) and t.is_cuda:
                            key = ("check_stop", t.dtype, tuple(t.shape))
                            idx = buffer_indices[key]
                            buffer_indices[key] += 1
                            cpu_t = self._get_pinned_d2h_buffer(
                                "check_stop", t.shape, t.dtype, idx,
                            )
                            cpu_t.copy_(t, non_blocking=True)
                            new_list.append(cpu_t)
                        else:
                            new_list.append(t)
                    cpu_per_rid[rid][name] = new_list
        ev = self._checkstop_barrier(side, defer)

        out = NodeOutput(
            per_request_output_tensors=cpu_per_rid,
            allocation_failed=output.allocation_failed,
            alloc_pages_short=output.alloc_pages_short,
            alloc_failed_request_id=output.alloc_failed_request_id,
            completion_event=output.completion_event,
        )
        out._checkstop_event = ev
        return out

    def _apply_pending_removes_safe_to_drop(
        self, in_flight_rids: set[str]
    ) -> None:
        """Apply ``REMOVE_REQUEST`` for any rid that is not currently held by
        an in-flight GPU step. Removes for in-flight rids stay deferred and
        are reattempted next iter.

        A rid running on the side stream (MSTAR_SIDE_PREFILL) is also in-flight:
        its GPU work may still be reading/writing that rid's KV pages, so its
        REMOVE must stay deferred until the side batch is postprocessed. We
        union ``self._side_in_flight_rids`` here so every caller respects it
        without having to thread the side set through each call site."""
        held = in_flight_rids | self._side_in_flight_rids
        to_apply = [r for r in self._pending_removes if r not in held]
        for rid in to_apply:
            self._pending_removes.discard(rid)
            if self._slim_emit and self._slim_emit_sent:
                self._slim_emit_sent = {
                    k for k in self._slim_emit_sent if k[0] != rid
                }
            # MSTAR_SLIM_EMIT2 layout entries ride the same lifecycle; not
            # nested under _slim_emit.
            if self._slim_emit_loop_layout:
                self._slim_emit_loop_layout = {
                    k: v for k, v in self._slim_emit_loop_layout.items()
                    if k[0] != rid
                }
            self._remove_request(RemoveRequest(request_id=rid, source=MessageSource.SELF))

    def run(self) -> None:
        switch_interval = os.environ.get("MSTAR_PY_SWITCH_INTERVAL_SEC", "")
        if switch_interval:
            try:
                sys.setswitchinterval(float(switch_interval))
                logger.info(
                    "Worker %s: Python thread switch interval set to %ss",
                    self.worker_id,
                    switch_interval,
                )
            except ValueError:
                logger.warning(
                    "Worker %s: ignoring invalid MSTAR_PY_SWITCH_INTERVAL_SEC=%r",
                    self.worker_id,
                    switch_interval,
                )

        # Bound the load-time asymmetry between workers before any
        # subgroup NCCL collective fires inside the per-bs CUDA-graph
        # capture loop. Without this fence, a worker with a small model
        # (e.g. an 8B Talker) can finish loading, enter warmup, and hit
        # its first subgroup barrier while a worker with a 30B Thinker
        # is still streaming safetensors shards. The subgroup NCCL comm
        # is created lazily on that first collective; its connect-retry
        # budget is ~33 s, which is shorter than the load-time delta on
        # large multi-tower models. Syncing here means every worker
        # reaches warmup at the same wall-clock instant, so subgroup
        # bootstrap completes within the retry budget.
        self.parallel_groups.barrier_all()

        # Hot-load persisted compile artifacts (inductor/dynamo/autotune) before
        # the first torch.compile inside warmup_all(). Default off; a missing or
        # stale artifact degrades to a normal cold compile. See mega_cache.
        boot_phase("compile_start")
        load_mega_cache(self.worker_id)

        # CUDA graph capture before entering the main loop
        self.engine_manager.warmup_all()
        # All torch.compile + inductor + CUDA-graph capture is done here.
        boot_phase("capture_done")

        # Sync every worker before the main loop opens. Per-batch-size
        # captures inside CudaGraphRunner are already barriered on the
        # node-local TP group, but that doesn't bound the time between
        # ``warmup_and_capture`` returning and ``run()`` starting to
        # schedule. Without this fence, a TP leader can finish warmup
        # quickly, schedule its first batch, and ZMQ-send
        # ``ScheduleTPNode`` to a follower that's still inside another
        # engine's ``warmup``. The follower can't service the message
        # yet, but the leader will sit on the first NCCL collective.
        self.parallel_groups.barrier_all()

        # Setup (weight load + warmup + CUDA-graph capture) is complete. Tell
        # the conductor this worker is ready. The conductor blocks its main
        # loop until every worker reports in, so the API server only advertises
        # readiness once all workers can actually serve.
        self.communicator.send(
            "conductor",
            ConductorMessage(
                message_type=ConductorMessageType.SETUP_DONE,
                body=SetupDone(worker_id=self.worker_id),
            ),
        )
        boot_phase("ready")

        # Persist compile artifacts for the next boot. Done after SETUP_DONE so
        # the first (cold) boot's readiness isn't delayed by the write; skips
        # entirely once the artifact exists (steady-state boots do no I/O), so
        # only the first boot at a given git sha pays this. Default off.
        save_mega_cache(self.worker_id)

        # The async worker path needs decode submission to return quickly so
        # the main loop can overlap queue/tensor polling and post-processing
        # with GPU execution. Run the engine unconditionally on a dedicated
        # 1-worker GPU thread.
        gpu_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mstar-gpu-{self.worker_id}",
            initializer=self._init_cuda_executor_thread,
        )
        logger.info(
            "Worker %s: engine runs on dedicated GPU thread",
            self.worker_id,
        )
        # Dedicated thread that pre-plans FlashInfer attention for the
        # speculatively-built next batch. Runs concurrent with main thread's
        # await_gpu (which releases the GIL), so plan()'s Python work isn't
        # contended by main thread's fast/slow postprocess
        #
        # With double-buffered wrappers (CudaGraphRunner.NUM_SLOTS=2) and
        # advance_event signaling, plan(N+1) runs concurrent with replay(N)
        # on the disjoint slot — the actual GPU overlap. plan_executor waits
        # on prev_advance_event (signaled right after advance_seq_lens(N) on
        # the GPU thread, ~tens of µs into replay)
        #
        # Default ON. Set MSTAR_PRE_PLAN_SPEC=0 to fall back to the
        # double-buffer-without-pre-plan baseline.
        pre_plan_spec = os.environ.get("MSTAR_PRE_PLAN_SPEC", "1") == "1"
        plan_executor = None
        if pre_plan_spec:
            plan_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"mstar-plan-{self.worker_id}",
                initializer=self._init_cuda_executor_thread,
            )
            logger.info(
                "Worker %s: plan_executor enabled — speculative plan() "
                "pre-runs on a dedicated thread",
                self.worker_id,
            )
        # In-flight: (batch, node_batch, batch_partition, future) | None.
        pending: PendingBatch | None = None

        # MSTAR_SIDE_PREFILL: a second 1-worker executor + a separate CUDA
        # stream that runs a prefill/encoder batch CONCURRENTLY with the decode
        # chain. The side executor only EXECUTES pre-built batches; the main
        # thread still owns all scheduling and postprocess. The side stream is
        # created at the least CUDA priority the device supports, so decode
        # replays on the default stream win SM arbitration and prefill fills
        # the gaps (protecting decode ITL). We fall back to a default-priority
        # stream if the priority query is unavailable. Lazily gated so non-CUDA
        # workers and the flag-off path allocate nothing.
        #
        # No explicit shutdown: run() is a `while True` loop and gpu_executor /
        # plan_executor are likewise never shut down (they die with the
        # process). side_executor follows the same convention.
        side_executor: ThreadPoolExecutor | None = None
        pending_side: PendingSide | None = None
        if self._side_prefill:
            side_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"mstar-side-{self.worker_id}"
            )
            if torch.cuda.is_available():
                # In CUDA, a MORE-NEGATIVE priority = HIGHER priority; the
                # default stream is priority 0. There is no user priority
                # lower than the default, so the best we can do to keep decode
                # ahead is leave the side stream at default priority and rely
                # on decode being launched first each step. If a wider
                # negative range exists we still keep the side stream at the
                # least-priority (largest) value the device reports.
                side_priority = 0
                try:
                    least, _greatest = torch.cuda.get_stream_priority_range()
                    side_priority = least
                except Exception:
                    side_priority = 0
                try:
                    self._side_stream = torch.cuda.Stream(
                        device=self.device, priority=side_priority
                    )
                except Exception:
                    self._side_stream = torch.cuda.Stream(device=self.device)
            logger.info(
                "Worker %s: side-stream prefill substrate enabled "
                "(MSTAR_SIDE_PREFILL=%s, MSTAR_ENC_OVERLAP_V2=%s) — prefill/"
                "encoder batches run on a side stream concurrent with decode",
                self.worker_id,
                os.environ.get("MSTAR_SIDE_PREFILL", "0"),
                os.environ.get("MSTAR_ENC_OVERLAP_V2", "0"),
            )

        # MSTAR_SPEC_PEEK_FOR_FAIRNESS=1: only break the spec chain when
        # MicroScheduler.has_ready_excluding finds another (node, walk)
        # ready RIGHT NOW. Single-walk workers always speculate; multi-walk
        # workers yield only when there's actual contention.
        max_consecutive_spec = int(os.environ.get("MSTAR_MAX_CONSECUTIVE_SPEC_STEPS", "1024"))
        spec_peek_for_fairness = (
            os.environ.get("MSTAR_SPEC_PEEK_FOR_FAIRNESS", "1") == "1"
        )
        # MSTAR_MIXED_SPEC: fold a mixable prefill chunk INTO the running decode
        # spec chain (thinker_mixed spec batch) instead of breaking the chain to
        # run mixed on the non-spec path (0cc7c71). Read once — the flag is
        # static for the process. Implies MSTAR_MIXED_BATCH (mixed_batch_spec_
        # enabled already ANDs it). ``self.mixed_batch_assert`` is set in
        # __init__ from MSTAR_MIXED_BATCH_ASSERT.
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            mixed_batch_spec_enabled as _mixed_batch_spec_enabled,
        )
        mixed_spec_enabled = _mixed_batch_spec_enabled()
        consecutive_spec_steps = 0
        # EAGER FOLD (MSTAR_EAGER_FOLD, idea o1): steps since the last eager fold,
        # for the frequency cap (MSTAR_EAGER_FOLD_MIN_GAP). Start high so the
        # first eligible arrival folds immediately; reset to 0 when we break the
        # chain for an eager fold.
        steps_since_eager_fold = 1 << 30
        yield_away_from_target: tuple[str, str] | None = None
        # MSTAR_SCHED_PACK (b): fairness-peek exponential backoff state
        # (mirrors the fold-peek backoff — doubling skip window after
        # consecutive negative peeks, capped, reset on any positive peek or
        # fresh chain).
        fair_peek_skip = 0
        fair_peek_backoff = 0

        def _set_pending(p: PendingBatch):
            nonlocal pending
            pending = p
            self._in_flight_rids = set(p.batch.node_objects.keys()) if p else set()

        # Per-phase wall-clock instrumentation, gated by MSTAR_PHASE_TIMING.
        # When enabled, every Nth speculative iter logs a histogram so we can
        # see whether await_gpu time = "GPU still running" (overlap working)
        # vs "GPU done, idle" (overlap not paying off). Set the env var to a
        # positive integer = the dump period in iters (e.g. 200).
        phase_period = int(os.environ.get("MSTAR_PHASE_TIMING", "0") or "0")
        phase_buf: dict[str, list[float]] = defaultdict(list)
        phase_iter = [0]

        def _phase_record(name: str, dt: float) -> None:
            if phase_period > 0:
                phase_buf[name].append(dt)

        def _phase_flush() -> None:
            if phase_period <= 0 or phase_iter[0] % phase_period != 0:
                return
            samples = sorted((k, v) for k, v in phase_buf.items() if v)
            parts = []
            for name, vs in samples:
                vs.sort()
                n = len(vs)
                p50 = vs[n // 2] * 1000
                p95 = vs[min(n - 1, int(n * 0.95))] * 1000
                mean = (sum(vs) / n) * 1000
                parts.append(f"{name}: p50={p50:.2f}ms p95={p95:.2f}ms mean={mean:.2f}ms n={n}")
            logger.info(
                "Worker %s phase-timing iter=%d: %s",
                self.worker_id, phase_iter[0], " | ".join(parts),
            )
            phase_buf.clear()

        _watch_ctr = 0
        while True:
            from mstar.utils.profiler import range_pop, range_push
            try:
                _iter_start = _time.perf_counter() if phase_period else 0.0
                # EMIT_SIDECAR death watch, ~every 50 iters (~0.4 s at the B32
                # step). A dead sidecar (or an earlier HWM trip) flips us to the
                # legacy path with a CRITICAL log and fail-fast aborts — never a
                # hang.
                _watch_ctr += 1
                if _watch_ctr % 50 == 0:
                    if (
                        self._sidecar_client is not None
                        and not self._sidecar_client.healthy()
                    ):
                        self._disable_sidecar(
                            "sidecar process died or send queue hit HWM"
                        )
                self._apply_pending_removes_safe_to_drop(
                    self._in_flight_rids
                )

                # 1. CPU preamble — overlaps with GPU(N).
                # synchronize=False on every range so torch.cuda.synchronize()
                # doesn't drain the in-flight GPU work and undo the overlap.
                if self.enable_nvtx:
                    range_push("worker.process_messages", synchronize=False)
                self._process_messages()
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                if self.enable_nvtx:
                    range_push("worker.check_ready_tensors", synchronize=False)
                self._check_ready_tensors()
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                if self.enable_nvtx:
                    range_push("worker.poll_stream_buffers", synchronize=False)
                self._poll_stream_buffers()
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                # 1b. Reap a finished side-stream prefill (MSTAR_SIDE_PREFILL).
                # Opportunistic: never blocks. Postprocess (routing + token
                # emit) runs here on the main thread. No-op when the flag is
                # off or nothing is in flight.
                if self._side_prefill:
                    pending_side = self._reap_side_if_done(pending_side)

                # 2. Speculatively schedule + build N+1 — overlaps with GPU(N).
                # Only when (a) there's a pending step and (b) it's AR-engine.
                # For non-AR or non-loop-body steps, falls through to the
                # non-speculative path below (drain, then schedule).
                speculation = None
                yield_away_from_target = None

                if pending is not None and self._can_speculate(pending.batch):
                    # Fairness check (peek-based, replaces the old iter-
                    # counter cap): only break the spec chain when there's
                    # another (node, walk) actually ready to schedule on
                    # this worker. On single-walk workers (Orpheus LLM,
                    # Orpheus SNAC) this returns False and we always speculate.
                    if consecutive_spec_steps <= 1:
                        # Fresh chain (right after a break / admission) —
                        # contention is most likely here; always peek and
                        # restart the backoff ladder.
                        fair_peek_skip = 0
                        fair_peek_backoff = 0
                    if (
                        self._sched_pack
                        and spec_peek_for_fairness
                        and fair_peek_skip > 0
                    ):
                        # MSTAR_SCHED_PACK (b): during backoff the full
                        # ready-scan peek is skipped; a fairness yield (and
                        # any fold boundary it would trigger) is delayed by
                        # at most MSTAR_SCHED_PACK_PEEK_CAP steps.
                        fair_peek_skip -= 1
                        self._ws_inc("fair_peek_skip")
                        must_yield_for_fairness = False
                    else:
                        must_yield_for_fairness = (
                            spec_peek_for_fairness
                            and consecutive_spec_steps >= 1
                            and self.scheduler.has_ready_excluding(
                                self.worker_graphs_manager,
                                (pending.node_name, pending.graph_walk),
                            )
                        )
                        if (
                            self._sched_pack
                            and spec_peek_for_fairness
                            and consecutive_spec_steps >= 1
                        ):
                            if must_yield_for_fairness:
                                fair_peek_backoff = 0
                                fair_peek_skip = 0
                            else:
                                fair_peek_backoff = min(
                                    max(1, fair_peek_backoff * 2),
                                    self._sched_pack_peek_cap,
                                )
                                fair_peek_skip = fair_peek_backoff
                                self._ws_inc("fair_peek_neg")
                    must_yield_away = (
                        consecutive_spec_steps >= max_consecutive_spec
                        or must_yield_for_fairness
                    )

                    # MSTAR_ADMIT_FASTPATH (fix #4): arrival-triggered admission.
                    # A brand-new request (fwd_index==0, no KV state) waiting on
                    # its FIRST prefill walk should not sit behind the spec-chain
                    # yield gate -- today it waits for a fairness peek (subject to
                    # SPEC_PEEK_FOR_FAIRNESS + backoff) or the consecutive-spec
                    # ceiling (~8% of steps). When on, force a yield-away at THIS
                    # decision point so the next scheduled batch admits the new
                    # prefill. Read per-call from os.environ (like the spec knobs
                    # above) so MSTAR_DYNFLAGS can A/B it without a reboot; default
                    # off -> byte-identical. Composes with the mixed path below:
                    # if the new prefill is a foldable chunk the eager/mixed probe
                    # still resets must_yield_away and folds it into the spec batch
                    # (same-step admission either way). Only WHEN eligibility is
                    # evaluated changes -- get_next_batch and the mixed fold keep
                    # their own budget/bucket gates, so tokens are unaffected.
                    # MSTAR_COADMIT (fix #1) and MSTAR_ADMIT_FASTPATH (fix #4)
                    # both key off "is a BRAND-NEW request waiting on its first
                    # prefill?" Compute that peek AT MOST ONCE per decision and
                    # share it (has_new_request_ready is a Python ready-scan).
                    # Both flags are read per-call from os.environ so
                    # MSTAR_DYNFLAGS can A/B them without reboot; when both are off
                    # the peek is never called (byte-identical). COADMIT also
                    # requires the spec-fold path (mixed_spec_enabled): it co-
                    # admits by FOLDING the new chunk into the decode step, which
                    # only the MIXED_SPEC chain-fold machinery does.
                    _admit_fp_on = (
                        os.environ.get("MSTAR_ADMIT_FASTPATH", "0") == "1"
                    )
                    _coadmit_on = (
                        os.environ.get("MSTAR_COADMIT", "0") == "1"
                        and mixed_spec_enabled
                    )
                    # MSTAR_ENC_OVERLAP_V2 (arrival-triggered side overlap). Only
                    # meaningful with a live side substrate (executor + stream)
                    # and a FREE side slot this step — otherwise there is nowhere
                    # to overlap the prefill, so we leave the normal yield alone.
                    # Read per-call so MSTAR_DYNFLAGS can A/B it.
                    _enc_overlap_v2_on = (
                        os.environ.get("MSTAR_ENC_OVERLAP_V2", "0") == "1"
                        and self._side_prefill
                        and side_executor is not None
                        and pending_side is None
                    )
                    # V2 peeks even when we are ALREADY yielding for fairness —
                    # that is exactly the case it converts into a side overlap.
                    # COADMIT / ADMIT_FASTPATH only peek when NOT yielding.
                    _v2_peek = _enc_overlap_v2_on and must_yield_for_fairness
                    new_req_ready = False
                    if (
                        ((_admit_fp_on or _coadmit_on) and not must_yield_away)
                        or _v2_peek
                    ):
                        new_req_ready = self.scheduler.has_new_request_ready(
                            self.worker_graphs_manager,
                            (pending.node_name, pending.graph_walk),
                        )
                    # fix #4: force a yield-away so get_next_batch admits the new
                    # prefill STANDALONE at the next decision. fix #1 (below)
                    # instead FOLDS the new chunk into THIS decode step when it is
                    # foldable; the two compose (fold preferred, standalone else,
                    # no double-admission — the fold clears must_yield_away).
                    if _admit_fp_on and not must_yield_away and new_req_ready:
                        must_yield_away = True
                        self._ws_inc("_admit_fastpath")

                    # MSTAR_ENC_OVERLAP_V2 (fix: encoder->Thinker handoff). A
                    # brand-new prefill just became ready. In the encoff/PD split
                    # topology has_new_request_ready() flips True *exactly* when
                    # the encoder's embeds arrive cross-rank at the Thinker (the
                    # prefill node is not ready until _check_ready_tensors has
                    # received them), so this IS the encoder-completion trigger.
                    # Instead of breaking the decode spec chain to run that
                    # prefill standalone (freezing the in-flight decodes for the
                    # prefill step), UNDO the fairness yield here: the decode
                    # chain keeps speculating on the default stream and section 3b
                    # dispatches the just-arrived prefill onto the side stream,
                    # so encode (rank 0) AND prefill (rank 1 side stream) both
                    # overlap decode continuation. Never suppresses the
                    # consecutive-spec ceiling (that stays the starvation
                    # backstop); guarded on a free side slot (checked in
                    # _enc_overlap_v2_on) so the side dispatch can actually take
                    # the prefill this step. ADMIT_FASTPATH's standalone yield, if
                    # both flags are set, still wins (it ran just above); V2 only
                    # acts on a still-standing fairness yield.
                    if (
                        _v2_peek
                        and new_req_ready
                        and must_yield_for_fairness
                        and consecutive_spec_steps < max_consecutive_spec
                    ):
                        must_yield_for_fairness = False
                        must_yield_away = (
                            consecutive_spec_steps >= max_consecutive_spec
                        )
                        self._ws_inc("_enc_overlap_v2_defer")

                    # Mixed batch: the ready contending work is a mixable
                    # prefill CHUNK on the decode's own node. Do NOT yield-away
                    # to a prefill-only step (yield-away schedules with
                    # exclude_target=decode while the decode rids are still
                    # _speculatively_scheduled and off the ready queue, so
                    # _try_assemble_mixed can never see the decode side there —
                    # which is why "thinker_mixed step" never fired on yield).
                    # Two ways to admit the chunk instead:
                    #
                    # * MSTAR_MIXED_SPEC off (0cc7c71): break the spec chain
                    #   WITHOUT yield-away — fall through to the non-speculative
                    #   path (section 4), where the in-flight decode completes,
                    #   un-flags, re-queues, and the plain get_next_batch
                    #   assembles decode + chunk into a thinker_mixed batch (mixed
                    #   is non-speculative there — see _can_speculate). Loses the
                    #   overlap for that step (measured 4-9%/admission).
                    #
                    # * MSTAR_MIXED_SPEC on: keep speculating the decode
                    #   continuation and fold the chunk row INTO that spec batch
                    #   (thinker_mixed) below, so the mixed step rides the chain
                    #   uninterrupted — no chain break, overlap preserved.
                    #
                    # No mixed opportunity → unchanged yield-away either way.
                    # Fold trigger. Default (P2): only at a must_yield_away
                    # boundary — the fold replaces the yield. EAGER
                    # (MSTAR_MIXED_SINGLE_CHUNK): attempt at EVERY chain step.
                    # With single-chunk on, every admission needs ~one fold
                    # slot per prompt walk; yield-boundary-only folding drains
                    # chunks ~8x slower than standalone prefill would and
                    # starves decode occupancy (measured 6.18 -> 3.48 req/s).
                    # The occupancy floor inside has_mixed_opportunity keeps
                    # ramp-up on the standalone path either way.
                    speculate_into_mixed = False
                    # Eager peeks (every chain step under an eager policy —
                    # MSTAR_MIXED_SINGLE_CHUNK or MSTAR_MIXED_BUDGET_TOKENS) scan
                    # the ready queues in Python. During a long pure-decode tail
                    # that's thousands of guaranteed-negative scans (~3400 peeks
                    # for 508 folds measured). Exponential backoff after negatives
                    # (1..32 steps) bounds the waste; a fold is delayed by at most
                    # the backoff, no worse than waiting for a natural yield
                    # boundary. must_yield_away peeks always run (rare; picking
                    # fold over yield there is the original P2 win).
                    # Eager (every-step) probing fires under the graveyard
                    # single-chunk flag OR the V2 budget policy. They differ in
                    # what makes a chunk foldable: single-chunk ALSO routes short
                    # standalone prefills through the chunk planner (the occupancy
                    # tax that closed it); the budget touches NOTHING on the
                    # admission side — it only folds chunks that already exist,
                    # capped at MSTAR_MIXED_BUDGET_TOKENS total tokens.
                    eager_probe = mixed_spec_enabled and (
                        self.mixed_single_chunk or self._mixed_budget_tokens > 0
                    )
                    # MSTAR_COADMIT (fix #1): a brand-new request's first chunk
                    # must co-admit into THIS decode step (vLLM's unified per-step
                    # admission), not wait for a backoff window or the occupancy
                    # floor. When a new request is ready, force the probe on this
                    # step and clear the negative-peek backoff so the fold is
                    # evaluated NOW; bypass_floor (below) lets it fold even at low
                    # decode occupancy. The fold still flows through the unchanged
                    # pop + per-request gates, so KV/capture safety is intact.
                    coadmit_fold = _coadmit_on and new_req_ready
                    if coadmit_fold:
                        eager_probe = True
                        self._peek_backoff = 0
                        self._peek_skip = 0
                        self._ws_inc("_coadmit_probe")
                    elif eager_probe and not must_yield_away and self._peek_skip > 0:
                        self._peek_skip -= 1
                        eager_probe = False
                        self._ws_inc("_n_peek_skipped")
                    fold_probe = must_yield_away or eager_probe
                    # Fold-decision token budget. MSTAR_COADMIT overrides the V2
                    # budget with its clamped unified budget (see
                    # _compute_coadmit_budget) — never MORE restrictive than V2 for
                    # a valid fold, and IMA-safe by construction. Passed to BOTH
                    # the peek and the pop so they agree on over-budget.
                    _fold_budget = (
                        self._coadmit_budget_tokens
                        if _coadmit_on
                        else self._mixed_budget_tokens
                    )
                    _peek_t0 = (
                        _time.perf_counter()
                        if (fold_probe and self._walk_stats is not None)
                        else 0.0
                    )
                    _n_decode_pending = len(pending.node_batch.request_ids)
                    _peek_hit = fold_probe and self.scheduler.has_mixed_opportunity(
                        self.worker_graphs_manager,
                        (pending.node_name, pending.graph_walk),
                        n_decode=_n_decode_pending,
                        budget_tokens=_fold_budget,
                        bypass_floor=coadmit_fold,
                    )
                    # V2 telemetry: a budget-accelerated fold fires on a step that
                    # was NOT a yield boundary (P2 would have stayed pure-decode).
                    # Capture before _peek_hit clears must_yield_away below.
                    _budget_fold = (
                        self._mixed_budget_tokens > 0
                        and eager_probe
                        and not must_yield_away
                    )
                    # Budget peek missed because the decode side is under the
                    # occupancy floor — the second graveyard anti-lesson, made
                    # visible (WALK_STATS budget_skips_floor). Only computed when
                    # WALK_STATS is on (the _mixed_min_decode read is otherwise
                    # skipped).
                    if (
                        self._walk_stats is not None
                        and _budget_fold
                        and not _peek_hit
                        and _n_decode_pending < self.scheduler._mixed_min_decode()
                    ):
                        self._ws_inc("budget_skips_floor")
                    if fold_probe:
                        if _peek_hit:
                            self._peek_backoff = 0
                            self._peek_skip = 0
                        else:
                            self._peek_backoff = min(
                                max(1, self._peek_backoff * 2), 32
                            )
                            self._peek_skip = self._peek_backoff
                    if _peek_t0:
                        # Eager-fold peek cost: a Python ready-queue scan per
                        # chain step. _ms_peek/_n_peek size it.
                        self._walk_stats["_ms_peek"] = self._walk_stats.get(
                            "_ms_peek", 0
                        ) + int((_time.perf_counter() - _peek_t0) * 1000)
                        self._ws_inc("_n_peek")
                    if _peek_hit:
                        must_yield_away = False
                        self._ws_inc("_mix_opp")
                        if mixed_spec_enabled:
                            speculate_into_mixed = True
                            break_chain_for_mixed = False
                        else:
                            break_chain_for_mixed = True
                    else:
                        break_chain_for_mixed = False

                    # EAGER FOLD (MSTAR_EAGER_FOLD, idea o1). The captured mixed
                    # fold above (_peek_hit, C<=512) could not claim this step for
                    # this arrival — either no chunk was ready or its chunk is
                    # LARGER than the captured cap. A brand-new large chunk
                    # (512 < C <= MSTAR_EAGER_FOLD_MAX_CHUNK) would otherwise run
                    # STANDALONE and freeze the concurrent decodes for that step
                    # (the i2t TTFT tail vLLM avoids by folding a full prefill
                    # EAGER). Break the decode chain and ARM the scheduler so the
                    # next get_next_batch (non-spec path, this same iteration once
                    # pending un-flags) assembles decode + the large chunk into a
                    # thinker_mixed step whose token count matches NO captured
                    # graph -> execute_forward runs it eager (_execute_batched).
                    # Overrides a pending yield-away (incl. ADMIT_FASTPATH's) for
                    # this arrival: folding beats standalone. Gated to arrivals
                    # (fwd_index==0, inside the peek) and throttled to one per
                    # MIN_GAP spec steps (an eager step is slower than a replay, so
                    # spacing protects decode throughput). Off -> the peek returns
                    # False, so byte-identical.
                    steps_since_eager_fold += 1
                    if (
                        not speculate_into_mixed
                        and not break_chain_for_mixed
                        and os.environ.get("MSTAR_EAGER_FOLD", "0") == "1"
                    ):
                        _ef_gap = int(
                            os.environ.get("MSTAR_EAGER_FOLD_MIN_GAP", "8") or "8"
                        )
                        if (
                            steps_since_eager_fold >= _ef_gap
                            and self.scheduler.has_eager_fold_opportunity(
                                self.worker_graphs_manager,
                                (pending.node_name, pending.graph_walk),
                            )
                        ):
                            must_yield_away = False
                            break_chain_for_mixed = True
                            self.scheduler._eager_fold_armed = True
                            steps_since_eager_fold = 0
                            self._ws_inc("_eager_fold")

                    if not must_yield_away and not break_chain_for_mixed:
                        if self.enable_nvtx:
                            range_push("worker.speculate", synchronize=False)
                        _t0 = _time.perf_counter() if phase_period else 0.0
                        speculation = self._try_speculate_next(pending)
                        self._ws_inc(
                            "_spec_ok" if speculation is not None else "_spec_none"
                        )
                        # Fold a ready mixable chunk into the decode continuation
                        # so the next spec batch is a thinker_mixed step that
                        # rides the chain. If no decode continuation survived
                        # (speculation is None — e.g. every decode rid stopping),
                        # there is nothing to ride the chain: fall back to the
                        # 0cc7c71 non-spec mixed path (break_chain_for_mixed) so
                        # the chunk still gets mixed, just off-chain.
                        if speculate_into_mixed:
                            if speculation is not None:
                                folded_len = self._try_fold_mixed_chunk_into_spec(
                                    speculation,
                                    budget_tokens=_fold_budget,
                                )
                                folded = folded_len is not None
                                self._ws_inc("_fold_ok" if folded else "_fold_miss")
                                if folded and coadmit_fold:
                                    # A fold that co-admitted a brand-new request
                                    # this step (would have gone standalone / waited
                                    # under the floor or backoff without COADMIT).
                                    self._ws_inc("_coadmit_fold")
                                if folded and _budget_fold:
                                    # V2 counters: folds the budget policy caused
                                    # (would not have happened at a yield boundary)
                                    # and the chunk tokens they admitted.
                                    self._ws_inc("budget_folds")
                                    if self._walk_stats is not None:
                                        self._walk_stats["budget_fold_tokens"] = (
                                            self._walk_stats.get(
                                                "budget_fold_tokens", 0
                                            )
                                            + int(folded_len)
                                        )
                                if (
                                    not folded
                                    and self.mixed_batch_assert
                                ):
                                    # Peek said a chunk was ready; a lost race is
                                    # tolerable, but under the assert flag surface
                                    # a persistent miss so it can't hide.
                                    logger.warning(
                                        "MIXED_SPEC: fold missed a peeked chunk "
                                        "opportunity (raced removal?)"
                                    )
                            else:
                                break_chain_for_mixed = True
                                self._ws_inc("_fold_lost_chain")
                        if phase_period:
                            _phase_record("speculate", _time.perf_counter() - _t0)
                        if self.enable_nvtx:
                            range_pop(synchronize=False)
                    if speculation is None and not break_chain_for_mixed:
                        yield_away_from_target = (
                            pending.node_name,
                            pending.graph_walk,
                        ) if must_yield_away else None
                        batch = self.scheduler.get_next_batch(
                            self.worker_graphs_manager,
                            exclude_target=yield_away_from_target,
                        )
                        if batch is not None:
                            node_batch = self._build_node_batch(batch)
                            batch_partition = self.worker_graphs_manager.get_partition_for_node(batch.node_name)
                            logger.debug(f"Yield away: {batch.node_name} {node_batch.request_ids}")
                            self._ws_inc("_yield_away")
                            speculation = Speculation(
                                scheduled_batch=batch,
                                node_batch=node_batch,
                                consumed_edges=set(),
                                continuing_rids=set(), # n/a
                                partition=batch_partition,
                                is_new_iter=False,
                                is_same_node=False,
                                is_yield_away=True
                            )

                            # send messages to follower ranks if relevant
                            self.maybe_send_zmq_to_tp_followers(node_batch)
                    if speculation is not None:
                        # Reserve the double-buffer slot for batch_(N+1) NOW
                        # so both pre-plan and replay (queued below) target
                        # the SAME slot — and the OPPOSITE slot from
                        # batch_N's in-flight replay. The reservation lives
                        # on spec_node_batch.metadata['cuda_graph_slot'];
                        # the engine forwards it to the runner.
                        if speculation is not None:
                            engine = self.engine_manager.get_engine(
                                speculation.node_batch.node_name
                            )
                            engine.reserve_replay_slot(speculation.node_batch)

                        # Kick off pre-planning on the plan_executor NOW —
                        # its Python work runs while the main thread is in
                        # await_gpu (releases GIL). plan_executor waits on
                        # prev's advance_event so  plan(N+1) starts before
                        # replay(N) finishes. replay(N) keeps running on
                        # the active slot.
                        if speculation is not None and plan_executor is not None:
                            engine = self.engine_manager.get_engine(
                                speculation.node_batch.node_name
                            )
                            prev_advance_event_for_plan: threading.Event | None = None
                            if pending is not None:
                                prev_advance_event_for_plan = (
                                    pending.node_batch.metadata.get("advance_event")
                                )
                            speculation.plan_future = plan_executor.submit(
                                self._pre_plan_for_speculative_batch,
                                engine,
                                speculation.node_batch,
                                prev_advance_event_for_plan,
                            )

                # 3. If pending: await GPU(N), submit speculated GPU(N+1)
                # asap, then post-process N (fast then slow) overlapping
                # with GPU(N+1).
                spec_pending = None
                if pending is not None:
                    if self.enable_nvtx:
                        range_push("worker.await_gpu", synchronize=False)
                    _t0 = _time.perf_counter() if phase_period else 0.0
                    output: NodeOutput = pending.future.result()
                    if phase_period:
                        _phase_record("await_gpu", _time.perf_counter() - _t0)
                    if self.enable_nvtx:
                        range_pop(synchronize=False)

                    # set node._speculatively_scheduled to false, since
                    # the node has just completed
                    for node in pending.batch.node_objects.values():
                        node._speculatively_scheduled = False

                    if output.allocation_failed:
                        # KV-cache OOM on pending. ``_handle_allocation_failure``
                        # offloads or holds the failed rids and pushes their
                        # GraphNodes back to the scheduler queue.
                        self._handle_allocation_failure(
                            pending.batch, pending.node_batch
                        )
                        for node in pending.batch.node_objects.values():
                            node._speculatively_scheduled = False
                        # Speculation cleanup splits by kind:
                        #
                        # * Non-yield-away spec depended on pending's outputs
                        #   (its inputs would be threaded from ``output`` below
                        #   via ``_thread_outputs_to_speculative``). Pending's
                        #   output is invalid, so the spec batch can't run.
                        #
                        # * Yield-away spec is independent of pending.
                        #   But ``_handle_allocation_failure`` may have shifted
                        #   the engine's KV-cache state (paused/offloaded rids),
                        #   so reset pre-plan.
                        if speculation is not None:
                            if speculation.plan_future is not None:
                                speculation.plan_future.result()
                                self._reset_skip_plan_flags(
                                    speculation.node_batch
                                )
                                speculation.plan_future = None
                            if not speculation.is_yield_away:
                                for rid, edges in speculation.consumed_streaming_edges.items():
                                    for edge in edges:
                                        self._return_speculative_streaming_edge(rid, edge)
                                speculation = None

                    if speculation is not None:
                        spec_batch = speculation.scheduled_batch
                        spec_node_batch = speculation.node_batch
                        # Promote per-rid speculative_signals → real inputs
                        if not speculation.is_yield_away:
                            self._thread_outputs_to_speculative(speculation, output)
                        # set node._speculatively_scheduled to true, so that it doesn't
                        # accidentally get put on the ready queue while already executing
                        for node in spec_batch.node_objects.values():
                            # this does not include the dropped rids
                            node._speculatively_scheduled = True

                        if spec_batch.node_objects:
                            if self.enable_nvtx:
                                range_push("worker.submit_spec", synchronize=False)
                            _t0 = _time.perf_counter() if phase_period else 0.0
                            # If pre-plan was dispatched but the spec_batch
                            # composition changed, fall back to inline planning
                            if speculation.plan_future is not None and speculation.dropped:
                                speculation.plan_future.result()
                                self._reset_skip_plan_flags(speculation.node_batch)
                                speculation.plan_future = None

                            # MSTAR_MIXED_PREPLAN validation + counter. A folded
                            # mixed step is preplanned when its pre-plan future
                            # survived (not reset by the drop path above) AND a
                            # packed slot was reserved; otherwise it plans inline
                            # (no slot match, or dropped membership). Under the
                            # assert flag, a live pre-plan future MUST report
                            # applied=True — a False there means the packed
                            # reserve/pre-plan silently missed and the run path
                            # will inline-plan without us noticing, which is the
                            # exact regression this hook guards against.
                            if (
                                self.mixed_batch_preplan
                                and spec_node_batch.metadata.get("mixed_preplan")
                                is not None
                            ):
                                slot_reserved = (
                                    "cuda_graph_slot" in spec_node_batch.metadata
                                )
                                if (
                                    speculation.plan_future is not None
                                    and slot_reserved
                                ):
                                    self._mixed_preplan_count += 1
                                    if self.mixed_batch_assert:
                                        applied = speculation.plan_future.result()
                                        assert applied, (
                                            "MIXED_PREPLAN: pre-plan future "
                                            "returned not-applied for a folded "
                                            "mixed step that reserved a packed "
                                            "slot — run path will inline-plan"
                                        )
                                else:
                                    self._mixed_inline_count += 1
                                if (
                                    self._mixed_preplan_count
                                    + self._mixed_inline_count
                                ) % 200 == 0:
                                    logger.info(
                                        "Worker %s mixed-preplan: preplanned=%d "
                                        "inline=%d",
                                        self.worker_id,
                                        self._mixed_preplan_count,
                                        self._mixed_inline_count,
                                    )

                            # Attach a fresh advance_event to this batch so
                            # the NEXT iter's plan_executor can gate on
                            # advance_seq_lens(THIS batch).
                            spec_advance_event = threading.Event()
                            spec_node_batch.metadata["advance_event"] = spec_advance_event

                            # Block the main thread until the GPU executor
                            # thread is about to launch CUDA kernels (set
                            # deep in the engine: before graph.replay() in
                            # CudaGraphRunner, or before forward/forward_batched
                            # in the eager AR path).
                            spec_launch_started_event = threading.Event()
                            spec_node_batch.metadata["launch_started_event"] = spec_launch_started_event
                            spec_future = gpu_executor.submit(
                                self._execute_on_gpu_thread,
                                spec_batch, spec_node_batch,
                                speculation.plan_future,
                                spec_advance_event,
                            )
                            self.wakeup_event.register_future(spec_future)
                            if self.enable_nvtx:
                                range_pop(synchronize=False)
                                range_push("worker.gpu_submit_queued", synchronize=False)
                            spec_launch_started_event.wait(timeout=0.005)
                            if phase_period:
                                _phase_record("submit_spec", _time.perf_counter() - _t0)
                            if self.enable_nvtx:
                                range_pop(synchronize=False)
                            spec_pending = PendingBatch(
                                batch=spec_batch,
                                node_batch=spec_node_batch,
                                node_name=spec_batch.node_name,
                                partition=speculation.partition,
                                graph_walk=spec_batch.graph_walk,
                                future=spec_future,
                                speculative_new_iter=speculation.is_new_iter,
                                loop_name=speculation.loop_name,
                            )
                        elif speculation.plan_future is not None:
                            # All continuing rids were dropped post-thread,
                            # so no spec batch was submitted. Drain the
                            # orphaned pre-plan future and reset the engine's
                            # skip flags so the next plan_attention call
                            # recomputes from scratch instead of trusting
                            # stale wrapper buffers from this aborted spec.
                            speculation.plan_future.result()
                            self._reset_skip_plan_flags(speculation.node_batch)

                    # Post-process N (routing stage) — runs concurrently with
                    # GPU(N+1) if we submitted one above. Skipped on
                    # allocation_failed since the output tensors aren't valid;
                    # ``_handle_allocation_failure`` already rehabilitated the
                    # failed rids upstream.
                    _t0 = _time.perf_counter() if phase_period else 0.0

                    if not output.allocation_failed:
                        if self.enable_nvtx:
                            range_push("worker.postprocess_batch", synchronize=False)
                        self._postprocess_batch(pending, output)
                        if self.enable_nvtx:
                            range_pop(synchronize=False)

                    # Removes for any rid not in the in-flight spec step
                    # are safe to apply now.
                    in_flight = set(spec_pending.batch.node_objects.keys()) if spec_pending else set()
                    self._apply_pending_removes_safe_to_drop(in_flight)
                    _set_pending(None)

                if spec_pending is not None:
                    if speculation.is_yield_away:
                        consecutive_spec_steps = 0
                    else:
                        consecutive_spec_steps += 1
                    # 3b. Dispatch a prefill/encoder batch onto the side stream
                    # to overlap the decode chain (MSTAR_SIDE_PREFILL). Only
                    # when the chain we just queued is a genuine (non-yield-away)
                    # decode/AR chain — a yield-away step is itself a handoff to
                    # another node, so there's no decode chain to overlap. The
                    # exclude_target keeps the scheduler off the active decode
                    # group; get_next_batch pops the prefill nodes so the
                    # decode speculation next iter won't re-see them.
                    if self._side_prefill and not speculation.is_yield_away:
                        pending_side = self._maybe_dispatch_side(
                            pending_side,
                            side_executor,
                            (spec_pending.node_name, spec_pending.graph_walk),
                        )
                    if phase_period:
                        _phase_record("iter_total", _time.perf_counter() - _iter_start)
                        phase_iter[0] += 1
                        _phase_flush()
                    _set_pending(spec_pending)
                    continue
                consecutive_spec_steps = 0

                # 4. Non-speculative path: no pending or speculation skipped
                # (e.g., non-AR engine, or loop ended). Run MicroScheduler.
                #
                # Chain-break drain (MSTAR_SIDE_PREFILL): the decode chain has
                # broken, so the get_next_batch below may schedule a decode
                # batch that reads KV pages a side prefill is still writing on
                # the side stream — with no ordering between the two streams.
                # Block on any in-flight side prefill and route it BEFORE
                # scheduling new default-stream work.
                if self._side_prefill and pending_side is not None:
                    self._drain_side(pending_side)
                    pending_side = None
                if self.enable_nvtx:
                    range_push("worker.schedule", synchronize=False)
                batch = None
                if yield_away_from_target is not None:
                    batch = self.scheduler.get_next_batch(
                        self.worker_graphs_manager,
                        exclude_target=yield_away_from_target,
                    )
                if batch is None:
                    batch = self.scheduler.get_next_batch(self.worker_graphs_manager)
                if self.enable_nvtx:
                    range_pop(synchronize=False)
                if batch is None:
                    self.communicator.wait_for_work(10)
                    continue

                if self.enable_nvtx:
                    range_push("worker.build_node_batch", synchronize=False)
                node_batch = self._build_node_batch(batch)
                batch_partition = self.worker_graphs_manager.get_partition_for_node(batch.node_name)

                for request_id, req_info in node_batch.per_request_info.items():
                    req_info.dynamic_loop_iter_counts.update(
                        self.worker_graphs_manager.get_dynamic_loop_iters(
                            request_id, partition=batch_partition,
                        )
                    )
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                # Reserve the double-buffer slot on the main thread before
                # submission so the per-key counter advances in main-thread
                # order. Without this, the GPU thread would advance the
                # counter at run time and races with main-thread reservations
                # from later iters.
                fallthrough_engine = self.engine_manager.get_engine(batch.node_name)
                fallthrough_engine.reserve_replay_slot(node_batch)

                # send messages to follower ranks if relevant
                self.maybe_send_zmq_to_tp_followers(node_batch)

                # Attach a fresh advance_event so the next iter's
                # plan_executor (if it speculates) can wait on this batch's
                # advance_seq_lens.
                fallthrough_advance_event = threading.Event()
                node_batch.metadata["advance_event"] = fallthrough_advance_event
                future = gpu_executor.submit(
                    self._execute_on_gpu_thread, batch, node_batch,
                    None, fallthrough_advance_event,
                )
                self.wakeup_event.register_future(future)
                logger.debug(f"Scheduling: {batch.node_name} {node_batch.request_ids}")
                _set_pending(PendingBatch(
                    batch=batch,
                    node_batch=node_batch,
                    node_name=batch.node_name,
                    partition=batch_partition,
                    graph_walk=batch.graph_walk,
                    future=future
                ))
            except Exception:
                logger.exception("Worker %s error in main loop", self.worker_id)
                sleep(0.01)
