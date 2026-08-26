"""Virtual-time discrete-event simulator for an mstar deployment.

The design rule is that **semantics are imported and costs are measured**.
Everything that decides *what runs when* comes from mstar's own code:

* graph readiness, loop iteration, and EOS handling — the real
  :class:`~mstar.graph.graph_io.WorkerGraphIO`, one deep copy per
  (request, worker graph), exactly as the worker does;
* placement — the real ``get_worker_graphs`` over the deployment YAML;
* streaming cadence — the model's own ``ChunkPolicy`` objects.

What is modeled rather than imported is *how long things take*: the worker's
two-lane (GPU / CPU) pipeline, the conductor hop, and tensor transfers. Step
costs are looked up in the measured stepdb.

## Walk sequencing

Which walk follows which is the model's decision, taken through the same two
calls the conductor makes — ``get_initial_forward_pass_args`` to open a
request and ``get_partition_forward_pass_args`` after each completed pass —
and applied per partition, since a backbone can be mid-decode while a codec
partition is still draining an earlier chunk. The inputs those calls return
are real ``GraphEdge`` objects, so seeding a walk is the model's decision too
rather than a guess about which nodes look like entry points.

The only thing layered on top is the conductor's own length cap, which stands
in for EOS: a simulator has no sampled tokens to inspect, so a request stops
at the workload's target length — what a measured run with a pinned
``max_tokens`` produces.

"""

from __future__ import annotations

import copy
import heapq
import itertools
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

from mstar.graph.special_destinations import EMIT_TO_CLIENT, SPECIAL_DESTINATIONS
from mstar.sim.deployment import Deployment, SimWorkerGraph
from mstar.sim.stepdb import Coverage, StepCost, StepDB, StepKey, pad_to_bucket

logger = logging.getLogger(__name__)


# ── Tunables measured from a real deployment ─────────────────────────────

@dataclass
class TimingModel:
    """Costs that are not per-step engine work.

    Defaults are order-of-magnitude placeholders drawn from the code's own
    constants (the conductor's 1 ms poll, the worker's 10 ms idle poll). The
    calibration step overwrites them from a real run; a simulation that never
    calibrated says so in its report rather than pretending these are
    measurements.
    """

    #: Conductor poll latency. Its loop sleeps 1 ms between drains, so a
    #: message waits ~half that on average.
    conductor_hop_s: float = 0.0005
    #: Fixed CPU cost the worker pays per step outside the engine: output
    #: routing, tensor store bookkeeping, ZMQ sends.
    worker_step_overhead_s: float = 0.0002
    #: One-way control message cost (pickle + ZMQ).
    control_msg_s: float = 0.0001
    #: Tensor transfer: fixed setup plus bytes / bandwidth.
    transfer_setup_s: float = 0.00005
    transfer_bandwidth_gbps: float = 20.0
    #: Client-side delivery: api-server poll + postprocess before a chunk is
    #: visible to the caller. Affects TTFT and inter-chunk gaps.
    client_delivery_s: float = 0.003
    #: Request preprocessing (tokenize, media decode) on the api server.
    preprocess_s: float = 0.002

    def transfer_s(self, nbytes: int) -> float:
        if nbytes <= 0:
            return 0.0
        return self.transfer_setup_s + nbytes / (self.transfer_bandwidth_gbps * 1e9 / 8)


# ── Events ───────────────────────────────────────────────────────────────

class EventType(IntEnum):
    """Ordered so same-timestamp events resolve deterministically."""

    ARRIVAL = 0
    CONDUCTOR = 1
    TRANSFER_DONE = 2
    STEP_DONE = 3
    WORKER_POLL = 4


@dataclass(order=True)
class Event:
    time: float
    kind: EventType
    seq: int
    payload: Any = field(compare=False, default=None)


class Calendar:
    """Event queue on a virtual clock."""

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter = itertools.count()
        self.now = 0.0

    def push(self, time: float, kind: EventType, payload: Any = None) -> None:
        # Never schedule into the past: a negative-duration cost (a bad
        # extrapolation) would otherwise rewind the clock and corrupt every
        # subsequent measurement.
        heapq.heappush(
            self._heap, Event(max(time, self.now), kind, next(self._counter), payload)
        )

    def pop(self) -> Event | None:
        if not self._heap:
            return None
        ev = heapq.heappop(self._heap)
        self.now = ev.time
        return ev

    def __len__(self) -> int:
        return len(self._heap)


# ── Request state ────────────────────────────────────────────────────────

@dataclass
class SimRequest:
    """One request in flight.

    Mirrors the conductor's ``RequestData``: state is held **per partition**,
    because that is the granularity the model's transition function works at.
    A speech model's backbone can be mid-decode while its codec partition is
    still consuming an earlier chunk, and the request is finished only when
    every partition says it is.
    """

    rid: str
    arrival_s: float
    #: Generated length in autoregressive steps. A simulator cannot read
    #: sampled tokens, so this stands in for EOS — the same thing a measured
    #: run with a pinned ``max_tokens`` produces.
    target_output_tokens: int
    prompt_tokens: int = 0
    #: What the request carries in and asks for; the model's own transition
    #: logic branches on these.
    input_modalities: list[str] = field(default_factory=lambda: ["text"])
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    model_kwargs: dict = field(default_factory=dict)
    #: Descriptors for the tensors ``process_prompt`` would have produced.
    input_signals: dict[str, list[Any]] = field(default_factory=dict)

    admitted_s: float | None = None
    ingest_s: float | None = None
    first_chunk_s: float | None = None
    last_chunk_s: float | None = None
    finish_s: float | None = None

    #: Tokens the conductor counts (edges flagged ``conductor_new_token``).
    #: Not every model flags one per decode step — Orpheus flags only the
    #: prefill's first token — so this is reporting, not the stop condition.
    output_tokens: int = 0
    #: Autoregressive steps executed for this request.
    decode_steps: int = 0
    #: Per-modality chunk emission times as seen by the client.
    chunks: list[tuple[str, float]] = field(default_factory=list)

    # ── conductor-level state, one entry per partition ───────────────────
    #: partition name -> the real ``PartitionState`` the model mutates
    partition_states: dict[str, Any] = field(default_factory=dict)
    #: "from->to" -> the real ``StreamingConnectionState``
    streaming_connections: dict[str, Any] = field(default_factory=dict)
    #: Tensors the conductor is holding across forward passes; the model
    #: reads this to build each pass's inputs.
    persist_signals: dict[str, list[Any]] = field(default_factory=dict)

    #: worker graph id -> its WorkerGraphIO for this request
    graph_ios: dict[str, Any] = field(default_factory=dict)
    #: worker graph id -> the rank set running it for this request
    wg_ranks: dict[str, list[int]] = field(default_factory=dict)
    #: worker graph id -> the walk it belongs to
    wg_walk: dict[str, str] = field(default_factory=dict)
    #: worker graph id -> the partition that owns it
    wg_partition: dict[str, str] = field(default_factory=dict)
    #: partition -> the walk it is currently executing
    partition_walk: dict[str, str] = field(default_factory=dict)
    #: partition -> steps dispatched but not yet completed. A partition with
    #: work in flight must not be ended, or its output is thrown away.
    partition_inflight: dict[str, int] = field(default_factory=dict)
    #: (consumer node) -> items buffered from a streaming producer, and the
    #: model's own ChunkPolicy deciding when a chunk is ready.
    stream_buffers: dict[str, int] = field(default_factory=dict)
    stream_policies: dict[str, Any] = field(default_factory=dict)
    fwd_index: int = 0
    #: KV context length, advanced as the request decodes.
    kv_len: int = 0
    #: Which walks have completed, for the transition function.
    done: bool = False

    def ttft_s(self) -> float | None:
        if self.first_chunk_s is None:
            return None
        return self.first_chunk_s - self.arrival_s

    def e2e_s(self) -> float | None:
        if self.finish_s is None:
            return None
        return self.finish_s - self.arrival_s


# ── Worker ───────────────────────────────────────────────────────────────

@dataclass
class ReadyItem:
    """A (request, node) pair whose inputs are all present."""

    rid: str
    node: str
    walk: str
    wg_id: str
    ready_s: float


class SimWorker:
    """One GPU rank: a serial GPU lane and a serial CPU lane.

    Batching follows the real MicroScheduler's rule: take every ready request
    for a single (node, graph_walk), choosing the node round-robin by
    least-recently-run so co-located nodes share the GPU fairly.
    """

    def __init__(self, rank: int, sim: "Simulator"):
        self.rank = rank
        self.worker_id = f"worker_{rank}"
        self.sim = sim
        self.gpu_free_s = 0.0
        self.cpu_free_s = 0.0
        #: (node, walk) -> ready items
        self.ready: dict[tuple[str, str], list[ReadyItem]] = {}
        #: (node, walk) -> last batch number, for round-robin selection
        self.last_run: dict[tuple[str, str], int] = {}
        self.batch_counter = 0
        self.busy = False
        #: Accumulated statistics
        self.gpu_busy_s = 0.0
        self.cpu_busy_s = 0.0
        self.steps = 0
        self.missing_cost_steps = 0

    def add_ready(self, item: ReadyItem) -> None:
        self.ready.setdefault((item.node, item.walk), []).append(item)

    def has_work(self) -> bool:
        return any(items for items in self.ready.values())

    def pick_batch(self) -> tuple[tuple[str, str], list[ReadyItem]] | None:
        """Round-robin over (node, walk) keys that have ready work.

        Mirrors ``MicroScheduler._select_node_rr``: the key whose last batch
        number is smallest (least recently run) wins, so a worker hosting
        several nodes alternates instead of starving one.
        """
        candidates = [(k, v) for k, v in self.ready.items() if v]
        if not candidates:
            return None
        key, items = min(candidates, key=lambda kv: self.last_run.get(kv[0], -1))
        self.ready[key] = []
        return key, items


# ── Simulator ────────────────────────────────────────────────────────────


def _synthetic_pointer(edge: Any) -> Any:
    """A descriptor standing in for a tensor the simulator never materialized.

    Edges routed inside the simulator carry no ``tensor_info`` — there are no
    tensors — but a model reading a persisted signal expects a pointer it can
    inspect. Dims are unknown here and reported as such rather than invented,
    so a model that sizes work off them will fail loudly instead of quietly
    computing from a fabricated shape.
    """
    from mstar.graph.base import TensorPointerInfo

    return TensorPointerInfo(
        dims=[], dtype="float32", nbytes=0, address=0, stride=[],
        uuid=f"sim-{edge.name}", source_session_id="sim", source_entity="worker",
    )


class Simulator:
    """Drives a deployment through a workload in virtual time."""

    def __init__(
        self,
        deployment: Deployment,
        stepdb: StepDB,
        timing: TimingModel | None = None,
        max_concurrent: int | None = None,
        seed: int = 0,
    ):
        self.dep = deployment
        self.db = stepdb
        self.timing = timing or TimingModel()
        self.cal = Calendar()
        self.seed = seed

        self.workers: dict[int, SimWorker] = {
            r: SimWorker(r, self) for r in deployment.ranks
        }
        self.requests: dict[str, SimRequest] = {}
        self.waiting: list[str] = []
        self.active: set[str] = set()
        self.finished: list[SimRequest] = []
        self.max_concurrent = (
            max_concurrent
            if max_concurrent is not None
            else deployment.max_concurrent_requests
        )

        #: Coverage flags OR'd across every step priced, so the report can say
        #: whether any number in it was extrapolated or missing.
        # The api server preprocesses on ONE worker thread, so concurrent
        # arrivals serialize through it. That stagger is not a detail: it
        # offsets each request's streaming buffer by a token or two, which
        # is what stops a codec consumer from batching every request in
        # lockstep with the backbone.
        self.preprocess_free_s = 0.0

        #: GPU completion time of the step being postprocessed, so an emit
        #: is never reported before its computation finished.
        self._gpu_end_of_current_step = 0.0

        self.coverage = Coverage.EXACT
        self.missing_keys: dict[str, int] = {}
        #: Transition-function failures, by call site. A model that cannot
        #: answer for a request is a modeling gap worth reporting, not a
        #: crash worth losing the whole run over.
        self.model_errors: dict[str, int] = {}
        self.step_count = 0
        #: (node, walk) -> steps executed, the V1 validation gate's input.
        self.step_counts_by_key: dict[tuple[str, str], int] = {}

        # Shapes actually measured, per (node, walk, mode) — used to snap a
        # requested batch onto the bucket the engine would have padded to.
        self._bucket_cache: dict[tuple[str, str, str], list[tuple[int, int]]] = {}

    # ── cost lookup ──────────────────────────────────────────────────────

    def _buckets(self, node: str, walk: str, mode: str) -> list[tuple[int, int]]:
        key = (node, walk, mode)
        hit = self._bucket_cache.get(key)
        if hit is None:
            hit = sorted({
                (k.padded_bs, k.padded_num_tokens)
                for k in self.db.keys(self.dep.model_key)
                if k.node == node and k.graph_walk == walk and k.mode == mode
            })
            self._bucket_cache[key] = hit
        return hit

    def step_cost(
        self, node: str, walk: str, bs: int, num_tokens: int, kv_len_total: int,
    ) -> StepCost:
        """Price one step, padding to the bucket the engine would replay into.

        Falls back to the eager regime when no captured bucket is large
        enough — which is what the real engine does on a capture miss.
        """
        tp = self.dep.tp_size_for(node)
        sp = self.dep.sp_size_for(node)

        for mode in ("graph", "eager", "sequential"):
            buckets = self._buckets(node, walk, mode)
            if not buckets:
                continue
            if mode == "graph":
                padded_bs = pad_to_bucket(bs, {b for b, _ in buckets})
                if padded_bs is None:
                    continue  # capture miss → try the eager rows
                token_opts = {t for b, t in buckets if b == padded_bs}
                padded_tokens = pad_to_bucket(num_tokens, token_opts)
                if padded_tokens is None:
                    continue
            else:
                padded_bs, padded_tokens = bs, num_tokens

            cost = self.db.lookup(
                StepKey(
                    model=self.dep.model_key, node=node, graph_walk=walk,
                    padded_bs=padded_bs, padded_num_tokens=padded_tokens,
                    tp_size=tp, sp_size=sp, mode=mode,
                ),
                kv_len_total,
            )
            if cost.coverage != Coverage.MISSING:
                self.coverage |= cost.coverage
                return cost

        self.coverage |= Coverage.MISSING
        tag = f"{node}/{walk}"
        self.missing_keys[tag] = self.missing_keys.get(tag, 0) + 1
        return StepCost(gpu_s=0.0, coverage=Coverage.MISSING,
                        note=f"no measured cost for {tag} bs={bs}")

    # ── request lifecycle ────────────────────────────────────────────────

    def submit(self, req: SimRequest) -> None:
        self.requests[req.rid] = req
        self.cal.push(req.arrival_s, EventType.ARRIVAL, req.rid)

    def _on_arrival(self, rid: str) -> None:
        # Queue for the single preprocess thread, then pay the conductor's
        # poll latency before admission.
        start = max(self.cal.now, self.preprocess_free_s)
        done = start + self.timing.preprocess_s
        self.preprocess_free_s = done
        # The request joins the conductor's queue only once preprocessing
        # has actually produced its inputs — enqueuing at arrival instead
        # would let one conductor tick admit a whole burst simultaneously
        # and erase the stagger the serial preprocess thread creates.
        self.cal.push(
            done + self.timing.conductor_hop_s,
            EventType.CONDUCTOR,
            ("ingest", rid, ""),
        )

    def _admit_waiting(self) -> None:
        """FIFO admission under the concurrency cap — the conductor's rule."""
        while self.waiting:
            if self.max_concurrent is not None and len(self.active) >= self.max_concurrent:
                return
            rid = self.waiting.pop(0)
            req = self.requests[rid]
            req.admitted_s = self.cal.now
            req.ingest_s = self.cal.now
            self.active.add(rid)
            self._ingest(req)

    # ── the conductor's request protocol, driven for real ────────────────

    def _ingest(self, req: SimRequest) -> None:
        """Kick off every partition, exactly as ``_do_ingest_request`` does.

        The model is asked for each partition's opening move. A partition
        that plays no part in this request answers ``request_done`` — that is
        how a speech model's codec partition stays idle for a text-only
        request without anyone special-casing it here.
        """
        from mstar.conductor.request_info import (
            CurrentForwardConductorMetadata,
            PartitionState,
            StreamingConnectionState,
        )

        req.persist_signals = dict(req.input_signals)

        for conn in (getattr(self.dep.partition_topology, "connections", None) or []):
            key = f"{conn.from_partition}->{conn.to_partition}"
            req.streaming_connections[key] = StreamingConnectionState(
                from_partition=conn.from_partition,
                to_partition=conn.to_partition,
                edge_name=conn.edge_name,
            )

        for p in self.dep.partitions:
            req.partition_states[p.name] = PartitionState(
                partition_name=p.name,
                metadata=CurrentForwardConductorMetadata(
                    input_modalities=list(req.input_modalities),
                    output_modalities=list(req.output_modalities),
                    graph_walk="",
                    is_prefill=True,
                ),
                random_seed=abs(hash(req.rid)) % (2 ** 31),
            )

        for p in self.dep.partitions:
            pstate = req.partition_states[p.name]
            try:
                fwd = self.dep.model.get_initial_forward_pass_args(
                    partition_name=p.name,
                    input_modalities=list(req.input_modalities),
                    output_modalities=list(req.output_modalities),
                    input_signals=req.input_signals,
                    model_kwargs=dict(req.model_kwargs),
                )
            except Exception as exc:
                # A model that cannot open this request is a modeling gap, not
                # a simulator crash: record it, leave the partition idle, and
                # let the run report which requests never started.
                logger.warning(
                    "%s: get_initial_forward_pass_args failed for partition "
                    "%s (%s); leaving it idle", self.dep.model_key, p.name, exc,
                )
                self.model_errors[f"initial:{p.name}"] = (
                    self.model_errors.get(f"initial:{p.name}", 0) + 1
                )
                pstate.is_done = True
                continue

            pstate.is_done = bool(fwd.request_done)
            pstate.metadata = fwd.full_metadata
            pstate.metadata.kwargs.update(fwd.step_metadata)
            if pstate.is_done:
                continue
            self._enter_walk(req, p.name, fwd.full_metadata.graph_walk, fwd.inputs)

        if all(ps.is_done for ps in req.partition_states.values()):
            self._finish(req)

    def _advance_partition(self, req: SimRequest, partition: str) -> None:
        """Ask the model what this partition does next — ``_process_done_forward``.

        Everything about which walk follows which is the model's decision,
        including the length cap the conductor applies on top of it. Nothing
        here inspects walk names.
        """
        pstate = req.partition_states.get(partition)
        if pstate is None or pstate.is_done or req.done:
            return
        if req.partition_inflight.get(partition, 0) > 0:
            # A step is still on the GPU. Ending the partition now would
            # discard whatever that step is about to emit — for a codec
            # draining its final chunk, that is the request's only audio.
            # Its own completion will advance it.
            return

        incoming = [
            c for c in req.streaming_connections.values()
            if c.to_partition == partition
        ]
        try:
            fwd = self.dep.model.get_partition_forward_pass_args(
                partition_name=partition,
                partition_metadata=pstate.metadata,
                persist_signals=req.persist_signals,
                incoming_connections=incoming,
            )
        except Exception as exc:
            logger.warning(
                "%s: get_partition_forward_pass_args failed for partition %s "
                "(%s); ending it", self.dep.model_key, partition, exc,
            )
            self.model_errors[f"advance:{partition}"] = (
                self.model_errors.get(f"advance:{partition}", 0) + 1
            )
            fwd = None

        if fwd is None:
            self._end_partition(req, partition)
            return

        pstate.metadata = fwd.full_metadata
        pstate.metadata.kwargs.update(fwd.step_metadata)
        pstate.fwd_pass_number += 1
        pstate.random_seed += 1

        request_done = bool(fwd.request_done)
        # The conductor's own cap, applied on top of the model's answer. A
        # simulator has no sampled tokens, so the workload's target length is
        # the stand-in for EOS.
        if req.decode_steps >= req.target_output_tokens:
            request_done = True

        if request_done:
            self._end_partition(req, partition)
            return

        self._enter_walk(req, partition, fwd.full_metadata.graph_walk, fwd.inputs)

    def _end_partition(self, req: SimRequest, partition: str) -> None:
        """Mark a partition finished and tell its consumers, as the conductor does."""
        pstate = req.partition_states.get(partition)
        if pstate is None or pstate.is_done:
            return
        pstate.is_done = True
        for conn in req.streaming_connections.values():
            if conn.from_partition == partition:
                conn.producer_done = True
                # The consumer drains whatever is still buffered, even though
                # the chunk policy's window was never met. Without this a
                # short request never runs its codec at all: a 126-token
                # generation cannot fill a 300-token window, yet the real
                # deployment still emits audio for it.
                if self._flush_stream(req, conn):
                    # Keep the consumer open until it has run that chunk, or
                    # the request would finish underneath the flush and the
                    # audio would never be produced.
                    consumer = req.partition_states.get(conn.to_partition)
                    if consumer is not None:
                        consumer.is_done = False
                # A consumer may have work left to do once its producer stops
                # (draining its buffer, or continuing on its own); give it a
                # chance to react rather than cutting it off.
                self.cal.push(
                    self.cal.now + self.timing.conductor_hop_s,
                    EventType.CONDUCTOR,
                    ("advance", req.rid, conn.to_partition),
                )
        if all(ps.is_done for ps in req.partition_states.values()):
            self._finish(req)

    def _flush_stream(self, req: SimRequest, conn: Any) -> bool:
        """Deliver a final partial chunk to a streaming consumer.

        Returns whether anything was scheduled.
        """
        from mstar.graph.base import GraphEdge

        scheduled = False
        for key, buffered in list(req.stream_buffers.items()):
            node, _, name = key.partition(":")
            if name != conn.edge_name or buffered <= 0:
                continue
            req.stream_buffers[key] = 0
            dest_wg_id, dest_wg = self._find_or_instantiate_dest(req, node)
            if dest_wg_id is None or dest_wg is None:
                continue
            self.cal.push(
                self.cal.now + self.timing.control_msg_s,
                EventType.TRANSFER_DONE,
                (req.rid, dest_wg_id, GraphEdge(next_node=node, name=name),
                 req.wg_walk.get(dest_wg_id, "")),
            )
            scheduled = True
        return scheduled

    def _enter_walk(
        self, req: SimRequest, partition: str, walk: str, inputs: list | None,
    ) -> None:
        """Instantiate a partition's next walk and seed it with the model's inputs.

        ``inputs`` are the ``GraphEdge`` objects the model itself returned —
        the same objects the conductor forwards to workers — so seeding is
        the model's decision, not a guess about which nodes look like entry
        points. When the model returns none, the walk is self-triggering
        (a streaming consumer waiting on its buffer) and is left to be woken
        by an arriving chunk.
        """
        if not walk:
            return
        wgs = [
            wg for wg in self.dep.walk_to_wgs.get(walk, [])
            if self._wg_partition(wg, walk) == partition
        ]
        if not wgs:
            # No worker graph for this partition's walk: nothing to run.
            self._end_partition(req, partition)
            return

        previous = req.partition_walk.get(partition)
        req.partition_walk[partition] = walk

        # Prefill/decode disaggregation: when a partition's walk moves to a
        # different GPU, the KV context it built has to follow. The real
        # engine pulls it page-by-page per layer over RDMA before the first
        # decode step can run, and a simulator that skips it reports a
        # disaggregated deployment as free of the very cost that motivates
        # measuring it.
        handoff_s = self._kv_handoff_s(req, partition, previous, wgs)

        # Retire this partition's previous graphs; other partitions keep theirs.
        if previous and previous != walk:
            for wg_id in [
                i for i, pn in req.wg_partition.items()
                if pn == partition and req.wg_walk.get(i) == previous
            ]:
                req.graph_ios.pop(wg_id, None)
                req.wg_walk.pop(wg_id, None)
                req.wg_partition.pop(wg_id, None)

        for wg in wgs:
            self._instantiate(req, wg, walk, partition)

        if not inputs:
            return

        # Route the model's declared inputs through ingest_input, which is
        # what keeps each graph's readiness, loop, and completion bookkeeping
        # correct — the same path the real worker uses for arriving edges.
        for edge in inputs:
            for wg in wgs:
                io = req.graph_ios.get(wg.wg_id)
                if io is not None and io.ingest_input(edge):
                    break
        for wg in wgs:
            self._drain_ready(req, wg, not_before=self.cal.now + handoff_s)

    def _kv_handoff_s(
        self, req: SimRequest, partition: str, previous: str | None,
        wgs: list[SimWorkerGraph],
    ) -> float:
        """Time to move a request's KV context to the walk's new GPUs.

        Zero unless the partition actually changed GPUs and the node it runs
        keeps a KV cache — a stateless encoder has nothing to carry, and a
        colocated walk transition moves nothing.
        """
        if not previous or self.dep.kv_bytes_per_token <= 0:
            return 0.0
        prev_ranks = {
            r
            for wg_id, pn in req.wg_partition.items()
            if pn == partition and req.wg_walk.get(wg_id) == previous
            for r in (req.wg_ranks.get(wg_id) or [])
        }
        new_ranks = {r for wg in wgs for r in (req.wg_ranks.get(wg.wg_id) or wg.ranks)}
        if not prev_ranks or not new_ranks or prev_ranks == new_ranks:
            return 0.0
        if not any(
            self.dep.node_engine_types.get(n) == "kv_cache"
            for wg in wgs for n in wg.node_names
        ):
            return 0.0
        nbytes = req.kv_len * self.dep.kv_bytes_per_token
        if nbytes <= 0:
            return 0.0
        return self.timing.transfer_s(nbytes)

    def _wg_partition(self, wg: SimWorkerGraph, walk: str) -> str:
        """Which partition owns a worker graph, via the walk it belongs to."""
        for p in self.dep.partitions:
            if walk in getattr(p, "graph_walks", set()):
                return p.name
        return "default"

    def _pick_instance(self, wg: SimWorkerGraph, req: SimRequest) -> list[int]:
        """Choose a DP replica, seeded per request like the real conductor."""
        import random
        rng = random.Random(f"{self.seed}:{req.rid}:{wg.group_id}")
        return list(rng.choice(wg.instance_ranks))

    @staticmethod
    def _entry_edges(io: Any) -> list[Any]:
        """Synthetic edges that start this graph running.

        Two sources, both taken from the graph's own declarations:

        * ``ext_inputs`` — what the section says it needs from outside.
        * each loop's loop-back inputs — a loop declares no external input
          (it feeds itself), but its *first* iteration still has to be
          primed, which is what the conductor's INPUT_SIGNALS does in the
          real system.

        The simulator fabricates these edges because it models the timing of
        tensors, never their contents.
        """
        from mstar.graph.base import GraphEdge

        wanted: set[tuple[str, str]] = set()
        try:
            wanted |= set(io.graph.get_inputs_outputs().ext_inputs)
        except Exception:
            pass
        for loop in io.loops.values():
            for pair in (getattr(loop, "_loop_back_inputs", None) or set()):
                wanted.add(pair)

        return [
            GraphEdge(next_node=node, name=name)
            for name, node in sorted(wanted)
            if node in io.nodes
        ]

    def _instantiate(
        self, req: SimRequest, wg: SimWorkerGraph, walk: str,
        partition: str | None = None,
    ) -> Any:
        """Create (or recreate) this request's copy of one worker graph.

        Deep-copies the section exactly as ``WorkerGraphQueues.add_request``
        does — per-request graph state (loop counters, readiness) must never
        be shared between requests.
        """
        from mstar.graph.graph_io import WorkerGraphIO

        io = WorkerGraphIO(copy.deepcopy(wg.section), wg_id=wg.wg_id)
        req.graph_ios[wg.wg_id] = io
        req.wg_walk[wg.wg_id] = walk
        req.wg_partition[wg.wg_id] = (
            partition if partition is not None else self._wg_partition(wg, walk)
        )
        if wg.wg_id not in req.wg_ranks:
            req.wg_ranks[wg.wg_id] = self._pick_instance(wg, req)
        return io

    def _find_or_instantiate_dest(
        self, req: SimRequest, node_name: str, source_wg_id: str | None = None
    ) -> tuple[str | None, SimWorkerGraph | None]:
        """Locate the graph that owns ``node_name``, instantiating on demand.

        Resolution order matters: the same node name appears in several walks
        (an AR backbone is "LLM" in both prefill and decode), so an unscoped
        search would route a decode loop-back into the finished prefill graph.

        1. the source graph itself — a loop-back edge never leaves it;
        2. graphs belonging to the walk currently being driven;
        3. any other live graph;
        4. otherwise instantiate the walk that declares the node. A streaming
           consumer (an audio codec fed by the backbone) lives in a walk the
           conductor never explicitly starts, so its graph is created here on
           first use and never seeded.
        """
        if source_wg_id is not None:
            io = req.graph_ios.get(source_wg_id)
            if io is not None and node_name in io.nodes:
                return source_wg_id, self._wg_by_id(source_wg_id)

        source_partition = req.wg_partition.get(source_wg_id) if source_wg_id else None
        if source_partition:
            for wg_id, io in req.graph_ios.items():
                if node_name in io.nodes and req.wg_partition.get(wg_id) == source_partition:
                    return wg_id, self._wg_by_id(wg_id)

        for wg_id, io in req.graph_ios.items():
            if node_name in io.nodes:
                return wg_id, self._wg_by_id(wg_id)

        for walk, wgs in self.dep.walk_to_wgs.items():
            for wg in wgs:
                if node_name in wg.node_names:
                    self._instantiate(req, wg, walk)
                    return wg.wg_id, wg
        return None, None

    def _is_stream_consumer(self, wg_id: str) -> bool:
        wg = self._wg_by_id(wg_id)
        return bool(wg and wg.consumes_stream)

    def _wg_by_id(self, wg_id: str) -> SimWorkerGraph | None:
        for wgs in self.dep.walk_to_wgs.values():
            for wg in wgs:
                if wg.wg_id == wg_id:
                    return wg
        return None

    def _drain_ready(
        self, req: SimRequest, wg: SimWorkerGraph, not_before: float | None = None,
    ) -> None:
        """Move every newly-ready node of this graph onto its worker's queue.

        ``not_before`` delays when the work counts as ready — used for a KV
        handoff, where the context has to land on the new GPU before the
        first step there can run.
        """
        io = req.graph_ios.get(wg.wg_id)
        if io is None:
            return
        ranks = req.wg_ranks.get(wg.wg_id) or wg.ranks
        # Lockstep instances run the same step on every rank; the leader's
        # timeline is the one that gates progress, so schedule on it.
        leader = ranks[0]
        worker = self.workers[leader]
        for node_name in list(io.ready_node_names):
            already = any(
                it.rid == req.rid and it.node == node_name
                for items in worker.ready.values() for it in items
            )
            if already:
                continue
            worker.add_ready(ReadyItem(
                rid=req.rid, node=node_name,
                # The graph's own walk, not the request's current one: a
                # streaming consumer runs its own walk (an audio codec's
                # chunk walk) while the producer is mid-decode, and pricing
                # its steps under the producer's walk would miss the table.
                walk=req.wg_walk.get(wg.wg_id, ""),
                wg_id=wg.wg_id,
                ready_s=max(self.cal.now, not_before or 0.0),
            ))
        self.cal.push(
            max(self.cal.now, not_before or 0.0), EventType.WORKER_POLL, leader
        )

    # ── worker execution ─────────────────────────────────────────────────

    def _worker_poll(self, rank: int) -> None:
        w = self.workers[rank]
        if w.busy or not w.has_work():
            return
        picked = w.pick_batch()
        if picked is None:
            return
        (node, walk), items = picked
        w.batch_counter += 1
        w.last_run[(node, walk)] = w.batch_counter
        w.busy = True

        bs = len(items)
        reqs = [self.requests[i.rid] for i in items]
        num_tokens, kv_total = self._batch_shape(node, walk, reqs)
        cost = self.step_cost(node, walk, bs, num_tokens, kv_total)
        if cost.coverage & Coverage.MISSING:
            w.missing_cost_steps += 1

        # Two-lane schedule mirroring the worker's speculation pipeline
        # (worker.py:1084-1098): the CPU *builds* the next batch while the
        # current GPU step is still running, submits it the moment that step
        # lands, and only then postprocesses the finished one.
        #
        #   CPU:  [ build N ][ submit N ][ post N-1 ][ build N+1 ] ...
        #   GPU:            [========= step N =========]
        #
        # Because the build overlaps the previous GPU step, the loop settles
        # at max(GPU, CPU) rather than their sum. Ordering the CPU lane as
        # build → submit → post (instead of submit → post → build) is what
        # produces that; getting it backwards silently turns every predicted
        # step into gpu + cpu and inflates every latency downstream.
        build_s = cost.prepare_s + cost.plan_s
        after_s = cost.launch_s + cost.sample_s + self.timing.worker_step_overhead_s

        ready_s = max(i.ready_s for i in items)
        build_start = max(w.cpu_free_s, ready_s, self.cal.now)
        build_end = build_start + build_s
        gpu_start = max(w.gpu_free_s, build_end)
        gpu_end = gpu_start + cost.gpu_s
        # The launch and everything after it (sampling, routing, ZMQ) run
        # while this step's kernels are already on the GPU — that is the
        # whole point of the async pipeline — so they are charged from
        # gpu_start, not from gpu_end. Charging them after the step would
        # serialize the two lanes and turn every predicted step into
        # gpu + cpu, inflating every latency downstream.
        cpu_done = gpu_start + after_s

        w.gpu_free_s = gpu_end
        w.cpu_free_s = cpu_done
        w.gpu_busy_s += cost.gpu_s
        w.cpu_busy_s += build_s + after_s
        w.steps += 1

        # A lockstep instance runs the same step on every one of its ranks.
        # Only the leader's timeline gates progress, but the followers' GPUs
        # are just as busy — leaving them idle would report a TP=2 deployment
        # as half-utilized and let the simulator schedule work onto a GPU
        # that is actually mid-collective.
        for follower in self._followers(items):
            if follower == rank:
                continue
            fw = self.workers.get(follower)
            if fw is None:
                continue
            fw.gpu_free_s = max(fw.gpu_free_s, gpu_end)
            fw.cpu_free_s = max(fw.cpu_free_s, cpu_done)
            fw.gpu_busy_s += cost.gpu_s
            fw.cpu_busy_s += build_s + after_s
            fw.steps += 1
        self.step_count += 1
        self.step_counts_by_key[(node, walk)] = (
            self.step_counts_by_key.get((node, walk), 0) + 1
        )

        # Routing fires when the CPU lane is free, not when the GPU step
        # lands. The real worker speculatively builds the next batch while
        # the current one is still on the GPU (worker.py:1084-1098); gating
        # the next build on GPU completion would serialize the two lanes and
        # make every step cost build + gpu instead of max(gpu, cpu).
        #
        # ``gpu_end`` rides along so anything the client can actually observe
        # — an emitted token or audio chunk — is still held until the GPU
        # has produced it.
        for item in items:
            r = self.requests.get(item.rid)
            if r is not None:
                pn = r.wg_partition.get(item.wg_id)
                if pn:
                    r.partition_inflight[pn] = r.partition_inflight.get(pn, 0) + 1

        self.cal.push(
            cpu_done, EventType.STEP_DONE, (rank, node, walk, items, gpu_end)
        )

    def _followers(self, items: list[ReadyItem]) -> set[int]:
        """Every rank of the lockstep instances this batch runs on."""
        ranks: set[int] = set()
        for item in items:
            req = self.requests.get(item.rid)
            if req is None:
                continue
            assigned = req.wg_ranks.get(item.wg_id)
            if assigned:
                ranks.update(assigned)
            else:
                wg = self._wg_by_id(item.wg_id)
                if wg is not None:
                    ranks.update(wg.ranks)
        return ranks

    def _batch_shape(
        self, node: str, walk: str, reqs: list[SimRequest]
    ) -> tuple[int, int]:
        """Token count and KV total for a batch.

        Decode contributes one token per request; a prefill contributes the
        whole prompt. Which one this is comes from the engine type and the
        request's progress, mirroring how the real batch is shaped.
        """
        is_kv = self.dep.node_engine_types.get(node) == "kv_cache"
        if not is_kv:
            return len(reqs), 0
        tokens = 0
        kv = 0
        for r in reqs:
            if "prefill" in walk and r.decode_steps == 0:
                tokens += max(1, r.prompt_tokens)
            else:
                tokens += 1
            kv += r.kv_len
        return tokens, kv

    def _on_step_done(
        self, rank: int, node: str, walk: str, items: list[ReadyItem],
        gpu_end: float = 0.0,
    ) -> None:
        w = self.workers[rank]
        w.busy = False
        self._gpu_end_of_current_step = gpu_end

        for item in items:
            req = self.requests.get(item.rid)
            if req is None:
                continue
            partition = req.wg_partition.get(item.wg_id)
            if partition:
                req.partition_inflight[partition] = max(
                    0, req.partition_inflight.get(partition, 0) - 1
                )
            if req.done:
                continue
            io = req.graph_ios.get(item.wg_id)
            if io is None:
                continue

            # A KV-cache node emits one token per step and grows the context.
            if self.dep.node_engine_types.get(node) == "kv_cache":
                if "prefill" in walk and req.decode_steps == 0:
                    req.kv_len += max(1, req.prompt_tokens)
                else:
                    req.kv_len += 1
                req.decode_steps += 1

            # EOS. The real worker learns a request is done by reading the
            # sampled token; a simulator has no token values, so the stop is
            # the workload's target length — the same condition a measured
            # run with a pinned max_tokens produces. Signalling it through
            # register_loop_finish_signal is exactly the worker's own path,
            # so loop teardown and completion accounting stay identical.
            if (
                req.decode_steps >= req.target_output_tokens
                and io.loops
            ):
                for loop_name in io.loops:
                    io.register_loop_finish_signal(loop_name)

            io.ready_node_names.discard(node)
            completion = io.mark_node_complete(node)
            self._route(req, item.wg_id, completion, walk)

        self.cal.push(self.cal.now, EventType.WORKER_POLL, rank)

    def _route(self, req: SimRequest, wg_id: str, completion: Any, walk: str) -> None:
        """Route a completed node's output edges, as the worker does."""
        wg_by_id = {
            wg.wg_id: wg for wgs in self.dep.walk_to_wgs.values() for wg in wgs
        }
        filtered = getattr(completion, "filtered_signals", set()) or set()

        for edge in completion.output_edges:
            if (edge.name, edge.next_node) in filtered:
                continue

            # Output-token accounting follows the edge flag the conductor
            # uses, not the destination: the backbone's token-count edge goes
            # to EMPTY_DESTINATION while the audio chunks go to the client.
            if getattr(edge, "conductor_new_token", False):
                req.output_tokens += 1

            # Absorb persisted tensors, as the conductor does when a worker
            # graph reports done. Models read these back by name to build the
            # next pass's inputs — Qwen3-Omni's Talker, for instance, needs
            # the embeds its Thinker produced — so a request whose persist
            # signals are missing cannot advance past the handoff.
            if getattr(edge, "persist", False):
                req.persist_signals[edge.name] = (
                    list(edge.tensor_info) if edge.tensor_info
                    else [_synthetic_pointer(edge)]
                )

            if edge.next_node == EMIT_TO_CLIENT:
                self._emit(req, edge)
                continue
            if edge.next_node in SPECIAL_DESTINATIONS:
                # EMPTY_DESTINATION and friends terminate here.
                continue

            dest_wg_id, dest_wg = self._find_or_instantiate_dest(
                req, edge.next_node, source_wg_id=wg_id
            )
            if dest_wg_id is None or dest_wg is None:
                continue

            # A streaming edge does not run the consumer per item: the
            # consumer's ChunkPolicy decides when enough have accumulated.
            # Modeling that is what makes codec cadence right — a SNAC step
            # every `stride` tokens, not one per token.
            if getattr(edge, "is_streaming", False):
                if not self._stream_arrival(req, edge, dest_wg):
                    continue

            same_worker = (
                (req.wg_ranks.get(dest_wg_id) or dest_wg.ranks)[0]
                == (req.wg_ranks.get(wg_id) or [0])[0]
            )
            if same_worker:
                delay = 0.0
            else:
                # Cross-worker edge: a tensor transfer plus a control hop.
                nbytes = self._edge_bytes(edge)
                delay = self.timing.transfer_s(nbytes) + self.timing.control_msg_s

            self.cal.push(
                self.cal.now + delay, EventType.TRANSFER_DONE,
                (req.rid, dest_wg_id, edge, req.wg_walk.get(dest_wg_id, walk)),
            )

        # This graph finished its walk. Every partition advances through the
        # same call the conductor makes — including a streaming consumer,
        # whose model decides whether another chunk follows.
        io = req.graph_ios.get(wg_id)
        if io is not None and io.wg_state_registry.is_done:
            partition = req.wg_partition.get(wg_id)
            if partition and self._partition_walk_done(req, partition):
                self.cal.push(
                    self.cal.now + self.timing.conductor_hop_s,
                    EventType.CONDUCTOR,
                    ("advance", req.rid, partition),
                )

    def _partition_walk_done(self, req: SimRequest, partition: str) -> bool:
        """True when every graph of this partition's current walk has finished."""
        ids = [i for i, pn in req.wg_partition.items() if pn == partition]
        if not ids:
            return False
        return all(
            (io := req.graph_ios.get(i)) is not None
            and io.wg_state_registry.is_done
            for i in ids
        )

    def _stream_arrival(
        self, req: SimRequest, edge: Any, dest_wg: SimWorkerGraph
    ) -> bool:
        """Buffer one streamed item; report whether a chunk is now ready.

        The policy object comes from the model's own PartitionTopology, so
        window/stride arithmetic — and therefore how often the consumer runs
        per producer token — is the deployment's, not the simulator's.
        """
        key = f"{edge.next_node}:{edge.name}"
        if key not in req.stream_policies:
            req.stream_policies[key] = self._chunk_policy_for(edge.name)
        policy = req.stream_policies[key]
        req.stream_buffers[key] = req.stream_buffers.get(key, 0) + 1

        if policy is None:
            return True  # no declared policy: consume item-by-item
        node = key
        buffered = req.stream_buffers[node]
        if not policy.is_ready(buffered):
            return False
        take = policy.next_chunk_size(buffered)
        policy.register_chunk(take)
        req.stream_buffers[node] = max(0, buffered - take)
        return True

    def _chunk_policy_for(self, edge_name: str) -> Any:
        """A fresh ChunkPolicy for a streaming edge, from the model's topology.

        ``Connection`` keys on the edge name and hands out policies through a
        factory, so each request gets its own policy instance with its own
        consumed-item counter — the same lifetime the real StreamBuffer gives
        it.
        """
        topo = self.dep.partition_topology
        for conn in (getattr(topo, "connections", None) or []):
            if getattr(conn, "edge_name", None) != edge_name:
                continue
            factory = getattr(conn, "chunk_policy_factory", None)
            if factory is not None:
                return factory()
        return None

    @staticmethod
    def _edge_bytes(edge: Any) -> int:
        total = 0
        for info in getattr(edge, "tensor_info", []) or []:
            total += getattr(info, "nbytes", 0) or 0
        return total

    def _on_transfer_done(self, rid: str, wg_id: str, edge: Any, walk: str) -> None:
        req = self.requests.get(rid)
        if req is None or req.done:
            return
        io = req.graph_ios.get(wg_id)
        if io is None:
            return
        io.ingest_input(edge)
        wg = next(
            (w for w in self.dep.walk_to_wgs.get(walk, []) if w.wg_id == wg_id),
            None,
        )
        if wg is not None:
            self._drain_ready(req, wg)

    def _emit(self, req: SimRequest, edge: Any) -> None:
        """A token/chunk reaches the client after the delivery path's latency.

        Held until the GPU step that produced it has actually landed — the
        CPU may have finished routing first, but nothing observable can
        precede the computation.
        """
        produced_at = max(self.cal.now, getattr(self, "_gpu_end_of_current_step", 0.0))
        t = produced_at + self.timing.control_msg_s + self.timing.client_delivery_s
        modality = getattr(edge, "output_modality", None) or "text"
        req.chunks.append((modality, t))
        if req.first_chunk_s is None:
            req.first_chunk_s = t
        req.last_chunk_s = t

    def _on_conductor(self, payload: Any) -> None:
        if payload is None:
            self._admit_waiting()
            return
        kind, rid, arg = payload
        if kind == "ingest":
            self.waiting.append(rid)
            self._admit_waiting()
            return
        req = self.requests.get(rid)
        if req is None or req.done:
            return
        if kind == "advance":
            req.fwd_index += 1
            self._advance_partition(req, arg)
            self._admit_waiting()

    def _finish(self, req: SimRequest) -> None:
        if req.done:
            return
        req.done = True
        req.finish_s = self.cal.now
        self.active.discard(req.rid)
        self.finished.append(req)
        for w in self.workers.values():
            for key, items in list(w.ready.items()):
                w.ready[key] = [i for i in items if i.rid != req.rid]

    # ── main loop ────────────────────────────────────────────────────────

    def run(
        self, max_events: int = 5_000_000, until_s: float | None = None,
        stop_on_completion: bool = False,
    ) -> None:
        handlers: dict[EventType, Callable[[Any], None]] = {
            EventType.ARRIVAL: self._on_arrival,
            EventType.CONDUCTOR: self._on_conductor,
            EventType.WORKER_POLL: self._worker_poll,
        }
        n = 0
        finished_at_entry = len(self.finished)
        while n < max_events:
            if stop_on_completion and len(self.finished) > finished_at_entry:
                return
            ev = self.cal.pop()
            if ev is None:
                break
            if until_s is not None and ev.time > until_s:
                break
            n += 1
            if ev.kind == EventType.STEP_DONE:
                self._on_step_done(*ev.payload)
            elif ev.kind == EventType.TRANSFER_DONE:
                self._on_transfer_done(*ev.payload)
            else:
                handlers[ev.kind](ev.payload)

        if n >= max_events:
            logger.warning(
                "simulation stopped at the %d-event cap with %d requests "
                "unfinished — the deployment may be deadlocked",
                max_events, len(self.active),
            )
