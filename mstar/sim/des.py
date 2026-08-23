"""Virtual-time discrete-event simulator for an mstar deployment.

The design rule is that **semantics are imported and costs are measured**.
Everything that decides *what runs when* comes from mstar's own code:

* graph readiness, loop iteration, and EOS handling — the real
  :class:`~mstar.graph.graph_io.WorkerGraphIO`, one deep copy per
  (request, worker graph), exactly as the worker does;
* walk sequencing — the model's own ``get_initial_forward_pass_args`` and
  ``get_partition_forward_pass_args``;
* placement — the real ``get_worker_graphs`` over the deployment YAML;
* streaming cadence — the model's own ``ChunkPolicy`` objects.

What is modeled rather than imported is *how long things take*: the worker's
two-lane (GPU / CPU) pipeline, the conductor hop, and tensor transfers. Step
costs are looked up in the measured stepdb.

## The worker timing model

A real worker runs one GPU stream and one busy main thread, and overlaps them
with a one-deep speculation pipeline: while GPU(N) runs, the CPU builds batch
N+1; after GPU(N) lands, it submits N+1 and only then postprocesses N. So the
steady-state cadence is ``max(GPU, CPU)`` rather than their sum.

This is modeled with two serial resources per worker and these dependencies::

    cpu_submit(N)  →  gpu(N)  →  cpu_post(N)

CPU work for a step is ``prepare + plan + launch`` before the launch and
``sample`` after it, both measured per step, plus a fixed per-step worker
overhead (routing, tensor bookkeeping, ZMQ) taken from the config. Because
both lanes are serial and the submit of N+1 does not wait for the post of N to
finish before the GPU can start, the loop settles at ``max(gpu, cpu)`` — the
behavior the real pipeline exists to produce — without hard-coding it.

The model deliberately does *not* reproduce the speculation machinery
step-for-step (fairness peeks, plan-thread gating, slot double-buffering). It
reproduces the resource structure those mechanisms exist to create. The
residual is what validation measures.
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
    """One request in flight."""

    rid: str
    arrival_s: float
    #: Requested output length; the workload sets it and EOS is modeled as
    #: reaching it, since a simulator has no token values to inspect.
    target_output_tokens: int
    prompt_tokens: int = 0

    admitted_s: float | None = None
    ingest_s: float | None = None
    first_chunk_s: float | None = None
    last_chunk_s: float | None = None
    finish_s: float | None = None

    #: Tokens the conductor counts (edges flagged ``conductor_new_token``).
    #: Not every model flags one per decode step — Orpheus flags only the
    #: prefill's first token — so this is reporting, not the stop condition.
    output_tokens: int = 0
    #: Autoregressive steps executed for this request. This is the generated
    #: token count in the sense ``max_tokens`` means, and what EOS is modeled
    #: against.
    decode_steps: int = 0
    #: Per-modality chunk emission times as seen by the client.
    chunks: list[tuple[str, float]] = field(default_factory=list)

    #: worker graph id -> its WorkerGraphIO for this request
    graph_ios: dict[str, Any] = field(default_factory=dict)
    #: worker graph id -> the rank set running it for this request
    wg_ranks: dict[str, list[int]] = field(default_factory=dict)
    #: worker graph id -> the walk it belongs to
    wg_walk: dict[str, str] = field(default_factory=dict)
    #: (consumer node) -> items buffered from a streaming producer, and the
    #: model's own ChunkPolicy deciding when a chunk is ready.
    stream_buffers: dict[str, int] = field(default_factory=dict)
    stream_policies: dict[str, Any] = field(default_factory=dict)
    current_walk: str = ""
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
        self.coverage = Coverage.EXACT
        self.missing_keys: dict[str, int] = {}
        self.step_count = 0

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
        req = self.requests[rid]
        # api-server preprocess, then the conductor's poll latency.
        t = self.cal.now + self.timing.preprocess_s
        self.waiting.append(rid)
        self.cal.push(t + self.timing.conductor_hop_s, EventType.CONDUCTOR, None)

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
            self._start_walk(req, self._initial_walk(req))

    def _initial_walk(self, req: SimRequest) -> str:
        """First walk of the request.

        Uses the model's declared initial walk for its first partition, which
        is what the conductor reads at ingest.
        """
        for p in self.dep.partitions:
            initial = getattr(p, "initial_walk", None)
            if initial:
                return initial
        # Fall back to any walk that looks like a prefill entry point.
        for name in self.dep.walk_to_wgs:
            if "prefill" in name:
                return name
        return next(iter(self.dep.walk_to_wgs))

    def _start_walk(self, req: SimRequest, walk: str) -> None:
        """Instantiate this walk's worker graphs for the request and seed them.

        Deep-copies each section exactly as ``WorkerGraphQueues.add_request``
        does — per-request graph state (loop counters, readiness) must not be
        shared between requests.
        """
        from mstar.graph.graph_io import WorkerGraphIO

        previous = req.current_walk
        req.current_walk = walk
        wgs = self.dep.walk_to_wgs.get(walk, [])
        if not wgs:
            self._finish(req)
            return

        # Drop the finished walk's graphs. Streaming-consumer graphs belong to
        # a different partition and keep running across the producer's walk
        # transitions, so they are kept.
        if previous and previous != walk:
            for wg_id in [
                i for i, w in req.wg_walk.items()
                if w == previous and not self._is_stream_consumer(i)
            ]:
                req.graph_ios.pop(wg_id, None)
                req.wg_walk.pop(wg_id, None)

        for wg in wgs:
            self._instantiate(req, wg, walk)
            # Seed by ingesting the inputs the graph declares as external.
            # Going through ingest_input (rather than forcing a node onto the
            # ready list) is what keeps the registry's readiness, loop, and
            # completion bookkeeping correct — the same reason the real
            # worker routes every arriving edge through it.
            io = req.graph_ios[wg.wg_id]
            for edge in self._entry_edges(io):
                io.ingest_input(edge)
            self._drain_ready(req, wg)

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

    def _instantiate(self, req: SimRequest, wg: SimWorkerGraph, walk: str) -> Any:
        """Create (or recreate) this request's copy of one worker graph.

        Deep-copies the section exactly as ``WorkerGraphQueues.add_request``
        does — per-request graph state (loop counters, readiness) must never
        be shared between requests.
        """
        from mstar.graph.graph_io import WorkerGraphIO

        io = WorkerGraphIO(copy.deepcopy(wg.section), wg_id=wg.wg_id)
        req.graph_ios[wg.wg_id] = io
        req.wg_walk[wg.wg_id] = walk
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

        for wg_id, io in req.graph_ios.items():
            if node_name in io.nodes and req.wg_walk.get(wg_id) == req.current_walk:
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

    def _drain_ready(self, req: SimRequest, wg: SimWorkerGraph) -> None:
        """Move every newly-ready node of this graph onto its worker's queue."""
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
                walk=req.wg_walk.get(wg.wg_id, req.current_walk),
                wg_id=wg.wg_id, ready_s=self.cal.now,
            ))
        self.cal.push(self.cal.now, EventType.WORKER_POLL, leader)

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

        # Two-lane schedule. The CPU submit must finish before the GPU can
        # start; the CPU is then free to build the next step while the GPU
        # runs; the postprocess lands after the GPU completes.
        submit_s = cost.prepare_s + cost.plan_s + cost.launch_s
        post_s = cost.sample_s + self.timing.worker_step_overhead_s

        ready_s = max(i.ready_s for i in items)
        cpu_submit_start = max(w.cpu_free_s, ready_s, self.cal.now)
        cpu_submit_end = cpu_submit_start + submit_s
        gpu_start = max(w.gpu_free_s, cpu_submit_end)
        gpu_end = gpu_start + cost.gpu_s
        post_start = max(cpu_submit_end, gpu_end)
        post_end = post_start + post_s

        w.gpu_free_s = gpu_end
        w.cpu_free_s = post_end
        w.gpu_busy_s += cost.gpu_s
        w.cpu_busy_s += submit_s + post_s
        w.steps += 1
        self.step_count += 1

        self.cal.push(post_end, EventType.STEP_DONE, (rank, node, walk, items))

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

    def _on_step_done(self, rank: int, node: str, walk: str, items: list[ReadyItem]) -> None:
        w = self.workers[rank]
        w.busy = False

        for item in items:
            req = self.requests.get(item.rid)
            if req is None or req.done:
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

        # This graph finished its walk → ask the model what comes next. Only
        # the walk the conductor is driving advances the request; a streaming
        # consumer finishing a chunk is not a walk transition.
        io = req.graph_ios.get(wg_id)
        if io is not None and io.wg_state_registry.is_done:
            if req.wg_walk.get(wg_id) == req.current_walk:
                self._walk_complete(req, walk)

    def _stream_arrival(
        self, req: SimRequest, edge: Any, dest_wg: SimWorkerGraph
    ) -> bool:
        """Buffer one streamed item; report whether a chunk is now ready.

        The policy object comes from the model's own PartitionTopology, so
        window/stride arithmetic — and therefore how often the consumer runs
        per producer token — is the deployment's, not the simulator's.
        """
        node = edge.next_node
        policy = req.stream_policies.get(node)
        if policy is None:
            policy = self._chunk_policy_for(node)
            req.stream_policies[node] = policy
        req.stream_buffers[node] = req.stream_buffers.get(node, 0) + 1

        if policy is None:
            return True  # no declared policy: consume item-by-item
        buffered = req.stream_buffers[node]
        if not policy.is_ready(buffered):
            return False
        take = policy.next_chunk_size(buffered)
        policy.register_chunk(take)
        req.stream_buffers[node] = max(0, buffered - take)
        return True

    def _chunk_policy_for(self, consumer_node: str) -> Any:
        """A fresh ChunkPolicy for a consumer, from the partition topology."""
        import copy as _copy

        topo = self.dep.partition_topology
        for attr in ("connections", "_connections"):
            conns = getattr(topo, attr, None)
            if not conns:
                continue
            for conn in (conns.values() if isinstance(conns, dict) else conns):
                policy = getattr(conn, "chunk_policy", None)
                target = getattr(conn, "consumer_node", None) or getattr(
                    conn, "next_node", None
                )
                if policy is not None and (target is None or target == consumer_node):
                    return _copy.deepcopy(policy)
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
        """A token/chunk reaches the client after the delivery path's latency."""
        t = self.cal.now + self.timing.control_msg_s + self.timing.client_delivery_s
        modality = getattr(edge, "output_modality", None) or "text"
        req.chunks.append((modality, t))
        if req.first_chunk_s is None:
            req.first_chunk_s = t
        req.last_chunk_s = t

    def _walk_complete(self, req: SimRequest, walk: str) -> None:
        """Ask the model for the next walk, through the conductor hop."""
        self.cal.push(
            self.cal.now + self.timing.conductor_hop_s,
            EventType.CONDUCTOR,
            ("walk_done", req.rid, walk),
        )

    def _on_conductor(self, payload: Any) -> None:
        if payload is None:
            self._admit_waiting()
            return
        kind, rid, walk = payload
        if kind != "walk_done":
            return
        req = self.requests.get(rid)
        if req is None or req.done:
            return

        req.fwd_index += 1
        next_walk = self._next_walk(req, walk)
        if next_walk is None:
            self._finish(req)
            self._admit_waiting()
            return
        self._start_walk(req, next_walk)

    def _next_walk(self, req: SimRequest, walk: str) -> str | None:
        """The model's transition function decides; length caps override it.

        The real conductor calls ``get_partition_forward_pass_args`` and also
        force-stops at ``max_output_tokens``. A simulator cannot inspect token
        values, so EOS is modeled as reaching the request's target length —
        which is exactly what a benchmark run with ``--ignore-eos`` produces,
        and what the workload spec describes.
        """
        if req.decode_steps >= req.target_output_tokens:
            return None
        # A decode walk keeps running until the length cap; other walks hand
        # off to the decode walk if one exists.
        if "decode" in walk:
            return walk
        for name in self.dep.walk_to_wgs:
            if "decode" in name:
                return name
        return None

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

    def run(self, max_events: int = 5_000_000, until_s: float | None = None) -> None:
        handlers: dict[EventType, Callable[[Any], None]] = {
            EventType.ARRIVAL: self._on_arrival,
            EventType.CONDUCTOR: self._on_conductor,
            EventType.WORKER_POLL: self._worker_poll,
        }
        n = 0
        while n < max_events:
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
