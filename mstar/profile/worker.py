
import time
from dataclasses import dataclass, field

from mstar.profile.format import GraphTiming

# (node, graph_walk) -> accumulated GraphTiming for a single request
GraphTimings = dict[tuple[str, str], GraphTiming]


@dataclass
class ExecTimings:
    """Per-node-execution timing for one batch. The dicts are keyed by
    request_id (a batch executes the same node/walk for several requests at
    once); all values are ``time.perf_counter()`` seconds."""
    start: float | None = None
    # Forward timing is batch-level, not per-request: with async GPU execution a
    # CPU perf_counter measures the launch/enqueue span of the whole forward
    # region, and a batched launch has no per-request boundary to attribute it
    # to. ``fwd_end`` is only set for sequential / max-batch-size paths.
    fwd_start: float | None = None
    fwd_end: float | None = None

    # ── GPU-time measurement ─────────────────────────────────────────────
    #
    # ``gpu_start_event`` is recorded on the default stream by the engine
    # just before the step's first launch; the worker pairs it with the
    # batch's completion event (already recorded in _execute_on_gpu_thread)
    # and fills ``gpu_time`` after the sync it performs anyway. Both events
    # are in-stream, so the measured span is this step's own GPU work even
    # when it was enqueued behind the previous step under speculation.
    #
    # For a batch split by execute_with_max_batch_size, the FIRST sub-batch's
    # start event is kept: paired with the single completion event recorded
    # after the whole split forward, it covers all sub-batches.
    gpu_start_event: object | None = None
    gpu_time: float | None = None

    # Engine-internal CPU phase spans (seconds), summed across sub-batches.
    prepare_s: float = 0.0
    plan_s: float = 0.0
    launch_s: float = 0.0
    sample_s: float = 0.0

    # ── Executed shape ───────────────────────────────────────────────────
    #
    # What the GPU actually ran. ``padded_*`` differ from ``real_*`` when a
    # CUDA-graph replay rounds the batch up to a captured bucket: the step
    # pays for the padded shape, so that — not the real shape — is what a
    # cost model must key on. ``mode`` is "graph" for a captured replay and
    # "eager" otherwise, which are different cost regimes rather than points
    # on one curve.
    #
    # Stamped by the CUDA-graph runner (graph path) or the engine (eager
    # path); left None when nothing recorded them.
    real_bs: int | None = None
    real_num_tokens: int | None = None
    padded_bs: int | None = None
    padded_num_tokens: int | None = None
    mode: str | None = None
    #: Sum of per-request KV lengths after the step, filled by the worker.
    kv_len_total: int | None = None
    #: How many sub-batches the forward was split into (>1 means the ready
    #: set exceeded the engine's max batch size and ran serially).
    num_sub_batches: int = 1

    def record_shape(
        self,
        real_bs: int,
        real_num_tokens: int,
        padded_bs: int,
        padded_num_tokens: int,
        mode: str,
    ) -> None:
        """Record the executed shape; first writer wins.

        A forward split by ``execute_with_max_batch_size`` calls this once
        per sub-batch. Keeping the first keeps the record consistent with
        ``gpu_start_event`` (also first-wins) — and the split case is
        reported by ``num_sub_batches`` rather than by mixing shapes.
        """
        if self.mode is None:
            self.real_bs = real_bs
            self.real_num_tokens = real_num_tokens
            self.padded_bs = padded_bs
            self.padded_num_tokens = padded_num_tokens
            self.mode = mode
        else:
            self.num_sub_batches += 1

    def update(self, other: "ExecTimings"):
        # Merge sub-batch forward windows (execute_with_max_batch_size) into one
        # span covering the whole batch: earliest launch to latest return.
        if other.start is not None:
            self.start = (
                other.start if self.start is None
                else min(self.start, other.start)
            )
        if other.fwd_start is not None:
            self.fwd_start = (
                other.fwd_start if self.fwd_start is None
                else min(self.fwd_start, other.fwd_start)
            )
        if other.fwd_end is not None:
            self.fwd_end = (
                other.fwd_end if self.fwd_end is None
                else max(self.fwd_end, other.fwd_end)
            )
        # Keep the earliest GPU start event: the parent batch's completion
        # event is recorded after every sub-batch has been submitted, so the
        # first sub-batch's start event brackets the whole split forward.
        if self.gpu_start_event is None:
            self.gpu_start_event = other.gpu_start_event
        self.prepare_s += other.prepare_s
        self.plan_s += other.plan_s
        self.launch_s += other.launch_s
        self.sample_s += other.sample_s


@dataclass
class WorkerProfileInfo:
    """Accumulates per-request graph timings on a worker.

    Every executed batch is recorded once, at postprocess time, via
    ``register_end`` — by then the batch's :class:`ExecTimings` carries
    ``start`` / ``fwd_start`` (stamped by the engine at the GPU-launch boundary)
    and ``fwd_end`` (stamped by the worker after the GPU completion event). All
    the data lives on the batch's ``ExecTimings`` (passed in by the caller), so
    no per-batch state has to be carried across the pipeline here; speculative
    and fallthrough batches alike flow through ``_postprocess_batch`` once and
    are recorded there.

    Timings accumulate per request and per ``(node, graph_walk)`` so repeated
    steps for a request (e.g. decode) sum into one entry with ``exec_count``.
    """
    # request_id -> {(node, graph_walk) -> accumulated GraphTiming}
    per_rid_graph_timings: dict[str, GraphTimings] = field(default_factory=dict)

    def register_end(
        self,
        node: str,
        walk: str,
        rids: list[str],
        timings: ExecTimings,
        end_time: float | None = None,
    ):
        """Emit a per-request GraphTiming for every request in a finished batch.

        ``timings`` is the batch's :class:`ExecTimings`; ``end_time`` defaults to
        now (the postprocess point). Forward timing is batch-level and shared
        across the batch's requests; the bracket bounds are used as fallbacks if
        a stamp is missing.
        """
        if timings.start is None:
            # Engine never stamped this batch (e.g. a path that ran no forward);
            # nothing meaningful to record.
            return
        if end_time is None:
            end_time = time.perf_counter()

        start = timings.start
        fwd_start = timings.fwd_start if timings.fwd_start is not None else start
        fwd_end = timings.fwd_end if timings.fwd_end is not None else end_time
        for rid in rids:
            timing = GraphTiming(
                node=node,
                graph_walk=walk,
                exec_count=1,
                total_time=end_time - start,
                forward_time=fwd_end - fwd_start,
                preprocess_time=fwd_start - start,
                postprocess_time=end_time - fwd_end,
                gpu_time=timings.gpu_time,
                prepare_time=timings.prepare_s,
                plan_time=timings.plan_s,
                launch_time=timings.launch_s,
                sample_time=timings.sample_s,
            )
            rid_timings = self.per_rid_graph_timings.setdefault(rid, {})
            if (node, walk) in rid_timings:
                rid_timings[(node, walk)] += timing
            else:
                rid_timings[(node, walk)] = timing

    def pop_request(self, rid: str) -> GraphTimings:
        """Drop and return a request's accumulated timings (called on removal)."""
        return self.per_rid_graph_timings.pop(rid, {})
