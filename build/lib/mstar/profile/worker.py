
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
            )
            rid_timings = self.per_rid_graph_timings.setdefault(rid, {})
            if (node, walk) in rid_timings:
                rid_timings[(node, walk)] += timing
            else:
                rid_timings[(node, walk)] = timing

    def pop_request(self, rid: str) -> GraphTimings:
        """Drop and return a request's accumulated timings (called on removal)."""
        return self.per_rid_graph_timings.pop(rid, {})
