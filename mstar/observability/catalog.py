"""Metric families whose values come from a process other than the API server.

Dependency-free, so the conductor and workers can import it: they record into
:mod:`mstar.observability.recorders` by these names, and the API server builds
the Prometheus families from the same entries, so the names cannot drift.
``model_name`` is added by the API server and is not listed in ``labels``.
"""

from __future__ import annotations

from dataclasses import dataclass

COUNTER, GAUGE, HISTOGRAM = "counter", "gauge", "histogram"


@dataclass(frozen=True)
class Family:
    name: str
    kind: str
    help: str
    labels: tuple[str, ...]
    #: Histogram upper bounds, ascending, without ``+Inf``.
    buckets: tuple[float, ...] = ()


#: How often the conductor sends its status message while metrics are on.
STATUS_INTERVAL_S = 1.0

#: ``waiting``: queued at ``max_concurrent_requests``, not dispatched yet.
#: ``running``: dispatched to workers and not finished. ``draining``: finished
#: (completed, aborted or failed) and waiting for every reader to release its
#: outputs before the workers free its state; it still counts toward
#: ``max_concurrent_requests``.
CONDUCTOR_REQUEST_STATES = ("waiting", "running", "draining")

CONDUCTOR_REQUESTS = Family(
    "mstar_conductor_requests", GAUGE,
    "Requests the conductor holds, by state.",
    ("state",),
)
COMPONENT_IN_FLIGHT = Family(
    "mstar_component_in_flight", GAUGE,
    "Running requests with a pass on this graph node that the conductor "
    "dispatched and whose completion it has not received yet.",
    ("component",),
)
REQUEST_QUEUE_SECONDS = Family(
    "mstar_request_queue_seconds", HISTOGRAM,
    "Successful requests: the conductor taking the request in to admitting it "
    "for dispatch (zero unless it queued at max_concurrent_requests).",
    ("output_modalities",),
    buckets=(
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
        10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
    ),
)

KV_PAGES_USED = Family(
    "mstar_kv_pages_used", GAUGE,
    "Pages of a paged KV cache not on its free list: held by requests, or kept "
    "by the prefix cache (see mstar_kv_pages_cached).",
    ("component", "resource"),
)
KV_PAGES_TOTAL = Family(
    "mstar_kv_pages_total", GAUGE,
    "Pages of a paged KV cache that requests can hold.",
    ("component", "resource"),
)
KV_PAGES_CACHED = Family(
    "mstar_kv_pages_cached", GAUGE,
    "Pages of a paged KV cache its prefix cache names, including pages "
    "running requests also hold; the cache frees the others on demand.",
    ("component", "resource"),
)
STEP_SECONDS = Family(
    "mstar_step_seconds", HISTOGRAM,
    "Engine steps: from the later of the step's hand-off to the GPU thread and "
    "the end of the worker's previous step, to the end of this step's "
    "postprocess. In steady state, the interval between step completions.",
    ("component", "graph_walk"),
    buckets=(
        0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03,
        0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
    ),
)
BATCH_SIZE = Family(
    "mstar_batch_size", HISTOGRAM,
    "Engine steps: rows the step's forward ran.",
    ("component", "graph_walk"),
    buckets=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0),
)
PREFIX_CACHE_QUERY_TOKENS = Family(
    "mstar_prefix_cache_query_tokens_total", COUNTER,
    "Prompt tokens looked up in the prefix cache, once per request.",
    ("component",),
)
PREFIX_CACHE_HIT_TOKENS = Family(
    "mstar_prefix_cache_hit_tokens_total", COUNTER,
    "Prompt tokens the prefix cache already held at that lookup.",
    ("component",),
)
KV_PREEMPTIONS = Family(
    "mstar_kv_preemptions_total", COUNTER,
    "Requests chosen to move to the host or drop when an older request's grant, or their own, ran short of KV pages.",
    ("component",),
)
KV_DROPS = Family(
    "mstar_kv_drops_total", COUNTER,
    "Preempted requests whose KV was dropped, to be rebuilt from their token ids and any host copies.",
    ("component",),
)
KV_REBUILD_TOKENS = Family(
    "mstar_kv_rebuild_tokens_total", COUNTER,
    "Tokens of kept history the rebuild rows prefilled, cache hits included.",
    ("component",),
)
KV_REBUILD_CACHED_PAGES = Family(
    "mstar_kv_rebuild_cached_pages_total", COUNTER,
    "Pages a rebuild row found in the prefix cache.",
    ("component",),
)
KV_PREEMPTED_AGAIN = Family(
    "mstar_kv_preempted_again_total", COUNTER,
    "Requests preempted a second time.",
    ("component",),
)
KV_RETURNS_PREEMPTED = Family(
    "mstar_kv_returns_preempted_total", COUNTER,
    "Dropped requests preempted again after their return, before they kept another token.",
    ("component",),
)
COMPONENT_PENDING_STEPS = Family(
    "mstar_component_pending_steps", GAUGE,
    "Requests whose next step on this graph node is ready (its inputs arrived) "
    "but not in a running step: waiting for a batch slot, or for the worker to "
    "schedule the node.",
    ("component",),
)

#: ``src`` / ``dst`` of a tensor edge into or out of the graph: the data
#: worker (a thread of the API server) sends the request's inputs and reads
#: its results. Every other endpoint is a graph node name.
TRANSFER_API_SERVER = "api_server"

_TRANSFER_LABELS = ("src", "dst", "transport")

# Counted by the consumer, once per edge read: the tensors of one graph edge it
# read from one producer, over the transport (not a tensor already in its own
# store). ``transport`` is the protocol: SHM, TCP or RDMA.
TRANSFER_BYTES = Family(
    "mstar_transfer_bytes_total", COUNTER,
    "Tensor bytes the consumer read from the producer over this edge.",
    _TRANSFER_LABELS,
)
TRANSFERS = Family(
    "mstar_transfers_total", COUNTER,
    "Tensors the consumer read from the producer over this edge.",
    _TRANSFER_LABELS,
)
TRANSFER_SECONDS = Family(
    "mstar_transfer_seconds", HISTOGRAM,
    "Consumer-side time to read one edge's tensors from the producer (one "
    "observation per edge read; the producer's own write or registration is "
    "not included).",
    _TRANSFER_LABELS,
    buckets=(
        0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025,
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    ),
)

#: Families the engine workers report to the conductor (``WorkerStatus``); the
#: conductor sums the workers' series and forwards them in its status message.
WORKER_FAMILIES: dict[str, Family] = {
    f.name: f for f in (
        KV_PAGES_USED, KV_PAGES_TOTAL, KV_PAGES_CACHED, STEP_SECONDS, BATCH_SIZE,
        PREFIX_CACHE_QUERY_TOKENS, PREFIX_CACHE_HIT_TOKENS, KV_PREEMPTIONS, KV_DROPS, KV_REBUILD_TOKENS,
        KV_PREEMPTED_AGAIN, KV_RETURNS_PREEMPTED, KV_REBUILD_CACHED_PAGES, COMPONENT_PENDING_STEPS,
        TRANSFER_BYTES, TRANSFERS, TRANSFER_SECONDS,
    )
}

#: Families the conductor's status message carries, exported from the latest
#: message.
STATUS_FAMILIES: dict[str, Family] = {
    f.name: f for f in (CONDUCTOR_REQUESTS, COMPONENT_IN_FLIGHT)
} | WORKER_FAMILIES
