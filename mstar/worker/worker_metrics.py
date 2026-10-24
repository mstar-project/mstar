"""The engine worker's metrics, built only while metrics are on.

Steps are recorded on the main thread as they finish (two histogram bumps, no
per-request work), and the tensor manager records each edge it reads
(``EdgeTransfers``). Everything else is sampled at most once per status
interval, when the worker sends the conductor one ``WorkerStatus`` with
cumulative snapshots; the conductor forwards them to the API server.

The gauges are live from worker start. The histograms and counters, and the
transfers, start at ``start_counting()``, which the API server triggers after
its warmup requests, as it does for its own families.

Engine families cover only the nodes this rank leads (``components``): the
other ranks of a lockstep instance run the same steps over the same page
counts. Transfers are every rank's own reads (each reads its slice).
Recording never changes what any rank runs.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import astuple, dataclass
from typing import TYPE_CHECKING

from mstar.observability import catalog
from mstar.observability.recorders import (
    Counter,
    EdgeTransfers,
    Gauge,
    Histogram,
    MetricSnapshot,
)
from mstar.utils.ipc_format import WorkerStatus

if TYPE_CHECKING:
    from mstar.engine.engine import Engine
    from mstar.model.submodule_base import NodeInputs
    from mstar.worker.micro_scheduler import MicroScheduler

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreemptionCounts:
    """A node's preemptions since its worker started."""
    # `rebuild_tokens`: KV slots the rebuild rows prefilled, image slots included.
    # `returns_preempted`: returned requests preempted again before they kept another token.
    victims: int = 0
    drops: int = 0
    rebuild_tokens: int = 0
    preempted_again: int = 0
    returns_preempted: int = 0


class WorkerMetrics:
    def __init__(
        self,
        worker_id: str,
        interval_s: float,
        components: frozenset[str],
        engine: Engine,
        scheduler: MicroScheduler,
        send: Callable[[WorkerStatus], None],
        *,
        enable_transfers: Callable[[], EdgeTransfers],
        preemptions: Mapping[str, PreemptionCounts],
    ):
        self.worker_id = worker_id
        self.components = components
        self._interval_s = interval_s
        self._next_report = 0.0
        self._report_failed = False
        self._scheduler = scheduler
        self._send = send
        self._enable_transfers = enable_transfers
        # empty until start_counting() swaps in the tensor manager's
        self._transfers = EdgeTransfers("")
        # steps whose time starts from then on are counted
        self._counting_since = math.inf
        # per cache, its prefix_tokens at start_counting(); None before
        self._prefix_base: list[dict[str, tuple[int, int]]] | None = None
        # `_preemptions`: the worker's counts per node.
        # `_preemption_base`, `_rebuild_base`: those counts, and each cache's rebuild_pages, at start_counting().
        self._preemptions = preemptions
        self._preemption_base: dict[str, tuple] = {}
        self._rebuild_base: list[dict[str, int]] = []
        # (resource key, its reported nodes, cache); a cache only follower
        # nodes use is left to their lead rank
        self._kv = [
            (kv.key, sorted(kv.nodes & components), kv.cache)
            for kv in engine.paged_kv_caches() if kv.nodes & components
        ]
        # when the worker's last step finished postprocessing
        self._last_done = 0.0
        self._step_seconds = Histogram(catalog.STEP_SECONDS)
        self._batch_size = Histogram(catalog.BATCH_SIZE)
        self._kv_used = Gauge(catalog.KV_PAGES_USED)
        self._kv_total = Gauge(catalog.KV_PAGES_TOTAL)
        self._kv_cached = Gauge(catalog.KV_PAGES_CACHED)
        self._prefix_query = Counter(catalog.PREFIX_CACHE_QUERY_TOKENS)
        self._prefix_hit = Counter(catalog.PREFIX_CACHE_HIT_TOKENS)
        self._pending = Gauge(catalog.COMPONENT_PENDING_STEPS)
        self._preemption_counters = [Counter(family) for family in (
            catalog.KV_PREEMPTIONS, catalog.KV_DROPS, catalog.KV_REBUILD_TOKENS, catalog.KV_PREEMPTED_AGAIN,
            catalog.KV_RETURNS_PREEMPTED,
        )]
        self._rebuild_pages = Counter(catalog.KV_REBUILD_CACHED_PAGES)

    def start_counting(self) -> None:
        """Count the steps whose time starts from now on (see ``step_done``),
        the prefix-cache lookups after this, and the tensors this rank reads
        from now on."""
        self._counting_since = time.perf_counter()
        self._prefix_base = [dict(cache.prefix_tokens) for _, _, cache in self._kv]
        self._preemption_base = {node: astuple(counts) for node, counts in self._preemptions.items()}
        self._rebuild_base = [dict(cache.rebuild_pages) for _, _, cache in self._kv]
        self._transfers = self._enable_transfers()

    def step_done(
        self, node: str, graph_walk: str, inputs: list[NodeInputs] | None,
        submitted_at: float,
    ) -> None:
        """A step finished postprocessing. Its time counts from
        ``submitted_at`` (the hand-off to the GPU thread) or from the end of
        the worker's previous step if that was later, so pipelined steps are
        not counted twice."""
        now = time.perf_counter()
        start = submitted_at if submitted_at > self._last_done else self._last_done
        self._last_done = now
        # A step whose rows were all declined or served from the prefix cache
        # ran no forward.
        if inputs and node in self.components and start >= self._counting_since:
            labels = (node, graph_walk)
            self._step_seconds.observe(labels, now - start)
            self._batch_size.observe(labels, len(inputs))

    def maybe_report(self) -> None:
        """Send the conductor a ``WorkerStatus`` once per status interval."""
        now = time.perf_counter()
        if now < self._next_report:
            return
        self._next_report = now + self._interval_s
        try:
            self._send(WorkerStatus(worker_id=self.worker_id, metrics=self.snapshot()))
        except Exception:
            # The run loop's error handler would fail the steps in flight; a
            # missed report only leaves the exported values one interval older.
            if not self._report_failed:
                self._report_failed = True
                logger.exception("Worker %s: could not send its metrics", self.worker_id)

    def snapshot(self) -> list[MetricSnapshot]:
        """Every worker family in ``catalog.WORKER_FAMILIES`` order."""
        prefix: dict[str, tuple[int, int]] = {}
        for i, (key, nodes, cache) in enumerate(self._kv):
            counts = cache.page_counts()
            for node in nodes:
                labels = (node, key)
                self._kv_used.set(labels, counts.used)
                self._kv_total.set(labels, counts.total)
                self._kv_cached.set(labels, counts.cached)
            if self._prefix_base is None:
                continue
            base = self._prefix_base[i]
            # list() first: the GPU thread adds a node's first lookup while this runs
            for node, (queried, hit) in list(cache.prefix_tokens.items()):
                if node in self.components:
                    q0, h0 = base.get(node, (0, 0))
                    q, h = prefix.get(node, (0, 0))
                    prefix[node] = (q + queried - q0, h + hit - h0)
        for node, (queried, hit) in prefix.items():
            # the caches keep cumulative totals, so these are set, not added
            self._prefix_query.set((node,), queried)
            self._prefix_hit.set((node,), hit)
        if self._prefix_base is not None:
            for node, counts in self._preemptions.items():
                if node in self.components:
                    base = self._preemption_base.get(node, (0,) * len(self._preemption_counters))
                    for counter, now, then in zip(self._preemption_counters, astuple(counts), base, strict=True):
                        counter.set((node,), now - then)
            for (_, _, cache), base in zip(self._kv, self._rebuild_base, strict=True):
                for node, pages in list(cache.rebuild_pages.items()):
                    if node in self.components:
                        self._rebuild_pages.set((node,), pages - base.get(node, 0))
        # a rank leading no node reports only its transfers
        pending = self._scheduler.pending_rows() if self.components else {}
        for node in self.components:
            self._pending.set((node,), pending.get(node, 0))
        return [
            self._kv_used.snapshot(), self._kv_total.snapshot(), self._kv_cached.snapshot(),
            self._step_seconds.snapshot(), self._batch_size.snapshot(),
            self._prefix_query.snapshot(), self._prefix_hit.snapshot(),
            *(counter.snapshot() for counter in self._preemption_counters), self._rebuild_pages.snapshot(),
            self._pending.snapshot(), *self._transfers.snapshot(),
        ]
