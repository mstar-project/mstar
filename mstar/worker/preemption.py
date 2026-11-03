"""The worker's half of preemption: the moves a paged KV pool chooses when a request runs short, made through the
engine, and the counts the metrics read. The pool keeps every victim, so nothing here outlives a call."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mstar.worker.worker_metrics import PreemptionCounts

if TYPE_CHECKING:
    from mstar.engine.engine import ExecutingBatch, PagedKVCache
    from mstar.engine.resources.kv.manager import Victims
    from mstar.worker.engine_manager import EngineManager
    from mstar.worker.micro_scheduler import MicroScheduler
    from mstar.worker.rebuild import Rebuilds


@dataclass(slots=True)
class Preempted:
    """How often a request was preempted, and how many samples it had kept when it last returned."""
    times: int = 0
    returned_at: int = -1


class Preemption:
    def __init__(
        self, engine_manager: EngineManager, rebuilds: Rebuilds, scheduler: MicroScheduler,
        drop: Callable[[int, str], int], rebuild_refusal: Callable[[int, str], str | None],
        log: Callable[..., None], rid_str: Callable[[int], str],
    ) -> None:
        """``drop`` drops a request's KV on a node and returns the pages freed, ``rebuild_refusal`` says why a
        request's kept ids can't rebuild it on a node, ``log`` is the worker's throttled log, and ``rid_str``
        names a request."""
        self._engine_manager = engine_manager
        self._rebuilds = rebuilds
        self._scheduler = scheduler
        self._drop = drop
        self._rebuild_refusal = rebuild_refusal
        self._log = log
        self._rid_str = rid_str
        # `_keys`: node -> the keys of its paged KV caches, and `_caches`: key -> its cache.
        # `_preempted`: handle -> its `Preempted`. `counts`: node -> its counts for the metrics.
        # `_waits_logged`: (node, walk) -> when its last wait was logged, and the waits since.
        # `_moves_logged`: node -> when its last move was logged, and the moves since.
        self._keys: dict[str, list[str]] = defaultdict(list)
        self._caches: dict[str, PagedKVCache] = {}
        for kv in engine_manager.engine.paged_kv_caches():
            self._caches[kv.key] = kv
            for node_name in kv.nodes:
                self._keys[node_name].append(kv.key)
        self._preempted: dict[int, Preempted] = {}
        self.counts: dict[str, PreemptionCounts] = defaultdict(PreemptionCounts)
        self._waits_logged: dict[tuple[str, str], tuple[float, int]] = {}
        self._moves_logged: dict[str, tuple[float, int]] = {}

    def preempt(self, node_batch: ExecutingBatch) -> bool:
        """`KVManager.preempt` chooses victims for the request that hit KV OOM. Move each victim to the host or
        drop it, or queue the request for its shortage when no move covers it, and return True. Return False, so
        the offload and the hold run, when the failed resource isn't a paged KV cache, when the node rebuilds no
        request here and the cache has no host pool or isn't the node's only one, or when `KVManager.preempt`
        returns None."""
        kv = self._caches.get(node_batch.failed_resource)
        node_name = node_batch.node_name
        if kv is None or not (
            node_name in self._rebuilds.nodes or (kv.cache.supports_eviction and self._keys[node_name] == [kv.key])
        ):
            return False
        reason, engine = node_batch.admit_error, self._engine_manager.get_engine(node_name)
        nodes = [node for node in kv.nodes if node in self._rebuilds.nodes]

        def droppable(rid: int) -> bool:
            held = [node for node in nodes if self._rebuilds.kept(rid, node) is not None]
            return bool(held) and all(
                self._rebuild_refusal(rid, node) is None and engine.drop_refusal(node, rid) is None for node in held
            )

        newer = engine.newer_than(reason.request_id)
        victims = kv.cache.preempt(reason.request_id, reason.pages_short, newer, droppable)
        if victims is None:
            return False
        if victims.waits:
            for node in kv.nodes:
                self._engine_manager.get_engine(node).readmit_request(node, reason.request_id)
            self._log(
                self._waits_logged, (node_name, node_batch.step_context.graph_walk),
                "KV short on node=%s walk=%s: %s waits in the queue for its shortage of %d pages, since moving what "
                "the pool can move wouldn't cover it (%d earlier waits not logged)",
                node_name, node_batch.step_context.graph_walk, self._rid_str(reason.request_id), reason.pages_short,
            )
        counts = self.counts[node_name]
        for rid in (*victims.to_host, *victims.to_drop, *victims.to_ask):
            record = self._preempted.setdefault(rid, Preempted())
            record.times += 1
            counts.victims += 1
            counts.preempted_again += record.times == 2
            kept = next((kept for node in nodes if (kept := self._rebuilds.kept(rid, node)) is not None), None)
            counts.returns_preempted += kept is not None and record.returned_at == len(kept.sampled)
        self._move(kv, victims, reason.pages_short)
        return True

    def returned(self, rid: int, node_name: str) -> None:
        """Note that a preempted request was admitted again on ``node_name``, with the samples it has kept."""
        if rid in self._preempted:
            self._preempted[rid].returned_at = len(self._rebuilds.kept(rid, node_name).sampled)

    def remove_request(self, rid: int) -> None:
        self._preempted.pop(rid, None)

    def _move(self, kv: PagedKVCache, victims: Victims, pages_short: int) -> None:
        """Move each victim to the host, drop it, or ask the node's resources again for one that gave back its
        allotment, as `KVManager.preempt` decided. A backlogged row goes back to the runtime and a queued rebuild
        row leaves its queue first, so neither runs on pages the move gave up."""
        for rid in (*victims.to_host, *victims.to_drop, *victims.to_ask):
            self._scheduler.return_backlogged(rid)
            self._scheduler.take_rebuilds((rid,))
        for rid in victims.to_host:
            for node_name in kv.nodes:
                freed = self._engine_manager.get_engine(node_name).offload_request(node_name, rid)
                self._log(
                    self._moves_logged, node_name,
                    "KV short on node=%s: moved %s to the host, %d pages freed for a shortage of %d, "
                    "preempted %d times (%d earlier moves not logged)",
                    node_name, self._rid_str(rid), freed, pages_short, self._preempted[rid].times,
                )
        for rid in victims.to_drop:
            for node_name in kv.nodes:
                if self._rebuilds.kept(rid, node_name) is not None:
                    freed = self._drop(rid, node_name)
                    self.counts[node_name].drops += 1
                    self._log(
                        self._moves_logged, node_name,
                        "KV short on node=%s: dropped %s, %d pages freed for a shortage of %d, "
                        "preempted %d times (%d earlier moves not logged)",
                        node_name, self._rid_str(rid), freed, pages_short, self._preempted[rid].times,
                    )
        for rid in victims.to_ask:
            for node_name in kv.nodes:
                self._engine_manager.get_engine(node_name).readmit_request(node_name, rid)
                self._log(
                    self._moves_logged, node_name,
                    "KV short on node=%s: %s, holding no page of its own, gave back its allotment for a shortage of "
                    "%d, preempted %d times (%d earlier moves not logged)",
                    node_name, self._rid_str(rid), pages_short, self._preempted[rid].times,
                )
