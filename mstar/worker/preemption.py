"""The worker's half of preemption: the moves a paged KV pool chooses when a request runs short or a waiting request
doesn't fit, made through the engine, and the counts the metrics read. The pool keeps every victim, so nothing here
outlives a call."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mstar.worker.worker_metrics import PreemptionCounts

if TYPE_CHECKING:
    from mstar.engine.engine import Engine, ExecutingBatch, PagedKVCache
    from mstar.engine.resources.kv.manager import Victims
    from mstar.worker.engine_manager import EngineManager
    from mstar.worker.micro_scheduler import MicroScheduler
    from mstar.worker.node_manager_utils import RequestStateManager
    from mstar.worker.rebuild import Prefilling, Rebuilds


@dataclass(slots=True)
class Preempted:
    """How often a request was preempted, and how many samples it had kept when it was last preempted."""
    times: int = 0
    sampled_at: int = -1


class Preemption:
    def __init__(
        self, engine_manager: EngineManager, rebuilds: Rebuilds, scheduler: MicroScheduler,
        request_state: RequestStateManager, log: Callable[..., None], rid_str: Callable[[int], str],
    ) -> None:
        """``log`` is the worker's throttled log, and ``rid_str`` names a request."""
        self._engine_manager = engine_manager
        self._rebuilds = rebuilds
        self._scheduler = scheduler
        self._request_state = request_state
        self._log = log
        self._rid_str = rid_str
        # `_keys`: node -> the keys of its paged KV caches, and `_caches`: key -> its cache.
        # `_preempted`: handle -> its `Preempted`. `counts`: node -> its counts for the metrics.
        # `_waits_logged`: (node, walk) -> when its last wait was logged, and the waits since.
        # `_moves_logged`: node -> when its last move was logged, and the moves since.
        # `_walked`: what the last walk of `make_room` that moved nothing read, or None.
        self._keys: dict[str, list[str]] = defaultdict(list)
        self._caches: dict[str, PagedKVCache] = {}
        for kv in engine_manager.engine.paged_kv_caches():
            self._caches[kv.key] = kv
            for node_name in kv.nodes:
                self._keys[node_name].append(kv.key)
        for kv in self._caches.values():
            kv.cache.moves = all(self._moves(kv, node_name) for node_name in kv.nodes)
        self._preempted: dict[int, Preempted] = {}
        self.counts: dict[str, PreemptionCounts] = defaultdict(PreemptionCounts)
        self._waits_logged: dict[tuple[str, str], tuple[float, int]] = {}
        self._moves_logged: dict[str, tuple[float, int]] = {}
        self._walked: tuple | None = None

    def preempt(self, node_batch: ExecutingBatch) -> bool:
        """`KVManager.preempt` chooses victims for the request that hit KV OOM. Move each victim to the host or
        drop it, or, when no move covers the shortage, move the request itself to the host or queue it for its
        shortage, and return True. Return False, so the offload and the hold run, when the failed resource isn't a
        paged KV cache, when the node rebuilds no request here and the cache has no host pool or isn't the node's
        only one, or when `KVManager.preempt` returns None."""
        kv = self._caches.get(node_batch.failed_resource)
        node_name = node_batch.node_name
        if kv is None or not self._moves(kv, node_name):
            return False
        reason, engine = node_batch.admit_error, self._engine_manager.get_engine(node_name)
        newer, older = engine.newer_than(reason.request_id), engine.older_than(reason.request_id)
        victims = kv.cache.preempt(reason.request_id, reason.pages_short, newer, older, self._droppable(kv, engine))
        if victims is None:
            return False
        if victims.waits:
            for node in kv.nodes:
                self._engine_manager.get_engine(node).readmit_request(node, reason.request_id)
            self.counts[node_name].take_back_waits += victims.takes_back
            self._log(
                self._waits_logged, (node_name, node_batch.step_context.graph_walk),
                "KV short on node=%s walk=%s: %s waits for %d pages %s, since moving "
                "what the pool can move wouldn't cover its shortage of %d (%d earlier waits not logged)",
                node_name, node_batch.step_context.graph_walk, self._rid_str(reason.request_id), victims.waits,
                "it lent" if victims.takes_back else "past its allotment", reason.pages_short,
            )
        self._move(kv, self.counts[node_name], victims, reason.request_id)
        return True

    def make_room(self) -> None:
        """Move the newer requests that wait holding pages for the oldest waiting request a paged KV pool refused
        that such moves can let in, as `KVManager.make_room` chooses them. No step of theirs runs short to move
        them, since none of them runs. An older request stuck on something no move frees, another resource or a
        pool that doesn't move, holds up no younger one, and one the room now fits moves nothing, so no younger one
        takes that room.

        A walk that moved nothing runs again only once the refusals or what the pools can move have changed. A
        refusal's mark changes once its room grows, but a request becomes droppable when its ids land, which grows
        nothing."""
        if not any(kv.cache.any_stalled() for kv in self._caches.values()):
            return
        engine = self._engine_manager.engine
        refusals = engine.refusals()
        walked = (refusals, [kv.cache.stalled_moves(self._droppable(kv, engine)) for kv in self._caches.values()])
        if walked == self._walked:
            return
        self._walked = None
        for refusal in refusals:
            kv = self._caches.get(refusal.key)
            if kv is None or not self._moves(kv, refusal.node_name):
                continue
            newer = engine.newer_than(refusal.rid)
            victims = None if newer is None else kv.cache.make_room(
                refusal.mark, newer, engine.older_than(refusal.rid), self._droppable(kv, engine),
            )
            if victims is not None:
                self._move(kv, self.counts[refusal.node_name], victims, refusal.rid)
                return
        self._walked = walked

    def rebuilt(self, rid: int, node_name: str, tokens: int) -> None:
        """Count the ``tokens`` a row of ``rid``'s rebuild on ``node_name`` ran."""
        self.counts[node_name].rebuild_tokens += tokens

    def refuse_drops(self, rids: Collection[int], node_name: str) -> None:
        """Tell each paged KV cache of ``node_name`` that no drop can free ``rids``, whose ids the node forgot."""
        for rid in rids:
            for key in self._keys[node_name]:
                self._caches[key].cache.refuse_drops(rid)

    def remove_request(self, rid: int) -> None:
        self._preempted.pop(rid, None)

    def _moves(self, kv: PagedKVCache, node_name: str) -> bool:
        """Whether the pool moves requests for ``node_name``: the node rebuilds requests here, or the cache has a host
        pool and is the node's only one."""
        return node_name in self._rebuilds.nodes or (kv.cache.supports_eviction and self._keys[node_name] == [kv.key])

    def _droppable(self, kv: PagedKVCache, engine: Engine) -> Callable[[int], bool]:
        """Whether a rebuild can write a request's KV in ``kv`` again on every node that holds it, and its launched
        steps allow the drop."""
        nodes = [node for node in kv.nodes if node in self._rebuilds.nodes]

        def droppable(rid: int) -> bool:
            held = self._held(rid, nodes)
            return bool(held) and all(
                self._rebuild_refusal(rid, node) is None
                and engine.drop_refusal(node, rid, self._tokens(rid, node)) is None
                for node in held
            )
        return droppable

    def _held(self, rid: int, nodes: list[str]) -> list[str]:
        """The nodes of ``nodes`` where a rebuild can write ``rid``'s KV again: it keeps ids there, or its prompt
        is mid-prefill there."""
        return [node for node in nodes if self._rebuilds.kept(rid, node) is not None or self._prefilling(rid, node)]

    def _prefilling(self, rid: int, node_name: str) -> Prefilling | None:
        info = self._request_state.get_fwd_info(rid, self._request_state.get_partition_for_node(node_name))
        return self._scheduler.prefilling(node_name, rid, info.graph_walk)

    def _rebuild_refusal(self, rid: int, node_name: str) -> str | None:
        """Why a rebuild can't write ``rid``'s KV on ``node_name`` again, or None."""
        info = self._request_state.get_fwd_info(rid, self._request_state.get_partition_for_node(node_name))
        return self._rebuilds.refusal(
            rid, node_name, info.graph_walk, info.dynamic_loop_iter_counts,
            self._scheduler.prefilling(node_name, rid, info.graph_walk),
        )

    def _tokens(self, rid: int, node_name: str) -> int:
        """`Rebuilds.tokens` for ``rid``, which keeps ids or is mid-prefill once `_rebuild_refusal` allows its drop."""
        return self._rebuilds.tokens(rid, node_name, self._prefilling(rid, node_name))

    def _drop(self, rid: int, node_name: str) -> int:
        """Drop ``rid``'s KV on ``node_name``, which a rebuild can write again, and return the pages freed."""
        why = self._rebuild_refusal(rid, node_name)
        if why is not None:
            raise RuntimeError(f"{self._rid_str(rid)} can't be dropped on {node_name}, since {why}")
        if any(rid in batch.request_to_worker_graph for (node, _), batch in self._scheduler.backlog.items()
               if node == node_name):
            raise RuntimeError(f"{self._rid_str(rid)} is in a backlog on {node_name}")
        return self._engine_manager.get_engine(node_name).drop_request(node_name, rid, self._tokens(rid, node_name))

    def _move(self, kv: PagedKVCache, counts: PreemptionCounts, victims: Victims, owner: int) -> None:
        """Count each victim in ``counts`` and move it to the host, drop it, or ask the node's resources again for
        one that gave back its allotment, as ``kv`` decided for ``owner``. A backlogged row goes back to the runtime
        first, so it doesn't run on pages the move gave up."""
        nodes = [node for node in kv.nodes if node in self._rebuilds.nodes]
        for rid in (*victims.to_host, *victims.to_drop, *victims.to_ask):
            record = self._preempted.setdefault(rid, Preempted())
            record.times += 1
            counts.victims += 1
            counts.take_backs += victims.takes_back
            counts.preempted_again += record.times == 2
            kept = next((kept for node in nodes if (kept := self._rebuilds.kept(rid, node)) is not None), None)
            if kept is not None:
                counts.returns_preempted += record.sampled_at == len(kept.sampled)
                record.sampled_at = len(kept.sampled)
            self._scheduler.return_backlogged(rid)
        for rid in victims.to_host:
            for node_name in kv.nodes:
                freed = self._engine_manager.get_engine(node_name).offload_request(node_name, rid)
                self._log(
                    self._moves_logged, node_name,
                    "KV short on node=%s: moved %s to the host, %d pages freed for %s, "
                    "preempted %d times (%d earlier moves not logged)",
                    node_name, self._rid_str(rid), freed, self._rid_str(owner), self._preempted[rid].times,
                )
        for rid in victims.to_drop:
            for node_name in self._held(rid, nodes):
                freed = self._drop(rid, node_name)
                self.counts[node_name].drops += 1
                self._log(
                    self._moves_logged, node_name,
                    "KV short on node=%s: dropped %s, %d pages freed for %s, "
                    "preempted %d times (%d earlier moves not logged)",
                    node_name, self._rid_str(rid), freed, self._rid_str(owner), self._preempted[rid].times,
                )
        for rid in victims.to_ask:
            for node_name in kv.nodes:
                self._engine_manager.get_engine(node_name).readmit_request(node_name, rid)
                self._log(
                    self._moves_logged, node_name,
                    "KV short on node=%s: %s, holding no page of its own, gave back its allotment for %s, "
                    "preempted %d times (%d earlier moves not logged)",
                    node_name, self._rid_str(rid), self._rid_str(owner), self._preempted[rid].times,
                )
