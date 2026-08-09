"""Step sequencing for engines with KV-cache resources.

The runner owns which lifecycle stage runs when. It holds no domain state
of its own: residency is admitted through the pool before inputs are
prepared, the step's plan surface is built before the forward runs, and
durable state is published through the pool once the step is done. What
each stage does lives on the resources; the runner only drives them in
order.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mstar.engine.resources.kv_pool import KVCachePool

if TYPE_CHECKING:
    from mstar.engine.cache_manager import BatchedCacheManager


@dataclass
class StepPlan:
    """One batch's planned execution shape: which path runs it and the plan
    surface each forward call uses. The graph path builds no surface here
    (the graph runner owns its captured one); the batched path shares one
    across the batch; the sequential path gets one per request."""
    mode: str
    cache_manager: "BatchedCacheManager | None" = None
    per_request_managers: list["BatchedCacheManager"] = field(default_factory=list)


class StepRunner:
    """Drives the per-batch resource lifecycle in dependency order."""

    def admit(
        self,
        per_request_info: dict[str, Any],
        pool: KVCachePool,
        kv_cache_string: str,
        tp_rank: int,
        tp_world_size: int,
        needed_labels: set[str] | None,
    ) -> None:
        """Establish residency for every stream this step reads: any state
        published by another process is retrieved into the pool before
        anything plans against it."""
        for req_id, info in per_request_info.items():
            if info.per_label_seq_info.world_size.get(
                kv_cache_string, tp_world_size
            ) != tp_world_size:
                raise RuntimeError(
                    "KV cache transfer across TP world size is currently disallowed"
                ) # TODO: figure out fanin/fanout for KV cache transfer
            for label, seq_info in info.per_label_seq_info.get(
                kv_cache_string, tp_rank
            ).items():
                if needed_labels is not None and label not in needed_labels:
                    continue
                pool.retrieve(req_id, label, seq_info)

    def plan(
        self,
        mode: str,
        request_ids: list[str],
        build_manager: Callable[[list[str]], "BatchedCacheManager"],
    ) -> StepPlan:
        """Build the step's plan surface for the chosen execution path. The
        model's plan calls run against this surface during its preprocess;
        the pools behind it were admitted before anything got here."""
        if mode == "graph":
            return StepPlan(mode=mode)
        if mode == "batched":
            return StepPlan(mode=mode, cache_manager=build_manager(request_ids))
        return StepPlan(
            mode=mode,
            per_request_managers=[build_manager([rid]) for rid in request_ids],
        )

    def publish(
        self,
        request_ids: list[str],
        per_request_info: dict[str, Any],
        pool: KVCachePool,
        kv_cache_string: str,
        tp_rank: int,
        tp_world_size: int,
    ) -> None:
        """Describe each request's durable pool state outward, after the
        step: the published descriptors are what another process retrieves
        from, and the next pass's routing reads them too."""
        for req_id in request_ids:
            info = per_request_info.get(req_id)
            if info is None:
                continue
            info.per_label_seq_info.add(
                kv_cache_string,
                tp_rank,
                tp_world_size,
                pool.publish(req_id),
            )
