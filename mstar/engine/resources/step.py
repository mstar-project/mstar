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

from mstar.engine.resources.base import Segment
from mstar.engine.resources.declare import StepDeclaration
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

    def drive(
        self,
        declaration: StepDeclaration,
        cache_manager: "BatchedCacheManager",
    ) -> None:
        """Execute a declared step's admission and planning: forks first,
        then each plan in declaration order. Every plan goes through the
        step surface's own plan calls, so a declared step admits, plans,
        and short-circuits pre-planned labels exactly as a facade-driven
        model would."""
        for from_label, to_label in declaration.pre_forks:
            cache_manager.snapshot_all(from_label, to_label)
        for plan in declaration.plans:
            if plan.combined:
                seq_lens = {
                    label: list(plan.spans[label]) for label in plan.labels
                }
                cache_manager.plan_attention_batched_cfg(
                    labels=list(plan.labels),
                    seq_lens=seq_lens,
                    is_causal=plan.is_causal,
                    write_store=plan.write_store,
                    combined_label=plan.combined_key,
                    dense_gen=plan.dense_gen,
                )
                if plan.rope:
                    cache_manager.plan_rope_batched_cfg(
                        labels=list(plan.labels),
                        seq_lens=seq_lens,
                        per_label_pos_ids=plan.rope_pos_ids,
                        combined_label=plan.combined_key,
                    )
            else:
                label = plan.labels[0]
                spans = list(plan.spans[label])
                cache_manager.plan_attention(
                    seq_lens=spans,
                    is_causal=plan.is_causal,
                    write_store=plan.write_store,
                    label=label,
                    dense_gen=plan.dense_gen,
                )
                if plan.rope:
                    cache_manager.plan_rope(
                        seq_lens=spans,
                        pos_ids=plan.rope_pos_ids,
                        label=label,
                    )

    def commit(
        self,
        declaration: StepDeclaration,
        cache_manager: "BatchedCacheManager",
    ) -> None:
        """Commit a declared step once its forward is launched: advance
        each committing plan's streams by their planned spans (and any
        declared position advances) straight on the pool, then apply the
        declared post-forward forks."""
        pool = cache_manager.kv_pool
        request_ids = cache_manager.request_ids
        for plan in declaration.plans:
            if not plan.commit:
                continue
            for label in plan.labels:
                spans = plan.spans[label]
                for i, rid in enumerate(request_ids):
                    pool.commit(
                        Segment(rid, label, spans[i]),
                        pos_advance=(
                            None if plan.pos_advance is None
                            else plan.pos_advance[i]
                        ),
                    )
        for from_label, to_label in declaration.post_forks:
            cache_manager.snapshot_all(from_label, to_label)

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
