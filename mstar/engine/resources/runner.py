"""sequencing for resource call cycle

runner only moves `.plan` outputs by putting under `StepContext.plan_results`
and handing to next. (in fact, maybe `plan` should do this and runner only moves
`StepContext`?)
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from mstar.engine.resources.base import CGSlotSpec, PublishedInfo, Resource
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import ADMIT_OK, AdmitOutcome, SubmoduleStep
from mstar.utils.profiler import range_pop, range_push


def topo_sort(resources: Mapping[str, Resource]) -> tuple[str, ...]:
    deps: dict[str, set[str]] = {
        key: set(resource.depends_on()) for key, resource in resources.items()
    }
    for key, required in deps.items():
        unknown = required - deps.keys()
        if unknown:
            raise ValueError(
                f"resource {key!r} depends on unknown key(s) {sorted(unknown)}; "
                f"available: {sorted(deps)}"
            )
        if key in required:
            raise ValueError(f"resource {key!r} depends on itself")

    order: list[str] = []
    remaining = dict(deps)
    while remaining:
        ready = sorted(
            key for key, required in remaining.items()
            if not (required & remaining.keys())
        )
        if not ready:
            raise ValueError(
                f"dependency cycle among resources {sorted(remaining)}"
            )
        order.extend(ready)
        for key in ready:
            del remaining[key]
    return tuple(order)


class StepRunner:
    """drives resources through per-step cycle"""

    def __init__(
        self, resources: Mapping[str, Resource],
        node_resources: Mapping[str, Collection[str]] | None = None,
        enable_nvtx: bool = False,
    ):
        # Per-resource NVTX. `engine.plan.promoted` only reports that the sweep
        # was slow; these say WHICH resource ate it. Gated because range_push
        # is a real CUDA call even with no profiler attached.
        self._nvtx = enable_nvtx
        self._resources: dict[str, Resource] = dict(resources)
        self._order = topo_sort(self._resources) # only toposort once to minimize cpu time on python
        self._preplan_order = [
            res for res in self._order if self._resources[res].supports_preplan
        ]
        self._check_preplan_deps()
        # keyset -> ordered keys; a step's resource set is fixed for a node, so
        # admit/plan/commit reuse this rather than re-filtering _order each call
        self._keys_cache: dict[frozenset[str], list[str]] = {}
        self._preplan_keys_cache: dict[frozenset[str], list[str]] = {}
        # publish runs per rid per step; most resources inherit the base
        # no-op, so settle who can actually publish once instead of calling
        # every resource bs times a step to be handed None
        self._publish_order = [
            key for key in self._order
            if type(self._resources[key]).publish is not Resource.publish
        ]
        # Same for admit_retrieve, which check_ready drives per ready request
        # per scheduler scan — the hottest sweep here. The base returns
        # ADMIT_OK, the identity for both the short-circuit and the `ready`
        # fold, so an inheritor cannot change the outcome.
        self._retrieve_order = [
            key for key in self._order
            if type(self._resources[key]).admit_retrieve
            is not Resource.admit_retrieve
        ]
        # Both sweeps run on behalf of one node and have no step to filter by,
        # so settle each node's share of them up front. `node_resources` must
        # name every node, including one that owns nothing — an absent node
        # falls back to the full sweep, which is the un-scoped behaviour.
        self._node_retrieve_order = self._per_node(node_resources, self._retrieve_order)
        self._node_publish_order = self._per_node(node_resources, self._publish_order)

    def _check_preplan_deps(self) -> None:
        """A pre-planning resource's dependencies must pre-plan too.

        A resource's `plan` reads its dependencies' output off
        `ctx.plan_results`, so pre-planning one whose dependency stays behind
        would read a result from the wrong (in-flight) step.
        """
        preplanned = set(self._preplan_order)
        for key in self._preplan_order:
            missing = sorted(self._resources[key].depends_on() - preplanned)
            if missing:
                raise ValueError(
                    f"resource {key!r} pre-plans but its dependencies {missing} "
                    "do not; a pre-planned step would read their plan output "
                    "from the step still in flight"
                )

    @property
    def order(self) -> tuple[str, ...]:
        return self._order

    @property
    def resources(self) -> Mapping[str, Resource]:
        return self._resources

    def _keys_for(self, step: SubmoduleStep) -> list[str]:
        keyset = frozenset(step.keys())
        cached = self._keys_cache.get(keyset)
        if cached is None:
            unknown = keyset - self._resources.keys()
            if unknown:
                raise KeyError(
                    f"step declares resource key(s) {sorted(unknown)} that this "
                    f"node does not have; available: {sorted(self._resources)}"
                )
            cached = self._keys_cache[keyset] = [k for k in self._order if k in keyset]
        return cached

    @staticmethod
    def _per_node(
        node_resources: Mapping[str, Collection[str]] | None, base: list[str],
    ) -> dict[str, list[str]] | None:
        """``base`` split into each node's own share, in ``base``'s order.

        ``None`` keeps every sweep global, which is what a runner built
        without the map (a test, say) had before.
        """
        if node_resources is None:
            return None
        return {
            node: [key for key in base if key in set(keys)]
            for node, keys in node_resources.items()
        }

    def _sweep(
        self, per_node: dict[str, list[str]] | None,
        base: list[str], node_name: str | None,
    ) -> list[str]:
        if per_node is None or node_name is None:
            return base
        return per_node.get(node_name, base)

    def _preplan_keys_for(self, step: SubmoduleStep) -> list[str]:
        keyset = frozenset(step.keys())
        cached = self._preplan_keys_cache.get(keyset)
        if cached is None:
            cached = self._preplan_keys_cache[keyset] = [
                k for k in self._preplan_order if k in keyset
            ]
        return cached



    def ingest_request(
        self, rid: str,
        overrides: Mapping[str, ResourceReqConfig] | None = None,
    ) -> None:
        """open state on all resources; `overrides` are per-resource"""
        for key in self._order:
            self._resources[key].ingest_request(
                rid, None if overrides is None else overrides.get(key)
            )

    def remove_request(self, rid: str) -> None:
        for key in self._order:
            self._resources[key].remove_request(rid)

    def admit_retrieve(
        self, rid: str, node_name: str, graph_walk: str,
        published: Mapping[str, PublishedInfo] | None = None,
    ) -> AdmitOutcome:
        """bring published state in

        gives `ready=False` when still has inflights; shortcircuit on failure

        Swept over `node_name`'s own resources. Asking a node about resources
        it does not own is not just wasted work: `KVManager.get_labels`
        answers for an unrecognised node with the default `["main"]`, so the
        node ends up gated on — and able to allocate against — another node's
        cache.
        """
        ready = True
        for key in self._sweep(
            self._node_retrieve_order, self._retrieve_order, node_name
        ):
            outcome = self._resources[key].admit_retrieve(
                rid, node_name, graph_walk,
                None if published is None else published.get(key),
            )
            if not outcome.ok:
                return outcome
            ready = ready and outcome.ready
        return ADMIT_OK if ready else AdmitOutcome(ok=True, ready=False)



    def admit(self, step: SubmoduleStep) -> AdmitOutcome:
        """reserve capacity for step"""
        ready = True
        for key in self._keys_for(step):
            if self._nvtx:
                range_push(f"res.admit.{key}")
            try:
                outcome = self._resources[key].admit(step.get(key), step.ctx)
            finally:
                if self._nvtx:
                    range_pop()
            if not outcome.ok:
                return outcome
            ready = ready and outcome.ready
        return ADMIT_OK if ready else AdmitOutcome(ok=True, ready=False)

    def plan(self, step: SubmoduleStep) -> dict[str, Any]:
        """plan in dependency order

        place plan in `step.ctx.plan_results` before next plan runs
        again could possibly move that into `plan` itself"""
        results = step.ctx.plan_results
        results.clear()
        for key in self._keys_for(step):
            if self._nvtx:
                range_push(f"res.plan.{key}")
            try:
                results[key] = self._resources[key].plan(step.get(key), step.ctx)
            finally:
                if self._nvtx:
                    range_pop()
        return results

    def pre_admit(self, step: SubmoduleStep) -> AdmitOutcome:
        """admit over the pre-planning subset, a step ahead

        the later full `admit` covers the rest; these resources see their own
        state as already reserved and no-op"""
        ready = True
        for key in self._preplan_keys_for(step):
            if self._nvtx:
                range_push(f"res.admit.{key}")
            try:
                outcome = self._resources[key].admit(step.get(key), step.ctx)
            finally:
                if self._nvtx:
                    range_pop()
            if not outcome.ok:
                return outcome
            ready = ready and outcome.ready
        return ADMIT_OK if ready else AdmitOutcome(ok=True, ready=False)

    def pre_plan(self, step: SubmoduleStep) -> dict[str, Any]:
        """plan the pre-planning subset, a step ahead

        `ctx.is_preplan` sends each one's output to its pending slot rather
        than the live one; the later full `plan` promotes it. Results are left
        in `ctx.plan_results` for the dependents in this same subset."""
        results = step.ctx.plan_results
        for key in self._preplan_keys_for(step):
            if self._nvtx:
                range_push(f"res.plan.{key}")
            try:
                results[key] = self._resources[key].plan(step.get(key), step.ctx)
            finally:
                if self._nvtx:
                    range_pop()
        return results

    def commit(self, step: SubmoduleStep) -> None:
        """record step consumption"""
        for key in self._keys_for(step):
            if self._nvtx:
                range_push(f"res.commit.{key}")
            try:
                self._resources[key].commit(step.get(key), step.ctx)
            finally:
                if self._nvtx:
                    range_pop()

    def publish(
        self, request_ids: list[str], node_name: str | None = None,
    ) -> dict[str, dict[str, PublishedInfo]]:
        """durable state outward publish

        non-publish resources noop; pairs with `admit_retrieve`

        Scoped to the node whose step just ran; no other node's state moved,
        and `merge_publish_info` keeps the entry it already published."""
        order = self._sweep(self._node_publish_order, self._publish_order, node_name)
        if not order:
            return {rid: {} for rid in request_ids}
        publishers = [(key, self._resources[key]) for key in order]
        out: dict[str, dict[str, PublishedInfo]] = {}
        for rid in request_ids:
            per_key: dict[str, PublishedInfo] = {}
            for key, resource in publishers:
                info = resource.publish(rid)
                if info is not None:
                    per_key[key] = info
            out[rid] = per_key
        return out



    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
    ) -> None:
        """resources preallocate the static buffers captured replays will read"""
        for key in self._order:
            self._resources[key].build_cuda_graph_buffers(
                slots, max_bs, max_seq_len
            )
