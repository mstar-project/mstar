"""sequencing for resource call cycle

runner only moves `.plan` outputs by putting under `StepContext.plan_results`
and handing to next. (in fact, maybe `plan` should do this and runner only moves
`StepContext`?)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from mstar.engine.resources.base import CGSlotSpec, PublishedInfo, Resource
from mstar.engine.resources.spec import ResourceReqConfig
from mstar.engine.resources.step import AdmitOutcome, SubmoduleStep


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

    def __init__(self, resources: Mapping[str, Resource]):
        self._resources: dict[str, Resource] = dict(resources)
        self._order = topo_sort(self._resources) # only toposort once to minimize cpu time on python

    @property
    def order(self) -> tuple[str, ...]:
        return self._order

    @property
    def resources(self) -> Mapping[str, Resource]:
        return self._resources

    def _keys_for(self, step: SubmoduleStep) -> list[str]:
        unknown = set(step.keys()) - set(self._resources)
        if unknown:
            raise KeyError(
                f"step declares resource key(s) {sorted(unknown)} that this "
                f"node does not have; available: {sorted(self._resources)}"
            )
        return [key for key in self._order if key in step]



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

        gives `ready=False` when still has inflights; shortcircuit on failure"""
        ready = True
        for key in self._order:
            outcome = self._resources[key].admit_retrieve(
                rid, node_name, graph_walk,
                None if published is None else published.get(key),
            )
            if not outcome.ok:
                return outcome
            ready = ready and outcome.ready
        return AdmitOutcome(ok=True, ready=ready)



    def admit(self, step: SubmoduleStep) -> AdmitOutcome:
        """reserve capacity for step"""
        if __debug__:
            step.validate()
        ready = True
        for key in self._keys_for(step):
            outcome = self._resources[key].admit(step.get(key), step.ctx)
            if not outcome.ok:
                return outcome
            ready = ready and outcome.ready
        return AdmitOutcome(ok=True, ready=ready)

    def plan(self, step: SubmoduleStep) -> dict[str, Any]:
        """plan in dependency order

        place plan in `step.ctx.plan_results` before next plan runs
        again could possibly move that into `plan` itself"""
        results = step.ctx.plan_results
        results.clear()
        for key in self._keys_for(step):
            results[key] = self._resources[key].plan(step.get(key), step.ctx)
        return results

    def commit(self, step: SubmoduleStep) -> None:
        """record step consumption"""
        for key in self._keys_for(step):
            self._resources[key].commit(step.get(key), step.ctx)

    def publish(
        self, request_ids: list[str],
    ) -> dict[str, dict[str, PublishedInfo]]:
        """durable state outward publish

        non-publish resources noop; pairs with `admit_retrieve`"""
        out: dict[str, dict[str, PublishedInfo]] = {}
        for rid in request_ids:
            per_key: dict[str, PublishedInfo] = {}
            for key in self._order:
                info = self._resources[key].publish(rid)
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
