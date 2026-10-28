"""sequencing for resource call cycle

runner only moves `.plan` outputs by putting under `StepContext.plan_results`
and handing to next. (in fact, maybe `plan` should do this and runner only moves
`StepContext`?)
"""

from __future__ import annotations

import bisect
import itertools
import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from mstar.engine.resources.base import CachedPrefix, CGSlotSpec, PublishedInfo, Resource
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import (
    FULL_ADMIT_NOT_READY,
    FULL_ADMIT_OK,
    FULL_ADMIT_QUEUED,
    FullAdmitOutcome,
    SubmoduleStep,
)
from mstar.utils.profiler import phase_end, phase_start, range_pop, range_push

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Waiting:
    """A waiting request: the node and walk that asked each resource it waits for, and every node
    whose readiness check waits on it, with that check's walk and what the request had published for
    that node then."""
    asked: dict[str, tuple[str, str]] = field(default_factory=dict)
    nodes: dict[str, str] = field(default_factory=dict)
    published: dict[str, Mapping[str, PublishedInfo] | None] = field(default_factory=dict)


def topo_sort(resources: Mapping[str, Resource | NodeResourceSpec]) -> tuple[str, ...]:
    """Keys in dependency order; built resources at load, their specs when the
    conductor validates a deployment."""
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
        self._ask_order = [key for key in self._order if type(self._resources[key]).ask is not Resource.ask]
        self._node_ask_order = self._per_node(node_resources, self._ask_order)
        # Admission state, see `Resource.ask`.
        # `_arrival` orders the requests: by the conductor's arrival number, then, for a request without
        # one, in the order it first asked.
        # `_waiting` holds what each waiting request asked.
        # `_queues` holds, per resource, the requests waiting for it in arrival order.
        # `_short` maps each resource whose first waiting request didn't fit to the mark it didn't fit
        # against, until that request leaves the front of the queue.
        # `_untried` is set when a queue has a new first request that `admit_waiting` hasn't tried.
        # `_admitted` holds the nodes each request was admitted on, with the walk each was admitted for.
        # `_started` holds, for a request admitted during one node's check, the other nodes waiting on
        # it, which `admit_waiting` reports so their parked checks resume.
        self._arrivals = itertools.count()
        self._arrival: dict[str, tuple[bool, int]] = {}
        self._waiting: dict[str, _Waiting] = {}
        self._queues: dict[str, list[str]] = {key: [] for key in self._ask_order}
        self._short: dict[str, object] = {}
        self._untried = False
        self._admitted: dict[str, dict[str, str]] = {}
        self._started: dict[str, set[str]] = {}
        self._node_publish_order = self._per_node(node_resources, self._publish_order)
        self._trail_order = [
            key for key in self._order if self._resources[key].page_trail() is not None
        ]
        self._node_trail_order = self._per_node(node_resources, self._trail_order)
        # capture-time buffer allocation, likewise scoped: a node's runner has
        # no business sizing a resource it never plans against
        self._node_order = self._per_node(node_resources, list(self._order))
        # the step whose pre-plan is staged across the pre-planning resources,
        # as `_step_key` describes it; None when nothing is staged
        self._staged: tuple | None = None
        # (step key, ready) of the last pre_admit whose stage is still up
        self._pre_admitted: tuple | None = None

    def resolve_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, inputs,
    ) -> CachedPrefix | None:
        """The longest prefix of this walk every resource on ``node_name`` can serve.

        The smallest answer wins, because a step is one span: a resource that
        holds less than another has to be caught up by the tokens that run.
        A resource with no opinion is not an answer of zero.
        """
        agreed = None
        for key in self._sweep(self._node_order, self._order, node_name):
            answer = self._resources[key].resolve_cached_prefix(
                rid, node_name, graph_walk, inputs,
            )
            if answer is not None and (agreed is None or answer.tokens < agreed.tokens):
                agreed = answer
        return agreed

    def apply_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str,
        inputs, prefix: CachedPrefix | None,
    ) -> None:
        """Give every resource the agreed prefix, with the untrimmed inputs."""
        for key in self._sweep(self._node_order, self._order, node_name):
            self._resources[key].apply_cached_prefix(
                rid, node_name, graph_walk, inputs, prefix,
            )

    def lease_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str,
    ) -> dict[str, list[int]]:
        leases = {}
        for key in self._sweep(self._node_order, self._order, node_name):
            pages = self._resources[key].lease_cached_prefix(rid, node_name, graph_walk)
            if pages is not None:
                leases[key] = pages
        return leases

    def adopt_cached_prefix(
        self, rid: str, node_name: str, graph_walk: str, leases: dict[str, list[int]], caught_up: bool,
    ) -> bool:
        ready = True
        for key, pages in leases.items():
            ready &= self._resources[key].adopt_cached_prefix(rid, node_name, graph_walk, pages, caught_up)
        return ready

    def release_cached_prefix(self, rid: str, node_name: str, graph_walk: str) -> None:
        for key in self._sweep(self._node_order, self._order, node_name):
            self._resources[key].release_cached_prefix(rid, node_name, graph_walk)

    def page_trails(self, node_name: str) -> dict[str, tuple[int, int]]:
        return {
            key: self._resources[key].page_trail()
            for key in self._sweep(self._node_trail_order, self._trail_order, node_name)
        }

    def prefix_chain_keys(self, node_name: str | None) -> list[str]:
        """The node's resources with a prefix cache open right now. Empty in
        the common case, which lets the caller skip a per-request sweep that
        was six no-op calls per request per step."""
        keys: list[str] = []
        for key in self._sweep(self._node_order, self._order, node_name):
            resource = self._resources[key]
            if resource.keys_prefix_chains:
                keys.append(key)
                continue
            # a resource that overrides the hook without saying when it keys
            # is swept as before the skip existed
            cls = type(resource)
            if (
                cls.extend_prefix_chain is not Resource.extend_prefix_chain
                and cls.keys_prefix_chains is Resource.keys_prefix_chains
            ):
                keys.append(key)
        return keys

    def extend_prefix_chains(
        self, rid: str, node_name: str, graph_walk: str, outputs,
        keys: list[str] | None = None,
    ) -> None:
        """Offer this step's sampled tokens to the node's own resources."""
        if keys is None:
            keys = self._sweep(self._node_order, self._order, node_name)
        for key in keys:
            self._resources[key].extend_prefix_chain(
                rid, node_name, graph_walk, outputs,
            )

    def extend_prefix_chains_batch(
        self, request_ids: list[str], node_name: str,
        walk_of: Callable[[str], str] | str, host_rows, outputs, keys: list[str],
    ) -> None:
        """``extend_prefix_chains`` for a step: a resource with a batch form
        takes the host rows; the others are offered each request's host copy.

        ``walk_of`` rather than one walk for the step, so a combined walk comes
        through here too instead of falling back to a call per row; a step of
        one walk passes that walk itself.
        """
        for key in keys:
            resource = self._resources[key]
            batched = getattr(resource, "extend_prefix_chains_batch", None)
            if batched is not None:
                batched(request_ids, node_name, walk_of, host_rows)
                continue
            for rid in request_ids:
                per_rid = outputs.get(rid)
                if isinstance(per_rid, dict):
                    resource.extend_prefix_chain(
                        rid, node_name,
                        walk_of if isinstance(walk_of, str) else walk_of(rid), per_rid,
                    )

    def complete_cached_walk(self, rid: str, node_name: str, graph_walk: str) -> None:
        """Tell the node's resources that the cache served this walk whole."""
        for key in self._sweep(self._node_order, self._order, node_name):
            self._resources[key].complete_cached_walk(rid, node_name, graph_walk)

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
        """open state on all resources; `overrides` are per-resource. All or
        nothing: a resource that raises leaves the request on none of them."""
        arrival = next((
            config.arrival for config in (overrides or {}).values()
            if getattr(config, "arrival", None) is not None
        ), None)
        if arrival is not None:
            self._arrival[rid] = (False, arrival)
        for i, key in enumerate(self._order):
            try:
                self._resources[key].ingest_request(
                    rid, None if overrides is None else overrides.get(key)
                )
            except Exception:
                # the raising resource may have taken part of it too
                for done in self._order[:i + 1]:
                    try:
                        self._resources[done].remove_request(rid)
                    except Exception:
                        logger.exception(
                            "rolling back request %s on resource %s failed", rid, done,
                        )
                raise

    def ingest_capture_row(self, rid: str) -> None:
        """open a capture padding row on all resources"""
        for key in self._order:
            self._resources[key].ingest_capture_row(rid)

    def remove_request(self, rid: str) -> None:
        entry = self._waiting.pop(rid, None)
        if entry is not None:
            self._leave(rid, entry.asked)
        self._admitted.pop(rid, None)
        self._started.pop(rid, None)
        self._arrival.pop(rid, None)
        for key in self._order:
            self._resources[key].remove_request(rid)

    def admit_retrieve(
        self, rid: str, node_name: str, graph_walk: str,
        published: Mapping[str, PublishedInfo] | None = None,
    ) -> FullAdmitOutcome:
        """bring published state in, once `node_name`'s resources have admitted the request

        gives `ready=False` when still has inflights; shortcircuit on failure

        Swept over `node_name`'s own resources. Asking a node about resources
        it does not own is not just wasted work: `KVManager.get_labels`
        answers for an unrecognised node with the default `["main"]`, so the
        node ends up gated on — and able to allocate against — another node's
        cache.
        """
        waits = self.admit_request(rid, node_name, graph_walk, published)
        if waits is not None:
            return waits
        ready = True
        for key in self._sweep(
            self._node_retrieve_order, self._retrieve_order, node_name
        ):
            outcome = self._resources[key].admit_retrieve(
                rid, node_name, graph_walk,
                None if published is None else published.get(key),
            )
            if not outcome.ok:
                logger.warning(
                    "Admit + retrieve for resource %s failed with error: %s",
                    key, outcome.reason.message
                )
                return FullAdmitOutcome(outcome, key)
            ready = ready and outcome.ready
        return FULL_ADMIT_OK if ready else FULL_ADMIT_NOT_READY

    def admit_request(
        self, rid: str, node_name: str, graph_walk: str,
        published: Mapping[str, PublishedInfo] | None = None,
    ) -> FullAdmitOutcome | None:
        """None once ``rid`` is admitted on ``node_name``, else `FULL_ADMIT_QUEUED` while it waits, or why
        it can never be admitted."""
        if node_name in self._admitted.get(rid, ()):
            return None
        return self._admit(rid, node_name, graph_walk, published)

    def readmit(self, rid: str, node_name: str) -> None:
        """Ask ``node_name``'s resources again for ``rid``, since a drop or a move to the host may have given
        back what they admitted it with. A resource that did asks now, so ``rid`` waits at its arrival and no
        later request takes the room first."""
        walk = self._admitted.get(rid, {}).pop(node_name, None)
        if walk is not None:
            self._admit(rid, node_name, walk, None)

    def _admit(
        self, rid: str, node_name: str, graph_walk: str, published: Mapping[str, PublishedInfo] | None,
    ) -> FullAdmitOutcome | None:
        entry = self._waiting.get(rid)
        # a request asked again while it waits retries a resource that has grown since it didn't fit
        waited = entry is not None
        if waited:
            for key in entry.asked:
                mark = self._short.get(key)
                if mark is not None and self._queues[key][0] == rid and self._resources[key].grew_since(mark):
                    del self._short[key]
        waits = False
        for key in self._sweep(self._node_ask_order, self._ask_order, node_name):
            if entry is not None and key in entry.asked:
                waits = True
                continue
            asked = self._resources[key].ask(rid, node_name, graph_walk)
            if asked is None:
                continue
            if not asked.ok:
                logger.warning("Admission for resource %s failed with error: %s", key, asked.reason.message)
                if entry is not None:
                    # a request a resource can't serve waits in no queue
                    del self._waiting[rid]
                    self._leave(rid, entry.asked)
                return FullAdmitOutcome(asked, key)
            if entry is None:
                entry = self._waiting[rid] = _Waiting()
                self._arrival.setdefault(rid, (True, next(self._arrivals)))
            entry.asked[key] = (node_name, graph_walk)
            queue = self._queues[key]
            bisect.insort(queue, rid, key=self._arrival.__getitem__)
            if queue[0] == rid:
                # an earlier arrival, such as a dropped request, goes ahead of the request the mark measured
                self._short.pop(key, None)
            waits = True
        if not waits:
            self._admitted.setdefault(rid, {})[node_name] = graph_walk
            return None
        entry.nodes[node_name] = graph_walk
        entry.published[node_name] = published
        tried = all(self._queues[key][0] == rid and key not in self._short for key in entry.asked)
        if tried and (nodes := self._reserve(rid, entry.published.get)):
            # The other nodes waiting on it resume, and this one too when its own check waited before: the
            # worker asks for a dropped request outside the scheduler's checks.
            others = set(nodes) if waited else nodes.keys() - {node_name}
            if others:
                self._started.setdefault(rid, set()).update(others)
            return None
        return FULL_ADMIT_QUEUED

    def _reserve(
        self, rid: str, published_of: Callable[[str], Mapping[str, PublishedInfo] | None],
    ) -> dict[str, str] | None:
        """Take everything waiting ``rid`` asked for and return the nodes waiting on it, or take
        nothing and return None. ``published_of`` gives what ``rid`` published for a node's
        resources."""
        entry = self._waiting[rid]
        taken: list[str] = []
        for key, (node_name, graph_walk) in entry.asked.items():
            published = published_of(node_name)
            mark = self._resources[key].reserve(
                rid, node_name, graph_walk, None if published is None else published.get(key),
            )
            if mark is not None:
                for done in taken:
                    self._resources[done].unreserve(rid)
                self._short[key] = mark
                return None
            taken.append(key)
        del self._waiting[rid]
        self._leave(rid, entry.asked)
        self._admitted.setdefault(rid, {}).update(entry.nodes)
        return entry.nodes

    def _leave(self, rid: str, keys: Collection[str]) -> None:
        """Take ``rid`` out of the queues of ``keys``. A queue it was first in drops its mark,
        which measured ``rid``, and its next request is left for `admit_waiting` to try."""
        for key in keys:
            queue = self._queues[key]
            if queue[0] == rid:
                self._short.pop(key, None)
                if len(queue) > 1:
                    self._untried = True
            queue.remove(rid)

    def admit_waiting(
        self, published_of: Callable[[str, str], Mapping[str, PublishedInfo] | None],
    ) -> list[tuple[str, str]]:
        """Admit each waiting request that is now first in every queue it is in and fits, and return
        the (node, rid) pairs that may start. ``published_of(rid, node)`` gives what the request
        published for that node's resources.

        Nothing is tried until a resource whose first waiting request didn't fit has grown, or a
        queue has a new first request, so a pass that changed neither costs one
        `Resource.grew_since` for each such resource.
        """
        started = []
        if self._started:
            started = [(node, rid) for rid, nodes in self._started.items() for node in nodes]
            self._started.clear()
        if not self._waiting:
            return started
        grown = [key for key, mark in self._short.items() if self._resources[key].grew_since(mark)]
        if not grown and not self._untried:
            return started
        for key in grown:
            del self._short[key]
        admitted = True
        while admitted:
            admitted = False
            for queue in self._queues.values():
                if not queue:
                    continue
                rid = queue[0]
                entry = self._waiting[rid]
                if any(self._queues[key][0] != rid or key in self._short for key in entry.asked):
                    continue
                nodes = self._reserve(rid, lambda node, rid=rid: published_of(rid, node))
                if nodes:
                    started.extend((node, rid) for node in nodes)
                    admitted = True
        # the last sweep admitted nothing, so it tried every queue's first request
        self._untried = False
        return started

    def preempt(
        self, rid: str, key: str, pages_short: int, droppable: Callable[[str], bool],
    ) -> list[str] | None:
        """Choose the requests to move off ``key``'s pool so that ``rid`` gets the ``pages_short`` pages its grant
        ran short of, and have the pool earmark the pages they free for ``rid``.

        The victims are the newest requests that arrived after ``rid`` and aren't already victims, taken until
        their moves cover the shortage. A victim the host can take whole moves there. Any other victim is dropped.
        When the newer requests that aren't victims free less than the shortage, return ``[rid]`` alone with no
        earmark if moving ``rid`` frees pages and no newer victim is still to move. Otherwise return an empty list,
        and ``rid`` waits until the room grows. It also returns an empty list when pages released since the grant
        ran short cover the shortage, and while ``rid`` or a victim chosen for it hasn't moved yet. None when the
        pool doesn't preempt or ``rid`` has no arrival number. ``droppable`` returns whether a request can be
        dropped once the steps it has launched land.
        """
        pool, arrival = self._resources[key], self._arrival.get(rid)
        if not pool.preempts or arrival is None or arrival[0]:
            return None
        with pool.lock:
            pages_short = pool.still_short(rid, pages_short)
            if pages_short <= 0:
                return []
            newer = sorted(
                (other for other, place in self._arrival.items() if place > arrival and not place[0]
                 and pool.holds(other)),
                key=self._arrival.__getitem__, reverse=True,
            )
            # a victim not yet moved frees its pages for the request it was chosen for
            moving = any(pool.is_victim(other) for other in newer)
            newer = [other for other in newer if not pool.is_victim(other)]
            on_host = left = pool.host_room()
            hosted: set[str] = set()
            victims: list[str] = []
            for other in newer:
                pages = pool.host_pages(other)
                if pages is not None and pages <= left:
                    left -= pages
                    hosted.add(other)
                elif not droppable(other):
                    continue
                victims.append(other)
                if pool.frees(victims) >= pages_short:
                    break
            else:
                pages = pool.host_pages(rid)
                to_host = pages is not None and pages <= on_host
                if not moving and (to_host or droppable(rid)) and pool.frees([rid]) > 0:
                    pool.preempt_itself(rid, to_host)
                    return [rid]
                pool.wait_for_room(rid)
                return []
            # a victim whose move frees nothing the rest don't is spared
            freed = pool.frees(victims)
            for victim in list(victims):
                rest = [other for other in victims if other != victim]
                if pool.frees(rest) == freed:
                    victims = rest
            pool.earmark(rid, pages_short, victims, hosted)
            return victims

    # ── Pre-plan bookkeeping ─────────────────────────────────────────────
    #
    # A pre-plan is staged across every pre-planning resource for one step,
    # and each resource promotes its share when that step's full `plan`
    # arrives. Only the runner sees the whole step, so it is the one that
    # decides whether the step reaching `admit`/`plan` is the staged one. Any
    # other step — a new request's prefill dispatched while a decode step sits
    # pre-planned, or the same rows re-declared without their lease — drops
    # the stage on every resource first. Otherwise a resource that promotes
    # blindly (the attention wrappers, positions) would run against the KV
    # layout its dependency just discarded and planned afresh.

    def _step_key(self, step: SubmoduleStep):
        """What identifies the step a pre-plan was staged for: its walk, the
        rows it runs (padding included), the replay slot it was leased on,
        its capture key, and every resource's segments."""
        ctx = step.ctx
        lease = ctx.slot_lease
        return (
            ctx.graph_walk,
            tuple(ctx.padded_request_ids),
            None if lease is None else lease.slot,
            step.cg_key_info,
            tuple(
                (key, tuple(step.get(key).segments or ()))
                for key in self._keys_for(step)
            ),
        )

    def _drop_stale_preplan(self, step: SubmoduleStep) -> tuple | None:
        """Returns the step's key when the stage is for this very step."""
        if self._staged is None or step.ctx.is_preplan:
            return None
        key = self._step_key(step)
        if self._staged != key:
            # Not the normal path: the worker meant to run the staged step
            # next. Name both steps so the source can be traced.
            logger.warning(
                "dropping the staged pre-plan (walk %s, %d rows) for a different "
                "step (walk %s, %d rows) that reached the GPU thread first",
                self._staged[0], len(self._staged[1]), key[0], len(key[1]),
            )
            self.clear_preplan()
            return None
        return key

    @property
    def staged(self) -> bool:
        """A pre-plan is staged and not yet promoted."""
        return self._staged is not None

    def clear_preplan(self) -> None:
        """Drop the staged pre-plan on every resource, and the record of it."""
        self._staged = None
        self._pre_admitted = None
        for key in self._order:
            self._resources[key].clear_preplan()

    def admit(self, step: SubmoduleStep) -> FullAdmitOutcome:
        """Reserve capacity for step. A refused admit gives back everything
        the step reserved.
        """
        staged_key = self._drop_stale_preplan(step)
        # A step whose ``pre_admit`` ran a step ahead, and whose stage is still
        # up for it, skips the resources that sweep covered: each would only
        # find its own reservation and no-op (0.1 ms of Python per step at
        # 128 rows, on the gpu thread). A refusal below still unwinds them.
        covered: list[str] = []
        ready = True
        pre = self._pre_admitted
        if pre is not None and staged_key is not None and pre[0] == staged_key:
            covered = self._preplan_keys_for(step)
            ready = pre[1]
        admitted: list[str] = []
        for key in self._keys_for(step):
            if key in covered:
                continue
            if self._nvtx:
                range_push(f"res.admit.{key}")
            t0 = phase_start()
            try:
                outcome = self._resources[key].admit(step.get(key), step.ctx)
            finally:
                if t0 is not None:
                    phase_end(f"res.admit.{key}", t0)
                if self._nvtx:
                    range_pop()
            if not outcome.ok:
                logger.warning(
                    "Admit for resource %s failed with error: %s",
                    key, outcome.reason.message
                )
                self._unwind(step, [*covered, *admitted, key])
                return FullAdmitOutcome(outcome, key)
            admitted.append(key)
            ready = ready and outcome.ready
        return FULL_ADMIT_OK if ready else FULL_ADMIT_NOT_READY

    def rollback_admit(self, step: SubmoduleStep) -> None:
        """Give back everything an admitted step reserved, for a step that
        will not run: the unwind a refused ``admit`` does, over every
        resource the step declares."""
        self._unwind(step, self._keys_for(step))

    def _unwind(self, step: SubmoduleStep, keys: list[str]) -> None:
        # reverse dependency order, off a list so every rank unwinds alike
        for key in reversed(keys):
            self._resources[key].rollback_admit(step.get(key), step.ctx)

    def plan(self, step: SubmoduleStep) -> dict[str, Any]:
        """plan in dependency order

        place plan in `step.ctx.plan_results` before next plan runs
        again could possibly move that into `plan` itself"""
        self._drop_stale_preplan(step)
        if not step.ctx.is_preplan:
            # the resources promote their staged share below (or plan afresh
            # if nothing was staged): either way nothing stays staged
            self._staged = None
            self._pre_admitted = None
        results = step.ctx.plan_results
        results.clear()
        for key in self._keys_for(step):
            if self._nvtx:
                range_push(f"res.plan.{key}")
            t0 = phase_start()
            try:
                results[key] = self._resources[key].plan(step.get(key), step.ctx)
            finally:
                if t0 is not None:
                    phase_end(f"res.plan.{key}", t0)
                if self._nvtx:
                    range_pop()
        return results

    def pre_admit(self, step: SubmoduleStep) -> FullAdmitOutcome:
        """admit over the pre-planning subset, a step ahead

        the later full `admit` covers the rest; these resources see their own
        state as already reserved and no-op

        Unwound on refusal exactly as ``admit`` is.
        """
        ready = True
        admitted: list[str] = []
        for key in self._preplan_keys_for(step):
            if self._nvtx:
                range_push(f"res.pre_admit.{key}")
            t0 = phase_start()
            try:
                outcome = self._resources[key].admit(step.get(key), step.ctx)
            finally:
                if t0 is not None:
                    phase_end(f"res.pre_admit.{key}", t0)
                if self._nvtx:
                    range_pop()
            if not outcome.ok:
                logger.warning(
                    "Admit for pre-planning resource %s failed with error: %s",
                    key, outcome.reason.message
                )
                self._unwind(step, [*admitted, key])
                self._pre_admitted = None
                return FullAdmitOutcome(outcome, key)
            admitted.append(key)
            ready = ready and outcome.ready
        self._pre_admitted = (self._step_key(step), ready)
        return FULL_ADMIT_OK if ready else FULL_ADMIT_NOT_READY

    def pre_plan(self, step: SubmoduleStep) -> dict[str, Any]:
        """plan the pre-planning subset, a step ahead

        `ctx.is_preplan` sends each one's output to its pending slot rather
        than the live one; the later full `plan` promotes it. Results are left
        in `ctx.plan_results` for the dependents in this same subset."""
        results = step.ctx.plan_results
        for key in self._preplan_keys_for(step):
            if self._nvtx:
                range_push(f"res.pre_plan.{key}")
            t0 = phase_start()
            try:
                results[key] = self._resources[key].plan(step.get(key), step.ctx)
            finally:
                if t0 is not None:
                    phase_end(f"res.pre_plan.{key}", t0)
                if self._nvtx:
                    range_pop()
        self._staged = self._step_key(step)
        return results

    def commit(self, step: SubmoduleStep) -> None:
        """record step consumption"""
        for key in self._keys_for(step):
            if self._nvtx:
                range_push(f"res.commit.{key}")
            t0 = phase_start()
            try:
                self._resources[key].commit(step.get(key), step.ctx)
            finally:
                if t0 is not None:
                    phase_end(f"res.commit.{key}", t0)
                if self._nvtx:
                    range_pop()

    def publish(
        self,
        request_ids: list[str],
        node_name: str | None = None,
        graph_walk: str | None = None,
        request_walks: Mapping[int, str] | None = None,
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
                info = resource.publish_for_step(
                    rid, node_name=node_name,
                    graph_walk=request_walks.get(rid, graph_walk) if request_walks else graph_walk,
                )
                if info is not None:
                    per_key[key] = info
            out[rid] = per_key
        return out

    def publish_snapshot(
        self,
        request_ids: list[str],
        node_name: str | None = None,
        graph_walk: str | None = None,
        request_walks: Mapping[int, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """``publish``, deferred: per rid the snapshot each publisher took.

        Finished by ``publish_from_snapshots`` for the rids that need it. A rid
        with nothing to publish has no entry. Under a combined walk each rid
        is snapshotted under its own walk (``request_walks``), as ``publish``
        does: a publisher may key what it publishes on the walk."""
        order = self._sweep(self._node_publish_order, self._publish_order, node_name)
        if not order:
            return {}
        if request_walks:
            groups: dict[str, list[str]] = {}
            for rid in request_ids:
                groups.setdefault(request_walks.get(rid, graph_walk), []).append(rid)
        else:
            groups = {graph_walk: request_ids}
        out: dict[str, dict[str, Any]] = {}
        for walk, rids in groups.items():
            # one call per publisher for the walk's rows, then one dict per
            # request that has anything to publish
            columns = [
                (key, self._resources[key].publish_snapshot_batch(
                    rids, node_name=node_name, graph_walk=walk,
                ))
                for key in order
            ]
            for i, rid in enumerate(rids):
                per_key: dict[str, Any] = {}
                for key, snaps in columns:
                    snap = snaps[i]
                    if snap is not None:
                        per_key[key] = snap
                if per_key:
                    out[rid] = per_key
        return out

    def publish_from_snapshots(
        self,
        snapshots: dict[str, dict[str, Any]],
        request_ids: list[str],
        node_name: str | None = None,
        graph_walk: str | None = None,
        request_walks: Mapping[int, str] | None = None,
    ) -> dict[str, dict[str, PublishedInfo]]:
        """The published info of ``request_ids`` from their snapshots, each
        under its own walk (see ``publish_snapshot``)."""
        out: dict[str, dict[str, PublishedInfo]] = {}
        for rid in request_ids:
            snaps = snapshots.get(rid)
            if not snaps:
                continue
            walk = request_walks.get(rid, graph_walk) if request_walks else graph_walk
            per_key: dict[str, PublishedInfo] = {}
            for key, snap in snaps.items():
                info = self._resources[key].publish_from_snapshot(
                    rid, snap, node_name=node_name, graph_walk=walk,
                )
                if info is not None:
                    per_key[key] = info
            out[rid] = per_key
        return out

    def publish_after_stop(
        self,
        request_ids: list[str],
        node_name: str | None = None,
        graph_walk: str | None = None,
        request_walks: Mapping[int, str] | None = None,
    ) -> dict[str, dict[str, PublishedInfo]]:
        """Publish resources configured for the end of a dynamic loop."""
        order = self._sweep(self._node_publish_order, self._publish_order, node_name)
        if not order:
            return {rid: {} for rid in request_ids}
        publishers = [(key, self._resources[key]) for key in order]
        out: dict[str, dict[str, PublishedInfo]] = {}
        for rid in request_ids:
            per_key: dict[str, PublishedInfo] = {}
            for key, resource in publishers:
                info = resource.publish_after_stop(
                    rid, node_name=node_name,
                    graph_walk=request_walks.get(rid, graph_walk) if request_walks else graph_walk,
                )
                if info is not None:
                    per_key[key] = info
            out[rid] = per_key
        return out

    def build_accelerator_graph_buffers(
        self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int,
        node_name: str | None = None,
    ) -> None:
        """resources preallocate the static buffers captured replays will read

        Swept over ``node_name``'s own resources: another node's slot count and
        batch size say nothing about a resource this node's graphs never touch.
        """
        for key in self._sweep(self._node_order, self._order, node_name):
            self._resources[key].build_accelerator_graph_buffers(
                slots, max_bs, max_seq_len
            )
