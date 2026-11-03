"""Admission across the resources of one runner, see `Resource.ask`."""

from __future__ import annotations

import bisect
import heapq
import itertools
import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

from mstar.engine.resources.base import PublishedInfo, Resource
from mstar.engine.resources.step import FULL_ADMIT_QUEUED, FullAdmitOutcome

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Waiting:
    """A waiting request: the node and walk that asked each resource it waits for, and every node
    whose readiness check waits on it, with that check's walk and what the request had published for
    that node then."""
    asked: dict[str, tuple[str, str]] = field(default_factory=dict)
    nodes: dict[str, str] = field(default_factory=dict)
    published: dict[str, Mapping[str, PublishedInfo] | None] = field(default_factory=dict)


class Admission:
    """The queues of the resources that ask, and the nodes each request was admitted on.

    A waiting request is admitted once it is first in every queue it is in and each resource it
    asked reserves for it, all or nothing. It may also pass a first request that can't start, in a
    queue whose resource overrides `Resource.reserve_past`, when it is first in every other queue it
    is in (`_pass`).
    """

    def __init__(self, resources: Mapping[str, Resource], ask_order: Collection[str]) -> None:
        self._resources = resources
        # `_arrival` orders the requests: by the conductor's arrival number, then, for a request without
        # one, in the order it first asked. `_numbered` lists the requests with a number, in that order.
        # `_waiting` holds what each waiting request asked.
        # `_queues` holds, per resource, the requests waiting for it in arrival order.
        # `_short` maps each resource whose first waiting request didn't fit to the mark it didn't fit
        # against, until that request leaves the front of the queue.
        # `_untried` is set when a queue has a new first request that `admit_waiting` hasn't tried.
        # `_admitted` holds the nodes each request was admitted on, with the walk each was admitted for.
        # `_started` holds, for a request admitted during one node's check, the other nodes waiting on
        # it, which `admit_waiting` reports so their parked checks resume.
        # `_passes` holds the resources that override `Resource.reserve_past`, whose queues a request may pass.
        # `_pass_short` maps each resource that refused a request passing since the last pass sweep to the
        # distinct marks it refused against.
        # `_unpassed` holds the requests not tried for passing because a first request ahead of them may fit
        # now, which `admit_waiting` tries once that request has been measured again.
        self._arrivals = itertools.count()
        self._arrival: dict[str, tuple[bool, int]] = {}
        self._numbered: list[str] = []
        self._waiting: dict[str, _Waiting] = {}
        self._queues: dict[str, list[str]] = {key: [] for key in ask_order}
        self._short: dict[str, object] = {}
        self._untried = False
        self._admitted: dict[str, dict[str, str]] = {}
        self._started: dict[str, set[str]] = {}
        self._passes = {key for key in ask_order if type(resources[key]).reserve_past is not Resource.reserve_past}
        self._pass_short: dict[str, list[object]] = {}
        self._unpassed: set[str] = set()

    def set_arrival(self, rid: str, arrival: int) -> None:
        """Order ``rid`` by the conductor's arrival number. A worker serving two of the request's partitions
        ingests it once for each, with the same number."""
        if rid not in self._arrival:
            self._arrival[rid] = (False, arrival)
            bisect.insort(self._numbered, rid, key=self._arrival.__getitem__)

    def newer_than(self, rid: str) -> list[str] | None:
        """The requests numbered after ``rid``, newest first, or None when the conductor gave ``rid`` no number."""
        arrival = self._arrival.get(rid)
        if arrival is None or arrival[0]:
            return None
        return self._numbered[bisect.bisect_right(self._numbered, arrival, key=self._arrival.__getitem__):][::-1]

    def admitted(self, rid: str, node_name: str) -> bool:
        return node_name in self._admitted.get(rid, ())

    def admit(
        self, rid: str, node_name: str, graph_walk: str, published: Mapping[str, PublishedInfo] | None,
        keys: Collection[str],
    ) -> FullAdmitOutcome | None:
        """None once ``rid`` is admitted on ``node_name``, else why it waits or fails. ``keys`` are the
        resources of ``node_name`` that ask."""
        entry = self._waiting.get(rid)
        waits = False
        for key in keys:
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
            entry = self._park_request(rid, key, node_name, graph_walk)
            waits = True
        if not waits:
            self._admitted.setdefault(rid, {})[node_name] = graph_walk
            return None
        entry.nodes[node_name] = graph_walk
        entry.published[node_name] = published
        tried = all(self._queues[key][0] == rid and key not in self._short for key in entry.asked)
        # a request that isn't first may pass a first request that can't start
        nodes = (
            self._reserve(rid, entry.published.get) if tried else self._pass(rid, entry.published.get, self._blocked())
        )
        if nodes:
            if others := nodes.keys() - {node_name}:
                self._started.setdefault(rid, set()).update(others)
            return None
        return FULL_ADMIT_QUEUED

    def readmit(self, rid: str, node_name: str, keys: Collection[str]) -> None:
        """Ask ``node_name``'s resources again for ``rid``, since a drop or a move to the host may have given
        back what they admitted it with. A resource that did asks now, so ``rid`` waits at its arrival and no
        later request takes the room first. ``keys`` are the resources of ``node_name`` that ask."""
        walk = self._admitted.get(rid, {}).pop(node_name, None)
        if walk is not None:
            self.admit(rid, node_name, walk, None, keys)

    def admit_waiting(
        self, published_of: Callable[[str, str], Mapping[str, PublishedInfo] | None],
    ) -> list[tuple[str, str]]:
        """Admit each waiting request that is now first in every queue it is in and fits, and return
        the (node, rid) pairs that may start. ``published_of(rid, node)`` gives what the request
        published for that node's resources.

        Nothing is tried until a resource whose first waiting request didn't fit has grown, or a
        queue has a new first request, so a pass that changed neither costs one
        `Resource.grew_since` for each such resource. After the in-order sweep, while a resource
        that lets requests pass has a first request that can't start, each later waiting request is
        tried, in arrival order, to pass it, but only once a queue has a new first request or a
        resource that refused a passing request has grown since. Otherwise only the requests in
        `_unpassed` and the first requests of the queues whose resource grew are tried.
        """
        started = []
        if self._started:
            started = [(node, rid) for rid, nodes in self._started.items() for node in nodes]
            self._started.clear()
        if not self._waiting:
            return started
        grown = [key for key, mark in self._short.items() if self._resources[key].grew_since(mark)]
        passable = any(
            self._resources[key].grew_since(mark) for key, marks in self._pass_short.items() for mark in marks
        )
        if not grown and not passable and not self._untried and not self._unpassed:
            return started
        for key in grown:
            del self._short[key]
        # a first request whose resource grew may now pass in a queue where it waits behind another
        heads = {self._queues[key][0] for key in grown}
        unpassed = self._unpassed.union(
            rid for rid in heads if any(self._queues[key][0] != rid for key in self._waiting[rid].asked)
        )
        self._unpassed = set()
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
        passable, self._untried = passable or self._untried, False
        if not passable and not unpassed:
            return started
        blocked = self._blocked()
        if passable:
            self._pass_short.clear()
            queues = [
                queue[1:] for key, queue in self._queues.items()
                if key in self._passes and queue and queue[0] in blocked
            ]
            # every queue is in arrival order, and a request waiting in two of them is tried once
            waiting = dict.fromkeys(heapq.merge(*queues, key=self._arrival.__getitem__))
        else:
            waiting = sorted(unpassed, key=self._arrival.__getitem__)
        for rid in waiting:
            if rid in self._waiting:
                nodes = self._pass(rid, lambda node, rid=rid: published_of(rid, node), blocked)
                started.extend((node, rid) for node in nodes or ())
        return started

    def remove_request(self, rid: str) -> None:
        entry = self._waiting.pop(rid, None)
        if entry is not None:
            self._leave(rid, entry.asked)
        self._admitted.pop(rid, None)
        self._started.pop(rid, None)
        arrival = self._arrival.pop(rid, None)
        if arrival is not None and not arrival[0]:
            self._numbered.remove(rid)

    def _park_request(self, rid: str, key: str, node_name: str, graph_walk: str) -> _Waiting:
        """Put ``rid`` in ``key``'s queue at its arrival, and return what it waits for."""
        entry = self._waiting.get(rid)
        if entry is None:
            entry = self._waiting[rid] = _Waiting()
            self._arrival.setdefault(rid, (True, next(self._arrivals)))
        entry.asked[key] = (node_name, graph_walk)
        queue = self._queues[key]
        bisect.insort(queue, rid, key=self._arrival.__getitem__)
        if queue[0] == rid:
            # an earlier arrival, such as a dropped request, goes ahead of the request the mark measured
            self._short.pop(key, None)
        return entry

    def _reserve(
        self, rid: str, published_of: Callable[[str], Mapping[str, PublishedInfo] | None],
        passing: Collection[str] = (),
    ) -> dict[str, str] | None:
        """Take everything waiting ``rid`` asked for and return the nodes waiting on it, or take
        nothing and return None. ``published_of`` gives what ``rid`` published for a node's
        resources, and ``passing`` names the resources where ``rid`` passes the first request."""
        entry = self._waiting[rid]
        taken: list[str] = []
        for key, (node_name, graph_walk) in entry.asked.items():
            published = published_of(node_name)
            resource = self._resources[key]
            reserve = resource.reserve_past if key in passing else resource.reserve
            mark = reserve(rid, node_name, graph_walk, None if published is None else published.get(key))
            if mark is not None:
                for done in taken:
                    self._resources[done].unreserve(rid)
                if key not in passing:
                    self._short[key] = mark
                elif mark not in (marks := self._pass_short.setdefault(key, [])):
                    marks.append(mark)
                return None
            taken.append(key)
        del self._waiting[rid]
        self._leave(rid, entry.asked)
        self._admitted.setdefault(rid, {}).update(entry.nodes)
        return entry.nodes

    def _pass(
        self, rid: str, published_of: Callable[[str], Mapping[str, PublishedInfo] | None], blocked: set[str],
    ) -> dict[str, str] | None:
        """Admit ``rid`` past the first request of each queue it waits behind, when that request is in
        ``blocked`` and the queue's resource lets requests pass, and ``rid`` is first wherever else it
        waits. Return the nodes waiting on it, or None. Add ``rid`` to ``blocked`` when it is first in a
        queue whose resource just refused it."""
        passing = []
        for key in self._waiting[rid].asked:
            first = self._queues[key][0]
            if first == rid:
                if key in self._short:
                    return None
            elif key in self._passes and first in blocked:
                passing.append(key)
            else:
                if key in self._passes and key in self._short:
                    self._unpassed.add(rid)
                return None
        if not passing:
            return None
        nodes = self._reserve(rid, published_of, passing)
        if nodes is None and any(self._queues[key][0] == rid for key in self._short):
            blocked.add(rid)
        return nodes

    def _blocked(self) -> set[str]:
        """The requests that can't start now: each is first in a queue whose resource it didn't fit, and
        that resource hasn't grown since."""
        return {self._queues[key][0] for key, mark in self._short.items() if not self._resources[key].grew_since(mark)}

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
        self._unpassed.discard(rid)
