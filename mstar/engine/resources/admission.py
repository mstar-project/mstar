"""Admission across the resources of one runner, see `Resource.ask`."""

from __future__ import annotations

import bisect
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
    whose readiness check waits on it, with what the request had published for that node then."""
    asked: dict[str, tuple[str, str]] = field(default_factory=dict)
    nodes: set[str] = field(default_factory=set)
    published: dict[str, Mapping[str, PublishedInfo] | None] = field(default_factory=dict)


class Admission:
    """The queues of the resources that ask, and the nodes each request was admitted on.

    A waiting request is admitted once it is first in every queue it is in and each resource it
    asked reserves for it, all or nothing.
    """

    def __init__(self, resources: Mapping[str, Resource], ask_order: Collection[str]) -> None:
        self._resources = resources
        # `_arrival` numbers each request in the order it first asked.
        # `_waiting` holds what each waiting request asked.
        # `_queues` holds, per resource, the requests waiting for it in arrival order.
        # `_short` maps each resource whose first waiting request didn't fit to the mark it didn't fit
        # against, until that request leaves the front of the queue.
        # `_untried` is set when a queue has a new first request that `admit_waiting` hasn't tried.
        # `_admitted` holds the nodes each request was admitted on.
        # `_started` holds, for a request admitted during one node's check, the other nodes waiting on
        # it, which `admit_waiting` reports so their parked checks resume.
        self._arrivals = itertools.count()
        self._arrival: dict[str, int] = {}
        self._waiting: dict[str, _Waiting] = {}
        self._queues: dict[str, list[str]] = {key: [] for key in ask_order}
        self._short: dict[str, object] = {}
        self._untried = False
        self._admitted: dict[str, set[str]] = {}
        self._started: dict[str, set[str]] = {}

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
            self._admitted.setdefault(rid, set()).add(node_name)
            return None
        entry.nodes.add(node_name)
        entry.published[node_name] = published
        tried = all(self._queues[key][0] == rid and key not in self._short for key in entry.asked)
        if tried and (nodes := self._reserve(rid, entry.published.get)):
            if nodes - {node_name}:
                self._started.setdefault(rid, set()).update(nodes - {node_name})
            return None
        return FULL_ADMIT_QUEUED

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

    def remove_request(self, rid: str) -> None:
        entry = self._waiting.pop(rid, None)
        if entry is not None:
            self._leave(rid, entry.asked)
        self._admitted.pop(rid, None)
        self._started.pop(rid, None)
        self._arrival.pop(rid, None)

    def _park_request(self, rid: str, key: str, node_name: str, graph_walk: str) -> _Waiting:
        """Put ``rid`` in ``key``'s queue at its arrival, and return what it waits for."""
        entry = self._waiting.get(rid)
        if entry is None:
            entry = self._waiting[rid] = _Waiting()
            self._arrival.setdefault(rid, next(self._arrivals))
        entry.asked[key] = (node_name, graph_walk)
        queue = self._queues[key]
        bisect.insort(queue, rid, key=self._arrival.__getitem__)
        if queue[0] == rid:
            # ahead of the request the mark measured, so `admit` tries it unless another queue holds it back
            self._short.pop(key, None)
        return entry

    def _reserve(
        self, rid: str, published_of: Callable[[str], Mapping[str, PublishedInfo] | None],
    ) -> set[str] | None:
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
        self._admitted.setdefault(rid, set()).update(entry.nodes)
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
