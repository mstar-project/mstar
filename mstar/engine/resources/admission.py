"""Admission across the resources of one runner, see `Resource.ask`."""

from __future__ import annotations

import bisect
import itertools
import logging
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

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


class Refusal(NamedTuple):
    """A waiting request that didn't fit, the resource that refused it, the node it asked that resource for, and the
    mark that resource returned."""
    rid: str
    key: str
    node_name: str
    mark: object


class Admission:
    """The queues of the resources that ask, and the nodes each request was admitted on.

    A waiting request is admitted once it is first in every queue it is in and each resource it
    asked reserves for it, all or nothing. It may also pass the first request of a queue whose resource
    overrides `Resource.reserve_past`, while that request can't start (`_pass`), and stand anywhere in the queue of
    such a resource once it asks nothing more of the request (`Resource.ask`), since that takes nothing from the
    requests ahead of it. A request that asks again (`readmit`) waits in no queue, and `admit_waiting` tries it,
    oldest first, whenever what it didn't fit has grown. It holds up no newer request unless it is the oldest
    request that has asked a resource it waits for, when that resource's queue admits no one it hasn't already
    reserved for, so newer prompts can't keep passing it.
    """

    def __init__(self, resources: Mapping[str, Resource], ask_order: Collection[str]) -> None:
        self._resources = resources
        # `_arrival` orders the requests: by the conductor's arrival number, then, for a request without
        # one, in the order it first asked. `_numbered` lists the requests with a number, in that order.
        # `_waiting` holds what each waiting request asked.
        # `_queues` holds, per resource, the requests waiting for it in arrival order.
        # `_askers` holds, per resource, every request that has asked it, waiting or admitted, in arrival order.
        # `_returns` maps each waiting request that asks again to the resource and the mark it didn't fit
        # against, or to None until `admit_waiting` tries it. Such a request is in no queue.
        # `_short` maps each resource whose first waiting request didn't fit to the mark it didn't fit
        # against, until that request leaves the front of the queue.
        # `_untried` is set when a queue has a new first request that `admit_waiting` hasn't tried.
        # `_admitted` holds the nodes each request was admitted on, with the walk each was admitted for.
        # `_started` holds, for a request admitted during one node's check, the other nodes waiting on
        # it, which `admit_waiting` reports so their parked checks resume.
        # `_passes` holds the resources that override `Resource.reserve_past`, whose queues a request may pass.
        # `_refused` maps each waiting request a resource refused while it passed to that resource and the mark it
        # didn't fit against. It can't start until that mark grows, so the requests behind it may pass it.
        # `_refusals` maps each such resource and mark to the requests refused against it.
        self._arrivals = itertools.count()
        self._arrival: dict[str, tuple[bool, int]] = {}
        self._numbered: list[str] = []
        self._waiting: dict[str, _Waiting] = {}
        self._queues: dict[str, list[str]] = {key: [] for key in ask_order}
        self._askers: dict[str, list[str]] = {key: [] for key in ask_order}
        self._returns: dict[str, tuple[str, object] | None] = {}
        self._short: dict[str, object] = {}
        self._untried = False
        self._admitted: dict[str, dict[str, str]] = {}
        self._started: dict[str, set[str]] = {}
        self._passes = {key for key in ask_order if type(resources[key]).reserve_past is not Resource.reserve_past}
        self._refused: dict[str, tuple[str, object]] = {}
        self._refusals: dict[tuple[str, object], set[str]] = {}

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

    def older_than(self, rid: str) -> list[str]:
        """The requests numbered before ``rid``, newest first, or none when the conductor gave ``rid`` no number."""
        arrival = self._arrival.get(rid)
        if arrival is None or arrival[0]:
            return []
        return self._numbered[:bisect.bisect_left(self._numbered, arrival, key=self._arrival.__getitem__)][::-1]

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
        if rid in self._returns:
            return FULL_ADMIT_QUEUED
        blocked = self._blocked(self._queues[key][0] for key in entry.asked if key in self._passes)
        if nodes := self._pass(rid, entry.published.get, blocked):
            if others := nodes.keys() - {node_name}:
                self._started.setdefault(rid, set()).update(others)
            return None
        return FULL_ADMIT_QUEUED

    def readmit(self, rid: str, node_name: str, keys: Collection[str]) -> None:
        """Ask ``node_name``'s resources again for ``rid``, since a drop or a move to the host may have given
        back what they admitted it with. A resource that did asks now, and ``rid`` waits as a return unless it
        already waits as a new request. ``keys`` are the resources of ``node_name`` that ask."""
        walk = self._admitted.get(rid, {}).pop(node_name, None)
        # a return moved again asks for something else, so `admit_waiting` tries it again
        if rid in self._returns or (walk is not None and rid not in self._waiting):
            self._returns[rid] = None
        if walk is not None:
            self.admit(rid, node_name, walk, None, keys)
        if rid not in self._waiting:
            self._returns.pop(rid, None)

    def refusals(self) -> list[Refusal]:
        """The waiting requests that have been tried and didn't fit, oldest first."""
        refused = [(rid, *held) for rid, held in self._returns.items() if held is not None]
        refused += [(self._queues[key][0], key, mark) for key, mark in self._short.items()]
        return sorted(
            (Refusal(rid, key, self._waiting[rid].asked[key][0], mark) for rid, key, mark in refused),
            key=lambda refusal: self._arrival[refusal.rid],
        )

    def admit_waiting(
        self, published_of: Callable[[str, str], Mapping[str, PublishedInfo] | None],
    ) -> list[tuple[str, str]]:
        """Admit each return that now fits, oldest first, then each waiting request that is now first
        in every queue it is in, held back by no return, and fits, then each that passes a first request that can't
        start, or waits where a resource holds what it asked for, in arrival order, and return the (node, rid) pairs
        that may start. ``published_of(rid, node)`` gives what the request published for that node's resources.

        Nothing is tried until a resource whose first waiting request didn't fit has grown, a mark that
        requests passing were refused against has grown, or a queue has a new first request, and a return
        is tried once it starts to return or what it didn't fit has grown. A pass that changed none of
        these costs one `Resource.grew_since` for each such resource, mark and return.
        """
        started = []
        if self._started:
            started = [(node, rid) for rid, nodes in self._started.items() for node in nodes]
            self._started.clear()
        if not self._waiting:
            return started
        grown = [key for key, mark in self._short.items() if self._resources[key].grew_since(mark)]
        returns = [
            rid for rid, held in self._returns.items() if held is None or self._resources[held[0]].grew_since(held[1])
        ]
        unblocked = [held for held in self._refusals if self._resources[held[0]].grew_since(held[1])]
        if not grown and not returns and not unblocked and not self._untried:
            return started
        for key in grown:
            del self._short[key]
        for held in unblocked:
            for rid in self._refusals.pop(held):
                del self._refused[rid]
        for rid in sorted(returns, key=self._arrival.__getitem__):
            nodes = self._reserve(rid, lambda node, rid=rid: published_of(rid, node))
            if nodes:
                started.extend((node, rid) for node in nodes)
        admitted = True
        while admitted:
            admitted = False
            for queue in self._queues.values():
                if not queue:
                    continue
                rid = queue[0]
                entry = self._waiting[rid]
                held = self._held_back(rid)
                if any(self._queues[key][0] != rid or key in self._short or key in held for key in entry.asked):
                    continue
                nodes = self._reserve(rid, lambda node, rid=rid: published_of(rid, node))
                if nodes:
                    started.extend((node, rid) for node in nodes)
                    admitted = True
        # the last sweep admitted nothing, so it tried every queue's first request
        swept, self._untried = grown or unblocked or self._untried, False
        if not swept:
            return started
        blocked = self._blocked(queue[0] for key, queue in self._queues.items() if key in self._passes and queue)
        # a request a resource refused while it passed waits for its mark, so the sweep skips it
        waiting = self._waiting.keys() - self._refused.keys() - self._returns.keys()
        for rid in sorted(waiting, key=self._arrival.__getitem__):
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
        for askers in self._askers.values():
            if rid in askers:
                askers.remove(rid)
        arrival = self._arrival.pop(rid, None)
        if arrival is not None and not arrival[0]:
            self._numbered.remove(rid)

    def _park_request(self, rid: str, key: str, node_name: str, graph_walk: str) -> _Waiting:
        """Put ``rid`` in ``key``'s queue at its arrival, unless it waits as a return, and return what it waits for."""
        entry = self._waiting.get(rid)
        if entry is None:
            entry = self._waiting[rid] = _Waiting()
            self._arrival.setdefault(rid, (True, next(self._arrivals)))
        entry.asked[key] = (node_name, graph_walk)
        askers = self._askers[key]
        at = bisect.bisect_left(askers, self._arrival[rid], key=self._arrival.__getitem__)
        if askers[at:at + 1] != [rid]:
            askers.insert(at, rid)
        if rid in self._returns:
            return entry
        queue = self._queues[key]
        bisect.insort(queue, rid, key=self._arrival.__getitem__)
        if queue[0] == rid and len(queue) > 1:
            # An earlier arrival goes ahead of the request the mark measured. That request may now pass it, and once
            # the mark is gone nothing else tries it again.
            self._short.pop(key, None)
            self._untried = True
        return entry

    def _reserve(
        self, rid: str, published_of: Callable[[str], Mapping[str, PublishedInfo] | None],
        passing: Collection[str] = (),
    ) -> dict[str, str] | None:
        """Take everything waiting ``rid`` asked for and return the nodes waiting on it, or take
        nothing and return None. ``published_of`` gives what ``rid`` published for a node's
        resources, and ``passing`` names the resources where ``rid`` passes the first request."""
        entry = self._waiting[rid]
        self._unrefuse(rid)
        taken: list[str] = []
        for key, (node_name, graph_walk) in entry.asked.items():
            published = published_of(node_name)
            resource = self._resources[key]
            reserve = resource.reserve_past if key in passing else resource.reserve
            mark = reserve(rid, node_name, graph_walk, None if published is None else published.get(key))
            if mark is not None:
                for done in taken:
                    self._resources[done].unreserve(rid)
                if rid in self._returns:
                    self._returns[rid] = (key, mark)
                elif key in passing:
                    self._refused[rid] = (key, mark)
                    self._refusals.setdefault((key, mark), set()).add(rid)
                else:
                    self._short[key] = mark
                return None
            taken.append(key)
        del self._waiting[rid]
        self._leave(rid, entry.asked)
        self._admitted.setdefault(rid, {}).update(entry.nodes)
        return entry.nodes

    def _pass(
        self, rid: str, published_of: Callable[[str], Mapping[str, PublishedInfo] | None], blocked: dict[str, str],
    ) -> dict[str, str] | None:
        """Admit waiting ``rid`` where it stands in each queue: first, or behind a first request in ``blocked``
        whose resource lets requests pass, in a queue no return holds back, or anywhere where such a resource asks
        nothing more of ``rid``. Return the nodes waiting on it, or None, and add ``rid`` to ``blocked`` when a
        resource refuses it."""
        if rid in self._refused:
            return None
        held = self._held_back(rid)
        passing, reserved = [], []
        for key, (node_name, graph_walk) in self._waiting[rid].asked.items():
            first = self._queues[key][0]
            if key in held:
                return None
            if key in self._passes and self._resources[key].ask(rid, node_name, graph_walk) is None:
                reserved.append(key)
            elif first == rid:
                if key in self._short:
                    return None
            elif key in self._passes and first in blocked:
                passing.append(key)
            else:
                return None
        # `Resource.reserve_past` leaves a first request what it needs only at the resource ``blocked`` names for it,
        # so ``rid`` passes a first request only if it also passes, or takes nothing, there.
        if any(blocked[self._queues[key][0]] not in passing + reserved for key in passing):
            return None
        nodes = self._reserve(rid, published_of, passing)
        if nodes is None:
            blocked.update(self._blocked([rid]))
        return nodes

    def _blocked(self, rids: Iterable[str]) -> dict[str, str]:
        """Those of the waiting ``rids`` that can't start now and may be passed, each with the resource that keeps it
        waiting. That is a resource whose queue a return holds back for it, or one that refused it, as a first request
        or while it passed, and hasn't grown since."""
        blocked = {}
        for rid in rids:
            back = self._held_back(rid)
            if key := next((key for key in self._waiting[rid].asked if key in back), None):
                blocked[rid] = key
                continue
            held = self._refused.get(rid) or next(
                ((key, mark) for key, mark in self._short.items() if self._queues[key][0] == rid), None,
            )
            if held is not None and not self._resources[held[0]].grew_since(held[1]):
                blocked[rid] = held[0]
        return blocked

    def _held_back(self, rid: str) -> set[str]:
        """The resources waiting ``rid`` asked whose oldest asker waits for them as a return, other than those that
        have already reserved for ``rid`` and so ask nothing more of it."""
        asked = self._waiting[rid].asked
        return {
            key for key, (node_name, graph_walk) in asked.items()
            if (askers := self._askers[key]) and askers[0] in self._returns and key in self._waiting[askers[0]].asked
            and self._resources[key].ask(rid, node_name, graph_walk) is not None
        }

    def _leave(self, rid: str, keys: Collection[str]) -> None:
        """Take ``rid`` out of the queues of ``keys``. A queue it was first in drops its mark,
        which measured ``rid``, and its next request is left for `admit_waiting` to try as a first
        request, even if a resource refused it while it passed, as are the first requests ``rid`` held back."""
        if rid in self._returns:
            del self._returns[rid]
            if any(self._askers[key][:1] == [rid] for key in keys):
                self._untried = True
            return
        self._unrefuse(rid)
        for key in keys:
            queue = self._queues[key]
            if queue[0] == rid:
                self._short.pop(key, None)
                if len(queue) > 1:
                    self._untried = True
                    self._unrefuse(queue[1])
            queue.remove(rid)

    def _unrefuse(self, rid: str) -> None:
        held = self._refused.pop(rid, None)
        if held is not None:
            rids = self._refusals[held]
            rids.discard(rid)
            if not rids:
                del self._refusals[held]
