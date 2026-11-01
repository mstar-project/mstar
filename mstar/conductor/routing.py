"""Choosing a new request's data-parallel replica in each node group.

A replica is one lockstep instance of a group: its ``tp_size * sp_size`` ranks,
or one rank. A replica's cost for the request is ``routing_weights.load`` times
its live requests, plus ``routing_cost`` of the facts the mirrors predict for
the request there:

- the live requests are counted per set of groups that pick together, so
  groups on other ranks that share a worker don't skew each other's picks, and
  only for a request one of whose partitions, not done at ingest, has a walk on
  the set;
- a resource whose spec offers a mirror for the set
  (``NodeResourceSpec.routing_mirror``) reports ``RoutingFacts``, a row per
  replica, such as a paged KV cache's predicted prefix hit. The rows of every
  mirror on the set add up before they are priced.

Every cost is in request-equivalents. The cheapest replica wins, ties at
random. Groups whose replicas sit on the same ranks, replica for replica, pick
one index together, so a request's walks stay on one set of workers. A
prediction never changes outputs: a wrong one makes a request slower.
"""

import logging
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from mstar.engine.resources.kv.config import prefix_caches_by_rank
from mstar.engine.resources.spec import (
    NodeResourceSpec,
    ResourceReqConfig,
    RoutingContext,
    RoutingFacts,
    RoutingMirror,
)
from mstar.model.base import Model, keyed_labels, names_checkpoint
from mstar.observability import catalog
from mstar.observability.recorders import Counter as MetricCounter

logger = logging.getLogger(__name__)

Instance = tuple[int, ...]


class RoutingWeights(NamedTuple):
    """The serving config's ``routing_weights``: one weight per term of the cost."""
    load: float = 1.0
    # provisional: the best weight in simulations of BAGEL and Qwen3.5 0.8B and 9B
    # that credit Qwen3.5's hits whole, as the KV mirror does, where its workers
    # serve them only up to a recurrent snapshot; above 16 a hot prefix piles
    # onto one replica
    prefix: float = 8.0


def routing_cost(facts: RoutingFacts, weights: RoutingWeights) -> float:
    """What a replica's facts add to its cost. A term is 0 where its facts are
    absent, so a resource with no mirror drops out.

    ``prefix`` is minus the share of the request's positions the replica's KV
    caches are predicted to hold: a prompt held whole is worth up to ``prefix``
    live requests. A share, not a token count, so one weight fits models whose
    prefill and decode costs differ.
    """
    return -weights.prefix * facts.kv_hit / facts.work if facts.work else 0.0


def _add(a: RoutingFacts, b: RoutingFacts) -> RoutingFacts:
    return RoutingFacts(*map(sum, zip(a, b, strict=True)))


# each term's weight alone, to count what it predicted on the picked replica
_TERMS = {
    term: RoutingWeights(**{name: float(name == term) for name in RoutingWeights._fields})
    for term in RoutingWeights._fields if term != "load"
}


class Placement(NamedTuple):
    """A new request's pick, for ``ReplicaRouter.add`` to record once the request dispatches."""
    # node group -> its replica
    replicas: dict[int, int]
    # unit index -> the facts on its picked replica
    facts: dict[int, RoutingFacts]


class _Mirror(NamedTuple):
    key: str
    # the unit's groups holding the resource's nodes: a request that runs none skips it
    groups: frozenset[int]
    mirror: RoutingMirror


class _Held(NamedTuple):
    mirror: _Mirror
    handle: Any


class RoutingHandle(NamedTuple):
    """What placing one request recorded, for ``ReplicaRouter.remove``."""
    # (unit index, replica) it counts on
    load: tuple[tuple[int, int], ...]
    mirrors: tuple[_Held, ...]


class _Unit(NamedTuple):
    """Node groups placed on the same ranks, replica for replica: one pick places them all."""
    groups: tuple[int, ...]
    instances: tuple[Instance, ...]
    mirrors: tuple[_Mirror, ...]


class ReplicaRouter:
    def __init__(
        self,
        model: Model,
        worker_graphs: Iterable,
        specs: Iterable[NodeResourceSpec] = (),
        weights: Mapping[str, Any] | None = None,
    ):
        """``weights`` are the serving config's ``routing_weights``, which
        ``validate_deployment`` checked. Each spec is asked for a mirror once per
        set of groups with replicas that holds one of its nodes."""
        worker_graphs, specs = list(worker_graphs), list(specs)
        self._weights = RoutingWeights(**{name: float(w) for name, w in (weights or {}).items()})
        units: dict[tuple[Instance, ...], list] = {}
        for wg in worker_graphs:
            units.setdefault(tuple(map(tuple, wg.replica_ranks)), []).append(wg)
        # the KV declarations a mirror reads: keyed_labels as the engine reads them, and
        # names_checkpoint in place of the engine's checkpoint_path() is not None
        keyed = {
            key: {label: kv_label.keys.walks for label, kv_label in labels.items()}
            for key, labels in keyed_labels(model).items()
        }
        by_rank = prefix_caches_by_rank(
            specs, ((wg.ranks, wg.tp_size, wg.section.get_nodes()) for wg in worker_graphs), names_checkpoint(model),
        )
        self._units: list[_Unit] = []
        for instances, wgs in units.items():
            # what every replica's engines open
            opened = frozenset.intersection(*(by_rank[rank] for instance in instances for rank in instance))
            mirrors = []
            for spec in specs if len(instances) > 1 else ():
                holding = [wg for wg in wgs if not spec.nodes.isdisjoint(wg.section.get_nodes())]
                if not holding:
                    continue
                walks = frozenset().union(*(wg.graph_walks for wg in holding))
                mirror = spec.routing_mirror(RoutingContext(
                    len(instances), walks, keyed.get(spec.resource_key, {}), spec.resource_key in opened,
                ))
                if mirror is not None:
                    mirrors.append(_Mirror(spec.resource_key, frozenset(wg._group_id for wg in holding), mirror))
            groups = tuple(dict.fromkeys(wg._group_id for wg in wgs))
            self._units.append(_Unit(groups, instances, tuple(mirrors)))
        if any(len(unit.instances) > 1 for unit in self._units):
            logger.info(
                "Replica pick: weights %s; mirrors %s", self._weights._asdict(),
                [(m.key, unit.instances) for unit in self._units for m in unit.mirrors],
            )
        self._partition_groups = {
            p.name: frozenset(wg._group_id for wg in worker_graphs if wg.graph_walks & p.graph_walks)
            for p in model.get_partitions()
        }
        # live requests per (unit index, replica)
        self._load: Counter[tuple[int, int]] = Counter()
        self.predicted_saving = MetricCounter(catalog.ROUTING_PREDICTED_SAVING)

    def groups_run(self, partitions: Iterable[str]) -> frozenset[int]:
        """The node groups of these partitions: what a request whose other partitions are done runs."""
        return frozenset().union(*(self._partition_groups[p] for p in partitions))

    def pick(self, configs: Mapping[str, ResourceReqConfig], groups_run: frozenset[int]) -> Placement:
        """Each node group's replica for a new request that runs ``groups_run``
        and opens its resources with ``configs``. Records nothing."""
        replicas: dict[int, int] = {}
        picked: dict[int, RoutingFacts] = {}
        for u, unit in enumerate(self._units):
            choice = 0
            if len(unit.instances) > 1:
                # the facts on each replica, every mirror's fields added
                row: list[RoutingFacts] | None = None
                for m in unit.mirrors:
                    if m.key in configs and not groups_run.isdisjoint(m.groups):
                        facts = m.mirror.facts(configs[m.key])
                        row = facts if row is None else list(map(_add, row, facts))
                cost = [
                    self._weights.load * self._load[(u, i)]
                    + (0.0 if row is None else routing_cost(row[i], self._weights))
                    for i in range(len(unit.instances))
                ]
                least = min(cost)
                choice = random.choice([i for i, c in enumerate(cost) if c == least])
                if row is not None:
                    picked[u] = row[choice]
            for group in unit.groups:
                replicas[group] = choice
        return Placement(replicas, picked)

    def add(
        self, placement: Placement, configs: Mapping[str, ResourceReqConfig], groups_run: frozenset[int],
    ) -> RoutingHandle:
        """Record a request placed by ``placement``, in every mirror of the groups it runs."""
        load = []
        handles = []
        for u, unit in enumerate(self._units):
            if groups_run.isdisjoint(unit.groups):
                continue
            replica = placement.replicas[unit.groups[0]]
            load.append((u, replica))
            for m in unit.mirrors:
                if m.key in configs and not groups_run.isdisjoint(m.groups):
                    mirror_handle = m.mirror.add_request(configs[m.key], replica)
                    if mirror_handle is not None:
                        handles.append(_Held(m, mirror_handle))
        if placement.facts:
            # the facts of every set of replicas it was placed on, priced once: a share of its whole prompt
            facts = RoutingFacts(*map(sum, zip(*placement.facts.values(), strict=True)))
            for term, alone in _TERMS.items():
                saving = -routing_cost(facts, alone)
                if saving:
                    self.predicted_saving.inc((term,), saving)
        handle = RoutingHandle(tuple(load), tuple(handles))
        self._load.update(handle.load)
        return handle

    def remove(self, handle: RoutingHandle, ran: Iterable[str]) -> None:
        """Release a request recorded by ``add``. ``ran`` names its partitions
        with a pass that came back."""
        self._load.subtract(handle.load)
        groups_ran = self.groups_run(ran)
        for held in handle.mirrors:
            held.mirror.mirror.remove_request(held.handle, not groups_ran.isdisjoint(held.mirror.groups))
