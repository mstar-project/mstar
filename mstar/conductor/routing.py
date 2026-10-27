"""Choosing a new request's data-parallel replica in each node group.

A replica is one lockstep instance of a group: its ``tp_size * sp_size`` ranks,
or one rank. Each replica's cost for the request is a weighted sum over mirrors,
conductor-side predictions of the replica's state:

- the forward mirror: the live requests on the replica, counted per set of
  groups that pick together, so groups on other ranks that share a worker
  don't skew each other's picks, and only for a request one of whose
  partitions, not done at ingest, has a walk on the set;
- one mirror per resource whose spec offers one (``NodeResourceSpec.routing_mirror``),
  such as a paged KV cache's predicted prefix hits.

Every cost is in request-equivalents, so a KV cache scores a replica minus the
share of the prompt it holds, which stays above -1: a hit weighs less than one
more live request. The cheapest replica wins, ties at random. Groups whose
replicas sit on the same ranks, replica for replica, pick one index together,
so a request's walks stay on one set of workers. The weights are the serving config's ``routing_weights``
(``load`` for the forward mirror, a resource key for its mirror), 1.0 unless set.
A prediction never changes outputs: a wrong one makes a request slower.
"""

import logging
import random
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NamedTuple

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig, RoutingContext, RoutingMirror
from mstar.observability import catalog
from mstar.observability.recorders import Counter as MetricCounter

logger = logging.getLogger(__name__)

Instance = tuple[int, ...]


class Placement(NamedTuple):
    """A new request's pick, for ``ReplicaRouter.add`` to record once the request dispatches."""
    # node group -> its replica
    replicas: dict[int, int]
    # resource key -> the saving its mirror predicted on the picked replica
    savings: dict[str, float]


class _Held(NamedTuple):
    mirror: RoutingMirror
    handle: Any


class RoutingHandle(NamedTuple):
    """What placing one request recorded, for ``ReplicaRouter.remove``."""
    # (unit index, replica) it counts on
    load: tuple[tuple[int, int], ...]
    mirrors: tuple[_Held, ...]


class _Mirror(NamedTuple):
    key: str
    weight: float
    # the unit's groups holding the resource's nodes: a request that runs none skips it
    groups: frozenset[int]
    mirror: RoutingMirror


class _Unit(NamedTuple):
    """Node groups placed on the same ranks, replica for replica: one pick places them all."""
    groups: tuple[int, ...]
    instances: tuple[Instance, ...]
    mirrors: tuple[_Mirror, ...]


class ReplicaRouter:
    def __init__(
        self,
        worker_graphs: Iterable,
        partitions: Iterable,
        specs: Iterable[NodeResourceSpec] = (),
        weights: Mapping[str, Any] | None = None,
        checkpoint_named: Callable[[], bool] = lambda: False,
    ):
        """``weights`` are the serving config's ``routing_weights``, which
        ``validate_deployment`` checked. ``checkpoint_named`` says whether the
        model names its checkpoint, which a prefix cache roots in. Asked only
        when a group has replicas, since a model may download its checkpoint to
        answer (BAGEL's ``checkpoint_path`` does)."""
        worker_graphs, specs = list(worker_graphs), list(specs)
        weights = {name: float(weight) for name, weight in (weights or {}).items()}
        self._load_weight = weights.get("load", 1.0)
        groups: dict[tuple[Instance, ...], dict[int, None]] = {}
        nodes: dict[tuple[Instance, ...], set[str]] = {}
        group_nodes: dict[int, set[str]] = {}
        for wg in worker_graphs:
            instances = wg.replica_instances
            groups.setdefault(instances, {})[wg._group_id] = None
            nodes.setdefault(instances, set()).update(wg.section.get_nodes())
            group_nodes.setdefault(wg._group_id, set()).update(wg.section.get_nodes())
        # a resource gets a mirror only where all its nodes, and so its one pool
        # per replica, sit in one unit
        units_of = Counter(node for unit_nodes in nodes.values() for node in unit_nodes)
        self._units: list[_Unit] = []
        named = None
        for instances, unit_groups in groups.items():
            mirrors = []
            if len(instances) > 1 and specs:
                if named is None:
                    named = checkpoint_named()
                for spec in specs:
                    if spec.nodes.isdisjoint(nodes[instances]):
                        continue
                    if not spec.nodes <= nodes[instances] or any(units_of[n] > 1 for n in spec.nodes):
                        logger.info(
                            "Replica pick: %s spans node groups on other ranks, so it takes no part",
                            spec.resource_key,
                        )
                        continue
                    mirror = spec.routing_mirror(RoutingContext(len(instances), len(instances[0]), named))
                    if mirror is not None:
                        holding = frozenset(g for g in unit_groups if not spec.nodes.isdisjoint(group_nodes[g]))
                        mirrors.append(_Mirror(spec.resource_key, weights.get(spec.resource_key, 1.0), holding, mirror))
            self._units.append(_Unit(tuple(unit_groups), instances, tuple(mirrors)))
        if any(len(unit.instances) > 1 for unit in self._units):
            mirrored = {m.key: m.weight for unit in self._units for m in unit.mirrors}
            logger.info(
                "Replica pick: load weight %g; resource weights %s; weights with no mirror here: %s",
                self._load_weight, mirrored, sorted(weights.keys() - mirrored.keys() - {"load"}),
            )
        self._partition_groups = {
            p.name: frozenset(wg._group_id for wg in worker_graphs if wg.graph_walks & p.graph_walks)
            for p in partitions
        }
        # the forward mirror: live requests per (unit index, replica)
        self._load: Counter[tuple[int, int]] = Counter()
        self.predicted_saving = MetricCounter(catalog.ROUTING_PREDICTED_SAVING)

    def groups_run(self, partitions: Iterable[str]) -> frozenset[int]:
        """The node groups of these partitions: what a request whose other partitions are done runs."""
        return frozenset().union(*(self._partition_groups[p] for p in partitions))

    def pick(self, configs: Mapping[str, ResourceReqConfig], groups_run: frozenset[int]) -> Placement:
        """Each node group's replica for a new request that runs ``groups_run``
        and opens its resources with ``configs``. Records nothing."""
        replicas: dict[int, int] = {}
        savings: dict[str, float] = {}
        for u, unit in enumerate(self._units):
            choice = 0
            if len(unit.instances) > 1:
                cost = [self._load_weight * self._load[(u, i)] for i in range(len(unit.instances))]
                scores = {}
                for m in unit.mirrors:
                    if m.key in configs and not groups_run.isdisjoint(m.groups):
                        scores[m.key] = m.mirror.score(configs[m.key])
                        for i, c in enumerate(scores[m.key]):
                            cost[i] += m.weight * c
                least = min(cost)
                choice = random.choice([i for i, c in enumerate(cost) if c == least])
                savings.update((key, -score[choice]) for key, score in scores.items() if score[choice])
            for group in unit.groups:
                replicas[group] = choice
        return Placement(replicas, savings)

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
                        handles.append(_Held(m.mirror, mirror_handle))
        for key, saving in placement.savings.items():
            self.predicted_saving.inc((key,), saving)
        handle = RoutingHandle(tuple(load), tuple(handles))
        self._load.update(handle.load)
        return handle

    def remove(self, handle: RoutingHandle) -> None:
        self._load.subtract(handle.load)
        for held in handle.mirrors:
            held.mirror.remove_request(held.handle)
