"""The batch-router harness: one interface, swappable backend.

Scope is **one worker**. Routing needs what a single worker graph cannot see —
which local worker graph owns a destination node, the request's sharding
config, the cross-worker fanout — so this sits one level above
``mstar.graph.runtime``, which owns a single worker graph's walk state.

The worker calls the wrapper; the wrapper calls the backend. As with the graph
runtime, ``shadow`` runs two backends and reports where they disagree.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mstar.worker.node_manager_utils import NodeOutputRouting


class RoutingWorld(Protocol):
    """What a router needs from the worker it serves.

    ``WorkerGraphsManager`` satisfies this; stating it as a protocol keeps the
    dependency one-way and makes a router testable without a whole worker.
    """

    queues: dict
    per_request_info: dict
    worker_id: str

    def routing_context(self, sharding_config, node_name: str, graph_walk: str): ...


class BatchRouterBase(ABC):
    """Completes a node for a batch of requests and routes its outputs."""

    #: Identifies the implementation in logs and divergence reports.
    backend: str = "?"

    def __init__(self, world: RoutingWorld):
        self.world = world

    @abstractmethod
    def route_batch(
        self,
        node_name: str,
        request_to_worker_graph: dict[str, str],
        graph_walk: str,
    ) -> dict["str", "NodeOutputRouting"]:
        """Complete ``node_name`` for every request and route what it emits.

        Returns one ``NodeOutputRouting`` per request — the same shape the
        per-request path produces, so the worker's send path is unchanged.
        """
