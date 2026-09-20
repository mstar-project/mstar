"""Reference router: the batched Python path.

Everything keyed by (node, graph walk) is resolved once into a
``RoutingContext`` and reused across the batch; what remains per request is
the completion, the edge clones and any sharded slice.
"""
import logging

from mstar.graph.routing.base import BatchRouterBase

logger = logging.getLogger(__name__)


class PythonBatchRouter(BatchRouterBase):
    backend = "python"

    def route_batch(self, node_name, request_to_worker_graph, graph_walk):
        world = self.world
        rids = list(request_to_worker_graph)
        if not rids:
            return {}

        # Sharing one RoutingContext assumes every request in the batch shares
        # a sharding config. That holds whenever the conductor gave them the
        # same worker_graph_to_workers map — i.e. always, today — but a batch
        # that mixes configs would get a fanout planned from the wrong one, so
        # fall back rather than assume.
        configs = {id(world.per_request_info[r].sharding_config) for r in rids}
        if len(configs) > 1:
            logger.debug(
                "Batch for node %s spans %d sharding configs; routing per request",
                node_name, len(configs),
            )
            return {
                rid: world.process_node_outputs(
                    rid, node_name=node_name, graph_walk=graph_walk,
                    outputs=self._complete(rid, wg, node_name),
                )
                for rid, wg in request_to_worker_graph.items()
            }

        ctx = world.routing_context(
            world.per_request_info[rids[0]].sharding_config, node_name, graph_walk,
        )
        return {
            rid: world._route_one(
                rid, self._complete(rid, request_to_worker_graph[rid], node_name), ctx,
            )
            for rid in rids
        }

    def _complete(self, rid: str, wg_id: str, node_name: str):
        """The completion half: what the node emitted, as fresh edges."""
        completion = self.world.queues[wg_id].runtime.complete(rid, node_name)
        return [edge.clone() for edge in completion.output_edges]
