"""Two behaviours of ``WorkerGraphsManager`` around loops that accumulate their
per-iteration output for the client.

The token loops back to a node that never reads it, so routing must not turn it
into a message to this same worker every iteration. And every persisted tensor
of a loop has to reach the conductor, not only the last iteration's, because the
node's output edges are reused between steps."""
import types

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.worker.node_manager_utils import WorkerGraphQueues, WorkerGraphsManager


class _StubTensorManager:
    def increment_ref(self, request_id, uuid, n=1):
        pass

    def dereference(self, request_id, uuid, n=1):
        pass


def _graph():
    return Sequential(sections=[
        GraphNode(
            name="prefill",
            input_names={"prompt"},
            outputs=[GraphEdge(name="text_inputs", next_node="decode")],
        ),
        Loop(
            name="decode_loop",
            section=GraphNode(
                name="decode",
                input_names={"text_inputs"},
                outputs=[
                    GraphEdge(name="new_token", next_node="decode", persist=True),
                    GraphEdge(name="text_inputs", next_node="decode"),
                ],
            ),
            outputs=[],
            accumulated_outputs=[
                GraphEdge(name="new_token", next_node=EMIT_TO_CLIENT, output_modality="text"),
            ],
            max_iters=8,
        ),
    ])


def _manager(walk="decode", wg_id="wg0", worker_id="worker0"):
    worker_graph = WorkerGraph(section=_graph(), graph_walks={walk}, ranks=[0], worker_graph_id=wg_id)
    queues = {wg_id: WorkerGraphQueues(
        worker_graph_id=wg_id, graph_walks={walk}, worker_graph=worker_graph,
        per_request_queues={}, tensor_manager=_StubTensorManager(),
    )}
    mgr = WorkerGraphsManager(
        queues=queues, per_request_info={},
        base_sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        worker_id=worker_id,
        all_worker_graph_ids_to_graph_walks={wg_id: {walk}},
        all_worker_graph_ids_to_nodes={wg_id: {"prefill", "decode"}},
        all_worker_graph_ids_to_dyn_loops={wg_id: {"decode_loop"}},
        node_to_partition={"prefill": "default", "decode": "default"},
    )
    mgr.add_request(
        request_id="rid", partition_worker_graph_ids=[wg_id],
        worker_graph_to_workers={wg_id: [worker_id]},
        current_fwd_info=CurrentForwardPassInfo(
            request_id="rid", graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=8,
        ),
    )
    return mgr, wg_id, walk


def _info(uuid):
    return types.SimpleNamespace(uuid=uuid)


def test_unread_loop_back_is_not_routed_to_this_worker():
    mgr, wg_id, walk = _manager()
    mgr.process_new_inputs("rid", [GraphEdge(name="prompt", next_node="prefill")])
    completion = mgr.mark_node_complete("rid", wg_id, "prefill")
    mgr.process_node_outputs("rid", "prefill", completion.output_edges, walk)

    completion = mgr.mark_node_complete("rid", wg_id, "decode")
    routing = mgr.process_node_outputs("rid", "decode", completion.output_edges, walk)

    assert routing.to_workers == {}
    assert [e.name for e in routing.persist] == ["new_token"]


def test_flush_keeps_every_persisted_tensor_in_order():
    mgr, _, _ = _manager()
    edge = GraphEdge(name="new_token", next_node="decode", persist=True)
    edge.tensor_info = [_info("t0")]
    mgr.buffer_persist_signals("rid", [edge])
    edge.tensor_info = [_info("t1")]
    mgr.buffer_persist_signals("rid", [edge])

    flushed = mgr.flush_persist_signals("rid")

    assert [i.uuid for i in flushed["new_token"]] == ["t0", "t1"]
    assert mgr.flush_persist_signals("rid") == {}
