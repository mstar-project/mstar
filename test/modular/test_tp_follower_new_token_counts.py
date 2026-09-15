"""New-token counting in ``Worker._send_outputs`` happens on the first TP rank only: the
conductor takes the counts from the rank-0 WORKER_GRAPHS_DONE, and follower ranks never hold
the emitted tensors the new-token edges point at (only the leader emits to the client), so a
follower looking them up raised KeyError on every decode step of a TP>1 model."""
import types

import torch

from mstar.graph.base import GraphEdge
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.worker.node_manager_utils import NodeOutputRouting
from mstar.worker.worker import Worker


def _fake_worker(store: dict):
    counts: dict = {}
    sent: list = []
    wgm = types.SimpleNamespace(
        get_graph_walk=lambda rid, part: "decode",
        get_fwd_info=lambda rid, part: None,
        buffer_persist_signals=lambda rid, p: None,
        buffer_new_token_counts=lambda rid, c: counts.update(c),
        buffer_output_signals=lambda rid, e: None,
        register_output_loop_indices=lambda **kw: None,
        per_request_info={"r": types.SimpleNamespace(stream_buffers={}, per_partition_info={})},
    )
    tm = types.SimpleNamespace(get_tensor=lambda request_id, uuid: store[request_id][uuid])
    comm = types.SimpleNamespace(send=lambda dst, msg: sent.append((dst, msg)))
    return types.SimpleNamespace(worker_graphs_manager=wgm, tensor_manager=tm, communicator=comm), counts, sent


def _routing(first_tp_rank: bool) -> NodeOutputRouting:
    edge = GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", output_modality="text", conductor_new_token=True)
    edge.tensor_info = [types.SimpleNamespace(uuid="u1")]
    return NodeOutputRouting(
        routed_to_this_worker_graph=[], is_first_tp_rank=first_tp_rank, persist=[], to_workers={},
        new_token_outputs=[edge],
    )


def test_follower_rank_skips_new_token_counting():
    # the follower's store has no tensor for the emitted edge: must neither raise nor count
    w, counts, sent = _fake_worker({"r": {}})
    Worker._send_outputs(w, "r", _routing(False), nested_loop_indices=None, graph_walk="decode", partition_name="p")
    assert counts == {} and sent == []


def test_first_rank_counts_new_tokens():
    w, counts, _ = _fake_worker({"r": {"u1": torch.zeros(3, dtype=torch.int64)}})
    Worker._send_outputs(w, "r", _routing(True), nested_loop_indices=None, graph_walk="decode", partition_name="p")
    assert counts == {"new_token": 3}
