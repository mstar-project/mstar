"""Persisted outputs are copied to the tensor transport only when another
worker could read them. With one worker the producer serves later walks from
its own store, so the copy would never be opened."""
import types

from mstar.graph.base import GraphEdge
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.worker.worker import Worker


class _Recorder:
    def __init__(self):
        self.registered = []

    def register_for_send(self, request_id, tensor_infos, skip_cuda_sync=False):
        self.registered.extend(info.uuid for info in tensor_infos)


def _edge(name, next_node, uuid, persist=False):
    edge = GraphEdge(next_node=next_node, name=name, persist=persist)
    edge.tensor_info = [types.SimpleNamespace(uuid=uuid)]
    return edge


def _stub(workers_by_node):
    stub = types.SimpleNamespace()
    stub.worker_id = "worker_0"
    stub.tensor_manager = _Recorder()
    stub._remote_workers_by_rid = {}
    info = types.SimpleNamespace(node_to_workers=workers_by_node)
    stub.worker_graphs_manager = types.SimpleNamespace(per_request_info={"r1": info})
    stub._has_remote_workers = lambda rid: Worker._has_remote_workers(stub, rid)
    return stub


def _run(stub):
    routing = types.SimpleNamespace(
        persist=[_edge("encoder_states", "decoder", "persisted", persist=True)],
        to_workers={},
        emit_to_client=[_edge("new_token", EMIT_TO_CLIENT, "emitted")],
        streaming_to_workers={},
    )
    batch = types.SimpleNamespace(node_objects={"r1": object()})
    Worker._register_outputs(stub, batch, {"r1": routing})
    return stub.tensor_manager.registered


def test_single_worker_skips_persisted_outputs():
    stub = _stub({("decoder", "decode"): ["worker_0"], ("encoder", "prefill"): ["worker_0"]})
    assert _run(stub) == ["emitted"]
    assert stub._remote_workers_by_rid == {"r1": False}


def test_remote_worker_keeps_persisted_outputs():
    stub = _stub({("decoder", "decode"): ["worker_0", "worker_1"]})
    assert sorted(_run(stub)) == ["emitted", "persisted"]


def test_unknown_request_is_treated_as_remote():
    stub = _stub({})
    stub.worker_graphs_manager.per_request_info.clear()
    assert Worker._has_remote_workers(stub, "r1") is True
