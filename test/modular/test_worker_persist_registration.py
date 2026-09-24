"""A persisted output is staged for other workers only when there are any.

A ``persist=True`` output goes back to the conductor, which feeds it to a later
walk of the same request. Staging it for a send (the SHM transport serializes
the tensor to a file) is only useful if that walk can land on another worker;
with a single worker it costs 13 ms per 7.5 MiB text embedding and nobody ever
reads the file. Outputs for the client and for other workers are staged as
before either way.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.graph.base import GraphEdge, TensorPointerInfo  # noqa: E402
from mstar.worker.node_manager_utils import NodeOutputRouting  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402


def _info(uuid: str) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=(1,), dtype=None, stride=(1,), nbytes=2, address=0, uuid=uuid,
        source_session_id="s", source_entity="worker_0",
    )


def _edge(name: str, next_node: str, uuid: str, persist: bool = False) -> GraphEdge:
    edge = GraphEdge(name=name, next_node=next_node, persist=persist)
    edge.tensor_info = [_info(uuid)]
    return edge


def _worker(peers: list[str]):
    registered: list[str] = []
    worker = SimpleNamespace(
        _peer_workers=peers,
        tensor_manager=SimpleNamespace(
            register_for_send=lambda request_id, tensor_infos, skip_cuda_sync=False:
                registered.extend(sorted(i.uuid for i in tensor_infos)),
        ),
    )
    worker._register_outputs = Worker._register_outputs.__get__(worker)
    return worker, registered


def _routing() -> NodeOutputRouting:
    return NodeOutputRouting(
        routed_to_this_worker_graph=[_edge("latents", "dit", "local")],
        is_first_tp_rank=True,
        persist=[_edge("text_embeds", "EMPTY_DESTINATION", "persisted", persist=True)],
        to_workers={"worker_1": [_edge("hidden", "decoder", "remote")]},
        emit_to_client=[_edge("image", "EMIT_TO_CLIENT", "client")],
    )


def _register(worker) -> None:
    batch = SimpleNamespace(node_objects={"r": object()})
    worker._register_outputs(batch, {"r": _routing()})


def test_single_worker_stages_client_and_remote_outputs_but_not_persisted_ones():
    worker, registered = _worker(peers=[])
    _register(worker)
    assert registered == ["client", "remote"]


def test_with_peer_workers_persisted_outputs_are_staged_too():
    worker, registered = _worker(peers=["worker_1"])
    _register(worker)
    assert registered == ["client", "persisted", "remote"]
