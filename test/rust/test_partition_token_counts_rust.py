"""The Rust runtime's twin of ``test_token_counts_stay_with_their_partition``
(test/modular/test_partition_bookkeeping.py): a colocated partition's
WORKER_GRAPHS_DONE carries only its own token counts. Frames are read off the
conductor's inbox, after the real transport."""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

from types import SimpleNamespace

import pytest

pytest.importorskip("mstar_rust")

import torch
from mstar_rust import ZmqCommunicator
from test_graph_runtime import _fwd_info, _ingest
from test_partition_bookkeeping import STEPS, add_both_partitions, colocated_partitions

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.communication.tensor_uuid import TensorUuidMinter
from mstar.communication.wire import decode
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.runtime.base import RouteInput, SendInput
from mstar.graph.runtime.rust import RustGraphRuntime
from mstar.utils.containers import ParallelList


def _drain(box):
    out = []
    msg = box.recv_timeout(1000)
    while msg is not None:
        out.append(decode(msg))
        msg = box.recv_timeout(50)
    return out


def test_token_counts_stay_with_their_partition_rust(tmp_path):
    book = RustTensorBookkeeping()
    conductor = ZmqCommunicator("conductor", str(tmp_path))
    api_server = ZmqCommunicator("api_server", str(tmp_path))
    rt = RustGraphRuntime(
        bookkeeping=book, communicator=SimpleNamespace(_inner=ZmqCommunicator("worker_0", str(tmp_path))),
        **colocated_partitions(),
    )
    rid = add_both_partitions(rt)
    minter = TensorUuidMinter("worker_0")
    for part, walk, wg, node, inp in STEPS:
        _ingest(rt, rid, [GraphEdge(next_node=node, name=inp)])
        assert rt.pop_rids(node, walk, [rid], check_ready=True)
        sig = rt.get_output_signals(node, walk)
        uuids = []
        for _ in sig:
            u = minter.mint()
            book.put_tensor(u, TensorPointerInfo(
                dims=[1], dtype=torch.float32, nbytes=4, address=0, stride=(1,), uuid=u,
                source_session_id="s", source_entity="worker_0"))
            book.increment_ref(u, 1)  # the safety hold
            uuids.append(u)
        out = rt.complete_and_route_batch(RouteInput(
            partition=part, graph_walk=walk, node_name=node, output_signals=sig,
            wg_ids=ParallelList([rid], [wg]), tensors=uuids, num_tensors=[1] * len(sig)))
        rt.send_outputs(SendInput(
            completion_id=out.completion_id,
            per_request_info=ParallelList([rid], [_fwd_info(walk, part)]),
            new_token_counts=ParallelList([rid], [{"tok": 1} if part == "P" else {}])))
    dones = [m.body for m in _drain(conductor)]
    _drain(api_server)
    assert [d.partition_name for d in dones] == ["C"]
    assert dones[0].new_token_counts == {}, f"C's WORKER_GRAPHS_DONE carries P's {dones[0].new_token_counts}"
