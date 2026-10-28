"""The Rust runtime wrapper hands ``SendInput.failed_rids`` to Rust, which
then builds no frame for those rids (no WORKER_GRAPHS_DONE among them); the
Python runtime's side is in test/modular/test_send_registration_failure.py."""
import pytest
import torch

pytest.importorskip("mstar_rust")

from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, TensorPointerInfo
from mstar.graph.runtime import rust as rust_runtime
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput, SendInput
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList


@pytest.mark.parametrize("failed", [False, True])
def test_a_failed_rid_gets_no_completion_frame(failed):
    section = GraphNode(
        name="gen", input_names={"tok"},
        outputs=[GraphEdge(name="out", next_node=EMIT_TO_CLIENT)],
    )
    wg = WorkerGraph(section=section, graph_walks={"w"}, ranks=[0], worker_graph_id=0)
    book = RustTensorBookkeeping()
    rt = rust_runtime.RustGraphRuntime(
        my_worker_id="worker_0", my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={0: {"w"}}, all_wg_ids_to_dyn_loops={0: set()},
        all_wg_ids_to_nodes={0: {"gen"}}, node_to_partition={"gen": "default"},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=book,
    )
    rid = rt.add_request(
        request_id="r", partition="default", graph_walk="w",
        partition_worker_graph_ids=[0],
        worker_graph_to_workers=ParallelList([0], [["worker_0"]]),
        loop_caps={},
    )
    block = ColumnarEdgeSpecs.empty()
    block.add(rid, "tok", [], False, next_node="gen")
    rt.ingest_inputs_batch(block)
    rt.pop_rids("gen", "w", [rid])
    book.put_tensor(1, TensorPointerInfo(
        dims=[1], dtype=torch.float32, nbytes=4, address=0, stride=(1,),
        uuid=1, source_session_id="h:1", source_entity="worker_0",
    ))
    book.increment_ref(1, 1)
    out = rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk="w", node_name="gen",
        output_signals=["out"], wg_ids=ParallelList([rid], [0]),
        tensors=[1], num_tensors=[1],
    ))

    sent_rids = rt.send_outputs(SendInput(
        completion_id=out.completion_id,
        per_request_info=ParallelList([rid], [None]),
        new_token_counts=ParallelList([], []),
        failed_rids=frozenset({rid}) if failed else frozenset(),
    ))

    assert sent_rids == ([] if failed else [rid])


def test_a_failed_rid_sends_no_peer_edge_persist_or_emit_next_to_a_kept_one(tmp_path):
    """Two rows complete together; the first failed after routing. Its peer
    edge, persist signal, client emit and WORKER_GRAPHS_DONE are all dropped,
    and the second row's go out as before."""
    from types import SimpleNamespace

    from mstar_rust import ZmqCommunicator

    import mstar.communication.wire_types  # noqa: F401  (registers the tags)
    from mstar.communication.wire import decode

    me, peer = "worker_0", "worker_1"
    section = GraphNode(
        name="gen", input_names={"tok"},
        outputs=[
            GraphEdge(name="out", next_node="remote"),
            GraphEdge(name="keep", next_node="", persist=True),
            GraphEdge(name="txt", next_node=EMIT_TO_CLIENT),
        ],
    )
    wg = WorkerGraph(section=section, graph_walks={"w"}, ranks=[0], worker_graph_id=0)
    book = RustTensorBookkeeping()
    comm = ZmqCommunicator(me, str(tmp_path))
    inboxes = {e: ZmqCommunicator(e, str(tmp_path)) for e in ("conductor", "api_server", peer)}
    rt = rust_runtime.RustGraphRuntime(
        my_worker_id=me, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={0: {"w"}, 1: {"w"}}, all_wg_ids_to_dyn_loops={0: set(), 1: set()},
        all_wg_ids_to_nodes={0: {"gen"}, 1: {"remote"}},
        node_to_partition={"gen": "default", "remote": "default"},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=book, communicator=SimpleNamespace(_inner=comm),
    )
    rids = [
        rt.add_request(
            request_id=name, partition="default", graph_walk="w",
            partition_worker_graph_ids=[0, 1],
            worker_graph_to_workers=ParallelList([0, 1], [[me], [peer]]),
            loop_caps={},
        )
        for name in ("failed", "kept")
    ]
    block = ColumnarEdgeSpecs.empty()
    for rid in rids:
        block.add(rid, "tok", [], False, next_node="gen")
    rt.ingest_inputs_batch(block)
    rt.pop_rids("gen", "w", rids)
    signals = rt.get_output_signals("gen", "w")
    uuids = []
    # one tensor per signal of each row, row-major
    for uuid in range(1, len(rids) * len(signals) + 1):
        book.put_tensor(uuid, TensorPointerInfo(
            dims=[1], dtype=torch.float32, nbytes=4, address=0, stride=(1,),
            uuid=uuid, source_session_id="h:1", source_entity=me,
        ))
        book.increment_ref(uuid, 1)
        uuids.append(uuid)
    out = rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk="w", node_name="gen",
        output_signals=signals, wg_ids=ParallelList(rids, [0, 0]),
        tensors=uuids, num_tensors=[1] * len(uuids),
    ))

    sent_rids = rt.send_outputs(SendInput(
        completion_id=out.completion_id,
        per_request_info=ParallelList(rids, [None, None]),
        new_token_counts=ParallelList([], []),
        failed_rids=frozenset({rids[0]}),
    ))

    def collect(box):
        got, msg = [], box.recv_timeout(1000)
        while msg is not None:
            got.append(decode(msg))
            msg = box.recv_timeout(50)
        return got

    got = {entity: collect(box) for entity, box in inboxes.items()}
    assert sent_rids == [rids[1]]
    assert [m.body.request_id for m in got[peer]] == ["kept"]
    assert [m.body.request_id for m in got["api_server"]] == ["kept"]
    (done,) = got["conductor"]
    assert done.body.request_id == "kept"
    assert list(done.body.persist_signals) == ["keep"]
