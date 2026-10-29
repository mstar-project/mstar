"""Finished loops exchange cache ownership for routed readers exactly once.

Real Python runtime, SHM tensor store and cleanup; synthetic producer tensors.
Client releases model completed reads/ACKs, not actual network transport.
"""
from types import SimpleNamespace

import pytest
import torch

from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList


def _runtime(
    tmp_path, *, accumulated, nested, readers, persist, limit, tail=False, remote=False, inner_accumulated=False,
):
    manager = SharedMemoryCommunicationManager(
        my_entity_id="worker0", hostname="localhost", device="cpu",
        communicator=SimpleNamespace(send=lambda *a, **kw: None), shm_dir=str(tmp_path),
    )
    node = GraphNode(name="producer", input_names=["token"],
                     outputs=[GraphEdge(name="token", next_node="producer")])
    section = node
    if tail:
        node.outputs.append(GraphEdge(name="tick", next_node="tail"))
        section = Sequential(sections=[node, GraphNode(
            name="tail", input_names=["tick"], outputs=[],
        )])
    if inner_accumulated:
        node.outputs.append(GraphEdge(name="frame", next_node=EMPTY_DESTINATION))
    if nested:
        section = Loop(name="inner", section=node, max_iters=2,
                       outputs=[GraphEdge(name="token", next_node="producer")],
                       accumulated_outputs=([GraphEdge(name="frame", next_node=EMIT_TO_CLIENT)
                                             for _ in range(readers)] if inner_accumulated else []))
    edges = ([GraphEdge(name="token", next_node=EMPTY_DESTINATION, persist=True)] if persist else
             [GraphEdge(name="token", next_node=EMIT_TO_CLIENT) for _ in range(readers)])
    if inner_accumulated:
        edges = []
    if remote:
        edges = [GraphEdge(name="token", next_node="consumer")]
    section = Loop(name="outer", section=section, max_iters=limit,
                   outputs=[] if accumulated else edges,
                   accumulated_outputs=edges if accumulated else [])
    graph = WorkerGraph(section=section, graph_walks={"walk"}, ranks=[0], worker_graph_id=0)
    walks = {0: {"walk"}}
    nodes = {0: set(section.get_nodes())}
    loops = {0: set(section.get_loops())}
    mapping = ParallelList([0], [["worker0"]])
    if remote:
        walks[1], nodes[1], loops[1] = {"walk"}, {"consumer"}, set()
        mapping = ParallelList([0, 1], [["worker0"], ["worker1", "worker2"]])
    runtime = PythonGraphRuntime(
        my_worker_id="worker0", my_worker_graphs=[graph],
        all_wg_ids_to_graph_walks=walks, all_wg_ids_to_nodes=nodes,
        all_wg_ids_to_dyn_loops=loops,
        node_to_partition=dict.fromkeys(section.get_nodes(), "default"),
        sharding_config=ShardingConfig(
            groups=[ShardingGroup(nodes={"consumer"}, tp_size=2)] if remote else [],
            tp_enabled_nodes={"consumer"} if remote else set(), shard_dim={},
        ),
        tensor_manager=manager,
    )
    caps = {name: loop.max_iters for name, loop in section.get_loops().items()}
    rid = runtime.add_request("request", "default", "walk", [0], mapping, caps)
    manager.register_request(rid, runtime.get_sharding_config(rid))
    block = ColumnarEdgeSpecs.empty()
    block.add_edge(rid, GraphEdge(name="token", next_node="producer"))
    assert runtime.ingest_inputs_batch(block) == []
    return runtime, manager, rid


def _step(runtime, manager, rid, value):
    runtime.cleanup_consumed_inputs("producer", [rid], [0])
    stored = manager.store_and_return_tensor_info_batch(
        [rid], {rid: {"token": [torch.tensor([value])]}}, ["token"],
    )
    manager.increment_ref_batch_uniform(stored.flat_uuids)
    result = runtime.complete_and_route_batch(RouteInput(
        partition="default", graph_walk="walk", node_name="producer", output_signals=["token"],
        wg_ids=ParallelList([rid], [0]), tensors=stored.flat_uuids, num_tensors=stored.num_tensors,
    ))
    return runtime._peek_completion(result.completion_id).routing[rid]


@pytest.mark.parametrize("accumulated", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("readers,persist", [(1, False), (2, False), (0, True)])
@pytest.mark.parametrize("explicit_stop", [False, True])
def test_loop_completion_transfers_cache_holds(tmp_path, accumulated, nested, readers, persist, explicit_stop):
    runtime, manager, rid = _runtime(
        tmp_path, accumulated=accumulated, nested=nested, readers=readers,
        persist=persist, limit=100 if explicit_stop else 2,
    )
    steps = 4 if nested else 2
    try:
        for frame in range(steps):
            if explicit_stop and frame == steps - 1:
                runtime.stop_loops_batched("default", "walk", "producer",
                                          ParallelList([rid], [["outer"]]))
            routing = _step(runtime, manager, rid, frame)
        edges = routing.persist if persist else routing.emit_to_client
        expected = ([1, 3] if nested else [0, 1]) if accumulated else [steps - 1]
        assert len(edges) == (1 if persist else readers)
        for edge in edges:
            assert [manager.get_tensor(i.uuid).item() for i in edge.tensor_info] == expected
        uuids = [i.uuid for i in edges[0].tensor_info]
        assert set(manager.tensor_store.get_all_uuids(rid)) == set(uuids)
        records = manager.tensor_store.bookkeeping._ref_info
        assert [records[u].ref_cnt for u in uuids] == [readers] * len(uuids)
        if persist:
            manager.cleanup_request(rid)
            assert set(manager.tensor_store.get_all_uuids(rid)) == set(uuids)
            for uuid in uuids:
                manager.tensor_store.set_metadata(uuid, persist=False)
            manager.cleanup_request(rid)
        else:
            for index, edge in enumerate(edges):
                manager.dereference_batch_uniform([i.uuid for i in edge.tensor_info])
                if index + 1 < readers:
                    assert [manager.get_tensor(u).item() for u in uuids] == expected
        assert manager.tensor_store.get_all_uuids(rid) == []
    finally:
        runtime.remove_request(rid)
        manager.force_cleanup_request(rid)


@pytest.mark.parametrize("nested", [False, True])
def test_abandoned_loop_hard_cleanup_reclaims_cached_outputs(tmp_path, nested):
    runtime, manager, rid = _runtime(tmp_path, accumulated=True, nested=nested,
                                    readers=1, persist=False, limit=100)
    for frame in range(3):
        _step(runtime, manager, rid, frame)
    assert manager.tensor_store.get_all_uuids(rid)
    runtime.remove_request(rid)
    manager.force_cleanup_request(rid)
    assert manager.tensor_store.get_all_uuids(rid) == []


@pytest.mark.parametrize("accumulated", [False, True])
@pytest.mark.parametrize("persist", [False, True])
def test_loop_finished_by_node_without_outputs_settles_older_holds(tmp_path, accumulated, persist):
    runtime, manager, rid = _runtime(tmp_path, accumulated=accumulated, nested=False,
                                    readers=1, persist=persist, limit=2, tail=True)
    try:
        for frame in range(2):
            runtime.cleanup_consumed_inputs("producer", [rid], [0])
            stored = manager.store_and_return_tensor_info_batch(
                [rid], {rid: {"token": [torch.tensor([frame])], "tick": [torch.tensor([0])]}},
                ["token", "tick"],
            )
            manager.increment_ref_batch_uniform(stored.flat_uuids)
            runtime.complete_and_route_batch(RouteInput(
                partition="default", graph_walk="walk", node_name="producer",
                output_signals=["token", "tick"], wg_ids=ParallelList([rid], [0]),
                tensors=stored.flat_uuids, num_tensors=stored.num_tensors,
            ))
            runtime.cleanup_consumed_inputs("tail", [rid], [0])
            result = runtime.complete_and_route_batch(RouteInput(
                partition="default", graph_walk="walk", node_name="tail",
                output_signals=[], wg_ids=ParallelList([rid], [0]), tensors=[], num_tensors=[],
            ))
        routing = runtime._peek_completion(result.completion_id).routing[rid]
        edges = routing.persist if persist else routing.emit_to_client
        assert len(edges) == 1
        uuids = [i.uuid for i in edges[0].tensor_info]
        assert [manager.get_tensor(u).item() for u in uuids] == ([0, 1] if accumulated else [1])
        records = manager.tensor_store.bookkeeping._ref_info
        assert [records[u].ref_cnt for u in uuids] == [int(not persist)] * len(uuids)
        if persist:
            for uuid in uuids:
                manager.tensor_store.set_metadata(uuid, persist=False)
            manager.cleanup_request(rid)
        else:
            manager.dereference_batch_uniform(uuids)
        assert manager.tensor_store.get_all_uuids(rid) == []
    finally:
        runtime.remove_request(rid)
        manager.force_cleanup_request(rid)


@pytest.mark.parametrize("accumulated", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_loop_output_remote_fanout_waits_for_each_reader(tmp_path, accumulated, nested):
    runtime, manager, rid = _runtime(tmp_path, accumulated=accumulated, nested=nested,
                                    readers=2, persist=False, limit=2, remote=True)
    try:
        for frame in range(4 if nested else 2):
            routing = _step(runtime, manager, rid, frame)
        assert set(routing.to_workers) == {"worker1", "worker2"}
        edges = [routing.to_workers[w][0] for w in ("worker1", "worker2")]
        uuids = [i.uuid for i in edges[0].tensor_info]
        assert [i.uuid for i in edges[1].tensor_info] == uuids
        expected = ([1, 3] if nested else [0, 1]) if accumulated else [3 if nested else 1]
        assert [manager.get_tensor(u).item() for u in uuids] == expected
        assert [manager.tensor_store.bookkeeping._ref_info[u].ref_cnt for u in uuids] == [2] * len(uuids)
        # One remote ACK must not reclaim the other reader's tensors.
        manager.dereference_batch_uniform(uuids)
        assert [manager.get_tensor(u).item() for u in uuids] == expected
        manager.dereference_batch_uniform(uuids)
        assert manager.tensor_store.get_all_uuids(rid) == []
    finally:
        runtime.remove_request(rid)
        manager.force_cleanup_request(rid)


@pytest.mark.parametrize("readers", [1, 2])
def test_finishing_parent_does_not_release_child_accumulation_before_routing(tmp_path, readers):
    runtime, manager, rid = _runtime(tmp_path, accumulated=False, nested=True,
                                    readers=readers, persist=False, limit=2, inner_accumulated=True)
    try:
        emitted = []
        for frame in range(4):
            runtime.cleanup_consumed_inputs("producer", [rid], [0])
            stored = manager.store_and_return_tensor_info_batch(
                [rid], {rid: {"token": [torch.tensor([frame])], "frame": [torch.tensor([frame])]}},
                ["token", "frame"],
            )
            manager.increment_ref_batch_uniform(stored.flat_uuids)
            result = runtime.complete_and_route_batch(RouteInput(
                partition="default", graph_walk="walk", node_name="producer",
                output_signals=["token", "frame"], wg_ids=ParallelList([rid], [0]),
                tensors=stored.flat_uuids, num_tensors=stored.num_tensors,
            ))
            edges = runtime._peek_completion(result.completion_id).routing[rid].emit_to_client
            if frame % 2:
                assert len(edges) == readers
                for edge in edges:
                    assert [manager.get_tensor(i.uuid).item() for i in edge.tensor_info] == [frame - 1, frame]
                    manager.dereference_batch_uniform([i.uuid for i in edge.tensor_info])
                emitted.extend([frame - 1, frame])
        assert emitted == [0, 1, 2, 3]
        assert manager.tensor_store.get_all_uuids(rid) == []
    finally:
        runtime.remove_request(rid)
        manager.force_cleanup_request(rid)
