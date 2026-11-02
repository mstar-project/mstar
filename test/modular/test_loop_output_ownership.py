"""Finished loops exchange cache ownership for routed readers exactly once.

Real runtime, SHM tensor store and cleanup; synthetic producer tensors.
Client releases model completed reads/ACKs, not actual network transport.
"""
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from rust_runtime import Inboxes, build_runtime

from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput, SendInput
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList


class _Sent(NamedTuple):
    """What one completion sent: one uuid list per edge."""
    persist: list[list[int]]
    emit_to_client: list[list[int]]
    to_workers: dict[str, list[list[int]]]


def _uuids(edge):
    return [i.uuid for i in edge.tensor_info]


def _send(runtime, inboxes, rid, result) -> _Sent:
    """Send a completion's frames and read them off the inboxes."""
    runtime.send_outputs(SendInput(
        completion_id=result.completion_id, per_request_info=ParallelList([rid], [None]),
        new_token_counts=ParallelList([rid], [{}]),
    ))
    sent = _Sent([], [], {})
    for peer, message in inboxes.take():
        if peer == "conductor":
            sent.persist.extend(
                [i.uuid for i in infos] for infos in message.body.persist_signals.values()
            )
        elif peer == "api_server":
            sent.emit_to_client.append(_uuids(message.body.graph_edge))
        else:
            sent.to_workers[peer] = [_uuids(e) for e in message.body.inputs]
    return sent


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
    inboxes = Inboxes(
        "worker0", tmp_path,
        peers=("conductor", "api_server", "worker1", "worker2") if remote else ("conductor", "api_server"),
    )
    runtime = build_runtime(
        bookkeeping=manager.tensor_store.bookkeeping, inboxes=inboxes,
        my_worker_id="worker0", my_worker_graphs=[graph],
        all_wg_ids_to_graph_walks=walks, all_wg_ids_to_nodes=nodes,
        all_wg_ids_to_dyn_loops=loops,
        node_to_partition=dict.fromkeys(section.get_nodes(), "default"),
        sharding_config=ShardingConfig(
            groups=[ShardingGroup(nodes={"consumer"}, tp_size=2)] if remote else [],
            tp_enabled_nodes={"consumer"} if remote else set(), shard_dim={},
        ),
    )
    caps = {name: loop.max_iters for name, loop in section.get_loops().items()}
    rid = runtime.add_request("request", "default", "walk", [0], mapping, caps)
    manager.register_request(rid, runtime.get_sharding_config(rid))
    block = ColumnarEdgeSpecs.empty()
    block.add_edge(rid, GraphEdge(name="token", next_node="producer"))
    assert runtime.ingest_inputs_batch(block) == []
    return runtime, manager, rid, inboxes


def _release_inputs(runtime, manager, rid, node="producer"):
    manager.cleanup_collectable(*runtime.cleanup_consumed_inputs(node, [rid], [0]))


def _route(runtime, manager, rid, inboxes, node, signals, stored):
    result = runtime.complete_and_route_batch(RouteInput(
        partition="default", graph_walk="walk", node_name=node, output_signals=signals,
        wg_ids=ParallelList([rid], [0]), tensors=stored, num_tensors=[1] * len(signals),
    ))
    manager.cleanup_collectable(*result.freed_inputs)
    return _send(runtime, inboxes, rid, result)


def _step(runtime, manager, rid, inboxes, value):
    _release_inputs(runtime, manager, rid)
    stored = manager.store_and_return_tensor_info_batch(
        [rid], {rid: {"token": [torch.tensor([value])]}}, ["token"],
    )
    manager.increment_ref_batch_uniform(stored.flat_uuids)
    return _route(runtime, manager, rid, inboxes, "producer", ["token"], stored.flat_uuids)


@pytest.mark.parametrize("accumulated", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("readers,persist", [(1, False), (2, False), (0, True)])
@pytest.mark.parametrize("explicit_stop", [False, True])
def test_loop_completion_transfers_cache_holds(tmp_path, accumulated, nested, readers, persist, explicit_stop):
    runtime, manager, rid, inboxes = _runtime(
        tmp_path, accumulated=accumulated, nested=nested, readers=readers,
        persist=persist, limit=100 if explicit_stop else 2,
    )
    steps = 4 if nested else 2
    try:
        for frame in range(steps):
            if explicit_stop and frame == steps - 1:
                runtime.stop_loops_batched("default", "walk", "producer",
                                          ParallelList([rid], [["outer"]]))
            sent = _step(runtime, manager, rid, inboxes, frame)
        edges = sent.persist if persist else sent.emit_to_client
        expected = ([1, 3] if nested else [0, 1]) if accumulated else [steps - 1]
        assert len(edges) == (1 if persist else readers)
        for edge in edges:
            assert [manager.get_tensor(u).item() for u in edge] == expected
        uuids = edges[0]
        assert set(manager.tensor_store.get_all_uuids(rid)) == set(uuids)
        book = manager.tensor_store.bookkeeping
        if persist:
            manager.cleanup_request(rid)
            assert set(manager.tensor_store.get_all_uuids(rid)) == set(uuids)
            for uuid in uuids:
                manager.tensor_store.set_metadata(uuid, persist=False)
            manager.cleanup_request(rid)
        else:
            for index, edge in enumerate(edges):
                assert not any(book.can_gc(u) for u in uuids), "each reader holds a reference"
                manager.dereference_batch_uniform(edge)
                if index + 1 < readers:
                    assert [manager.get_tensor(u).item() for u in uuids] == expected
        assert manager.tensor_store.get_all_uuids(rid) == []
    finally:
        runtime.remove_request(rid)
        manager.force_cleanup_request(rid)


@pytest.mark.parametrize("nested", [False, True])
def test_abandoned_loop_hard_cleanup_reclaims_cached_outputs(tmp_path, nested):
    runtime, manager, rid, inboxes = _runtime(tmp_path, accumulated=True, nested=nested,
                                              readers=1, persist=False, limit=100)
    for frame in range(3):
        _step(runtime, manager, rid, inboxes, frame)
    assert manager.tensor_store.get_all_uuids(rid)
    runtime.remove_request(rid)
    manager.force_cleanup_request(rid)
    assert manager.tensor_store.get_all_uuids(rid) == []


@pytest.mark.parametrize("accumulated", [False, True])
@pytest.mark.parametrize("persist", [False, True])
def test_loop_finished_by_node_without_outputs_settles_older_holds(tmp_path, accumulated, persist):
    runtime, manager, rid, inboxes = _runtime(tmp_path, accumulated=accumulated, nested=False,
                                              readers=1, persist=persist, limit=2, tail=True)
    try:
        for frame in range(2):
            _release_inputs(runtime, manager, rid)
            stored = manager.store_and_return_tensor_info_batch(
                [rid], {rid: {"token": [torch.tensor([frame])], "tick": [torch.tensor([0])]}},
                ["token", "tick"],
            )
            manager.increment_ref_batch_uniform(stored.flat_uuids)
            _route(runtime, manager, rid, inboxes, "producer", ["token", "tick"], stored.flat_uuids)
            _release_inputs(runtime, manager, rid, "tail")
            sent = _route(runtime, manager, rid, inboxes, "tail", [], [])
        edges = sent.persist if persist else sent.emit_to_client
        assert len(edges) == 1
        uuids = edges[0]
        assert [manager.get_tensor(u).item() for u in uuids] == ([0, 1] if accumulated else [1])
        book = manager.tensor_store.bookkeeping
        assert not any(book.can_gc(u) for u in uuids), "held by its reader, or by the persist mark"
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
    runtime, manager, rid, inboxes = _runtime(tmp_path, accumulated=accumulated, nested=nested,
                                              readers=2, persist=False, limit=2, remote=True)
    try:
        for frame in range(4 if nested else 2):
            sent = _step(runtime, manager, rid, inboxes, frame)
        assert set(sent.to_workers) == {"worker1", "worker2"}
        uuids, others = (sent.to_workers[w][0] for w in ("worker1", "worker2"))
        assert others == uuids
        expected = ([1, 3] if nested else [0, 1]) if accumulated else [3 if nested else 1]
        assert [manager.get_tensor(u).item() for u in uuids] == expected
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
    runtime, manager, rid, inboxes = _runtime(tmp_path, accumulated=False, nested=True,
                                              readers=readers, persist=False, limit=2, inner_accumulated=True)
    try:
        emitted = []
        for frame in range(4):
            _release_inputs(runtime, manager, rid)
            stored = manager.store_and_return_tensor_info_batch(
                [rid], {rid: {"token": [torch.tensor([frame])], "frame": [torch.tensor([frame])]}},
                ["token", "frame"],
            )
            manager.increment_ref_batch_uniform(stored.flat_uuids)
            edges = _route(
                runtime, manager, rid, inboxes, "producer", ["token", "frame"], stored.flat_uuids,
            ).emit_to_client
            if frame % 2:
                assert len(edges) == readers
                for edge in edges:
                    assert [manager.get_tensor(u).item() for u in edge] == [frame - 1, frame]
                    manager.dereference_batch_uniform(edge)
                emitted.extend([frame - 1, frame])
        assert emitted == [0, 1, 2, 3]
        assert manager.tensor_store.get_all_uuids(rid) == []
    finally:
        runtime.remove_request(rid)
        manager.force_cleanup_request(rid)
