"""Both runtimes release transferred loop-cache holds after the final reader.

Synthetic tensors, real runtime/bookkeeping; no GPU or transport execution.
"""
import pytest
import torch

pytest.importorskip("mstar_rust", reason="ownership checks require the freshly built extension")

from test_runtime_lockstep import WORKER, _build, _finish_teardown, _info  # noqa: E402

from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential  # noqa: E402
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput  # noqa: E402
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION  # noqa: E402
from mstar.utils.containers import ParallelList  # noqa: E402


def _reported(store, uuids):
    """What a cleanup dereference that frees ``uuids`` returns: a store that
    frees in place reports only what needs the transport's teardown, which
    none of these synthetic tensors do."""
    return [] if store.frees_in_place else uuids


def _caps(section):
    """What the conductor sends when a request names no caps: every loop at its max_iters."""
    return {name: loop.max_iters for name, loop in section.get_loops().items()}


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("accumulated", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("readers,persist", [(1, False), (2, False), (0, True)])
def test_finished_loop_ownership(backend, accumulated, nested, readers, persist):
    section = GraphNode(name="producer", input_names=["token"],
                        outputs=[GraphEdge(name="token", next_node="producer")])
    if nested:
        section = Loop(name="inner", section=section, max_iters=2,
                       outputs=[GraphEdge(name="token", next_node="producer")])
    edges = ([GraphEdge(name="token", next_node=EMPTY_DESTINATION, persist=True)] if persist else
             [GraphEdge(name="token", next_node=EMIT_TO_CLIENT) for _ in range(readers)])
    section = Loop(name="outer", section=section, max_iters=2,
                   outputs=[] if accumulated else edges, accumulated_outputs=edges if accumulated else [])
    runtime, book, store = _build(backend, section)
    rid = runtime.add_request("ownership", "default", "decode", [0], ParallelList([0], [[WORKER]]), _caps(section))
    block = ColumnarEdgeSpecs.empty()
    block.add_edge(rid, GraphEdge(name="token", next_node="producer"))
    assert runtime.ingest_inputs_batch(block) == []
    steps = 4 if nested else 2
    for frame in range(steps):
        _finish_teardown(store, runtime.cleanup_consumed_inputs("producer", [rid], [0]))
        uuid = frame + 100
        store.put_tensor(rid, uuid, torch.tensor([frame]), _info(uuid))
        store.increment_ref(uuid)
        result = runtime.complete_and_route_batch(RouteInput(
            partition="default", graph_walk="decode", node_name="producer", output_signals=["token"],
            wg_ids=ParallelList([rid], [0]), tensors=[uuid], num_tensors=[1],
        ))
        _finish_teardown(store, result.freed_inputs)
    expected = ([101, 103] if nested else [100, 101]) if accumulated else [99 + steps]
    assert sorted(result.register_uuids) == ([] if persist else expected)
    for uuid in range(100, 100 + steps):
        if uuid not in expected:
            assert book.get_info(uuid) is None or book.can_gc(uuid)
    for uuid in expected:
        assert book.get_info(uuid) is not None
        assert not book.can_gc(uuid)
    if persist:
        for uuid in expected:
            book.set_persist(uuid, False)
            assert book.can_gc(uuid), "persistence must not conceal a leftover cache hold"
    else:
        for reader in range(readers):
            freed, _ = store.dereference_batch_uniform(expected, cleanup=True)
            if reader + 1 < readers:
                assert freed == []
                assert all(book.get_info(u) is not None and not book.can_gc(u) for u in expected)
            else:
                assert sorted(freed) == _reported(store, expected)
                assert all(book.get_info(u) is None for u in expected)
    runtime.remove_request(rid)


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("readers", [1, 2])
def test_child_accumulation_survives_parent_finish(backend, readers):
    node = GraphNode(name="producer", input_names=["token"], outputs=[
        GraphEdge(name="token", next_node="producer"), GraphEdge(name="frame", next_node=EMPTY_DESTINATION),
    ])
    inner = Loop(name="inner", section=node, max_iters=2,
                 outputs=[GraphEdge(name="token", next_node="producer")],
                 accumulated_outputs=[GraphEdge(name="frame", next_node=EMIT_TO_CLIENT) for _ in range(readers)])
    outer = Loop(name="outer", section=inner, max_iters=2, outputs=[])
    runtime, book, store = _build(backend, outer)
    rid = runtime.add_request("child", "default", "decode", [0], ParallelList([0], [[WORKER]]), _caps(outer))
    block = ColumnarEdgeSpecs.empty()
    block.add_edge(rid, GraphEdge(name="token", next_node="producer"))
    assert runtime.ingest_inputs_batch(block) == []
    for frame in range(4):
        _finish_teardown(store, runtime.cleanup_consumed_inputs("producer", [rid], [0]))
        for uuid in (100 + frame, 200 + frame):
            store.put_tensor(rid, uuid, torch.tensor([frame]), _info(uuid))
            store.increment_ref(uuid)
        result = runtime.complete_and_route_batch(RouteInput(
            partition="default", graph_walk="decode", node_name="producer", output_signals=["token", "frame"],
            wg_ids=ParallelList([rid], [0]), tensors=[100 + frame, 200 + frame], num_tensors=[1, 1],
        ))
        _finish_teardown(store, result.freed_inputs)
        if frame % 2:
            expected = [199 + frame, 200 + frame]
            assert sorted(result.register_uuids) == expected
            assert all(book.get_info(u) is not None for u in expected)
            for reader in range(readers):
                freed, _ = store.dereference_batch_uniform(expected, cleanup=True)
                assert sorted(freed) == (_reported(store, expected) if reader + 1 == readers else [])
    assert all(book.get_info(u) is None for u in range(200, 204))
    runtime.remove_request(rid)


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("accumulated", [False, True])
@pytest.mark.parametrize("persist", [False, True])
def test_loop_finished_by_node_without_outputs_settles_older_holds(backend, accumulated, persist):
    """The node that finishes the loop produces nothing, so only handed-over loop holds are settled."""
    producer = GraphNode(name="producer", input_names=["token"], outputs=[
        GraphEdge(name="token", next_node="producer"), GraphEdge(name="tick", next_node="tail"),
    ])
    body = Sequential(sections=[producer, GraphNode(name="tail", input_names=["tick"], outputs=[])])
    edges = ([GraphEdge(name="token", next_node=EMPTY_DESTINATION, persist=True)] if persist else
             [GraphEdge(name="token", next_node=EMIT_TO_CLIENT)])
    loop = Loop(name="outer", section=body, max_iters=2,
                outputs=[] if accumulated else edges, accumulated_outputs=edges if accumulated else [])
    runtime, book, store = _build(backend, loop)
    rid = runtime.add_request("tail", "default", "decode", [0], ParallelList([0], [[WORKER]]), _caps(loop))
    block = ColumnarEdgeSpecs.empty()
    block.add_edge(rid, GraphEdge(name="token", next_node="producer"))
    assert runtime.ingest_inputs_batch(block) == []
    for frame in range(2):
        _finish_teardown(store, runtime.cleanup_consumed_inputs("producer", [rid], [0]))
        for uuid in (100 + frame, 200 + frame):
            store.put_tensor(rid, uuid, torch.tensor([frame]), _info(uuid))
            store.increment_ref(uuid)
        result = runtime.complete_and_route_batch(RouteInput(
            partition="default", graph_walk="decode", node_name="producer", output_signals=["token", "tick"],
            wg_ids=ParallelList([rid], [0]), tensors=[100 + frame, 200 + frame], num_tensors=[1, 1],
        ))
        _finish_teardown(store, result.freed_inputs)
        _finish_teardown(store, runtime.cleanup_consumed_inputs("tail", [rid], [0]))
        result = runtime.complete_and_route_batch(RouteInput(
            partition="default", graph_walk="decode", node_name="tail", output_signals=[],
            wg_ids=ParallelList([rid], [0]), tensors=[], num_tensors=[],
        ))
        _finish_teardown(store, result.freed_inputs)
    expected = [100, 101] if accumulated else [101]
    assert sorted(result.register_uuids) == ([] if persist else expected)
    for uuid in expected:
        assert book.get_info(uuid) is not None
        assert not book.can_gc(uuid)
    if persist:
        for uuid in expected:
            book.set_persist(uuid, False)
            assert book.can_gc(uuid), "persistence must not conceal a leftover cache hold"
    else:
        freed, _ = store.dereference_batch_uniform(expected, cleanup=True)
        assert sorted(freed) == _reported(store, expected)
        assert all(book.get_info(u) is None for u in expected)
    runtime.remove_request(rid)


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("nested", [False, True])
def test_abandoned_loop_keeps_its_holds_until_hard_cleanup(backend, nested):
    """Removing a request mid-loop releases nothing early or twice; hard cleanup then reclaims everything."""
    section = GraphNode(name="producer", input_names=["token"],
                        outputs=[GraphEdge(name="token", next_node="producer")])
    if nested:
        section = Loop(name="inner", section=section, max_iters=2,
                       outputs=[GraphEdge(name="token", next_node="producer")])
    loop = Loop(name="outer", section=section, max_iters=100, outputs=[],
                accumulated_outputs=[GraphEdge(name="token", next_node=EMIT_TO_CLIENT)])
    runtime, book, store = _build(backend, loop)
    rid = runtime.add_request("abandoned", "default", "decode", [0], ParallelList([0], [[WORKER]]), _caps(loop))
    block = ColumnarEdgeSpecs.empty()
    block.add_edge(rid, GraphEdge(name="token", next_node="producer"))
    assert runtime.ingest_inputs_batch(block) == []
    produced = [100 + frame for frame in range(3)]
    for uuid in produced:
        _finish_teardown(store, runtime.cleanup_consumed_inputs("producer", [rid], [0]))
        store.put_tensor(rid, uuid, torch.tensor([uuid]), _info(uuid))
        store.increment_ref(uuid)
        result = runtime.complete_and_route_batch(RouteInput(
            partition="default", graph_walk="decode", node_name="producer", output_signals=["token"],
            wg_ids=ParallelList([rid], [0]), tensors=[uuid], num_tensors=[1],
        ))
        _finish_teardown(store, result.freed_inputs)
    runtime.remove_request(rid)
    # Every tensor is either already freed or still referenced (by the unfinished loop's cache).
    assert any(book.get_info(u) is not None for u in produced)
    assert all(book.get_info(u) is None or not book.can_gc(u) for u in produced)
    for uuid in store.get_all_uuids(rid):  # what the tensor manager's hard cleanup does
        store.remove_tensor(uuid)
    assert store.get_all_uuids(rid) == []
    assert all(book.get_info(u) is None for u in produced)
