"""``Worker._settle_chunks``: what a step of chunked rows routes, and what waits.

A row whose node needs more chunks is held: its outputs go to the
accumulator, it leaves the step unrouted, and its node goes back to the ready
queue. A final chunk takes the held outputs back, joined, and its whole input
is released. A step left with nothing to route ends postprocess early, but a
row the prefix cache served whole still has to be routed (#768).
"""

from types import SimpleNamespace

import torch

from mstar.worker.chunk_outputs import ChunkOutputAccumulator
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker

NODE, WALK = "LLM", "prefill"


class _Scheduler:
    def __init__(self):
        self.chunk_outputs = ChunkOutputAccumulator()
        self.advanced: list[tuple[int, str, int]] = []

    def advance_chunk(self, rid, node_name, end):
        self.advanced.append((rid, node_name, end))


class _Engine:
    def __init__(self):
        self.released: list[tuple[int, str]] = []

    def chunked_prefill_output_policies(self, node_name, graph_walk):
        return {}  # CONCAT on dim 0

    def release_chunk_inputs(self, rid, node_name):
        self.released.append((rid, node_name))


class _Runtime:
    def __init__(self):
        self.pushed_back: list[tuple[str, list[int], list[int]]] = []

    def push_back_node(self, node, rids, wg_ids):
        self.pushed_back.append((node, rids, wg_ids))


def _worker():
    w = Worker.__new__(Worker)
    w.scheduler = _Scheduler()
    w.engine_manager = SimpleNamespace(get_engine=lambda name: w._engine)
    w._engine = _Engine()
    w._graph_runtime = _Runtime()
    return w


def _pending(ranges: dict, incomplete: set, ran: list[int], cached=()):
    batch = ScheduledBatch(
        node_name=NODE, graph_walk=WALK,
        request_to_worker_graph={rid: 7 for rid in ran},
        chunk_ranges=ranges, incomplete_node_rids=set(incomplete),
    )
    node_batch = SimpleNamespace(
        request_ids=list(ran), per_request_info={rid: object() for rid in ran},
        cached_rids=set(cached),
    )
    return SimpleNamespace(batch=batch, node_name=NODE, node_batch=node_batch)


def _outputs(**per_rid):
    return SimpleNamespace(per_rid_outputs={
        int(rid[1:]): {"hidden": [torch.tensor(v)]} for rid, v in per_rid.items()
    })


def test_a_non_final_chunk_is_held_and_its_node_readied_again():
    w = _worker()
    pending = _pending({1: (0, 4, 10)}, incomplete={1}, ran=[1, 2])
    outputs = _outputs(r1=[1.0], r2=[9.0])

    assert w._settle_chunks(pending, outputs)

    assert 1 not in outputs.per_rid_outputs, "a held chunk routes nothing"
    assert pending.node_batch.request_ids == [2] and 1 not in pending.node_batch.per_request_info
    assert 1 not in pending.batch.request_to_worker_graph
    assert w._graph_runtime.pushed_back == [(NODE, [1], [7])]
    assert w.scheduler.advanced == [(1, NODE, 4)]
    assert w._engine.released == []


def test_the_final_chunk_routes_every_chunk_joined_and_releases_the_input():
    w = _worker()
    w._settle_chunks(_pending({1: (0, 4, 10)}, {1}, [1]), _outputs(r1=[1.0, 2.0]))
    final = _outputs(r1=[3.0])

    assert w._settle_chunks(_pending({1: (4, 10, 10)}, set(), [1]), final)

    assert final.per_rid_outputs[1]["hidden"][0].tolist() == [1.0, 2.0, 3.0]
    assert w._engine.released == [(1, NODE)]
    assert w.scheduler.advanced[-1] == (1, NODE, 10)


def test_a_step_of_only_held_chunks_has_nothing_to_route():
    w = _worker()

    assert not w._settle_chunks(_pending({1: (0, 4, 10)}, {1}, [1]), _outputs(r1=[1.0]))


def test_a_row_the_cache_served_is_still_routed_beside_held_chunks():
    w = _worker()
    pending = _pending({1: (0, 4, 10)}, {1}, [1], cached={2})

    assert w._settle_chunks(pending, _outputs(r1=[1.0]))


def test_a_row_dropped_from_the_step_is_left_alone():
    """A row the step lost (failed, removed) keeps no progress update."""
    w = _worker()
    pending = _pending({1: (0, 4, 10), 3: (0, 4, 10)}, {1, 3}, [1])

    w._settle_chunks(pending, _outputs(r1=[1.0]))

    assert w.scheduler.advanced == [(1, NODE, 4)]
