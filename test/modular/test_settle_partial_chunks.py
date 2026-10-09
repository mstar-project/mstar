"""A chunked prefill's non-final chunk, when a consumer opted into partial
input: its streamed outputs are routed now and the rest wait for the last
chunk (``Worker._settle_chunks``)."""
from types import SimpleNamespace

import torch

from mstar.model.submodule_base import BatchedModelOutput
from mstar.worker.chunk_outputs import ChunkOutputAccumulator
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker

NODE = "thinker"


def _worker(partial_edges):
    pushed = []
    worker = SimpleNamespace(
        _partial_stream_edges=frozenset(partial_edges),
        scheduler=SimpleNamespace(
            advance_chunk=lambda rid, node, end: None,
            chunk_outputs=ChunkOutputAccumulator(),
        ),
        engine_manager=SimpleNamespace(get_engine=lambda node: SimpleNamespace(
            chunked_prefill_output_policies=lambda node, walk: {},
        )),
        _graph_runtime=SimpleNamespace(
            push_back_node=lambda node, rids, wgs: pushed.extend(rids),
        ),
    )
    return worker, pushed


def _pending():
    batch = ScheduledBatch(
        node_name=NODE, graph_walk="prefill",
        request_to_worker_graph={0: 0, 1: 0},
    )
    batch.chunk_ranges = {0: (0, 4)}
    batch.incomplete_node_rids = {0}
    node_batch = SimpleNamespace(request_ids=[0, 1], per_request_info={0: "i0", 1: "i1"})
    return SimpleNamespace(batch=batch, node_name=NODE, node_batch=node_batch)


def _outputs():
    return BatchedModelOutput(per_rid_outputs={
        0: {"states": [torch.tensor([1.0])], "text": [torch.tensor([2.0])]},
        1: {"states": [torch.tensor([3.0])], "text": [torch.tensor([4.0])]},
    })


def _settle(worker, pending, outputs):
    return Worker._settle_chunks(worker, pending, outputs)


def test_an_opted_in_stream_is_routed_now_and_the_rest_waits():
    worker, pushed = _worker({"states"})
    pending, outputs = _pending(), _outputs()

    assert _settle(worker, pending, outputs)

    assert set(outputs.per_rid_outputs[0]) == {"states"}, "text waits for the last chunk"
    assert 0 in pending.batch.request_to_worker_graph, "the row stays to route its stream"
    assert pending.node_batch.request_ids == [1], "and never reaches the stop check"
    assert pending.node_batch.per_request_info == {0: "i0", 1: "i1"}
    assert pushed == [0]
    held = worker.scheduler.chunk_outputs.release(0, NODE, {})
    assert set(held) == {"text"}


def test_a_row_with_no_opted_in_stream_is_held_whole():
    worker, pushed = _worker(set())
    pending, outputs = _pending(), _outputs()

    assert _settle(worker, pending, outputs)

    assert 0 not in outputs.per_rid_outputs
    assert 0 not in pending.batch.request_to_worker_graph
    assert pending.node_batch.per_request_info == {1: "i1"}
    assert pushed == [0]


def test_a_batch_left_with_only_partial_rows_is_still_routed():
    worker, _pushed = _worker({"states"})
    pending, outputs = _pending(), _outputs()
    pending.batch.discard_rid(1)
    pending.node_batch.request_ids = [0]

    assert _settle(worker, pending, outputs), "its stream still has to go out"
    assert pending.node_batch.request_ids == []


def test_completion_flags_name_only_the_unfinished_rows():
    pending = _pending()
    pending.node_batch.incomplete_walk_rids = set()
    assert Worker._completion_flags(pending, [0, 1]) == ([False, True], None)
    pending.batch.incomplete_node_rids = set()
    pending.node_batch.incomplete_walk_rids = {1}
    assert Worker._completion_flags(pending, [0, 1]) == (None, [True, False])
