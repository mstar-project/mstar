"""A chunked prefill can stream its non-final chunks early, flagged as not
finishing the producer's walk, to a consumer whose ChunkPolicy opted into
partial input. A chunk of only such items does not finish the consumer's walk."""
import torch

from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer


class _Partial(FixedChunkPolicy):
    def allow_partial_input(self) -> bool:
        return True


def _buffer(policy, flags):
    buf = StreamBuffer(request_id=0, edge_name="states", from_partition="thinker", policy=policy)
    for i, finished in enumerate(flags):
        buf.pre_read_register(f"t{i}", finished)
        buf.put(f"t{i}", torch.tensor([float(i)]))
    return buf


def test_a_partial_consumer_takes_items_as_they_arrive():
    buf = _buffer(_Partial(1), [False, True])

    first, second = buf.pop_chunk(), buf.pop_chunk()

    assert (first.finished_graph_walk, second.finished_graph_walk) == (False, True)
    assert first.info.finished_graph_walk is False


def test_a_chunk_holding_the_finishing_item_finishes_the_walk():
    buf = _buffer(FixedChunkPolicy(3), [False, False, True])

    assert buf.pop_chunk().finished_graph_walk


def test_an_unflagged_item_finishes_its_walk():
    buf = StreamBuffer(request_id=0, edge_name="s", from_partition="p", policy=FixedChunkPolicy(1))
    buf.pre_read_register("t0")
    buf.put("t0", torch.tensor([0.0]))

    assert buf.has_chunk_ready() and buf.pop_chunk().finished_graph_walk


def test_a_step_on_a_partial_chunk_does_not_finish_its_walk():
    from mstar.engine.engine import ExecutingBatch
    from mstar.model.submodule_base import InputMetadata
    from mstar.streaming.stream_buffer import StreamChunkInfo

    partial = StreamChunkInfo(0, 0, 1, False, finished_graph_walk=False)
    batch = ExecutingBatch(
        node_name="Talker", per_request_info={0: None, 1: None}, step_context=None,
        per_request_input_metadata={0: InputMetadata(stream_chunks={"states": partial})},
    )

    assert (batch.completes_walk(0), batch.completes_walk(1)) == (False, True)
