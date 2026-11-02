"""Joining a chunked producer's pieces on arrival.

A node whose prefill is chunked emits one piece per chunk, but a consumer's
ChunkPolicy counts ITEMS -- so to a FixedChunkPolicy(1) the pieces look like
separate items and it reads only the first. Qwen3-Omni's Talker hit exactly
that: four prefill chunks, and it was handed the first one, which happened to
be empty because the prompt opened with a system span the producer drops.

Joined on arrival rather than at the producer: the consumer owns these tensors
outright, so there is no cross-stream ordering to get wrong.
"""

import sys

sys.path.insert(0, ".")

import torch

from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer


def _buf(join_dim):
    return StreamBuffer(
        request_id=1, edge_name="thinker_states", from_partition="Thinker",
        policy=FixedChunkPolicy(chunk_size=1, continue_after_done=True),
        join_dim=join_dim,
    )


def _feed(buf, pieces, joined=True):
    ids = [f"t{i}" for i in range(len(pieces))]
    if joined:
        buf.pre_read_register_joined(ids)
    else:
        for i in ids:
            buf.pre_read_register(i)
    for i, piece in zip(ids, pieces, strict=True):
        buf.put(i, piece)
    return buf


def test_the_pieces_arrive_as_one_item():
    buf = _feed(_buf(0), [torch.ones(4, 8), torch.ones(6, 8) * 2])
    assert buf.has_chunk_ready()
    data = buf.pop_chunk().data["data"]
    assert data.shape == (10, 8)
    assert data[:4].eq(1).all() and data[4:].eq(2).all(), "order must be kept"


def test_an_empty_piece_still_counts():
    """A prompt chunk can be all system prompt, which the producer drops every
    position of -- the piece is real and its neighbours must still join."""
    buf = _feed(_buf(0), [torch.zeros(0, 8), torch.ones(5, 8)])
    assert buf.pop_chunk().data["data"].shape == (5, 8)


def test_one_piece_is_the_tensor_itself():
    buf = _feed(_buf(0), [torch.ones(3, 8)])
    assert buf.pop_chunk().data["data"].shape == (3, 8)


def test_a_1d_edge_joins_too():
    """``thinker_mask`` reaches the Talker 1-D: the producer flattens it."""
    buf = _feed(_buf(0), [torch.ones(4), torch.zeros(6)])
    assert buf.pop_chunk().data["data"].shape == (10,)


def test_without_a_join_dim_the_pieces_stay_separate():
    """A consumer that wants them apart gets what it got before."""
    buf = _feed(_buf(None), [torch.ones(4, 8), torch.ones(6, 8)])
    assert buf.pop_chunk().data["data"].shape == (4, 8), "one item per piece"


def test_the_consumer_sees_one_item_however_many_pieces():
    """What the ChunkPolicy counts is buffered items, so the pieces have to
    become one. The producer-unit counters are a separate concern (see
    test_a_joined_group_counts_in_the_producers_units)."""
    joined, apart = _buf(0), _buf(None)
    pieces = [torch.ones(2, 8)] * 4
    _feed(joined, pieces)
    _feed(apart, pieces, joined=False)
    joined._update_buffer()
    apart._update_buffer()
    assert len(joined._buffer) == 1
    assert len(apart._buffer) == 4


def test_a_joined_group_counts_in_the_producers_units():
    """The producer reports one item per piece in ``signal_done``, so the
    buffer's own counters have to match it. Counting a group as one leaves it
    permanently short of the producer's total and it never agrees the producer
    is finished -- the consumer then runs on a stream that never closes."""
    buf = _feed(_buf(0), [torch.ones(2, 8)] * 4)
    buf.signal_done(num_items=4)
    assert buf.has_chunk_ready()
    assert buf._producer_done_and_all_read(), "the stream must be able to close"
    assert buf.pop_chunk().data["data"].shape == (8, 8), "still one item"
