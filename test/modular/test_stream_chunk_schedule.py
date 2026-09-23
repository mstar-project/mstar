"""``ScheduledLeftContextChunkPolicy`` and the chunk geometry a StreamBuffer reports.

A streaming vocoder decodes each popped window and must drop the audio of the
leading ``context_items`` (frames an earlier chunk already delivered). These
tests pin (a) the window / context sequence of the ramped policy, (b) that every
item reaches the consumer exactly once as new data whatever the ramp, and (c)
that ``StreamChunk.context_items`` is right for every policy in the tree, so a
consumer can rely on it instead of re-deriving the policy's schedule.
"""

import pytest
import torch

from mstar.graph.base import GraphEdge
from mstar.streaming.chunk_policy import (
    FixedChunkPolicy,
    LeftContextChunkPolicy,
    ScheduledLeftContextChunkPolicy,
    SlidingWindowChunkPolicy,
)
from mstar.streaming.stream_buffer import StreamBuffer


def _drive(policy, total_items, drain_before_done=True):
    """Feed ``total_items`` one-row items; return (chunks, is_final flags)."""
    buffer = StreamBuffer(request_id="r", edge_name="codec_tokens", from_partition="Talker", policy=policy)
    chunks = []

    def poll():
        for _ in range(total_items + 50):
            if not buffer.has_chunk_ready():
                return
            chunks.append(buffer.pop_chunk())
        raise AssertionError("has_chunk_ready never went False")

    for i in range(total_items):
        buffer.pre_read_register(f"t{i}")
        buffer.put(f"t{i}", torch.tensor([i]))
        if drain_before_done:
            poll()
    buffer.signal_done()
    poll()
    return chunks


def _geometry(chunks):
    """(window, context, start_offset) per data-carrying chunk."""
    out = []
    for chunk in chunks:
        data = chunk.data["data"]
        if data is None:
            assert chunk.num_items == 0
            continue
        items = data.reshape(-1).tolist()
        assert chunk.num_items == len(items)
        out.append((len(items), chunk.context_items, chunk.start_offset, items))
    return out


def _new_items(chunks):
    delivered = []
    for window, context, _, items in _geometry(chunks):
        delivered.extend(items[context:window])
    return delivered


def test_scheduled_policy_ramps_windows_and_reports_context():
    policy = ScheduledLeftContextChunkPolicy(schedule=(4, 8, 16), chunk=25, left_context=25)
    chunks = _drive(policy, total_items=120)
    geometry = [(w, c, o) for w, c, o, _ in _geometry(chunks)]
    # window = context + new: 4 | 4+8 | 12+16 | 25+25 | 25+25 | flush
    assert geometry[:5] == [(4, 0, 0), (12, 4, 0), (28, 12, 0), (50, 25, 3), (50, 25, 28)]
    # The terminal flush hands over the remaining 17 new items behind 25 of context.
    assert geometry[-1] == (42, 25, 78)
    assert _new_items(chunks) == list(range(120))
    assert sum(chunk.is_final for chunk in chunks) == 1 and chunks[-1].is_final
    # Every window's leading context is exactly the items the previous chunk ended with.
    geometry_all = _geometry(chunks)
    for prev, curr in zip(geometry_all, geometry_all[1:], strict=False):
        assert curr[3][:curr[1]] == prev[3][len(prev[3]) - curr[1]:]


@pytest.mark.parametrize("drain_before_done", [True, False])
@pytest.mark.parametrize("total_items", [0, 1, 3, 4, 5, 12, 28, 53, 78, 100, 153])
def test_scheduled_policy_delivers_everything_once_with_one_final(total_items, drain_before_done):
    policy = ScheduledLeftContextChunkPolicy(schedule=(4, 8, 16), chunk=25, left_context=25)
    chunks = _drive(policy, total_items, drain_before_done)
    assert _new_items(chunks) == list(range(total_items))
    assert sum(chunk.is_final for chunk in chunks) == 1 and chunks[-1].is_final


def test_scheduled_policy_context_smaller_than_first_chunks():
    # Left context shorter than the ramp steps: context saturates at 2.
    policy = ScheduledLeftContextChunkPolicy(schedule=(1, 3), chunk=5, left_context=2)
    chunks = _drive(policy, total_items=14)
    assert [(w, c, o) for w, c, o, _ in _geometry(chunks)] == [
        (1, 0, 0), (4, 1, 0), (7, 2, 2), (7, 2, 7), (2, 2, 12),
    ]
    assert _new_items(chunks) == list(range(14))


def test_scheduled_policy_rejects_bad_sizes():
    with pytest.raises(ValueError):
        ScheduledLeftContextChunkPolicy(schedule=(0,), chunk=5, left_context=1)
    with pytest.raises(ValueError):
        ScheduledLeftContextChunkPolicy(schedule=(), chunk=5, left_context=-1)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (FixedChunkPolicy(chunk_size=3), [(3, 0), (3, 0), (3, 0), (1, 0)]),
        (SlidingWindowChunkPolicy(window=4, stride=2), [(4, 0), (4, 2), (4, 2), (4, 2), (2, 2)]),
        (LeftContextChunkPolicy(chunk=4, left_context=1), [(4, 0), (5, 1), (3, 1)]),
    ],
)
def test_existing_policies_report_context_items(policy, expected):
    chunks = _drive(policy, total_items=10)
    assert [(w, c) for w, c, _, _ in _geometry(chunks)] == expected
    assert _new_items(chunks) == list(range(10))


def test_graph_edge_clone_keeps_stream_chunk_geometry():
    edge = GraphEdge(next_node="Codec", name="codec_tokens", _final_stream_chunk=True,
                     _stream_chunk_offset=7, _stream_chunk_context=3, _stream_chunk_items=12)
    clone = edge.clone()
    assert (clone._stream_chunk_offset, clone._stream_chunk_context, clone._stream_chunk_items,
            clone._final_stream_chunk) == (7, 3, 12, True)
    plain = GraphEdge(next_node="x", name="y")
    assert plain._stream_chunk_context is None and plain._stream_chunk_items is None
