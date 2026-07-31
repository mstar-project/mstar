"""Regression tests for StreamBuffer terminal-chunk emission.

A streaming consumer (e.g. the Zonos2 DAC vocoder) only runs its final
flush — emitting the withheld crossfade tail and closing the client stream —
when it receives a chunk marked ``is_final``. The bug this guards against:
when a producer finishes at a moment its consumer has already drained the
buffer to empty (which happens exactly when the total item count is a multiple
of the chunk size and the consumer kept up), no chunk was ever marked final,
so the request hung forever waiting on a terminal chunk that never came.

See ``test/zonos2/BUG_realtext_concurrency_hang.md``.
"""
import pytest
import torch

from mstar.streaming.chunk_policy import (
    FixedChunkPolicy,
    LeftContextChunkPolicy,
    SlidingWindowChunkPolicy,
)
from mstar.streaming.stream_buffer import StreamBuffer


def _drive(policy, total_frames, drain_before_done):
    """Feed ``total_frames`` one-frame items through a StreamBuffer and return
    the ``is_final`` flag of every popped chunk.

    ``drain_before_done`` models a consumer that keeps up with the producer,
    popping every ready chunk as items arrive — the timing that empties the
    buffer before ``signal_done`` and triggered the hang.
    """
    sb = StreamBuffer(
        request_id="r", edge_name="new_token",
        from_partition="LLM", policy=policy,
    )
    finals = []

    def poll():
        # Hard cap so a broken buffer that never terminates fails loudly
        # instead of hanging the test.
        for _ in range(total_frames + 50):
            if not sb.has_chunk_ready():
                return
            finals.append(sb.pop_chunk().is_final)
        raise AssertionError("has_chunk_ready never went False (spin/hang)")

    for i in range(total_frames):
        uid = f"t{i}"
        sb.pre_read_register(uid)
        sb.put(uid, torch.zeros(1, 4))
        if drain_before_done:
            poll()
    sb.signal_done()
    poll()
    return finals


@pytest.mark.parametrize("drain_before_done", [True, False])
@pytest.mark.parametrize(
    "total_frames",
    [0, 1, 29, 30, 31, 60, 90, 100],  # 30/60/90 are exact multiples of chunk
)
def test_fixed_policy_emits_exactly_one_final(total_frames, drain_before_done):
    finals = _drive(FixedChunkPolicy(chunk_size=30), total_frames, drain_before_done)
    # Exactly one terminal chunk, regardless of how the frame count lines up
    # with the chunk size or how eagerly the consumer drained.
    assert sum(finals) == 1, finals
    # The final chunk is the last one popped.
    assert finals[-1] is True, finals


@pytest.mark.parametrize("drain_before_done", [True, False])
@pytest.mark.parametrize("total_frames", [0, 7, 14, 28, 35, 56])
def test_sliding_window_policy_emits_exactly_one_final(total_frames, drain_before_done):
    finals = _drive(
        SlidingWindowChunkPolicy(window=28, stride=7), total_frames, drain_before_done
    )
    assert sum(finals) == 1, finals
    assert finals[-1] is True, finals


@pytest.mark.parametrize("drain_before_done", [True, False])
@pytest.mark.parametrize("total_frames", [0, 5, 10, 20, 40])
def test_left_context_policy_emits_exactly_one_final(total_frames, drain_before_done):
    finals = _drive(
        LeftContextChunkPolicy(chunk=10, left_context=2), total_frames, drain_before_done
    )
    assert sum(finals) == 1, finals
    assert finals[-1] is True, finals


def test_continue_after_producer_done_never_marks_final():
    """Policies that keep the consumer running past producer-done (e.g. the
    Qwen Thinker->Talker handoff) must never emit a final chunk — the consumer
    decides completion via its own model logic. The final-flush fix must not
    change that."""
    policy = FixedChunkPolicy(chunk_size=30, continue_after_done=True)
    sb = StreamBuffer(
        request_id="r", edge_name="new_token", from_partition="LLM", policy=policy,
    )
    for i in range(90):  # exact multiple — the trigger for the plain-policy hang
        uid = f"t{i}"
        sb.pre_read_register(uid)
        sb.put(uid, torch.zeros(1, 4))
        while sb.has_chunk_ready():
            assert sb.pop_chunk().is_final is False
    sb.signal_done()
    # Keeps offering (empty) chunks, none marked final.
    for _ in range(5):
        assert sb.has_chunk_ready() is True
        assert sb.pop_chunk().is_final is False
