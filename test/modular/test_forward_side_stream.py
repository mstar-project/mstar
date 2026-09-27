"""The forward runs off the default stream, and everyone orders against it.

Under TCP/RDMA a rank sitting on a collective on the DEFAULT stream blocks its
peers from reading out of its tensor store, so a rank that reaches its forward
first can fail a peer's read. Moving the forward to a side stream fixes that,
and the whole cost is that the default stream stops being the thing to order
against: the H2D of a step's inputs, a KV reload and the events that fence a
peer's read are all issued from the scheduler thread, whose current stream is
NOT the one running attention.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.utils import cuda_streams
from mstar.worker.worker import Worker

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


@pytest.fixture(autouse=True)
def _restore_declared_stream():
    """The declared stream is a module global; leaking it would repoint every
    later test's ordering at a stream from this one."""
    before = cuda_streams._compute_stream
    yield
    cuda_streams.set_compute_stream(before)


def _worker(stream) -> SimpleNamespace:
    """Just enough Worker to call ``_on_compute_stream``."""
    return SimpleNamespace(
        _compute_stream=stream,
        _on_compute_stream=Worker._on_compute_stream.__get__(
            SimpleNamespace(_compute_stream=stream)
        ),
    )


def test_undeclared_falls_back_to_the_calling_threads_stream():
    """Every non-worker process (the API server, these tests) has to keep its
    old default-stream behaviour."""
    cuda_streams.set_compute_stream(None)
    if torch.cuda.is_available():
        # ``==``, not ``is``: current_stream() hands back a fresh wrapper each call
        assert cuda_streams.compute_stream() == torch.cuda.current_stream()


@cuda_only
def test_declared_stream_wins_over_the_calling_threads_stream():
    """The property the ordering rests on: a reload or a fence issued from the
    scheduler thread names the forward's stream, not its own."""
    forward = torch.cuda.Stream()
    cuda_streams.set_compute_stream(forward)

    seen: list[torch.cuda.Stream] = []
    other = threading.Thread(target=lambda: seen.append(cuda_streams.compute_stream()))
    other.start()
    other.join()

    assert cuda_streams.compute_stream() is forward
    assert seen == [forward], (
        "a thread that is not the GPU thread resolved to its own stream, so "
        "its wait_stream/record_event would miss the forward entirely"
    )


@cuda_only
def test_the_forward_stream_is_not_the_default_one():
    forward = torch.cuda.Stream()
    cuda_streams.set_compute_stream(forward)
    assert cuda_streams.compute_stream() != torch.cuda.default_stream(), (
        "the whole point is to get the forward's collectives off the default "
        "stream, where they block peers' tensor-store reads"
    )


@cuda_only
def test_the_block_redirects_bare_current_stream_calls():
    """The engine's own event records and syncs say ``current_stream()`` without
    naming anything. They have to land on the forward's stream."""
    forward = torch.cuda.Stream()
    worker = _worker(forward)

    outside = torch.cuda.current_stream()
    with worker._on_compute_stream():
        assert torch.cuda.current_stream() == forward
    assert torch.cuda.current_stream() == outside, (
        "the GPU thread is pooled and reused; a leaked stream would put the "
        "next step's work on the wrong one"
    )


def test_no_stream_declared_leaves_the_thread_alone():
    """CPU workers and MSTAR_FORWARD_SIDE_STREAM=0 keep running exactly as
    before, with no stream context at all."""
    worker = _worker(None)
    with worker._on_compute_stream():
        pass  # a nullcontext; nothing to assert beyond it not raising
