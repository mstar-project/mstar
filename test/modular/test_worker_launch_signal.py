"""The submitter's launch signal: set by the engine at the replay launch by
default, or by the gpu thread after commit and staging when the knob is on."""

import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mstar.model.submodule_base import BatchedModelOutput  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402


class _Engine:
    def __init__(self):
        self.seen_event = "unset"

    def exec_and_postprocess(self, node_batch):
        # what the engine would hand to the replay as its launch signal
        self.seen_event = node_batch.launch_started_event
        return BatchedModelOutput()


class _Span:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _worker(deferred):
    w = object.__new__(Worker)
    w._launch_signal_after_commit = deferred
    w._span = lambda name: _Span()
    return w


def test_by_default_the_engine_gets_the_event_to_set_at_the_launch():
    engine = _Engine()
    event = threading.Event()
    batch = SimpleNamespace(launch_started_event=event, completion_event=None)
    Worker._exec_step(_worker(False), engine, batch, None)
    assert engine.seen_event is event
    assert not event.is_set(), "the stub engine never launched, so nobody set it"


def test_with_the_knob_the_gpu_thread_signals_after_the_step():
    engine = _Engine()
    event = threading.Event()
    batch = SimpleNamespace(launch_started_event=event, completion_event=None)
    Worker._exec_step(_worker(True), engine, batch, None)
    assert engine.seen_event is None, "the engine must not signal at the launch"
    assert event.is_set() and batch.launch_started_event is event


def test_a_raising_step_still_releases_the_submitter():
    class _Raising(_Engine):
        def exec_and_postprocess(self, node_batch):
            raise RuntimeError("forward failed")

    event = threading.Event()
    batch = SimpleNamespace(launch_started_event=event, completion_event=None)
    with pytest.raises(RuntimeError):
        Worker._exec_step(_worker(True), _Raising(), batch, None)
    assert event.is_set() and batch.launch_started_event is event
