"""What happens to a batch whose admit refused it.

Admit failure means no forward ran, so the step's per-rid outputs are empty.
Two things follow, and neither used to be keyed off ``admit_error`` — they
read ``allocation_failed``, which was true only because ``AllocationFailed``
was the only reason a resource could give:

1. the engine must not run ``postprocess`` over those empty outputs, and
2. the worker must re-queue the batch, or the requests are never retried.

Only an ``AllocationFailed`` additionally calls for an eviction;
``RequestOffloading`` says the rid is already on its way to the host.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest

from mstar.engine.engine import Engine
from mstar.engine.resources import AllocationFailed, RequestOffloading
from mstar.worker.worker import Worker


def _alloc_failed(rid: str = "r0") -> AllocationFailed:
    return AllocationFailed(
        message="out of pages", pages_short=4, label="main", request_id=rid,
    )


def _offloading(rid: str = "r0") -> RequestOffloading:
    return RequestOffloading(
        message="being offloaded", label="main", request_id=rid,
    )


# --- engine: the tail must not run over a step that never ran --------------


class _FakeEngine:
    """Just enough engine to exercise the ``exec``/``postprocess`` branch."""

    exec_and_postprocess = Engine.exec_and_postprocess

    def __init__(self, admit_error):
        self._admit_error = admit_error
        self.postprocessed = False

    def exec(self, batch):
        batch.admit_error = self._admit_error
        # what `_exec_single` returns when admit refused the step
        return {rid: {} for rid in batch.request_ids}

    def postprocess_batch(self, batch, outputs):
        del batch, outputs
        self.postprocessed = True


def _batch():
    return SimpleNamespace(request_ids=["r0", "r1"], admit_error=None)


@pytest.mark.parametrize(
    "reason", [_alloc_failed(), _offloading()], ids=["allocation", "offloading"]
)
def test_postprocess_skipped_on_admit_error(reason):
    engine = _FakeEngine(reason)
    engine.exec_and_postprocess(_batch())
    assert engine.postprocessed is False


def test_postprocess_runs_on_a_step_that_admitted():
    engine = _FakeEngine(None)
    engine.exec_and_postprocess(_batch())
    assert engine.postprocessed is True


# --- worker: every admit failure is re-queued ------------------------------


class _Queue:
    def __init__(self):
        self.pushed_back: list[str] = []

    def push_back_node(self, request_id, node):
        del node
        self.pushed_back.append(request_id)


class _FakeWorker:
    """Binds the two handlers onto stubs for their collaborators."""

    _handle_admit_failure = Worker._handle_admit_failure

    def __init__(self):
        self.queue = _Queue()
        self.worker_graphs_manager = SimpleNamespace(queues={"wg": self.queue})
        self.held: list[str] = []
        self.scheduler = SimpleNamespace(hold_requests=self.held.extend)
        self.offload_calls: list[str] = []

    def _handle_allocation_failure(self, batch, node_batch):
        self.offload_calls.append(node_batch.node_name)
        # the real one push-backs and holds; stand in for both
        for rid in batch.node_objects:
            self.queue.push_back_node(rid, None)
        self.scheduler.hold_requests(list(batch.node_objects))


def _batches(reason):
    batch = SimpleNamespace(
        node_name="node", graph_walk="walk",
        node_objects={"r0": object(), "r1": object()},
        request_to_worker_graph={"r0": "wg", "r1": "wg"},
    )
    node_batch = SimpleNamespace(node_name="node", admit_error=reason)
    return batch, node_batch


def test_offloading_requeues_without_evicting():
    worker = _FakeWorker()
    worker._handle_admit_failure(*_batches(_offloading()))

    assert sorted(worker.queue.pushed_back) == ["r0", "r1"]
    # nothing to evict: the rid is already on its way to the host, and
    # `check_ready` gates the retry on reloading it
    assert worker.offload_calls == []
    assert worker.held == []


def test_allocation_failure_still_evicts():
    worker = _FakeWorker()
    worker._handle_admit_failure(*_batches(_alloc_failed()))

    assert worker.offload_calls == ["node"]
    # exactly one push-back per request: the delegation must not double up
    assert sorted(worker.queue.pushed_back) == ["r0", "r1"]
    assert sorted(worker.held) == ["r0", "r1"]
