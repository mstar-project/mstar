"""Registration of captured CUDA graph buckets.

A bucket's slots are double buffers of one shape: replay(N) runs on one while
pre-plan(N+1) writes the other. So a bucket is only usable with all of its
slots — registering a partial one would hand both to the same buffers, and
(because slots are appended in index order) would file the surviving slot
under a lower index than the one it captured under.

Capture failure is also a per-rank event, so the barrier count must not depend
on whether a capture succeeded.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.cuda_graph_runner import CudaGraphRunner
from mstar.engine.resources import BucketKey, CGSlotSpec

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="capture allocates a graph pool"
)


class _Barrier:
    def __init__(self):
        self.count = 0

    def barrier(self):
        self.count += 1


class _FakeRunner:
    """`warmup_and_capture` and `_register_slot` bound onto stubs."""

    warmup_and_capture = CudaGraphRunner.warmup_and_capture
    _register_slot = CudaGraphRunner._register_slot

    def __init__(self, specs, fail: set[tuple[str, int]] = frozenset(), num_slots=2):
        self._device = torch.device("cuda")
        self._submodule_name = "node"
        self._num_slots = num_slots
        self._specs = specs
        self._fail = fail
        self._buckets = {}
        self._memory_pool = None
        self.barrier = _Barrier()
        self._comm_group = SimpleNamespace(
            tp_group=self.barrier, sp_group=self.barrier
        )
        self.declared: list = []

    def prepare_for_capture(self):
        return self._specs

    def _capture_one(self, spec):
        if (spec.bucket.graph_walk, spec.slot) in self._fail:
            raise RuntimeError("capture failed")
        # stands in for the CudaGraphSlot; identity is what the test checks
        return f"{spec.bucket.graph_walk}:slot{spec.slot}"

    def _get_addtl_slot_specs(self, spec):
        del spec
        return []

    def declare_inputs_for(self, lease):
        self.declared.append(lease)
        return []

    def _log_memory(self, before, after):
        del before, after


def _specs(walks=("decode",), num_slots=2):
    out = []
    for walk in walks:
        bucket = BucketKey(graph_walk=walk, bs=1, num_tokens=8, cg_key_info=None)
        for slot in range(num_slots):
            out.append(CGSlotSpec(
                bucket=bucket, slot=slot, config=SimpleNamespace(), config_idx=0,
            ))
    return out


@requires_cuda
def test_a_fully_captured_bucket_registers_every_slot_in_index_order():
    runner = _FakeRunner(_specs())

    runner.warmup_and_capture()

    (bucket,) = runner._buckets.values()
    assert bucket.slots == ["decode:slot0", "decode:slot1"], (
        "list position must be the slot index"
    )


@requires_cuda
def test_a_bucket_missing_a_slot_is_dropped_whole():
    """The regression: slot 0 failing used to leave the bucket registered with
    slot 1's graph sitting at index 0, and only one slot to double-buffer on."""
    runner = _FakeRunner(_specs(), fail={("decode", 0)})

    runner.warmup_and_capture()

    assert runner._buckets == {}, "a half-captured bucket must not be usable"


@requires_cuda
def test_one_bucket_failing_does_not_take_the_others_with_it():
    runner = _FakeRunner(
        _specs(walks=("decode", "prefill")), fail={("decode", 1)},
    )

    runner.warmup_and_capture()

    assert [key.graph_walk for key in runner._buckets] == ["prefill"]
    (bucket,) = runner._buckets.values()
    assert bucket.slots == ["prefill:slot0", "prefill:slot1"]


@requires_cuda
def test_every_rank_barriers_once_per_spec_whatever_happens():
    """Capture can fail on one rank and not another; if the failing rank
    barriered fewer times the others would hang waiting for it."""
    clean = _FakeRunner(_specs(walks=("decode", "prefill")))
    failed = _FakeRunner(_specs(walks=("decode", "prefill")), fail={("decode", 0)})

    clean.warmup_and_capture()
    failed.warmup_and_capture()

    assert failed.barrier.count == clean.barrier.count == 2 * len(_specs(
        walks=("decode", "prefill")
    ))


@requires_cuda
def test_single_slot_runners_still_register():
    """No pre-planning resource means one slot per bucket, which is complete."""
    runner = _FakeRunner(_specs(num_slots=1), num_slots=1)

    runner.warmup_and_capture()

    (bucket,) = runner._buckets.values()
    assert bucket.slots == ["decode:slot0"]
    assert runner.declared == [], "nothing to pre-plan with a single slot"
