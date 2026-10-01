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

from mstar.engine.cuda_graph_runner import (
    CudaGraphRunner,
    capture_into_graph,
    fail_if_graphs_required,
)
from mstar.engine.resources import BucketKey, CGSlotSpec

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="capture allocates a graph pool"
)


class _Group:
    """A one-rank comm group, or a two-rank one whose peer's flags are given."""

    def __init__(self, peer_flags: list[bool] | None = None):
        self.count = 0
        self.world_size = 1 if peer_flags is None else 2
        self._peer = peer_flags

    def barrier(self):
        self.count += 1

    def all_gather(self, tensor, dim=0):
        del dim
        peer = torch.tensor(self._peer, dtype=tensor.dtype, device=tensor.device)
        # rank-major, matching CommGroup.all_gather
        return torch.cat([tensor, peer])


class _FakeRunner:
    """`warmup_and_capture` and `_register_slot` bound onto stubs."""

    warmup_and_capture = CudaGraphRunner.warmup_and_capture
    _register_slot = CudaGraphRunner._register_slot
    _buckets_captured_everywhere = CudaGraphRunner._buckets_captured_everywhere
    _report_dropped = CudaGraphRunner._report_dropped

    def __init__(
        self, specs, fail: set[tuple[str, int]] = frozenset(), num_slots=2,
        peer_flags: list[bool] | None = None,
    ):
        self._device = torch.device("cuda")
        self._submodule_name = "node"
        self._num_slots = num_slots
        self._specs = specs
        self._fail = fail
        self._buckets = {}
        self._memory_pool = None
        self.barrier = _Group(peer_flags)
        self._comm_group = SimpleNamespace(
            tp_group=self.barrier, sp_group=self.barrier
        )
        self.declared: list = []
        self._dummy_rows = SimpleNamespace(
            released=False,
            release_all=lambda: setattr(self._dummy_rows, "released", True),
        )

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
def test_a_bucket_another_rank_dropped_is_dropped_here_too():
    """Capture failure is per-rank. If this rank kept a bucket the peer
    dropped, it would lease and replay while the peer ran eager — and a
    captured region holding a collective hangs on the mismatch."""
    runner = _FakeRunner(
        _specs(walks=("decode", "prefill")),
        # local captures both; the peer failed the second bucket
        peer_flags=[True, False],
    )

    runner.warmup_and_capture()

    assert [key.graph_walk for key in runner._buckets] == ["decode"]


@requires_cuda
def test_a_bucket_this_rank_dropped_stays_dropped_when_the_peer_kept_it():
    runner = _FakeRunner(
        _specs(walks=("decode", "prefill")),
        fail={("decode", 1)},
        peer_flags=[True, True],
    )

    runner.warmup_and_capture()

    assert [key.graph_walk for key in runner._buckets] == ["prefill"]


@requires_cuda
def test_ranks_agree_on_the_full_candidate_list_not_just_local_successes():
    """The reduced vector is ordered by the configs, which every rank shares,
    so a rank that captured nothing still lines its flags up with the rest."""
    runner = _FakeRunner(
        _specs(walks=("decode", "prefill")),
        fail={("decode", 0), ("decode", 1), ("prefill", 0), ("prefill", 1)},
        peer_flags=[True, True],
    )

    runner.warmup_and_capture()

    assert runner._buckets == {}


@requires_cuda
def test_capture_hands_the_padding_rows_pages_back():
    """A capture gives its padding rows real spans; a replay pads with
    zero-length ones, so that storage is residue the traffic should get."""
    runner = _FakeRunner(_specs())

    runner.warmup_and_capture()

    assert runner._dummy_rows.released


@requires_cuda
def test_single_slot_runners_still_register():
    """No pre-planning resource means one slot per bucket, which is complete."""
    runner = _FakeRunner(_specs(num_slots=1), num_slots=1)

    runner.warmup_and_capture()

    (bucket,) = runner._buckets.values()
    assert bucket.slots == ["decode:slot0"]
    assert runner.declared == [], "nothing to pre-plan with a single slot"


@requires_cuda
def test_a_dropped_bucket_is_listed_and_logged_as_an_error(caplog):
    """A bucket that runs eagerly is a 10-20x latency cliff, so it has to be
    visible in the log and to the engine's strict mode."""
    runner = _FakeRunner(
        _specs(walks=("decode", "prefill")), fail={("decode", 1)},
    )

    with caplog.at_level("ERROR", logger="mstar.engine.cuda_graph_runner"):
        runner.warmup_and_capture()

    assert [key.graph_walk for key in runner.dropped_buckets] == ["decode"]
    summary = [r for r in caplog.records if "captured 1 of 2 buckets" in r.message]
    assert summary and summary[0].levelname == "ERROR"

    clean = _FakeRunner(_specs(walks=("decode", "prefill")))
    clean.warmup_and_capture()
    assert clean.dropped_buckets == []


@requires_cuda
def test_a_failed_capture_leaves_the_pool_and_stream_usable():
    """A capture that dies part way used to leave the allocator recording
    into the shared pool (every later bucket then failed with "already
    recording to mempool_id") and the thread on the capture stream."""
    device = torch.device("cuda")
    x = torch.ones(8, device=device)
    pool = torch.cuda.graph_pool_handle()

    def bad():
        # a pageable host-to-device copy is not permitted while capturing
        return x + torch.tensor([1.0], device=device)

    with pytest.raises(RuntimeError):
        capture_into_graph(bad, pool, device, None)

    assert torch.cuda.current_stream(device) == torch.cuda.default_stream(device)

    graph, out = capture_into_graph(lambda: x * 2, pool, device, None)
    graph.replay()
    torch.cuda.synchronize()
    assert out.tolist() == [2.0] * 8


def test_required_graphs_turn_a_dropped_bucket_into_a_startup_failure(monkeypatch):
    monkeypatch.delenv("MSTAR_REQUIRE_CUDA_GRAPHS", raising=False)
    fail_if_graphs_required(["node decode[bs=1]"])

    monkeypatch.setenv("MSTAR_REQUIRE_CUDA_GRAPHS", "1")
    fail_if_graphs_required([])
    with pytest.raises(RuntimeError, match="node decode"):
        fail_if_graphs_required(["node decode[bs=1]"])


@requires_cuda
def test_a_recompile_during_capture_fails_with_the_guard_that_broke():
    """dynamo saves the CUDA RNG state before compiling, which CUDA refuses
    mid capture, so a forward that would recompile inside the capture cannot
    succeed. Fail it before any CUDA call, naming the guard."""
    device = torch.device("cuda")
    x = torch.ones(4, device=device)

    @torch.compile(dynamic=False)
    def scale(t, n):
        return t * n

    scale(x, 1)  # warm: compiled and cached for n == 1
    pool = torch.cuda.graph_pool_handle()

    with pytest.raises(RuntimeError, match="recompile"):
        capture_into_graph(lambda: scale(x, 2), pool, device, None)

    graph, out = capture_into_graph(lambda: scale(x, 1), pool, device, None)
    graph.replay()
    torch.cuda.synchronize()
    assert out.tolist() == [1.0] * 4

