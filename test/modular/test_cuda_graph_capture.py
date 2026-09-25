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


@pytest.fixture(autouse=True)
def fake_cuda_runtime(request, monkeypatch):
    """The only real CUDA calls on this path are the graph pool handle and the
    memory readings around it; `_FakeRunner` stands in for the capture itself.
    Stubbing them keeps these policy tests running where there is no GPU.

    Stubbed on a GPU too: `_FakeRunner` is on the CPU device, and the real
    `memory_allocated(cpu)` raises once any earlier test has initialised CUDA.
    Only the `requires_cuda` tests, which capture for real, keep the runtime."""
    if requires_cuda.mark in request.node.iter_markers():
        return
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda.graphs, "graph_pool_handle", object)


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
        # only carries the rank-agreement flag vector; nothing is captured here
        self._device = torch.device("cpu")
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


def test_a_fully_captured_bucket_registers_every_slot_in_index_order():
    runner = _FakeRunner(_specs())

    runner.warmup_and_capture()

    (bucket,) = runner._buckets.values()
    assert bucket.slots == ["decode:slot0", "decode:slot1"], (
        "list position must be the slot index"
    )


def test_a_bucket_missing_a_slot_is_dropped_whole():
    """The regression: slot 0 failing used to leave the bucket registered with
    slot 1's graph sitting at index 0, and only one slot to double-buffer on."""
    runner = _FakeRunner(_specs(), fail={("decode", 0)})

    runner.warmup_and_capture()

    assert runner._buckets == {}, "a half-captured bucket must not be usable"


def test_one_bucket_failing_does_not_take_the_others_with_it():
    runner = _FakeRunner(
        _specs(walks=("decode", "prefill")), fail={("decode", 1)},
    )

    runner.warmup_and_capture()

    assert [key.graph_walk for key in runner._buckets] == ["prefill"]
    (bucket,) = runner._buckets.values()
    assert bucket.slots == ["prefill:slot0", "prefill:slot1"]


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


def test_a_bucket_this_rank_dropped_stays_dropped_when_the_peer_kept_it():
    runner = _FakeRunner(
        _specs(walks=("decode", "prefill")),
        fail={("decode", 1)},
        peer_flags=[True, True],
    )

    runner.warmup_and_capture()

    assert [key.graph_walk for key in runner._buckets] == ["prefill"]


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


def test_capture_hands_the_padding_rows_pages_back():
    """A capture gives its padding rows real spans; a replay pads with
    zero-length ones, so that storage is residue the traffic should get."""
    runner = _FakeRunner(_specs())

    runner.warmup_and_capture()

    assert runner._dummy_rows.released


def test_single_slot_runners_still_register():
    """No pre-planning resource means one slot per bucket, which is complete."""
    runner = _FakeRunner(_specs(num_slots=1), num_slots=1)

    runner.warmup_and_capture()

    (bucket,) = runner._buckets.values()
    assert bucket.slots == ["decode:slot0"]
    assert runner.declared == [], "nothing to pre-plan with a single slot"


class _InternRunner:
    """`_intern_static_buffer` bound onto the three fields it touches."""

    _seq_dim = staticmethod(CudaGraphRunner._seq_dim)
    _intern_static_buffer = CudaGraphRunner._intern_static_buffer

    def __init__(self):
        self._shared_static_buffers = {}
        self._static_buffer_seq_dims = {}
        self._capture_clone_bytes_naive = 0


def test_seq_dim_picks_the_only_matching_dim_even_at_batch_size_one():
    """A [1, seq_len] tensor at bs=1: dim 0's size (1) never collides with
    ``batch_size`` here because `_seq_dim` no longer takes one — it just
    scans for ``seq_len``, so a coincidental dim-0 size never shadows the
    real seq dim. This is the case from the PR review comment on
    cuda_graph_runner.py:621."""
    value = torch.zeros(1, 512)
    assert CudaGraphRunner._seq_dim(value, seq_len=512) == 1


def test_seq_dim_guesses_the_wrong_dim_when_button_collides_with_seq_len():
    """Documents `_seq_dim`'s plain size scan on Waypoint's ``button``
    shape ``[bs, 1, n_buttons]``: at 360p bs=2 the bucket's token count
    (2*128=256) equals ``n_buttons``, so the scan hoists dim 2 instead of the
    real batch-varying dim 0. Config-declared ``input_seq_dims`` is how
    Waypoint overrides this guess — see
    `test_button_shares_its_buffer_via_input_seq_dims_override`."""
    value = torch.zeros(2, 1, 256)
    assert CudaGraphRunner._seq_dim(value, seq_len=256) == 2


def test_button_shares_its_buffer_via_input_seq_dims_override():
    """Waypoint's config declares ``input_seq_dims={"button": 0, ...}``, so
    ``_intern_static_buffer`` uses that dim instead of `_seq_dim`'s guess
    (which would hoist dim 2, see
    `test_seq_dim_guesses_the_wrong_dim_when_button_collides_with_seq_len`).
    button then reslices the shared buffer like every other per-row tensor
    instead of falling back to a private allocation."""
    runner = _InternRunner()
    big = torch.arange(2 * 256, dtype=torch.float32).reshape(2, 1, 256)
    small = -torch.arange(256, dtype=torch.float32).reshape(1, 1, 256)

    shared_view = runner._intern_static_buffer(
        0, "button", big, seq_len=256, seq_dim_override=0
    )
    assert shared_view.shape == (2, 1, 256)
    assert torch.equal(shared_view, big)

    resliced = runner._intern_static_buffer(
        0, "button", small, seq_len=128, seq_dim_override=0
    )

    assert runner._static_buffer_seq_dims[(0, "button")] == 0
    assert resliced.shape == (1, 1, 256)
    assert resliced.data_ptr() == shared_view.data_ptr()
    assert torch.equal(resliced, small)
    # the reslice writes through the shared buffer — row 0 now reads back
    # as `small`, which is the whole point of sharing rather than cloning
    assert torch.equal(shared_view[:1], small)


def test_a_smaller_bucket_that_shrinks_off_the_hoisted_axis_raises():
    """Without an override, a fixed-width tensor can still coincidentally
    match ``seq_len`` on a non-batch dim (here: button's n_buttons=256 lines
    up with the bs=2 bucket's token count), hoisting the wrong axis. The bs=1
    bucket then shrinks along dim 0, not the hoisted one, and can't reslice
    the shared buffer. This stays a hard failure rather than a silent private
    allocation; the real fix is a config-declared ``input_seq_dims`` override
    — see `test_button_shares_its_buffer_via_input_seq_dims_override`."""
    runner = _InternRunner()
    big = torch.arange(2 * 256, dtype=torch.float32).reshape(2, 1, 256)
    small = -torch.arange(256, dtype=torch.float32).reshape(1, 1, 256)

    runner._intern_static_buffer(0, "button", big, seq_len=256)
    with pytest.raises(RuntimeError, match="captures must be largest-first"):
        runner._intern_static_buffer(0, "button", small, seq_len=128)


def test_a_smaller_bucket_along_the_hoisted_axis_still_reslices_the_shared_buffer():
    runner = _InternRunner()
    big = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)
    small = torch.zeros(3, 4)

    shared_view = runner._intern_static_buffer(0, "mrope", big, seq_len=8)
    resliced = runner._intern_static_buffer(0, "mrope", small, seq_len=4)

    assert runner._static_buffer_seq_dims[(0, "mrope")] == 1
    assert resliced.shape == (3, 4)
    assert resliced.data_ptr() == shared_view.data_ptr()
    assert torch.equal(shared_view[:, :4], small)


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
