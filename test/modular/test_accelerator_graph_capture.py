"""Registration of captured CUDA graph buckets.

A bucket's slots are double buffers of one shape: replay(N) runs on one while
pre-plan(N+1) writes the other. So a bucket is only usable with all of its
slots — registering a partial one would hand both to the same buffers, and
(because slots are appended in index order) would file the surviving slot
under a lower index than the one it captured under.

Capture failure is also a per-rank event, so the barrier count must not depend
on whether a capture succeeded. Same for the TP group's capture scope, which
can run a collective of its own when the capture finishes.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from torch.distributed import ReduceOp

from mstar.distributed.ar_fusion import GraphCaptureScope
from mstar.distributed.communication import JointGroups
from mstar.engine.accelerator_graph_backend import AcceleratorGraphPool
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfigType
from mstar.engine.accelerator_graph_runner import (
    AcceleratorGraphRunner,
    CaptureCost,
    capture_into_graph,
    fail_if_graphs_required,
)
from mstar.engine.resources import BucketKey, CGSlotSpec

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="capture allocates a graph pool"
)


class _Scope(GraphCaptureScope):
    """A capture scope that can reject a capture, like a strategy does when
    ranks recorded different collectives."""

    def __init__(self, drop: bool):
        super().__init__()
        self._drop = drop

    def __exit__(self, exc_type, exc, tb):
        super().__exit__(exc_type, exc, tb)
        self.usable = self.usable and not self._drop
        return False


class _Group:
    """A one-rank comm group, or a two-rank one with the peer's flags given.
    ``drop_scopes`` lists, by entry order, the captures its scope rejects."""

    def __init__(self, peer_flags: list[bool] | None = None, drop_scopes=()):
        self.count = 0
        self.scopes = 0
        self.world_size = 1 if peer_flags is None else 2
        self._peer = peer_flags
        self._drop_scopes = set(drop_scopes)

    def barrier(self):
        self.count += 1

    def agree(self, values, op):
        if self._peer is None:
            return list(values)
        pick = min if op == ReduceOp.MIN else max
        return [int(pick(a, b)) for a, b in zip(values, self._peer, strict=True)]

    def graph_capture(self):
        scope = _Scope(drop=self.scopes in self._drop_scopes)
        self.scopes += 1
        return scope


class _FakeRunner:
    """`warmup_and_capture`, with every bucket planned, and `_register_slot` bound onto stubs."""

    _register_slot = AcceleratorGraphRunner._register_slot
    _buckets_captured_everywhere = AcceleratorGraphRunner._buckets_captured_everywhere
    _report_dropped = AcceleratorGraphRunner._report_dropped
    _refused_buckets = AcceleratorGraphRunner._refused_buckets

    def __init__(
        self, specs, fail: set[tuple[str, int]] = frozenset(), num_slots=2,
        peer_flags: list[bool] | None = None, drop_scopes=(),
        refuse: frozenset[str] = frozenset(),
    ):
        # only carries the rank-agreement flag vector; nothing is captured here
        self._device = torch.device("cpu")
        self._graph_backend = SimpleNamespace(
            is_available=lambda: True,
            graph_pool_handle=lambda: AcceleratorGraphPool(object()),
            memory_allocated=lambda: 0,
        )
        self._submodule_name = "node"
        self._num_slots = num_slots
        self._specs = specs
        self._fail = fail
        # every bucket alike, so capture keeps the specs' order
        self._capture_costs = {spec.bucket: CaptureCost(0, 0, 0, num_slots) for spec in specs}
        # walks a resource cannot run inside a graph
        self._refuse = refuse
        self.attempted: list = []
        self._buckets = {}
        self._memory_pool = None
        self._blocked_buckets = []
        self._executables = {}
        self.barrier = _Group(peer_flags, drop_scopes)
        self._comm_group = JointGroups(tp_group=self.barrier, sp_group=self.barrier)
        self.declared: list = []
        self._dummy_rows = SimpleNamespace(
            released=False,
            release_all=lambda: setattr(self._dummy_rows, "released", True),
        )
        # what the node's resources hear was kept
        self.reported: list = []
        self._step_runner = SimpleNamespace(
            buckets_captured=lambda buckets, node_name=None: self.reported.append((node_name, list(buckets))),
        )

    def prepare_for_capture(self):
        return self._specs

    def warmup_and_capture(self):
        AcceleratorGraphRunner.warmup_and_capture(self, {spec.bucket for spec in self._specs})

    def _capture_refusal(self, spec):
        return "no capturable kernel" if spec.bucket.graph_walk in self._refuse else None

    def _capture_one(self, spec, forwards):
        self.attempted.append((spec.bucket.graph_walk, spec.slot))
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


def _specs(walks=("decode",), num_slots=2, packed=()):
    out = []
    for walk in walks:
        bucket = BucketKey(graph_walk=walk, bs=1, num_tokens=8, cg_key_info=None)
        kind = (
            AcceleratorGraphConfigType.FLASH_INFER_PACKED if walk in packed
            else AcceleratorGraphConfigType.BASIC_BATCHED
        )
        for slot in range(num_slots):
            out.append(CGSlotSpec(
                bucket=bucket, slot=slot, config_idx=0,
                config=SimpleNamespace(get_config_type=lambda kind=kind: kind),
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


def test_every_rank_enters_the_capture_scope_once_per_spec_whatever_happens():
    """The scope may sync with other ranks on exit, so a rank whose capture
    failed has to go through it as many times as the rest."""
    clean = _FakeRunner(_specs(walks=("decode", "prefill")))
    failed = _FakeRunner(_specs(walks=("decode", "prefill")), fail={("decode", 0)})

    clean.warmup_and_capture()
    failed.warmup_and_capture()

    assert failed.barrier.scopes == clean.barrier.scopes == len(_specs(
        walks=("decode", "prefill")
    ))


def test_a_capture_its_scope_rejects_drops_the_whole_bucket():
    """A rejected graph must never replay, and its bucket can't run on the
    other slot alone."""
    # specs run in the given order, so decode slot 0 is entry 0
    runner = _FakeRunner(_specs(walks=("decode", "prefill")), drop_scopes={0})

    runner.warmup_and_capture()

    assert [key.graph_walk for key in runner._buckets] == ["prefill"]


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


def test_only_a_batched_bucket_gets_pre_seeded_rows():
    """A batch that hasn't built its inputs can only lease a batched bucket, so
    no step ever reads a packed bucket's template rows. Pre-seeded for every
    bucket, they held over 0.5 GiB for the prefill buckets of Qwen3-Omni's Thinker."""
    runner = _FakeRunner(_specs(walks=("decode", "prefill"), packed=("prefill",)))

    runner.warmup_and_capture()

    assert [lease.bucket.graph_walk for lease in runner.declared] == ["decode"], (
        "a packed bucket's rows are never read, so they must not be built"
    )


class _InternRunner:
    """`_intern_static_buffer` bound onto the three fields it touches."""

    _seq_dim = staticmethod(AcceleratorGraphRunner._seq_dim)
    _intern_static_buffer = AcceleratorGraphRunner._intern_static_buffer

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
    assert AcceleratorGraphRunner._seq_dim(value, seq_len=512) == 1


def test_seq_dim_guesses_the_wrong_dim_when_button_collides_with_seq_len():
    """Documents `_seq_dim`'s plain size scan on Waypoint's ``button``
    shape ``[bs, 1, n_buttons]``: at 360p bs=2 the bucket's token count
    (2*128=256) equals ``n_buttons``, so the scan hoists dim 2 instead of the
    real batch-varying dim 0. Config-declared ``input_seq_dims`` is how
    Waypoint overrides this guess — see
    `test_button_shares_its_buffer_via_input_seq_dims_override`."""
    value = torch.zeros(2, 1, 256)
    assert AcceleratorGraphRunner._seq_dim(value, seq_len=256) == 2


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

    with caplog.at_level("ERROR", logger="mstar.engine.accelerator_graph_runner"):
        runner.warmup_and_capture()

    assert [key.graph_walk for key in runner.dropped_buckets] == ["decode"]
    summary = [r for r in caplog.records if "captured 1 of 2 buckets" in r.message]
    assert summary and summary[0].levelname == "ERROR"

    clean = _FakeRunner(_specs(walks=("decode", "prefill")))
    clean.warmup_and_capture()
    assert clean.dropped_buckets == []


def test_the_resources_hear_only_the_buckets_kept():
    """A resource that picks between captures (the sampler's unfiltered one)
    must not pick a bucket that was dropped on any rank."""
    runner = _FakeRunner(_specs(walks=("decode", "prefill")), fail={("decode", 1)})

    runner.warmup_and_capture()

    ((node, kept),) = runner.reported
    assert node == "node"
    assert [key.graph_walk for key in kept] == ["prefill"]


def test_a_bucket_a_resource_refuses_is_never_captured_and_runs_eagerly(caplog):
    """A resource that cannot run a bucket in a graph says so before capture
    starts: no capture is attempted, the bucket runs eagerly with one warning
    saying why, and it stays listed, so a deployment that requires every graph
    still fails. Every rank skips it, scope and all, as it skips an unplanned
    bucket."""
    specs = _specs(walks=("decode", "prefill"))
    runner = _FakeRunner(specs, refuse=frozenset({"prefill"}))

    with caplog.at_level("WARNING", logger="mstar.engine.accelerator_graph_runner"):
        runner.warmup_and_capture()

    assert [key.graph_walk for key in runner._buckets] == ["decode"]
    assert {walk for walk, _ in runner.attempted} == {"decode"}
    assert [key.graph_walk for key in runner.dropped_buckets] == ["prefill"]
    assert runner.barrier.scopes == len([spec for spec in specs if spec.bucket.graph_walk == "decode"])
    refusals = [r for r in caplog.records if "runs eagerly: no capturable kernel" in r.message]
    assert len(refusals) == 1 and refusals[0].levelname == "WARNING"
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


class _Refusing:
    """A resource that refuses buckets above a token count."""

    def __init__(self, limit):
        self.limit = limit

    def capture_refusal(self, spec):
        return f"over {self.limit}" if spec.max_row_tokens > self.limit else None


def test_the_step_runner_asks_only_the_nodes_own_resources():
    from mstar.engine.resources.base import Resource
    from mstar.engine.resources.runner import StepRunner

    class Plain(Resource):
        @classmethod
        def build(cls, spec, info):
            return cls()

    class Refusing(Plain):
        capture_refusal = _Refusing(64).capture_refusal

    runner = StepRunner(
        {"plain": Plain(), "gdn": Refusing()},
        node_resources={"llm": ["plain", "gdn"], "vision": ["plain"]},
    )
    big = CGSlotSpec(
        bucket=BucketKey(graph_walk="prefill", bs=1, num_tokens=128), slot=0,
        config=SimpleNamespace(get_config_type=lambda: None),
    )
    assert runner.capture_refusal(big, node_name="llm") == "gdn: over 64"
    assert runner.capture_refusal(big, node_name="vision") is None


def test_the_longest_row_a_bucket_can_hold():
    """A batched capture gives each row the same share of the bucket's
    tokens; a packed one may hand one row all of them."""
    from mstar.engine.accelerator_graph_config import (
        BatchedAcceleratorGraphConfig,
        PackedAcceleratorGraphConfig,
    )

    batched = BatchedAcceleratorGraphConfig(
        capture_graph_walk="decode", single_request_inputs=SimpleNamespace(input_seq_len=1),
    )
    packed = PackedAcceleratorGraphConfig(
        capture_graph_walk="prefill", capture_token_lengths=[1024], make_node_input=None,
    )
    bucket = BucketKey(graph_walk="w", bs=16, num_tokens=1024)
    assert CGSlotSpec(bucket=bucket, slot=0, config=batched).max_row_tokens == 64
    assert CGSlotSpec(bucket=bucket, slot=0, config=packed).max_row_tokens == 1024


@requires_cuda
def test_a_failed_capture_leaves_the_pool_and_stream_usable():
    """A capture that dies part way used to leave the allocator recording
    into the shared pool (every later bucket then failed with "already
    recording to mempool_id") and the thread on the capture stream."""
    device = torch.device("cuda")
    x = torch.ones(8, device=device)
    pool = AcceleratorGraphPool(torch.cuda.graph_pool_handle())

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


@requires_cuda
@pytest.mark.parametrize("fails", [False, True], ids=["captured", "failed"])
def test_benchmark_is_off_inside_the_capture_and_restored_after(torch_flag, fails):
    """Qwen3-Omni turns cuDNN's benchmark mode on for Code2Wav's warm-up.
    Inside a capture its search fails the capture, which cost Code2Wav its
    bs=32 graph and kept that graph's pool."""
    device = torch.device("cuda")
    x = torch.ones(8, device=device)
    seen = []

    def run():
        seen.append(torch.backends.cudnn.benchmark)
        # a pageable host-to-device copy is not permitted while capturing
        return x + torch.tensor([1.0], device=device) if fails else x * 2

    for benchmark in (True, False):
        torch_flag(torch.backends.cudnn, "benchmark", benchmark)
        with pytest.raises(RuntimeError) if fails else nullcontext():
            capture_into_graph(run, AcceleratorGraphPool(torch.cuda.graph_pool_handle()), device, None)
        assert torch.backends.cudnn.benchmark is benchmark, "the caller's setting must come back"
    assert seen == [False, False], "cuDNN must not search inside the capture"


def test_required_graphs_turn_a_dropped_bucket_into_a_startup_failure(monkeypatch):
    monkeypatch.delenv("MSTAR_REQUIRE_ACCELERATOR_GRAPHS", raising=False)
    fail_if_graphs_required(["node decode[bs=1]"])

    monkeypatch.setenv("MSTAR_REQUIRE_ACCELERATOR_GRAPHS", "1")
    fail_if_graphs_required([])
    with pytest.raises(RuntimeError, match="node decode"):
        fail_if_graphs_required(["node decode[bs=1]"])


def test_accelerator_graph_requirement_can_be_disabled_explicitly(monkeypatch):
    monkeypatch.setenv("MSTAR_REQUIRE_ACCELERATOR_GRAPHS", "0")
    fail_if_graphs_required(["node decode[bs=1]"])

    monkeypatch.setenv("MSTAR_REQUIRE_ACCELERATOR_GRAPHS", "1")
    with pytest.raises(RuntimeError, match="Accelerator graph|accelerator graph"):
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
    pool = AcceleratorGraphPool(torch.cuda.graph_pool_handle())

    with pytest.raises(RuntimeError, match="recompile"):
        capture_into_graph(lambda: scale(x, 2), pool, device, None)

    graph, out = capture_into_graph(lambda: scale(x, 1), pool, device, None)
    graph.replay()
    torch.cuda.synchronize()
    assert out.tolist() == [1.0] * 4
