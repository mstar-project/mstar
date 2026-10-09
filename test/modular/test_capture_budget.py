"""CUDA graph capture leaves the largest eager step free.

Each runner used to capture every bucket it could, and its graph pool kept
whatever the captures took. Qwen3-Omni colocated on one H100 ended
capture with 3 MiB free, and every text request then failed with CUDA OOM in
the Thinker's prefill: a dropped bucket runs eager, and nothing had kept room
for it. The floor is the largest eager step of any runner on the device,
measured before any of them captures.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine import cuda_graph_runner
from mstar.engine.cuda_graph_runner import CaptureCost, CudaGraphRunner, WarmedSpec, plan_captures
from mstar.engine.engine import Engine
from mstar.engine.resources import BucketKey, CGSlotSpec

_GIB = 2**30
_MIB = 2**20

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a throwaway capture needs a GPU")


def _bucket(walk: str, bs: int, tokens: int) -> BucketKey:
    return BucketKey(graph_walk=walk, bs=bs, num_tokens=tokens)


_DECODE = _bucket("thinker_decode", 32, 32)
_PREFILL = _bucket("prefill_vision", 1, 16384)


class _StubMeasuredRunner:
    """`size_captures` over stub warm-ups of ``steps``, two slots each, in capture order."""

    size_captures = CudaGraphRunner.size_captures

    def __init__(self, device: torch.device, steps: dict[BucketKey, CaptureCost]):
        self._steps = steps
        self._device = device
        self._autocast_dtype = None
        self._submodule_name = "Thinker"
        self._num_slots = 2
        self._capture_costs = {}
        group = SimpleNamespace(world_size=1, barrier=lambda: None)
        self._comm_group = SimpleNamespace(tp_group=group, sp_group=group)
        self.warmed: list[tuple[BucketKey, int]] = []
        self.failing: set[tuple[BucketKey, int]] = set()

    def prepare_for_capture(self):
        return [CGSlotSpec(bucket=b, slot=s, config=SimpleNamespace()) for b in self._steps for s in (0, 1)]

    @contextmanager
    def _warmed(self, spec):
        self.warmed.append((spec.bucket, spec.slot))
        if (spec.bucket, spec.slot) in self.failing:
            raise RuntimeError(f"capture admit failed for {spec.bucket}")
        step = self._steps[spec.bucket]
        yield WarmedSpec(
            run=lambda: step, static_inputs={}, static_input_keys=(), dummy_rids=[], dummy_metadata={},
            peak=step.peak,
        )


@pytest.fixture
def measured(monkeypatch):
    """Each sized bucket's cost, with the throwaway capture stubbed out."""
    runs = []

    def capture(capture, what):
        _, cost = capture(pool=None)
        runs.append(cost)
        return cost.graph, cost.kept

    monkeypatch.setattr(cuda_graph_runner, "_capture_thrown_away", capture)
    monkeypatch.setattr(
        cuda_graph_runner, "capture_into_graph", lambda run, pool, device, autocast_dtype: (None, run()),
    )
    return runs


def test_every_bucket_is_measured_once_before_any_capture(measured):
    """The first spec in capture order is a 32-row decode; the one-row 16k-token
    prefill behind it peaks ten times higher. Measuring only the first would
    set the floor far too low."""
    steps = {_DECODE: CaptureCost(_GIB // 10, _GIB // 5, 0, 2), _PREFILL: CaptureCost(_GIB, 2 * _GIB, 0, 2)}

    kept = _StubMeasuredRunner(torch.device("cuda", 0), steps).size_captures()

    assert measured == list(steps.values()), "each bucket is measured once, in capture order"
    assert kept == steps, "every bucket's step must be kept to size its capture"


def test_every_slot_is_warmed_before_any_capture(measured):
    """A slot's first plan builds state of its own: FlashInfer's graph wrappers,
    and a 512 MiB workspace for the Talker's second slot. Warmed at slot 0 only,
    that came after free memory was read, and the Talker's capture took 844 MiB
    against 98 predicted."""
    runner = _StubMeasuredRunner(torch.device("cuda", 0), {_DECODE: CaptureCost(_GIB, 2 * _GIB, 0, 2)})

    runner.size_captures()

    assert runner.warmed == [(_DECODE, 0), (_DECODE, 1)], "every slot must be warmed before the plan reads free memory"


def test_a_bucket_whose_second_slot_cannot_run_is_never_planned(measured):
    """A bucket registers all of its slots or none, so one whose second slot
    fails its warm-up would be captured at slot 0 for nothing."""
    runner = _StubMeasuredRunner(torch.device("cuda", 0), {_DECODE: CaptureCost(_GIB, 2 * _GIB, 0, 2)})
    runner.failing = {(_DECODE, 1)}

    costs = runner.size_captures()

    assert costs[_DECODE].graph is None, "a bucket that can't run every slot must not be planned"
    assert costs[_DECODE].peak == _GIB, "its eager peak still counts toward the floor"


def test_a_cpu_device_never_asks_the_driver(monkeypatch):
    """Asking the CUDA driver for memory fails an XPU or CPU worker at start. A
    CPU device skips the plan even on a host where CUDA exists."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    driver_calls = []
    for name in ("mem_get_info", "empty_cache", "graph_pool_handle", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, name=name, **kwargs: driver_calls.append(name))
    runner = _StubMeasuredRunner(torch.device("cpu"), {_DECODE: CaptureCost(_GIB, 2 * _GIB, 0, 2)})
    engine = SimpleNamespace(_device=torch.device("cpu"), _submodules={"Thinker": None})

    plan = Engine._plan_captures(engine, {"Thinker": runner}, {"Thinker": {}})

    assert plan is None
    assert driver_calls == [], "a CPU worker must not ask the CUDA driver for memory"


def _succeeds(x: torch.Tensor) -> torch.Tensor:
    return x @ x


def _runs_out_of_memory(x: torch.Tensor) -> torch.Tensor:
    x @ x
    return torch.empty(2 * torch.cuda.mem_get_info(x.device)[0], dtype=torch.uint8, device=x.device)


def _fails_in_capture_end(x: torch.Tensor) -> torch.Tensor:
    """A side stream never joined back: the capture runs, and capture_end refuses it."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        return x @ x


@requires_cuda
@pytest.mark.parametrize("forward", [_succeeds, _runs_out_of_memory, _fails_in_capture_end])
def test_a_throwaway_capture_frees_its_memory_pass_or_fail(forward, monkeypatch):
    """torch releases a graph's pool only for a capture that ended. One that
    failed in capture_end kept its pool, and the cuBLAS workspace the capture
    stream took in it, for good: memory every later bucket was sized without."""
    # a captured log record would keep the failed capture's frames, and its memory, alive
    monkeypatch.setattr(cuda_graph_runner.logger, "disabled", True)
    device = torch.device("cuda")
    x = torch.randn(2048, 2048, device=device)
    # warmed up, as before any capture: cuBLAS can't make its handle inside one
    _succeeds(x)
    # a failed capture drops every stream's workspace, so start with none
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    reserved = torch.cuda.memory_reserved(device)

    taken, _ = cuda_graph_runner._capture_thrown_away(
        partial(cuda_graph_runner.capture_into_graph, lambda: forward(x), device=device, autocast_dtype=None),
        forward.__name__,
    )

    assert (taken is None) == (forward is not _succeeds)
    assert torch.cuda.memory_reserved(device) == reserved, "a throwaway capture must hand back all it took"


def _cost(graph: int, kept: int = 0, peak: int = 0, slots: int = 1) -> CaptureCost:
    """A bucket's cost, in MiB."""
    return CaptureCost(peak=peak * _MIB, graph=graph * _MIB, kept=kept * _MIB, slots=slots)


def test_the_plan_takes_small_graphs_from_every_pool_first():
    """700 MiB beside a 1 GiB eager step: largest first, Code2Wav's bs=32 and
    bs=1 would take it all and leave the Talker none."""
    pools = {
        "Code2Wav": {"bs=32": _cost(600), "bs=1": _cost(100)},
        "Talker": {"bs=32": _cost(500, peak=1024), "bs=1": _cost(100)},
    }

    plan = plan_captures(pools, free=(1024 + 700) * _MIB)

    assert plan.floor == 1024 * _MIB, "the floor is the largest eager step in any pool"
    assert plan.buckets == {"Code2Wav": {"bs=1"}, "Talker": {"bs=1", "bs=32"}}, (
        "every pool's cheapest graphs must come before any pool's biggest"
    )


def test_a_bucket_goes_in_with_all_its_slots_or_not_at_all():
    """Two slots of a 200 MiB graph that keeps 100 MiB of outputs take 300 MiB:
    the second slot reuses the first's scratch but not its outputs."""
    pools = {"Thinker": {"decode": _cost(200, kept=100, slots=2)}}

    assert plan_captures(pools, free=250 * _MIB).buckets == {"Thinker": set()}, "room for one slot is room for none"
    assert plan_captures(pools, free=300 * _MIB).predicted == {"Thinker": 300 * _MIB}


def test_a_graph_keeps_its_outputs_in_whole_segments():
    """Code2Wav's bs=2 graph keeps 373 KB of outputs. Captured after bs=8 into
    one pool, it took a fresh 2 MiB segment: bs=8's scratch had no small block
    left for it, and the pool came out over the plan."""
    bs2 = CaptureCost(peak=0, graph=390 * _MIB, kept=381780, slots=1)
    pools = {"Code2Wav": {"bs=8": _cost(1552, kept=2), "bs=2": bs2}}

    plan = plan_captures(pools, free=2 * _GIB)

    assert plan.predicted == {"Code2Wav": (1550 + 2 + 2) * _MIB}, "a small tensor must count as the segment it may need"


def test_the_same_costs_give_the_same_plan_in_any_order():
    """Nodes came from a set, so the order a plan sees its pools in changed
    from boot to boot. A tie goes to the node named first."""
    talker, code2wav = {"bs=1": _cost(100)}, {"bs=1": _cost(100)}

    one = plan_captures({"Talker": talker, "Code2Wav": code2wav}, free=150 * _MIB)
    other = plan_captures({"Code2Wav": code2wav, "Talker": talker}, free=150 * _MIB)

    assert one.buckets == other.buckets == {"Code2Wav": {"bs=1"}, "Talker": set()}, (
        "the plan must not depend on the order the pools come in"
    )


class _PeerGroup:
    """One of two ranks, whose peer's flags are ``peer``."""

    world_size = 2

    def __init__(self, peer: list[bool]):
        self._peer = peer

    def barrier(self):
        pass

    def all_gather(self, tensor, dim=0):
        return torch.cat([tensor, torch.tensor(self._peer, dtype=tensor.dtype)])


class _StubPlannedRunner:
    """The real capture loop over stub captures, one slot per bucket, on a rank whose peer planned ``peer``."""

    warmup_and_capture = CudaGraphRunner.warmup_and_capture
    _buckets_captured_everywhere = CudaGraphRunner._buckets_captured_everywhere
    _register_slot = CudaGraphRunner._register_slot
    _report_dropped = CudaGraphRunner._report_dropped

    def __init__(self, buckets: list[BucketKey], peer: list[bool]):
        self._specs = [CGSlotSpec(bucket=bucket, slot=0, config=SimpleNamespace()) for bucket in buckets]
        self._capture_costs = {bucket: _cost(100) for bucket in buckets}
        self._device = torch.device("cpu")
        self._submodule_name = "Thinker"
        self._num_slots = 1
        self._buckets = {}
        self._comm_group = SimpleNamespace(
            tp_group=_PeerGroup(peer), sp_group=SimpleNamespace(world_size=1, barrier=lambda: None),
        )
        self._dummy_rows = SimpleNamespace(release_all=lambda: None)
        self.tried: list[BucketKey] = []

    def prepare_for_capture(self):
        return self._specs

    def _capture_one(self, spec):
        self.tried.append(spec.bucket)
        return spec.bucket

    _get_addtl_slot_specs = staticmethod(lambda spec: [])
    _log_memory = staticmethod(lambda before, after: None)


@pytest.fixture
def capture_loop(monkeypatch):
    """The CUDA calls around the capture loop, stubbed so it runs on the CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.graphs, "graph_pool_handle", object)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)


def test_a_tp_runner_captures_only_what_every_rank_planned(capture_loop):
    """Each rank plans on its own device. A bucket only this rank planned would
    be replayed here while the peer runs it eager, and a collective inside it
    would hang."""
    runner = _StubPlannedRunner([_DECODE, _PREFILL], peer=[True, False])

    runner.warmup_and_capture({_DECODE, _PREFILL})

    assert runner.tried == [_DECODE], "a bucket the peer did not plan must not be captured here"


def test_a_runner_captures_its_costliest_bucket_first(capture_loop):
    """The Codec's buckets come in batch-size order, its smallest window first
    at each size. Captured that way, every larger window outgrew what the last
    had left, and the pool took 11222 MiB against 9068 predicted."""
    runner = _StubPlannedRunner([_DECODE, _PREFILL], peer=[True, True])
    runner._capture_costs = {_DECODE: _cost(100), _PREFILL: _cost(900)}

    runner.warmup_and_capture({_DECODE, _PREFILL})

    assert runner.tried == [_PREFILL, _DECODE], "the costliest graph must go first, so the rest reuse its pool"


def test_a_runner_with_no_graphs_keeps_its_largest_bucket_as_its_batch_cap():
    """A runner that captured nothing runs every bucket eager, and the floor
    covers eager steps only up to its largest bucket."""
    code2wav = SimpleNamespace(
        submodule=SimpleNamespace(max_batch_size=lambda walk: None),
        cuda_graph_runner=None,
        capture_runner=SimpleNamespace(max_batch_size_for=lambda walk: 32),
    )
    engine = SimpleNamespace(_submodules={"Code2Wav": code2wav})

    cap = Engine.get_max_batch_size(engine, "Code2Wav", "code2wav_chunk")

    assert cap == 32, "a batch past the largest bucket is an eager step the floor never measured"
