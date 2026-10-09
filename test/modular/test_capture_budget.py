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
from mstar.engine.cuda_graph_runner import CaptureCost, CudaGraphRunner, WarmedSpec
from mstar.engine.engine import Engine
from mstar.engine.resources import BucketKey, CGSlotSpec

_GIB = 2**30

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
