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
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine import cuda_graph_runner
from mstar.engine.cuda_graph_config import PiecewiseCaptureShape
from mstar.engine.cuda_graph_runner import (
    CaptureBudget,
    CudaGraphRunner,
    EagerStep,
    PiecewiseCudaGraphRunner,
    WarmedRegion,
    WarmedSpec,
)
from mstar.engine.engine import Engine
from mstar.engine.resources import BucketKey, CGSlotSpec

_GIB = 2**30


def _bucket(walk: str, bs: int, tokens: int) -> BucketKey:
    return BucketKey(graph_walk=walk, bs=bs, num_tokens=tokens)


_DECODE = _bucket("thinker_decode", 32, 32)
_PREFILL = _bucket("prefill_vision", 1, 16384)


class _StubMeasuredRunner:
    """`measure_eager_steps` over stub warm-ups of ``steps``, two slots each, in capture order."""

    measure_eager_steps = CudaGraphRunner.measure_eager_steps

    def __init__(self, device: torch.device, steps: dict[BucketKey, EagerStep]):
        self._steps = steps
        self._device = device
        self._autocast_dtype = None
        self._submodule_name = "Thinker"
        self._eager_steps = {}
        group = SimpleNamespace(world_size=1, barrier=lambda: None)
        self._comm_group = SimpleNamespace(tp_group=group, sp_group=group)
        self._dummy_rows = SimpleNamespace(release_all=lambda: None)

    def prepare_for_capture(self):
        return [CGSlotSpec(bucket=b, slot=s, config=SimpleNamespace()) for b in self._steps for s in (0, 1)]

    @contextmanager
    def _warmed(self, spec):
        step = self._steps[spec.bucket]
        yield WarmedSpec(
            run=lambda: step, static_inputs={}, static_input_keys=(), dummy_rids=[], dummy_metadata={},
        )


@pytest.fixture
def measured(monkeypatch):
    """Each measured forward's step, with the pool it would run in stubbed out."""
    runs = []

    def measure(device, run):
        runs.append(run())
        return runs[-1]

    monkeypatch.setattr(cuda_graph_runner, "_eager_step", measure)
    return runs


def test_the_floor_is_the_largest_eager_peak_on_the_device(measured):
    cuda = torch.device("cuda", 0)
    talker = _StubMeasuredRunner(cuda, {_DECODE: EagerStep(peak=_GIB, reserved=2 * _GIB)})
    code2wav = _StubMeasuredRunner(cuda, {_DECODE: talker._steps[_DECODE], _PREFILL: EagerStep(3 * _GIB, 6 * _GIB)})

    budget = CaptureBudget.measure(cuda, [talker, code2wav])

    assert budget.floor == 3 * _GIB, "the floor must fit every runner's largest eager step"


def test_every_bucket_is_measured_once_before_any_capture(measured):
    """The first spec in capture order is a 32-row decode; the one-row 16k-token
    prefill behind it peaks ten times higher. Measuring only the first would
    set the floor far too low."""
    steps = {_DECODE: EagerStep(peak=_GIB // 10, reserved=_GIB // 5), _PREFILL: EagerStep(peak=_GIB, reserved=2 * _GIB)}

    kept = _StubMeasuredRunner(torch.device("cuda", 0), steps).measure_eager_steps()

    assert measured == list(steps.values()), "each bucket is measured once, in capture order"
    assert kept == steps, "every bucket's step must be kept to size its capture"


def test_a_cpu_device_never_asks_the_driver(monkeypatch):
    """Asking the CUDA driver for memory fails an XPU or CPU worker at start. A
    CPU device skips the budget even on a host where CUDA exists."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    driver_calls = []
    for name in ("mem_get_info", "synchronize", "empty_cache", "MemPool", "use_mem_pool"):
        monkeypatch.setattr(torch.cuda, name, lambda *args, name=name, **kwargs: driver_calls.append(name))
    runner = _StubMeasuredRunner(torch.device("cpu"), {_DECODE: EagerStep(peak=_GIB, reserved=2 * _GIB)})

    budget = CaptureBudget.measure(torch.device("cpu"), [runner])

    assert budget is None
    assert driver_calls == [], "a CPU worker must not ask the CUDA driver for memory"


def _taking(gpu: SimpleNamespace, cost: int):
    """A forward whose capture takes ``cost`` off ``gpu``, or fails for want of it."""

    def run():
        if gpu.free < cost:
            raise torch.OutOfMemoryError("CUDA out of memory")
        gpu.free -= cost

    return run


class _StubCaptureRunner:
    """The real capture loop and budget check over stub captures, one slot per bucket."""

    warmup_and_capture = CudaGraphRunner.warmup_and_capture
    _wanted = CudaGraphRunner._wanted
    _capture_one = CudaGraphRunner._capture_one
    _buckets_captured_everywhere = CudaGraphRunner._buckets_captured_everywhere
    _register_slot = CudaGraphRunner._register_slot
    _report_dropped = CudaGraphRunner._report_dropped
    _build_slot_from_capture = CudaGraphRunner._build_slot_from_capture

    def __init__(self, gpu: SimpleNamespace, reserved: dict[int, int]):
        # reserved bytes by batch size; the buckets run largest first, as sorted
        self._gpu = gpu
        self._submodule_name = "Code2Wav"
        self._num_slots = 1
        self._device = torch.device("cuda", 0)
        self._autocast_dtype = None
        self._buckets = {}
        self._eager_steps = {
            _bucket("decode", bs, bs): EagerStep(peak=size // 2, reserved=size) for bs, size in reserved.items()
        }
        group = SimpleNamespace(world_size=1, barrier=lambda: None)
        self._comm_group = SimpleNamespace(tp_group=group, sp_group=group)
        self._dummy_rows = SimpleNamespace(release_all=lambda: None)
        self.tried: list[int] = []

    def prepare_for_capture(self):
        buckets = sorted(self._eager_steps, key=lambda bucket: bucket.bs, reverse=True)
        return [CGSlotSpec(bucket=bucket, slot=0, config=SimpleNamespace()) for bucket in buckets]

    @contextmanager
    def _warmed(self, spec):
        self.tried.append(spec.bs)
        yield WarmedSpec(
            run=_taking(self._gpu, self._eager_steps[spec.bucket].reserved), static_inputs={},
            static_input_keys=(), dummy_rids=[], dummy_metadata={},
        )

    _get_addtl_slot_specs = staticmethod(lambda spec: [])
    _log_memory = staticmethod(lambda before, after: None)

    def kept(self) -> list[int]:
        return sorted((key.bs for key in self._buckets), reverse=True)


class _StubCaptureRegion:
    """The real piecewise capture loop and budget check over stub captures."""

    warmup_and_capture = PiecewiseCudaGraphRunner.warmup_and_capture
    _wanted = PiecewiseCudaGraphRunner._wanted
    _capture_one = PiecewiseCudaGraphRunner._capture_one
    _bucket = PiecewiseCudaGraphRunner._bucket

    def __init__(self, gpu: SimpleNamespace, reserved: dict[tuple[int, int], int]):
        self._gpu = gpu
        self._shapes = [
            PiecewiseCaptureShape(bs=bs, seq_lens=[total // bs] * bs, total_tokens=total) for bs, total in reserved
        ]
        self._device = torch.device("cuda", 0)
        self._autocast_dtype = None
        self._label = "vit"
        self._comm_group = None
        self._num_slots = 1
        self._graphs = {}
        self.dropped_shapes = []
        self._eager_steps = {
            self._bucket(shape): EagerStep(peak=size // 2, reserved=size)
            for shape, size in zip(self._shapes, reserved.values(), strict=True)
        }
        self.tried: list[tuple[int, int]] = []

    def prepare_for_capture(self):
        return self._shapes

    _normalize_output = staticmethod(lambda raw: raw)

    @contextmanager
    def _warmed(self, shape, slot):
        self.tried.append((shape.bs, shape.total_tokens))
        yield WarmedRegion(
            run=_taking(self._gpu, self._eager_steps[self._bucket(shape)].reserved), static_inputs={}, dummy_rids=[],
        )


@pytest.fixture
def gpu(monkeypatch):
    """A fake device the capture path reads free memory from, where a capture runs the forward once."""
    gpu = SimpleNamespace(free=0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda.graphs, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (gpu.free, 80 * _GIB))
    monkeypatch.setattr(
        cuda_graph_runner, "capture_into_graph", lambda run, pool, device, autocast_dtype: (run(), None),
    )
    return gpu


def test_buckets_that_would_take_the_floor_are_dropped_largest_first(gpu):
    """4 GiB held for eager steps out of 6.5 free leaves 2.5 for graphs: bs=32
    doesn't fit, bs=16 does and leaves 0.5, bs=8 then doesn't, and bs=1 takes
    the rest."""
    gpu.free = 13 * _GIB // 2
    budget = CaptureBudget(torch.device("cuda", 0), floor=4 * _GIB)
    runner = _StubCaptureRunner(gpu, {32: 4 * _GIB, 16: 2 * _GIB, 8: 1 * _GIB, 1: _GIB // 2})

    runner.warmup_and_capture(budget)

    assert runner.tried == runner.kept() == [16, 1], "a bucket that would take the floor must not even be tried"
    assert gpu.free == budget.floor, "capture must leave the floor free"


def test_a_second_runner_on_the_device_sees_what_the_first_left(gpu):
    """Runners capture one after another from one budget: Code2Wav's bs=32 fits
    the device at the start, not what the Talker left above the floor."""
    gpu.free = 13 * _GIB // 2
    budget = CaptureBudget(torch.device("cuda", 0), floor=3 * _GIB)
    talker = _StubCaptureRunner(gpu, {4: 2 * _GIB, 1: _GIB // 2})
    code2wav = _StubCaptureRunner(gpu, {32: 3 * _GIB // 2, 1: 1 * _GIB})

    talker.warmup_and_capture(budget)
    code2wav.warmup_and_capture(budget)

    assert talker.kept() == [4, 1]
    assert code2wav.tried == code2wav.kept() == [1], "the second runner only gets what the first left above the floor"


def test_a_region_shape_that_would_take_the_floor_runs_eagerly(gpu):
    gpu.free = 4 * _GIB
    region = _StubCaptureRegion(gpu, {(2, 4096): 2 * _GIB, (1, 1024): _GIB // 2})

    region.warmup_and_capture(CaptureBudget(torch.device("cuda", 0), floor=3 * _GIB))

    assert region.tried == [(1, 1024)], "a shape that would take the floor must not even be tried"
    assert region.dropped_shapes == [(2, 4096)], "a refused shape must be listed as running eagerly"


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
