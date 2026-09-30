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
from mstar.engine.cuda_graph_runner import CaptureBudget, CudaGraphRunner, EagerStep, WarmedSpec
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
