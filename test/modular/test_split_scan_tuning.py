"""A split scan's block size stays at or above the one its workspace is sized for
(`keep_split_scans_in_their_workspace`): coordinate descent never steps below it, every
config is raised to it before launch, and other kernels are left alone.

Drives Inductor's own `CoordescTuner.autotune` with a timing function that favours small
blocks, so the test fails if the tuner could still reach one. No GPU needed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

triton = pytest.importorskip("triton")
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner  # noqa: E402
from torch._inductor.runtime.triton_heuristics import CachingAutotuner  # noqa: E402

import mstar.engine.engine as engine_module  # noqa: E402
import mstar.platforms.compile as compile_mod  # noqa: E402
from mstar.platforms.compile import keep_split_scans_in_their_workspace  # noqa: E402
from mstar.testing_utils.engine import make_engine  # noqa: E402

MIN_RBLOCK = 256


def _meta(grid_type: str) -> dict:
    return {"grid_type": grid_type, "min_split_scan_rblock": MIN_RBLOCK}


def _tuned_rblock(grid_type: str) -> int:
    tuner = CoordescTuner(size_hints={"x": 1, "r0_": 16384}, inductor_meta=_meta(grid_type))
    baseline = triton.Config({"R0_BLOCK": MIN_RBLOCK}, num_warps=4)
    # smaller blocks time faster, as they did on the media buckets that faulted
    best = tuner.autotune(lambda cfg: float(cfg.kwargs["R0_BLOCK"]), baseline)
    return best.kwargs["R0_BLOCK"]


@pytest.fixture(autouse=True)
def _installed():
    keep_split_scans_in_their_workspace()


def test_a_split_scan_keeps_its_minimum_block():
    assert _tuned_rblock("SplitScanGrid") == MIN_RBLOCK


def test_other_kernels_still_tune_below_it():
    assert _tuned_rblock("Grid2D") < MIN_RBLOCK


def test_a_split_scan_may_still_tune_up():
    tuner = CoordescTuner(size_hints={"x": 1, "r0_": 16384}, inductor_meta=_meta("SplitScanGrid"))
    baseline = triton.Config({"R0_BLOCK": MIN_RBLOCK}, num_warps=4)
    best = tuner.autotune(lambda cfg: 1.0 / cfg.kwargs["R0_BLOCK"], baseline)
    assert best.kwargs["R0_BLOCK"] > MIN_RBLOCK


def test_installing_twice_wraps_once():
    first = CoordescTuner.value_too_small
    keep_split_scans_in_their_workspace()
    assert CoordescTuner.value_too_small is first


class _Result:
    """A compile result: only its config matters here."""

    def __init__(self, cfg):
        self.config = cfg


class _Tuner:
    """The `CachingAutotuner` fields the clamp reads, and a compile that records its configs."""

    def __init__(self, grid_type: str, rblocks: list[int], *, from_worker: bool = False):
        self.inductor_meta = _meta(grid_type)
        self.compile_results = [_Result(triton.Config({"R0_BLOCK": r}, num_warps=4)) for r in rblocks]
        self.fn = SimpleNamespace(fn=None if from_worker else "source")
        self._reload_kernel = lambda: SimpleNamespace(fn=SimpleNamespace(fn="reloaded"))
        self.compiled = []

    def _precompile_config(self, cfg):
        self.compiled.append(cfg)
        return _Result(cfg)


def _rblocks(tuner) -> list[int]:
    return [r.config.kwargs["R0_BLOCK"] for r in tuner.compile_results]


def test_a_split_scan_config_below_its_minimum_is_raised_before_launch():
    tuner = _Tuner("SplitScanGrid", [1, MIN_RBLOCK // 2, 2 * MIN_RBLOCK])
    compile_mod._raise_split_scan_blocks(tuner)
    assert _rblocks(tuner) == [MIN_RBLOCK, MIN_RBLOCK, 2 * MIN_RBLOCK]
    assert len(tuner.compiled) == 2
    # coordinate descent retunes from the raised config instead of trusting the old pick
    assert all(cfg.found_by_coordesc is False for cfg in tuner.compiled)


def test_a_worker_compiled_split_scan_reloads_its_source_to_recompile():
    tuner = _Tuner("SplitScanGrid", [MIN_RBLOCK // 2], from_worker=True)
    compile_mod._raise_split_scan_blocks(tuner)
    assert tuner.fn.fn == "reloaded" and _rblocks(tuner) == [MIN_RBLOCK]


def test_other_kernels_launch_their_configs_as_tuned():
    tuner = _Tuner("Grid2D", [MIN_RBLOCK // 2])
    compile_mod._raise_split_scan_blocks(tuner)
    assert _rblocks(tuner) == [MIN_RBLOCK // 2] and not tuner.compiled


def test_launchers_go_through_the_clamp():
    assert "keep_split_scans_in_their_workspace" in CachingAutotuner._make_launchers.__qualname__


def test_the_engine_installs_it(monkeypatch):
    calls = []
    monkeypatch.setattr(engine_module, "keep_split_scans_in_their_workspace", lambda: calls.append(1))
    make_engine()
    assert calls == [1]
