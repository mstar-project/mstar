"""A split scan's block size stays at or above the one its workspace is sized for
(`keep_split_scans_in_their_workspace`), in coordinate descent and in cached picks, and
other kernels' tuning is left alone.

Drives Inductor's own `CoordescTuner.autotune` with a timing function that favours small
blocks, so the test fails if the tuner could still reach one. No GPU needed."""

from __future__ import annotations

import pytest

triton = pytest.importorskip("triton")
from torch._inductor.runtime import autotune_cache  # noqa: E402
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner  # noqa: E402

import mstar.engine.engine as engine_module  # noqa: E402
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


def _cached_pick(grid_type: str, rblock: int):
    """What the autotune cache hands back for a pick coordinate descent saved earlier."""
    saved = {"R0_BLOCK": rblock, "num_warps": 4, "num_stages": 1,
             "found_by_coordesc": True, "configs_hash": "h"}
    meta = {**_meta(grid_type), "coordinate_descent_tuning": True}
    return autotune_cache._load_cached_autotuning(saved, "h", [], meta)


def test_a_split_scan_pick_cached_below_its_minimum_is_retuned():
    assert _cached_pick("SplitScanGrid", MIN_RBLOCK // 2) is None


def test_other_cached_picks_still_load():
    assert _cached_pick("SplitScanGrid", MIN_RBLOCK).kwargs["R0_BLOCK"] == MIN_RBLOCK
    assert _cached_pick("Grid2D", MIN_RBLOCK // 2).kwargs["R0_BLOCK"] == MIN_RBLOCK // 2


def test_the_engine_installs_it(monkeypatch):
    calls = []
    monkeypatch.setattr(engine_module, "keep_split_scans_in_their_workspace", lambda: calls.append(1))
    make_engine()
    assert calls == [1]
