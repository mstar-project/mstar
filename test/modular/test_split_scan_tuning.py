"""Coordinate descent never tunes a split scan's block size below the one its workspace is
sized for (`keep_split_scans_in_their_workspace`), and leaves other kernels' tuning alone.

Drives Inductor's own `CoordescTuner.autotune` with a timing function that favours small
blocks, so the test fails if the tuner could still reach one. No GPU needed."""

from __future__ import annotations

import pytest

triton = pytest.importorskip("triton")
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner  # noqa: E402

from mstar.platforms.compile import keep_split_scans_in_their_workspace  # noqa: E402

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
