"""Unit tests for the measured step-cost table.

No GPU, no model, no server: the table is a pure function of what was
written into it, and these pin the parts a wrong answer would be silent
about — bucket snapping, KV interpolation, and the coverage flags that tell
a caller how much to trust a number.
"""

import os
import tempfile

import pytest

from mstar.sim.stepdb import (
    Coverage,
    StepDB,
    StepKey,
    StepSample,
    pad_to_bucket,
)


@pytest.fixture
def db():
    d = tempfile.mkdtemp()
    database = StepDB(os.path.join(d, "step.db"), gpu_name="TEST-GPU")
    yield database
    database.close()


def _key(**over):
    base = dict(
        model="m", node="LLM", graph_walk="decode",
        padded_bs=4, padded_num_tokens=4,
    )
    base.update(over)
    return StepKey(**base)


def test_exact_lookup_returns_measured_value(db):
    db.add(StepSample(_key(), kv_len_total=1024, gpu_s=0.005, plan_s=0.001))
    cost = db.lookup(_key(), 1024)
    assert cost.gpu_s == pytest.approx(0.005)
    assert cost.plan_s == pytest.approx(0.001)
    assert cost.coverage == Coverage.EXACT


def test_kv_interpolation_is_linear_and_flagged(db):
    db.add_many([
        StepSample(_key(), kv_len_total=1000, gpu_s=0.010),
        StepSample(_key(), kv_len_total=2000, gpu_s=0.020),
    ])
    cost = db.lookup(_key(), 1500)
    assert cost.gpu_s == pytest.approx(0.015)
    assert cost.coverage & Coverage.INTERPOLATED


def test_outside_measured_range_is_extrapolated_not_silent(db):
    db.add_many([
        StepSample(_key(), kv_len_total=1000, gpu_s=0.010),
        StepSample(_key(), kv_len_total=2000, gpu_s=0.020),
    ])
    high = db.lookup(_key(), 4000)
    assert high.coverage & Coverage.EXTRAPOLATED
    assert high.gpu_s > 0.020
    assert "above" in high.note


def test_extrapolation_never_goes_negative(db):
    # A steep downward slope extrapolated far below the measured range
    # would otherwise produce a negative duration and rewind the clock.
    db.add_many([
        StepSample(_key(), kv_len_total=1000, gpu_s=0.100),
        StepSample(_key(), kv_len_total=2000, gpu_s=0.001),
    ])
    assert db.lookup(_key(), 0).gpu_s >= 0.0


def test_missing_key_is_reported_not_raised(db):
    db.add(StepSample(_key(), kv_len_total=0, gpu_s=0.005))
    cost = db.lookup(_key(node="nonexistent"), 0)
    assert cost.coverage & Coverage.MISSING
    assert cost.gpu_s == 0.0
    assert "nonexistent" in cost.note


def test_mode_is_part_of_identity(db):
    # A captured replay and an eager fallback are different execution
    # regimes; they must not answer each other's lookups.
    db.add(StepSample(_key(mode="graph"), kv_len_total=0, gpu_s=0.004))
    assert db.lookup(_key(mode="graph"), 0).coverage == Coverage.EXACT
    assert db.lookup(_key(mode="eager"), 0).coverage & Coverage.MISSING


def test_reprofiling_overwrites_metrics_not_identity(db):
    db.add(StepSample(_key(), kv_len_total=512, gpu_s=0.005))
    db.add(StepSample(_key(), kv_len_total=512, gpu_s=0.007))
    assert db.count() == 1
    assert db.lookup(_key(), 512).gpu_s == pytest.approx(0.007)


def test_gpu_name_scopes_rows(db):
    db.add(StepSample(_key(), kv_len_total=0, gpu_s=0.005))
    other = StepDB(db.path, gpu_name="OTHER-GPU")
    try:
        assert other.lookup(_key(), 0).coverage & Coverage.MISSING
    finally:
        other.close()


def test_cpu_total_sums_the_phases(db):
    db.add(StepSample(
        _key(), kv_len_total=0, gpu_s=0.01,
        prepare_s=0.001, plan_s=0.002, launch_s=0.003, sample_s=0.004,
    ))
    assert db.lookup(_key(), 0).cpu_s == pytest.approx(0.010)


@pytest.mark.parametrize("value,buckets,expected", [
    (3, {1, 2, 4, 8}, 4),
    (4, {1, 2, 4, 8}, 4),
    (1, {1, 2, 4, 8}, 1),
    (9, {1, 2, 4, 8}, None),   # capture miss → caller must model eager
    (5, set(), None),
])
def test_pad_to_bucket(value, buckets, expected):
    assert pad_to_bucket(value, buckets) == expected


def test_coverage_describe_is_readable():
    assert Coverage.EXACT.describe() == "exact"
    combined = Coverage.INTERPOLATED | Coverage.MISSING
    assert "interpolated" in combined.describe()
    assert "missing" in combined.describe()
