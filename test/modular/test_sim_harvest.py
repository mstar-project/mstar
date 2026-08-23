"""Unit tests for turning step traces into a cost table.

The load-bearing behaviors are the ones that decide *which* observations
become a number: skipping records that cannot be attributed, bucketing KV so
repeated observations actually aggregate, and using a median so the long
right tail of a step-time distribution does not drag the estimate up.
"""

import os
import tempfile

import pytest

from mstar.sim.harvest import bucket_kv, harvest_records
from mstar.sim.stepdb import Coverage, StepDB, StepKey


def _rec(**over):
    base = dict(
        node="LLM", graph_walk="decode", bs=4, real_bs=4,
        real_num_tokens=4, padded_bs=4, padded_num_tokens=4,
        mode="graph", kv_len_total=1000, num_sub_batches=1,
        tp_size=1, sp_size=1, requires_cfg=False,
        gpu_s=0.005, prepare_s=0.0001, plan_s=0.0002,
        launch_s=0.0003, sample_s=0.0004, total_s=0.006,
    )
    base.update(over)
    return base


@pytest.fixture
def db():
    d = tempfile.mkdtemp()
    database = StepDB(os.path.join(d, "step.db"), gpu_name="TEST-GPU")
    yield database
    database.close()


def test_repeated_observations_collapse_to_one_row(db):
    report = harvest_records([_rec() for _ in range(10)], db, model="m")
    assert report.records_used == 10
    assert report.rows_written == 1
    assert db.count() == 1


def test_estimator_is_the_median_not_the_mean(db):
    # Nine ordinary steps and one 20x outlier — the shape of a real capture,
    # where the first replay of a bucket pays warmup. A mean would be
    # dragged ~3x; a median must not move.
    recs = [_rec(gpu_s=0.005) for _ in range(9)] + [_rec(gpu_s=0.100)]
    harvest_records(recs, db, model="m")
    cost = db.lookup(
        StepKey(model="m", node="LLM", graph_walk="decode",
                padded_bs=4, padded_num_tokens=4),
        bucket_kv(1000),
    )
    assert cost.gpu_s == pytest.approx(0.005)


def test_records_without_gpu_time_are_skipped(db):
    report = harvest_records([_rec(gpu_s=None)], db, model="m")
    assert report.skipped_no_gpu_time == 1
    assert report.rows_written == 0


def test_records_without_a_shape_are_skipped(db):
    report = harvest_records([_rec(mode=None, padded_bs=None)], db, model="m")
    assert report.skipped_no_shape == 1


def test_split_forwards_are_skipped(db):
    # One GPU window spanning several shapes cannot be attributed to any
    # single one, so it must not become a row.
    report = harvest_records([_rec(num_sub_batches=3)], db, model="m")
    assert report.skipped_split == 1
    assert report.rows_written == 0


def test_distinct_shapes_become_distinct_rows(db):
    harvest_records(
        [_rec(padded_bs=1, padded_num_tokens=1),
         _rec(padded_bs=8, padded_num_tokens=8)],
        db, model="m",
    )
    assert db.count() == 2


def test_kv_bucketing_aggregates_neighbouring_lengths(db):
    # Decode KV grows one token per step; without bucketing every step is
    # its own row and nothing ever aggregates.
    recs = [_rec(kv_len_total=600 + i) for i in range(50)]
    report = harvest_records(recs, db, model="m", kv_bucket=512)
    assert report.rows_written == 1


def test_kv_bucketing_still_separates_across_a_boundary(db):
    # Aggregation is per bucket, so a run that crosses a boundary yields two
    # rows — which is the point: distant KV lengths should not be averaged.
    recs = [_rec(kv_len_total=1000 + i) for i in range(50)]  # spans 1023/1024
    report = harvest_records(recs, db, model="m", kv_bucket=512)
    assert report.rows_written == 2


def test_kv_bucket_centres_the_value():
    assert bucket_kv(0) == 0
    assert bucket_kv(None) == 0
    assert bucket_kv(100, 512) == 256
    assert bucket_kv(600, 512) == 768


def test_min_observations_filters_thin_points(db):
    recs = [_rec(kv_len_total=1000)] + [_rec(kv_len_total=9000) for _ in range(5)]
    report = harvest_records(recs, db, model="m", min_observations=3)
    assert report.rows_written == 1


def test_harvested_rows_are_queryable_end_to_end(db):
    harvest_records([_rec() for _ in range(4)], db, model="m")
    cost = db.lookup(
        StepKey(model="m", node="LLM", graph_walk="decode",
                padded_bs=4, padded_num_tokens=4),
        bucket_kv(1000),
    )
    assert cost.coverage == Coverage.EXACT
    assert cost.gpu_s == pytest.approx(0.005)
    assert cost.cpu_s == pytest.approx(0.001)
