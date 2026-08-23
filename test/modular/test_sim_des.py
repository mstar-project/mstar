"""Unit tests for the simulator's timing model and event loop.

These use a hand-built deployment rather than a real model, so they run
anywhere and pin the behaviors that a wrong answer would be quiet about:
the pipeline settling at max(GPU, CPU), the clock never going backwards,
and coverage flags surviving into the report.
"""

import os
import tempfile

import pytest

from mstar.sim.des import Calendar, EventType, Simulator, SimWorker, TimingModel
from mstar.sim.stepdb import Coverage, StepDB, StepKey, StepSample


class _FakeDeployment:
    """The narrow surface Simulator needs, without loading a model."""

    def __init__(self):
        self.model_key = "fake"
        self.ranks = [0]
        self.node_engine_types = {"N": "kv_cache"}
        self.walk_to_wgs = {}
        self.partitions = []
        self.partition_topology = None
        self.max_concurrent_requests = None
        self.max_output_tokens = 128

    def tp_size_for(self, node):
        return 1

    def sp_size_for(self, node):
        return 1


@pytest.fixture
def sim():
    d = tempfile.mkdtemp()
    db = StepDB(os.path.join(d, "s.db"), gpu_name="T")
    yield Simulator(_FakeDeployment(), db, timing=TimingModel())
    db.close()


def _add(db, gpu_s, cpu_each=0.0, bs=1):
    db.add(StepSample(
        StepKey(model="fake", node="N", graph_walk="w",
                padded_bs=bs, padded_num_tokens=bs),
        kv_len_total=0, gpu_s=gpu_s,
        prepare_s=cpu_each, plan_s=cpu_each,
        launch_s=cpu_each, sample_s=cpu_each,
    ))


# ── calendar ─────────────────────────────────────────────────────────────

def test_calendar_pops_in_time_order():
    cal = Calendar()
    cal.push(3.0, EventType.STEP_DONE, "c")
    cal.push(1.0, EventType.STEP_DONE, "a")
    cal.push(2.0, EventType.STEP_DONE, "b")
    assert [cal.pop().payload for _ in range(3)] == ["a", "b", "c"]


def test_calendar_never_schedules_into_the_past():
    # A negative-duration cost from a bad extrapolation must not rewind the
    # clock; every later measurement would be corrupted.
    cal = Calendar()
    cal.push(5.0, EventType.STEP_DONE, "x")
    cal.pop()
    cal.push(1.0, EventType.STEP_DONE, "y")
    assert cal.pop().time == 5.0


def test_same_timestamp_events_resolve_by_kind():
    cal = Calendar()
    cal.push(1.0, EventType.WORKER_POLL, "poll")
    cal.push(1.0, EventType.ARRIVAL, "arrival")
    assert cal.pop().payload == "arrival"


# ── the pipeline model ───────────────────────────────────────────────────

def test_steady_state_cadence_is_max_of_gpu_and_cpu(sim):
    """The load-bearing property: overlap, not addition.

    A step with 10 ms GPU and 4 ms total CPU must settle at 10 ms per step,
    not 14 ms. Modeling it as a sum inflates every downstream latency.
    """
    _add(sim.db, gpu_s=0.010, cpu_each=0.001)   # 4 x 1 ms CPU
    sim.timing.worker_step_overhead_s = 0.0
    w = sim.workers[0]

    starts = []
    for _ in range(6):
        cost = sim.step_cost("N", "w", 1, 1, 0)
        build_s = cost.prepare_s + cost.plan_s
        after_s = cost.launch_s + cost.sample_s
        build_end = max(w.cpu_free_s, 0.0) + build_s
        gpu_start = max(w.gpu_free_s, build_end)
        starts.append(gpu_start)
        w.gpu_free_s = gpu_start + cost.gpu_s
        w.cpu_free_s = gpu_start + after_s

    cadence = starts[-1] - starts[-2]
    assert cadence == pytest.approx(0.010, abs=1e-6)


def test_cpu_bound_step_sets_the_cadence(sim):
    """When CPU exceeds GPU, the CPU lane is the bottleneck."""
    _add(sim.db, gpu_s=0.001, cpu_each=0.002)   # 1 ms GPU, 8 ms CPU
    sim.timing.worker_step_overhead_s = 0.0
    w = sim.workers[0]

    starts = []
    for _ in range(6):
        cost = sim.step_cost("N", "w", 1, 1, 0)
        build_end = w.cpu_free_s + cost.prepare_s + cost.plan_s
        gpu_start = max(w.gpu_free_s, build_end)
        starts.append(gpu_start)
        w.gpu_free_s = gpu_start + cost.gpu_s
        w.cpu_free_s = gpu_start + cost.launch_s + cost.sample_s
    cadence = starts[-1] - starts[-2]
    assert cadence == pytest.approx(0.008, abs=1e-6)


# ── cost lookup ──────────────────────────────────────────────────────────

def test_batch_is_padded_to_the_captured_bucket(sim):
    _add(sim.db, gpu_s=0.004, bs=4)
    cost = sim.step_cost("N", "w", bs=3, num_tokens=3, kv_len_total=0)
    assert cost.gpu_s == pytest.approx(0.004)
    assert not (cost.coverage & Coverage.MISSING)


def test_capture_miss_is_reported_when_no_bucket_fits(sim):
    _add(sim.db, gpu_s=0.004, bs=4)
    cost = sim.step_cost("N", "w", bs=64, num_tokens=64, kv_len_total=0)
    assert cost.coverage & Coverage.MISSING
    assert sim.coverage & Coverage.MISSING


def test_missing_costs_are_counted_for_the_report(sim):
    sim.step_cost("N", "w", 1, 1, 0)
    sim.step_cost("N", "w", 1, 1, 0)
    assert sim.missing_keys["N/w"] == 2


def test_eager_rows_answer_when_no_graph_bucket_fits(sim):
    # A capture miss falls back to the eager regime, as the engine does.
    sim.db.add(StepSample(
        StepKey(model="fake", node="N", graph_walk="w",
                padded_bs=9, padded_num_tokens=9, mode="eager"),
        kv_len_total=0, gpu_s=0.02,
    ))
    _add(sim.db, gpu_s=0.004, bs=4)
    cost = sim.step_cost("N", "w", bs=9, num_tokens=9, kv_len_total=0)
    assert cost.gpu_s == pytest.approx(0.02)


# ── worker batching ──────────────────────────────────────────────────────

def test_worker_picks_least_recently_run_node():
    """Round-robin over (node, walk), matching MicroScheduler's rule."""
    from mstar.sim.des import ReadyItem

    w = SimWorker(0, sim=None)
    w.add_ready(ReadyItem("r1", "A", "w", "g", 0.0))
    w.add_ready(ReadyItem("r2", "B", "w", "g", 0.0))
    w.last_run[("A", "w")] = 5
    w.last_run[("B", "w")] = 1
    (node, _), _ = w.pick_batch()
    assert node == "B"


def test_worker_batches_every_ready_request_for_one_key():
    from mstar.sim.des import ReadyItem

    w = SimWorker(0, sim=None)
    for i in range(4):
        w.add_ready(ReadyItem(f"r{i}", "A", "w", "g", 0.0))
    w.add_ready(ReadyItem("other", "B", "w", "g", 0.0))
    (node, _), items = w.pick_batch()
    assert len(items) == 4 and node == "A"


def test_picking_a_batch_clears_it_from_the_queue():
    from mstar.sim.des import ReadyItem

    w = SimWorker(0, sim=None)
    w.add_ready(ReadyItem("r1", "A", "w", "g", 0.0))
    w.pick_batch()
    assert w.pick_batch() is None
