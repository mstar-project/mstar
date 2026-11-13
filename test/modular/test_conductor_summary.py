"""The conductor's INFO summary, which stands in for the per-request lines it
logs at DEBUG."""
import pytest

pytest.importorskip("torch")

from mstar.conductor import conductor as conductor_mod  # noqa: E402
from mstar.conductor.conductor import Conductor  # noqa: E402


def _conductor(requests=(), waiting=()) -> Conductor:
    c = Conductor.__new__(Conductor)
    c.requests = dict.fromkeys(requests)
    c.waiting_queue = list(waiting)
    return c


def _summaries(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("Last ")]


def test_nothing_is_summarized_before_the_run_loop_starts_or_the_period_ends(caplog):
    c = _conductor(requests=["a"])
    with caplog.at_level("INFO", logger=conductor_mod.logger.name):
        c._maybe_log_summary()                  # the loop hasn't started
        c._next_summary = conductor_mod.time.perf_counter() + 60
        c._done_since_summary = 3
        c._maybe_log_summary()                  # the period hasn't ended
    assert _summaries(caplog) == []


def test_a_period_logs_what_finished_queued_runs_and_waits(caplog):
    c = _conductor(requests=["a", "b"], waiting=["c"])
    c._next_summary = 0.0
    c._done_since_summary, c._queued_since_summary = 5, 2
    with caplog.at_level("INFO", logger=conductor_mod.logger.name):
        c._maybe_log_summary()
    assert _summaries(caplog) == ["Last 60 s: 5 requests done, 2 queued for admission; 2 running, 1 waiting"]
    assert (c._done_since_summary, c._queued_since_summary) == (0, 0)
    assert c._next_summary > conductor_mod.time.perf_counter() + 59


def test_a_server_that_stopped_finishing_still_logs_and_an_idle_one_does_not(caplog):
    stalled, idle = _conductor(requests=["a"], waiting=["b", "c"]), _conductor()
    stalled._next_summary = idle._next_summary = 0.0
    with caplog.at_level("INFO", logger=conductor_mod.logger.name):
        stalled._maybe_log_summary()
        idle._maybe_log_summary()
    assert _summaries(caplog) == ["Last 60 s: 0 requests done, 0 queued for admission; 1 running, 2 waiting"]


def test_the_wait_for_messages_ends_at_the_summary():
    c = _conductor()
    c._next_liveness_check = conductor_mod.time.perf_counter() + 30
    c._draining_deadlines = c._early_reads_done_deadlines = c._early_abort_deadlines = []
    c._next_summary = conductor_mod.time.perf_counter() + 1
    assert c._ms_to_next_timed_check() <= 1000
