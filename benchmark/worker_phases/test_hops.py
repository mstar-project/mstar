"""The hop-stamp parser over strings, and the profiler's stamp lines through it."""
import logging

from benchmark.worker_phases import hops
from mstar.utils import profiler

LOG = (
    "2026-10-29 12:00:00,001 INFO [api_server] mstar.hops: hop-stamps wall_offset=1000.000000: "
    "r#1 api.http_in 10.000000; r#1 api.submit 10.000100; r#2 api.http_in 20.000000\n"
    "2026-10-29 12:00:00,002 INFO [worker_0] mstar.hops: hop-stamps wall_offset=1000.000000: "
    "r#1 wk.ingest 10.002000; r#1 wk.ready 10.003000; r#1 wk.ready 10.002500\n"
    "unrelated line\n"
    "2026-10-29 12:00:00,003 INFO [api_server] mstar.hops: hop-stamps wall_offset=1000.000000: "
    "r#1 api.first_out 10.010000; r#2 api.submit 20.000200; garbage\n"
)


def test_parse_keeps_the_earliest_stamp_per_hop():
    reqs = hops.parse(LOG)
    assert set(reqs) == {"r#1", "r#2"}
    assert reqs["r#1"].stamps["wk.ready"] == 10.0025
    assert reqs["r#1"].wall("api.http_in") == 1010.0


def test_each_hop_is_timed_from_the_last_one_reached():
    rows = {r.hop: r for r in hops.hop_table(hops.parse(LOG))}
    assert round(rows["api.submit"].mean_ms, 3) == round((0.0001 + 0.0002) / 2 * 1e3, 3)
    assert rows["api.submit"].n == 2
    # r#1 has no data worker or conductor stamps: ingest is timed from submit
    assert round(rows["wk.ingest"].p50_ms, 3) == 1.9
    assert round(rows["api.first_out"].p50_ms, 3) == 7.5
    # only r#1 reached the last hop
    assert rows["total"].n == 1 and round(rows["total"].p50_ms, 3) == 10.0


def test_window_selects_by_first_stamp():
    reqs = hops.parse(LOG)
    assert set(hops.in_window(reqs, 1015.0, 1025.0)) == {"r#2"}


def test_the_profilers_lines_parse(monkeypatch, caplog):
    monkeypatch.setattr(profiler, "PHASE_PERIOD", 1)
    stamps = profiler._HopStamps()
    stamps._FLUSH_N = 3
    with caplog.at_level(logging.INFO, logger="mstar.hops"):
        stamps.stamp("a#0", "api.http_in", 5.0)
        stamps.stamp("a#0", "api.submit", 5.25)
        stamps.stamp("a#0", "api.first_out", 5.5)
    reqs = hops.parse("\n".join(r.getMessage() for r in caplog.records))
    assert reqs["a#0"].stamps == {"api.http_in": 5.0, "api.submit": 5.25, "api.first_out": 5.5}


def test_stamps_are_free_with_timing_off(monkeypatch):
    monkeypatch.setattr(profiler, "PHASE_PERIOD", 0)
    stamps = profiler._HopStamps()
    stamps.stamp("a#0", "api.http_in")
    assert stamps._buf == []
