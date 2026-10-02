"""Tests for the phase parser and segmenter.

These are the reviewable part: segmentation decides which numbers get averaged
together, so it is asserted against hand-written logs rather than trusted.
"""
from phases import Record, parse_line, parse_log, render, segment

LINE = (
    "2026-09-23 02:15:01,123 INFO [worker_0] mstar.worker.worker: "
    "Worker worker_0 phase-timing iter=300 bs=15.94: "
    "await_gpu: p50=0.00ms p95=0.01ms mean=0.00ms n=100 | "
    "iter_total: p50=6.55ms p95=7.99ms mean=6.68ms n=100"
)


def _rec(worker, it, bs, mean=1.0, n=100):
    return Record(worker=worker, iter=it, bs=bs,
                  phases={"iter_total": __import__("phases").Phase(
                      mean, mean, mean, n)})


def test_parses_a_record():
    r = parse_line(LINE)
    assert r is not None
    assert r.worker == "worker_0" and r.iter == 300 and r.bs == 15.94
    assert r.phases["iter_total"].mean == 6.68
    assert r.phases["iter_total"].n == 100
    assert r.ts is not None


def test_ignores_unrelated_lines():
    assert parse_line("2026-09-23 INFO [worker_0] something else") is None
    assert parse_line("") is None
    # a flush with no parseable fields is not a record
    assert parse_line("Worker worker_0 phase-timing iter=1 bs=1.0: junk") is None


def test_bs_is_optional_so_older_logs_still_parse():
    old = LINE.replace(" bs=15.94", "")
    r = parse_line(old)
    assert r is not None and r.bs == 0.0


def test_steady_state_is_one_segment():
    recs = [_rec("w0", i * 100, 16.0) for i in range(1, 11)]
    segs = segment(recs, bs_tolerance=0.15)
    assert len(segs) == 1
    assert len(segs[0].records) == 10


def test_drain_starts_a_new_segment():
    # ten records at full concurrency, then a drain down to one
    recs = [_rec("w0", i * 100, 16.0) for i in range(1, 11)]
    recs += [_rec("w0", 1100 + i * 100, bs) for i, bs in enumerate([8.0, 4.0, 1.0])]
    segs = segment(recs, bs_tolerance=0.15, min_records=1)
    assert len(segs) > 1, "the drain must not be averaged into steady state"
    steady = max(segs, key=lambda s: len(s.records))
    assert steady.mean_bs == 16.0
    assert all(r.bs == 16.0 for r in steady.records)


def test_min_records_drops_brief_transitions():
    recs = [_rec("w0", i * 100, 16.0) for i in range(1, 11)]
    recs += [_rec("w0", 1100, 2.0)]              # one-off blip
    recs += [_rec("w0", i * 100, 16.0) for i in range(12, 22)]
    segs = segment(recs, bs_tolerance=0.15, min_records=3)
    assert all(s.mean_bs == 16.0 for s in segs), \
        "the single-record blip should be dropped, not reported"


def test_skip_warmup_drops_leading_records_per_worker():
    recs = ([_rec("w0", i * 100, 16.0) for i in range(1, 6)]
            + [_rec("w1", i * 100, 16.0) for i in range(1, 6)])
    segs = segment(recs, skip_warmup=2)
    assert {s.worker for s in segs} == {"w0", "w1"}
    for s in segs:
        assert len(s.records) == 3
        assert s.records[0].iter == 300


def test_workers_are_segmented_independently():
    recs = [_rec("w0", i * 100, 16.0) for i in range(1, 6)]
    recs += [_rec("w1", i * 100, 4.0) for i in range(1, 6)]
    segs = segment(recs)
    assert len(segs) == 2
    assert {round(s.mean_bs) for s in segs} == {16, 4}


def test_mean_is_n_weighted_and_exact():
    import phases as P
    recs = [
        Record("w0", 100, 16.0, {"x": P.Phase(1.0, 1.0, 1.0, 10)}),
        Record("w0", 200, 16.0, {"x": P.Phase(2.0, 2.0, 2.0, 90)}),
    ]
    s = segment(recs)[0].summary()["x"]
    # (1*10 + 2*90) / 100 = 1.9, not the unweighted 1.5
    assert abs(s["mean_ms"] - 1.9) < 1e-9
    assert s["samples"] == 100


def test_render_mentions_batch_and_iters():
    out = render(segment([_rec("w0", 100, 16.0), _rec("w0", 200, 16.0)]))
    assert "worker_0" not in out and "w0" in out
    assert "batch=16.00" in out and "iters 100-200" in out


def test_end_to_end_from_log_text():
    log = "\n".join([
        "irrelevant startup line",
        LINE,
        LINE.replace("iter=300", "iter=400"),
    ])
    recs = parse_log(log)
    assert len(recs) == 2
    assert len(segment(recs)) == 1


def test_request_steps_is_bs_times_iters():
    """Work normalisation: a segment's request-steps is the sum over records
    of bs x iter_total samples, which is what makes phase cost comparable
    between runs whose iteration mix differs (e.g. orpheus' snac decoder)."""
    log = (
        "2026-09-23 21:17:31,409 INFO [worker_0] mstar.worker.worker: "
        "Worker worker_0 phase-timing iter=100 bs=10.00: "
        "iter_total: p50=1.00ms p95=1.00ms mean=1.00ms n=100 | "
        "foo: p50=0.50ms p95=0.50ms mean=0.50ms n=100\n"
        "2026-09-23 21:17:32,409 INFO [worker_0] mstar.worker.worker: "
        "Worker worker_0 phase-timing iter=200 bs=10.00: "
        "iter_total: p50=1.00ms p95=1.00ms mean=1.00ms n=100 | "
        "foo: p50=0.50ms p95=0.50ms mean=0.50ms n=100\n"
    )
    segs = segment(parse_log(log), skip_warmup=0, bs_tolerance=1.0)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.request_steps == 2000.0          # 2 records x 10.00 bs x 100
    # foo: 200 samples x 0.5ms = 100ms over 2000 steps = 50us per 1k steps.
    # foo total 200 x 0.5ms = 100ms; 100ms / 2000 steps x 1e6 = 50000 us/1k.
    assert seg.summary()["foo"]["us_per_1k"] == 50_000.0


def test_us_per_1k_is_zero_without_bs():
    """A log from a worker predating the bs field still parses; the work
    column is simply absent rather than wrong."""
    log = (
        "2026-09-23 21:17:31,409 INFO [worker_0] mstar.worker.worker: "
        "Worker worker_0 phase-timing iter=100: "
        "iter_total: p50=1.00ms p95=1.00ms mean=1.00ms n=100\n"
    )
    segs = segment(parse_log(log), skip_warmup=0, bs_tolerance=1.0)
    assert segs[0].request_steps == 0.0
    assert segs[0].summary()["iter_total"]["us_per_1k"] == 0.0
