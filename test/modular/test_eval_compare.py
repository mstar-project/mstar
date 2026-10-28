"""benchmark.eval compare on CPU: the e2e manifest, base-vs-head checks, verdicts (perf noise, A/A floors and
correctness limits, length-matched latency, ABBA reruns), calibration, the e2e suite against a fake server, the
GPU-slot queue, and one model run end to end through the SSH code path with a fake transport and fake servers."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

pytest.importorskip("aiohttp")

from benchmark.eval.e2e import calibrate, checks, compare, hosts, report, trees  # noqa: E402
from benchmark.eval.e2e import manifest as mf  # noqa: E402
from benchmark.eval.e2e.manifest import Case, Model  # noqa: E402
from benchmark.eval.e2e.report import Perf, Stat  # noqa: E402

RUN = {"id": "r", "base": {"label": "main", "sha": "a" * 40}, "head": {"label": "PR #1", "sha": "b" * 40},
       "gpu": "h200", "perf_pct": 5, "noise_sigma": 2, "gpu_hours": 1.0}

# ---------------------------------------------------------------- manifest

BASE = {
    "defaults": {"conc": [1, 8], "repeats": 3, "thresholds": {"perf_pct": 5}, "env": {"A": 1}},
    "datasets": {"tts": "tts/manifest.jsonl", "clone": "clone/manifest.jsonl"},
    "models": {
        "kokoro": {"serve": "kokoro", "family": "TTS", "tags": ["tts", "small"],
                   "cases": [{"id": "tts", "suite": "tts", "data": "tts", "check": {"kind": "wer", "max_increase": 1},
                              "perf": {"latency": "e2e_s_p50", "throughput": "audio_s_per_s"}},
                             {"id": "inv", "invariants": "speech", "optional": True,
                              "check": {"kind": "status_match"}}]},
        "z": {"serve": "z_image_turbo", "family": "Image", "tags": ["image-gen"], "est_min": 3,
              "thresholds": {"perf_pct": 8},
              "cases": [{"id": "t2i", "client": "image", "conc": [1], "args": ["--size", 512],
                         "check": {"kind": "psnr", "min_db": 30}, "perf": {"latency": "lat_p50_s"}}]},
    },
}


def test_manifest_defaults_and_overrides():
    m = mf.parse(copy.deepcopy(BASE))
    k, z = m.models["kokoro"], m.models["z"]
    assert k.cases[0].conc == [1, 8] and k.cases[0].repeats == 3
    assert z.cases[0].conc == [1] and z.cases[0].args == ["--size", "512"]
    assert z.est_min == 3 and k.est_min == 10 and k.env == {"A": "1"}
    assert z.thresholds["perf_pct"] == 8 and k.thresholds["perf_pct"] == 5
    assert Model.from_dict(json.loads(json.dumps(z.to_dict()))) == z


def test_manifest_select_tags_names_exclusions_optional():
    m = mf.parse(copy.deepcopy(BASE))
    assert [x.name for x in m.select("all")] == ["kokoro", "z"]
    assert [x.name for x in m.select("tag:tts")] == ["kokoro"]
    assert [x.name for x in m.select("tag:image")] == ["z"]          # family match
    assert [x.name for x in m.select("all,-tag:small")] == ["z"]
    assert [c.id for c in m.select("kokoro")[0].cases] == ["tts"]
    assert [c.id for c in m.select("kokoro", include_optional=True)[0].cases] == ["tts", "inv"]
    with pytest.raises(mf.ManifestError, match="unknown model"):
        m.select("nope")
    raw = copy.deepcopy(BASE)
    raw["models"]["kokoro_mc"] = {**copy.deepcopy(raw["models"]["kokoro"]), "optional": True,
                                  "tags": ["tts", "mooncake"]}
    m = mf.parse(raw)
    assert [x.name for x in m.select("all")] == ["kokoro", "z"]
    assert [x.name for x in m.select("tag:tts")] == ["kokoro"]           # a tag regular models share
    assert [x.name for x in m.select("tag:mooncake")] == ["kokoro_mc"]


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r["models"]["kokoro"]["cases"][0].pop("check"), "missing 'check'"),
    (lambda r: r["models"]["kokoro"]["cases"][0].update(client="image"), "exactly one of"),
    (lambda r: r["models"]["kokoro"]["cases"][0].update(data="nope"), "unknown dataset"),
    (lambda r: r["models"]["kokoro"]["cases"][0]["check"].update(kind="vibes"), "check.kind"),
    (lambda r: r["models"]["kokoro"]["cases"][0]["check"].pop("max_increase"), "needs max_increase"),
    (lambda r: r["models"]["z"]["cases"][0].update(client="teleport"), "client must be one of"),
    (lambda r: r["models"]["z"]["cases"].append(dict(r["models"]["z"]["cases"][0])), "duplicate case ids"),
    (lambda r: r["models"]["z"]["cases"][0].update(conc=[0]), "positive ints"),
    (lambda r: r["models"]["z"]["cases"][0].update(perf={}), "perf needs"),
    (lambda r: r["models"]["z"]["cases"][0].update(data="nope"), "unknown dataset"),
])
def test_manifest_validation_errors(mutate, msg):
    raw = copy.deepcopy(BASE)
    mutate(raw)
    with pytest.raises(mf.ManifestError, match=msg):
        mf.parse(raw)


def test_shipped_manifest():
    m = mf.load()
    regular = [x for x in m.models.values() if not x.optional]
    assert len(regular) == 16
    assert {c.kind for x in m.models.values() for c in x.cases} >= {"tts", "asr", "duplex", *mf.CLIENT_KINDS}
    assert {c.check["kind"] for x in m.models.values() for c in x.cases} == set(mf.CHECKS)
    mc = m.select("tag:mooncake")
    assert {x.name for x in mc} == {"qwen3_omni_mooncake", "orpheus_mooncake"}   # not the 2- and 3-GPU w2w entries
    assert all(x.optional and x.serve_args[:2] == ["--tensor-comm-protocol", "TCP"] for x in mc)
    for x in m.models.values():
        for c in x.cases:
            assert not c.data or c.data in m.datasets
    # the shipped calibration is in the tree and every entry names a shipped model and case
    cases = {(x.name, c.id) for x in m.models.values() for c in x.cases}
    shipped = calibrate.load()
    assert len(shipped) > 100 and calibrate.DEFAULT.suffix == ".yaml"
    for key in (k for k in shipped if not k.startswith("_")):
        model, case = key.split(":", 1)[1].split("/")[:2]
        assert (model, case) in cases, key


# ---------------------------------------------------------------- verdicts

def perf(base, head, lower=True, metric="lat_p50_s"):
    return report.judge_perf(Perf(metric, 1, Stat.of(base), Stat.of(head), lower), perf_pct=5, noise_sigma=2)


def test_judge_perf_same_regression_better_noisy():
    assert perf([1.0, 1.0, 1.0], [1.02, 1.02, 1.02]).verdict == "same"
    p = perf([1.0, 1.0, 1.0], [1.10, 1.10, 1.10])
    assert p.verdict == "regression" and p.delta_pct == pytest.approx(-10)
    assert perf([1.0, 1.0, 1.0], [0.8, 0.8, 0.8]).verdict == "better"
    assert perf([1.0, 1.4, 0.6], [1.1, 1.5, 0.7]).verdict == "noisy"     # +10% but the spread is wider
    t = perf([10, 10, 10], [9, 9, 9], lower=False, metric="req_per_s")
    assert t.verdict == "regression" and t.delta_pct == pytest.approx(-10)
    assert perf([], [1.0]).verdict == "n/a"
    assert report.lower_is_better("lat_p50_s") and report.lower_is_better("ttfa_ms_p50")
    assert not any(map(report.lower_is_better, ("req_per_s", "audio_s_per_s", "take_turn_rate", "rtfx")))


def case(kind, **ck):
    return Case(id="c", modality="m", driver="client", kind="image", check={"kind": kind, **ck})


@pytest.mark.parametrize("c,facts,errors,want", [
    (case("text_match", min_match=0.5), [{"text_n": 16, "text_identical": 15}], {}, "pass"),
    (case("text_match", min_match=0.5), [{"text_n": 16, "text_identical": 4}], {}, "FAIL"),
    (case("text_match", min_match=0.5), [{"text_n": 16, "text_identical": 16}], {"base": 0, "head": 2}, "FAIL"),
    (case("wer", max_increase=1.0), [{"wer_base": 0.020, "wer_head": 0.025}], {}, "pass"),
    (case("wer", max_increase=1.0), [{"wer_base": 0.020, "wer_head": 0.050}], {}, "FAIL"),
    (case("psnr", min_db=30), [{"psnr_min_db": math.inf, "psnr_median_db": math.inf}], {}, "pass"),
    (case("psnr", min_db=30), [{"psnr_min_db": 22.0, "psnr_median_db": 35.0}], {}, "FAIL"),
    (case("psnr", min_db=30), [{"psnr_min_db": None, "psnr_median_db": 40.0, "shape_mismatch": 1}], {}, "FAIL"),
    (case("action_diff", tol=0.05), [{"action_max_abs_diff": 0.01}], {}, "pass"),
    (case("action_diff", tol=0.05), [{"action_max_abs_diff": 0.3}], {}, "FAIL"),
    (case("audio_len", pct=10), [{"audio_s_base": 10.0, "audio_s_head": 10.5}], {}, "pass"),
    (case("audio_len", pct=10), [{"audio_s_base": 10.0, "audio_s_head": 5.0}], {}, "FAIL"),
    (case("duplex", tol=0.1), [{"take_turn_rate_base": 0.9, "take_turn_rate_head": 0.85}], {}, "pass"),
    (case("duplex", tol=0.1), [{"take_turn_rate_base": 0.9, "take_turn_rate_head": 0.5}], {}, "FAIL"),
    (case("status_match"), [{"n": 5, "diffs": []}], {}, "pass"),
    (case("status_match"), [{"n": 5, "diffs": [{"case": "x", "base": "200", "head": "500"}]}], {}, "FAIL"),
    (case("psnr", min_db=30), [{"missing": ["base"]}], {}, "incomplete"),
    (case("psnr", min_db=30), [{"missing": ["head"]}], {}, "FAIL"),
    (case("psnr", min_db=30), [], {}, "incomplete"),
])
def test_judge_correct(c, facts, errors, want):
    assert report.judge_correct(c, facts, errors)[0] == want


def test_text_match_accepts_a_late_fork_but_not_truncation_and_reports_base_noise():
    c = case("text_match", min_match=0.5)
    fork = [{"text_n": 16, "text_identical": 0, "text_sim_median": 0.93, "text_len_ratio_median": 0.98}]
    cut = [{"text_n": 16, "text_identical": 0, "text_sim_median": 0.62, "text_len_ratio_median": 0.5}]
    assert report.judge_correct(c, fork, {})[0] == "pass"
    assert report.judge_correct(c, cut, {})[0] == "FAIL"
    _, detail = report.judge_correct(c, [{"text_n": 8, "text_identical": 6},
                                         {"self_base": {"text_n": 8, "text_identical": 7}}], {})
    assert detail == "6/8 identical (base vs base 7/8)"


def test_correct_limit_widens_from_aa_swings_with_the_manifest_as_floor():
    """Qwen3-TTS: main vs main moved WER 3.62% to 1.09% (2.53 points) against a 2-point limit."""
    wer = {"kind": "wer", "max_increase": 2.0}
    ck, note = report.correct_limit(wer, [2.53])
    assert ck["max_increase"] == pytest.approx(3.795) and "from A/A" in note
    assert report.correct_limit(wer, [0.4]) == (wer, "")
    assert report.correct_limit({"kind": "text_match", "min_match": 0.5}, [1.0])[0]["min_match"] == 0.1
    assert report.correct_limit({"kind": "psnr", "min_db": 30}, [32.0, 40.0])[0]["min_db"] == 29.0
    assert report.correct_limit({"kind": "psnr", "min_db": 30}, [12.0])[0]["min_db"] == report.PSNR_FLOOR_DB
    c = Case(id="tts", modality="tts", driver="suite", kind="tts", check=wer)
    facts = [{"wer_base": 0.0109, "wer_head": 0.0362}]
    assert report.judge_correct(c, facts, {})[0] == "FAIL"
    v, detail = report.judge_correct(c, facts, {}, swings=[2.53])
    assert v == "pass" and "limit 3.8 points from A/A (manifest 2 points)" in detail


def test_mixed_directions_are_not_called_faster():
    better = Perf("lat_p50_s", 1, Stat.of([1.0] * 3), Stat.of([0.9] * 3), True)
    worse_noisy = Perf("req_per_s", 8, Stat.of([10, 14, 6]), Stat.of([9, 13, 5]), False)
    r = report.Row(model="m", family="F", case="c", modality="x", starts={"base": True, "head": True}, correct="pass",
                   correct_detail="", latency=report.judge_perf(better, 5, 2),
                   throughput=report.judge_perf(worse_noisy, 5, 2))
    assert r.verdict() == "mixed, within noise"
    assert report.fmt_val("lat_p50_s", 0.4123) == "412 ms" and report.fmt_val("lat_p50_s", 5.18) == "5.18 s"
    assert report.fmt_val("ttfa_ms_p50", 19.73) == "19.7 ms" and report.fmt_val("req_per_s", 4.961) == "4.96 req/s"


# Result directories as compare writes them: results/<model>/<arm>/s<k>/<case>/*-c<N>.json

def chat_model(name="m", kind="chat_text", check=None, conc=(1, 8)):
    tts = kind == "tts"
    p = {"latency": "e2e_s_p50", "throughput": "audio_s_per_s"} if tts else {"latency": "lat_p50_s",
                                                                              "throughput": "req_per_s"}
    return Model(name=name, serve=name, family="Omni", tags=[], thresholds={"perf_pct": 5, "noise_sigma": 2},
                 cases=[Case(id="c", modality="text", driver="suite" if tts else "client", kind=kind, conc=list(conc),
                             perf=p, check=check or {"kind": "text_match", "min_match": 0.5})])


def write_arm(md: Path, arm: str, reqs1, reqsn, wall_n: float, audio=False, session=0, passes=3, errors=0,
              failed=()):
    """reqs: (e2e_s, output length) per request; the pass metrics are what the suites compute. ``failed``: the
    concurrencies at which every request failed (a server answering each one with a 500)."""
    d = md / arm / f"s{session}" / "c"
    d.mkdir(parents=True, exist_ok=True)
    (md / arm / f"s{session}" / "startup.json").write_text(json.dumps({"healthy": True, "startup_s": 10}))
    for conc, reqs, wall in ((1, reqs1, sum(e for e, _ in reqs1)), (8, reqsn, wall_n)):
        rows = []
        for i, (e, ln) in enumerate(reqs):
            r = {"id": f"{i:03d}", "ok": True, "e2e_s": e}
            r.update({"audio_s": ln} if audio else {"text": "x" * int(ln), "n_chunks": int(ln), "audio_s": 0.0})
            rows.append(r)
        lat = sorted(r["e2e_s"] for r in rows)
        p50 = (lat[len(lat) // 2 - 1] + lat[len(lat) // 2]) / 2
        metrics = {"lat_p50_s": p50, "e2e_s_p50": p50, "req_per_s": len(rows) / wall,
                   "audio_s_per_s": sum(ln for _, ln in reqs) / wall, "errors": errors}
        if conc in failed:
            rows = [{"id": r["id"], "ok": False, "error": "HTTP 500"} for r in rows]
            metrics = {"lat_p50_s": None, "e2e_s_p50": None, "req_per_s": 0.0, "audio_s_per_s": 0.0,
                       "errors": len(rows)}
        reps = [{"index": k, "wall_s": wall, "metrics": metrics, "requests": rows} for k in range(passes)]
        (d / f"x-{arm}-c{conc}.json").write_text(json.dumps({"schema": "mstar-eval/1", "suite": "e2e", "label": "x",
                                                             "config": {}, "meta": {}, "repeats": reps,
                                                             "summary": {}}))


def finish(rd: Path, md: Path, facts=None, run=None):
    (md / "done.json").write_text(json.dumps({"status": "done"}))
    (md / "compare.json").write_text(json.dumps({"c": facts or [{"text_n": 8, "text_identical": 8}]}))
    (rd / "run.json").write_text(json.dumps({**RUN, "models": ["m"], **(run or {})}))


def test_rows_markdown_chart_and_summary(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.01, 96)] * 8, [(1.01, 96)] * 8, wall_n=2.02)
    finish(rd, md)
    path, rows = report.write(rd, [chat_model()], chart_url="https://example/x.png", calibration={})
    r = rows[0]
    assert r.correct == "pass" and r.latency.verdict == "same" and not r.flagged and r.no_baseline
    text = path.read_text()
    assert "No regressions" in text and "https://example/x.png" in text and "No A/A noise baseline on H200" in text
    assert "\u2014" not in text and "\u2013" not in text
    assert json.loads((rd / "report" / "summary.json").read_text())[0]["verdict"] == "same"
    pytest.importorskip("matplotlib")
    assert (rd / "report" / "chart.png").stat().st_size > 1000


def test_regression_correctness_and_a_head_that_does_not_start_are_flagged(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.2, 96)] * 8, [(1.2, 96)] * 8, wall_n=2.4)
    finish(rd, md, [{"text_n": 8, "text_identical": 1, "text_sim_median": 0.3, "text_len_ratio_median": 0.4}])
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.flagged and r.correct == "FAIL"
    assert "regression" in r.verdict() and "correctness changed" in r.verdict()
    assert report.flagged_perf_cases([r]) == ["c"]
    assert "1 of 1 rows flagged" in report.render_md([r], json.loads((rd / "run.json").read_text()))
    (md / "head" / "s0" / "startup.json").write_text(json.dumps({"healthy": False}))
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.correct == "FAIL" and r.flagged and "head does not start" in r.verdict()


def test_aa_floor_turns_a_drift_into_noise_and_calibrate_folds_each_run_once(tmp_path):
    rd, md = tmp_path / "aa", tmp_path / "aa" / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.08, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)       # +8% at c=1, tight passes
    finish(rd, md)
    assert report.build_rows(rd, [chat_model()], calibration={})[0].latency.verdict == "regression"
    cal = {"h200:m/c/lat_p50_s@c1": {"deltas": [-6.0, 2.0]}}                  # A/A drift up to 6%: floor 9%
    p = report.build_rows(rd, [chat_model()], calibration=cal)[0].latency
    assert p.floor_pct == 9.0 and p.verdict == "noisy"
    path = tmp_path / "cal.json"
    with pytest.raises(ValueError, match="not an A/A run"):
        calibrate.add_run(rd, [chat_model()], path)
    run = json.loads((rd / "run.json").read_text())
    run["head"]["sha"] = run["base"]["sha"]
    (rd / "run.json").write_text(json.dumps(run))
    assert calibrate.add_run(rd, [chat_model()], path) > 0
    assert calibrate.add_run(rd, [chat_model()], path) == 0
    assert calibrate.floor_pct(calibrate.load(path), "h200:m/c/lat_p50_s@c1") == pytest.approx(12.0)


def test_calibrate_folds_correctness_swings_into_a_run_seen_before(tmp_path):
    rd, md = tmp_path / "aa", tmp_path / "aa" / "results" / "m"
    write_arm(md, "base", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True)
    write_arm(md, "head", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True)
    finish(rd, md, [{"wer_base": 0.0362, "wer_head": 0.0109}], run={"head": RUN["base"], "id": "aa"})
    m = chat_model(kind="tts", check={"kind": "wer", "max_increase": 2})
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"_runs": ["aa"]}))                         # perf folded in before swings existed
    assert calibrate.add_run(rd, [m], path) == 1
    cal = calibrate.load(path)
    assert cal["h200:m/c/correct"] == {"kind": "wer", "swings": [2.53]} and "h200:m/c/e2e_s_p50@c1" not in cal
    rd2, md2 = tmp_path / "pr", tmp_path / "pr" / "results" / "m"
    write_arm(md2, "base", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True)
    write_arm(md2, "head", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True)
    finish(rd2, md2, [{"wer_base": 0.0109, "wer_head": 0.0362}])
    assert report.build_rows(rd2, [m], calibration={})[0].correct == "FAIL"
    assert report.build_rows(rd2, [m], calibration=cal)[0].correct == "pass"


def test_calibrate_leaves_failed_and_errored_rows_out_of_the_floors(tmp_path, monkeypatch, capsys):
    """A drift from requests that errored or a job that broke is not A/A noise: folded in, it would set a floor
    that hides a real regression of that size."""
    rd, models = tmp_path / "aa", []
    for name, head_s, (eb, eh), status in (("ok", 1.03, (0, 0), "done"), ("failing", 0.57, (7, 19), "done"),
                                           ("errored", 1.4, (3, 3), "done"), ("broke", 1.3, (0, 0), "error")):
        md = rd / "results" / name
        write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0, errors=eb)
        write_arm(md, "head", [(head_s, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0, errors=eh)
        (md / "done.json").write_text(json.dumps({"status": status, "reason": "" if status == "done" else "boom"}))
        (md / "compare.json").write_text(json.dumps({"c": [{"text_n": 8, "text_identical": 8}]}))
        models.append(chat_model(name))
    (rd / "run.json").write_text(json.dumps({**RUN, "head": RUN["base"], "models": [m.name for m in models]}))
    rows = {r.model: r for r in report.build_rows(rd, models, calibration={})}
    assert [rows[n].correct for n in ("ok", "failing", "errored", "broke")] == ["pass", "FAIL", "pass", "incomplete"]
    assert all(rows[n].latency.delta_pct is not None for n in rows)          # every row has a drift to fold
    path, excluded = tmp_path / "cal.json", {}
    assert calibrate.add_run(rd, models, path, excluded) > 0
    floors = {k.split(":", 1)[1].split("/", 1)[0] for k in calibrate.load(path) if "@c" in k}
    assert floors == {"ok"}
    assert excluded == {"failing": 1, "errored": 1, "broke": 1}
    monkeypatch.setattr(compare.mf, "load", lambda _: mf.Manifest(models={m.name: m for m in models}, datasets={},
                                                                    defaults={}))
    path.unlink()
    assert compare.cmd_calibrate(Namespace(cases="x", runs=[str(rd)], calibration=str(path))) == 0
    assert capsys.readouterr().out.rstrip().endswith("left out of the floors: broke 1, errored 1, failing 1")


def test_diverged_lengths_compare_matched_requests_at_c1_and_tokens_per_s_at_cmax(tmp_path):
    """Head's greedy text forks into replies twice as long on half the prompts: per pass p50 latency and req/s say
    -50%, while every request of equal length is exactly as fast."""
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 96)] * 4 + [(2.0, 192)] * 4, [(1.0, 96)] * 4 + [(2.0, 192)] * 4, wall_n=3.0)
    finish(rd, md, [{"text_n": 8, "text_identical": 4, "text_sim_median": 0.7, "text_len_ratio_median": 1.5}])
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.latency.basis == "matched length 4/8" and r.latency.verdict == "same"
    assert r.throughput.metric == "tok_per_s" and r.throughput.verdict == "same"
    assert "[matched length 4/8]" in report.fmt_perf(r.latency)


def test_matched_set_is_fixed_across_sessions(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    lats = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    for s in (0, 1):
        write_arm(md, "base", [(x, 96) for x in lats], [(1.0, 96)] * 8, wall_n=2.0, session=s)
    write_arm(md, "head", [(x, 96) for x in lats], [(1.0, 96)] * 8, wall_n=2.0, session=0)
    write_arm(md, "head", [(x, 96) for x in lats[:6]] + [(1.5, 140), (1.6, 150)], [(1.0, 96)] * 8, wall_n=2.0,
              session=1)
    finish(rd, md)
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.latency.basis == "matched length 6/8" and r.latency.base.n == 6
    assert r.latency.base.sd == 0 and r.latency.head.sd == 0 and r.latency.verdict == "same"


def test_too_few_matched_lengths_fall_back_to_per_token_and_tts_per_audio_second(tmp_path):
    rd, md = tmp_path / "a", tmp_path / "a" / "results" / "m"
    write_arm(md, "base", [(1.0, 100)] * 8, [(1.0, 100)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.3, 130)] * 7 + [(1.0, 100)], [(1.3, 130)] * 8, wall_n=2.6)
    finish(rd, md)
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.latency.metric == "s_per_tok_p50" and r.latency.basis.startswith("per token")
    assert r.latency.verdict == "same" and report.fmt_val("s_per_tok_p50", 0.0104) == "10.4 ms/token"
    rd, md = tmp_path / "b", tmp_path / "b" / "results" / "m"
    write_arm(md, "base", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True)
    write_arm(md, "head", [(0.4, 4.0)] * 8, [(0.8, 4.0)] * 8, wall_n=1.6, audio=True)   # 50% slower per audio s
    finish(rd, md, [{"wer_base": 0.02, "wer_head": 0.02}])
    r = report.build_rows(rd, [chat_model(kind="tts", check={"kind": "wer", "max_increase": 2})], calibration={})[0]
    assert r.latency.metric == "s_per_audio_s_p50" and r.latency.verdict == "regression"
    assert r.throughput.metric == "audio_s_per_s" and not r.throughput.basis


def test_matching_lengths_and_fixed_size_outputs_are_left_alone(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 97)] * 8, [(1.0, 97)] * 8, wall_n=2.0)                # within one token
    finish(rd, md)
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.latency.metric == "lat_p50_s" and not r.latency.basis and r.throughput.metric == "req_per_s"
    lat = Perf("lat_p50_s", 1, Stat.of([1.0]), Stat.of([2.0]), True)
    assert report.length_aware(chat_model(kind="image").cases[0], {}, {}, lat, None) == (lat, None)


def test_abba_rerun_replaces_a_one_session_flag(tmp_path):
    """A server process that settles at one of two speeds: one session flags -11%, two sessions ABBA do not."""
    rd, md, rr = tmp_path, tmp_path / "results" / "m", tmp_path / "results" / "m__abba"
    write_arm(md, "base", [(1.25, 96)] * 8, [(1.25, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.385, 96)] * 8, [(1.385, 96)] * 8, wall_n=2.0)
    finish(rd, md)
    assert report.build_rows(rd, [chat_model()], calibration={})[0].flagged
    for s, (b, h) in enumerate(((1.25, 1.385), (1.385, 1.25))):           # the speed level follows the process
        write_arm(rr, "base", [(b, 96)] * 8, [(b, 96)] * 8, wall_n=2.0, session=s)
        write_arm(rr, "head", [(h, 96)] * 8, [(h, 96)] * 8, wall_n=2.0, session=s)
    (rr / "done.json").write_text(json.dumps({"status": "done"}))
    (rr / "compare.json").write_text(json.dumps({"c": [{"text_n": 8, "text_identical": 8}] * 2}))
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert not r.flagged and r.latency.verdict in ("same", "noisy")
    assert r.rerun.startswith("flagged with 1 session (c=1 -11%") and r.latency.base.n == 6
    text = report.render_md([r], json.loads((rd / "run.json").read_text()))
    assert "rerun with two (ABBA)" in text and "(2-session rerun)" in text


# ---------------------------------------------------------------- checks

def test_saved_outputs_text_psnr_actions_audio(tmp_path):
    from PIL import Image

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "000.txt").write_text("same"), (b / "000.txt").write_text("same")
    (a / "001.txt").write_text("x"), (b / "001.txt").write_text("y")
    img = np.zeros((8, 8, 3), np.uint8)
    Image.fromarray(img).save(a / "002.png")
    Image.fromarray(img).save(b / "002.png")
    img2 = img.copy()
    img2[0, 0] = 255
    Image.fromarray(img).save(a / "003.png")
    Image.fromarray(img2).save(b / "003.png")
    np.array([0.0, 1.0], np.float32).tofile(a / "004.f32")
    np.array([0.0, 1.25], np.float32).tofile(b / "004.f32")
    np.zeros(24000, np.int16).tofile(a / "005.pcm")
    np.zeros(12000, np.int16).tofile(b / "005.pcm")
    f = checks.saved(a, b)
    assert f["text_identical"] == 1 and f["text_n"] == 2 and 0 <= f["text_sim_median"] <= 1
    assert f["psnr_median_db"] > 30 and f["psnr_min_db"] < float("inf")
    assert f["action_max_abs_diff"] == 0.25 and f["audio_s_base"] == 1.0 and f["audio_s_head"] == 0.5
    Image.fromarray(np.zeros((4, 8, 3), np.uint8)).save(b / "002.png")
    assert checks.saved(a, b)["shape_mismatch"] == 1
    assert checks.saved(a, tmp_path / "nope")["missing"] == ["head"]


def test_text_facts_fork_and_truncation():
    fox = "the quick brown fox jumps over the lazy dog"
    late = checks.text_facts([(fox, fox.replace("the lazy", "a lazy"))])
    assert late["text_identical"] == 0 and late["text_sim_median"] > 0.9 and late["text_prefix_median"] > 0.7
    cut = checks.text_facts([(fox, "the quick")])
    assert cut["text_len_ratio_min"] < 0.3


def test_rp_and_suite_facts(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "rp.json").write_text(json.dumps({"x": {"status": 200}, "y": {"status_line": "HTTP/1.1 400 Bad"},
                                           "z": {"status": 200, "identical": True}, "_elapsed_s": 1}))
    (b / "rp.json").write_text(json.dumps({"x": {"status": 200}, "y": {"status_line": "HTTP/1.1 200 OK"},
                                           "z": {"status": 200, "identical": False}}))
    f = checks.rp_facts(a, b)
    assert f["n"] == 3 and {d["case"] for d in f["diffs"]} == {"y", "z"}

    def write(d, wer, hyps):
        reqs = [{"id": f"i{k}", "ok": True, "hyp": h} for k, h in enumerate(hyps)]
        (d / "x-c1.json").write_text(json.dumps({"schema": "mstar-eval/1", "suite": "asr", "label": "x", "config": {},
                                                 "meta": {}, "summary": {"wer": {"mean": wer}},
                                                 "repeats": [{"index": 0, "metrics": {}, "requests": reqs}]}))
    write(a, 0.02, ["a", "b", "c"])
    write(b, 0.03, ["a", "b", "d"])
    f = checks.suite_facts(a, b, 1)
    assert f["wer_base"] == 0.02 and f["wer_head"] == 0.03 and f["text_identical"] == 2 and f["text_n"] == 3
    (b / "x-c1.json").write_text((b / "x-c1.json").read_text()[:40])      # truncated head results
    assert checks.suite_facts(a, b, 1) == {"missing": ["head"]}


# ---------------------------------------------------------------- the e2e suite and the CLI

FAKE_SERVE = r'''
import argparse, json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DELAY, SUFFIX, DONE = {delay}, {suffix!r}, {done}
p = argparse.ArgumentParser()
p.add_argument("cmd"); p.add_argument("model"); p.add_argument("--port", type=int)
a, _ = p.parse_known_args()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        words = (req["messages"][0]["content"] + SUFFIX).split()
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close"); self.end_headers()
        for w in words:
            time.sleep(DELAY)
            msg = {{"choices": [{{"delta": {{"content": w + " "}}}}]}}
            self.wfile.write(b"data: " + json.dumps(msg).encode() + b"\n\n"); self.wfile.flush()
        if DONE:
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        self.close_connection = True


ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
'''


def fake_tree(d: Path, delay: float = 0.0, suffix: str = "", done: bool = True) -> Path:
    """A checkout whose `python -m mstar.cli.main serve` is a stdlib chat server streaming the prompt back."""
    (d / "mstar" / "cli").mkdir(parents=True)
    (d / "mstar" / "__init__.py").write_text("")
    (d / "mstar" / "cli" / "__init__.py").write_text("")
    (d / "mstar" / "cli" / "main.py").write_text(FAKE_SERVE.format(delay=delay, suffix=suffix, done=done))
    (d / ".complete").touch()
    return d


def test_e2e_suite_runs_without_data_and_saves_outputs(tmp_path):
    from benchmark.eval.cli import main
    from benchmark.eval.schema import RunResult

    tree = fake_tree(tmp_path / "tree")
    out = tmp_path / "out"
    main(["run", "e2e", "--kind", "chat_text", "--serve-cmd",
          f"{sys.executable} {tree}/mstar/cli/main.py serve fake --port {{port}}", "--limit", "4",
          "--concurrency", "1", "--repeats", "2", "--warmup", "1", "--label", "fake", "--out", str(out)])
    res = RunResult.load(out / "e2e-fake-c1.json")
    assert res.config["data"]["path"] is None and res.config["suite"]["kind"] == "chat_text"
    assert res.summary["errors"]["mean"] == 0 and res.summary["lat_p50_s"]["n"] == 2
    row = res.repeats[0].requests[0]
    assert row["ok"] and row["n_chunks"] == len(row["text"].split()) and "_audio" not in row
    saved = checks.saved_dir(out, 1)
    assert sorted(p.name for p in saved.iterdir()) == ["000.txt", "001.txt", "002.txt", "003.txt"]
    cut = fake_tree(tmp_path / "cut", done=False)                             # replies that end without [DONE]
    main(["run", "e2e", "--kind", "chat_text", "--serve-cmd",
          f"{sys.executable} {cut}/mstar/cli/main.py serve fake --port {{port}}", "--limit", "2",
          "--concurrency", "1", "--repeats", "1", "--warmup", "0", "--label", "cut", "--out", str(out)])
    rows = RunResult.load(out / "e2e-cut-c1.json").repeats[0].requests
    assert [r["error"] for r in rows] == ["incomplete stream"] * 2
    with pytest.raises(SystemExit, match="needs --data"):
        main(["run", "tts", "--url", "http://127.0.0.1:1"])
    with pytest.raises(SystemExit, match="needs --media"):
        main(["run", "e2e", "--kind", "chat_image", "--url", "http://127.0.0.1:1"])


# ---------------------------------------------------------------- slots and the queue

def test_parse_hosts_and_provider_hook(monkeypatch):
    assert hosts.parse_hosts("u@a, u@b:1,u@b:0") == [("u@a", 0), ("u@b", 1), ("u@b", 0)]
    with pytest.raises(ValueError, match="same GPU twice"):
        hosts.parse_hosts("a,a:0")
    mod = type(sys)("fake_launcher")
    released = []

    class Launcher(hosts.HostProvider):
        def hosts(self):
            return "u@vm1,u@vm2:1"

        def release(self):
            released.append(True)

    mod.make = lambda args: Launcher()
    monkeypatch.setitem(sys.modules, "fake_launcher", mod)
    args = Namespace(hosts=None, host_provider="fake_launcher:make")
    provider = compare.make_provider(args)
    slots = compare.make_slots(args, provider)
    assert [s.name for s in slots] == ["u@vm1:0", "u@vm2:1"]
    assert all(isinstance(s.transport, hosts.SSHTransport) for s in slots)
    provider.release()
    assert released == [True]

    class Broken(Launcher):
        def hosts(self):
            raise RuntimeError("quota")

    mod.make = lambda args: Broken()
    from benchmark.eval.cli import build_parser

    cli_args = build_parser().parse_args(["compare", "--head", "main", "--host-provider", "fake_launcher:make",
                                          "--out", "/nonexistent-run-dir"])
    monkeypatch.setattr(trees, "resolve", lambda repo, spec, remote=None: trees.Ref(spec, "a" * 40, spec))
    with pytest.raises(RuntimeError, match="quota"):
        compare.cmd_compare(cli_args)
    assert released == [True, True]                     # released although hosts() failed


def slot(name: str) -> hosts.Slot:
    return hosts.Slot(Namespace(name=name), 0)


def test_queue_longest_first_reruns_first_and_idle_slots_take_queued_work():
    q = hosts.WorkQueue(["small", "big", "mid"], {"small": 1, "big": 30, "mid": 10})
    assert q.items == ["big", "mid", "small"]
    done: dict[str, list[str]] = {"x:0": [], "y:0": []}
    release = threading.Event()

    def run_one(s, name):
        if name == "big":
            release.wait(10)        # whichever slot took "big" holds it until the other has drained the queue
        done[s.name].append(name)
        if name == "small":
            release.set()
        return "mid__abba" if name == "mid" else None

    hosts.run_slots([slot("x"), slot("y")], q, run_one)
    assert sorted(done.values()) == [["big"], ["mid", "mid__abba", "small"]]     # the rerun jumps the queue


def test_a_lost_host_requeues_its_model_and_gives_up_when_every_slot_is_gone():
    q = hosts.WorkQueue(["a", "b"], {"a": 2, "b": 1})
    dead, alive = slot("dead"), slot("alive")
    ran, lost = [], []
    took = threading.Event()

    def run_one(s, name):
        if s is dead:
            took.set()
            raise hosts.HostLostError("ssh: connection refused")
        took.wait(10)               # the dead slot has taken one model before the live one finishes any
        ran.append(name)

    hosts.run_slots([dead, alive], q, run_one, on_lost=lambda s, n, e: lost.append(n))
    assert sorted(ran) == ["a", "b"] and len(lost) == 1
    q2, gave_up = hosts.WorkQueue(["a", "b"], {}), []

    def always_lost(s, name):
        raise hosts.HostLostError("gone")

    hosts.run_slots([slot("only")], q2, always_lost, on_given_up=gave_up.append)
    assert sorted(gave_up) == ["a", "b"]


# ---------------------------------------------------------------- one model, end to end, through the SSH path

class FakeRemote:
    """An "SSH host" on this machine: commands run in its own home, files are copied there, never shared."""

    def __init__(self, home: Path):
        self.name, self.home = "bench@fake-host", home

    def sh(self, script, timeout=None):
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=timeout,
                              cwd=self.home, check=False)

    def push(self, local, remote):   # as SSHTransport.push: the parent first, then rsync
        self.sh(f"mkdir -p {shlex.quote(str(Path(remote).parent))}")
        hosts._rsync([f"{local}/" if local.is_dir() else str(local), remote])

    def pull(self, remote, local, exclude=()):
        local.mkdir(parents=True, exist_ok=True)
        hosts._rsync([*[f"--exclude={e}" for e in exclude], f"{remote}/", f"{local}/"])


def test_one_model_runs_both_arms_on_a_remote_slot_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "POLL_S", 0.2)
    home = tmp_path / "home"
    home.mkdir()
    man = mf.parse({"defaults": {"conc": [1, 2], "repeats": 2, "warmup": 1, "startup_timeout_s": 60}, "models": {
        "fake": {"serve": "fake", "family": "LLM", "cases": [
            {"id": "text", "client": "chat_text", "n": 4, "check": {"kind": "text_match", "min_match": 0.5},
             "perf": {"latency": "lat_p50_s", "throughput": "req_per_s"}}]}}})
    models = man.select("all")
    base, head = trees.Ref("main", "a" * 40, "main"), trees.Ref("x", "b" * 40, "PR #1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({**RUN, "base": base.__dict__, "head": head.__dict__,
                                                  "models": ["fake"]}))
    (tmp_path / "data" / "media").mkdir(parents=True)
    args = Namespace(python=sys.executable, host_python=sys.executable, host_dir="mstar-eval",
                     work_dir=str(tmp_path / "work"), data_dir=str(tmp_path / "data"), sessions=1, repeats=None,
                     no_rerun=True, calibration=str(tmp_path / "cal.json"))
    slot = hosts.Slot(FakeRemote(home), None)
    runner = compare.Runner(args, man, models, base, head, run_dir, [slot])
    runner.cache = tmp_path / "cache"
    fake_tree(runner.cache / "src" / base.sha)
    fake_tree(runner.cache / "src" / head.sha, suffix=" indeed")
    stale = run_dir / "results" / "fake" / "head" / "s0" / "text" / "e2e-old-c2.json"   # a lost attempt's pull
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    assert runner.run_one(slot, "fake") is None
    assert not stale.exists()
    host_dir = home / "mstar-eval"
    assert slot.host_dir == str(host_dir) and (host_dir / "evalfw" / runner.evalfw / "benchmark" / "eval").is_dir()
    res = run_dir / "results" / "fake"
    assert json.loads((res / "done.json").read_text())["status"] == "done"
    assert not list(res.rglob("audio"))                       # saved outputs stay on the host
    assert list((host_dir / "runs" / runner.run_key / "fake.1").rglob("audio/*-c1/r1/000.txt"))     # attempt 1
    assert runner.run_key.startswith("run-") and runner.run_key != compare.Runner(
        args, man, models, base, head, run_dir, [slot]).run_key                    # unique per invocation
    for arm in ("base", "head"):
        info = json.loads((res / arm / "s0" / "startup.json").read_text())
        assert info["healthy"] and info["mstar_from_arm_src"] and info["cases"]["text"]["rc"] == 0
    facts = json.loads((res / "compare.json").read_text())["text"][0]
    assert facts["text_n"] == 4 and facts["text_identical"] == 0 and facts["text_sim_median"] > 0.6
    job = json.loads((run_dir / "jobs" / "fake.json").read_text())
    assert job["order"] == ["base", "head"] and job["arms"]["head"]["src"] == f"{host_dir}/src/{head.sha}"
    row = report.build_rows(run_dir, models, calibration={})[0]
    assert row.starts == {"base": True, "head": True} and row.correct == "pass"
    assert row.latency.base.n == 2 and row.latency.head.n == 2
    assert runner.state["models"]["fake"]["status"] == "done"
    assert runner.state["models"]["fake"]["elapsed_s"] > 1          # polled: the launch did not wait for the job


def test_stopping_compare_stops_the_job_and_its_servers(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "POLL_S", 0.2)
    man = mf.parse({"defaults": {"conc": [1], "repeats": 50, "warmup": 0, "startup_timeout_s": 60}, "models": {
        "fake": {"serve": "fake", "family": "LLM", "cases": [
            {"id": "text", "client": "chat_text", "n": 4, "check": {"kind": "text_match", "min_match": 0.5},
             "perf": {"latency": "lat_p50_s"}}]}}})
    base, head = trees.Ref("main", "a" * 40, "main"), trees.Ref("x", "b" * 40, "PR #1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({**RUN, "models": ["fake"]}))
    (tmp_path / "data").mkdir()
    args = Namespace(python=sys.executable, host_python=sys.executable, host_dir="mstar-eval",
                     work_dir=str(tmp_path / "work"), data_dir=str(tmp_path / "data"), sessions=1, repeats=None,
                     no_rerun=True, calibration=str(tmp_path / "cal.json"))
    slot = hosts.Slot(hosts.LocalTransport(), None)
    runner = compare.Runner(args, man, man.select("all"), base, head, run_dir, [slot])
    runner.cache = tmp_path / "cache"
    for ref in (base, head):
        fake_tree(runner.cache / "src" / ref.sha, delay=0.05)
    out = run_dir / "attempts" / "fake.1"
    th = threading.Thread(target=runner.run_one, args=(slot, "fake"))
    th.start()
    for _ in range(300):
        if '"phase": "case"' in ((out / "status.json").read_text() if (out / "status.json").exists() else ""):
            break
        threading.Event().wait(0.1)
    pid = int((out / "pid").read_text())
    assert procs_under(runner.cache / "src")              # the base server is up
    runner.stop.set()
    th.join(60)
    assert not th.is_alive() and runner.state["models"]["fake"]["status"] == "interrupted"
    assert json.loads((out / "done.json").read_text())["status"] == "interrupted"
    assert json.loads((run_dir / "results" / "fake" / "done.json").read_text())["status"] == "interrupted"
    assert not Path(f"/proc/{pid}").exists()
    assert not procs_under(runner.cache / "src")          # servers run with their checkout as working directory


def procs_under(root: Path) -> list[int]:
    found = []
    for d in Path("/proc").glob("[0-9]*"):
        try:
            if Path(os.readlink(d / "cwd")).is_relative_to(root.resolve()):
                found.append(int(d.name))
        except OSError:
            continue
    return found


def test_a_flagged_model_queues_an_abba_rerun_of_its_flagged_cases(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.2, 96)] * 8, [(1.2, 96)] * 8, wall_n=2.4)
    finish(rd, md)
    man = mf.Manifest(models={"m": chat_model()}, datasets={}, defaults={})
    args = Namespace(sessions=1, no_rerun=False, repeats=None, calibration=str(tmp_path / "none.json"),
                     work_dir=str(tmp_path / "work"), data_dir=str(tmp_path))
    runner = compare.Runner(args, man, [chat_model()], trees.Ref("a", "a" * 40, "a"), trees.Ref("b", "b" * 40, "b"),
                            rd, [])
    assert runner.maybe_rerun("m") == "m__abba" and runner.rerun_cases["m__abba"] == ["c"]
    assert runner.maybe_rerun("m__abba") is None                   # never a rerun of a rerun
    assert [c.id for c in runner.model_for("m__abba").cases] == ["c"]
    args.no_rerun = True
    assert runner.maybe_rerun("m") is None


def test_trees_resolve_and_export(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    (repo / "rust").mkdir()
    (repo / "rust" / "Cargo.toml").write_text("[package]\n")
    (repo / "a.txt").write_text("one")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "one"], check=True)
    ref = trees.resolve(repo, "main", remote="origin")
    assert ref.label == "main" and len(ref.sha) == 40 and ref.rust_tree
    by_sha = trees.resolve(repo, ref.sha[:10], remote="origin")
    assert by_sha.sha == ref.sha and by_sha.label == ref.sha[:9]
    with pytest.raises(trees.TreeError, match="cannot resolve"):
        trees.resolve(repo, "no-such-branch", remote="origin")
    clone = tmp_path / "clone"                                                 # a branch fetched from a remote
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True)
    subprocess.run([*git, "checkout", "-q", "-b", "feat"], check=True)
    (repo / "a.txt").write_text("two")
    subprocess.run([*git, "commit", "-q", "-am", "two"], check=True)
    feat = trees.resolve(clone, "feat", remote="origin")
    assert feat.label == "feat" and feat.sha == trees.git(repo, "rev-parse", "feat") != ref.sha
    assert not trees.git(clone, "for-each-ref", "refs/mstar-eval")                # the temporary ref is gone
    out = trees.export(repo, ref.sha, tmp_path / "src" / ref.sha)
    assert (out / "a.txt").read_text() == "one" and (out / ".complete").exists()
    assert compare.gpu_type("NVIDIA H200") == "h200" and compare.gpu_type("NVIDIA B200") == "b200"
    assert [compare.gpu_type(n) for n in ("NVIDIA L4", "NVIDIA L40", "NVIDIA L40S", "NVIDIA GB200",
                                          "NVIDIA A100-SXM4-80GB")] == ["l4", "l40", "l40s", "gb200", "a100"]


def test_every_session_is_judged_and_more_head_errors_fail_even_without_outputs():
    c = case("text_match", min_match=0.5)
    good, bad = {"text_n": 16, "text_identical": 16}, {"text_n": 16, "text_identical": 0, "text_sim_median": 0.1,
                                                       "text_len_ratio_median": 0.2}
    v, detail = report.judge_correct(c, [good, bad, {"self_base": good}], {})
    assert v == "FAIL" and detail.startswith("session 2: 0/16 identical")
    v, detail = report.judge_correct(c, [good, good], {})
    assert v == "pass" and "all 2 sessions pass" in detail
    v, detail = report.judge_correct(c, [{"missing": ["base", "head"]}], {"base": 0, "head": 48})
    assert v == "FAIL" and detail.endswith("errors 0 vs 48")
    assert report.judge_correct(c, [{"missing": ["base", "head"]}], {})[0] == "incomplete"


def test_report_reasons_carry_no_paths_addresses_or_host_names(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    md.mkdir(parents=True)
    (md / "done.json").write_text(json.dumps({"status": "error",
                                              "reason": "OSError: /home/alice/x on 192.0.2.7, log /root/w/server.log"}))
    (rd / "run.json").write_text(json.dumps({**RUN, "models": ["m"]}))
    text = report.render_md(report.build_rows(rd, [chat_model()], calibration={}), RUN)
    assert "error: OSError" in text and "/home/alice" not in text and "192.0.2.7" not in text and "/root" not in text
    runner = compare.Runner.__new__(compare.Runner)
    runner.slots = [hosts.Slot(hosts.SSHTransport("bench@gpu7.example"), 0)]
    assert runner.redact("HostLostError: bench@gpu7.example: refused by gpu7.example") == \
        "HostLostError: <host>: refused by <host>"
    runner.slots[0].host_dir = "/data/eval"
    assert runner.redact("see /data/eval/runs/r/m.1/arm.log", runner.slots[0]) == "see <work dir>/runs/r/m.1/arm.log"


def test_a_reused_run_directory_is_refused(tmp_path):
    from benchmark.eval.cli import build_parser

    (tmp_path / "run.json").write_text("{}")
    args = build_parser().parse_args(["compare", "--head", "main", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="already holds a run"):
        compare.cmd_compare(args)


def test_a_job_whose_lease_is_not_renewed_stops_itself_and_its_servers(tmp_path):
    man = mf.parse({"defaults": {"conc": [1], "repeats": 50, "warmup": 0, "startup_timeout_s": 60}, "models": {
        "fake": {"serve": "fake", "family": "LLM", "cases": [
            {"id": "text", "client": "chat_text", "n": 4, "check": {"kind": "text_match", "min_match": 0.5},
             "perf": {"latency": "lat_p50_s"}}]}}})
    base = trees.Ref("main", "a" * 40, "main")
    args = Namespace(python=sys.executable, work_dir=str(tmp_path / "work"), data_dir=str(tmp_path / "data"),
                     sessions=1, repeats=None)
    slot = hosts.Slot(hosts.LocalTransport(), None, host_dir=str(tmp_path / "host"))
    runner = compare.Runner(args, man, man.select("all"), base, base, tmp_path / "run", [slot])
    runner.cache = tmp_path / "cache"
    fake_tree(runner.cache / "src" / base.sha, delay=0.05)
    out = tmp_path / "out"
    out.mkdir()
    job = {**runner.job(slot, "fake", str(out)), "lease_s": 1.5}
    (out / "job.json").write_text(json.dumps(job))
    (out / "lease").touch()
    r = subprocess.run([sys.executable, "-m", "benchmark.eval.e2e.arm", str(out / "job.json")], capture_output=True,
                       text=True, timeout=120, check=False, env={**os.environ, "PYTHONPATH": str(compare.TOOL_ROOT)})
    done = json.loads((out / "done.json").read_text())
    assert r.returncode != 0 and done["status"] == "interrupted" and "lease not renewed" in done["reason"]
    assert json.loads((out / "base" / "s0" / "startup.json").read_text())["healthy"]
    assert not procs_under(runner.cache / "src")


def test_head_gaps_missing_outputs_and_nan_actions_fail(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    (md / "head" / "s0" / "c" / "x-head-c8.json").unlink()                # head timed out at c=8
    finish(rd, md)
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.flagged and r.correct_detail.startswith("head wrote no results for s0 c=8")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    for i in range(3):
        (a / f"{i:03d}.txt").write_text("hi"), (b / f"{i:03d}.txt").write_text("hi")
        np.zeros(24000, np.int16).tofile(a / f"{i:03d}.pcm")
    np.zeros(24000, np.int16).tofile(b / "000.pcm")                       # head kept the text, lost the audio
    f = checks.saved(a, b)
    assert f["only_base"] == ["001.pcm", "002.pcm"] and f["audio_s_base"] == 3.0 and f["audio_s_head"] == 1.0
    v, detail = report.judge_correct(case("audio_len", pct=25), [f], {})
    assert v == "FAIL" and detail.startswith("head lacks 2 outputs")
    assert report.judge_correct(case("psnr", min_db=30), [{"missing": ["head"]}], {})[0] == "FAIL"
    c, d = tmp_path / "c", tmp_path / "d"
    c.mkdir(), d.mkdir()
    for name, x, y in (("000.f32", [0.0, 1.0], [0.0, 1.0]), ("001.f32", [0.0, 1.0], [float("nan"), 1.0])):
        np.array(x, np.float32).tofile(c / name), np.array(y, np.float32).tofile(d / name)
    assert checks.saved(c, d)["action_max_abs_diff"] == math.inf


def test_signals_wait_until_server_teardown_is_done():
    import os as _os
    import signal as _signal

    from benchmark.eval.e2e import arm

    old = _signal.signal(_signal.SIGTERM, arm._on_term)
    try:
        finished = []
        with pytest.raises(SystemExit) as e:
            with arm.signals_deferred():
                _os.kill(_os.getpid(), _signal.SIGTERM)
                finished.append(True)                  # the teardown keeps going after the signal
        assert finished == [True] and e.value.code == 128 + _signal.SIGTERM
    finally:
        _signal.signal(_signal.SIGTERM, old)


def test_in_band_stream_errors_count_as_a_changed_outcome(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "rp.json").write_text(json.dumps({"gen": {"status": 200, "text": "hi"}, "ws": {"j1": "hi"},
                                           "sse": {"status": 200, "sse_done": True},
                                           "rt": {"events": "transcription_session.updated"},
                                           "rt2": {"events": "transcription_session.updated"}}))
    (b / "rp.json").write_text(json.dumps({"gen": {"status": 200, "error": "boom", "error_status": 500},
                                           "ws": {"j1": "", "j1_error": "boom"},
                                           "sse": {"status": 200, "sse_done": False},
                                           "rt": {"events": "error", "error": "bad session"},
                                           "rt2": {"events": "transcription_session.created"}}))
    diffs = {d["case"]: d for d in checks.rp_facts(a, b)["diffs"]}
    assert diffs["gen"]["head"] == "200+error 500" and diffs["ws"]["head"] == "ok+error" and "sse" in diffs
    assert diffs["rt"]["head"] == "ok+error" and "rt2" in diffs                 # an error event, other events


def test_points_neither_arm_wrote_leave_the_row_incomplete(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    for arm in ("base", "head"):
        (md / arm / "s0" / "c" / f"x-{arm}-c8.json").unlink()            # both c=8 clients timed out
    finish(rd, md)
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.correct == "incomplete" and r.correct_detail.startswith("no results for s0 c=8")
    assert "No regressions:" not in report.render_md([r], RUN)


def test_cmax_throughput_is_normalized_on_its_own_lengths(tmp_path):
    """Identical c=1 replies, but at c=8 head's replies are twice as long and take twice as long: the same
    tokens per second, not a 50% drop in requests per second."""
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 96)] * 8, [(2.0, 192)] * 8, wall_n=4.0)
    finish(rd, md)
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.latency.metric == "lat_p50_s" and not r.latency.basis
    assert r.throughput.metric == "tok_per_s" and r.throughput.verdict == "same" and not r.flagged


def test_non_finite_actions_fail_and_never_calibrate_a_limit(tmp_path):
    c = case("action_diff", tol=0.05)
    assert report.judge_correct(c, [{"action_max_abs_diff": math.inf}], {})[0] == "FAIL"
    assert report.judge_correct(c, [{"action_max_abs_diff": math.inf}], {}, swings=[math.inf])[0] == "FAIL"
    assert report.correct_limit(c.check, [math.inf]) == (c.check, "")
    rd, md = tmp_path / "aa", tmp_path / "aa" / "results" / "m"
    md.mkdir(parents=True)
    (md / "done.json").write_text(json.dumps({"status": "done"}))
    (md / "compare.json").write_text(json.dumps({"c": [{"action_max_abs_diff": math.inf}]}))
    (rd / "run.json").write_text(json.dumps({**RUN, "head": RUN["base"], "id": "aa", "models": ["m"]}))
    m = Model(name="m", serve="m", family="VLA", tags=[], cases=[c])
    calibrate.add_run(rd, [m], tmp_path / "cal.json")
    assert "h200:m/c/correct" not in calibrate.load(tmp_path / "cal.json")


def test_sse_errors_dead_servers_undecodable_outputs_and_base_only_gaps(tmp_path):
    from benchmark.eval.e2e import request_path

    class Resp:
        status_code, headers = 200, {"content-type": "text/event-stream"}
        content = b'data: {"error": {"message": "boom", "code": 500}}\n\ndata: [DONE]\n\n'

    s = request_path.summarize(Resp())
    assert s["error_status"] == 500 and checks.status_of(s) == "200+error 500"
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    for arm, alive in (("base", True), ("head", False)):
        (md / arm / "s0" / "startup.json").write_text(json.dumps(
            {"healthy": True, "startup_s": 10, "cases": {"c": {"rc": 0, "server_alive_after": alive}}}))
    finish(rd, md)
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.flagged and r.correct_detail.startswith("head server died during this case (s0)")
    (md / "head" / "s0" / "startup.json").write_text(json.dumps({"healthy": True, "startup_s": 10}))
    (md / "base" / "s0" / "c" / "x-base-c8.json").unlink()                 # only head wrote c=8
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.correct == "incomplete" and r.correct_detail.startswith("no results for s0 c=8")
    from PIL import Image

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(a / "000.png")
    (b / "000.png").write_bytes(b"\x89PNG truncated")
    f = checks.saved(a, b)
    assert f["head_undecodable"] == ["000.png"]
    assert report.judge_correct(case("psnr", min_db=30), [f], {})[0] == "FAIL"


def test_a_truncated_results_file_counts_as_a_missing_point(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    f = md / "head" / "s0" / "c" / "x-head-c8.json"
    f.write_text(f.read_text()[:200])                                      # the client died while saving
    finish(rd, md)
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.flagged and r.correct_detail.startswith("head wrote no results for s0 c=8")


SIGTERM_RUN = r"""
import json, signal, sys
sys.path[:0] = [{root!r}, {tests!r}]
from argparse import Namespace
from pathlib import Path
import test_eval_compare as T
from benchmark.eval import cli
from benchmark.eval.e2e import compare, hosts, manifest as mf, trees

compare.POLL_S = 0.2
tmp = Path({tmp!r})
man = mf.parse({{"defaults": {{"conc": [1], "repeats": 200, "warmup": 0, "startup_timeout_s": 60}}, "models": {{
    "fake": {{"serve": "fake", "family": "LLM", "cases": [
        {{"id": "text", "client": "chat_text", "n": 4, "check": {{"kind": "text_match", "min_match": 0.5}},
         "perf": {{"latency": "lat_p50_s"}}}}]}}}}}})
base = trees.Ref("main", "a" * 40, "main")
(tmp / "run").mkdir()
(tmp / "run" / "run.json").write_text(json.dumps({{**T.RUN, "models": ["fake"]}}))
(tmp / "data").mkdir()
args = Namespace(python=sys.executable, host_python=sys.executable, host_dir="mstar-eval", data_dir=str(tmp / "data"),
                 work_dir=str(tmp / "work"), sessions=1, repeats=None, no_rerun=True, calibration=str(tmp / "cal.json"))
(tmp / "home").mkdir()
slot = hosts.Slot(T.FakeRemote(tmp / "home"), None)
runner = compare.Runner(args, man, man.select("all"), base, base, tmp / "run", [slot])
runner.cache = tmp / "cache"
T.fake_tree(runner.cache / "src" / base.sha, delay=0.05)
for sig in (signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, cli._exit_on_signal)
hosts.run_slots([slot], hosts.WorkQueue(["fake"], {{}}), runner.run_one, stop=runner.stop)
"""


def wait_for_fake_job_in_a_case(tmp_path: Path) -> Path:
    """The fake job's status.json once its first server is serving a case."""
    for _ in range(300):
        found = list((tmp_path / "home" / "mstar-eval" / "runs").glob("run-*/fake.1/status.json"))
        if found:
            break
        threading.Event().wait(0.1)
    status = found[0]
    for _ in range(300):
        if '"phase": "case"' in (status.read_text() if status.exists() else "") and procs_under(tmp_path / "home"):
            break
        threading.Event().wait(0.1)
    assert procs_under(tmp_path / "home")
    return status


def test_sigterm_to_compare_stops_the_remote_job_before_compare_exits(tmp_path):
    """A signal that lands while the main thread waits must not let compare exit before its slot threads have
    stopped their jobs (Thread.join can return early after an interrupted join)."""
    code = SIGTERM_RUN.format(root=str(compare.TOOL_ROOT), tests=str(Path(__file__).parent), tmp=str(tmp_path))
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    status = wait_for_fake_job_in_a_case(tmp_path)
    proc.send_signal(subprocess.signal.SIGTERM)
    proc.wait(120)
    done = json.loads((status.parent / "done.json").read_text())
    assert done["status"] == "interrupted"
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["models"]["fake"]["status"] == "interrupted"
    assert not procs_under(tmp_path / "home")


CTRL_C_RUN = SIGTERM_RUN.split("for sig in", maxsplit=1)[0] + r"""
def fake_compare(args):
    hosts.run_slots([slot], hosts.WorkQueue(["fake"], {{}}), runner.run_one, stop=runner.stop)
    return 0
compare.cmd_compare = fake_compare
cli.main(["compare", "--head", "main"])
"""


def test_ctrl_c_on_compare_stops_the_remote_job_and_prints_one_line(tmp_path):
    code = CTRL_C_RUN.format(root=str(compare.TOOL_ROOT), tests=str(Path(__file__).parent), tmp=str(tmp_path))
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    status = wait_for_fake_job_in_a_case(tmp_path)
    os.killpg(proc.pid, subprocess.signal.SIGINT)          # a terminal's Ctrl-C: the whole foreground group
    _, err = proc.communicate(timeout=120)
    assert proc.returncode == 130
    assert "Traceback" not in err and err.strip().splitlines()[-1] == "interrupted; remote job stopped"
    assert json.loads((status.parent / "done.json").read_text())["status"] == "interrupted"
    assert not procs_under(tmp_path / "home")


def test_ctrl_c_messages_say_whether_the_jobs_were_stopped():
    from benchmark.eval import cli

    assert cli._interrupted(KeyboardInterrupt()) == "interrupted; no job was running"
    assert cli._interrupted(hosts.Interrupted(True)) == "interrupted; remote job stopped"
    try:
        try:
            raise hosts.Interrupted(False)
        except KeyboardInterrupt:
            raise KeyboardInterrupt from None              # a second Ctrl-C during cleanup keeps the first's news
    except KeyboardInterrupt as e:
        assert cli._interrupted(e) == "interrupted; a job still running stops itself within 10 min (its lease)"


def test_a_job_killed_during_head_fails_the_rows_head_never_answered(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    (md / "head" / "s0").mkdir(parents=True)
    (md / "head" / "s0" / "startup.json").write_text(json.dumps({"healthy": True, "startup_s": 10}))
    finish(rd, md)
    (md / "done.json").write_text(json.dumps({"status": "error", "reason": "the job exited without a result"}))
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.flagged and r.correct_detail.startswith("head wrote no results for s0 c=1, s0 c=8")


def test_poll_output_parsing_treats_a_truncated_done_json_as_absent():
    p = compare.parse_poll
    assert p('A:1\nS:{"phase": "case"}\nD:\n') == (True, '{"phase": "case"}', None)
    assert p('A:0\nS:{"phase": "done"}\nD:{"status": "done"}') == (False, '{"phase": "done"}', {"status": "done"})
    assert p('A:0\nS:{"phase": "case"}\nD:{"status": "do') == (False, '{"phase": "case"}', None)
    assert p('A:0\nS:\nD:{"status": "skipped"}') == (False, "", {"status": "skipped"})   # ended before any status
    assert p("") == (False, "", None)


def test_unreadable_run_files_leave_the_report_standing(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    finish(rd, md)
    (md / "head" / "s0" / "startup.json").write_text('{"healthy": tr')        # killed while writing
    (md / "compare.json").write_text('{"c": [')
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.starts["head"] is False and r.flagged                               # unknown health is not a start
    (md / "head" / "s0" / "startup.json").write_text(json.dumps({"healthy": True}))
    assert report.build_rows(rd, [chat_model()], calibration={})[0].correct == "incomplete"


def test_a_zombie_job_counts_as_dead():
    script = 'sleep 0.2 & z=$!; exec sleep 3'                    # the child exits; nobody reaps it
    p = subprocess.Popen(["bash", "-c", script])
    try:
        threading.Event().wait(1.0)
        kids = subprocess.run(["ps", "-o", "pid=,stat=", "--ppid", str(p.pid)], capture_output=True, text=True,
                              check=False).stdout.split()
        zombie = kids[0] if len(kids) >= 2 and kids[1].startswith("Z") else None
        if zombie is None:
            pytest.skip("could not make a zombie here")
        alive = f'case "$(ps -o stat= -p {zombie} 2>/dev/null)" in ""|Z*) false;; *) true;; esac'
        assert subprocess.run(["bash", "-c", f"kill -0 {zombie}"], check=False).returncode == 0   # kill -0 says alive
        assert subprocess.run(["bash", "-c", alive], check=False).returncode != 0
    finally:
        p.kill()
        p.wait()


def test_inputs_are_keyed_by_content_and_calibration_by_invocation(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, text in ((a, "one"), (b, "two")):                                 # same names, same sizes
        (d / "media").mkdir(parents=True)
        (d / "media" / "x.txt").write_text(text)
    assert compare.tree_id(a) != compare.tree_id(b) and compare.tree_id(a) == compare.tree_id(a)
    path = tmp_path / "cal.json"
    for root, drift in (("one", 1.08), ("two", 1.30)):                      # two A/A runs, both called aa
        rd, md = tmp_path / root / "aa", tmp_path / root / "aa" / "results" / "m"
        write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
        write_arm(md, "head", [(drift, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
        finish(rd, md, run={"head": RUN["base"], "id": f"aa-{root}"})
        assert calibrate.add_run(rd, [chat_model()], path) > 0
    assert len(calibrate.load(path)["h200:m/c/lat_p50_s@c1"]["deltas"]) == 2


def test_calibration_keeps_the_drift_that_triggered_an_abba_rerun(tmp_path):
    rd, md, rr = tmp_path, tmp_path / "results" / "m", tmp_path / "results" / "m__abba"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.1, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)       # one session: -10% at c=1
    finish(rd, md, run={"head": RUN["base"], "id": "aa"})
    for s, (b, h) in enumerate(((1.0, 1.1), (1.1, 1.0))):                       # ABBA: it cancels out
        write_arm(rr, "base", [(b, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0, session=s)
        write_arm(rr, "head", [(h, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0, session=s)
    (rr / "done.json").write_text(json.dumps({"status": "done"}))
    (rr / "compare.json").write_text(json.dumps({"c": [{"text_n": 8, "text_identical": 8}] * 2}))
    path = tmp_path / "cal.yaml"
    calibrate.add_run(rd, [chat_model()], path)
    deltas = calibrate.load(path)["h200:m/c/lat_p50_s@c1"]["deltas"]
    assert -10.0 in deltas and calibrate.floor_pct(calibrate.load(path), "h200:m/c/lat_p50_s@c1") == 15.0
    assert path.read_text().startswith("# A/A noise floors")
    calibrate.save(calibrate.load(path), tmp_path / "cal.json")                  # a private JSON copy round-trips
    assert calibrate.load(tmp_path / "cal.json") == calibrate.load(path)


def test_concurrent_exports_of_one_commit_both_get_a_whole_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    for i in range(200):
        (repo / f"f{i}.txt").write_text("x" * 1000)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "one"], check=True)
    sha = trees.resolve(repo, "main", remote="origin").sha
    dest, errors = tmp_path / "src" / sha, []

    def go():
        try:
            trees.export(repo, sha, dest)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=go) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors and len(list(dest.glob("f*.txt"))) == 200 and (dest / ".complete").exists()
    assert {p.name for p in dest.parent.iterdir()} == {sha, f".{sha}.lock"}           # no temp dirs left


def test_a_lost_host_retires_all_its_gpus_so_the_retry_lands_elsewhere():
    q = hosts.WorkQueue(["a"], {})
    bad0, bad1, good = (hosts.Slot(Namespace(name=n), g) for n, g in (("u@bad", 0), ("u@bad", 1), ("u@good", 0)))
    ran, gave_up, release = [], [], threading.Event()

    def run_one(s, name):
        if s.transport.name == "u@bad":
            raise hosts.HostLostError("gone")
        release.wait(2)             # usually a slot of the bad host took "a" first and lost it
        ran.append((s.name, name))

    def lost(s, name, e):
        release.set()

    hosts.run_slots([bad0, bad1, good], q, run_one, on_lost=lost, on_given_up=gave_up.append)
    assert ran == [("u@good:0", "a")] and not gave_up


def test_an_abba_rerun_keeps_a_correctness_failure_from_the_first_run(tmp_path):
    rd, md, rr = tmp_path, tmp_path / "results" / "m", tmp_path / "results" / "m__abba"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.2, 96)] * 8, [(1.2, 96)] * 8, wall_n=2.4, errors=2)
    finish(rd, md)
    for s in (0, 1):
        write_arm(rr, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0, session=s)
        write_arm(rr, "head", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0, session=s)
    (rr / "done.json").write_text(json.dumps({"status": "done"}))
    (rr / "compare.json").write_text(json.dumps({"c": [{"text_n": 8, "text_identical": 8}] * 2}))
    r = report.build_rows(rd, [chat_model()], calibration={})[0]
    assert r.rerun and r.latency.verdict == "same"                               # the timings come from the rerun
    assert r.flagged and r.correct == "FAIL" and r.correct_detail.startswith("first run:") and r.errors["head"] > 0


def test_a_truncated_request_path_result_fails_its_row_and_spares_the_others(tmp_path):
    md = tmp_path / "m"
    for arm in ("base", "head"):
        (md / arm / "s0" / "rp").mkdir(parents=True)
        (md / arm / "s0" / "rp" / "rp.json").write_text(json.dumps({"x": {"status": 200}}))
        (md / arm / "s0" / "t").mkdir(parents=True)
    (md / "head" / "s0" / "rp" / "rp.json").write_text('{"x": {"sta')            # killed while writing
    cases = [{"id": "rp", "driver": "rp", "conc": [1]}, {"id": "t", "driver": "suite", "conc": [1]}]
    res = checks.compare_model(md, cases, 1)
    assert res["rp"] == [{"missing": ["head"]}] and res["t"] == [{"missing": ["base", "head"]}]
    assert report.judge_correct(case("status_match"), res["rp"], {})[0] == "FAIL"


def test_a_head_that_does_not_build_is_flagged_and_rows_without_a_verdict_exit_2(tmp_path, monkeypatch):
    from benchmark.eval.e2e import arm
    m = chat_model()
    man = mf.Manifest(models={"m": m}, datasets={}, defaults={})
    calib = tmp_path / "calib.yaml"
    calib.write_text("{}\n")
    monkeypatch.setattr(arm, "build_rust", lambda a: "mstar_rust build failed: boom" if a["src"] == "h" else None)
    monkeypatch.setattr(arm, "job", {"model": {"name": "m", "cases": [], "serve_args": []}, "order": ["base", "head"],
                                     "sessions": 1, "arms": {"base": {"src": "b", "rust_tree": "t1"},
                                                             "head": {"src": "h", "rust_tree": "t2"}}})
    rd = tmp_path / "run"
    md = rd / "results" / "m"
    md.mkdir(parents=True)
    monkeypatch.setattr(arm, "out", md)
    arm.run()
    assert json.loads((md / "done.json").read_text())["broken"] == "head"
    (rd / "run.json").write_text(json.dumps({**RUN, "models": ["m"]}))
    r = report.build_rows(rd, [m], calibration={})[0]
    assert r.flagged and r.correct == "FAIL" and r.verdict() == "**head does not build**"
    assert compare.render(rd, man, str(calib))[1] == 1
    for broken, verdict, code in (("base", "base does not build", 2), (None, "error: job failed", 2)):
        (md / "done.json").write_text(json.dumps({"status": "error", "reason": "job failed",
                                                  **({"broken": broken} if broken else {})}))
        r = report.build_rows(rd, [m], calibration={})[0]
        assert not r.flagged and r.unfinished and r.verdict() == verdict
        assert compare.render(rd, man, str(calib))[1] == code
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    finish(rd, md)
    assert compare.render(rd, man, str(calib))[1] == 0


def test_a_local_ref_resolves_to_this_checkout_not_the_remote(tmp_path):
    repo, clone = tmp_path / "repo", tmp_path / "clone"
    repo.mkdir()
    git = ["-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "-C", str(repo), *git, "init", "-q", "-b", "main"], check=True)
    (repo / "a.txt").write_text("one")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), *git, "commit", "-q", "-m", "one"], check=True)
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True)
    (clone / "a.txt").write_text("unpushed")
    subprocess.run(["git", "-C", str(clone), *git, "commit", "-q", "-am", "unpushed"], check=True)
    subprocess.run(["git", "-C", str(clone), "tag", "v1"], check=True)
    local = trees.git(clone, "rev-parse", "HEAD")
    assert local != trees.git(repo, "rev-parse", "HEAD")
    head = trees.resolve(clone, "HEAD", remote="origin")
    assert head.sha == local and head.label == f"HEAD ({local[:9]})"
    assert trees.resolve(clone, "v1", remote="origin").sha == local
    assert trees.resolve(clone, "HEAD~1", remote="origin").sha == trees.git(repo, "rev-parse", "HEAD")
    for name in ("main", "heads/main", "refs/heads/main"):                       # a branch, however named: fresh
        assert trees.resolve(clone, name, remote="origin").sha == trees.git(repo, "rev-parse", "HEAD")


def test_truncated_text_replies_fail_even_when_enough_others_are_identical():
    fox = "the quick brown fox jumps over the lazy dog"
    facts = checks.text_facts([(fox, fox)] * 8 + [(fox, fox[:14])] * 4 + [(fox, "")] * 4)
    assert facts["text_identical"] == 8 and facts["text_truncated"] == 8
    assert checks.text_facts([(fox, fox.replace("the lazy", "a lazy")), (fox, fox[:-4])])["text_truncated"] == 0
    c = case("text_match", min_match=0.5)
    v, detail = report.judge_correct(c, [facts], {})
    assert v == "FAIL" and "8 head replies are base's cut short" in detail
    assert report.judge_correct(c, [{**facts, "text_truncated": 0}], {})[0] == "pass"


# ---------------------------------------------------------------- arms that answer no request, environment checks

def visible(md: str) -> str:
    """The part of a report outside its collapsed sections."""
    return re.sub(r"<details>.*?</details>", "", md, flags=re.S)


def silent_run(tmp_path: Path, base_failed, head_failed):
    """Whisper-like ASR rows where an arm answers every request at the given concurrencies with a 500: WER reads
    100% on a failed arm, so a WER check alone sees no change when both fail."""
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True, failed=base_failed)
    write_arm(md, "head", [(0.2, 3.0)] * 8, [(0.4, 3.0)] * 8, wall_n=1.0, audio=True, failed=head_failed)
    wer = {"base": 1.0 if 1 in base_failed else 0.02, "head": 1.0 if 1 in head_failed else 0.02}
    finish(rd, md, [{"wer_base": wer["base"], "wer_head": wer["head"]}])
    m = chat_model(kind="tts", check={"kind": "wer", "max_increase": 0.5})
    calib = tmp_path / "calib.yaml"
    calib.write_text("{}\n")
    man = mf.Manifest(models={"m": m}, datasets={}, defaults={})
    row = report.build_rows(rd, [m], calibration={})[0]
    md_path, code = compare.render(rd, man, str(calib))
    return row, md_path.read_text(), code


def test_every_request_failing_on_both_arms_has_no_verdict_and_says_so(tmp_path):
    r, text, code = silent_run(tmp_path, (1, 8), (1, 8))
    assert r.correct == "base broken" and r.unfinished and not r.flagged and code == 2
    assert r.silent == "both" and r.verdict() == "neither arm answers a request"
    assert r.correct_detail.startswith("neither arm answered a request at s0 c=1, s0 c=8")
    top = visible(text)
    assert "keeps its outputs" not in top and "No regressions:" not in top
    assert "Errors, base / head" in top and "**48 / 48**" in top          # 8 requests x 3 passes x 2 points
    assert "**1 rows had request errors** (base 48, head 48 in total)" in top


def test_a_base_that_answers_no_request_leaves_the_row_without_a_verdict(tmp_path):
    r, text, code = silent_run(tmp_path, (1, 8), ())
    assert r.correct == "base broken" and r.silent == "base" and code == 2
    assert r.verdict() == "base answers no request" and "base answered no request at s0 c=1, s0 c=8" in text
    r, _, code = silent_run(tmp_path / "cmax", (8,), ())                  # base fails only at c=max
    assert r.correct == "base broken" and r.correct_detail.startswith("base answered no request at s0 c=8") \
        and code == 2


def test_a_head_that_answers_no_request_fails_the_row(tmp_path):
    r, text, code = silent_run(tmp_path, (), (1, 8))
    assert r.correct == "FAIL" and r.flagged and code == 1 and r.verdict() == "**head answers no request**"
    assert "head answered no request at s0 c=1, s0 c=8" in r.correct_detail and "**0 / 48**" in visible(text)
    r, _, code = silent_run(tmp_path / "cmax", (), (8,))                  # c=1 outputs agree, c=max all 500s
    assert r.correct == "FAIL" and code == 1 and "head answered no request at s0 c=8" in r.correct_detail


def test_a_shared_rust_tree_that_did_not_build_is_said_in_the_report(tmp_path):
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    finish(rd, md)
    why = "mstar_rust not built: the M* environment has no pip and no uv is on PATH"
    (md / "done.json").write_text(json.dumps({"status": "done", "rust": {"base": why, "head": why}}))
    text = report.render_md(report.build_rows(rd, [chat_model()], calibration={}), json.loads((rd / "run.json")
                                                                                              .read_text()))
    assert f"**No per-arm `rust/` build for m** ({why})" in visible(text)
    assert "own `rust/` build, except m: see above" in text


def fake_bin(d: Path, name: str, script: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("#!/bin/sh\n" + script)
    (d / name).chmod(0o755)
    return d / name


def test_rust_build_uses_uv_pip_when_the_environment_has_no_pip(tmp_path, monkeypatch):
    from benchmark.eval.e2e import arm

    bin_dir = tmp_path / "venv" / "bin"
    py = fake_bin(bin_dir, "python", 'if [ "$2" = pip ]; then echo "No module named pip" >&2; exit 1; fi\n')
    log = tmp_path / "uv.log"
    fake_bin(bin_dir, "uv", f'echo "$@" > {log}\nmkdir -p "$8"\n')
    monkeypatch.setattr(arm, "job", {"python": str(py), "host_dir": str(tmp_path / "host"), "model": {}})
    a = {"src": str(tmp_path / "src"), "rust_tree": "t"}
    assert arm.build_rust(a) is None
    site = tmp_path / "host" / "rust" / "t" / "site"
    assert log.read_text().split() == ["pip", "install", "--python", str(py), "-q", "--no-deps", "--target",
                                       str(site), f"{a['src']}/rust"]
    assert a["rust_site"] == str(site) and (site / ".complete").exists()
    (bin_dir / "uv").unlink()
    monkeypatch.setattr(arm, "base_env", lambda: {"PATH": str(tmp_path / "empty")})
    assert "no pip and no uv" in arm.build_rust({"src": "s", "rust_tree": "other"})


def test_requires_can_call_the_servers_own_loader(tmp_path, monkeypatch):
    from benchmark.eval.e2e import arm

    (tmp_path / "eager_pkg").mkdir()          # like qwen_tts under transformers 5.x: its __init__ raises
    (tmp_path / "eager_pkg" / "__init__.py").write_text("raise TypeError('wants transformers 4.57.3')\n")
    (tmp_path / "loader.py").write_text("def load():\n    return 1\n\ndef bad():\n    raise ImportError('x')\n")
    for requires, want in ((["loader:load", "json"], ""), (["eager_pkg"], "eager_pkg: TypeError"),
                           (["loader:bad"], "loader:bad: ImportError"),
                           (["loader:nope"], "loader:nope: AttributeError")):
        monkeypatch.setattr(arm, "job", {"python": sys.executable, "model": {"requires": requires}})
        assert arm.requirements_problem({"src": str(tmp_path)}) == want
    models = mf.load().models
    qwen = models["qwen3_tts"].requires
    assert qwen == ["mstar.model.qwen3_tts.qwen3_tts_model:_load_qwen3_tts_codec_classes"]
    for m in models.values():   # the split and transport entries serve the same model through the same loader
        if m.serve.startswith("qwen3_tts"):
            assert qwen[0] in m.requires and "qwen_tts" not in m.requires, m.name
    module, fn = qwen[0].split(":")
    assert f"def {fn}(" in Path(module.replace(".", "/") + ".py").read_text()



def test_requires_reads_only_its_result_line_whatever_an_import_prints(tmp_path, monkeypatch):
    """qwen_tts prints transformers' docstring warnings to stdout while the codec loads; the model was skipped
    as lacking its loader. Output can land before or after the check's own line."""
    from benchmark.eval.e2e import arm

    (tmp_path / "noisy.py").write_text(
        "import atexit\n"
        "print('[ERROR] `cache_position` is part of the forward signature, but not documented.')\n"
        "atexit.register(print, '[ERROR] printed at exit')\n"
        "def load():\n    return 1\n")
    (tmp_path / "dies.py").write_text("import os\nos._exit(3)\n")
    for requires, want in ((["noisy:load"], ""), (["noisy:load", "no_such_module_xyz"],
                                                   "no_such_module_xyz: ModuleNotFoundError"),
                           (["dies"], "requirement check exited 3")):
        monkeypatch.setattr(arm, "job", {"python": sys.executable, "model": {"requires": requires}})
        assert arm.requirements_problem({"src": str(tmp_path)}) == want

def test_results_pulls_work_with_an_rsync_older_than_3_2_3(tmp_path, monkeypatch):
    """RHEL 8's rsync 3.1.3 has no --mkpath; every push and pull lands in a directory whose parents may not exist."""
    real = subprocess.run(["bash", "-c", "command -v rsync"], capture_output=True, text=True,
                          check=False).stdout.strip()
    if not real:
        pytest.skip("no rsync")
    fake_bin(tmp_path / "bin", "rsync", 'for a in "$@"; do [ "$a" = --mkpath ] && { echo "rsync: --mkpath: unknown '
             f'option" >&2; exit 1; }}; done\nexec {real} "$@"\n')
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "f.txt").write_text("x")
    (src / "audio").mkdir()
    t = hosts.LocalTransport()
    t.push(src, str(tmp_path / "host" / "a" / "b"))
    assert (tmp_path / "host" / "a" / "b" / "sub" / "f.txt").read_text() == "x"
    t.pull(str(src), tmp_path / "results" / "m.partial", exclude=("audio/",))
    assert (tmp_path / "results" / "m.partial" / "sub" / "f.txt").exists()
    assert not (tmp_path / "results" / "m.partial" / "audio").exists()
    calls = []
    ssh = hosts.SSHTransport("u@h")
    monkeypatch.setattr(ssh, "sh", lambda script, timeout=None: calls.append(script) or
                        subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(hosts, "_rsync", lambda args, host="local": calls.append(args))
    ssh.push(src, "/w/evalfw/x/benchmark/eval")
    assert calls[0] == "mkdir -p /w/evalfw/x/benchmark" and calls[1][-1] == "u@h:/w/evalfw/x/benchmark/eval"


def test_work_dir_flag_and_xdg_cache_home(tmp_path, monkeypatch):
    import argparse

    parser = argparse.ArgumentParser()
    compare.add_args(parser.add_subparsers())
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    man = mf.Manifest(models={}, datasets={}, defaults={})
    ref = trees.Ref("a", "a" * 40, "a")
    for argv, cache, data in (([], tmp_path / "xdg" / "mstar-eval", tmp_path / "xdg" / "mstar-eval" / "e2e-data"),
                              (["--work-dir", str(tmp_path / "w")], tmp_path / "w", tmp_path / "w" / "e2e-data"),
                              (["--work-dir", str(tmp_path / "w"), "--data-dir", str(tmp_path / "d")], tmp_path / "w",
                               tmp_path / "d")):
        args = parser.parse_args(["compare", "--head", "x", *argv])
        runner = compare.Runner(args, man, [], ref, ref, tmp_path / "run", [])
        assert runner.cache == cache / "compare" and runner.data == data
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert compare.default_work_dir() == Path("~/.cache/mstar-eval").expanduser()


COLD_START = '''import os, time
marker = os.environ["FAKE_COMPILE_CACHE"]
if not os.path.exists(marker):   # the first server to start compiles; the caches it leaves serve every later one
    time.sleep({cold})
    open(marker, "w").close()
'''


def test_whichever_arm_starts_first_on_cold_caches_still_starts(tmp_path, monkeypatch):
    """The first start pays the cold compiles: with a timeout sized for a warm start, the arm that goes first used
    to read "does not start" (base on one order, head on the other)."""
    from benchmark.eval.e2e import arm

    for order in (["base", "head"], ["head", "base"]):
        root = tmp_path / order[0]
        srcs = {a: fake_tree(root / "src" / a) for a in ("base", "head")}
        for s in srcs.values():
            main = s / "mstar" / "cli" / "main.py"
            main.write_text(COLD_START.format(cold=2.5) + main.read_text())
        monkeypatch.setattr(arm, "out", root / "out")
        (root / "out").mkdir()
        monkeypatch.setattr(arm, "job", {
            "python": sys.executable, "gpu": None, "host_dir": str(root / "host"),
            "model": {"name": "fake", "serve": "fake", "startup_timeout_s": 2.0, "cases": [], "serve_args": [],
                      "env": {"FAKE_COMPILE_CACHE": str(root / "compiled")}},
            "arms": {a: {"src": str(s)} for a, s in srcs.items()}})
        infos = {a: arm.run_arm(a, 0) for a in order}
        assert all(i["healthy"] for i in infos.values()), (order, infos)
        assert infos[order[0]]["startup_s"] > 2.0 and infos[order[1]]["startup_s"] < 2.0
        assert arm.run_arm(order[0], 1)["startup_timeout_s"] == 2.0     # later sessions: the manifest's timeout


def test_one_noisy_aa_run_does_not_widen_a_floor_for_good(tmp_path):
    """qwen c=8 on a loaded node: one A/A drift of 29.7% next to 7.1% took the floor from 10.7% to 44.5%."""
    assert report.floor_from([7.13]) == pytest.approx(10.7, abs=0.01)
    assert report.floor_from([7.13, 29.67]) == pytest.approx(10.7, abs=0.01)          # a lone outlier
    assert report.floor_from([7.13, 29.67, -31.0]) == pytest.approx(46.5)              # confirmed by a second run
    assert report.floor_from([0.5, 7.1]) == pytest.approx(10.65)                       # under 10%: always counts
    assert report.floor_from([40.0, 41.0, 42.0] + [5.0] * 8) == pytest.approx(7.5)     # the last 8: floors shrink
    assert report.floor_from([]) is None and report.floor_from([math.nan]) is None
    rd, md = tmp_path, tmp_path / "results" / "m"
    write_arm(md, "base", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.0)
    write_arm(md, "head", [(1.0, 96)] * 8, [(1.0, 96)] * 8, wall_n=2.5)               # -20% at c=8
    finish(rd, md)
    cal = {"h200:m/c/req_per_s@c8": {"deltas": [7.13, -29.67]}}
    assert calibrate.floor_pct(cal, "h200:m/c/req_per_s@c8") == pytest.approx(10.7, abs=0.01)
    r = report.build_rows(rd, [chat_model()], calibration=cal)[0]
    assert r.throughput.verdict == "regression" and r.flagged
