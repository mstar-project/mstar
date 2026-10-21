"""benchmark.eval compare's perceptual quality checks on CPU, with fake judges: the manifest's ``quality`` lists,
scoring each arm's saved outputs (quality.py), the facts in compare.json, the verdicts (floors, A/A swings, "changed,
quality same"), calibration, and a seeded degradation (head speech low-passed to 4 kHz) that gets flagged while its
WER holds. Result directories follow the layout ``compare`` writes."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

pytest.importorskip("aiohttp")
PIL = pytest.importorskip("PIL.Image")

from benchmark.eval.audio import float_to_pcm16, write_wav  # noqa: E402
from benchmark.eval.e2e import arm, calibrate, checks, quality, report  # noqa: E402
from benchmark.eval.e2e import manifest as mf  # noqa: E402

RATE = 24000
RUN = {"id": "q", "base": {"label": "main", "sha": "a" * 40}, "head": {"label": "PR #1", "sha": "b" * 40},
       "gpu": "h200", "perf_pct": 5, "noise_sigma": 2, "gpu_hours": 0.1}
PERF = {"latency": "lat_p50_s", "throughput": "req_per_s"}
MANIFEST = {
    "datasets": {"tts": "tts/manifest.jsonl", "clone": "clone/manifest.jsonl"},
    "models": {
        "tts": {"serve": "kokoro", "family": "TTS", "cases": [
            {"id": "tts", "suite": "tts", "data": "tts", "conc": [1, 8], "quality": ["mos"],
             "check": {"kind": "wer", "max_increase": 1.5}, "perf": PERF}]},
        "clone": {"serve": "omnivoice", "family": "TTS", "cases": [
            {"id": "clone_upload", "client": "gen_clone", "data": "clone", "conc": [1, 8],
             "quality": ["mos", "speaker_sim"], "check": {"kind": "audio_len", "pct": 10}, "perf": PERF}]},
        "img": {"serve": "z_image_turbo", "family": "Image", "cases": [
            {"id": "t2i", "client": "image", "conc": [1], "quality": ["clip"],
             "check": {"kind": "psnr", "min_db": 30}, "perf": {"latency": "lat_p50_s"}}]},
    },
}
IDS = [f"{i:03d}" for i in range(4)]


# ---------------------------------------------------------------- fake judges and outputs

def hf_share(x: np.ndarray, rate: int, cut: float = 4000.0) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    return float(spec[np.fft.rfftfreq(len(x), 1 / rate) > cut].sum() / max(spec.sum(), 1e-12))


def bands(x: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x)) ** 2
    v = np.array([b.sum() for b in np.array_split(spec, 32)])
    return v / np.linalg.norm(v)


class FakeJudges:
    """Stand-ins that react to the signal: "MOS" rises with the energy above 4 kHz (so a 4 kHz low-pass scores
    lower), "speaker similarity" is the cosine of coarse band energies, "CLIP" follows the image's mean red."""

    def __init__(self):
        self.ids = {"mos": "fake-mos", "speaker_sim": "fake-sim", "clip": "fake-clip"}
        self.calls = {"mos": 0, "speaker_sim": 0, "clip": 0}

    def mos(self, clip):
        self.calls["mos"] += 1
        return 2.0 + 10 * hf_share(*clip)

    def speaker_sim(self, clip, ref):
        self.calls["speaker_sim"] += 1
        return float(bands(clip[0]) @ bands(ref[0]))

    def clip(self, images, text):
        self.calls["clip"] += 1
        assert text in ("a red fox sitting in fresh snow at sunrise, photorealistic",
                        "a bowl of ramen on a wooden table, studio lighting",
                        "a lighthouse on a cliff during a storm, oil painting",
                        "a small robot watering plants in a greenhouse")
        return float(np.mean([np.asarray(im)[..., 0].mean() for im in images]) / 255 * 10 + 25)


def voice(i: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """A 1 s harmonic "voice" (f0 120 Hz plus 10 Hz per item, harmonics to 11 kHz)."""
    t = np.arange(RATE) / RATE
    f0 = 120 + 10 * i
    x = sum(np.sin(2 * np.pi * k * f0 * t) / np.sqrt(k) for k in range(1, int(11000 // f0)))
    x = 0.3 * x / np.max(np.abs(x))
    return x + noise * np.random.default_rng(seed).standard_normal(len(x))


def low_pass(x: np.ndarray, cut: float = 4000.0) -> np.ndarray:
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(len(x), 1 / RATE) > cut] = 0
    return np.fft.irfft(spec, len(x))


def picture(i: int, noise: float = 0.0, dark: bool = False) -> np.ndarray:
    rng = np.random.default_rng(i)
    img = rng.integers(0, 255, (64, 64, 3)).astype(np.float64)
    if dark:
        img[...] = 20.0
    img += noise * np.random.default_rng(100 + i).standard_normal(img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def results(d: Path, conc: int, metrics: dict) -> None:
    rows = [{"id": i, "ok": True, "e2e_s": 1.0} for i in IDS]
    reps = [{"index": k, "wall_s": 4.0, "metrics": metrics, "requests": rows} for k in range(3)]
    (d / f"x-c{conc}.json").write_text(json.dumps({
        "schema": "mstar-eval/1", "suite": "e2e", "label": "x", "config": {}, "meta": {}, "repeats": reps,
        "summary": {k: {"mean": v} for k, v in metrics.items()}}))


def write_case(md: Path, arm_name: str, session: int, case: mf.Case, outputs: dict[str, object]) -> None:
    """One arm's results for a case: a results file per point and the saved c=1 outputs (as the suites save them)."""
    sd = md / arm_name / f"s{session}"
    d = sd / case.id
    keep = d / "audio" / "x-c1" / "r2"
    keep.mkdir(parents=True, exist_ok=True)
    (sd / "startup.json").write_text(json.dumps({"healthy": True, "startup_s": 10}))
    for conc in case.conc:
        results(d, conc, {"lat_p50_s": 1.0, "req_per_s": 4.0, "errors": 0, "wer": 0.02})
    for item, out in outputs.items():
        if case.kind == "tts":
            write_wav(keep / f"{item}.wav", out, RATE)
        elif case.kind == "gen_clone":
            (keep / f"{item}.pcm").write_bytes(float_to_pcm16(out))
        else:
            PIL.fromarray(out).save(keep / f"{item}.png")


def data_dir(tmp: Path) -> Path:
    data = tmp / "data"
    (data / "tts").mkdir(parents=True)
    (data / "clone" / "ref").mkdir(parents=True)
    (data / "tts" / "manifest.jsonl").write_text("".join(json.dumps({"id": i, "text": "hi"}) + "\n" for i in IDS))
    rows = []
    for k, i in enumerate(IDS):
        write_wav(data / "clone" / "ref" / f"{i}.wav", voice(k), RATE)
        rows.append(json.dumps({"id": i, "text": "hi", "ref_audio": f"ref/{i}.wav", "ref_text": "hello"}) + "\n")
    (data / "clone" / "manifest.jsonl").write_text("".join(rows))
    return data


def run_matrix(tmp: Path, head_speech, head_picture, sessions: int = 1, same_sha: bool = False,
               judges: FakeJudges | None = None, calibration: dict | None = None):
    """Base and head outputs for the three models, scored (quality.py) and compared (checks.py) as ``arm`` does
    after the last session, then reported."""
    man = mf.parse(copy.deepcopy(MANIFEST))
    models = man.select("all")
    data, rd = data_dir(tmp), tmp / "run"
    judges = judges or FakeJudges()
    for m in models:
        md = rd / "results" / m.name
        case = m.cases[0]
        for s in range(sessions):
            for arm_name in ("base", "head"):
                head = arm_name == "head"
                if case.kind == "image":
                    outs = {i: head_picture(k) if head else picture(k) for k, i in enumerate(IDS)}
                else:
                    outs = {i: head_speech(k, s) if head else voice(k, 0.002 * s, seed=s) for k, i in enumerate(IDS)}
                write_case(md, arm_name, s, case, outs)
        cases = [c.to_dict() for c in m.cases]
        assert quality.score_model(md, cases, sessions, data, man.datasets, judges) == 2 * sessions
        checks.compare_model(md, cases, sessions)
        (md / "done.json").write_text(json.dumps({"status": "done"}))
    run = {**RUN, "models": [m.name for m in models], "sessions": sessions}
    if same_sha:
        run["head"] = {**run["head"], "sha": run["base"]["sha"]}
    (rd / "run.json").write_text(json.dumps(run))
    path, rows = report.write(rd, models, calibration={} if calibration is None else calibration)
    return rd, models, {r.model: r for r in rows}, path.read_text()


# ---------------------------------------------------------------- manifest

@pytest.mark.parametrize("model,update,msg", [
    ("tts", {"quality": ["vibes"]}, "quality must be a list"),
    ("img", {"quality": ["mos"]}, "need speech outputs"),
    ("tts", {"quality": ["clip"]}, "clip needs generated images"),
    ("clone", {"data": None}, "speaker_sim needs a dataset"),
])
def test_quality_lists_are_validated(model, update, msg):
    raw = copy.deepcopy(MANIFEST)
    raw["models"][model]["cases"][0].update(update)
    with pytest.raises(mf.ManifestError, match=msg):
        mf.parse(raw)


def test_shipped_manifest_scores_every_speech_clone_and_generated_image_case():
    man = mf.load()
    for m in man.models.values():
        for c in m.cases:
            speech = c.kind in ("tts", "gen_clone") or "--audio-out" in c.args
            assert ("mos" in c.quality) == speech, (m.name, c.id)
            assert ("speaker_sim" in c.quality) == (speech and c.data == "clone"), (m.name, c.id)
            assert ("clip" in c.quality) == (c.kind in mf.CLIP_KINDS), (m.name, c.id)
    assert mf.QUALITY == {"mos": 0.1, "speaker_sim": 0.02, "clip": 0.5}


# ---------------------------------------------------------------- verdicts

def qcase(*kinds):
    return mf.Case(id="c", modality="m", driver="suite", kind="tts", check={"kind": "wer", "max_increase": 1},
                   quality=list(kinds))


def qfacts(**scores):
    return [{"quality": {k: {"base": b, "head": h, "n": 4} for k, (b, h) in scores.items()}}]


@pytest.mark.parametrize("kind,base,head,want", [
    ("mos", 4.20, 4.11, "same"), ("mos", 4.20, 4.09, "worse"), ("mos", 4.0, 4.2, "better"),
    ("speaker_sim", 0.90, 0.881, "same"), ("speaker_sim", 0.90, 0.879, "worse"),
    ("clip", 31.0, 30.51, "same"), ("clip", 31.0, 30.49, "worse"),
])
def test_quality_floors(kind, base, head, want):
    q = report.judge_quality(qcase(kind), qfacts(**{kind: (base, head)}))[0]
    assert q.verdict == want and q.limit == mf.QUALITY[kind]


def test_aa_swings_widen_a_quality_limit_and_the_worst_session_decides():
    c = qcase("mos")
    q = report.judge_quality(c, qfacts(mos=(4.2, 4.0)), {"mos": [0.2]})[0]
    assert q.verdict == "same" and q.limit == pytest.approx(0.3) and "from A/A" in q.note
    assert report.judge_quality(c, qfacts(mos=(4.2, 4.0)), {"mos": [0.01]})[0].verdict == "worse"   # floor holds
    two = qfacts(mos=(4.2, 4.18)) + qfacts(mos=(4.2, 3.9)) + [{"self_base": {"quality": {"mos": {"base": 4.2,
                                                                                               "head": 4.19}}}}]
    q = report.judge_quality(c, two)[0]
    assert q.verdict == "worse" and q.head == 3.9
    assert report.quality_swings(two[2]) == {"mos": pytest.approx(0.01)}
    none = report.judge_quality(c, [{"quality": {"mos": {"n": 0, "error": "RuntimeError: no hub"}}}])[0]
    assert none.verdict == "n/a" and "no hub" in none.text()


def test_only_a_limit_failure_can_read_changed():
    assert report.limit_only([{"psnr_min_db": 24.0, "psnr_median_db": 26.0}], {})
    assert not report.limit_only([{"psnr_min_db": None, "shape_mismatch": 1}], {})
    assert not report.limit_only([{"text_n": 8, "text_identical": 0, "text_truncated": 2}], {})
    assert not report.limit_only([{"psnr_min_db": 24.0}], {"base": 0, "head": 1})
    assert not report.limit_only([{"missing": ["head"]}], {})


# ---------------------------------------------------------------- dry run: scored, compared, reported

def test_quality_that_holds_passes_and_a_psnr_drop_stays_flagged_with_clip_holding(tmp_path):
    """A PSNR drop is flagged even when CLIP holds: a seed the server stops honouring gives different pixels of
    the same quality."""
    judges = FakeJudges()
    rd, _, rows, md = run_matrix(tmp_path, lambda k, s: voice(k), lambda k: picture(k, noise=12.0),
                                      judges=judges)
    assert judges.calls == {"mos": 16, "speaker_sim": 8, "clip": 8}        # 4 outputs per arm and case, c=1 only
    q = json.loads((rd / "results" / "clone" / "head" / "s0" / "clone_upload" / "quality.json").read_text())
    assert set(q) == {"mos", "speaker_sim"} and q["speaker_sim"]["judge"] == "fake-sim"
    assert len(q["mos"]["scores"]) == 4 and q["speaker_sim"]["scores"]["000"] > 0.99
    tts, clone, img = rows["tts"], rows["clone"], rows["img"]
    assert tts.correct == "pass" and [x.verdict for x in tts.quality] == ["same"] and not tts.flagged
    assert [x.kind for x in clone.quality] == ["mos", "speaker_sim"] and not clone.flagged
    facts = json.loads((rd / "results" / "img" / "compare.json").read_text())["t2i"][0]
    assert facts["psnr_min_db"] < 30 and facts["quality"]["clip"]["n"] == 4
    assert img.correct == "FAIL" and img.flagged and [x.verdict for x in img.quality] == ["same"]
    assert img.verdict().startswith("**correctness changed**") and "quality worse" not in img.verdict()
    assert "| Quality, base to head |" in md and "**correctness changed**" in md and "UTMOS" in md
    assert "CLIP score" in md and "**1 of 3 rows flagged**" in md
    assert "\u2014" not in md and "\u2013" not in md
    summary = json.loads((rd / "report" / "summary.json").read_text())
    assert {s["model"]: s["quality"][0]["verdict"] for s in summary} == {"tts": "same", "clone": "same",
                                                                         "img": "same"}
    if importlib.util.find_spec("matplotlib"):
        assert (rd / "report" / "chart.png").stat().st_size > 1000


def test_head_speech_low_passed_to_4khz_is_flagged_while_wer_holds(tmp_path):
    _, _, rows, md = run_matrix(tmp_path, lambda k, s: low_pass(voice(k)), picture)
    tts, clone, img = rows["tts"], rows["clone"], rows["img"]
    assert tts.correct == "pass" and "WER 2.00% to 2.00%" in tts.correct_detail       # WER alone would pass it
    (mos,) = tts.quality
    assert mos.verdict == "worse" and mos.head < mos.base - 1 and tts.flagged
    assert "**quality worse** (UTMOS" in tts.verdict()
    assert clone.flagged and clone.correct == "pass"           # audio length holds, the voice does not
    assert img.correct == "pass" and not img.flagged
    assert "**2 of 3 rows flagged**" in md and "a quality score that dropped" in md


def test_a_psnr_drop_whose_clip_score_drops_stays_a_failure(tmp_path):
    _, _, rows, _ = run_matrix(tmp_path, lambda k, s: voice(k), lambda k: picture(k, dark=True))
    img = rows["img"]
    assert img.correct == "FAIL" and img.quality[0].verdict == "worse" and img.flagged
    assert "**correctness changed**, **quality worse** (CLIP score" in img.verdict()


def test_calibrate_folds_quality_swings_once_and_they_widen_the_limit(tmp_path):
    """Two sessions per arm of identical code: base against base gives each score's A/A swing."""
    rd, models, _, _ = run_matrix(tmp_path, lambda k, s: voice(k, 0.002 * s, seed=s), picture,
                                  sessions=2, same_sha=True)
    facts = json.loads((rd / "results" / "tts" / "compare.json").read_text())["tts"]
    assert len(facts) == 3 and "quality" in facts[2]["self_base"]
    cal = tmp_path / "cal.json"
    calibrate.save({"_runs": ["q"], "_runs_correct": ["q"]}, cal)       # folded before quality swings existed
    assert calibrate.add_run(rd, models, cal) > 0
    entries = calibrate.load(cal)
    assert entries["_runs_quality"] == ["q"]
    for key, kind in (("h200:tts/tts/mos", "mos"), ("h200:clone/clone_upload/speaker_sim", "speaker_sim"),
                      ("h200:img/t2i/clip", "clip")):
        assert entries[key]["kind"] == kind and len(entries[key]["swings"]) == 3, key   # 2 sessions + base vs base
    assert "h200:tts/tts/correct" not in entries                         # correctness was folded before
    assert calibrate.add_run(rd, models, cal) == 0
    wide = {"h200:tts/tts/mos": {"kind": "mos", "swings": [1.0]}}
    rows = {r.model: r for r in report.build_rows(rd, models, calibration=wide)}
    assert rows["tts"].quality[0].limit == pytest.approx(1.5)
    assert rows["clone"].quality[0].limit == mf.QUALITY["mos"]


def test_a_judge_that_fails_leaves_its_score_out_and_the_others_standing(tmp_path):
    class Broken(FakeJudges):
        def speaker_sim(self, clip, ref):
            raise RuntimeError("no WavLM weights in /home/someone/.cache/huggingface")

    _, _, rows, md = run_matrix(tmp_path, lambda k, s: voice(k), picture, judges=Broken())
    mos, sim = rows["clone"].quality
    assert mos.verdict == "same" and sim.verdict == "n/a" and "no WavLM weights" in sim.note
    assert "/home/someone" not in md
    assert not rows["clone"].flagged and rows["clone"].correct == "pass"
    assert "speaker similarity n/a" in md


def test_the_arm_scores_quality_in_its_own_process_after_the_sessions(tmp_path, monkeypatch):
    out = tmp_path / "job"
    out.mkdir()
    case = mf.parse(copy.deepcopy(MANIFEST)).models["tts"].cases[0].to_dict()
    job = {"model": {"name": "tts", "cases": [case]}, "out": str(out), "sessions": 1, "python": sys.executable,
           "evalfw": str(Path.cwd()), "data": str(tmp_path), "datasets": MANIFEST["datasets"], "gpu": None}
    (out / "job.json").write_text(json.dumps(job))
    monkeypatch.setattr(arm, "job", job)
    monkeypatch.setattr(arm, "out", out)
    monkeypatch.setattr(arm, "job_path", out / "job.json")
    arm.score_quality()      # no saved outputs yet: the judges never load
    assert "quality: scored 0 case directories" in (out / "quality.log").read_text()
    assert json.loads((out / "status.json").read_text())["phase"] == "quality"


@pytest.mark.parametrize("stress", [False, True])
def test_quality_is_scored_after_the_sessions_unless_stress_replaced_the_cases(tmp_path, monkeypatch, stress):
    case = mf.parse(copy.deepcopy(MANIFEST)).models["tts"].cases[0].to_dict()
    assert case["quality"]
    job = {"model": {"name": "tts", "cases": [case]}, "arms": {"base": {"src": "b"}, "head": {"src": "h"}},
           "order": ["base", "head"], "sessions": 1, **({"stress": {"max_tokens": 16}} if stress else {})}
    calls = []
    monkeypatch.setattr(arm, "job", job)
    monkeypatch.setattr(arm, "out", tmp_path)
    monkeypatch.setattr(arm, "build_rust", lambda a: None)
    monkeypatch.setattr(arm, "requirements_problem", lambda a: "")
    monkeypatch.setattr(arm, "run_arm", lambda name, s: {"healthy": True, "mstar_from_arm_src": True})
    monkeypatch.setattr(arm, "score_quality", lambda: calls.append("quality"))
    monkeypatch.setattr(checks, "compare_model", lambda *a: calls.append("compare"))
    arm.run()
    assert calls == ([] if stress else ["quality", "compare"])
    assert json.loads((tmp_path / "done.json").read_text())["status"] == "done"
