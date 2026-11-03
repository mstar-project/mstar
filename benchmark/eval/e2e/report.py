"""Base vs head rows, verdicts, a PR-ready markdown summary and a PNG chart, from a ``compare`` run directory.

Run layout read here (written by ``compare`` and ``arm``):
  run.json                                         refs, GPU type, models, settings, GPU-hours
  results/<model>/done.json                        status: done | skipped | error | interrupted
  results/<model>/<arm>/s<k>/startup.json          healthy, startup_s, peak VRAM, which checkout the server imported
  results/<model>/<arm>/s<k>/<case>/*-c<N>.json    mstar-eval/1
  results/<model>/compare.json                     checks.compare_model
  results/<model>__abba/                           a flagged model's 2-session (ABBA) rerun of its flagged cases
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.eval.e2e.manifest import QUALITY, Case, Model
from benchmark.eval.schema import scrub, validate

DASHES = ("\u2014", "\u2013")
RERUN_TAG = "__abba"
FLOOR_FACTOR = 1.5          # a perf delta must clear 1.5x the largest drift two processes of the same code showed
CORRECT_FACTOR = 1.5        # and a correctness limit 1.5x the largest swing of its number
TEXT_MATCH_FLOOR = 0.1      # A/A noise never lowers the identical-output share below this
PSNR_FLOOR_DB = 20.0        # nor the PSNR limit below this
FLOOR_WINDOW = 8            # a floor follows the most recent A/A drifts of its row, so it can shrink again
OUTLIER_FACTOR = 3.0        # a drift over 3x the median of the row's others is one noisy run until another agrees
OUTLIER_MIN_PCT = 10.0      # and only a drift over 10% is ever left out
# say "different", not "worse": quality scores that hold clear the row. Not PSNR: a seed the server stops honouring
# gives different pixels of the same quality, and no output says which seed was used
CHANGE_KINDS = ("text_match",)
QUALITY_NAME = {"mos": "UTMOS", "speaker_sim": "speaker similarity", "clip": "CLIP score"}
QUALITY_DIGITS = {"mos": 2, "speaker_sim": 3, "clip": 1}


def floor_from(deltas: list[float]) -> float | None:
    """The cross-process noise floor (% of base) from a row's A/A drifts: FLOOR_FACTOR times the largest of the
    last FLOOR_WINDOW, leaving out a lone outlier (a second drift of that size makes it count)."""
    d = [abs(x) for x in deltas[-FLOOR_WINDOW:] if isinstance(x, (int, float)) and math.isfinite(x)]
    if not d:
        return None
    kept = [x for i, x in enumerate(d) if len(d) == 1 or x <= OUTLIER_MIN_PCT
            or x <= OUTLIER_FACTOR * statistics.median(d[:i] + d[i + 1:])]
    return FLOOR_FACTOR * max(kept)


@dataclass
class Stat:
    mean: float | None
    sd: float | None
    n: int

    @classmethod
    def of(cls, xs: list[float]) -> Stat:
        xs = [x for x in xs if x is not None and math.isfinite(x)]
        if not xs:
            return cls(None, None, 0)
        return cls(statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0, len(xs))


@dataclass
class Perf:
    metric: str
    conc: int
    base: Stat
    head: Stat
    lower_is_better: bool
    delta_pct: float | None = None      # head vs base, positive = head better
    noise_pct: float | None = None      # the larger of noise_sigma combined sd and the A/A floor, in % of base
    floor_pct: float | None = None      # cross-process noise floor from A/A runs (calibrate)
    verdict: str = "n/a"                # same | better | regression | noisy | n/a
    basis: str = ""                     # set when output lengths differ: "matched length 6/8", "per token", ...


@dataclass
class Quality:
    """One perceptual score of a row, each arm scored on its own outputs (higher is better for every kind)."""
    kind: str                           # manifest.QUALITY
    base: float | None
    head: float | None
    limit: float                        # the largest drop below base that still passes
    verdict: str = "n/a"                # same | better | worse | n/a
    note: str = ""                      # why there is no score, or where the limit came from

    def delta(self) -> str:
        return f"{self.head - self.base:+.{QUALITY_DIGITS[self.kind]}f}"

    def text(self) -> str:
        name, k = QUALITY_NAME[self.kind], QUALITY_DIGITS[self.kind]
        if self.base is None or self.head is None:
            return f"{name} n/a ({self.note})" if self.note else f"{name} n/a"
        return f"{name} {self.base:.{k}f} to {self.head:.{k}f} ({self.delta()})"


@dataclass
class Row:
    model: str
    family: str
    case: str
    modality: str
    starts: dict[str, bool | None]
    correct: str                        # pass | FAIL | changed (quality same) | incomplete | base broken
    correct_detail: str
    latency: Perf | None
    throughput: Perf | None
    extra: list[Perf] = field(default_factory=list)
    startup_s: dict[str, float | None] = field(default_factory=dict)
    peak_vram_gb: dict[str, float | None] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    request_path: dict | None = None
    note: str = ""
    no_baseline: bool = False           # perf row with no A/A noise floor for this GPU (calibrate it)
    rerun: str = ""                     # flagged with one session, judged on a 2-session (ABBA) rerun
    broken: str = ""                    # the one arm whose rust/ (or Rust frontend) did not build
    silent: str = ""                    # head | base | both: an arm with a point where no request succeeded
    rust: dict[str, str] = field(default_factory=dict)   # arms whose shared rust/ tree did not build: why
    quality: list[Quality] = field(default_factory=list)

    @property
    def quality_worse(self) -> list[Quality]:
        return [q for q in self.quality if q.verdict == "worse"]

    @property
    def flagged(self) -> bool:
        perf_bad = any(p and p.verdict == "regression" for p in (self.latency, self.throughput))
        lost = self.starts.get("head") is False and self.starts.get("base")
        return bool(perf_bad or self.correct == "FAIL" or lost or self.quality_worse)

    @property
    def unfinished(self) -> bool:
        """No verdict either way (job error, interrupted, skipped, not run, or base broken): exit 2 unless flagged."""
        return self.correct in ("incomplete", "base broken") and not self.flagged

    def verdict(self) -> str:
        if self.broken:
            return "**head does not build**" if self.broken == "head" else "base does not build"
        if self.starts.get("base") is False and self.starts.get("head") is False:
            return "neither arm started" + (f" ({self.note})" if self.note else "")
        if self.starts.get("head") is False:
            return "**head does not start**"
        if self.starts.get("base") is False:
            return "base does not start"
        if self.silent:
            return {"head": "**head answers no request**", "base": "base answers no request",
                    "both": "neither arm answers a request"}[self.silent]
        perfs_with_data = [p for p in (self.latency, self.throughput) if p and p.verdict != "n/a"]
        if self.correct in ("incomplete", "base broken") and not perfs_with_data and self.request_path is None:
            return self.note or self.correct
        parts = []
        if self.correct == "FAIL":
            parts.append("**correctness changed**")
        elif self.correct == "changed":
            parts.append("changed, quality same")
        if self.quality_worse:
            parts.append("**quality worse** (" + ", ".join(
                f"{QUALITY_NAME[q.kind]} {q.delta()}" for q in self.quality_worse) + ")")
        perfs = [p for p in (self.latency, self.throughput) if p]
        if any(p.verdict == "regression" for p in perfs):
            parts.append("**regression** " + ", ".join(
                f"c={p.conc} {p.delta_pct:+.0f}%" for p in perfs if p.verdict == "regression"))
        elif any(p.verdict == "better" for p in perfs):
            # one metric clearly better while another moved the other way inside its noise: not a speed-up claim
            mixed = any(p.verdict == "noisy" and (p.delta_pct or 0) < 0 for p in perfs)
            parts.append("mixed, within noise" if mixed else "faster")
        elif any(p.verdict == "noisy" for p in perfs):
            parts.append("same (within noise)")
        elif perfs:
            parts.append("same")
        return ", ".join(parts) or "same"


def _vals(files: list[Path], metric: str) -> list[float]:
    vals = []
    for f in files:
        try:
            js = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for rep in js.get("repeats", []):
            v = rep.get("metrics", {}).get(metric)
            if isinstance(v, (int, float)):
                vals.append(float(v))
    return vals


def lower_is_better(metric: str) -> bool:
    m = metric.lower()
    return not ("per_s" in m or m.endswith("rate") or m.startswith(("rtfx", "sim_", "take_turn", "yield")))


def judge_perf(p: Perf, perf_pct: float, noise_sigma: float, floor_pct: float | None = None) -> Perf:
    b, h = p.base, p.head
    if b.mean is None or h.mean is None or b.mean == 0:
        p.verdict = "n/a"
        return p
    change = (h.mean / b.mean - 1) * 100
    p.delta_pct = -change if p.lower_is_better else change
    sd = math.sqrt((b.sd or 0) ** 2 + (h.sd or 0) ** 2)
    p.floor_pct = floor_pct
    p.noise_pct = max(noise_sigma * sd / abs(b.mean) * 100, floor_pct or 0.0)
    if abs(p.delta_pct) <= perf_pct:
        p.verdict = "same"
    elif abs(p.delta_pct) <= p.noise_pct:
        p.verdict = "noisy"
    else:
        p.verdict = "better" if p.delta_pct > 0 else "regression"
    return p


def aa_swing(kind: str, f: dict) -> float | None:
    """How far one comparison of identical code moved a check's number (compare.json facts of an A/A run). Units
    follow the check: WER points, audio %, share of non-identical texts, min PSNR dB, action or rate difference."""
    if "self_base" in f:
        f = f["self_base"]
    if kind == "wer" and f.get("wer_base") is not None and f.get("wer_head") is not None:
        return abs(f["wer_head"] - f["wer_base"]) * 100
    if kind == "audio_len" and f.get("audio_s_base"):
        return abs(f["audio_s_head"] - f["audio_s_base"]) / f["audio_s_base"] * 100
    if kind == "text_match" and f.get("text_n"):
        return 1 - f["text_identical"] / f["text_n"]
    if kind == "psnr" and f.get("psnr_min_db") is not None and math.isfinite(f["psnr_min_db"]):
        return f["psnr_min_db"]
    if kind == "action_diff" and f.get("action_max_abs_diff") is not None:
        return f["action_max_abs_diff"]
    if kind == "duplex" and f.get("take_turn_rate_base") is not None and f.get("take_turn_rate_head") is not None:
        d = [abs(f["take_turn_rate_head"] - f["take_turn_rate_base"])]
        if f.get("yield_rate_base") is not None and f.get("yield_rate_head") is not None:
            d.append(abs(f["yield_rate_head"] - f["yield_rate_base"]))
        return max(d)
    return None


def quality_swings(f: dict) -> dict[str, float]:
    """Per quality judge, how far one comparison of identical code moved its mean score (compare.json facts)."""
    if "self_base" in f:
        f = f["self_base"]
    return {k: abs(q["head"] - q["base"]) for k, q in (f.get("quality") or {}).items()
            if q.get("base") is not None and q.get("head") is not None}


def judge_quality(case: Case, facts: list[dict], swings: dict[str, list[float]] | None = None) -> list[Quality]:
    """Each of the case's quality scores, judged on its worst session: head fails when it scores below base by more
    than the larger of the manifest floor (``manifest.QUALITY``) and CORRECT_FACTOR times the largest A/A swing."""
    sessions = [f for f in facts if "self_base" not in f]
    out = []
    for kind in case.quality:
        floor = QUALITY[kind]
        sw = [x for x in (swings or {}).get(kind) or [] if math.isfinite(x)]
        limit = max(floor, CORRECT_FACTOR * max(sw)) if sw else floor
        note = f"limit {limit:.2g} from A/A (floor {floor:.2g})" if limit > floor else ""
        scored = [q for q in ((f.get("quality") or {}).get(kind) or {} for f in sessions)
                  if q.get("base") is not None and q.get("head") is not None]
        if not scored:
            why = next((q.get("error") for q in ((f.get("quality") or {}).get(kind) or {} for f in sessions)
                        if q.get("error")), "not scored")
            out.append(Quality(kind, None, None, limit, "n/a", why))
            continue
        worst = min(scored, key=lambda q: q["head"] - q["base"])
        d = worst["head"] - worst["base"]
        verdict = "worse" if d < -limit else "better" if d > limit else "same"
        out.append(Quality(kind, worst["base"], worst["head"], limit, verdict, note))
    return out


def limit_only(facts: list[dict], errors: dict[str, int]) -> bool:
    """Whether every session that failed did so only on its check's limit (a lower PSNR, fewer identical texts),
    with no missing, cut-short, undecodable or reshaped outputs and no extra head errors."""
    hard = ("missing", "only_base", "head_undecodable", "shape_mismatch", "text_truncated")
    sessions = [f for f in facts if "self_base" not in f]
    return bool(sessions) and errors.get("head", 0) <= errors.get("base", 0) and not any(
        f.get(k) for f in sessions for k in hard)


def correct_limit(check: dict, swings: list[float] | None) -> tuple[dict, str]:
    """The check with its limit widened to clear CORRECT_FACTOR times the largest A/A swing; the manifest value
    stays the floor (calibration only loosens a limit that noise alone would trip). Returns (check, note)."""
    swings = [x for x in swings or [] if math.isfinite(x)]
    if not swings:
        return check, ""
    kind, ck = check["kind"], dict(check)
    worst = max(swings) if kind != "psnr" else min(swings)
    if kind in ("wer", "audio_len", "duplex", "action_diff"):
        key = {"wer": "max_increase", "audio_len": "pct", "duplex": "tol", "action_diff": "tol"}[kind]
        ck[key] = max(float(check[key]), CORRECT_FACTOR * worst)
        unit = {"wer": " points", "audio_len": "%"}.get(kind, "")
        changed = ck[key] > float(check[key])
        note = f"limit {ck[key]:.2g}{unit} from A/A (manifest {float(check[key]):.2g}{unit})" if changed else ""
    elif kind == "text_match":
        ck["min_match"] = min(float(check["min_match"]), max(TEXT_MATCH_FLOOR, 1 - CORRECT_FACTOR * worst))
        changed = ck["min_match"] < float(check["min_match"])
        note = (f"identical share limit {ck['min_match']:.2f} from A/A (manifest {check['min_match']})"
                if changed else "")
    elif kind == "psnr":
        ck["min_db"] = min(float(check["min_db"]), max(PSNR_FLOOR_DB, worst - 3))
        changed = ck["min_db"] < float(check["min_db"])
        note = f"PSNR limit {ck['min_db']:.1f} dB from A/A (manifest {check['min_db']})" if changed else ""
    else:
        return check, ""
    return ck, note


def _judge_session(ck: dict, f: dict) -> tuple[str, str]:
    """(verdict, detail) of one session's facts against the (already widened) check."""
    if "missing" in f:   # head alone produced nothing where base did: a failure, not a gap
        return ("FAIL" if f["missing"] == ["head"] else "incomplete"), f"outputs missing ({', '.join(f['missing'])})"
    if f.get("unexercised"):   # base never reached the state the case is about: nothing was tested
        return "incomplete", f"not exercised on {f['unexercised']}"
    if f.get("only_base"):
        return "FAIL", f"head lacks {len(f['only_base'])} outputs base returned ({', '.join(f['only_base'][:3])})"
    if f.get("head_undecodable"):
        bad = f["head_undecodable"]
        return "FAIL", f"{len(bad)} head outputs do not decode ({', '.join(bad[:3])})"
    kind = ck["kind"]
    ok, detail = True, ""
    if kind == "text_match":
        n, same = f.get("text_n", 0), f.get("text_identical", 0)
        if not n:
            return "incomplete", "no text outputs"
        sim, lr = f.get("text_sim_median"), f.get("text_len_ratio_median")
        close = sim is not None and sim >= float(ck.get("min_sim", 0.6)) and 0.8 <= (lr or 0) <= 1.25
        cut = int(f.get("text_truncated", 0))
        ok = (same / n >= float(ck["min_match"]) or close) and not cut
        detail = f"{same}/{n} identical"
        if same < n and sim is not None:
            detail += f", similarity {sim:.2f}, length ratio {lr:.2f}"
        if cut:
            detail += f", {cut} head replies are base's cut short"
    elif kind == "wer":
        wb, wh = f.get("wer_base"), f.get("wer_head")
        if wb is None or wh is None:
            return "incomplete", "no WER"
        ok = (wh - wb) * 100 <= float(ck["max_increase"])
        detail = f"WER {wb * 100:.2f}% to {wh * 100:.2f}%"
        if f.get("text_n"):
            detail += f", {f['text_identical']}/{f['text_n']} identical transcripts"
    elif kind == "psnr":
        mn, med = f.get("psnr_min_db"), f.get("psnr_median_db")
        if f.get("shape_mismatch"):
            ok, detail = False, f"{f['shape_mismatch']} outputs changed shape"
        elif mn is None and med is None:
            return "incomplete", "no media outputs"
        else:
            ok = mn is None or mn >= float(ck["min_db"])
            detail = "bit-identical" if mn == math.inf else f"PSNR min {mn:.1f} dB, median {med:.1f} dB"
    elif kind == "action_diff":
        d = f.get("action_max_abs_diff")
        if d is None:
            return "incomplete", "no actions"
        if not math.isfinite(d):
            return "FAIL", "actions changed shape or are not finite"
        ok = d <= float(ck["tol"])
        detail = "bit-identical" if d == 0 else f"max action diff {d:.4f}"
    elif kind == "audio_len":
        ab, ah = f.get("audio_s_base"), f.get("audio_s_head")
        if not ab:
            return "incomplete", "no audio"
        ok = abs(ah - ab) / ab * 100 <= float(ck["pct"])
        detail = f"audio {ab:.1f} s to {ah:.1f} s"
        if f.get("text_n"):
            detail += f", {f['text_identical']}/{f['text_n']} identical texts"
    elif kind == "duplex":
        tol = float(ck["tol"])
        tb, th = f.get("take_turn_rate_base"), f.get("take_turn_rate_head")
        yb, yh = f.get("yield_rate_base"), f.get("yield_rate_head")
        if tb is None or th is None:
            return "incomplete", "no duplex metrics"
        ok = th >= tb - tol and (yb is None or yh is None or yh >= yb - tol)
        detail = f"take-turn {tb:.2f} to {th:.2f}"
        if yb is not None and yh is not None:
            detail += f", yield {yb:.2f} to {yh:.2f}"
    elif kind == "status_match":
        diffs = f.get("diffs", [])
        ok = not diffs
        detail = f"{f.get('n', 0) - len(diffs)}/{f.get('n', 0)} same" + (
            "; " + ", ".join(f"{d['case']} {d['base']} to {d['head']}" for d in diffs[:4]) if diffs else "")
    return ("pass" if ok else "FAIL"), detail


def judge_correct(case: Case, facts: list[dict], errors: dict[str, int],
                  swings: list[float] | None = None) -> tuple[str, str]:
    """(verdict, detail) from compare.json facts for one case: every session must pass, and head must not have
    more request errors than base (even when outputs are missing). ``swings`` are this row's A/A swings
    (calibrate) that widen the manifest limit."""
    ck, limit_note = correct_limit(case.check, swings)
    eb, eh = errors.get("base", 0), errors.get("head", 0)
    sessions = [f for f in facts if "self_base" not in f]
    results = [_judge_session(ck, f) for f in sessions] or [("incomplete", "no comparison")]
    label = (lambda i: f"session {i + 1}: ") if len(results) > 1 else (lambda i: "")
    worst = next((i for want in ("FAIL", "incomplete") for i, r in enumerate(results) if r[0] == want), None)
    verdict, detail = (results[worst][0], label(worst) + results[worst][1]) if worst is not None else results[0]
    if worst is None and len(results) > 1:
        detail += f"; all {len(results)} sessions pass"
    selfb = next((x["self_base"] for x in facts if "self_base" in x), None)
    if ck["kind"] == "text_match" and selfb and selfb.get("text_n"):
        detail += f" (base vs base {selfb['text_identical']}/{selfb['text_n']})"
    if limit_note:
        detail += f" ({limit_note})"
    if eb or eh:
        detail += f"; errors {eb} vs {eh}"
    return ("FAIL" if eh > eb else verdict), detail


# Output length is the model's choice for these (greedy text can fork, sampled speech varies); latency then measures
# the length as much as the server. Images, video, frames, actions and ASR have a fixed output size.
LENGTH_KINDS = {"chat_text", "chat_image", "chat_audio", "chat_video", "chat_count", "gen_clone", "tts"}
E2E_LATENCY = {"lat_p50_s", "e2e_s_p50"}    # c=1 latency metrics that are the p50 of per-request e2e_s
LENGTH_TOL = 0.02                           # two outputs "match" within 2% (and one token)
MIN_MATCHED = 3                             # matched requests per pass needed to judge latency on them alone
UNIT_NAME = {"tok": "token", "char": "character", "audio_s": "audio second"}


def out_len(r: dict) -> tuple[float, str] | None:
    """One request's output length: audio seconds when it returned audio, else SSE content deltas (one per decode
    step), else characters."""
    if r.get("audio_s"):
        return float(r["audio_s"]), "audio_s"
    if r.get("n_chunks"):
        return float(r["n_chunks"]), "tok"
    if r.get("text"):
        return float(len(r["text"])), "char"
    return None


def same_len(a: float, b: float, unit: str) -> bool:
    return abs(a - b) <= max(LENGTH_TOL * max(a, b), 1.0 if unit == "tok" else 0.0)


def _requests(files: list[Path]) -> list[tuple[float, dict[str, tuple[float, float, str]]]]:
    """Per measured pass: (wall_s, {request id: (e2e_s, output length, unit)}) over the ok requests."""
    out = []
    for f in files:
        try:
            js = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        for rep in js.get("repeats", []):
            reqs = {}
            for r in rep.get("requests", []):
                ln = out_len(r) if r.get("ok") and isinstance(r.get("e2e_s"), (int, float)) else None
                if ln and ln[0] > 0:
                    reqs[str(r.get("id"))] = (float(r["e2e_s"]), ln[0], ln[1])
            out.append((float(rep.get("wall_s") or 0.0), reqs))
    return out


def _matched(files: dict[str, list[Path]]):
    """Per-pass request maps of both arms at one concurrency, the request ids every pass has, and those whose
    lengths matched in every pass (one fixed set: a subset that changed from pass to pass would add its own
    spread). None when the arms share no requests."""
    rb, rh = _requests(files["base"]), _requests(files["head"])
    pairs = [(b, h, sorted(b.keys() & h.keys())) for (_, b), (_, h) in zip(rb, rh, strict=False)]
    pairs = [p for p in pairs if p[2]]
    if not pairs:
        return None
    every = set.intersection(*(set(ids) for _, _, ids in pairs))
    same = sorted(i for i in every if all(same_len(b[i][1], h[i][1], b[i][2]) for b, h, _ in pairs))
    return rb, rh, pairs, every, same, pairs[0][0][pairs[0][2][0]][2]


def length_aware(case: Case, files_c1: dict[str, list[Path]], files_cn: dict[str, list[Path]],
                 lat: Perf | None, thru: Perf | None) -> tuple[Perf | None, Perf | None]:
    """When base and head returned different output lengths for the same requests, compare like with like: c=1
    latency over only the requests whose lengths matched (when at least MIN_MATCHED per pass did), else latency per
    output token or audio second; c=max throughput as output tokens (audio seconds) per second. Each concurrency
    is judged on its own outputs' lengths; unchanged when every length matched, or for fixed-size outputs."""
    if case.kind not in LENGTH_KINDS:
        return lat, thru
    m1 = _matched(files_c1)
    if lat and lat.metric in E2E_LATENCY and m1 and len(m1[4]) < len(m1[3]):
        rb, rh, pairs, every, same, unit = m1
        if len(same) >= MIN_MATCHED and len(same) >= 0.25 * len(every):
            vb = [statistics.median(b[i][0] for i in same) for b, _, _ in pairs]
            vh = [statistics.median(h[i][0] for i in same) for _, h, _ in pairs]
            lat = Perf(lat.metric, lat.conc, Stat.of(vb), Stat.of(vh), True,
                       basis=f"matched length {len(same)}/{len(every)}")
        else:
            def per_unit(passes):
                return [statistics.median(e / ln for e, ln, _ in reqs.values()) for _, reqs in passes if reqs]

            lat = Perf(f"s_per_{unit}_p50", lat.conc, Stat.of(per_unit(rb)), Stat.of(per_unit(rh)), True,
                       basis=f"per {UNIT_NAME[unit]}, lengths differ in {len(every) - len(same)}/{len(every)}")
    mn = _matched(files_cn)
    if thru and mn and len(mn[4]) < len(mn[3]) and thru.metric != f"{mn[5]}_per_s":
        rb, rh, _, _, _, unit = mn

        def rate(passes):
            return [sum(ln for _, ln, _ in reqs.values()) / wall for wall, reqs in passes if wall > 0 and reqs]

        tb, th = rate(rb), rate(rh)
        if tb and th:
            thru = Perf(f"{unit}_per_s", thru.conc, Stat.of(tb), Stat.of(th), False,
                        basis=f"{UNIT_NAME[unit]}s per second, lengths differ")
    return lat, thru


def _read(path: Path, absent: dict, unreadable: dict) -> dict:
    """A run file, or ``absent`` / ``unreadable`` (a job killed while writing it) instead of an error."""
    if not path.exists():
        return absent
    try:
        js = json.loads(path.read_text())
    except (OSError, ValueError):
        return unreadable
    return js if isinstance(js, dict) else unreadable


@dataclass
class _Arms:
    """What one results directory (a model, or its ABBA rerun) says about both arms."""
    done: dict
    compare: dict
    starts: dict[str, bool | None]
    startup: dict[str, float | None]
    vram: dict[str, float | None]
    md: Path

    @classmethod
    def load(cls, md: Path) -> _Arms:
        done = _read(md / "done.json", {"status": "not run"}, {"status": "error", "reason": "unreadable done.json"})
        compare = _read(md / "compare.json", {}, {})
        starts, startup, vram = {}, {}, {}
        for arm in ("base", "head"):
            infos = [_read(p, {}, {"healthy": None}) for p in sorted((md / arm).glob("s*/startup.json"))]
            starts[arm] = all(i.get("healthy") for i in infos) if infos else None
            ups = [i["startup_s"] for i in infos if i.get("healthy") and i.get("startup_s") is not None]
            startup[arm] = min(ups) if ups else None
            vs = [i.get("peak_vram_gb") for i in infos if i.get("peak_vram_gb")]
            vram[arm] = max(vs) if vs else None
        return cls(done, compare, starts, startup, vram, md)

    def alive_after(self, case_id: str) -> dict[str, dict[str, bool | None]]:
        """Per session: whether each arm's server still answered /health after this case."""
        out: dict[str, dict[str, bool | None]] = {}
        for arm in ("base", "head"):
            for p in sorted((self.md / arm).glob("s*/startup.json")):
                case = _read(p, {}, {}).get("cases", {}).get(case_id) or {}
                out.setdefault(p.parent.name, {"base": None, "head": None})[arm] = case.get("server_alive_after")
        return out

    def hung(self, case_id: str) -> dict[str, float | None]:
        """Per arm, the case's wall cap in seconds when that arm's case ran past it in any session, else None."""
        out: dict[str, float | None] = {"base": None, "head": None}
        for arm in ("base", "head"):
            for p in sorted((self.md / arm).glob("s*/startup.json")):
                case = _read(p, {}, {}).get("cases", {}).get(case_id) or {}
                if case.get("hung"):
                    out[arm] = float(case.get("cap_s") or 0)
        return out

    def health(self, case_id: str) -> dict[str, dict[str, str]]:
        """Per arm, the worker health findings after this case in any session (``arm.health_after``), by kind."""
        out: dict[str, dict[str, str]] = {"base": {}, "head": {}}
        for arm in ("base", "head"):
            for p in sorted((self.md / arm).glob("s*/startup.json")):
                case = _read(p, {}, {}).get("cases", {}).get(case_id) or {}
                for x in case.get("health") or []:
                    out[arm].setdefault(_finding_kind(x), x)
        return out

    def files(self, case_id: str) -> dict[str, list[Path]]:
        """Each arm's results files for a case; a file that is not a complete mstar-eval result (a client that died
        while writing it) does not count, so its point shows up as missing."""
        return {arm: [f for f in sorted((self.md / arm).glob(f"s*/{case_id}/*-c*.json")) if _complete(f)]
                for arm in ("base", "head")}


def _finding_kind(finding: str) -> str:
    """``/dev/shm +3 entries`` -> ``/dev/shm leak``: findings compare by kind, not count."""
    if finding.startswith("/dev/shm"):
        return "/dev/shm leak"
    if finding.endswith("server process(es) exited"):
        return "server process exited"
    if finding.endswith("server did not start"):
        return "server did not start"
    return finding


CRASH_KINDS = ("server process exited", "log: segfault", "server did not start")


def _complete(f: Path) -> bool:
    try:
        js = json.loads(f.read_text())
    except (OSError, ValueError):
        return False
    return validate(js) == [] and bool(js["repeats"])


def calibration_key(gpu: str, model: str, case: str, metric: str, conc: int) -> str:
    return f"{gpu}:{model}/{case}/{metric}@c{conc}"


def correct_key(gpu: str, model: str, case: str) -> str:
    return f"{gpu}:{model}/{case}/correct"


def quality_key(gpu: str, model: str, case: str, kind: str) -> str:
    return f"{gpu}:{model}/{case}/{kind}"


_ROOT_HOME = re.compile(r"/root(?=/|\b)")


def _public(text: str) -> str:
    return _ROOT_HOME.sub("~", scrub(text))


def _row(a: _Arms, m: Model, c: Case, status: dict, gpu: str, calibration: dict) -> Row:
    done = a.done
    note = "" if done["status"] == "done" else (
        status.get("reason") if done["status"] == "not run" and status.get("reason")
        else f"{done['status']}: {done.get('reason', '')}".strip(": "))
    perf_pct, sigma = float(m.thresholds.get("perf_pct", 5)), float(m.thresholds.get("noise_sigma", 2))
    files = a.files(c.id)
    mute = ""

    def at(conc: int) -> dict[str, list[Path]]:
        return {k: [f for f in v if f.name.endswith(f"-c{conc}.json")] for k, v in files.items()}

    def perf(metric: str | None, conc: int) -> Perf | None:
        if not metric:
            return None
        fs = at(conc)
        return Perf(metric, conc, Stat.of(_vals(fs["base"], metric)), Stat.of(_vals(fs["head"], metric)),
                    lower_is_better(metric))

    def floor_of(p: Perf, orig: str | None) -> float | None:
        for metric in dict.fromkeys((p.metric, orig)):   # a normalized metric falls back to its source
            e = calibration.get(calibration_key(gpu, m.name, c.id, metric, p.conc)) or {}
            if (f := floor_from(e.get("deltas") or [])) is not None:
                return f
        return None

    errs = {arm: int(sum(_vals(files[arm], "errors"))) for arm in ("base", "head")}
    timed = c.driver in ("suite", "client")
    lat = perf(c.perf.get("latency"), min(c.conc)) if timed else None
    thru = perf(c.perf.get("throughput"), max(c.conc)) if timed else None
    lat, thru = length_aware(c, at(min(c.conc)), at(max(c.conc)), lat, thru)
    floors = {}
    for p, orig in ((lat, c.perf.get("latency")), (thru, c.perf.get("throughput"))):
        if p:
            floors[id(p)] = floor_of(p, orig)
            judge_perf(p, perf_pct, sigma, floors[id(p)])
    extra = [p for p in (perf(x, min(c.conc)) for x in c.perf.get("extra", [])) if p]
    for p in extra:
        judge_perf(p, perf_pct, sigma, floor_of(p, p.metric))
    broken = done.get("broken") if done["status"] == "error" and done.get("broken") in ("base", "head") else ""
    quality: list[Quality] = []
    if broken:
        corr, detail = ("FAIL" if broken == "head" else "base broken"), f"{broken} did not build; {note}"
    elif a.starts.get("base") is False:
        corr, detail = "base broken", "base did not start"
    elif a.starts.get("head") is False:
        corr, detail = "FAIL", "head did not start"
    elif done["status"] == "error" and timed and point_gaps(a, files, c)[0]:
        gaps = point_gaps(a, files, c)[0]   # the job broke, and head is the arm that lacks results
        corr, detail = "FAIL", f"head wrote no results for {', '.join(gaps)}; {note}"
    elif done["status"] != "done":
        corr, detail = "incomplete", note
    else:
        facts = a.compare.get(c.id, [])
        swings = (calibration.get(correct_key(gpu, m.name, c.id)) or {}).get("swings")
        corr, detail = judge_correct(c, facts, errs, swings)
        quality = judge_quality(c, facts, {k: (calibration.get(quality_key(gpu, m.name, c.id, k)) or {}).get("swings")
                                           for k in c.quality})
        if (corr == "FAIL" and c.check["kind"] in CHANGE_KINDS and quality and limit_only(facts, errs)
                and all(q.verdict in ("same", "better") for q in quality)):
            corr = "changed"
        gaps, absent = point_gaps(a, files, c) if timed else ([], [])
        silent = silent_points(files) if timed else {"base": [], "head": []}
        died = [s for s, alive in a.alive_after(c.id).items() if alive["head"] is False and alive["base"] is not False]
        if gaps:
            corr, detail = "FAIL", f"head wrote no results for {', '.join(gaps)}; {detail}"
        elif died:
            corr, detail = "FAIL", f"head server died during this case ({', '.join(died)}); {detail}"
        elif head_only := [p for p in silent["head"] if p not in silent["base"]]:
            mute = "head"
            corr, detail = "FAIL", f"head answered no request at {', '.join(head_only)}; {detail}"
        elif silent["base"]:   # nothing to judge head against at those points, whatever head did
            mute = "both" if silent["head"] else "base"
            who = "neither arm answered a request" if mute == "both" else "base answered no request"
            corr, detail = "base broken", f"{who} at {', '.join(silent['base'])}; {detail}"
        elif absent and corr == "pass":
            corr, detail = "incomplete", f"no results for {', '.join(absent)}; {detail}"
        hung = a.hung(c.id)
        if hung["base"] is not None:   # nothing to judge head against, timings included
            corr, detail = "base broken", (f"base hung: still running after the {hung['base']:.0f} s case cap "
                                           "(stacks saved)")
            for p in (lat, thru, *extra):
                if p:
                    p.verdict = "n/a"
        elif hung["head"] is not None:
            corr, detail = "FAIL", (f"head hung: still running after the {hung['head']:.0f} s case cap; base finished "
                                    "(stacks in head/s*/" + c.id + "/stacks.txt)")
        health = a.health(c.id)
        head_only = [health["head"][k] for k in health["head"] if k not in health["base"]]
        base_crash = [health["base"][k] for k in CRASH_KINDS if k in health["base"]]
        if m.health and any(alive["base"] is False for alive in a.alive_after(c.id).values()):
            base_crash = base_crash or ["server stopped answering"]
        if base_crash and hung["base"] is None:   # nothing to judge head against, timings included
            corr, detail = "base broken", f"base worker crashed: {', '.join(base_crash)}; {detail}"
            for p in (lat, thru, *extra):
                if p:
                    p.verdict = "n/a"
        elif head_only and corr != "base broken" and hung["head"] is None:
            seen = ", ".join(health["base"].values()) or "none"
            corr, detail = "FAIL", f"head worker health: {', '.join(head_only)} (base: {seen}); {detail}"
    if corr not in ("pass", "changed", "FAIL"):
        quality = []
    with_data = [p for p in (lat, thru) if p and p.base.mean is not None and p.head.mean is not None]
    rp = scrub((a.compare.get(c.id) or [None])[0]) if c.driver in ("rp", "invariants") else None
    note, detail = _public(note), _public(detail)   # reasons can quote server errors: no paths or addresses
    for q in quality:
        q.note = _public(q.note)
    return Row(model=m.name, family=m.family, case=c.id, modality=c.modality, starts=a.starts, correct=corr,
               correct_detail=detail, latency=lat, throughput=thru, extra=extra, startup_s=a.startup,
               peak_vram_gb=a.vram, errors=errs, request_path=rp, note=note, broken=broken, silent=mute,
               rust={k: _public(str(v)) for k, v in (done.get("rust") or {}).items() if v}, quality=quality,
               no_baseline=bool(with_data) and all(floors.get(id(p)) is None for p in with_data))


def point_gaps(a: _Arms, files: dict[str, list[Path]], c: Case) -> tuple[list[str], list[str]]:
    """(points base wrote results for and head did not, points base did not write), a point being a server session
    and a concurrency (a client that timed out, a server that died)."""
    def points(arm: str) -> set[tuple[str, int]]:
        return {(f.parent.parent.name, conc) for f in files[arm] for conc in c.conc
                if f.name.endswith(f"-c{conc}.json")}

    base, head = points("base"), points("head")
    sessions = {p.parent.name for arm in ("base", "head") for p in (a.md / arm).glob("s*/startup.json")}
    expected = {(s, conc) for s in sessions for conc in c.conc}
    fmt = lambda pts: [f"{s} c={conc}" for s, conc in sorted(pts)]  # noqa: E731
    return fmt(base - head), fmt(expected - base)   # base missing a point: nothing to judge head against


def silent_points(files: dict[str, list[Path]]) -> dict[str, list[str]]:
    """Per arm, the points (server session and concurrency) it wrote results for in which no request succeeded:
    a server that answers every request with an error still yields metrics (WER 100%, 0 req/s) that compare equal."""
    out = {}
    for arm, fs in files.items():
        ok: dict[tuple[str, int], int] = {}
        for f in fs:
            conc = re.search(r"-c(\d+)\.json$", f.name)
            if not conc:
                continue
            try:
                reps = json.loads(f.read_text()).get("repeats", [])
            except (OSError, ValueError):
                continue
            point = (f.parent.parent.name, int(conc.group(1)))
            ok[point] = ok.get(point, 0) + sum(1 for rep in reps for r in rep.get("requests", []) if r.get("ok"))
        out[arm] = [f"{s} c={conc}" for (s, conc), n in sorted(ok.items()) if n == 0]
    return out


def build_rows(run_dir: Path, models: list[Model], calibration: dict | None = None,
               reruns: bool = True) -> list[Row]:
    if calibration is None:
        from benchmark.eval.e2e import calibrate

        calibration = calibrate.load()
    run = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    gpu, status = run.get("gpu", "unknown"), run.get("model_status", {})
    rows: list[Row] = []
    res = run_dir / "results"
    for m in models:
        first = _Arms.load(res / m.name)
        rerun = _Arms.load(res / (m.name + RERUN_TAG)) if reruns and (res / (m.name + RERUN_TAG)).exists() else None
        for c in m.cases:
            r = _row(first, m, c, status.get(m.name, {}), gpu, calibration)
            if rerun and rerun.done.get("status") == "done" and any(rerun.files(c.id).values()):
                r2 = _row(rerun, m, c, {}, gpu, calibration)
                was = ", ".join(f"c={p.conc} {p.delta_pct:+.0f}%" for p in (r.latency, r.throughput)
                                if p and p.verdict == "regression")
                r2.rerun = f"flagged with 1 session ({was or r.verdict()}); judged on a 2-session ABBA rerun"
                r2.startup_s, r2.peak_vram_gb = r.startup_s, r.peak_vram_gb
                # The rerun replaces the timings only: a hard failure seen in the first run stays.
                r2.errors = {arm: r.errors.get(arm, 0) + r2.errors.get(arm, 0) for arm in ("base", "head")}
                if r.correct == "FAIL":
                    r2.correct, r2.correct_detail = "FAIL", f"first run: {r.correct_detail}"
                    r2.silent = r2.silent or r.silent
                if r.quality_worse:
                    r2.quality = r.quality
                r = r2
            rows.append(r)
    return rows


def flagged_perf_cases(rows: list[Row]) -> list[str]:
    """Cases whose perf verdict is a regression: what an ABBA rerun repeats."""
    return [r.case for r in rows if any(p and p.verdict == "regression" for p in (r.latency, r.throughput))]


def fmt_val(metric: str, v: float | None) -> str:
    if v is None:
        return "n/a"
    m = metric.lower()
    if m.startswith("s_per_"):   # length-normalized latency (length_aware)
        unit = m[len("s_per_"):].removesuffix("_p50")
        return f"{v:.3f} s/audio s" if unit == "audio_s" else f"{v * 1000:.1f} ms/{UNIT_NAME.get(unit, unit)}"
    if m in ("tok_per_s", "char_per_s"):
        return f"{v:.0f} {m.split('_')[0]}/s"
    if "_ms" in m:
        return f"{v:.1f} ms"
    if m.endswith(("_s", "_s_p50", "_s_p90", "_s_mean")) and "per_s" not in m:
        return f"{v * 1000:.0f} ms" if v < 2 else f"{v:.2f} s"
    unit = {"req_per_s": "req/s", "audio_s_per_s": "audio s/s", "frames_per_s": "frames/s"}.get(m, "")
    return f"{v:.3g} {unit}".strip() if v < 100 else f"{v:.0f} {unit}".strip()


def fmt_perf(p: Perf | None, with_sd: bool = False) -> str:
    if not p or (p.base.mean is None and p.head.mean is None):
        return ""

    def one(s: Stat) -> str:
        t = fmt_val(p.metric, s.mean)
        if with_sd and s.sd and s.mean:
            t += f" +- {s.sd / abs(s.mean) * 100:.1f}%"
        return t

    d = f" ({p.delta_pct:+.1f}%)" if p.delta_pct is not None else ""
    return f"{one(p.base)} to {one(p.head)}{d}" + (f" [{p.basis}]" if p.basis else "")


def starts_cell(r: Row) -> str:
    b, h = r.starts.get("base"), r.starts.get("head")
    if b and h:
        return "yes"
    return f"base {'yes' if b else 'no' if b is False else '?'}, head {'yes' if h else 'no' if h is False else '?'}"


def errors_cell(r: Row) -> str:
    eb, eh = r.errors.get("base", 0), r.errors.get("head", 0)
    return f"**{eb} / {eh}**" if eh > eb or r.silent else f"{eb} / {eh}"


def hosts_text(run: dict) -> str:
    slots = run.get("slots") or []
    hosts = sorted({s.rsplit(":", 1)[0] for s in slots})
    if not slots:
        return "no GPU"
    return f"{len(slots)} GPU{'s' if len(slots) > 1 else ''} on {len(hosts)} host{'s' if len(hosts) > 1 else ''}"


def render_md(rows: list[Row], run: dict, chart_url: str | None = None) -> str:
    base, head = run["base"], run["head"]
    gpu = run.get("gpu", "?").upper()
    pct, sigma = run.get("perf_pct", 5), run.get("noise_sigma", 2)
    perf_rows = [r for r in rows if r.latency or r.throughput]
    rp_rows = [r for r in rows if not (r.latency or r.throughput)]
    flagged = [r for r in rows if r.flagged]
    unfinished = [r for r in rows if r.unfinished]
    models = sorted({r.model for r in rows})
    lines = [f"## Base vs head matrix: {head['label']} against {base['label']}", ""]
    if flagged:
        lines.append(f"**{len(flagged)} of {len(rows)} rows flagged** (regression over {pct}% beyond noise, a "
                     "correctness change, a quality score that dropped, or a head that no longer builds or starts).")
    elif not unfinished:
        lines.append(f"No regressions: {len(rows)} rows across {len(models)} models, every one starts, answers its "
                     f"requests, keeps its outputs, and stays within {pct}% (or within noise).")
    else:
        lines.append(f"No regressions in {len(rows) - len(unfinished)} complete rows.")
    if unfinished:
        lines.append(f"**{len(unfinished)} rows have no verdict** (incomplete, a broken base, or a base that answered "
                     f"no request; {', '.join(sorted({r.model for r in unfinished}))}); see the table.")
    erred = [r for r in rows if any(r.errors.values())]
    if erred:
        lines.append(f"**{len(erred)} rows had request errors** (base {sum(r.errors.get('base', 0) for r in erred)}, "
                     f"head {sum(r.errors.get('head', 0) for r in erred)} in total); see the Errors column.")
    no_rust = {r.model: r.rust for r in rows if r.rust}
    if no_rust:
        lines.append(f"**No per-arm `rust/` build for {', '.join(no_rust)}** "
                     f"({'; '.join(sorted({p for d in no_rust.values() for p in d.values()}))}): both arms of those "
                     "models ran the environment's `mstar_rust`, so they do not cover `rust/`.")
    reruns = [r for r in rows if r.rerun]
    if reruns:
        lines.append(f"{len(reruns)} rows flagged with one server session per arm were rerun with two (ABBA) and are "
                     f"judged on the rerun: {', '.join(f'{r.model} / {r.case}' for r in reruns)}.")
    bare = sorted({r.model for r in rows if r.no_baseline})
    if bare:
        lines.append(f"No A/A noise baseline on {gpu} for {', '.join(bare)}: those rows are judged on the passes' "
                     "spread alone (`python -m benchmark.eval calibrate` on a main-vs-main run of them).")
    multi = run.get("gpus_per_model") or {}
    where = (f"on one {gpu} per model ({', '.join(f'{n}: {k}' for n, k in sorted(multi.items()))} {gpu}s of one host)"
             if multi else f"on one {gpu} per model")
    lines += ["", f"Served through `mstar serve` {where}, base and head back to back on the same GPUs "
              f"(order alternated per model), {run.get('repeats', 3)} measured passes per point, "
              f"{run.get('sessions', 1)} server session(s) per arm. Deltas are signed so that positive means head "
              "is better (lower latency, higher throughput).", ""]
    scored = any(r.quality for r in rows)
    qcol = (lambda r: f" {'; '.join(q.text() for q in r.quality)} |") if scored else (lambda r: "")  # noqa: E731
    lines += ["| Model | Modality | Starts | Correct | Errors, base / head | c=1 latency, base to head "
              "| c=max throughput, base to head |" + (" Quality, base to head |" if scored else "") + " Verdict |",
              "|---|---|---|---|---|---|---|" + ("---|" if scored else "") + "---|"]
    for r in perf_rows:
        corr = {"pass": "yes", "FAIL": f"**no** ({r.correct_detail})",
                "changed": f"changed ({r.correct_detail})"}.get(r.correct, r.correct)
        if r.silent and r.correct != "FAIL":
            corr = f"{corr} ({r.correct_detail})"
        lines.append(f"| {r.model} | {r.modality} | {starts_cell(r)} | {corr} | {errors_cell(r)} | "
                     f"{fmt_perf(r.latency)} | {fmt_perf(r.throughput)} |{qcol(r)} "
                     f"{r.verdict()}{' (2-session rerun)' if r.rerun else ''} |")
    for r in rp_rows:
        corr = r.correct if r.correct != "pass" else "yes"
        verdict = {"FAIL": "**changed**", "pass": "same"}.get(r.correct, r.note or r.correct)
        lines.append(f"| {r.model} | {r.modality} | {starts_cell(r)} | {corr} ({r.correct_detail}) | | | |"
                     f"{qcol(r)} {verdict} |")
    if chart_url:
        lines += ["", "![Per model and modality, head's c=1 latency and c=max throughput relative to base on one "
                  f"{gpu}; positive means head is better, the grey band marks {pct}%, error bars are the noise margin.]"
                  f"({chart_url})"]
    lines += ["", "<details>", "<summary>Per-row numbers (mean +- std over passes), correctness detail, extra metrics"
              "</summary>", "", "| Model / case | Correctness | c=1 latency | c=max throughput | Extra (c=1) |",
              "|---|---|---|---|---|"]
    for r in perf_rows:
        extra = "; ".join(f"{p.metric} {fmt_perf(p)}" for p in r.extra if p.base.mean is not None)
        detail = r.correct_detail + (f"; {r.rerun}" if r.rerun else "") + "".join(
            f"; {q.text()}, {q.verdict}" + (f", {q.note}" if q.note and q.base is not None else "")
            for q in r.quality)
        lines.append(f"| {r.model} / {r.case} | {detail} | {fmt_perf(r.latency, True)} | "
                     f"{fmt_perf(r.throughput, True)} | {extra} |")
    lines += ["", "</details>", "", "<details>", "<summary>Startup, peak VRAM, errors</summary>", "",
              "| Model | Startup s, base / head | Peak VRAM GB, base / head | Request errors, base / head |",
              "|---|---|---|---|"]

    def pair(d: dict) -> str:
        return " / ".join("n/a" if d.get(a) is None else f"{d[a]:.0f}" for a in ("base", "head"))

    for name in dict.fromkeys(r.model for r in rows):
        mine = [x for x in rows if x.model == name]
        eb, eh = sum(x.errors.get("base", 0) for x in mine), sum(x.errors.get("head", 0) for x in mine)
        lines.append(f"| {name} | {pair(mine[0].startup_s)} | {pair(mine[0].peak_vram_gb)} | {eb} / {eh} |")
    lines += ["", "Startup shares warm compile caches between the arms, so the first arm can pay compiles the second "
              "reuses; it is reported, not judged.", "", "</details>", ""]
    rp_with_diffs = [r for r in rows if r.request_path and r.request_path.get("diffs")]
    if rp_with_diffs:
        lines += ["<details>", "<summary>Request-path differences</summary>", "", "| Model | Case | base | head |",
                  "|---|---|---|---|"]
        for r in rp_with_diffs:
            lines += [f"| {r.model} | {d['case']} | {d['base']} | {d['head']} |" for d in r.request_path["diffs"]]
        lines += ["", "</details>", ""]
    lines += ["<details>", "<summary>Setup</summary>", "",
              f"- base: {base['label']} (`{base['sha'][:9]}`), head: {head['label']} (`{head['sha'][:9]}`)",
              f"- GPU: {gpu}; one model per GPU at a time, {hosts_text(run)}; each arm imports its own checkout "
              "(and its own `rust/` build" + (f", except {', '.join(no_rust)}: see above" if no_rust else "")
              + ") in one Python environment",
              f"- Command: `{run.get('command', 'python -m benchmark.eval compare')}`",
              "- Checks: greedy text identity (or close similarity with no truncation) at c=1, Whisper-large-v3 WER "
              "for speech, PSNR against base for seeded images and video, max action difference for VLA, duplex "
              "take-turn and yield rates, status codes for the request path"
              + ("; quality per arm on its own c=1 outputs where a case lists it, scored after the servers stop: "
                 "UTMOS22 for speech, WavLM speaker similarity to the reference clip for voice clones, CLIP score "
                 "against the prompt for images and video. Head fails when it scores below base by more than "
                 f"{CORRECT_FACTOR} times the row's A/A swing, at least "
                 + ", ".join(f"{v} ({QUALITY_NAME[k]})" for k, v in QUALITY.items())
                 + "; a greedy-text change with every quality score holding reads \"changed, quality same\" and "
                 "is not flagged (a PSNR drop always is)" if scored else ""),
              f"- Regression: worse than {pct}%, more than {sigma} combined standard deviations of the passes, and "
              f"more than {FLOOR_FACTOR} times the largest drift the same row showed between two processes of "
              f"identical code in its last {FLOOR_WINDOW} A/A runs, a lone outlier run left out (A/A calibration); "
              "better by the same margin is reported as faster. Where base and "
              "head returned different output lengths, c=1 latency uses only the requests whose lengths matched "
              "(else latency per token or audio second) and throughput counts output tokens or audio seconds per "
              "second",
              f"- Correctness limits: the manifest's, widened to {CORRECT_FACTOR} times the largest swing the row "
              "showed between two processes of identical code; never tighter than the manifest"
              + ("; a perf row flagged with one session is rerun with two (ABBA) before it is reported"
                 if run.get("rerun_flagged", True) and run.get("sessions", 1) == 1 else ""),
              f"- GPU time: {run.get('gpu_hours', '?')} GPU-hours", "", "</details>"]
    text = "\n".join(lines) + "\n"
    for d in DASHES:
        text = text.replace(d, "-")
    return text


def chart(rows: list[Row], run: dict, path: Path) -> Path | None:
    perf_rows = [r for r in rows if (r.latency and r.latency.delta_pct is not None)
                 or (r.throughput and r.throughput.delta_pct is not None)]
    if not perf_rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band = float(run.get("perf_pct", 5))
    names = [f"{r.model} / {r.modality}" + ("  [correctness changed]" if r.correct == "FAIL" else "")
             + ("  [quality worse]" if r.quality_worse else "") for r in perf_rows]
    scored = any(q.base is not None for r in perf_rows for q in r.quality)
    fig, axes = plt.subplots(1, 3 if scored else 2, figsize=(19 if scored else 13, 0.42 * len(perf_rows) + 1.9),
                             sharey=True)
    ys = list(range(len(perf_rows)))[::-1]
    if scored:
        quality_panel(axes[2], ys, perf_rows)
    for ax, key, title in ((axes[0], "latency", "c=1 latency"), (axes[1], "throughput", "c=max throughput")):
        vals, abs_vals, labels = [], [], []
        for y, r in zip(ys, perf_rows, strict=True):
            p: Perf | None = getattr(r, key)
            if not p or p.delta_pct is None:
                ax.text(0, y, "n/a", va="center", ha="center", fontsize=8, color="#888")
                continue
            v, err = p.delta_pct, p.noise_pct or 0
            vals.append(abs(v) + err)
            abs_vals.append(abs(v))
            color = {"regression": "#c92a2a", "better": "#2b8a3e"}.get(p.verdict, "#5c7cfa")
            ax.barh(y, v, xerr=err, color=color, alpha=0.85, error_kw={"elinewidth": 0.8, "capsize": 2})
            labels.append((y, f"{v:+.1f}%", color))
        ax.axvspan(-band, band, color="#adb5bd", alpha=0.25, lw=0)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"{title}: head vs base (positive = head better)", fontsize=10)
        ax.set_xlabel("% change")
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=8.5)
        # Axis from the bars, not the error bars: one noisy metric (+-150%) would flatten every other row.
        lim = max(band * 3, min(max(vals, default=0), max(abs_vals, default=0) * 2 + band) + 6)
        ax.set_xlim(-lim, lim)
        for y, text, color in labels:
            ax.text(lim * 0.97, y, text, va="center", ha="right", fontsize=8,
                    color=color if color != "#5c7cfa" else "#212529",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"{run['head']['label']} against {run['base']['label']}, served end to end, one "
                 f"{run.get('gpu', '?').upper()} per model; grey band = within {band:.0f}%", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def quality_panel(ax, ys: list[int], rows: list[Row]) -> None:
    """Head minus base per quality score, in units of the row's limit: below -1 fails."""
    kinds = [k for k in QUALITY if any(q.kind == k and q.base is not None for r in rows for q in r.quality)]
    h = 0.8 / max(len(kinds), 1)
    colors = dict(zip(QUALITY, ("#5c7cfa", "#f08c00", "#0c8599"), strict=True))
    for y, r in zip(ys, rows, strict=True):
        for q in r.quality:
            if q.base is None or q.kind not in kinds:
                continue
            off = (kinds.index(q.kind) - (len(kinds) - 1) / 2) * h
            v = (q.head - q.base) / q.limit
            ax.barh(y + off, v, height=h, color="#c92a2a" if q.verdict == "worse" else colors[q.kind], alpha=0.85)
            ax.text(1.95, y + off, q.delta(), va="center", ha="right", fontsize=7)
    from matplotlib.patches import Patch

    ax.axvspan(-1, 1, color="#adb5bd", alpha=0.25, lw=0)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-2, 2)
    ax.set_title("quality: head - base, in units of the row's limit (below -1 fails)", fontsize=10)
    ax.set_xlabel("score change / limit (clipped at 2)")
    ax.legend(handles=[Patch(color=colors[k], label=QUALITY_NAME[k]) for k in kinds] + (
        [Patch(color="#c92a2a", label="worse")] if any(r.quality_worse for r in rows) else []),
        fontsize=7, loc="best")
    ax.grid(axis="x", alpha=0.3)


def summary_json(rows: list[Row]) -> list[dict[str, Any]]:
    def p(x: Perf | None):
        if not x:
            return None
        return {"metric": x.metric, "conc": x.conc, "base": x.base.__dict__, "head": x.head.__dict__,
                "delta_pct": x.delta_pct, "noise_pct": x.noise_pct, "verdict": x.verdict, "basis": x.basis}

    return [{"model": r.model, "case": r.case, "modality": r.modality, "starts": r.starts, "correct": r.correct,
             "correct_detail": r.correct_detail, "latency": p(r.latency), "throughput": p(r.throughput),
             "extra": [p(x) for x in r.extra], "startup_s": r.startup_s, "peak_vram_gb": r.peak_vram_gb,
             "errors": r.errors, "flagged": r.flagged, "verdict": r.verdict(), "note": r.note,
             "no_baseline": r.no_baseline, "rerun": r.rerun, "silent": r.silent, "rust": r.rust,
             "quality": [q.__dict__ for q in r.quality]}
            for r in rows]


def write(run_dir: Path, models: list[Model], chart_url: str | None = None, calibration: dict | None = None,
          with_chart: bool = True) -> tuple[Path, list[Row]]:
    """report/report.md, report/summary.json and (with matplotlib) report/chart.png; returns the markdown path."""
    run = json.loads((run_dir / "run.json").read_text())
    rows = build_rows(run_dir, models, calibration)
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    png = None
    if with_chart:
        try:
            png = chart(rows, run, out / "chart.png")
        except ImportError:
            png = None
    md = render_md(rows, run, chart_url or (png.name if png else None))
    (out / "report.md").write_text(md)
    (out / "summary.json").write_text(json.dumps(summary_json(rows), indent=1, default=str))
    return out / "report.md", rows
