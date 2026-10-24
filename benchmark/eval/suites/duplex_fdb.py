"""Full-Duplex-Bench v1.5 on a frame-synchronous duplex model over streamed ``/generate``.

Opt-in: the benchmark's code and data are CC BY-NC 4.0 (non-commercial), so both are
downloaded at run time from the official sources (``data fdb-v15``) and never vendored;
see ``benchmark/eval/fdb_reference.py``. Metric definitions follow the reference runner
(``benchmarks/duplex/fdb_v15``, pinned there) so the numbers are comparable:

* Each item is a pair: ``input.wav`` (the user's query plus a later overlapping event:
  an interruption, a backchannel, speech to someone else, or background speech) and
  ``clean_input.wav`` (the query alone). Each is its own session.
* Observation window: the model's output over ``[0, T]``, T the input length, as 16 kHz
  mono with exactly the input's sample count. A frame-synchronous model emits one output
  frame per input frame, so output sample i plays at input time i / rate: this is the
  audio a client hears with no transport delay when the server keeps up live
  (``on_time_rate`` says whether it did; ``wall_ms_per_frame`` bounds the difference).
* Stop and response latency: the official ``get_timing.py`` on Silero VAD spans (user
  spans merged across 0.6 s gaps, model spans across 0.5 s). Stop intervals are overlaps
  of user and model speech, response intervals run from a user span end to the next
  later model start. Whole-file intervals of the overlap session, pooled per category:
  the mean over every interval of every timed session (null when there are none).
  Lower stop latency is better on ``user_interruption``; on the other three the model
  should keep its turn, so the overlap mostly tracks how long the other speech lasts.
* Behavior labels: Parakeet TDT 0.6B v2 word-timestamp transcripts of the four files,
  then the official prompt, one label per pair per judge: ``C_RESPOND`` (addressed the
  overlap), ``C_RESUME`` (ignored it and continued), ``C_UNCERTAIN_HANDLING`` (asked for
  a repeat), ``C_UNKNOWN`` (off-target or silent). Shares are over valid labels only;
  ``judged_pairs`` is how many there were.
* Semantic quality (non-official, one local judge): ``quality = A/(A+F)`` and
  ``coverage = (A+F)/N`` per axis and for ``joint``, N every selected pair.
* Judge agreement: with two or more judges, raw label agreement and Cohen's kappa over
  the pairs both judges labeled validly, per repeat; ``agreement`` renders the pooled
  confusion matrix.

Failed sessions stay in every denominator. A pair whose session failed is not timed or
judged and counts as such.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

from benchmark.eval import stats
from benchmark.eval.audio import pcm16_to_float, read_wav, resample, write_wav
from benchmark.eval.datasets import Item
from benchmark.eval.fdb_reference import CATEGORIES, RATE, provenance
from benchmark.eval.llm_judge import ACCEPTABLE_OUTCOMES, AXES, C_LABELS, SEMANTIC_BATCH, SEMANTIC_DECODING
from benchmark.eval.suites.base import Suite, error_row
from benchmark.eval.suites.duplex import SessionError, _special, generate_session

GROUPS = ("all", *CATEGORIES)
VARIANTS = (("overlap", "input", "output"), ("clean", "clean_input", "clean_output"))
SEMANTIC_AXES = (*AXES, "joint")
# Model speech ending this close to T counts as still playing at the cutoff (the reference runner's EOF tolerance).
EOF_TOLERANCE_S = 0.05
# Output after T below this peak (-40 dBFS) is the codec's idle floor, not more reply.
SILENCE_PEAK = 0.01


# --- metric definitions (pure, unit tested) ---------------------------------------------------


def pool_intervals(sessions: list[list[list[float]]]) -> dict[str, float | None]:
    """Pooled mean and median of interval lengths (end - start) over every interval of every
    session; ``None`` when there are no intervals."""
    lengths = [e - s for intervals in sessions for s, e in intervals]
    return {"mean": stats.mean(lengths) if lengths else None,
            "median": stats.percentile(lengths, 50), "intervals": len(lengths), "sessions": len(sessions)}


def label_shares(labels: list[str | None]) -> dict[str, Any]:
    """Share of each ``C_*`` label over the valid labels (``None`` entries are not valid)."""
    valid = [x for x in labels if x is not None]
    return {"judged": len(valid), **{lab: (valid.count(lab) / len(valid) if valid else None) for lab in C_LABELS}}


def afu(statuses: list[str]) -> dict[str, float | None]:
    """``quality = A/(A+F)`` and ``coverage = (A+F)/N`` over N statuses (every selected pair)."""
    a, f, n = statuses.count("accept"), statuses.count("fail"), len(statuses)
    return {"quality": a / (a + f) if a + f else None, "coverage": (a + f) / n if n else None}


def valid_label_pairs(rows: list[dict[str, Any]], a: str, b: str) -> tuple[list[str], list[str]]:
    """Labels of judges ``a`` and ``b`` on the pairs both labeled validly."""
    xs, ys = [], []
    for r in rows:
        la, lb = ((r.get("labels") or {}).get(j, {}).get("label") for j in (a, b))
        if la is not None and lb is not None:
            xs.append(la)
            ys.append(lb)
    return xs, ys


def fdb_metrics(rows: list[dict[str, Any]], judges: list[str], semantic: bool) -> dict[str, float | None]:
    """One repeat's metrics, per category and for ``all``."""
    out: dict[str, float | None] = {}
    for group in GROUPS:
        members = rows if group == "all" else [r for r in rows if r.get("category") == group]
        if not members:
            continue
        timed = [r for r in members if (r.get("overlap") or {}).get("timing")]
        out[f"{group}.pairs"] = len(members)
        out[f"{group}.timed_overlap_sessions"] = len(timed)
        for kind in ("stop", "response"):
            pooled = pool_intervals([r["overlap"]["timing"][f"{kind}_intervals"] for r in timed])
            out[f"{group}.{kind}_latency_s"] = pooled["mean"]
            out[f"{group}.{kind}_latency_s_median"] = pooled["median"]
        for j in judges:
            shares = label_shares([(r.get("labels") or {}).get(j, {}).get("label") for r in members])
            out[f"{group}.{j}.judged_pairs"] = shares.pop("judged")
            out |= {f"{group}.{j}.{lab}": v for lab, v in shares.items()}
        if semantic:
            for axis in SEMANTIC_AXES:
                res = afu([(r.get("semantic") or {}).get(axis, "unresolved") for r in members])
                out[f"{group}.semantic.{axis}_quality"] = res["quality"]
                out[f"{group}.semantic.{axis}_coverage"] = res["coverage"]
    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            ag = stats.label_agreement(*valid_label_pairs(rows, a, b))
            out[f"agreement.{a}.{b}.n"] = ag["n"]
            out[f"agreement.{a}.{b}.agreement"] = ag["agreement"]
            out[f"agreement.{a}.{b}.kappa"] = ag["kappa"]
    return out


def window(pcm: bytes, rate: int, n: int) -> tuple[np.ndarray, bool]:
    """The output over ``[0, T]`` at 16 kHz with exactly ``n`` samples (cut or padded with
    silence), and whether the model produced audible output after T. The last frame usually
    runs past T (the input is not a whole number of frames); its silence does not count."""
    y = resample(pcm16_to_float(pcm), rate, RATE)
    after = len(y) > n and float(np.abs(y[n:]).max()) > SILENCE_PEAK
    y = y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y), dtype=np.float32)])
    return y, after


def native_text(tokens: list[str], frame_s: float, t_s: float) -> str:
    """Agent text of the frames that start before T (the server's text, auxiliary evidence)."""
    return "".join(t for i, t in enumerate(tokens) if i * frame_s < t_s and not _special(t)).strip()


def semantic_packet(row: dict[str, Any], opaque_id: str) -> dict[str, Any]:
    """One pair as the semantic judge sees it, in the reference judge's field order (model identity withheld)."""
    asr, ov = row["asr"], row["overlap"]
    return {
        "sample_id": opaque_id,
        "category": row["category"],
        "original_task": row["context_text"],
        "event_text": row["event_text"],
        "event_s": row["event_s"],
        "window_s": [0, ov["T_s"]],
        "input_asr_text": asr["input"]["text"],
        "clean_input_asr_text": asr["clean_input"]["text"],
        "output_asr_text": asr["output"]["text"],
        "output_asr_segments": asr["output"]["chunks"],
        "native_output_text": ov["native_text"],
        "clean_output_asr_text": asr["clean_output"]["text"],
        "boundary": ov["boundary"],
        "available_context_and_tool_results": [],
        "acceptable_outcomes": ACCEPTABLE_OUTCOMES,
    }


# --- judging (shared by the suite and `rejudge`) ----------------------------------------------

# Row fields a judge reads; a stored row whose copy of them was redacted is not judged again.
JUDGE_FIELDS = {"asr", "overlap", "context_text", "event_text"}


def assessable(row: dict[str, Any]) -> bool:
    return bool(row.get("ok") and row.get("asr")) and not JUDGE_FIELDS & set(row.get("scrubbed") or [])



class Judging:
    """The LLM judges of a run, loaded once and reused across repeats."""

    def __init__(self, judges: list, semantic_name: str | None, concurrency: int = 8, behavior: bool = True):
        from benchmark.eval import llm_judge

        self.judges = [llm_judge.resolve(j) for j in judges]
        self.available = [j for j in self.judges if j.available]
        for j in self.judges:
            if not j.available:
                print(f"[fdb] judge {j.name!r} skipped: {j.skipped}", file=sys.stderr)
        names = [j.name for j in self.judges]
        if semantic_name not in (None, "none") and semantic_name not in names:
            raise SystemExit(f"--semantic-judge {semantic_name!r} is not one of --judge {names}")
        if semantic_name is None:  # default: the first local judge, as the reference runner runs it on Qwen only
            semantic_name = next((j.name for j in self.available if j.kind == "local"), "none")
        self.semantic = next((j for j in self.available if j.name == semantic_name), None)
        self.concurrency = concurrency
        self.behavior = behavior  # False: the judges only regrade semantic quality
        self._behavior = self._assets = self._controls = None

    def identity(self) -> dict[str, Any]:
        sem = None
        if self.semantic is not None:
            sem = {"judge": self.semantic.name, "identity": self.semantic.identity(), "decoding": SEMANTIC_DECODING,
                   "batch_pairs": SEMANTIC_BATCH, "enable_thinking": self.semantic.kind == "local",
                   "controls": self._controls}
        return {"judges": [j.identity() for j in self.judges], "semantic": sem}

    def usage(self) -> dict[str, dict[str, int]]:
        """Requests, transport retries and prompt and completion token counts, per judge."""
        return {j.name: dict(j.usage) for j in self.available}

    def label(self, rows: list[dict[str, Any]], retry_failed: bool = False) -> None:
        """Behavior labels from every available judge, and semantic outcomes, into ``rows``.
        ``retry_failed`` relabels only the pairs whose label from that judge failed."""
        from benchmark.eval import fdb_reference, llm_judge

        if not self.available:
            return
        if self._behavior is None:
            self._behavior = fdb_reference.load_behavior()
        for j in self.available if self.behavior else []:
            todo = [not retry_failed or (r.get("labels") or {}).get(j.name, {}).get("status") in (None, "failed")
                    for r in rows]
            transcripts = [r.get("asr") if t and assessable(r) else None for r, t in zip(rows, todo, strict=True)]
            labels = llm_judge.label_pairs(j, self._behavior, transcripts, self.concurrency)
            for r, t, res in zip(rows, todo, labels, strict=True):
                if t:
                    r.setdefault("labels", {})[j.name] = res
        if self.semantic is None:
            return
        if self._assets is None:
            self._assets = llm_judge.SemanticAssets.load()
            self._controls = llm_judge.run_controls(self.semantic, self._assets)
            c = self._controls
            print(f"[fdb] semantic controls: {c['total'] - len(c['mismatches'])}/{c['total']} axis statuses match",
                  file=sys.stderr)
            for m in c["mismatches"]:
                print(f"  {m}", file=sys.stderr)
        if not self._controls["passed"]:
            for r in rows:
                r["semantic"] = {ax: "unresolved" for ax in SEMANTIC_AXES} | {"conversions": "controls failed"}
            return
        order = sorted(range(len(rows)), key=lambda i: rows[i].get("sample_id") or rows[i]["id"])
        opaque = {i: f"p{k:04d}" for k, i in enumerate(order, start=1)}
        packets = [semantic_packet(rows[i], opaque[i]) for i in order if assessable(rows[i])]
        graded = llm_judge.grade(self.semantic, self._assets, packets, self.concurrency)
        for i, r in enumerate(rows):
            axes = graded.get(opaque[i]) or {ax: {"status": "unresolved", "conversion": "pair not assessable"}
                                             for ax in AXES}
            statuses = {ax: axes[ax]["status"] for ax in AXES}
            r["semantic"] = statuses | {"joint": llm_judge.joint_status(statuses),
                                        "conversions": {ax: axes[ax]["conversion"] for ax in AXES
                                                        if axes[ax].get("conversion")}}


def _derived(metric: str) -> bool:
    """A metric ``fdb_metrics`` computes (per group, or agreement)."""
    return metric.split(".", 1)[0] in (*GROUPS, "agreement") and "." in metric


def rejudge(result, judging: Judging, retry_failed: bool = False):
    """Label a stored ``fdb`` results file with more judges (or again), from the transcripts
    its rows hold: no server, no GPU. Judges of the same name are replaced, or with
    ``retry_failed`` keep their labels and relabel only the pairs that failed."""
    if result.suite != "fdb":
        raise SystemExit(f"rejudge reads fdb results, not {result.suite!r}")
    suite_cfg = result.config.setdefault("suite", {})
    old = suite_cfg.get("judges") or []
    new = {j["name"]: j for j in judging.identity()["judges"]} if judging.behavior else {}
    # A judge skipped now (no key) keeps the labels and identity it stored; only a judge that runs replaces them.
    for name in [n for n, j in new.items() if j.get("skipped") and n in {o["name"] for o in old}]:
        print(f"[fdb] judge {name!r} skipped; its stored labels are kept", file=sys.stderr)
        del new[name]
    if retry_failed:
        from benchmark.eval.schema import scrub

        same = {o["name"]: o for o in old}
        for name, ident in new.items():
            # Stored identities were scrubbed when saved (a remote base_url is redacted), so compare like with like.
            stored = {k: v for k, v in (same.get(name) or {}).items() if k != "skipped"}
            if not stored or stored != {k: v for k, v in scrub(ident).items() if k != "skipped"}:
                raise SystemExit(f"--retry-failed: judge {name!r} is not the judge that labeled this file")
    # A replaced judge keeps its place, so agreement keys keep their order across rejudges.
    suite_cfg["judges"] = [new.get(j["name"], j) for j in old] + [j for n, j in new.items()
                                                                  if n not in {o["name"] for o in old}]
    sem_cfg = suite_cfg.get("semantic")
    if sem_cfg and judging.semantic is None and sem_cfg.get("judge") in new:
        print(f"[fdb] semantic grades by the replaced judge {sem_cfg['judge']!r} are dropped; pass "
              f"--semantic-judge {sem_cfg['judge']} to regrade them", file=sys.stderr)
        suite_cfg["semantic"] = sem_cfg = None
    regrade = judging.semantic is not None or sem_cfg is None
    names = [j["name"] for j in suite_cfg["judges"] if not j.get("skipped")]
    before = judging.usage()
    for rep in result.repeats:
        for r in rep.requests:
            for name in new if not retry_failed else ():
                (r.get("labels") or {}).pop(name, None)
            if regrade:
                r.pop("semantic", None)
        judging.label(rep.requests, retry_failed)
        # Every derived metric is recomputed from the rows; none of the old ones survive.
        kept = {k: v for k, v in rep.metrics.items() if not _derived(k)}
        rep.metrics = kept | fdb_metrics(rep.requests, names, judging.semantic is not None or bool(sem_cfg))
    if judging.semantic is not None:
        suite_cfg["semantic"] = judging.identity()["semantic"]
    result.config.setdefault("rejudged", []).append(
        {"judges": sorted(new), "semantic": judging.semantic.name if judging.semantic else None,
         "semantic_only": not judging.behavior, "retry_failed": retry_failed,
         "usage": {n: {k: v - before[n][k] for k, v in u.items()} for n, u in judging.usage().items()}})
    result.summary = {}
    return result.finalize()


# --- the suite --------------------------------------------------------------------------------


class FDBSuite(Suite):
    name = "fdb"
    help = ("Full-Duplex-Bench v1.5 (CC BY-NC 4.0, fetched at run time): stop and response latency, "
            "LLM-judged behavior labels and semantic quality, judge agreement")
    requires = ("audio_path",)

    @staticmethod
    def add_args(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("fdb")
        g.add_argument("--temperature", type=float, default=None,
                       help="sampling temperature sent to the model (default: the model's own sampling, so "
                            "repeats measure generation variance)")
        g.add_argument("--frame-ms", type=float, default=80.0, help="model frame duration")
        g.add_argument("--timeout", type=float, default=600.0, help="per-session timeout, seconds")
        g.add_argument("--asr", choices=("parakeet", "none"), default="parakeet",
                       help="transcripts for the judges (none: timing only)")
        g.add_argument("--judge", action="append", default=[], metavar="NAME=openai|URL",
                       help="an LLM judge: NAME=openai (gpt-4o-2024-08-06, needs OPENAI_API_KEY; skipped "
                            "without it) or NAME=URL of an OpenAI-compatible endpoint (local, e.g. a "
                            "server for Qwen/Qwen3.8-27B). Repeat for agreement")
        g.add_argument("--judge-revision", action="append", default=[], metavar="NAME=REV",
                       help="checkpoint revision of a local judge, for the results file")
        g.add_argument("--semantic-judge", default=None,
                       help="judge for the semantic A/F/U grade (default: the first local judge; 'none' skips)")
        g.add_argument("--judge-concurrency", type=int, default=8)
        g.add_argument("--judge-device", default=None, help="device for Parakeet ASR (default: cuda if available)")

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        from benchmark.eval.llm_judge import parse_judges

        try:
            judges = parse_judges(args.judge, args.judge_revision)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        if judges and args.asr == "none":
            raise SystemExit("--judge needs transcripts; drop --asr none")
        self.judging = Judging(judges, args.semantic_judge, args.judge_concurrency)
        self._timing = self._asr = None
        self._dataset_revision: set[str] = set()

    def check_items(self, items: list[Item]) -> None:
        super().check_items(items)
        for it in items:
            if it.extra.get("category") not in CATEGORIES:
                raise ValueError(f"fdb: item {it.id!r} has category {it.extra.get('category')!r}")
            if not self._clean_path(it).exists():
                raise ValueError(f"fdb: item {it.id!r} lacks its clean input")
            self._dataset_revision.add(it.extra.get("dataset_revision"))

    @staticmethod
    def _clean_path(item: Item) -> Path:
        return Path(item.audio_path).parent / item.extra.get("clean_audio", "clean_input.wav")

    def config(self) -> dict[str, Any]:
        a = self.args
        return {
            "temperature": a.temperature, "frame_ms": a.frame_ms, "window": "[0, T] of the frame-synchronous output",
            "dataset_revision": sorted(r for r in self._dataset_revision if r) or None,
            "reference": provenance(),
            "timing": self._timing.record if self._timing else None,
            "asr": self._asr.id if self._asr else a.asr,
            **self.judging.identity(),
            "judge_usage": self.judging.usage(),
        }

    async def request(
        self, session: aiohttp.ClientSession, url: str, index: int, item: Item, t0: float
    ) -> dict[str, Any]:
        a = self.args
        kwargs = {} if a.temperature is None else {"temperature": a.temperature}
        base = {"category": item.extra["category"], "sample_id": item.extra.get("sample_id")}
        sessions, failed = {}, None
        for variant, path in (("overlap", item.audio_path), ("clean", str(self._clean_path(item)))):
            try:
                sessions[variant] = await generate_session(session, url, path, kwargs, a.timeout,
                                                           time.perf_counter())
            except SessionError as exc:
                failed = error_row(item, f"{variant}: {exc.error}", exc.detail, **base, **exc.extra)
            except Exception as exc:  # noqa: BLE001 - every failure is an error row
                failed = error_row(item, f"{variant}: {type(exc).__name__}", f"{type(exc).__name__}: {exc}", **base)
            if failed is not None:
                break
        frame_s = a.frame_ms / 1000
        # A pair whose clean session failed is an error and is not judged, but its overlap session is
        # still timed: the reference times each session on its own.
        row = failed or {"id": item.id, "ok": True, "error": None, **base}
        if not sessions:
            return row
        row |= {"e2e_s": time.perf_counter() - t0, "context_text": item.extra.get("context_text"),
                "event_text": item.extra.get("event_text"), "event_s": item.extra.get("event_s"),
                "_sessions": sessions, "_item": item}
        for variant, s in sessions.items():
            row[variant] = {"frames": s["frames"], "wall_ms_per_frame": 1000 * s["e2e_s"] / s["frames"],
                            "on_time": s["e2e_s"] <= s["frames"] * frame_s, "sample_rate": s["rate"]}
        return row

    def _scorers(self) -> None:
        from benchmark.eval import fdb_reference

        if self._timing is None:
            self._timing = fdb_reference.load_timing()
        if self._asr is None and self.args.asr == "parakeet":
            device = self.args.judge_device
            if device is None:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._asr = fdb_reference.ParakeetASR(device)

    def score(self, items: list[Item], rows: list[dict[str, Any]], wall_s: float, audio_dir: Path | None) -> dict:
        from benchmark.eval import fdb_reference

        self._scorers()
        work = audio_dir or Path(tempfile.mkdtemp(prefix="mstar-fdb-"))
        frame_s = self.args.frame_ms / 1000
        try:
            for r in rows:
                if "_sessions" in r:
                    self._score_pair(r, work / r["id"], frame_s, fdb_reference)
            self.judging.label(rows)
        finally:
            if audio_dir is None:
                shutil.rmtree(work, ignore_errors=True)
        ok = [r for r in rows if r["ok"]]
        frames = [r[v]["wall_ms_per_frame"] for r in ok for v in ("overlap", "clean")]
        out = {
            "requests": len(rows),
            "errors": len(rows) - len(ok),
            **fdb_metrics(rows, [j.name for j in self.judging.available], self.judging.semantic is not None),
            "timing_failed": sum(bool((r.get("overlap") or {}).get("timing_error")) for r in rows),
            "asr_failed": sum(bool(r.get("asr_error")) for r in ok),
            "wall_ms_per_frame_p50": stats.percentile(frames, 50),
            "wall_ms_per_frame_max": max(frames, default=None),
            "on_time_rate": stats.mean(float(r[v]["on_time"]) for r in ok for v in ("overlap", "clean")),
            "frames_per_s": sum(r[v]["frames"] for r in ok for v in ("overlap", "clean")) / wall_s if wall_s else None,
        }
        return out

    def _score_pair(self, r: dict[str, Any], d: Path, frame_s: float, ref) -> None:
        """Window, timing and transcripts of one pair; frees its raw session audio."""
        sessions, item = r.pop("_sessions"), r.pop("_item")
        d.mkdir(parents=True, exist_ok=True)
        inputs = {"overlap": Path(item.audio_path), "clean": self._clean_path(item)}
        for variant, in_name, out_name in VARIANTS:
            if variant not in sessions:
                continue
            x, rate = read_wav(inputs[variant])
            if rate != RATE:
                raise ValueError(f"{inputs[variant]}: expected {RATE} Hz")
            src = d / f"{in_name}.wav"
            src.unlink(missing_ok=True)
            os.symlink(inputs[variant].resolve(), src)
            y, after = window(sessions[variant]["pcm"], sessions[variant]["rate"], len(x))
            write_wav(d / f"{out_name}.wav", y, RATE)
            t_s = len(x) / RATE
            v = r[variant]
            v |= {"T_s": t_s, "native_text": native_text(sessions[variant]["texts"], frame_s, t_s)}
            try:
                v["timing"] = ref.run_timing(self._timing, src, d / f"{out_name}.wav", d / "timing" / variant)
            except Exception as exc:  # noqa: BLE001 - a session that cannot be timed counts as untimed
                v["timing_error"] = type(exc).__name__
                print(f"[fdb] {r['id']} {variant}: timing failed: {exc}", file=sys.stderr)
            segs = (v.get("timing") or {}).get("model_segments") or []
            v["boundary"] = {"playout_active_at_T": bool(segs) and segs[-1][1] >= t_s - EOF_TOLERANCE_S,
                             "audio_received_after_T": after}
        if self._asr is not None and len(sessions) == len(VARIANTS):
            try:
                r["asr"] = {k: self._asr.transcribe(d / f"{k}.wav", d / ".asr-stage")
                            for k in ("input", "clean_input", "output", "clean_output")}
            except Exception as exc:  # noqa: BLE001 - an untranscribed pair is not judged
                r["asr_error"] = type(exc).__name__
                print(f"[fdb] {r['id']}: ASR failed: {exc}", file=sys.stderr)
