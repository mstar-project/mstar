"""A/B of a host-overhead change on the ASR models, on one box (the benchmarking skill's patch method).

The two arms are the same checkout with ``change.patch`` (``git diff`` of ``mstar/``, written next to this
file before the box is launched) reversed (``base``) or applied (``change``). Both share the venv and the
configs. Every server is fresh, arms alternate within a round (A B, then B A), and each model's two arms run
back to back. Cells are the 318 harness's (``benchmark/asr318_run/run.py``): LibriSpeech test-clean
utterances, closed loop, warmed with 5 waves at c=32 then one wave per point, RTFx = audio s / wall s.

Phases:
  check      tree state, patch applies both ways, imports, GPUs
  ab         rounds of timed cells (``--rounds``, ``--models``, ``--concs``), results/ab.jsonl
  profile    MSTAR_PHASE_TIMING=100 at c=32 per arm and model, untimed; server logs + phase tables
  summarize  summary.md: RTFx mean +- sd per (model, c, arm) and the delta; key phase means per arm

  python benchmark/asr_perf/ab.py ab --out ~/results/asr_perf --models turbo,qasr --rounds 3
"""
from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PATCH = HERE / "change.patch"

_spec = importlib.util.spec_from_file_location("asr318", REPO / "benchmark" / "asr318_run" / "run.py")
H = importlib.util.module_from_spec(_spec)
sys.modules["asr318"] = H
_spec.loader.exec_module(H)

KEY_PHASES = ("iter_total", "gpu_exec", "worker.postprocess.event_sync", "submit_spec.launch_wait",
              "engine.plan.fresh", "engine.plan.promoted", "engine.preprocess", "engine.collect_outputs",
              "worker.gpu_thread.prepare_inputs", "worker.postprocess.check_stop",
              "worker.postprocess.prematerialize", "worker.postprocess_batch")


def log(msg: str) -> None:
    print(f"[ab {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- unique audio per request

def make_audio_unique() -> None:
    """Every request's WAV gets its own count of trailing silent samples (1..2048, about 0.13 s at most), so
    no two requests carry the same bytes. A server that caches encoder outputs by audio content (sglang-omni's
    Whisper keeps 1,024 by default) then encodes every request, as it would on real traffic; the words, and
    so the transcripts, are unchanged. Applied to every engine alike."""
    import io
    import itertools
    import wave
    counter = itertools.count(1)
    post = H._post

    def unique(audio: bytes) -> bytes:
        with wave.open(io.BytesIO(audio), "rb") as r:
            params, frames = r.getparams(), r.readframes(r.getnframes())
        pad = (next(counter) % 2048) + 1
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params)
            w.writeframes(frames + b"\x00" * (pad * params.sampwidth * params.nchannels))
        return out.getvalue()

    async def _post(s, url, model, audio, name, fields, *rest, **kw):
        return await post(s, url, model, unique(audio), name, fields, *rest, **kw)

    H._post = _post
    log("unique audio per request: on")


# ----------------------------------------------------------------------------- the tree's arm

def _patch(*flags: str) -> bool:
    r = subprocess.run(["patch", "-p1", "-f", "-s", *flags], cwd=REPO, stdin=PATCH.open(),
                       capture_output=True, text=True, check=False)
    return r.returncode == 0


def arm_now() -> str:
    if PATCH.stat().st_size == 0:
        return "change"  # no patch: the tree as it is; a single-arm (profile) run only
    if _patch("--dry-run", "-R"):
        return "change"
    if _patch("--dry-run"):
        return "base"
    raise RuntimeError("change.patch applies neither way: the tree is in an unknown state")


def set_arm(arm: str) -> None:
    if arm_now() == arm:
        return
    if PATCH.stat().st_size == 0:
        raise RuntimeError(f"change.patch is empty: there is no {arm} arm")
    if not _patch(*(["-R"] if arm == "base" else [])):
        raise RuntimeError(f"switching to {arm} failed")
    for d in (REPO / "mstar").rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run([str(H.MSTAR_BIN / "python"), "-c",
                        "import mstar, mstar.worker.worker, mstar.model.whisper.submodules, "
                        "mstar.model.qwen3_asr.submodules; print(mstar.__file__)"],
                       cwd=REPO, capture_output=True, text=True, check=False)
    if r.returncode or str(REPO) not in r.stdout or arm_now() != arm:
        raise RuntimeError(f"import check after switching to {arm} failed: {r.stdout} {r.stderr[-600:]}")
    log(f"tree -> {arm}")


def restore() -> None:
    try:
        if PATCH.exists() and arm_now() == "base":
            set_arm("change")
    except Exception as e:  # noqa: BLE001
        print(f"restore failed: {e!r}", file=sys.stderr)


# ----------------------------------------------------------------------------- phases

def phase_check(a) -> int:
    log(f"tree: {arm_now()}; patch: {PATCH.stat().st_size} bytes")
    set_arm("base")
    set_arm("change")
    print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False).stdout, flush=True)
    for m in a.models:
        log(f"{m}: {len(H.load('test_clean_200'))} utterances, model key {H.MODELS[m]['key']}")
    # the change's own unit tests, and the models' CPU suites it touches
    r = subprocess.run([str(H.MSTAR_BIN / "python"), "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "test/modular/test_whisper_model.py", "test/modular/test_qwen3_asr_model.py",
                        "test/modular/test_batched_model_output.py", "test/modular/test_qwen3_5_check_stop_batched.py",
                        *sorted(str(x.relative_to(REPO)) for x in (REPO / "test" / "modular").glob("test_kv_*.py")),
                        *sorted(str(x.relative_to(REPO)) for x in (REPO / "test" / "modular").glob("test_cross_*.py"))],
                       cwd=REPO, capture_output=True, text=True, check=False)
    print(r.stdout[-3000:], r.stderr[-1500:], flush=True)
    return r.returncode


def run_server_cells(a, arm: str, m: str, r: int, engine: str = "mstar", spec: "Arm | None" = None) -> list[dict]:
    """One fresh server, every cell. ``arm`` labels the row: an Arm's label for mstar (``spec``), the
    engine's name for a baseline."""
    mps_env = {}
    if engine == "mstar":
        set_arm(spec.tree if spec else arm)
        if spec and spec.mps:
            mps_env = mps(True)
    srv = H.Server(a.out, m, engine, 0, 8100, f"{arm}_r{r}", extra=(spec.args if spec else []),
                   env_extra={**(spec.env if spec else {}), **mps_env})
    rows = []
    with H.serving(srv) as up:
        if not up:
            row = {"arm": arm, "model": m, "round": r, "kind": "start_failed", "ts": time.time()}
            H.append_jsonl(a.out / "results" / "ab.jsonl", row)
            return [row]
        items = H.load("test_clean_200")
        H.run_requests(srv.url, H.req_model(m, engine), H.repeat(items, H.WARM_FIRST_WAVES * H.WARM_CONC),
                       H.WARM_CONC, H.json_fields())
        for c in a.concs:
            H.run_requests(srv.url, H.req_model(m, engine), H.repeat(items, c), c, H.json_fields())
            res = H.cell(srv, m, engine, H.repeat(items, H.REQS[c]), c, H.json_fields())
            row = {k: v for k, v in res.items() if k != "records"}
            row.update({"arm": arm, "model": m, "round": r, "start_s": getattr(srv, "start_s", None),
                        "variant": srv.variant, "ts": time.time()})
            H.append_jsonl(a.out / "results" / "ab.jsonl", row)
            rows.append(row)
            log(f"r{r} {m} {arm} c={c}: RTFx {res['rtfx']:.0f} p50 {res['p50_ms'] or 0:.0f} ms "
                f"WER {100 * (res['wer'] or 0):.2f}% errors {res['errors']}")
    if mps_env:
        mps(False)
    return rows


def phase_ab(a) -> int:
    failures = 0
    arms = parse_arms(a.arms)
    for r in range(1, a.rounds + 1):
        for m in a.models:
            order = arms if r % 2 else list(reversed(arms))
            for spec in order:
                rows = run_server_cells(a, spec.label, m, r, spec=spec)
                failures += sum(1 for x in rows if x.get("kind") == "start_failed" or x.get("errors"))
    restore()
    return 1 if failures else 0


def phase_baselines(a) -> int:
    """vLLM / sglang-omni on the same box and cells, ``--rounds`` fresh servers each (default configs, as
    the 318 run's tuning kept). Rows land in ab.jsonl with the engine as the arm."""
    failures = 0
    for r in range(1, a.rounds + 1):
        for m in a.models:
            for e in H.engines_of(m):
                if e == "mstar":
                    continue
                rows = run_server_cells(a, e, m, r, engine=e)
                failures += sum(1 for x in rows if x.get("kind") == "start_failed" or x.get("errors"))
    return 1 if failures else 0


class Arm(NamedTuple):
    label: str
    tree: str            # base or change (change.patch reversed / applied)
    env: dict            # extra server env: runtime-switch arms on one build
    args: list           # extra `mstar serve` args, e.g. --config configs/x.yaml
    mps: bool            # run this arm's server under an MPS daemon


def parse_arms(spec: str) -> list[Arm]:
    """``label:tree[:ENV=V+ENV2=V2[:ARG+ARG[:mps]]],...``, e.g.
    ``plain:change,kv:change:MSTAR_KV_PLAN_CACHE=1,split:change::--config+configs/x.yaml:mps``."""
    arms = []
    for item in spec.split(","):
        parts = item.split(":") + ["", "", ""]
        env = dict(kv.split("=", 1) for kv in parts[2].split("+")) if parts[2] else {}
        args = parts[3].split("+") if parts[3] else []
        arms.append(Arm(parts[0], parts[1], env, args, parts[4] == "mps"))
    return arms


MPS_SH = REPO / ".claude" / "skills" / "benchmarking" / "scripts" / "server.sh"


def mps(on: bool) -> dict:
    """Start (on) or stop the per-user MPS daemon for GPU 0; the client env when on."""
    r = subprocess.run(["bash", str(MPS_SH), "mps-start" if on else "mps-stop", "0"],
                       capture_output=True, text=True, check=False)
    log(f"mps {'start' if on else 'stop'}: {(r.stdout + r.stderr).strip().splitlines()[:1]}")
    if not on:
        return {}
    m = re.search(r"export (CUDA_MPS_PIPE_DIRECTORY=\S+) (CUDA_MPS_LOG_DIRECTORY=\S+)", r.stdout)
    if r.returncode or not m:
        raise RuntimeError(f"MPS did not start: {r.stdout} {r.stderr}")
    return dict(kv.split("=", 1) for kv in m.groups())


def phase_profile(a) -> int:
    pdir = a.out / "profile"
    pdir.mkdir(parents=True, exist_ok=True)
    for m in a.models:
        for arm_spec in parse_arms(a.arms):
            arm, tree, env = arm_spec.label, arm_spec.tree, arm_spec.env
            set_arm(tree)
            mps_env = mps(True) if arm_spec.mps else {}
            srv = H.Server(a.out, m, "mstar", 0, 8100, f"{arm}_profile", extra=arm_spec.args,
                           env_extra={"MSTAR_PHASE_TIMING": "100", **env, **mps_env})
            with H.serving(srv) as up:
                if not up:
                    log(f"profile {m} {arm}: server failed")
                    continue
                items = H.load("test_clean_200")
                H.run_requests(srv.url, H.req_model(m, "mstar"), H.repeat(items, 64), 32, H.json_fields())
                res = H.cell(srv, m, "mstar", H.repeat(items, 400), a.profile_conc, H.json_fields())
                log(f"profile {m} {arm}: RTFx {res['rtfx']:.0f} p50 {res['p50_ms'] or 0:.0f} ms "
                    f"(timing on, not a headline number)")
                H.append_jsonl(a.out / "results" / "profile.jsonl",
                               {"arm": arm, "tree": tree, "env": env, "model": m, "conc": a.profile_conc,
                                **{k: v for k, v in res.items() if k != "records"}, "ts": time.time()})
            if arm_spec.mps:
                mps(False)
            shutil.copy(srv.log_path, pdir / f"{m}_{arm}.log")
            (pdir / f"{m}_{arm}.phases.txt").write_text(render_phases(pdir / f"{m}_{arm}.log"))
    restore()
    return 0


# ----------------------------------------------------------------------------- reading the results

def render_phases(log_path: Path) -> str:
    sys.path.insert(0, str(REPO))
    from benchmark.worker_phases.phases import parse_log, render, segment
    return render(segment(parse_log(log_path.read_text(errors="replace"))))


def steady_means(text: str) -> dict[str, float]:
    """n-weighted mean_ms per phase over the steady-state segments (the batch line's 'iters a-b' with b - a
    >= 300, i.e. not the first windows), plus the mean batch."""
    tot: dict[str, list[float]] = {}
    batches = []
    take = False
    for line in text.splitlines():
        m = re.match(r"=== worker_\d+\s+iters (\d+)-(\d+).*batch=([\d.]+)", line)
        if m:
            take = int(m.group(2)) - int(m.group(1)) >= 300
            if take:
                batches.append(float(m.group(3)))
            continue
        if take:
            parts = line.split()
            if len(parts) >= 5 and parts[0] in KEY_PHASES:
                mean, n = float(parts[1]), float(parts[4])
                acc = tot.setdefault(parts[0], [0.0, 0.0])
                acc[0] += mean * n
                acc[1] += n
    out = {k: v[0] / v[1] for k, v in tot.items() if v[1]}
    if batches:
        out["batch"] = statistics.mean(batches)
    return out


def phase_summarize(a) -> int:
    path = a.out / "results" / "ab.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    cells: dict[tuple, list[float]] = {}
    for x in rows:
        if x.get("kind") == "start_failed" or x.get("errors"):
            continue
        cells.setdefault((x["model"], x["conc"], x["arm"]), []).append(x["rtfx"])
    labels = [x.label for x in parse_arms(a.arms)] + ["vllm", "sgl_omni"]
    present = [lab for lab in labels if any(k[2] == lab for k in cells)]
    lines = ["# ASR A/B", "",
             f"arms: {a.arms}; rounds alternate arm order, fresh server each; first arm is the reference", "",
             "| model | c | " + " | ".join(present) + " | "
             + " | ".join(f"{p} vs {present[0]}" for p in present[1:]) + " |",
             "|---|---:|" + "---:|" * (2 * len(present) - 1)]

    def ms(v):
        return (f"{statistics.mean(v):.0f} +- {statistics.stdev(v):.0f} (n={len(v)})" if len(v) > 1
                else f"{v[0]:.0f} (n=1)") if v else "-"

    for m in a.models:
        for c in a.concs:
            vals = [cells.get((m, c, p), []) for p in present]
            if not any(vals):
                continue
            ref = statistics.mean(vals[0]) if vals[0] else None
            deltas = [f"{100 * (statistics.mean(v) / ref - 1):+.1f}%" if v and ref else "-" for v in vals[1:]]
            lines.append(f"| {m} | {c} | " + " | ".join(ms(v) for v in vals) + " | " + " | ".join(deltas) + " |")
    pdir = a.out / "profile"
    if pdir.exists():
        arms = [x.label for x in parse_arms(a.arms)]
        prof = [json.loads(x) for x in (a.out / "results" / "profile.jsonl").read_text().splitlines()
                if x.strip()] if (a.out / "results" / "profile.jsonl").exists() else []
        lines += ["", f"## Profile pass, c={a.profile_conc} (MSTAR_PHASE_TIMING=100; steady-state means, ms)", ""]
        for m in a.models:
            got = {arm: steady_means((pdir / f"{m}_{arm}.phases.txt").read_text())
                   for arm in arms if (pdir / f"{m}_{arm}.phases.txt").exists()}
            if not got:
                continue
            rtfx = {x["arm"]: x["rtfx"] for x in prof if x["model"] == m}
            lines += [f"### {m}", "", "| phase | " + " | ".join(got) + " |", "|---|" + "---:|" * len(got)]
            lines.append("| RTFx (timing on) | " + " | ".join(f"{rtfx.get(k, 0):.0f}" for k in got) + " |")
            for k in ("batch", *KEY_PHASES):
                if any(k in d for d in got.values()):
                    lines.append(f"| {k} | "
                                 + " | ".join(f"{d[k]:.3f}" if k in d else "-" for d in got.values()) + " |")
            lines.append("")
    (a.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "ab", "baselines", "profile", "summarize"])
    ap.add_argument("--out", type=Path, default=Path.home() / "results" / "prasrperf")
    ap.add_argument("--models", default="turbo,qasr")
    ap.add_argument("--concs", default="1,8,32,64")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arms", default="base:base,change:change",
                    help="profile arms, label:tree[:ENV=V+ENV2=V2],... (tree = base or change)")
    ap.add_argument("--profile-conc", type=int, default=32)
    ap.add_argument("--unique-audio", action="store_true",
                    help="give every request distinct audio bytes (defeats content-keyed encoder caches)")
    a = ap.parse_args()
    a.out = a.out.expanduser()
    a.models = a.models.split(",")
    a.concs = [int(c) for c in a.concs.split(",")]
    (a.out / "results").mkdir(parents=True, exist_ok=True)
    atexit.register(restore)
    if a.unique_audio:
        make_audio_unique()
    H.OUT = a.out
    sys.exit({"check": phase_check, "ab": phase_ab, "baselines": phase_baselines, "profile": phase_profile,
              "summarize": phase_summarize}[a.phase](a))


if __name__ == "__main__":
    main()
