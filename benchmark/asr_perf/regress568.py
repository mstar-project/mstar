"""Does a change regress models other than the ones it targets? Two or more arms on one GPU.

The motivating case was #568 (or #318 under it): arms for main, model/asr (#318) and #568 on top of it.

Each arm is a full source tree given as --arm LABEL=PATH; the shared venv's editable mstar is shadowed by
PYTHONPATH=<arm tree>, so the arms must not differ under rust/ (the Rust extension is the venv's). Clients and
data come from this tree.

  python benchmark/asr_perf/regress568.py run --out DIR --arm main=/path/to/main --arm pr=/path/to/pr
      [--arm LABEL=PATH ...] [--rounds 2] [--models qwen3_tts,higgs,qwen35] [--venv-bin REPO/.venv/bin]
  python benchmark/asr_perf/regress568.py wer --out DIR --arm main=/path/to/main --arm pr=/path/to/pr
      (TTS audio of round 1, c=8, through Whisper turbo)

The first --arm is the baseline; odd rounds run the arms in --arm order, even rounds in reverse.

Models, chosen for what #568's arms changed:
  qwen3_tts  Qwen3-TTS-12Hz-1.7B (talker and code predictor GQA 16/8 = 2: the decode-kernel switch), streamed
             audio (many chunks per request through the API server's output path). tts_speech_bench at
             c=1/8/32: TTFA, audio s/s.
  higgs      Higgs-audio v3 STT (GQA 2), transcriptions at c=1/8/32: RTFx, WER on LibriSpeech test-clean.
  qwen35     Qwen3.5-0.8B text (GQA 4: unchanged kernel; the CPU-overhead litmus), 256 tokens at c=32/64.
Fresh server per (round, model, arm), then one MSTAR_PHASE_TIMING server per
(model, arm) at c=8 (untimed). Pass bars, fixed before the run: 0 errors everywhere; a TTS WER within 1 point
of the baseline's; a throughput change counts only past both arms' round-to-round spread.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
_spec = importlib.util.spec_from_file_location("asr318", REPO / "benchmark" / "asr318_run" / "run.py")
H = importlib.util.module_from_spec(_spec)
sys.modules["asr318"] = H
_spec.loader.exec_module(H)
_spec2 = importlib.util.spec_from_file_location("litmus", HERE / "litmus_qwen35.py")
LIT = importlib.util.module_from_spec(_spec2)
sys.modules["litmus"] = LIT
_spec2.loader.exec_module(LIT)

PORT = 8300
VENV_BIN = REPO / ".venv" / "bin"  # overridden by --venv-bin
ARMS: dict[str, Path] = {}  # label -> tree, in --arm order; filled by main()
MODELS = {
    "qwen3_tts": ["qwen3_tts_1p7b", "--config", "configs/qwen3tts_1p7b.yaml"],
    "higgs": ["higgs_audio", "--config", "configs/higgs_audio.yaml"],
    "qwen35": ["qwen3_5_0.8b"],
}


def log(msg: str) -> None:
    print(f"[regress {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_arm(spec: str) -> tuple[str, Path]:
    label, sep, path = spec.partition("=")
    if not sep or not label or not path or "/" in label:
        raise argparse.ArgumentTypeError(f"--arm wants LABEL=PATH (LABEL without '/'), got {spec!r}")
    return label, Path(path).expanduser().resolve()


def start(arm: str, model: str, name: str, out: Path, env_extra: dict | None = None) -> subprocess.Popen:
    tree = ARMS[arm]
    ts = int(time.time())
    cmd = [str(VENV_BIN / "mstar"), "serve", *MODELS[model], "--gpus", "0", "--port", str(PORT),
           "--socket-path-prefix", f"/tmp/rg_{name}_{ts}/", "--upload-dir", f"/tmp/rgu_{name}_{ts}/"]
    env = {**os.environ, "PYTHONPATH": str(tree), "MSTAR_SHM_ARENA": "1",
           "PATH": f"{VENV_BIN}:{os.environ.get('PATH', '')}", **(env_extra or {})}
    logf = open(out / "servers" / f"{name}.log", "w")
    p = subprocess.Popen(cmd, cwd=tree, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f"{name}: server exited during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return p
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    stop(p)
    raise RuntimeError(f"{name}: not up in 40 min")


def stop(p: subprocess.Popen) -> None:
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGINT)
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=30)


def sentences_file(out: Path) -> Path:
    path = out / "sentences.txt"
    if not path.exists():
        lines = [u.reference.lower() for u in H.load("test_clean_200")[:128]]
        path.write_text("\n".join(lines) + "\n")
    return path


def tts_cell(out: Path, tag: str, conc: int, save_audio: bool) -> dict:
    n = {1: 24, 8: 64, 32: 128}[conc]
    report = out / "tts" / f"{tag}_c{conc}.json"
    cmd = [str(VENV_BIN / "python"), "-m", "benchmark.tts_speech_bench", "--engine", "mstar",
           "--url", f"http://127.0.0.1:{PORT}", "--model", "qwen3_tts_1p7b", "--sentences", str(sentences_file(out)),
           "--num-sentences", str(n), "--concurrency", str(conc), "--repeats", "1", "--warmup", "3",
           "--voice", "vivian", "--language", "English", "--out", str(report)]
    if save_audio:
        cmd += ["--save-audio-dir", str(out / "tts_audio" / tag)]
    subprocess.run(cmd, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, check=True,
                   stdout=subprocess.DEVNULL, timeout=3600)
    med = json.loads(report.read_text())
    m = med["median_over_repeats"]
    return {"conc": conc, "n": n, "errors": med["errors_total"], "ttfa_p50_ms": round(m["ttfa_p50_ms"], 1),
            "ttfa_p95_ms": round(m["ttfa_p95_ms"], 1), "audio_s_per_s": round(m["audio_s_per_wall_s"], 2),
            "rtf_mean": round(m["rtf_mean"], 4)}


def asr_cell(conc: int) -> dict:
    items = H.load("test_clean_200")
    n = {1: 48, 8: 200, 32: 400}[conc]
    batch = H.repeat(items, n)
    H.run_requests(f"http://127.0.0.1:{PORT}", "higgs_audio", H.repeat(items, conc), conc, H.json_fields())
    recs, wall = H.run_requests(f"http://127.0.0.1:{PORT}", "higgs_audio", batch, conc, H.json_fields())
    audio = sum(u.duration for u in batch)
    ok = [r for r in recs if r.get("status") == 200]
    refs = {u.uid: u.reference for u in items}
    w = H.wer([refs[r["uid"]] for r in ok], [r.get("text", "") for r in ok]) if ok else None
    return {"conc": conc, "n": n, "errors": len(recs) - len(ok), "rtfx": round(audio / wall, 1),
            "wer": None if w is None else round(w, 4)}


def text_cell(conc: int) -> dict:
    LIT.PORT = PORT
    asyncio.run(LIT.run(conc, 2 * conc))
    r = asyncio.run(LIT.run(conc, 5 * conc))
    return {"conc": conc, "n": r["n"], "errors": r["errors"], "tok_s": r["tok_s"], "p50_ms": r["p50_ms"]}


def cells(model: str, out: Path, tag: str, save_audio: bool, concs=None) -> list[dict]:
    if model == "qwen3_tts":
        return [tts_cell(out, tag, c, save_audio and c == 8) for c in (concs or (1, 8, 32))]
    if model == "higgs":
        return [asr_cell(c) for c in (concs or (1, 8, 32))]
    return [text_cell(c) for c in (concs or (32, 64))]


def phase_run(a) -> int:
    for d in ("results", "servers", "tts", "tts_audio", "profile"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    rows = a.out / "results" / "regress.jsonl"
    done = set()
    if rows.exists():
        for line in rows.read_text().splitlines():
            r = json.loads(line)
            done.add((r["round"], r["model"], r["arm"]))
    for r in range(1, 0 if a.profile_only else a.rounds + 1):
        for model in a.models:
            order = list(ARMS) if r % 2 else list(reversed(ARMS))
            for arm in order:
                if (r, model, arm) in done:
                    continue
                name = f"{model}_{arm}_r{r}"
                try:
                    p = start(arm, model, name, a.out)
                except RuntimeError as e:
                    row = {"round": r, "model": model, "arm": arm, "error": str(e)}
                else:
                    try:
                        row = {"round": r, "model": model, "arm": arm,
                               "cells": cells(model, a.out, f"{arm}_r{r}", save_audio=(r == 1))}
                    except Exception as e:  # noqa: BLE001 - recorded; the next arm still runs
                        row = {"round": r, "model": model, "arm": arm, "error": repr(e)[:300]}
                    finally:
                        stop(p)
                row["ts"] = time.time()
                with open(rows, "a") as f:
                    f.write(json.dumps(row) + "\n")
                log(json.dumps(row)[:400])
    for model in a.models:  # the profile pass: untimed, one server per (model, arm)
        for arm in ARMS:
            name = f"{model}_{arm}_profile"
            if (a.out / "profile" / f"{name}.phases.txt").exists():
                continue
            try:
                p = start(arm, model, name, a.out, {"MSTAR_PHASE_TIMING": "100"})
            except RuntimeError as e:
                log(f"profile {name}: {e}")
                continue
            try:
                cells(model, a.out, f"{arm}_profile", save_audio=False, concs=(8,))
            finally:
                stop(p)
            from benchmark.worker_phases.phases import parse_log, render, segment
            text = (a.out / "servers" / f"{name}.log").read_text(errors="replace")
            (a.out / "profile" / f"{name}.phases.txt").write_text(render(segment(parse_log(text))))
            log(f"profile {name} written")
    return 0


def phase_wer(a) -> int:
    for arm in ARMS:
        audio = a.out / "tts_audio" / f"{arm}_r1"
        if not audio.is_dir():
            continue
        report = a.out / "tts" / f"wer_{arm}.json"
        subprocess.run([str(VENV_BIN / "python"), "-m", "benchmark.tts_speech_wer", "--audio-dir", str(audio),
                        "--sentences", str(sentences_file(a.out)), "--out", str(report)],
                       cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, check=False)
        log(f"wer {arm}: {report.read_text()[:300] if report.exists() else 'no report'}")
    return 0


def main() -> int:
    global VENV_BIN
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "wer"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--models", default="qwen3_tts,higgs,qwen35")
    ap.add_argument("--profile-only", action="store_true", help="only the MSTAR_PHASE_TIMING pass")
    ap.add_argument("--arm", type=parse_arm, action="append", default=[], metavar="LABEL=PATH",
                    help="an arm's label and source tree; repeat for each arm (at least two)")
    ap.add_argument("--venv-bin", type=Path, default=VENV_BIN,
                    help="bin/ of the venv all arms share (default: REPO/.venv/bin)")
    a = ap.parse_args()
    a.models = a.models.split(",")
    labels = [label for label, _ in a.arm]
    if len(labels) < 2:
        ap.error("give at least two --arm LABEL=PATH")
    if len(set(labels)) != len(labels):
        ap.error(f"duplicate --arm labels: {labels}")
    for label, tree in a.arm:
        if not tree.is_dir():
            ap.error(f"--arm {label}: {tree} is not a directory")
    VENV_BIN = a.venv_bin.expanduser().absolute()
    ARMS.update(a.arm)
    sys.path.insert(0, str(REPO))
    return phase_run(a) if a.phase == "run" else phase_wer(a)


if __name__ == "__main__":
    sys.exit(main())
