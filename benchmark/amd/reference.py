"""Reference implementations without a server, timed one request at a time.

They write the same files as a benchmark.runner cell (aNrK/results.json with per-request records,
req_<i>.wav, cell.json), so the table and the correctness pass read them like any other arm.

    HIP_VISIBLE_DEVICES=3 python -m benchmark.amd.reference kokoro --set-dir SET --out CELL_DIR --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmark.amd.launch import REPO

SAMPLE_RATE = 24000


def _stats(values: list[float]) -> dict | None:
    from benchmark.request import _latency_stats

    if not values:
        return None
    s = _latency_stats(values)
    return {"mean": s.mean, "p50": s.p50, "p95": s.p95, "p99": s.p99}


def _write_pcm16(path: Path, audio) -> None:
    import numpy as np

    from benchmark.utils import _write_wav

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    _write_wav(pcm, str(path))


def kokoro_cell(set_dir: Path, out: Path, repeats: int, warmup: int, voice: str, correctness_src: Path) -> dict:
    """Kokoro's own package, sentence by sentence. TTFA is the first segment KPipeline yields."""
    import numpy as np
    import torch

    from benchmark.amd import correctness

    correctness.load(correctness_src)
    from benchmark.correctness.families import tts_models

    tts_models.use_refenv()
    tts_models._stub_missing_espeak()
    from kokoro import KModel, KPipeline

    model = KModel(repo_id="hexgrad/Kokoro-82M").to("cuda").eval()
    pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M", model=model)
    items = json.loads((set_dir / "manifest.json").read_text())["items"]

    def synth(text: str, seed: int):
        torch.manual_seed(seed)
        t0 = time.monotonic()
        first, parts = None, []
        for r in pipe(text, voice=voice, speed=1.0):
            if r.audio is None:
                continue
            parts.append(r.audio.detach().float().cpu().numpy())  # the copy waits for the GPU
            if first is None:
                first = time.monotonic() - t0
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return audio, first, time.monotonic() - t0

    # Warm up on every sentence once: MIOpen tunes each new convolution shape on first use, which
    # would otherwise land in the timed pass (mstar's own decoder shapes are captured at startup).
    for w in range(warmup if warmup >= 0 else len(items)):
        synth(items[w % len(items)]["prompt"], w)
    reps = []
    for k in range(1, repeats + 1):
        rdir = out / f"a1r{k}"
        rdir.mkdir(parents=True, exist_ok=True)
        per, ttfa, rtf, durs = [], [], [], []
        t_wall = time.monotonic()
        for i, it in enumerate(items):
            audio, first, e2e = synth(it["prompt"], i)
            dur = len(audio) / SAMPLE_RATE
            _write_pcm16(rdir / f"req_{i}.wav", audio)
            per.append({"request_id": i, "jct_ms": e2e * 1000.0, "type": "text_to_speech",
                        "output_bytes": {"audio": 2 * len(audio)}, "ttft_s": {"audio": first},
                        "output_text_tokens": 0, "input_tokens": 0, "response_chunks": {}})
            ttfa.append(first)
            durs.append(dur)
            if dur > 0:
                rtf.append(e2e / dur)
        wall = time.monotonic() - t_wall
        e2es = [p["jct_ms"] / 1000.0 for p in per]
        results = {"system": "reference", "inference_system": "kokoro-package", "max_concurrency": 1,
                   "request_type": "text_to_speech", "profiling_type": "sequential", "num_requests": len(items),
                   "num_warmup": warmup, "completed": len(items), "failed": 0, "wall_time_s": wall,
                   "request_throughput": len(items) / wall,
                   "aggregate": {"max_concurrency": 1, "ttft_s": {"audio": _stats(ttfa)}, "e2e_latency_s": _stats(e2es),
                                 "rtf": _stats(rtf), "audio_seconds_throughput": sum(durs) / wall,
                                 "audio_duration_mean_s": statistics.mean(durs) if durs else None},
                   "per_request": per, "errors": []}
        (rdir / "results.json").write_text(json.dumps(results, indent=1))
        reps.append(str(rdir))
    return {"repeats": reps, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None)}


def run_arm(plan, row, arm, gpus: list[int], arm_dir: Path) -> dict:
    """Run a reference arm in a child process that sees only its GPU, then build its cell like run_cell."""
    from benchmark.amd import sweep

    wl = next(w for w in plan.workloads if w.family == row.family)
    if arm.name != "ref-kokoro" or row.runner_model != "kokoro":
        return {"skipped": f"no reference timing for {row.name} in {arm.name}"}
    set_dir = sweep.SETS_ROOT / "sets" / wl.prompts[4:]
    cell_dir = arm_dir / "cells" / wl.name / "c1"
    cell_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HIP_VISIBLE_DEVICES": str(gpus[0]), "CUDA_VISIBLE_DEVICES": str(gpus[0]),
           "HF_HUB_OFFLINE": "1", "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])}
    cmd = [sys.executable, "-m", "benchmark.amd.reference", "kokoro", "--set-dir", str(set_dir), "--out",
           str(cell_dir), "--repeats", str(plan.repeats), "--correctness-src", str(plan.correctness_src)]
    (cell_dir / "cotenants.json").write_text(json.dumps(sweep.cotenants(os.environ.get("SLURM_JOB_ID")), indent=1))
    with open(cell_dir / "reference.log", "w") as log:
        p = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    reps = []
    for k in range(1, plan.repeats + 1):
        res = cell_dir / f"a1r{k}" / "results.json"
        if p.returncode == 0 and res.is_file():
            st = sweep.repeat_stats(json.loads(res.read_text()), row.family)
            st["dir"] = str(res.parent)
            reps.append(st)
        else:
            reps.append({"error": f"reference rc {p.returncode}", "dir": str(res.parent)})
    good = [r for r in reps if "error" not in r]
    summary = sweep.summarize_repeats(good)
    cell = {"status": "done" if len(good) == len(reps) else "error", "arm": arm.name, "row": row.name,
            "workload": wl.name, "concurrency": 1, "num_requests": None, "summary": summary,
            "spread": sweep.spread(summary, wl.primary[0]), "primary": wl.primary[0], "flags": ["single-stream"],
            "errors": [r["error"] for r in reps if "error" in r],
            "attempts": [{"attempt": 1, "repeats": reps, "summary": summary}]}
    (cell_dir / "cell.json").write_text(json.dumps(cell, indent=1))
    return {"cell": str(cell_dir), "status": cell["status"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("impl", choices=["kokoro"])
    ap.add_argument("--set-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=-1, help="warmup requests (default: one pass over the set)")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--correctness-src", default=str(REPO))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = kokoro_cell(Path(args.set_dir), out, args.repeats, args.warmup, args.voice, Path(args.correctness_src))
    (out / "reference.json").write_text(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
