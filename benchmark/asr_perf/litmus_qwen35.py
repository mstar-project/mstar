"""AGENTS.md's CPU-overhead litmus for a change that runs for every model: Qwen3.5-0.8B text, closed loop at
c=32 and c=64, base vs change (ab.py's change.patch), on one GPU, plus an untimed MSTAR_PHASE_TIMING pass.

    python benchmark/asr_perf/litmus_qwen35.py --out DIR [--rounds 2]

Every request asks for exactly 256 tokens (ignore_eos), so output tok/s = requests x 256 / wall. Fresh server
per (round, arm), arm order alternating; 2 warmup waves at the cell's concurrency before each cell.
Rows go to DIR/results/litmus.jsonl, phase tables to DIR/profile/qwen35_<arm>.phases.txt.
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
_spec = importlib.util.spec_from_file_location("ab", HERE / "ab.py")
AB = importlib.util.module_from_spec(_spec)
sys.modules["ab"] = AB
_spec.loader.exec_module(AB)
REPO = AB.REPO
PORT = 8200
OUT_TOKENS = 256
PROMPTS = [f"Write a detailed story about a lighthouse keeper, number {i}, and the storm that changed everything."
           for i in range(64)]


def log(msg: str) -> None:
    print(f"[litmus {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def start(name: str, out: Path, env_extra: dict | None = None) -> subprocess.Popen:
    ts = int(time.time())
    cmd = [str(REPO / ".venv" / "bin" / "mstar"), "serve", "qwen3_5_0.8b", "--gpus", "0", "--port", str(PORT),
           "--socket-path-prefix", f"/tmp/lq_{name}_{ts}/", "--upload-dir", f"/tmp/lqu_{name}_{ts}/"]
    logf = open(out / "servers" / f"qwen35_{name}.log", "w")
    env = {**os.environ, "MSTAR_SHM_ARENA": "1", **(env_extra or {})}
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f"server {name} exited during startup (see its log)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return p
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    raise RuntimeError(f"server {name} not up in 30 min")


def stop(p: subprocess.Popen) -> None:
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGINT)
        try:
            p.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=30)


async def run(conc: int, n: int) -> dict:
    import aiohttp

    sem = asyncio.Semaphore(conc)
    lat, errors = [], []

    async def one(i, s):
        body = {"model": "qwen3_5_0.8b", "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
                "max_tokens": OUT_TOKENS, "max_output_tokens": OUT_TOKENS, "ignore_eos": True, "temperature": 0}
        async with sem:
            t0 = time.perf_counter()
            try:
                async with s.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body) as r:
                    payload = await r.read()
                    if r.status != 200:
                        errors.append(f"{r.status} {payload[:120]!r}")
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e)[:120])
            lat.append(time.perf_counter() - t0)

    conn = aiohttp.TCPConnector(limit=conc + 8)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), connector=conn) as s:
        t0 = time.perf_counter()
        await asyncio.gather(*(one(i, s) for i in range(n)))
        wall = time.perf_counter() - t0
    lat.sort()
    return {"conc": conc, "n": n, "wall_s": round(wall, 3), "tok_s": round(n * OUT_TOKENS / wall, 1),
            "p50_ms": round(1000 * lat[len(lat) // 2], 1), "errors": len(errors), "error_examples": errors[:3]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    for d in ("results", "servers", "profile"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    rows_path = a.out / "results" / "litmus.jsonl"
    arms = ["base", "change"]
    try:
        for r in range(1, a.rounds + 1):
            for arm in (arms if r % 2 else arms[::-1]):
                AB.set_arm(arm)
                p = start(f"{arm}_r{r}", a.out)
                try:
                    for conc in (32, 64):
                        asyncio.run(run(conc, 2 * conc))  # warmup
                        row = {"arm": arm, "round": r, **asyncio.run(run(conc, 5 * conc)), "ts": time.time()}
                        with open(rows_path, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        log(f"r{r} {arm} c={conc}: {row['tok_s']} tok/s p50 {row['p50_ms']} ms errors {row['errors']}")
                finally:
                    stop(p)
        for arm in arms:  # untimed phase tables at c=64
            AB.set_arm(arm)
            p = start(f"{arm}_profile", a.out, {"MSTAR_PHASE_TIMING": "100"})
            try:
                asyncio.run(run(64, 128))
                res = asyncio.run(run(64, 640))
                log(f"profile {arm} c=64: {res['tok_s']} tok/s (timing on)")
            finally:
                stop(p)
            log_path = a.out / "servers" / f"qwen35_{arm}_profile.log"
            (a.out / "profile" / f"qwen35_{arm}.phases.txt").write_text(AB.render_phases(log_path))
    finally:
        AB.restore()
    return 0


if __name__ == "__main__":
    sys.exit(main())
