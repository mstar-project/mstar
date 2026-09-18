#!/usr/bin/env python3
"""Image-generation benchmark client over the OpenAI ``/v1/images/generations`` route.

Drives any server that speaks the route — mstar, vLLM-Omni (``vllm serve <id> --omni``),
SGLang-Diffusion (``sglang serve``) — with the same prompts, size, step count and seeds,
and reports what the benchmark protocol asks for:

* ``--mode latency``     B=1 end-to-end latency at the given size (median / p95 over N prompts
                         after warmup)
* ``--mode throughput``  images/s at a fixed concurrency (``--concurrency 4 8 16``): that many
                         requests kept in flight, prompts round-robin from the prompt file;
                         each level is repeated (``--repeats``, default 3) and the median reported

Peak VRAM (``memory.used`` of ``--vram-gpu`` sampled by ``nvidia-smi`` during the timed run)
is recorded when the client runs on the server's node, as the protocol does.

Prompts come from a text file (one per line; the shared protocol set lives under
``commons/bench/data/image/``), so every engine sees identical inputs. Results are
written as JSON (``--out``) with the exact request parameters and timing samples.

Examples::

   python benchmark/flux2_klein/bench_images.py --port 8000 --model flux2_klein --mode latency \\
       --prompts prompts.txt --size 1024x1024 --steps 4 --n 20 --warmup 3 --out results/mstar_latency.json
   python benchmark/flux2_klein/bench_images.py --port 8002 --model black-forest-labs/FLUX.2-klein-4B \\
       --mode throughput --concurrency 4 8 16 --prompts prompts.txt --steps 4 --n 32 --out results/vllm_tp.json
"""

from __future__ import annotations

import argparse
import base64
import functools
import json
import shutil
import statistics
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class VramPoller:
    """Peak ``memory.used`` of one GPU (MiB) sampled by ``nvidia-smi`` while a run is timed.

    The client runs on the server's node, so this is the device's high-water mark (the
    engine's allocator plus context), which is what the protocol's "peak VRAM" means.
    """

    def __init__(self, gpu: str, interval_s: float = 0.5):
        self.gpu, self.interval_s, self.peak_mib = gpu, interval_s, None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = bool(gpu) and shutil.which("nvidia-smi") is not None

    def _sample(self) -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "-i", self.gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
            return int(out.splitlines()[0])
        except (subprocess.SubprocessError, ValueError, IndexError, OSError):
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            mib = self._sample()
            if mib is not None:
                self.peak_mib = mib if self.peak_mib is None else max(self.peak_mib, mib)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "VramPoller":
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            mib = self._sample()  # the level right after the run, for the record
            if mib is not None:
                self.peak_mib = mib if self.peak_mib is None else max(self.peak_mib, mib)


def image_format(data: bytes | None) -> str | None:
    """``png`` / ``jpeg`` / ``webp`` from the magic bytes, else None."""
    if not data:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def _post(url: str, body: dict, timeout: float) -> tuple[float, bytes | None]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    dt = time.perf_counter() - t0
    data = payload.get("data") or []
    b64 = data[0].get("b64_json") if data else None
    return dt, (base64.b64decode(b64) if b64 else None)


def _body(args, prompt: str, seed: int) -> dict:
    body = {
        "model": args.model, "prompt": prompt, "size": args.size, "n": 1, "response_format": "b64_json",
        "num_inference_steps": args.steps, "seed": seed,
    }
    if args.guidance is not None:
        body["guidance_scale"] = args.guidance
    if args.output_format:
        body["output_format"] = args.output_format
    return body


def _summary(samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "n": len(s), "median_s": statistics.median(s), "mean_s": statistics.fmean(s),
        "p95_s": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))], "min_s": s[0], "max_s": s[-1],
    }


def run_latency(args, url: str, prompts: list[str]) -> dict:
    for i in range(args.warmup):
        _post(url, _body(args, prompts[i % len(prompts)], args.seed + i), args.timeout)
    samples, saved, observed = [], 0, None
    with VramPoller(args.vram_gpu) as vram:
        for i in range(args.n):
            dt, png = _post(url, _body(args, prompts[i % len(prompts)], args.seed + i), args.timeout)
            samples.append(dt)
            observed = observed or image_format(png)
            if args.save_dir and png is not None:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                Path(args.save_dir, f"{args.tag}_{i:03d}.png").write_bytes(png)
                saved += 1
            print(f"  [{i + 1}/{args.n}] {dt:.3f}s", flush=True)
    result = {
        "mode": "latency", **_summary(samples), "samples_s": samples, "images_saved": saved,
        "peak_vram_mib": vram.peak_mib, "observed_output_format": observed,
    }
    print(f"latency B=1 {args.size} steps={args.steps}: median {result['median_s']:.3f}s  p95 {result['p95_s']:.3f}s")
    return result


class _RoundRobin:
    """Thread-safe request counter: request ``i`` takes prompt ``i % len(prompts)`` and seed ``seed + i``."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next = 0

    def take(self) -> int:
        with self._lock:
            i = self._next
            self._next += 1
            return i

    def reset(self) -> None:
        with self._lock:
            self._next = 0


_OBSERVED: dict[str, str | None] = {"format": None}


def _one_request(args, url: str, prompts: list[str], counter: _RoundRobin, _index: int = 0) -> float:
    """One request; ``_index`` is the executor's map argument and is ignored (the counter
    assigns prompts/seeds in completion-independent submission order)."""
    i = counter.take()
    dt, png = _post(url, _body(args, prompts[i % len(prompts)], args.seed + i), args.timeout)
    if _OBSERVED["format"] is None:
        _OBSERVED["format"] = image_format(png)
    return dt


def run_throughput(args, url: str, prompts: list[str]) -> dict:
    results = {}
    for concurrency in args.concurrency:
        counter = _RoundRobin()
        request = functools.partial(_one_request, args, url, prompts, counter)
        runs = []
        with VramPoller(args.vram_gpu) as vram, ThreadPoolExecutor(max_workers=concurrency) as pool:
            # warm the shape / batch buckets before timing
            list(pool.map(request, range(min(args.warmup, concurrency))))
            for _ in range(max(1, args.repeats)):
                counter.reset()
                t0 = time.perf_counter()
                lat = list(pool.map(request, range(args.n)))
                wall = time.perf_counter() - t0
                runs.append({"wall_s": wall, "images_per_s": args.n / wall, "request_latency": _summary(lat)})
        rates = sorted(r["images_per_s"] for r in runs)
        results[str(concurrency)] = {
            "concurrency": concurrency, "images": args.n, "repeats": len(runs),
            "images_per_s": statistics.median(rates), "images_per_s_min": rates[0], "images_per_s_max": rates[-1],
            "wall_s": statistics.median(r["wall_s"] for r in runs),
            "request_latency": runs[len(runs) // 2]["request_latency"], "runs": runs,
            "peak_vram_mib": vram.peak_mib,
        }
        print(f"throughput concurrency={concurrency}: {statistics.median(rates):.3f} images/s "
              f"(median of {len(runs)}; min {rates[0]:.3f}, max {rates[-1]:.3f}), "
              f"request median {runs[len(runs) // 2]['request_latency']['median_s']:.3f}s", flush=True)
    return {"mode": "throughput", "by_concurrency": results, "observed_output_format": _OBSERVED["format"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True, help="model id the server expects in the request")
    ap.add_argument("--mode", choices=["latency", "throughput"], default="latency")
    ap.add_argument("--prompts", required=True, help="text file, one prompt per line")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=None, help="omit for distilled models")
    ap.add_argument("--output-format", default=None, choices=[None, "png", "jpeg", "webp"],
                    help="request this output_format (servers default differently: SGLang to JPEG, others to PNG)")
    ap.add_argument("--seed", type=int, default=0, help="request i uses seed + i")
    ap.add_argument("--n", type=int, default=20, help="measured requests (latency) / images per concurrency level")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--save-dir", default="", help="save the PNGs of the latency run here (for PSNR checks)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--repeats", type=int, default=3,
                    help="throughput: repeat each concurrency level this many times and report the median images/s")
    ap.add_argument("--vram-gpu", default="0",
                    help="GPU index polled with nvidia-smi for peak memory.used during the run ('' to disable)")
    ap.add_argument("--out", default="", help="write the JSON result here")
    args = ap.parse_args()

    prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts}")
    url = f"http://{args.host}:{args.port}/v1/images/generations"
    print(f"=== {args.tag}: {url} model={args.model} size={args.size} steps={args.steps} mode={args.mode}", flush=True)
    result = run_latency(args, url, prompts) if args.mode == "latency" else run_throughput(args, url, prompts)
    result.update({
        "tag": args.tag, "url": url, "model": args.model, "size": args.size, "steps": args.steps,
        "guidance": args.guidance, "seed": args.seed, "prompts_file": args.prompts, "num_prompts": len(prompts),
        "output_format": args.output_format,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
