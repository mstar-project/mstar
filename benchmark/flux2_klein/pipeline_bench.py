#!/usr/bin/env python3
"""Protocol benchmark for image engines that expose only a Python API (diffusers, xDiT, LightX2V).

``bench_images.py`` drives HTTP servers; this driver runs the same protocol in-process and writes
the same JSON, so ``summarize_bench.py`` renders every engine in one table:

* latency: B=1, ``--n`` measured requests after ``--warmup`` warmups; request ``i`` uses prompt
  ``i % len(prompts)`` and seed ``--seed + i``; one sample = pipeline call + PNG encoding (what a
  server's latency includes), with the pipeline-only time kept next to it (``pipe_samples_s``).
* throughput: a Python API has no request queue, so "concurrency c" means c requests in flight =
  one batched call of c images (a list of c prompts, one generator per image); ``--n`` images per
  level, ``--repeats`` repeats, median images/s. Engines that take one prompt per call (LightX2V)
  run the c images back to back, which the JSON records under ``batching``.
* peak VRAM: ``nvidia-smi memory.used`` sampled during the timed run (the client's poller).

Seeds go through a CPU ``torch.Generator`` per image where the engine lets us (diffusers, xDiT),
the convention of the M* oracles, so those rows can be compared with the eager reference images
(``psnr.py --dirs``). PNG bytes come from the engine's own path: PIL ``Image.save`` for pipelines
that return PIL images, the engine's writer for LightX2V.

Run it from the engine's own environment; xDiT needs ``torchrun --nproc_per_node=1`` because its
runner initialises torch.distributed::

   commons/envs/diffusers/bin/python benchmark/flux2_klein/pipeline_bench.py --engine diffusers \\
       --model black-forest-labs/FLUX.2-klein-4B --steps 4 --guidance 1.0 --prompts prompts_100.txt \\
       --modes latency throughput --tag diffusers --out-prefix results/diffusers_flux2_klein --save-dir ref_png
   commons/envs/xfuser/bin/torchrun --nproc_per_node=1 benchmark/flux2_klein/pipeline_bench.py --engine xdit ...
   commons/envs/lightx2v/bin/python benchmark/flux2_klein/pipeline_bench.py --engine lightx2v \\
       --engine-config notes/cfg_lightx2v/klein_t2i.json ...
"""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_images import VramPoller, _summary, image_format  # noqa: E402  (same directory, stdlib only)


def _parse_size(size: str) -> tuple[int, int]:
    width, height = size.lower().split("x")
    return int(width), int(height)


def _png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _version(dist: str) -> str:
    try:
        return f"{dist} {importlib.metadata.version(dist)}"
    except importlib.metadata.PackageNotFoundError:
        return dist


def _local_snapshot(model: str) -> str:
    """The cached snapshot directory of ``model`` (compute nodes run offline), else the id itself."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, local_files_only=True)
    except Exception:  # noqa: BLE001 - any failure means "let the engine resolve it"
        return model


class DiffusersEngine:
    """Stock diffusers ``Flux2KleinPipeline`` / ``ZImagePipeline``, eager or ``torch.compile``d
    (transformer and VAE decode, static shapes)."""

    name = "diffusers"
    batching = "batched call: list of prompts, one CPU generator per image"

    def __init__(self, args):
        import torch
        from diffusers import Flux2KleinPipeline, ZImagePipeline

        self.torch = torch
        cls = ZImagePipeline if "z-image" in args.model.lower() else Flux2KleinPipeline
        self.pipe = cls.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        if args.compile:
            self.pipe.transformer.compile(mode=args.compile_mode, dynamic=False)
            self.pipe.vae.decode = torch.compile(self.pipe.vae.decode, mode=args.compile_mode, dynamic=False)
        self.width, self.height = _parse_size(args.size)
        self.steps, self.guidance = args.steps, args.guidance
        self.version = f"{_version('diffusers')}, torch {torch.__version__}"
        self.last_pipe_s = 0.0

    def generate(self, prompts: list[str], seeds: list[int]) -> list[bytes]:
        gens = [self.torch.Generator(device="cpu").manual_seed(int(s)) for s in seeds]
        t0 = time.perf_counter()
        out = self.pipe(
            prompt=list(prompts), height=self.height, width=self.width, num_inference_steps=self.steps,
            guidance_scale=self.guidance, generator=gens,
        )
        self.last_pipe_s = time.perf_counter() - t0
        return [_png_bytes(image) for image in out.images]

    def close(self) -> None:
        pass


class XditEngine:
    """xDiT's unified runner (``xfuser.runner.xFuserModelRunner``) on one GPU. The runner applies the
    engine's own optimisations (``--use_torch_compile``, fp8/int8 GEMMs, caches) exactly as the ``xdit``
    CLI does; extra runner flags pass through ``--engine-args``."""

    name = "xdit"
    batching = "batched call: list of prompts, one CPU generator per call"

    def __init__(self, args):
        import torch
        from xfuser import xFuserArgs
        from xfuser.config import FlexibleArgumentParser
        from xfuser.runner import xFuserModelRunner

        self.width, self.height = _parse_size(args.size)
        parser = FlexibleArgumentParser(description="xFuser Arguments")
        xFuserArgs.add_runner_args(parser)
        argv = [
            "--model", args.model, "--height", str(self.height), "--width", str(self.width),
            "--num_inference_steps", str(args.steps), "--guidance_scale", str(args.guidance),
            "--prompt", "warmup", "--seed", "0", "--num_iterations", "1", "--warmup_calls", "0",
            "--output_directory", tempfile.mkdtemp(prefix="xdit_bench_"),
        ]
        if args.compile:
            argv.append("--use_torch_compile")
        argv += args.engine_args
        config = vars(parser.parse_args(argv))
        self.runner = xFuserModelRunner(config)
        self.input_args = self.runner.preprocess_args(config)
        self.runner.initialize(self.input_args)
        # the oracles' convention: noise from a seeded CPU generator (the runner's default is a CUDA one)
        self.runner.model._make_generator = lambda seed: torch.Generator(device="cpu").manual_seed(int(seed))
        self.version = _version("xfuser")
        self.last_pipe_s = 0.0

    def generate(self, prompts: list[str], seeds: list[int]) -> list[bytes]:
        request = dict(self.input_args)
        request["prompt"] = list(prompts)
        request["seed"] = int(seeds[0])
        t0 = time.perf_counter()
        out = self.runner.model._run_pipe(request)
        self.last_pipe_s = time.perf_counter() - t0
        return [_png_bytes(image) for image in out.images]

    def close(self) -> None:
        self.runner.cleanup()


class LightX2VEngine:
    """LightX2V's ``LightX2VPipeline``: one prompt per call, the engine writes the PNG itself
    (``--engine-config`` is its JSON: steps, guidance, attention, rope)."""

    name = "lightx2v"
    batching = "none: one prompt per call, the c images run back to back"

    def __init__(self, args):
        from lightx2v import LightX2VPipeline

        if not args.engine_config:
            raise SystemExit("--engine-config <lightx2v json> is required for --engine lightx2v")
        model_path = _local_snapshot(args.model)
        if "z-image" in args.model.lower():
            self.pipe = LightX2VPipeline(model_path=model_path, model_cls="z_image", task="t2i")
        else:
            self.pipe = LightX2VPipeline(model_path=model_path, model_cls="flux2", model_variant="klein", task="t2i")
        self.pipe.create_generator(config_json=args.engine_config)
        self.width, self.height = _parse_size(args.size)
        self.out_dir = Path(tempfile.mkdtemp(prefix="lightx2v_bench_"))
        self.version = _version("lightx2v")
        self.last_pipe_s = 0.0

    def generate(self, prompts: list[str], seeds: list[int]) -> list[bytes]:
        pngs = []
        t0 = time.perf_counter()
        for i, (prompt, seed) in enumerate(zip(prompts, seeds, strict=True)):
            path = self.out_dir / f"{int(seed)}_{i}.png"
            self.pipe.generate(
                seed=int(seed), prompt=prompt, save_result_path=str(path), size=(self.height, self.width),
            )
            pngs.append(path.read_bytes())
            path.unlink(missing_ok=True)
        self.last_pipe_s = time.perf_counter() - t0  # includes the engine's own PNG write
        return pngs

    def close(self) -> None:
        pass


ENGINES = {"diffusers": DiffusersEngine, "xdit": XditEngine, "lightx2v": LightX2VEngine}


def run_latency(engine, args, prompts: list[str]) -> dict:
    for i in range(args.warmup):
        engine.generate([prompts[i % len(prompts)]], [args.seed + i])
    samples, pipe_samples, saved, observed = [], [], 0, None
    with VramPoller(args.vram_gpu) as vram:
        for i in range(args.n):
            t0 = time.perf_counter()
            png = engine.generate([prompts[i % len(prompts)]], [args.seed + i])[0]
            dt = time.perf_counter() - t0
            samples.append(dt)
            pipe_samples.append(engine.last_pipe_s)
            observed = observed or image_format(png)
            if args.save_dir:
                Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                Path(args.save_dir, f"{args.tag}_{i:03d}.png").write_bytes(png)
                saved += 1
            print(f"  [{i + 1}/{args.n}] {dt:.3f}s (pipeline {engine.last_pipe_s:.3f}s)", flush=True)
    result = {
        "mode": "latency", **_summary(samples), "samples_s": samples, "pipe_samples_s": pipe_samples,
        "pipe_median_s": statistics.median(pipe_samples), "images_saved": saved,
        "peak_vram_mib": vram.peak_mib, "observed_output_format": observed,
    }
    print(f"latency B=1 {args.size} steps={args.steps}: median {result['median_s']:.3f}s  p95 {result['p95_s']:.3f}s  "
          f"(pipeline only {result['pipe_median_s']:.3f}s)")
    return result


def run_throughput(engine, args, prompts: list[str]) -> dict:
    results = {}
    for concurrency in args.concurrency:
        try:
            results[str(concurrency)] = _throughput_level(engine, args, prompts, concurrency)
        except Exception as exc:  # noqa: BLE001 - one level failing (OOM at a large batch) must not lose the others
            print(f"throughput concurrency={concurrency}: FAILED {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            results[str(concurrency)] = {
                "concurrency": concurrency, "images": args.n, "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
            _free_gpu_memory()
    return {"mode": "throughput", "by_concurrency": results, "observed_output_format": "png"}


def _free_gpu_memory() -> None:
    try:
        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _throughput_level(engine, args, prompts: list[str], concurrency: int) -> dict:
    """One concurrency level: warm the batch shape, then ``--repeats`` timed passes over ``--n`` images."""
    engine.generate(prompts[:concurrency], list(range(args.seed, args.seed + concurrency)))  # warm the batch shape
    runs = []
    if True:
        with VramPoller(args.vram_gpu) as vram:
            for _ in range(max(1, args.repeats)):
                t0, latencies, index, pipe_wall = time.perf_counter(), [], 0, 0.0
                while index < args.n:
                    batch = min(concurrency, args.n - index)
                    tb = time.perf_counter()
                    engine.generate(
                        [prompts[(index + j) % len(prompts)] for j in range(batch)],
                        [args.seed + index + j for j in range(batch)],
                    )
                    latencies.extend([time.perf_counter() - tb] * batch)  # every image in the batch waited the batch
                    pipe_wall += engine.last_pipe_s
                    index += batch
                wall = time.perf_counter() - t0
                runs.append({
                    "wall_s": wall, "images_per_s": args.n / wall, "request_latency": _summary(latencies),
                    "pipe_wall_s": pipe_wall, "images_per_s_pipeline": args.n / pipe_wall,
                })
        rates = sorted(r["images_per_s"] for r in runs)
        pipeline_rates = sorted(r["images_per_s_pipeline"] for r in runs)
        level = {
            "concurrency": concurrency, "images": args.n, "repeats": len(runs),
            "images_per_s": statistics.median(rates), "images_per_s_min": rates[0], "images_per_s_max": rates[-1],
            "images_per_s_pipeline": statistics.median(pipeline_rates),
            "wall_s": statistics.median(r["wall_s"] for r in runs),
            "request_latency": runs[len(runs) // 2]["request_latency"], "runs": runs,
            "peak_vram_mib": vram.peak_mib,
        }
        print(f"throughput concurrency={concurrency}: {statistics.median(rates):.3f} images/s "
              f"(median of {len(runs)}; min {rates[0]:.3f}, max {rates[-1]:.3f}; pipeline only "
              f"{statistics.median(pipeline_rates):.3f})", flush=True)
        return level


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=sorted(ENGINES), required=True)
    ap.add_argument("--model", required=True, help="Hugging Face id (resolved from the offline cache)")
    ap.add_argument("--prompts", required=True, help="text file, one prompt per line")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0, help="klein 1.0, Z-Image-Turbo 0.0")
    ap.add_argument("--seed", type=int, default=0, help="request i uses seed + i")
    ap.add_argument("--modes", nargs="+", choices=["latency", "throughput"], default=["latency", "throughput"])
    ap.add_argument("--n", type=int, default=20, help="measured requests (latency) / images per level (throughput)")
    ap.add_argument("--n-throughput", type=int, default=32, help="images per concurrency level")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--compile", action="store_true", help="the engine's torch.compile path")
    ap.add_argument("--compile-mode", default="max-autotune-no-cudagraphs", help="diffusers: torch.compile mode")
    ap.add_argument("--engine-args", nargs=argparse.REMAINDER, default=[], help="xdit: extra runner flags")
    ap.add_argument("--engine-config", default="", help="lightx2v: its config json")
    ap.add_argument("--save-dir", default="", help="save the PNGs of the latency run here (for PSNR checks)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--vram-gpu", default="0", help="GPU index polled with nvidia-smi ('' to disable)")
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>_latency.json / <prefix>_throughput.json")
    args = ap.parse_args()

    prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts in {args.prompts}")
    print(f"=== {args.tag}: engine={args.engine} model={args.model} size={args.size} steps={args.steps} "
          f"compile={args.compile} modes={args.modes}", flush=True)
    t_load = time.perf_counter()
    engine = ENGINES[args.engine](args)
    print(f"loaded {engine.version} in {time.perf_counter() - t_load:.1f}s", flush=True)
    common = {
        "tag": args.tag, "url": f"python-api:{args.engine}", "model": args.model, "size": args.size,
        "steps": args.steps, "guidance": args.guidance, "seed": args.seed, "prompts_file": args.prompts,
        "num_prompts": len(prompts), "output_format": "png", "engine": args.engine, "engine_version": engine.version,
        "compile": args.compile, "compile_mode": args.compile_mode if args.compile else None,
        "engine_args": args.engine_args, "engine_config": args.engine_config, "batching": engine.batching,
        "generator": "cpu, seeded per image" if args.engine != "lightx2v" else "engine seed",
    }
    try:
        for mode in args.modes:
            if mode == "latency":
                result = run_latency(engine, args, prompts)
            else:
                throughput_args = argparse.Namespace(**{**vars(args), "n": args.n_throughput})
                result = run_throughput(engine, throughput_args, prompts)
            result.update(common)
            result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            out = Path(f"{args.out_prefix}_{mode}.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2))
            print(f"wrote {out}", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
