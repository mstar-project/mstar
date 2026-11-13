"""Run the step cache's methods against dense and report speed against quality.

    python -m benchmark.step_cache.run --backend inproc --model wan22_ti2v_5b \\
        --size 832x480 --frames 33 --steps 30 --prompts 16 --out runs/wan5b

Dense runs first (twice on the first two prompts, for the run-to-run floor), then every method on every prompt;
each clip is scored against the dense clip of its prompt and seed and written to ``records.jsonl`` as it finishes, so
a killed run picks up with ``--resume``. The run ends with ``report.md``, ``summary.json`` and ``pareto.svg``.

``--ci`` is the regression mode: three prompts on a short schedule, dense, the default and a policy that skips
nothing. The last must be bit-identical to dense, and the default must hold the PSNR, SSIM and speedup of
``ci_baseline/<model>.json`` within its tolerances; the exit status is 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from benchmark.step_cache import methods as method_grid
from benchmark.step_cache.backends.base import BACKENDS, Backend, Generated, backend_class
from benchmark.step_cache.methods import DENSE, NO_SKIP, Method
from benchmark.step_cache.metrics import CLIP_MODEL, Quality, Scorer, frames_md5, temporal_flicker
from benchmark.step_cache.pareto import METRICS, Point
from benchmark.step_cache.prompts import CI_PROMPTS, Prompt, take
from benchmark.step_cache.report import RunConfig, append_record, load_records, write_report

BASELINE_DIR = Path(__file__).parent / "ci_baseline"
# what a method may lose against its baseline before the regression mode fails it: dB, SSIM, a share of the speedup
TOLERANCE = {"psnr": 0.5, "ssim": 0.01, "speedup": 0.1}
# (size, frames, steps) of a sweep and of the regression mode; MiniMax-H3 takes no clip under five seconds
_SHAPES = {
    "minimax_h3": (("832x480", 124, 30), ("832x480", 124, 20)),
    "tiny": (("64x64", 17, 8), ("64x64", 17, 8)),
    None: (("832x480", 33, 30), ("832x480", 17, 20)),
}


def _dense_path(out_dir: Path, prompt: Prompt) -> Path:
    return out_dir / "dense" / f"p{prompt.index}.pt"


def _record(kind: str, method: Method, prompt: Prompt, generated: Generated, quality: Quality | None,
            **more: Any) -> dict[str, Any]:
    record = {
        "kind": kind, "method": method.name, "label": method.label, "step_cache": method.step_cache,
        "is_default": method.is_default, "prompt_index": prompt.index, "prompt": prompt.text, "seed": prompt.seed,
        "seconds": generated.seconds, "dit_seconds": generated.dit_seconds,
        "computed_steps": generated.computed_steps, "branch_forwards": generated.branch_forwards,
        "frames": int(generated.frames.shape[0]), "frames_md5": frames_md5(generated.frames),
        "extra": generated.extra or {},
    }
    if quality is not None:
        record.update(quality.as_dict())
    record.update(more)
    return record


def run_methods(
    backend: Backend, methods: Sequence[Method], prompts: Sequence[Prompt], config: RunConfig, out_dir: str | Path,
    scorer: Scorer, *, resume: bool = False, floor_prompts: int = 2, log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Generate and score every (method, prompt) the records in ``out_dir`` don't hold yet, dense first, and return
    all of the run's records. A method with nothing left to do is never configured, so a resumed run starts no
    server for it."""
    out_dir = Path(out_dir)
    (out_dir / "dense").mkdir(parents=True, exist_ok=True)
    shape = {"size": config.size, "frames": config.frames, "steps": config.steps}
    stamp = {"config": config._asdict()}
    stamp_path = out_dir / "run.json"
    records = load_records(out_dir)
    if records or stamp_path.exists():
        if not resume:
            raise SystemExit(f"{out_dir} already holds a run; pass --resume to continue it, or a new --out")
        if stamp_path.exists() and json.loads(stamp_path.read_text()) != stamp:
            raise SystemExit(f"{out_dir} holds a run of another config ({stamp_path.read_text().strip()})")
    stamp_path.write_text(json.dumps(stamp))
    # a name's clips are reused, so it has to mean the same setting as when they were made
    recorded = {r["method"]: r["step_cache"] for r in records}
    for method in methods:
        if method.name in recorded and recorded[method.name] != json.loads(json.dumps(method.step_cache)):
            raise SystemExit(
                f"{out_dir} has clips of a method named {method.name!r} with another step_cache "
                f"({json.dumps(recorded[method.name])}); rename the new one, or use a new --out"
            )
    done = {(r["kind"], r["method"], r["prompt_index"]) for r in records}

    def keep(record: dict[str, Any]) -> None:
        records.append(record)
        append_record(out_dir, record)
        done.add((record["kind"], record["method"], record["prompt_index"]))

    configured: list[Method | None] = [None]

    def generate(method: Method, prompt: Prompt) -> Generated:
        if configured[0] != method:
            log(f"[{method.name}] {method.label}")
            backend.configure(method)
            configured[0] = method
        generated = backend.generate(prompt.text, prompt.seed, **shape)
        log(f"[{method.name}] p{prompt.index} {generated.seconds:.2f} s")
        return generated

    dense = next((m for m in methods if m.step_cache is None), DENSE)
    for prompt in prompts:
        path = _dense_path(out_dir, prompt)
        if ("dense", dense.name, prompt.index) in done and path.exists():
            continue
        generated = generate(dense, prompt)
        torch.save(generated.frames, path)
        keep(_record(
            "dense", dense, prompt, generated, None, clip=scorer.clip_score(generated.frames, prompt.text),
            flicker=temporal_flicker(generated.frames),
        ))
    dense_clip = {r["prompt_index"]: r.get("clip") for r in records if r["kind"] == "dense"}

    def score(kind: str, method: Method, prompt: Prompt) -> None:
        generated = generate(method, prompt)
        reference = torch.load(_dense_path(out_dir, prompt))
        quality = scorer.compare(reference, generated.frames, prompt.text, clip_ref=dense_clip.get(prompt.index))
        keep(_record(kind, method, prompt, generated, quality))

    for prompt in prompts[:floor_prompts]:
        if ("floor", dense.name, prompt.index) not in done:
            score("floor", dense, prompt)
    for method in methods:
        if method.step_cache is None:
            continue
        for prompt in prompts:
            if ("method", method.name, prompt.index) not in done:
                score("method", method, prompt)
    return records


def identity_failures(records: Sequence[Mapping[str, Any]], method: Method = NO_SKIP) -> list[str]:
    """The prompts where ``method``, which skips nothing, did not come out bit-identical to dense."""
    rows = [r for r in records if r["kind"] == "method" and r["method"] == method.name]
    if not rows:
        return [f"{method.name}: no clip to compare with dense"]
    return [
        f"{method.name} differs from dense on prompt {r['prompt_index']} (PSNR {r['psnr']:.2f} dB): the cache's "
        f"compute path must change no output"
        for r in rows if not r["identical"]
    ]


def baseline_failures(points: Sequence[Point], baseline: Mapping[str, Any]) -> list[str]:
    """The methods that fell below ``baseline`` by more than its tolerances."""
    tolerance = {**TOLERANCE, **baseline.get("tolerance", {})}
    by_name = {p.name: p for p in points}
    failures = []
    for name, floor in baseline["methods"].items():
        point = by_name.get(name)
        if point is None:
            failures.append(f"{name}: in the baseline, not in this run")
            continue
        checks = (
            ("psnr", point.psnr, floor["psnr"] - tolerance["psnr"]),
            ("ssim", point.ssim, floor["ssim"] - tolerance["ssim"]),
            ("speedup", point.speedup, floor["speedup"] * (1 - tolerance["speedup"])),
        )
        for key, value, limit in checks:
            if value is None:
                failures.append(f"{name}: no {key} in this run (baseline {floor[key]:.3f})")
            elif value < limit:
                failures.append(f"{name}: {key} {value:.3f} is under {limit:.3f} (baseline {floor[key]:.3f})")
    return failures


def make_baseline(points: Sequence[Point], config: RunConfig, prompts: Sequence[Prompt], env: Mapping[str, Any],
                  skip: Sequence[str] = (NO_SKIP.name,)) -> dict[str, Any]:
    return {
        "config": config._asdict(), "prompts": [p.index for p in prompts], "env": dict(env),
        "tolerance": dict(TOLERANCE),
        "methods": {
            p.name: {"psnr": round(p.psnr, 3), "ssim": round(p.ssim, 4), "speedup": round(p.speedup, 3)}
            for p in points if p.name not in skip
        },
    }


def _parser(backend: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default=backend)
    parser.add_argument("--model", choices=method_grid.MODELS, required=True)
    parser.add_argument("--out", required=True, help="the run's directory: records, dense clips, the report")
    parser.add_argument("--size", help="WxH (default 832x480; the toy model's is 64x64)")
    parser.add_argument("--frames", type=int, help="default 33, or 17 with --ci (MiniMax-H3: 124)")
    parser.add_argument("--steps", type=int, help="default 30, or 20 with --ci")
    parser.add_argument("--prompts", type=int, default=16, help="how many of the fixed prompts (ignored with --ci)")
    parser.add_argument("--methods", action="append", default=[],
                        help="keep the grid's methods whose name starts with one of these, comma-separated "
                             "(repeatable); 'none' keeps only dense")
    parser.add_argument("--methods-file",
                        help="a JSON list of {name, step_cache, label?, is_default?} run after the grid's methods")
    parser.add_argument("--list-methods", action="store_true", help="print the model's grid and exit")
    parser.add_argument("--default", help="the method the frontier check is about (default: the grid's)")
    parser.add_argument("--mag-ratios", help="the model's calibrated table the MagCache methods read")
    parser.add_argument("--metric", choices=sorted(METRICS), default="psnr", help="the frontier's quality axis")
    parser.add_argument("--ssim-window", choices=("gaussian", "uniform"), default="gaussian",
                        help="uniform is scikit-image's default, which earlier quality tables used")
    parser.add_argument("--metrics-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-optional-metrics", action="store_true", help="skip LPIPS and the CLIP score")
    parser.add_argument("--clip-model", default=CLIP_MODEL,
                        help="the CLIP checkpoint: a hub id, or a snapshot directory when the hub cache is the model's")
    parser.add_argument("--floor-prompts", type=int, default=2, help="prompts whose dense clip is generated twice")
    parser.add_argument("--resume", action="store_true", help="continue the run in --out")
    parser.add_argument("--strict", action="store_true", help="exit 1 when the default fails the frontier check")
    parser.add_argument("--ci", action="store_true", help="the regression mode (see above)")
    parser.add_argument("--baseline", help="the regression mode's baseline json (default: ci_baseline/<model>.json)")
    parser.add_argument("--write-baseline", action="store_true", help="with --ci: write the baseline from this run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # a first pass for the two flags the rest depends on: the backend adds its own, and listing needs no run
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--backend", default="server")
    first.add_argument("--model")
    first.add_argument("--mag-ratios")
    first.add_argument("--list-methods", action="store_true")
    early = first.parse_known_args(argv)[0]
    chosen = early.backend
    parser = _parser(chosen)
    if early.list_methods:
        if early.model is None:
            parser.error("--list-methods needs --model")
        for method in method_grid.grid(early.model, early.mag_ratios):
            print(f"{method.name}{' (default)' if method.is_default else ''}: {method.label}")
        return 0
    cls = backend_class(chosen)
    if hasattr(cls, "add_arguments"):
        cls.add_arguments(parser)
    args = parser.parse_args(argv)

    grid = method_grid.grid(args.model, args.mag_ratios)
    extra = method_grid.load_methods(args.methods_file) if args.methods_file else ()
    default = method_grid.default_of(method_grid.combine(grid, extra), args.default)
    sweep, ci = _SHAPES.get(args.model, _SHAPES[None])
    size, frames, steps = (given or preset for given, preset in zip(
        (args.size, args.frames, args.steps), ci if args.ci else sweep, strict=True))
    config = RunConfig(model=args.model, backend=chosen, size=size, frames=frames, steps=steps)
    if args.ci:
        methods, prompts = (DENSE, default, NO_SKIP), CI_PROMPTS
    else:
        prefixes = [p for group in args.methods for p in group.split(",") if p]
        methods = method_grid.combine(method_grid.select(grid, prefixes), extra)
        prompts = take(args.prompts)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = Scorer.build(
        args.metrics_device, args.ssim_window, optional=not args.no_optional_metrics, clip_model=args.clip_model,
    )
    backend = cls.from_args(args)
    try:
        env = backend.describe()
        records = run_methods(
            backend, methods, prompts, config, out_dir, scorer, resume=args.resume,
            floor_prompts=args.floor_prompts,
        )
    finally:
        backend.close()
    report = write_report(
        out_dir, records, env=env, config=config, metric=args.metric, default=default.name, notes=scorer.notes,
    )
    print(f"report: {report.path}")
    print(f"default: {'PASS' if report.verdict.ok else 'FAIL'}: {report.verdict.reason}")
    if not args.ci:
        return 1 if args.strict and not report.verdict.ok else 0

    failures = identity_failures(records)
    baseline_path = Path(args.baseline) if args.baseline else BASELINE_DIR / f"{args.model}.json"
    if args.write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(make_baseline(report.points, config, prompts, env), indent=2) + "\n")
        print(f"baseline written: {baseline_path}")
    elif not baseline_path.exists():
        failures.append(f"no baseline at {baseline_path}: run once with --write-baseline on a tree known good")
    else:
        baseline = json.loads(baseline_path.read_text())
        if baseline["config"] != config._asdict() or baseline["prompts"] != [p.index for p in prompts]:
            failures.append(f"{baseline_path} is for another config ({baseline['config']}); this run is {config}")
        else:
            failures += baseline_failures(report.points, baseline)
    for failure in failures:
        print(f"FAIL {failure}")
    print("regression mode:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
