"""One kernel search on this process's GPU.

    python -m benchmark.kernels.search --spec rms_norm [--families registry,triton] [--budget 8]
        [--shapes shapes/qwen3_5.json] [--store DIR] [--table OUT.md] [--json OUT.json] [--job ID]

For each shape and dtype of the spec, every candidate (built once per run)
is checked against the reference; those that pass are timed together in
captured graphs (``bench``); the record goes to the results store and a row
to the table: the fastest passing candidate, its time and percent of the
roofline floor, beside the baseline (the shipped op, else torch). Earlier
winners in the store are tried first. ``--list`` prints the candidates and
exits.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import os
import subprocess
import sys
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch

from benchmark.kernels.bench import Bench, Timing
from benchmark.kernels.candidates import Candidate
from benchmark.kernels.roofline import roofline
from benchmark.kernels.spec import KernelSpec, load_shapes
from benchmark.kernels.store import DEFAULT_ROOT, ResultsStore, best_name, shape_key, stack_key
from mstar.kernels.numerics import as_tensors, compare, reference_outputs
from mstar.platforms import current_platform
from mstar.platforms.facts import DeviceFacts

logger = logging.getLogger(__name__)
BASELINES = ("registry", "torch")


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _clone(args: Sequence[Any]) -> list[Any]:
    return [a.clone() if isinstance(a, torch.Tensor) else a for a in args]


def references(spec: KernelSpec, args: Sequence[Any]) -> list[torch.Tensor]:
    """The reference's outputs (and mutated arguments) for ``args``."""
    if not spec.oracle_on_device:
        return reference_outputs(spec.reference, args, spec.mutated)
    upcast = [a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in _clone(args)]
    result = spec.reference(*upcast)
    return [*as_tensors(result), *(upcast[i] for i in spec.mutated)]


def check(spec: KernelSpec, fn: Callable[..., Any], args: Sequence[Any], expected) -> dict[str, Any]:
    """The oracle's verdict on ``fn``: every output within the spec's tolerance."""
    try:
        run_args = _clone(args)
        got = [*as_tensors(fn(*run_args)), *(run_args[i] for i in spec.mutated)]
        if len(got) != len(expected):
            return {"ok": False, "error": f"{len(got)} outputs, the reference has {len(expected)}"}
        results = [compare(out, ref, spec.tolerance) for out, ref in zip(got, expected, strict=True)]
    except Exception as exc:  # noqa: BLE001  a candidate that cannot run fails the oracle
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    worst = max(results, key=lambda r: (not r.ok, r.max_ulps))
    return {"ok": all(r.ok for r in results), "summary": worst.summary(), "max_ulps": worst.max_ulps,
            "rel_norm": worst.rel_norm, "mismatched": worst.mismatched}


class Built(NamedTuple):
    """A candidate after its build: the callable, or why there is none."""
    candidate: Candidate
    fn: Callable[..., Any] | None
    error: str | None


def build(candidates: Sequence[Candidate]) -> dict[str, Built]:
    built = {}
    for cand in candidates:
        try:
            built[cand.name] = Built(cand, cand.build(), None)
        except Exception as exc:  # noqa: BLE001  recorded for every shape
            built[cand.name] = Built(cand, None, f"{type(exc).__name__}: {exc}"[:500])
            logger.warning("candidate %s did not build: %s", cand.name, built[cand.name].error)
    return built


def search_shape(
    spec: KernelSpec, shape, dtype, built, bench: Bench, facts: DeviceFacts, node_us: float | None,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Check and time every built candidate on one shape; the store record."""
    device = bench.device
    # the same inputs for a shape in every run
    base_seed = zlib.crc32(json.dumps([spec.name, dict(shape)], sort_keys=True).encode())

    def make(i: int) -> tuple:
        generator = torch.Generator(device=device).manual_seed(base_seed + i)
        return spec.inputs(shape, dtype, device, generator)

    arg_sets = bench.operand_sets(make)
    expected = references(spec, arg_sets[0])
    line = roofline(spec.bytes_moved(shape, dtype), spec.flops(shape, dtype), dtype, facts)
    record_cands: dict[str, dict[str, Any]] = {}
    passing: dict[str, Callable] = {}
    for name, (cand, fn, error) in built.items():
        entry: dict[str, Any] = {"family": cand.family, "knobs": dict(cand.knobs), **meta}
        if fn is None:
            record_cands[name] = {**entry, "error": f"build: {error}"}
            continue
        entry["oracle"] = check(spec, fn, arg_sets[0], expected)
        record_cands[name] = entry
        if entry["oracle"]["ok"]:
            passing[name] = fn
    timings = bench.time(passing, arg_sets, capture=spec.capture) if passing else {}
    for name, timing in timings.items():
        entry = record_cands[name]
        if not isinstance(timing, Timing):
            entry["error"] = f"timing: {timing}"
            continue
        entry.update(us=timing.us, runs_us=timing.runs_us, captured=timing.captured, calls=timing.calls,
                     copies=timing.copies, note=timing.note, pct_of_floor=line.pct_of_floor(timing.us),
                     gbps=line.bytes_moved / timing.us / 1e3, tflops=line.flops / timing.us / 1e6)
        cand, fn = built[name].candidate, built[name].fn
        if cand.isa is not None:
            try:
                entry["isa"] = cand.isa(fn, arg_sets[0])
            except Exception as exc:  # noqa: BLE001
                entry["isa"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return {
        "kernel": spec.name, "shape": dict(shape), "dtype": str(dtype).removeprefix("torch."),
        "shape_class": spec.shape_class(shape), "arch": facts.arch, "product": facts.product,
        "device": facts.name, "bytes": line.bytes_moved, "flops": line.flops, "floor_us": line.floor_us,
        "bound": line.bound, "node_us": node_us, "candidates": record_cands, "best": best_name(record_cands),
    }


def summarize(record: dict[str, Any]) -> dict[str, Any]:
    """One table row from a record."""
    cands = record["candidates"]
    baseline = next((b for b in BASELINES if isinstance(cands.get(b, {}).get("us"), float)), None)
    best = record["best"]
    row = {"shape": shape_key(record["shape"], getattr(torch, record["dtype"])), "floor_us": record["floor_us"],
           "node_us": record["node_us"], "baseline": baseline, "best": best,
           "failed": sorted(n for n, c in cands.items() if not c.get("oracle", {}).get("ok"))}
    row["baseline_us"] = cands[baseline]["us"] if baseline else None
    row["best_us"] = cands[best]["us"] if best else None
    row["best_pct_of_floor"] = cands[best].get("pct_of_floor") if best else None
    if row["baseline_us"] and row["best_us"]:
        row["speedup"] = row["baseline_us"] / row["best_us"]
    return row


def _fmt(value, spec=".2f") -> str:
    return "-" if value is None else format(value, spec)


def table(rows: Sequence[dict[str, Any]], title: str) -> str:
    lines = [title, "",
             "| shape | floor us | node us | baseline | baseline us | best | best us | x baseline | % of floor |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['shape']} | {_fmt(row['floor_us'], '.3f')} | {_fmt(row['node_us'])} | {row['baseline'] or '-'} "
            f"| {_fmt(row['baseline_us'])} | {row['best'] or '-'} | {_fmt(row['best_us'])} "
            f"| {_fmt(row.get('speedup'))} | {_fmt(row['best_pct_of_floor'], '.1f')} |"
        )
    return "\n".join(lines) + "\n"


def run(
    spec: KernelSpec, *, families: Sequence[str] | None = None, budget: int | None = None,
    store: ResultsStore | None = None, bench: Bench | None = None, facts: DeviceFacts | None = None,
    meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The search; returns the records written (or that would be, without a store)."""
    facts = facts or current_platform.device_facts()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bench = bench or Bench(device, rotate_bytes=2 * facts.mall_bytes if facts.mall_bytes else None)
    seeds = {}
    if store is not None:
        for family in ("triton", "hip"):
            seeds[family] = store.seeds(facts, spec.name, family)
    cands = [c for c in spec.candidates(facts, budget=budget, seeds=seeds)
             if families is None or c.family in families]
    built = build(cands)
    node_us = bench.node_floor_us()
    meta = {"time": datetime.datetime.now().isoformat(timespec="seconds"), **(meta or {})}
    records = []
    for dtype in spec.dtypes:
        for shape in spec.shapes:
            record = search_shape(spec, shape, dtype, built, bench, facts, node_us, meta)
            if store is not None:
                store.write(facts, spec.name, shape, dtype, record)
            records.append(record)
            row = summarize(record)
            logger.info("%s: best %s %s us (baseline %s %s us)", row["shape"], row["best"],
                        _fmt(row["best_us"]), row["baseline"], _fmt(row["baseline_us"]))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return records


def main(argv: Sequence[str] | None = None) -> int:
    from benchmark.kernels.specs import SPECS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, choices=sorted(SPECS))
    parser.add_argument("--shapes", help="a harvested shape file (trace/harvest.py) to take the shapes from")
    parser.add_argument("--max-shapes", type=int, help="the first N shapes only")
    parser.add_argument("--families", help="comma-separated candidate families (default: all)")
    parser.add_argument("--budget", type=int, default=8, help="knob points per searched family")
    parser.add_argument("--calls", type=int, default=256, help="calls per captured graph")
    parser.add_argument("--rounds", type=int, default=3, help="interleaved replays per candidate")
    parser.add_argument("--store", default=str(DEFAULT_ROOT), help="results store root ('' to write none)")
    parser.add_argument("--table", help="also write the markdown table here")
    parser.add_argument("--json", help="also write the records here")
    parser.add_argument("--job", default=os.environ.get("SLURM_JOB_ID"), help="recorded with every result")
    parser.add_argument("--list", action="store_true", help="print the candidates and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    shapes = load_shapes(args.shapes, args.spec) if args.shapes else None
    spec = SPECS[args.spec](shapes)
    if args.max_shapes:
        spec = dataclasses.replace(spec, shapes=spec.shapes[: args.max_shapes])
    families = args.families.split(",") if args.families else None
    facts = current_platform.device_facts()
    if args.list:
        for cand in spec.candidates(facts, budget=args.budget):
            if families is None or cand.family in families:
                print(cand.name)
        return 0
    # Inductor's candidate compiles once per shape
    for limit in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, limit):
            setattr(torch._dynamo.config, limit, 4096)
    store = ResultsStore(args.store) if args.store else None
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bench = Bench(device, calls=args.calls, rounds=args.rounds,
                  rotate_bytes=2 * facts.mall_bytes if facts.mall_bytes else None)
    meta = {"job": args.job, "sha": git_sha()}
    records = run(spec, families=families, budget=args.budget, store=store, bench=bench, facts=facts, meta=meta)
    title = (f"{spec.name} on {facts.name} ({facts.arch}, {facts.product}), {stack_key()}, "
             f"job {args.job}, sha {meta['sha']}")
    text = table([summarize(r) for r in records], title)
    print(text)
    if args.table:
        Path(args.table).write_text(text)
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
