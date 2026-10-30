#!/usr/bin/env python3
"""Sweep a kernel op's candidate configs at a model's shapes on this GPU: check each against a
reference output, time the ones that pass, and write the winners to the op's table.

    python -m benchmark.kernels.sweep moe_gemm --dtype bfloat16 \\
        --shape E=128,N=1536,K=2048,top_k=8 --note "Qwen3-Omni thinker" --out ~/kernel-tables

A server logs the exact command for each shape it ran without a table ("kernel table: no
moe_gemm ... To tune it: ..."). Each bucket prints what runs now (``current``: the table's row,
or the built-in config), the built-in config, and the best candidate, in microseconds per call.
A candidate whose output sits outside its implementation's tolerance of the reference output is
never timed, so never written, and neither is a current row that fails the same check. A bucket
keeps its current row unless the best candidate beats it by ``--min-gain``.

``--out DIR`` writes ``DIR/<op>/<GPU>.json``, which a server reads with
``MSTAR_KERNEL_TABLES=DIR``: the way to A/B a sweep end to end before committing it.
``--write`` writes into the package's tables (``mstar/kernels/tables``), to commit.
``--fresh`` decides from scratch: what runs now is taken to be the built-in config.
``--list`` prints the ops and the shapes each table file covers.
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

from mstar.kernels import Choice, device_key, get_op, op_names, table_dirs
from mstar.kernels.registry import KernelOp, describe_rows, reset_lookups
from mstar.kernels.table import TABLE_DIR, Rows, TableEntry, dtype_name, file_stem, load_table, write_entry

# op -> the class that sweeps it ("module:Class")
OP_SWEEPS = {
    "moe_gemm": "benchmark.kernels.moe_gemm:MoeGemmSweep",
    "moe_decode": "benchmark.kernels.moe_decode:MoeDecodeSweep",
}


@dataclass
class BucketResult:
    """One bucket's measurements, in microseconds per call; NaN where a choice failed the
    reference check or would not run."""

    bucket: int
    current: Choice
    current_us: float
    default_us: float
    best: Choice | None  # the fastest candidate within tolerance
    best_us: float
    impl_us: dict[str, float] = field(default_factory=dict)  # each implementation's fastest
    timed: int = 0
    rejected: int = 0


def pick_row(r: BucketResult, min_gain: float) -> tuple[Choice, float] | None:
    """The row a bucket gets: what runs now unless the best candidate beats it by ``min_gain``
    (near-ties would otherwise flip between sweeps, and every re-sweep would rewrite rows that
    measure the same), the best when what runs now failed the reference check, and nothing when
    no choice passed it."""
    if not math.isnan(r.current_us) and (r.best is None or r.current_us <= r.best_us * (1 + min_gain)):
        return r.current, r.current_us
    if r.best is not None:
        return r.best, r.best_us
    return None


def merge_runs(chosen: list[tuple[int, Choice]]) -> list[tuple[int | None, Choice]]:
    """Buckets' winners as rows: a run of buckets with one winner is one row, keyed by its
    largest bucket."""
    rows: list[tuple[int | None, Choice]] = []
    for bucket, choice in chosen:
        if rows and rows[-1][1] == choice:
            rows[-1] = (bucket, choice)
        else:
            rows.append((bucket, choice))
    return rows


def assemble_rows(chosen: list[tuple[int, Choice]]) -> Rows:
    """Buckets' winners as table rows, the last one ``"*"``: counts above the last bucket take
    its winner."""
    rows = merge_runs(chosen)
    rows[-1] = (None, rows[-1][1])
    return tuple(rows)


def _sweeper(op: str):
    if op not in OP_SWEEPS:
        raise SystemExit(f"no sweep for op {op!r}; sweeps exist for {sorted(OP_SWEEPS)}")
    module, cls = OP_SWEEPS[op].split(":")
    return getattr(importlib.import_module(module), cls)


def _parse_shape(text: str) -> dict[str, int]:
    shape = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        if not value:
            raise SystemExit(f"--shape takes NAME=INT pairs separated by commas; got {text!r}")
        shape[name.strip()] = int(value)
    return shape


def _mstar_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=9", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=False)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    commit = root / "COMMIT"
    return commit.read_text().strip() if commit.is_file() else None


def _provenance(device, command: str, note: str | None) -> dict:
    prov = {
        "device": device.name,
        "arch": device.arch,
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d"),
        "harness": command,
        "torch": torch.__version__,
    }
    try:
        prov["triton"] = importlib.import_module("triton").__version__
    except Exception:  # noqa: BLE001 - not installed, or fails to import here
        pass
    commit = _mstar_commit()
    if commit:
        prov["mstar"] = commit
    if note:
        prov["note"] = note
    return prov


def list_tables() -> None:
    for name in op_names():
        op = get_op(name)
        print(f"{op.name}: shape {op.shape}, rows by {op.bucket or '-'}, implementations "
              f"{[impl.name for impl in op.impls]}. {op.doc}")
        for directory in table_dirs():
            for path in sorted((directory / name).glob("*.json")):
                for e in load_table(path, name):
                    print(f"  {path}: {e.dtype} {dict(e.shape)}: {describe_rows(e.rows)}; "
                          f"swept on {e.provenance.get('device')} {e.provenance.get('date')}")


def _us(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.1f}"


def sweep(op_name: str, shape: dict[str, int], dtype: torch.dtype, buckets, quick: bool, note: str | None,
          out_dir: Path | None, report: list, min_gain: float, fresh: bool = False) -> bool:
    """Sweep one shape; False when it found rows it can't write."""
    op: KernelOp = get_op(op_name)
    cls = _sweeper(op_name)
    device = torch.device("cuda", torch.cuda.current_device())
    dev = device_key(device)
    reasons = op.unavailable(op.target(dtype, shape, device))
    if len(reasons) == len(op.impls):
        raise SystemExit(f"{op_name} can't run {dtype_name(dtype)} {shape} here: {reasons}")
    sweeper = cls(shape, dtype, device, quick=quick)
    entry = None if fresh else op.lookup(dtype, shape, device)
    buckets = tuple(buckets or cls.default_buckets)
    print(f"# {op_name} {dtype_name(dtype)} {op.format_shape(shape)} on {dev.name} ({dev.arch}), "
          f"buckets {list(buckets)}", flush=True)
    now = describe_rows(entry.rows) if entry is not None else "built-in configs" + (" (--fresh)" if fresh else "")
    print(f"# now: {now}", flush=True)
    print(f"{op.bucket or 'bucket':>8} {'current':>8} {'default':>8} {'best':>8} {'chosen':>8}  row", flush=True)
    chosen, measured = [], {}
    for bucket in buckets:
        r = sweeper.run_bucket(bucket, None if entry is None else op.row_for(entry.rows, bucket))
        picked = pick_row(r, min_gain)
        if picked is None:
            print(f"{bucket:>8}  no candidate passed the reference check; no row", flush=True)
            continue
        row, us = picked
        chosen.append((bucket, row))
        times = {"us": round(us, 2), "default_us": round(r.default_us, 2),
                 **{f"{n}_us": round(t, 2) for n, t in r.impl_us.items()}}
        measured[bucket] = {k: v for k, v in times.items() if not math.isnan(v)}
        report.append({"op": op_name, "dtype": dtype_name(dtype), "shape": shape, "bucket": bucket,
                       "current": r.current.to_json(), "current_us": r.current_us, "default_us": r.default_us,
                       "best": None if r.best is None else r.best.to_json(), "best_us": r.best_us,
                       "chosen": row.to_json(), "timed": r.timed, "rejected": r.rejected})
        print(f"{bucket:>8} {_us(r.current_us):>8} {_us(r.default_us):>8} {_us(r.best_us):>8} {_us(us):>8}  "
              f"{row.describe()}", flush=True)
    if not chosen:
        print("# no rows: nothing to write", flush=True)
        return True
    command = op.sweep_command(dtype, shape)
    if buckets != tuple(cls.default_buckets):
        command += f" --buckets {','.join(map(str, buckets))}"
    try:
        rows = sweeper.rows(chosen) if hasattr(sweeper, "rows") else assemble_rows(chosen)
        new = TableEntry(op=op_name, dtype=dtype_name(dtype), shape=op.shape_key(shape), rows=rows,
                         provenance=_provenance(dev, command, note), source="this sweep", measured=measured)
        # what a server would refuse at its first call, refused here instead
        op.check_rows(new.rows, new.source)
        sweeper.check(new)
    except ValueError as e:
        print(f"# not written: {e}", flush=True)
        return False
    print(f"# rows: {describe_rows(rows)}", flush=True)
    if out_dir is not None:
        path = out_dir / op_name / f"{file_stem(dev.name)}.json"
        write_entry(path, new)
        reset_lookups()
        print(f"# wrote {path}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("op", nargs="?", help=f"one of {sorted(OP_SWEEPS)}")
    parser.add_argument("--shape", action="append", default=[], help="NAME=INT,... in the op's shape keys")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--buckets", help="comma-separated; default: the op's")
    parser.add_argument("--quick", action="store_true", help="a smaller candidate grid")
    parser.add_argument("--fresh", action="store_true",
                        help="take the built-in config, not the tables, as what runs now")
    parser.add_argument("--note", help="recorded in the entry's provenance, e.g. the model and TP")
    parser.add_argument("--min-gain", type=float, default=0.02,
                        help="replace a bucket's current row only when the best is this much faster (0.02: 2%%)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--out", type=Path, help="write DIR/<op>/<GPU>.json (for MSTAR_KERNEL_TABLES)")
    where.add_argument("--write", action="store_true", help=f"write into {TABLE_DIR}")
    parser.add_argument("--json", type=Path, help="also write every bucket's timings here")
    parser.add_argument("--list", action="store_true", help="list ops and table coverage, then exit")
    args = parser.parse_args()
    if args.list:
        list_tables()
        return
    if not args.op or not args.shape:
        parser.error("give an op and at least one --shape (or --list)")
    if not torch.cuda.is_available():
        raise SystemExit("a sweep times kernels on a GPU; none is visible")
    dtype = getattr(torch, dtype_name(args.dtype))
    buckets = sorted({int(b) for b in args.buckets.split(",")}) if args.buckets else None
    out_dir = TABLE_DIR if args.write else args.out
    report: list = []
    ok = [sweep(args.op, _parse_shape(text), dtype, buckets, args.quick, args.note, out_dir, report,
                args.min_gain, args.fresh) for text in args.shape]
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str) + "\n")
    if not all(ok):
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
