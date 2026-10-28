#!/usr/bin/env python3
"""Sweep a kernel op's candidate configs at a model's shapes on this GPU, and write the winners
to the op's tuned-config table.

    python -m benchmark.kernels.sweep moe_gemm --dtype bfloat16 \\
        --shape E=128,N=1536,K=2048,top_k=8 --note "Qwen3-Omni thinker" --out ~/kernel-tables

A server logs the exact command for each shape it ran without a table ("kernel table: no
moe_gemm ... To tune it: ..."). Each bucket prints what runs now (``current``: the table's row,
or the default), the op's default, and the best candidate, in microseconds per call.

``--out DIR`` writes ``DIR/<op>/<GPU>.json``, which a server reads with
``MSTAR_KERNEL_TABLES=DIR``: the way to A/B a sweep end to end before committing it.
``--write`` writes into the package's tables (``mstar/kernels/tables``), to commit.
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
from pathlib import Path

import torch

from mstar.kernels import device_key, get_op, op_names, table_dirs
from mstar.kernels.registry import reset_lookups
from mstar.kernels.table import TABLE_DIR, TableEntry, dtype_name, file_stem, load_table, write_entry

# op -> the class that sweeps it ("module:Class")
OP_SWEEPS = {
    "moe_gemm": "benchmark.kernels.moe_gemm:MoeGemmSweep",
}


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
        import triton

        prov["triton"] = triton.__version__
    except ImportError:
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
        print(f"{op.name}: shape {op.shape}, rows by {op.bucket or '-'}, backends "
              f"{[b.name for b in op.backends]}. {op.doc}")
        for directory in table_dirs():
            for path in sorted((directory / name).glob("*.json")):
                for e in load_table(path, name):
                    print(f"  {path}: {e.dtype} {dict(e.shape)} ({e.backend}), {len(e.rows)} rows, "
                          f"swept on {e.provenance.get('device')} {e.provenance.get('date')}")


def _speedup(a: float, b: float) -> str:
    return "-" if math.isnan(a) or math.isnan(b) or b <= 0 else f"{a / b:.2f}x"


def sweep(op_name: str, shape: dict[str, int], dtype: torch.dtype, buckets, quick: bool, note: str | None,
          out_dir: Path | None, report: list, min_gain: float) -> None:
    op = get_op(op_name)
    cls = _sweeper(op_name)
    device = torch.device("cuda", torch.cuda.current_device())
    dev = device_key(device)
    reason = op.backend(cls.backend).unavailable(device, dtype, shape)
    if reason:
        raise SystemExit(f"{op_name} backend {cls.backend} can't run {dtype} {shape} here: {reason}")
    op.shape_key(shape)
    sweeper = cls(shape, dtype, device, quick=quick)
    buckets = tuple(buckets or cls.default_buckets)
    print(f"# {op_name} {dtype_name(dtype)} {op.format_shape(shape)} on {dev.name} ({dev.arch}), "
          f"{cls.backend}, buckets {list(buckets)}", flush=True)
    print(f"{op.bucket or 'bucket':>8} {'current_us':>11} {'default_us':>11} {'best_us':>9} "
          f"{'vs current':>10} {'vs default':>10}  row", flush=True)
    rows, measured = [], {}
    for bucket in buckets:
        r = sweeper.run_bucket(bucket)
        # Keep what runs now unless the best beats it by min_gain: near-ties would otherwise flip
        # between sweeps and every re-sweep would rewrite rows that measure the same.
        better = math.isnan(r.current_us) or r.current_us > r.best_us * (1 + min_gain)
        rows.append((bucket, r.row if better else r.current_row))
        us = r.best_us if better else r.current_us
        measured[bucket] = {"us": round(us, 1), "default_us": round(r.default_us, 1)}
        report.append({"op": op_name, "dtype": dtype_name(dtype), "shape": shape, "bucket": bucket,
                       "current_us": r.current_us, "default_us": r.default_us, "best_us": r.best_us,
                       "row": r.row, "timed": r.timed, "rejected": r.rejected})
        print(f"{bucket:>8} {r.current_us:>11.1f} {r.default_us:>11.1f} {r.best_us:>9.1f} "
              f"{_speedup(r.current_us, r.best_us):>10} {_speedup(r.default_us, r.best_us):>10}  "
              f"{json.dumps(r.row)}", flush=True)
    if out_dir is None:
        return
    command = op.sweep_command(dtype, shape)
    if buckets != tuple(cls.default_buckets):
        command += f" --buckets {','.join(map(str, buckets))}"
    entry = TableEntry(op=op_name, dtype=dtype_name(dtype), shape=op.shape_key(shape), backend=cls.backend,
                       rows=tuple(rows), provenance=_provenance(dev, command, note), source="", measured=measured)
    path = out_dir / op_name / f"{file_stem(dev.name)}.json"
    write_entry(path, entry)
    reset_lookups()
    print(f"# wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("op", nargs="?", help=f"one of {sorted(OP_SWEEPS)}")
    parser.add_argument("--shape", action="append", default=[], help="NAME=INT,... in the op's shape keys")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--buckets", help="comma-separated; default: the op's")
    parser.add_argument("--quick", action="store_true", help="a smaller candidate grid")
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
    buckets = [int(b) for b in args.buckets.split(",")] if args.buckets else None
    out_dir = TABLE_DIR if args.write else args.out
    report: list = []
    for text in args.shape:
        sweep(args.op, _parse_shape(text), dtype, buckets, args.quick, args.note, out_dir, report, args.min_gain)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
