"""fp8_linear table rows for exactly the block-fp8 linear shapes a model dispatches: one sweep command (or its run)
over every shape a server reported without rows, or over a model's per-rank shapes at a TP size.

A server logs each shape its fp8 linears dispatch that no table covers on its GPU (``kernel table: no fp8_linear
... entry for <GPU>; built-in configs. To tune it: <command>``), so booting a model once and reading its log gives
its declared per-rank shapes at that deployment's TP size, whatever the model:

    python -m benchmark.quantization.fp8_rows --from-log server.log --out ~/tables --run
    python -m benchmark.quantization.fp8_rows --set glm53_tp8,qwen3_5_27b_tp4 --out ~/tables   # print the command

The rows land in ``DIR/fp8_linear/<GPU>.json``; a server reads them with ``MSTAR_KERNEL_TABLES=DIR`` (``--write``
instead puts them in the package's tables, to commit).
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

# per-rank (N, K, glu): N per half for glu (a [gate; up] weight's fused SwiGLU)
SETS = {
    # GLM-5.3 (GLM-5.2's geometry) at TP8: q_a + kv_a fused (unsharded), q_b, o, the dense layers' MLP
    "glm53_tp8": [(2624, 6144, 0), (2048, 2048, 0), (6144, 2048, 0), (1536, 6144, 1), (6144, 1536, 0)],
    # Qwen3.5-27B at TP4: qkv, o, the MLP (fused SwiGLU), the delta net's fp8 input and output
    "qwen3_5_27b_tp4": [(3584, 5120, 0), (5120, 1536, 0), (4352, 5120, 1), (5120, 4352, 0), (4096, 5120, 0)],
    # Qwen3.5-27B at TP1 (Qwen3.5-27B-FP8 as stored): qkv, the delta net's fp8 input, the MLP, down, o / delta-net out
    "qwen3_5_27b_tp1": [(14336, 5120, 0), (16384, 5120, 0), (17408, 5120, 1), (5120, 17408, 0), (5120, 6144, 0)],
    # Qwen3.5-397B-A17B at TP4: the delta net's fp8 input, qkv, o / delta-net out, the shared expert (its routed
    # experts are fused_experts_fp8 shapes)
    "qwen3_5_397b_a17b_tp4": [(5120, 4096, 0), (4608, 4096, 0), (4096, 2048, 0), (256, 4096, 1), (4096, 256, 0)],
    # Qwen3-Omni's Thinker attention at TP1 (its experts are moe_decode / fused_experts_fp8 shapes)
    "qwen3_omni_thinker_tp1": [(5120, 2048, 0), (2048, 4096, 0)],
}
_MISSING = re.compile(r"kernel table: no fp8_linear \S+ N=(\d+),K=(\d+),glu=(\d)\b")


def shapes_from_logs(paths) -> list[tuple[int, int, int]]:
    """The fp8_linear shapes a server logged without rows, in first-seen order."""
    seen: dict[tuple[int, int, int], None] = {}
    for path in paths:
        for m in _MISSING.finditer(Path(path).read_text(errors="replace")):
            seen[(int(m[1]), int(m[2]), int(m[3]))] = None
    return list(seen)


def sweep_command(shapes, out: str | None, write: bool, note: str | None, extra=()) -> list[str]:
    cmd = [sys.executable, "-m", "benchmark.kernels.sweep", "fp8_linear", "--dtype", "float8_e4m3fn"]
    for n, k, glu in shapes:
        cmd += ["--shape", f"N={n},K={k},glu={glu}"]
    cmd += ["--write"] if write else ["--out", out] if out else []
    return cmd + (["--note", note] if note else []) + list(extra)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-log", action="append", default=[], help="a server log to read missing shapes from")
    p.add_argument("--set", default="", help=f"named per-rank shape sets: {', '.join(SETS)}")
    p.add_argument("--shape", action="append", default=[], help="N,K,glu (per rank)")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--out", help="write DIR/fp8_linear/<GPU>.json (for MSTAR_KERNEL_TABLES)")
    where.add_argument("--write", action="store_true", help="write into the package's tables")
    p.add_argument("--note", default=None)
    p.add_argument("--quick", action="store_true", help="the sweep's smaller w8a16 tile grid")
    p.add_argument("--buckets", help="token counts to sweep, comma-separated (default: the sweep's)")
    p.add_argument("--run", action="store_true", help="run the sweep here (a GPU), else print the command")
    args = p.parse_args()
    shapes = shapes_from_logs(args.from_log)
    for name in filter(None, args.set.split(",")):
        shapes += [s for s in SETS[name] if s not in shapes]
    for text in args.shape:
        n, k, glu = (int(v) for v in text.split(","))
        if (n, k, glu) not in shapes:
            shapes.append((n, k, glu))
    if not shapes:
        p.error("no shapes: give --from-log, --set or --shape")
    note = args.note or ", ".join(filter(None, [args.set] + [Path(f).name for f in args.from_log]))
    extra = (["--quick"] if args.quick else []) + (["--buckets", args.buckets] if args.buckets else [])
    cmd = sweep_command(shapes, args.out, args.write, note, extra)
    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.run:
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
