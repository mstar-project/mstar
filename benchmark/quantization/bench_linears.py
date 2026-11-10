"""Linear formats against bf16 at a model's per-rank GEMM shapes, each call replayed in a CUDA graph.

    python -m benchmark.quantization.bench_linears --formats fp8_tensor,fp8_channel,fp8_block \\
        --shapes qwen3_5_27b --tp 1 --tokens 1,16,64,256,1024,4096 --out linears.jsonl

One row per (format, shape, tokens): median microseconds of a replay over ``--iters`` calls, the speedup over
F.linear in bf16, and the output's relative L2 distance from it. A format the GPU can't run is skipped with
the reason. Times are the GEMM and the activation quantize together, as a served linear runs them.

``--impls w8a16,flashinfer,cublas`` also times each of fp8_block's kernels on its own (``fp8_block[impl]``
rows; a ``gate_up`` layer also as the fused SwiGLU, ``layer`` ``gate_up_glu``), and ``--table-out DIR`` writes
the fastest per token count as fp8_linear table entries, ``DIR/fp8_linear/<GPU>.json``, for a server to try
with ``MSTAR_KERNEL_TABLES=DIR`` before a table is committed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

# per-rank [N, K] at TP 1 (divide N for column-parallel, K for row-parallel by --tp)
SHAPES = {
    # hidden 5120, q 24 heads x 256 x 2 (gated), kv 4 x 256, inter 17408, delta-net value 6144
    # gdn_in: the delta net's fused input less its bf16 gate rows (q|k|v|z, 2 x 2048 + 2 x 6144)
    "qwen3_5_27b": {"qkv": (14336, 5120, "col"), "o": (5120, 6144, "row"), "gate_up": (34816, 5120, "col"),
                    "down": (5120, 17408, "row"), "gdn_out": (5120, 6144, "row"), "gdn_in": (16384, 5120, "col")},
    "qwen3_5_9b": {"qkv": (6144, 4096, "col"), "o": (4096, 4096, "row"), "gate_up": (24576, 4096, "col"),
                   "down": (4096, 12288, "row")},
    # Qwen3-Omni thinker: attention only (its experts run fused_experts_fp8)
    "qwen3_omni_thinker": {"qkv": (5120, 2048, "col"), "o": (2048, 4096, "row")},
}


def _time(fn, iters: int, repeats: int = 5) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iters)
    return statistics.median(times)


def _forced(impl: str):
    """Run fp8_block's linear on ``impl`` whatever its dispatch would pick."""
    from mstar.utils import dense_fp8

    real = dense_fp8._choose

    class Forced:
        def __enter__(self):
            dense_fp8._choose = lambda *a, **k: (impl, None)

        def __exit__(self, *exc):
            dense_fp8._choose = real

    return Forced()


def _per_impl(layer, fmt, x, weight, stored, ref, base_us, impls, iters) -> list[dict]:
    """fp8_block on each kernel in ``impls``; a gate_up layer also as its fused SwiGLU (glu=1), against bf16's
    GEMM and act-and-mul."""
    n, k = weight.shape
    m = x.shape[0]
    rows = []
    cases = [(layer, n, fmt.linear, ref, base_us)]
    if layer == "gate_up" and fmt.has_glu:
        def bf16_glu():
            gate, up = F.linear(x, weight).chunk(2, dim=-1)
            return F.silu(gate) * up
        cases.append((f"{layer}_glu", n // 2, fmt.glu_linear, bf16_glu(), _time(bf16_glu, iters)))
    for name, n_out, fn, want, bf16_us in cases:
        if name.endswith("_glu"):
            rows.append({"layer": name, "n": n_out, "k": k, "tokens": m, "format": "bf16", "us": round(bf16_us, 2)})
        for impl in impls:
            row = {"layer": name, "n": n_out, "k": k, "tokens": m, "format": f"fp8_block[{impl}]"}
            try:
                with _forced(impl):
                    out = fn(x, *stored)
                    us = _time(lambda: fn(x, *stored), iters)  # noqa: B023
            except Exception as e:  # noqa: BLE001  (an impl this GPU or shape can't run)
                rows.append({**row, "skipped": f"{type(e).__name__}: {e}"[:200]})
                continue
            err = ((out.float() - want.float()).norm() / want.float().norm()).item()
            rows.append({**row, "us": round(us, 2), "speedup": round(bf16_us / us, 3), "rel_l2": round(err, 4)})
    return rows


# a kernel whose output is further than this from bf16's is wrong, not rounded: e4m3 lands 2-4% on a GEMM and
# ~5.5% on a SwiGLU of one with fp8 activations (job 1438), a wrong scale or layout order 1
_WRONG = 0.15


def _write_tables(rows: list[dict], out: Path, impls: list[str], gpu: str, arch: str | None,
                  harness: str = "") -> list[Path]:
    """The fastest measured fp8_block kernel at each measured token count, per (N, K, glu), as fp8_linear table
    entries: a token count between two measured ones takes the larger's row, one above them all the largest's.
    Every measured count gets the fastest of the kernels timed there, the one the built-in rule runs among them,
    so no measured count runs slower than without the table."""
    from types import MappingProxyType

    try:
        import flashinfer
        fi_version = flashinfer.__version__
    except Exception:  # noqa: BLE001
        fi_version = "none"
    from mstar.kernels.table import Choice, TableEntry, file_stem, write_entry
    from mstar.utils.fp8_block import FP8_DTYPE

    timed: dict[tuple[int, int, int], dict[int, dict[str, float]]] = {}
    for row in rows:
        fmt = row["format"]
        if fmt.startswith("fp8_block[") and "us" in row and row.get("rel_l2", 1) < _WRONG:
            glu = int(row["layer"].endswith("_glu"))
            timed.setdefault((row["n"], row["k"], glu), {}).setdefault(row["tokens"], {})[fmt[10:-1]] = row["us"]
        elif fmt == "bf16" and "us" in row:
            glu = int(row["layer"].endswith("_glu"))
            timed.setdefault((row["n"], row["k"], glu), {}).setdefault(row["tokens"], {})["bf16"] = row["us"]
    path = out / "fp8_linear" / f"{file_stem(gpu)}.json"
    written = []
    for (n, k, glu), by_tokens in sorted(timed.items()):
        measured = {m: t for m, t in sorted(by_tokens.items()) if any(i in t for i in impls)}
        if not measured:
            continue
        best = {m: min((i for i in impls if i in t), key=t.get) for m, t in measured.items()}
        buckets = sorted(best)
        table_rows = tuple((m, Choice(best[m])) for m in buckets[:-1]) + ((None, Choice(best[buckets[-1]])),)
        entry = TableEntry(
            op="fp8_linear", dtype=str(FP8_DTYPE).removeprefix("torch."),
            shape=(("N", n), ("K", k), ("glu", glu)), rows=table_rows,
            provenance=MappingProxyType({
                "device": gpu, "arch": arch, "date": datetime.date.today().isoformat(),
                "harness": harness or "benchmark/quantization/bench_linears.py --impls (CUDA-graph replay, one "
                                      "weight copy)",
                "torch": torch.__version__, "flashinfer": fi_version,
                "note": "fastest measured kernel per token count; the largest count's for '*'",
            }),
            source=str(path),
            measured=MappingProxyType({(m if m != buckets[-1] else None): MappingProxyType(dict(t))
                                       for m, t in measured.items()}),
        )
        write_entry(path, entry)
        written.append(path)
    return sorted(set(written))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formats", default="fp8_tensor,fp8_channel,fp8_block")
    parser.add_argument("--shapes", default="qwen3_5_27b", choices=sorted(SHAPES))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--tokens", default="1,16,64,256,1024,4096")
    parser.add_argument("--layers", default="", help="only these of the shape set's layers (comma-separated)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", default=None)
    parser.add_argument("--impls", default="", help="fp8_block's kernels to time one by one (w8a16,flashinfer,cublas)")
    parser.add_argument("--table-out", type=Path, default=None)
    parser.add_argument("--tables-from", type=Path, default=None,
                        help="write --table-out from this --out file's per-kernel rows (no GPU), its gpu and shapes")
    parser.add_argument("--arch", default="sm_90", help="with --tables-from: the measured GPU's architecture")
    args = parser.parse_args()
    impls = [i for i in args.impls.split(",") if i]
    if args.tables_from is not None:
        rows = [json.loads(line) for line in args.tables_from.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r.get("shapes") == args.shapes and r.get("tp") == args.tp]
        gpus = {r["gpu"] for r in rows}
        assert len(gpus) == 1, f"one GPU's rows, got {gpus}"
        for path in _write_tables(rows, args.table_out, impls, gpus.pop(), args.arch,
                                  f"benchmark/quantization/bench_linears.py --impls, {args.tables_from.name}"):
            print(f"wrote {path}")
        return
    from mstar.model.components.quant_linear import (  # here: bench_vllm_linears imports SHAPES
        as_container,
        linear_format,
    )
    device = torch.device("cuda")
    torch.manual_seed(0)
    rows = []
    only = {name for name in args.layers.split(",") if name}
    for layer, (n, k, kind) in SHAPES[args.shapes].items():
        if only and layer not in only:
            continue
        n, k = (n // args.tp, k) if kind == "col" else (n, k // args.tp)
        weight = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
        for m in (int(t) for t in args.tokens.split(",")):
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            ref = F.linear(x, weight)
            base_us = _time(lambda: F.linear(x, weight), args.iters)  # noqa: B023
            rows.append({"layer": layer, "n": n, "k": k, "tokens": m, "format": "bf16", "us": round(base_us, 2)})
            for name in args.formats.split(","):
                fmt = linear_format(name)
                reason = fmt.unavailable_reason(device)
                try:
                    fmt.check(n, k, torch.bfloat16)
                except ValueError as e:
                    reason = reason or str(e)
                if reason:
                    rows.append({"layer": layer, "n": n, "k": k, "tokens": m, "format": name, "skipped": reason})
                    continue
                fmt.prepare_device(device)
                buffers = fmt.quantize_weight(weight, None)
                # as a served layer hands them to linear(): viewed as what the bytes stand for
                stands_for = {st.name: st.stands_for for st in fmt.storage(n, k)}
                stored = [as_container(buffers[b]).view(stands_for.get(b, buffers[b].dtype)) for b in fmt.buffers]
                first = len(rows)
                out = fmt.linear(x, *stored)
                us = _time(lambda: fmt.linear(x, *stored), args.iters)  # noqa: B023
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                rows.append({"layer": layer, "n": n, "k": k, "tokens": m, "format": name, "us": round(us, 2),
                             "speedup": round(base_us / us, 3), "rel_l2": round(err, 4)})
                if name == "fp8_block":
                    rows += _per_impl(layer, fmt, x, weight, stored, ref, base_us, impls, args.iters)
                for row in rows[first:]:
                    print(json.dumps(row), flush=True)
    if args.table_out is not None and impls:
        from mstar.kernels.table import device_key

        dev = device_key(device)
        for path in _write_tables(rows, args.table_out, impls, torch.cuda.get_device_name(), dev.arch if dev else None):
            print(f"wrote {path}", flush=True)
    if args.out:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps({"gpu": torch.cuda.get_device_name(), "shapes": args.shapes, "tp": args.tp,
                                    **row}) + "\n")


if __name__ == "__main__":
    main()
