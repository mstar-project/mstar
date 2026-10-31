"""Tuned hipBLASLt solutions for a flow transformer's GEMM shapes, as the table ``mstar.platforms.rocm_dit_gemm`` reads.

    python -m benchmark.kernels.dit_linear --rows 4128,8256 --nk 20480x3840,3840x10240 --out dit_linear.json
        [--solutions SHORTLIST.json] [--min-gain 0.03] [--job NAME]

Each solution is timed in a captured graph that rotates through weight copies past the MALL, as a DiT step reads its
weights (see ``benchmark.kernels.specs.linear``). A full search takes minutes per shape, so ``--solutions`` takes a list
that an earlier search's winners give. The fastest few and the default are then timed again in alternation, which
takes the selection bias of a best-of-many out of the number, and a shape enters the table only for a gain of at least
``--min-gain``. ``--out`` is rewritten after every shape.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.kernels.specs.linear import _graph_us
from mstar.platforms.rocm_gemm import _decode_key, _stack

WEIGHT_BYTES = 1200 << 20  # past the MALL, as the many weights of a step are
ROUNDS = 5  # alternating retimes of the finalists
FINALISTS = 3


def solution_us(aiter, x, weights):
    return lambda sol: _graph_us(lambda a, b: aiter.hipb_mm(a, b.t(), sol), x, weights, calls=len(weights))


def search_shape(aiter, rows: int, n: int, k: int, solutions: list[int] | None, generator) -> dict:
    x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda", generator=generator)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=generator) * k ** -0.5
    copies = max(2, min(24, -(-WEIGHT_BYTES // (w.numel() * w.element_size()))))
    weights = [w] + [w.clone() for _ in range(copies - 1)]
    timed_solution = solution_us(aiter, x, weights)
    default = lambda: _graph_us(F.linear, x, weights, calls=copies)  # noqa: E731
    ranked = []
    for sol in solutions if solutions is not None else aiter.hipb_findallsols(x, w.t()):
        try:
            ranked.append((timed_solution(sol), sol))
        except Exception:  # noqa: BLE001  a solution that cannot run this shape
            torch.cuda.synchronize()
    ranked.sort()
    finalists = [sol for _, sol in ranked[:FINALISTS]]
    default_us, retimes = [], {sol: [] for sol in finalists}
    for _ in range(ROUNDS):
        default_us.append(default())
        for sol in finalists:
            retimes[sol].append(timed_solution(sol))
    medians = {sol: statistics.median(us) for sol, us in retimes.items()}
    best = min(medians, key=medians.get)
    return {"rows": rows, "n": n, "k": k, "solution": best, "us": round(medians[best], 2),
            "default_us": round(statistics.median(default_us), 2), "tried": len(ranked), "copies": copies,
            "top": [[round(us, 1), sol] for us, sol in ranked[:8]]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="comma list")
    ap.add_argument("--nk", required=True, help="comma list of NxK")
    ap.add_argument("--out", required=True)
    ap.add_argument("--solutions", default=None, help="JSON list of solution indices (default: all of hipBLASLt's)")
    ap.add_argument("--min-gain", type=float, default=0.03)
    ap.add_argument("--job", default="")
    args = ap.parse_args()
    import aiter

    from mstar.platforms import current_platform

    aiter.hipb_create_extension()
    facts = current_platform.device_facts(0)
    solutions = json.loads(Path(args.solutions).read_text()) if args.solutions else None
    generator = torch.Generator(device="cuda").manual_seed(0)
    records, classes = [], {}
    log = Path(args.out).with_suffix(".records.jsonl")
    with torch.no_grad():
        for rows in (int(r) for r in args.rows.split(",")):
            for n, k in (tuple(int(d) for d in nk.split("x")) for nk in args.nk.split(",")):
                t0 = time.perf_counter()
                rec = search_shape(aiter, rows, n, k, solutions, generator)
                rec["s"] = round(time.perf_counter() - t0, 1)
                records.append(rec)
                gain = rec["default_us"] / rec["us"] - 1
                print(f"rows={rows} n={n} k={k}: solution {rec['solution']} {rec['us']} us, default {rec['default_us']}"
                      f" us, {gain:+.1%} of {rec['tried']}", flush=True)
                with open(log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if gain >= args.min_gain:
                    classes[_decode_key(rows, n, k)] = {
                        "impl": "hipblaslt", "solution": rec["solution"],
                        "us": rec["us"], "default_us": rec["default_us"],
                    }
                Path(args.out).write_text(json.dumps({
                    "kernel": "dit_linear", "arch": facts.arch, "product": facts.product, "classes": classes,
                    "provenance": {"stack": _stack(), "job": args.job, "min_gain": args.min_gain, "source":
                                   "benchmark/kernels/dit_linear.py, captured graphs, weights rotated past the MALL"},
                }, indent=1) + "\n")


if __name__ == "__main__":
    main()
