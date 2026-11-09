"""Kimi K3.1's dense decode GEMMs per TP8 rank, M = 1..64 tokens: cuBLAS against FlashInfer's small-M bf16 kernels,
and the projections that read one input merged into one GEMM.

Backends: ``mstar.utils.small_m_gemm.BACKENDS`` (cuBLAS through torch, FlashInfer's ``cublaslt``, ``tinygemm``,
``tgv``, ``cute-dsl`` and its Kimi-K3 router GEMM), each where the GPU, the output dtype and M take it. Each call
is CUDA-graph replayed over rotating weight copies (from HBM, as layer
after layer in a step) and checked against an fp32 product. Prints per shape and M each backend's microseconds, then
per M the step's total for the shipped projections on cuBLAS, the merged ones on cuBLAS, and the merged ones on each
shape's fastest backend.

    python -m benchmark.kernels.k31_dense_gemm --out dense_gemm.jsonl --table-out table.json [--tokens 1,2,4,8,16,32,64]

``--table-out`` writes, for this GPU, each shape and M whose fastest backend beats cuBLAS by 5% or more, in the format
``small_m_gemm.load_table`` reads (a serving config names it under ``model_kwargs.small_m_gemm_table``).
"""
from __future__ import annotations

import argparse
import json

import torch

from benchmark.kernels.timing import time_graph

HIDDEN = 7168
TOKENS = (1, 2, 4, 8, 16, 32, 64)
ROTATION_BYTES = 512 << 20
# per-rank (name, N, K, calls a decode step) at TP8: 69 KDA + 24 MLA layers, 92 latent MoE + 1 dense MLP layer
SHIPPED = [
    ("kda_in", 6288, HIDDEN, 69), ("kda_o", HIDDEN, 1536, 69),
    ("mla_qkv_a", 2112, HIDDEN, 24), ("mla_g", 1536, HIDDEN, 24), ("mla_q_b", 2304, 1536, 24),
    ("mla_o", HIDDEN, 1536, 24),
    ("moe_down", 3584, HIDDEN, 92), ("moe_shared_gate_up", 1536, HIDDEN, 92),
    # the latent's up projection is row parallel: a rank's 3584 / 8 latent columns
    ("moe_up", HIDDEN, 448, 92), ("moe_shared_down", HIDDEN, 768, 92),
    ("dense_gate_up", 8448, HIDDEN, 1), ("dense_down", HIDDEN, 4224, 1), ("lm_head", 20480, HIDDEN, 1),
]
# merged: the projections of one input as one GEMM (name, N, K, calls, the shipped ones it replaces)
MERGED = [
    ("mla_qkv_a_g", 2112 + 1536, HIDDEN, 24, ("mla_qkv_a", "mla_g")),
    ("moe_down_shared_gate_up", 3584 + 1536, HIDDEN, 92, ("moe_down", "moe_shared_gate_up")),
    # the latent up projection and the shared experts' down projection along K: one row-parallel partial
    ("moe_up_shared_down", HIDDEN, 448 + 768, 92, ("moe_up", "moe_shared_down")),
]
# measured, not in the step totals: the K-packed GEMM with K padded to 1280, a multiple of cute-dsl's 128-wide K tile
EXTRA = [("moe_up_shared_down_k1280", HIDDEN, 1280, 92)]
ROUTER = ("moe_router_fp32", 896, HIDDEN, 92)
# the router's fp32 logits with the rest of the MoE input's projections: one fp32-out GEMM
ROUTER_MERGED = ("moe_router_down_shared_fp32", 896 + 3584 + 1536, HIDDEN, 92)


def _candidates(major: int, out_dtype: torch.dtype, n: int, k: int, m: int):
    """name -> fn(x, w, out) for this GPU, output dtype, shape and M; w ``(N, K)`` row-major."""
    from mstar.utils.small_m_gemm import BACKENDS

    arch = {"fi_tinygemm": 9, "fi_tgv": 10, "fi_cute_dsl": 10}
    out = {}
    for name, (fn, dtypes, limit) in BACKENDS.items():
        if out_dtype not in dtypes or (limit is not None and m > limit) or major < arch.get(name, 0):
            continue
        if name == "fi_router_n896" and (n, k) != (896, HIDDEN):
            continue
        out[name] = _bind(fn, out_dtype)
    return out


def _bind(fn, out_dtype):
    return lambda x, w, y: fn(x, w, y, out_dtype if out_dtype != x.dtype else None)


def _time(fn, x, ws, y) -> float:
    return time_graph(lambda: [fn(x, w, y) for w in ws], replays=20) / len(ws)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    p.add_argument("--hbm-tbps", type=float, default=None, help="HBM bandwidth for the floor (B300 ~8, H200 4.8)")
    p.add_argument("--out", default=None)
    p.add_argument("--table-out", default=None)
    a = p.parse_args()
    dev = torch.device("cuda", torch.cuda.current_device())
    major = torch.cuda.get_device_capability(dev)[0]
    gpu = torch.cuda.get_device_name(dev)
    tbps = a.hbm_tbps or (8.0 if major >= 10 else 4.8)
    table: list[dict] = []
    tokens = [int(t) for t in a.tokens.split(",")]
    sink = open(a.out, "a") if a.out else None  # noqa: SIM115
    best: dict[tuple[str, int], float] = {}
    torch_us: dict[tuple[str, int], float] = {}

    def log(row):
        row = {"gpu": gpu, "bench": "k31_dense_gemm", **row}
        print(json.dumps(row), flush=True)
        if sink:
            sink.write(json.dumps(row) + "\n")

    shapes = [(n, nn, k, c) for n, nn, k, c in SHIPPED] + [(n, nn, k, c) for n, nn, k, c, _ in MERGED]
    for name, n, k, calls in [*shapes, ROUTER, ROUTER_MERGED, *EXTRA]:
        copies = max(2, min(32, ROTATION_BYTES // (n * k * 2)))
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        for m in tokens:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            ref = x.float() @ ws[0].float().t()
            res = {}
            out_dtype = torch.float32 if name in (ROUTER[0], ROUTER_MERGED[0]) else torch.bfloat16
            cands = _candidates(major, out_dtype, n, k, m)
            for bname, fn in cands.items():
                y = torch.empty(m, n, device=dev, dtype=out_dtype)
                try:
                    got = fn(x, ws[0], y)
                    torch.cuda.synchronize()
                    err = ((got.float() - ref).norm() / ref.norm()).item()
                    if not err < 1e-2:
                        res[bname + "_error"] = f"rel err {err:.2e}"
                        continue
                    res[bname] = round(_time(fn, x, ws, y), 2)
                except Exception as exc:  # noqa: BLE001 - a backend this shape, M or GPU doesn't take
                    res[bname + "_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
            timed = {b: v for b, v in res.items() if isinstance(v, float)}
            if timed:
                win = min(timed, key=timed.get)
                best[(name, m)] = timed[win]
                torch_us[(name, m)] = timed.get("torch", timed[win])
                res["best"] = win
                if win != "torch" and "torch" in timed and timed[win] <= 0.95 * timed["torch"]:
                    table.append({"n": n, "k": k, "out": "fp32" if out_dtype is torch.float32 else "bf16",
                                  "m": m, "backend": win, "us": timed[win], "torch_us": timed["torch"]})
            log({"shape": name, "n": n, "k": k, "calls": calls, "tokens": m,
                 "floor_us": round(n * k * 2 / (tbps * 1e12) * 1e6, 2), **res})
        ws.clear()
        torch.cuda.empty_cache()
    merged_away = {s for *_, parts in MERGED for s in parts}
    def total(table, names, m):
        return sum(c * table[(nm, m)] for nm, _, _, c in names if (nm, m) in table)

    for m in tokens:
        shipped = [*SHIPPED, ROUTER]
        merged = [s for s in SHIPPED if s[0] not in merged_away] + [s[:4] for s in MERGED] + [ROUTER]
        log({"summary": "step_ms", "tokens": m,
             "shipped_torch": round(total(torch_us, shipped, m) / 1e3, 3),
             "merged_torch": round(total(torch_us, merged, m) / 1e3, 3),
             "merged_best": round(total(best, merged, m) / 1e3, 3),
             # the router in the MoE input's one fp32-out GEMM (its down / shared outputs cast to bf16 after)
             "merged3_torch": round(total(torch_us, [s for s in merged if s[0] not in (
                 ROUTER[0], "moe_down_shared_gate_up")] + [ROUTER_MERGED], m) / 1e3, 3),
             "merged3_best": round(total(best, [s for s in merged if s[0] not in (
                 ROUTER[0], "moe_down_shared_gate_up")] + [ROUTER_MERGED], m) / 1e3, 3)})
    if a.table_out:
        with open(a.table_out, "w") as f:
            json.dump({"gpu": gpu, "capability": list(torch.cuda.get_device_capability(dev)), "entries": table},
                      f, indent=1)


if __name__ == "__main__":
    main()
