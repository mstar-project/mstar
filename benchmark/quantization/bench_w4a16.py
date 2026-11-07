"""The 4-bit dense GEMM's paths and tiles (``mstar.utils.w4a16``) against bf16 at a model's per-rank shapes, each call
replayed in a CUDA graph, plus torch's own int4 GEMM (tinygemm, ``aten._weight_int4pack_mm``) as a second reference:

    python -m benchmark.quantization.bench_w4a16 --shapes qwen3_5_27b --tp 1 --kinds int4,int4zp,mxfp4 \\
        --tokens 1,2,4,8,16,32,64,128,256,512,1024 --out w4a16.jsonl

``int4`` is symmetric group 128 (GPTQ / compressed-tensors), ``int4zp`` group 128 with zero points (AutoAWQ, as
QuantTrio/Qwen3.5-27B-AWQ), ``mxfp4`` e2m1 with e8m0 scales per 32. Per (layer, tokens, kind): the built-in choice
(``w4a16.config``) and every candidate tile of the paths that apply, each checked against the reference GEMM
(relative L2 below 1e-2) before it is timed; the summary line names the fastest and how far the built-in choice is
from it, so the built-in rule can be set from the numbers. The vLLM column (Machete, Marlin) is
``bench_vllm_w4a16`` in a vLLM environment.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from benchmark.quantization.bench_linears import SHAPES, _time

GEMV = [{"BN": bn, "BK": bk, "num_warps": w} for bn, bk, w in
        ((16, 256, 4), (32, 256, 4), (32, 128, 4), (64, 128, 4), (32, 512, 4), (64, 256, 8))]
DOT = [{"BN": bn, "BK": bk, "STAGES": st, "num_warps": w} for bn, bk, st, w in
       ((64, 128, 3, 4), (128, 128, 3, 8), (64, 64, 4, 4), (128, 64, 3, 4), (32, 128, 3, 4), (64, 256, 2, 4))]


def _weights(kind: str, n: int, k: int, device):
    from mstar.model.components import int4_formats as I

    w = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
    if kind == "mxfp4":
        q = I.MXFP4.quantize_weight(w, None)
        return q["weight_packed"], q["weight_scale"], None, 32, I.KIND_E2M1, None
    q = I.W4A16Format(group=128).quantize_weight(w, None)
    zp = None
    if kind == "int4zp":
        zp = I.pack_int4(torch.randint(4, 12, (n, k // 128), device=device), dim=0)
    codes = I.unpack_int4(q["weight_packed"])
    return (q["weight_packed"].view(torch.uint8), q["weight_scale"], zp, 128,
            I.KIND_INT4_SYM if zp is None else I.KIND_INT4_ZP, codes)


def _tinygemm(codes: torch.Tensor, scale: torch.Tensor, group: int):
    """torch's int4 GEMM on the symmetric weight, or None where this build has none. Its packing takes uint8
    [N, K/2]; which nibble holds the even column is checked against the reference, both tried."""
    try:
        sz = torch.stack([scale.t().float(), torch.zeros_like(scale.t(), dtype=torch.float)], dim=-1)
        sz = sz.to(torch.bfloat16).contiguous()
        packed = []
        for hi_first in (True, False):
            a, b = (codes[:, 0::2], codes[:, 1::2]) if hi_first else (codes[:, 1::2], codes[:, 0::2])
            u8 = ((a << 4) | b).to(torch.uint8).contiguous()
            packed.append(torch.ops.aten._convert_weight_to_int4pack(u8, 8))
        return [(lambda x, p=p: torch.ops.aten._weight_int4pack_mm(x, p, group, sz)) for p in packed]
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"tinygemm": f"unavailable: {type(e).__name__}: {e}"[:200]}), flush=True)
        return None


def main() -> None:
    from mstar.model.components import int4_formats as I
    from mstar.utils import w4a16

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", default="qwen3_5_27b", choices=sorted(SHAPES))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--kinds", default="int4,int4zp,mxfp4")
    parser.add_argument("--tokens", default="1,2,4,8,16,32,64,128,256,512,1024")
    parser.add_argument("--layers", default="", help="only these of the shape set's layers (comma-separated)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    only = {name for name in args.layers.split(",") if name}
    rows = []
    for layer, (n, k, kind) in SHAPES[args.shapes].items():
        if only and layer not in only:
            continue
        n, k = (n // args.tp, k) if kind == "col" else (n, k // args.tp)
        for wkind in args.kinds.split(","):
            packed, scale, zp, group, code, codes = _weights(wkind, n, k, dev)
            ref_w = I.dequantize_kind(packed, scale, zp, group, code).to(torch.bfloat16)
            tiny = _tinygemm(codes, scale, group) if wkind == "int4" else None
            for m in (int(t) for t in args.tokens.split(",")):
                x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
                ref = F.linear(x, ref_w)
                base_us = _time(lambda: F.linear(x, ref_w), args.iters)  # noqa: B023
                base = {"layer": layer, "n": n, "k": k, "tokens": m, "kind": wkind}
                rows.append({**base, "impl": "bf16", "us": round(base_us, 2)})
                builtin = w4a16.config(m, n, k, group, code, sms)
                cands = [("builtin", builtin)]
                if m <= 8:
                    cands += [("gemv", ("gemv", {**c, "SPLIT_K": 0})) for c in GEMV]
                if m <= 1024:
                    cands += [("dot", ("dot", {**c, "SPLIT_K": 0})) for c in DOT]
                cands.append(("dequant", ("dequant", {})))
                timed = []
                for name, (path, cfg) in cands:
                    cfg = dict(cfg)
                    try:
                        if path != "dequant":
                            bk = w4a16._block_k(group, k, cfg["BK"])
                            cfg["BK"] = bk
                            if path == "dot":
                                cfg.setdefault("TB", 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128)
                                ctas = -(-n // cfg["BN"]) * -(-m // cfg["TB"])
                                waves = 2
                            else:
                                ctas, waves = -(-n // cfg["BN"]), 4
                            if not cfg.get("SPLIT_K"):
                                cfg["SPLIT_K"] = w4a16._split(ctas, k // bk, sms, waves)
                        w4a16.CONFIG_OVERRIDE = lambda *a, p=path, c=cfg: (p, c)
                        run = lambda: w4a16.gemm(x, packed, scale, zp, group, code)  # noqa: E731,B023
                        out = run()
                        err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                        if not err < 1e-2:
                            rows.append({**base, "impl": name, "config": cfg, "rejected": round(err, 4)})
                            continue
                        us = _time(run, args.iters)
                    except Exception as e:  # noqa: BLE001
                        rows.append({**base, "impl": name, "config": cfg, "error": f"{type(e).__name__}: {e}"[:200]})
                        continue
                    finally:
                        w4a16.CONFIG_OVERRIDE = None
                    row = {**base, "impl": name, "path": path, "config": cfg, "us": round(us, 2),
                           "speedup": round(base_us / us, 3), "rel_l2": round(err, 5)}
                    rows.append(row)
                    timed.append(row)
                for i, fn in enumerate(tiny or []):
                    out = fn(x)
                    err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                    if err < 1e-2:
                        us = _time(lambda fn=fn: fn(x), args.iters)  # noqa: B023
                        rows.append({**base, "impl": "tinygemm", "config": {"hi_first": i == 0}, "us": round(us, 2),
                                     "speedup": round(base_us / us, 3), "rel_l2": round(err, 5)})
                best = max(timed, key=lambda r: r["speedup"]) if timed else None
                built = next((r for r in timed if r["impl"] == "builtin"), None)
                tiny_row = next((r for r in rows[::-1] if r.get("impl") == "tinygemm" and r["tokens"] == m
                                 and r["layer"] == layer and r["kind"] == wkind), None)
                summary = {**base, "bf16_us": round(base_us, 2),
                           "builtin": built and {"path": built["path"], "speedup": built["speedup"]},
                           "best": best and {"path": best["path"], "config": best["config"],
                                             "speedup": best["speedup"]},
                           "tinygemm": tiny_row and tiny_row["speedup"]}
                print(json.dumps(summary), flush=True)
                rows.append({**summary, "impl": "summary"})
    if args.out:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps({"gpu": torch.cuda.get_device_name(), "shapes": args.shapes, "tp": args.tp, **row})
                        + "\n")


if __name__ == "__main__":
    main()
