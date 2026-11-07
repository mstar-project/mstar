"""The 4-bit dense GEMM's paths and tiles (``mstar.utils.w4a16``) against bf16 at a model's per-rank shapes, each call
replayed in a CUDA graph, torch's own int4 GEMM (tinygemm, ``aten._weight_int4pack_mm``, on the copy the format
builds at load) among them:

    python -m benchmark.quantization.bench_w4a16 --shapes qwen3_5_27b --tp 1 --kinds int4,int4zp,mxfp4 \\
        --tokens 1,2,4,8,16,32,64,128,256,512,1024 --out w4a16.jsonl

``int4`` is symmetric group 128 (GPTQ / compressed-tensors), ``int4zp`` group 128 with zero points (AutoAWQ, as
QuantTrio/Qwen3.5-27B-AWQ), ``mxfp4`` e2m1 with e8m0 scales per 32. Per (layer, tokens, kind): the built-in choice
(``w4a16.config``) and every candidate tile of the paths that apply, each checked against the reference GEMM
(relative L2 below 1e-2) before it is timed; the summary line names the fastest, the best of each path and the
built-in choice, so the built-in rule can be set from the numbers. The vLLM column (Machete, Marlin) is
``bench_vllm_w4a16`` in a vLLM environment.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from benchmark.quantization.bench_linears import SHAPES, _time

GEMV = [{"BN": bn, "NB": nb, "num_warps": w} for bn, nb, w in
        ((16, 4, 4), (16, 8, 4), (16, 2, 4), (32, 2, 4), (32, 4, 4), (64, 1, 4), (64, 2, 8))]
DOT = [{"BN": bn, "NB": nb, "STAGES": st, "num_warps": w} for bn, nb, st, w in
       ((64, 1, 3, 4), (64, 2, 2, 4), (128, 1, 3, 8), (128, 2, 2, 8), (64, 1, 3, 8), (32, 1, 3, 4), (32, 2, 3, 4))]
DOT8 = [{"BN": bn, "NB": nb, "STAGES": st, "num_warps": w} for bn, nb, st, w in
        ((64, 1, 3, 4), (128, 1, 3, 8), (64, 2, 2, 4), (128, 2, 2, 8), (32, 1, 3, 4))]


def _weights(kind: str, n: int, k: int, device):
    """The served buffers of a random weight (planes layout, scales, zero points, tinygemm's copy) and its kind."""
    from mstar.model.components import int4_formats as I

    w = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
    if kind == "mxfp4":
        q = I.MXFP4.quantize_weight(w, None)
        return q["weight_planes"].view(torch.uint8), q["weight_scale"], None, 32, I.KIND_E2M1, None, None
    fmt = I.W4A16Format(group=128, symmetric=kind == "int4")
    stored = fmt.quantize_storage(w)
    if kind == "int4zp":
        stored["weight_zero_point"] = I.pack_int4(torch.randint(4, 12, (n, k // 128), device=device), dim=0)
    q = fmt.after_load(stored)
    tg = q["tinygemm_weight"] if q["tinygemm_weight"].numel() else None
    return (q["weight_planes"].view(torch.uint8), q["weight_scale"], q.get("weight_zero_point"), 128,
            I.KIND_INT4_SYM if kind == "int4" else I.KIND_INT4_ZP, tg, q["tinygemm_scale_zero"])


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
            packed, scale, zp, group, code, tg_w, tg_sz = _weights(wkind, n, k, dev)
            ref_w = I.dequantize_kind(packed, scale, zp, group, code).to(torch.bfloat16)
            blocks = k // 128
            for m in (int(t) for t in args.tokens.split(",")):
                x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
                ref = F.linear(x, ref_w)
                base_us = _time(lambda: F.linear(x, ref_w), args.iters)  # noqa: B023
                base = {"layer": layer, "n": n, "k": k, "tokens": m, "kind": wkind}
                rows.append({**base, "impl": "bf16", "us": round(base_us, 2)})
                # the built-in dispatch as gemm runs it (tinygemm first at 1-2 tokens), timed with no override
                tiny_first = tg_w is not None and m <= w4a16.TINYGEMM_MAX
                cands = [("builtin", ("tinygemm", {}) if tiny_first else w4a16.config(m, n, k, group, code, sms))]
                if tg_w is not None and m <= 64:
                    cands.append(("tinygemm", ("tinygemm", {})))
                if m <= w4a16.GEMV_MAX:
                    cands += [("gemv", ("gemv", dict(c))) for c in GEMV if blocks % c["NB"] == 0]
                if m < 256:
                    for path, table in (("dot", DOT), ("dot8", DOT8)):
                        for c in table:
                            if blocks % c["NB"]:
                                continue
                            tb = w4a16._tb(m)
                            ctas = -(-n // c["BN"]) * -(-m // tb)
                            splits = {w4a16._split(ctas, blocks // c["NB"], sms, waves) for waves in (1, 2, 4)}
                            cands += [(path, (path, {**c, "TB": tb, "SPLIT_K": sk})) for sk in sorted(splits)]
                cands.append(("dequant", ("dequant", {})))
                timed = []
                for name, (path, cfg) in cands:
                    cfg = dict(cfg)
                    if path == "gemv":
                        cfg["SPLIT_K"] = w4a16._split(-(-n // cfg["BN"]), blocks // cfg["NB"], sms, 4)
                    try:
                        w4a16.CONFIG_OVERRIDE = None if name == "builtin" else (lambda *a, p=path, c=cfg: (p, c))
                        run = lambda: w4a16.gemm(x, packed, scale, zp, group, code, tg_w, tg_sz)  # noqa: E731,B023
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
                best = max(timed, key=lambda r: r["speedup"]) if timed else None
                built = next((r for r in timed if r["impl"] == "builtin"), None)
                per_path = {}
                for r in timed:
                    if r["impl"] != "builtin" and r["speedup"] > per_path.get(r["path"], (0,))[0]:
                        per_path[r["path"]] = (r["speedup"], r["config"])
                summary = {**base, "bf16_us": round(base_us, 2),
                           "builtin": built and {"path": built["path"], "config": built["config"],
                                                 "speedup": built["speedup"]},
                           "best": best and {"path": best["path"], "config": best["config"],
                                             "speedup": best["speedup"]},
                           "per_path": {p: v[0] for p, v in per_path.items()},
                           "per_path_config": {p: v[1] for p, v in per_path.items()}}
                print(json.dumps(summary), flush=True)
                rows.append({**summary, "impl": "summary"})
    if args.out:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps({"gpu": torch.cuda.get_device_name(), "shapes": args.shapes, "tp": args.tp, **row})
                        + "\n")


if __name__ == "__main__":
    main()
