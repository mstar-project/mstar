"""Routed-experts formats against bf16 at a model's per-rank expert shapes, each call replayed in a CUDA graph.

    python -m benchmark.quantization.bench_experts_formats --formats fp8_block,w4a16,w4a16_g32,mxfp4 \\
        --shapes qwen3_omni_thinker --tokens 1,8,32,128,512 --out experts.jsonl

One row per (format, shape, tokens): median microseconds of a replay of the whole routed-experts call (gate/up
GEMM, SwiGLU, down GEMM, top-k sum) over ``--iters`` calls, the speedup over the bf16 fused MoE, and the output's
relative L2 distance from it. Routing is uniform random top-k, the same for every format. ``w4a16`` is int4
symmetric group 128 (quantized here, round-to-nearest), ``w4a16_g32`` group 32 (Kimi K2.x's), ``mxfp4`` e2m1 with
e8m0 scales per 32 (Kimi K3's); all three run ``fused_experts_w4a16``; ``fp8_block`` runs ``fused_experts_fp8``.
"""

from __future__ import annotations

import argparse
import json

import torch

from benchmark.quantization.bench_linears import _time

# (experts on the rank, hidden, expert width on the rank, top-k)
SHAPES = {
    "qwen3_omni_thinker": (128, 2048, 768, 8),
    # Kimi K2.6 at TP 8 (384 experts, 2048 wide) and Kimi K3.1 at TP 8 (896 experts, 3072 wide, latent 3584)
    "kimi_k2_6_tp8": (384, 7168, 256, 8),
    "kimi_k3_1_tp8": (896, 3584, 384, 16),
}


def _stack(fmt, w):
    parts = [fmt.quantize_weight(w[i], None) for i in range(w.shape[0])]
    return {k: torch.stack([p[k] for p in parts]) for k in parts[0]}


def main() -> None:
    from mstar.model.components import int4_formats as I
    from mstar.utils.fp8_block import quantize_fp8_block_weight
    from mstar.utils.fused_moe import fused_experts, fused_experts_fp8
    from mstar.utils.fused_moe.w4a16 import fused_experts_w4a16

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formats", default="fp8_block,w4a16,w4a16_g32,mxfp4")
    parser.add_argument("--shapes", default="qwen3_omni_thinker", choices=sorted(SHAPES))
    parser.add_argument("--tokens", default="1,8,32,128,512")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    e, hidden, inter, top_k = SHAPES[args.shapes]
    w1 = (torch.randn(e, 2 * inter, hidden, device=dev) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(e, hidden, inter, device=dev) * 0.02).to(torch.bfloat16)
    runs = {}
    for name in args.formats.split(","):
        if name == "fp8_block":
            q = [[quantize_fp8_block_weight(w[i]) for i in range(e)] for w in (w1, w2)]
            (q1, s1), (q2, s2) = ([torch.stack(t) for t in zip(*qs, strict=True)] for qs in q)
            runs[name] = lambda x, wt, ids, a=(q1, q2, s1.contiguous(), s2.contiguous()): fused_experts_fp8(
                x, a[0], a[1], a[2], a[3], wt, ids)
            continue
        fmt = {"w4a16": I.W4A16, "w4a16_g32": I.W4A16Format(group=32), "mxfp4": I.MXFP4}[name]
        b1, b2 = _stack(fmt, w1), _stack(fmt, w2)
        kind = I.KIND_E2M1 if name == "mxfp4" else I.KIND_INT4_SYM
        runs[name] = lambda x, wt, ids, b1=b1, b2=b2, kind=kind, g=fmt.group: fused_experts_w4a16(
            x, b1["weight_packed"].view(torch.uint8), b2["weight_packed"].view(torch.uint8), b1["weight_scale"],
            b2["weight_scale"], None, None, wt, ids, kind, g, "silu", False)
    rows = []
    for m in (int(t) for t in args.tokens.split(",")):
        x = torch.randn(m, hidden, device=dev, dtype=torch.bfloat16)
        ids = torch.rand(m, e, device=dev).topk(top_k, dim=-1).indices.to(torch.int32)
        wt = torch.softmax(torch.randn(m, top_k, device=dev), -1)
        ref = fused_experts(x, w1, w2, wt, ids)
        base_us = _time(lambda: fused_experts(x, w1, w2, wt, ids), args.iters)  # noqa: B023
        rows.append({"tokens": m, "format": "bf16", "us": round(base_us, 2)})
        for name, run in runs.items():
            out = run(x, wt, ids)
            us = _time(lambda: run(x, wt, ids), args.iters)  # noqa: B023
            err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            rows.append({"tokens": m, "format": name, "us": round(us, 2), "speedup": round(base_us / us, 3),
                         "rel_l2": round(err, 4)})
        for row in rows[-(1 + len(runs)):]:
            print(json.dumps({"shapes": args.shapes, **row}), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps({"gpu": torch.cuda.get_device_name(), "shapes": args.shapes, **row}) + "\n")


if __name__ == "__main__":
    main()
