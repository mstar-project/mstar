"""dense_fp8's block-fp8 linear per kernel at a model's per-rank shapes, against bf16: which kernel each token
count should take on this GPU (w8a16, flashinfer's SM90 swapAB GEMM, cuBLAS's block-scaled GEMM, flashinfer's
Blackwell groupwise GEMM, whichever run here). CUDA-graph replays (``timing.time_graph``); one JSON line per
(shape, tokens, kernel) with microseconds, the speedup over bf16 and the relative distance from bf16's output.

    python -m benchmark.kernels.fp8_linear_impls --shapes glm53_tp8,qwen3_5_27b_tp4 --out fp8_linear.jsonl
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from benchmark.kernels.timing import time_graph

# per-rank (name, N, K, glu): N per half for a fused [gate; up] with its SwiGLU
SHAPES = {
    # GLM-5.3 at TP8: q_a + kv_a fused (2624 rows, unsharded), q_b, o_proj, the dense layers' MLP
    "glm53_tp8": [("qkv_a", 2624, 6144, False), ("q_b", 2048, 2048, False), ("o", 6144, 2048, False),
                  ("gate_up", 1536, 6144, True), ("down", 6144, 1536, False)],
    # Qwen3.5-27B at TP4
    "qwen3_5_27b_tp4": [("qkv", 3584, 5120, False), ("o", 5120, 1536, False), ("gate_up", 4352, 5120, True),
                        ("down", 5120, 4352, False)],
}


def _block_quant(w: torch.Tensor, fp8: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 with one fp32 scale per 128 x 128 block (the last row block partial), as checkpoints store it."""
    n, k = w.shape
    pad = -n % 128
    tiles = F.pad(w.float(), (0, 0, 0, pad)).reshape((n + pad) // 128, 128, k // 128, 128)
    scale = (tiles.abs().amax(dim=(1, 3)) / torch.finfo(fp8).max).clamp_min(1e-12)
    q = (tiles / scale[:, None, :, None]).to(fp8).reshape(n + pad, k)[:n]
    return q.contiguous(), scale.contiguous()


def main() -> None:
    from mstar.utils import dense_fp8
    from mstar.utils.fp8_block import FP8_DTYPE, dequantize_fp8_block_weight

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", default="glm53_tp8,qwen3_5_27b_tp4")
    p.add_argument("--tokens", default="1,2,4,8,16,32,64,128,256,512,1024")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = torch.device("cuda", torch.cuda.current_device())
    dense_fp8.reserve(dev)
    kernels = {"w8a16": lambda x, w, s, glu: dense_fp8._w8a16(x, w, s, (128, 128), glu)}
    if dense_fp8._FI_OK.get(dev):
        kernels["flashinfer"] = dense_fp8._fi
    if dense_fp8._W8A8_OK.get(dev):
        kernels["cublas"] = lambda x, w, s, glu: dense_fp8._w8a8(x, w, s, (128, 128), glu)
    if dense_fp8._BW_OK.get(dev):
        kernels["groupwise"] = dense_fp8._bw
    gpu = torch.cuda.get_device_name(dev)
    rows = []
    torch.manual_seed(0)
    for shapes in args.shapes.split(","):
        for name, n, k, glu in SHAPES[shapes]:
            rows_w = 2 * n if glu else n
            q, s = _block_quant((torch.randn(rows_w, k, device=dev) * 0.02).to(torch.bfloat16), FP8_DTYPE)
            w, s = q.view(torch.uint8).contiguous(), s.float().contiguous()
            ref_w = dequantize_fp8_block_weight(q.view(FP8_DTYPE), s, (128, 128)).to(torch.bfloat16)
            for m in (int(t) for t in args.tokens.split(",")):
                x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)

                def bf16():
                    h = F.linear(x, ref_w)  # noqa: B023
                    if glu:  # noqa: B023
                        g, u = h.chunk(2, dim=-1)
                        return F.silu(g) * u
                    return h

                want = bf16()
                base = time_graph(bf16)
                head = {"gpu": gpu, "shapes": shapes, "layer": name, "n": n, "k": k, "glu": glu, "tokens": m}
                rows.append({**head, "kernel": "bf16", "us": round(base, 2)})
                for kname, fn in kernels.items():
                    try:
                        out = fn(x, w, s, glu)
                        us = time_graph(lambda: fn(x, w, s, glu))  # noqa: B023
                    except Exception as e:  # noqa: BLE001 - this kernel can't take the shape here
                        rows.append({**head, "kernel": kname, "skipped": f"{type(e).__name__}: {e}"[:160]})
                        continue
                    err = ((out.float() - want.float()).norm() / want.float().norm()).item()
                    rows.append({**head, "kernel": kname, "us": round(us, 2), "speedup": round(base / us, 3),
                                 "rel_l2": round(err, 4)})
                for row in rows[-(1 + len(kernels)):]:
                    print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
