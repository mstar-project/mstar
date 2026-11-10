"""flashinfer's Blackwell groupwise fp8 GEMM (``gemm_fp8_nt_groupwise``) on dense_fp8's block-fp8 weights: which
scale layout, backend and MMA width reproduce the dequantized product, and how the kernels that run on this GPU
compare per token count. sm_100 / sm_103 only; synthetic weights, no checkpoint.

1. Variants, each against the dequantized product with the activations quantized per 128 columns as the kernel's
   caller does (a right layout is off by rounding, ~1e-3; a wrong one by ~20%): scale layout K-major row-major
   (dense_fp8's first probe candidate), MN-major (its second), K-major column-major (what 01 shipped; failed on a
   B300), MN with the weight scales as stored; mma_sm 1 and 2; the CUTLASS and trtllm backends.
2. Timing (unless --no-time): bf16 (cuBLAS), w8a16, the dequantized bf16 GEMM (dense_fp8 "dequant"), and groupwise
   in the layout dense_fp8's probe picked, per token count, CUDA-graph replays.

    python -m benchmark.kernels.bw_linear_trials --out bw_linear.jsonl
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from benchmark.kernels.timing import time_graph

FP8 = torch.float8_e4m3fn
# per-rank (name, N, K): GLM-5.3 TP8's o / q_b / dense down, Qwen3.5-27B TP4's down, GLM's 2624-row q_a + kv_a
SHAPES = [("glm_o", 6144, 2048), ("glm_q_b", 2048, 2048), ("glm_down", 6144, 1536), ("q27_down", 5120, 4352),
          ("glm_qkv_a", 2624, 6144)]
BOUND = 0.02


def block_fp8(n, k, device, gen):
    pad = -n % 128
    tiles = F.pad(torch.randn(n, k, device=device, generator=gen) * 0.02, (0, 0, 0, pad)).reshape(
        (n + pad) // 128, 128, k // 128, 128)
    s = (tiles.abs().amax(dim=(1, 3)) / torch.finfo(FP8).max).clamp_min(1e-12).contiguous()
    q = (tiles / s[:, None, :, None]).to(FP8).reshape(n + pad, k)[:n].contiguous()
    return q, s


def quantized_x(x):
    """x as the kernel's caller quantizes it: e4m3 per 128 columns, scales (M, K / 128) row-major."""
    fmax = torch.finfo(FP8).max
    g = x.float().reshape(x.shape[0], -1, 128)
    xs = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / fmax
    xq = (g / xs).clamp(-fmax, fmax).to(FP8)
    return xq.reshape(x.shape), xs.squeeze(-1).contiguous()


def variant(xq, xs, q, s, layout, backend, mma_sm):
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    m = xq.shape[0]
    pad = -m % 4
    if pad:
        xq = torch.cat([xq, torch.zeros(pad, xq.shape[1], dtype=xq.dtype, device=xq.device)])
        xs = torch.cat([xs, torch.full((pad, xs.shape[1]), 1e-10 / 448, device=xs.device)])
    a_scale, b_scale, mode = {
        "K-row": (xs, s, "K"),
        "MN": (xs.t().contiguous(), s.t().contiguous(), "MN"),
        "K-col (01)": (xs.t().contiguous().t(), s, "K"),
        "MN, b as stored": (xs.t().contiguous(), s, "MN"),
        "trtllm": (xs.t().contiguous().t(), s.t(), None),  # both "contiguous on the first dimension"
    }[layout]
    kw = dict(scale_major_mode=mode, mma_sm=mma_sm, scale_granularity_mnk=(1, 128, 128), out_dtype=torch.bfloat16)
    if backend != "cutlass":
        kw = dict(out_dtype=torch.bfloat16, backend=backend)
    return gemm_fp8_nt_groupwise(xq, q, a_scale, b_scale, **kw)[:m]


def main() -> None:
    from mstar.utils import dense_fp8
    from mstar.utils.fp8_block import dequantize_fp8_block_weight

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", default="1,8,64,128,256,512,1024,4096")
    p.add_argument("--no-time", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = torch.device("cuda", torch.cuda.current_device())
    gpu = torch.cuda.get_device_name(dev)
    out = open(args.out, "a") if args.out else None  # noqa: SIM115

    def log(row):
        row = {"gpu": gpu, **row}
        print(json.dumps(row), flush=True)
        if out:
            out.write(json.dumps(row) + "\n")

    dense_fp8.reserve(dev)
    log({"probe": {"groupwise_ok": dense_fp8._BW_OK.get(dev), "layout": dense_fp8._BW_LAYOUT.get(dev),
                   "blackwell": dense_fp8._BLACKWELL.get(dev)}})
    gen = torch.Generator(device=dev).manual_seed(0)
    matches: dict[str, int] = {}
    for name, n, k in SHAPES:
        q, s = block_fp8(n, k, dev, gen)
        w32 = dequantize_fp8_block_weight(q, s, (128, 128), torch.float32)
        for m in (1, 5, 64, 300):
            x = torch.randn(m, k, device=dev, generator=gen).bfloat16()
            xq, xs = quantized_x(x)
            want = (xq.float() * xs.repeat_interleave(128, 1)) @ w32.T
            for layout in ("K-row", "MN", "K-col (01)", "MN, b as stored", "trtllm"):
                for backend, mma in ((("cutlass", 1), ("cutlass", 2)) if layout != "trtllm" else (("trtllm", 1),)):
                    row = {"shape": name, "n": n, "k": k, "tokens": m, "layout": layout, "backend": backend,
                           "mma_sm": mma}
                    try:
                        got = variant(xq, xs, q, s, layout, backend, mma).float()
                        err = ((got - want).norm() / want.norm()).item() if torch.isfinite(got).all() else 1e9
                        row.update(rel_l2=round(err, 5), result="MATCH" if err <= BOUND else "mismatch")
                    except Exception as exc:  # noqa: BLE001 - a variant the kernel refuses is a result too
                        row["result"] = f"error: {type(exc).__name__}: {str(exc)[:160]}"
                    if row["result"] == "MATCH":
                        key = f"{layout}/{backend}/mma{mma}"
                        matches[key] = matches.get(key, 0) + 1
                    log(row)
    log({"summary": "variants matching (count of shape x token cases)", "match": matches})
    if args.no_time:
        return
    for name, n, k in SHAPES:
        q, s = block_fp8(n, k, dev, gen)
        wb = dequantize_fp8_block_weight(q, s, (128, 128), torch.bfloat16)
        qu = q.view(torch.uint8)
        for m in (int(t) for t in args.tokens.split(",")):
            x = torch.randn(m, k, device=dev, generator=gen).bfloat16()
            row = {"shape": name, "n": n, "k": k, "tokens": m, "bf16_us": round(time_graph(lambda: F.linear(x, wb)), 2)}  # noqa: B023
            kernels = {"w8a16": lambda: dense_fp8._w8a16(x, qu, s, (128, 128), False),  # noqa: B023
                       "dequant": lambda: dense_fp8._dequant_linear(x, qu, s, (128, 128), False)}  # noqa: B023
            if dense_fp8._BW_OK.get(dev) and n % 128 == 0:
                kernels["groupwise"] = lambda: dense_fp8._bw(x, qu, s, False)  # noqa: B023
            want = F.linear(x, wb).float()
            for kname, fn in kernels.items():
                try:
                    err = ((fn().float() - want).norm() / want.norm()).item()
                    row[f"{kname}_us"] = round(time_graph(fn), 2)
                    row[f"{kname}_rel_l2"] = round(err, 4)
                except Exception as exc:  # noqa: BLE001
                    row[f"{kname}_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            log(row)


if __name__ == "__main__":
    main()
