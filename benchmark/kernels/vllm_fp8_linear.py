"""vLLM's block-fp8 linear at fp8_linear's shapes, timed as ``benchmark.kernels.sweep fp8_linear`` times M*'s: one
call per weight copy in a CUDA graph, the copies rotated so the weights come from HBM. Run it in a vLLM venv with this
tree on ``PYTHONPATH`` (it imports nothing from mstar):

    python -m benchmark.kernels.vllm_fp8_linear --shape N=17408,K=5120,glu=1 --buckets 1,16,64,256 --json out.json

``vllm`` is what vLLM 0.33 runs on Hopper for a Qwen/DeepSeek FP8 checkpoint
(``FlashInferFp8DeepGEMMDynamicBlockScaledKernel``): FlashInfer's swapAB GEMM below 32 tokens, DeepGEMM from 32, x
quantized per 128 columns either way; ``deepgemm`` and ``flashinfer`` are its two halves at every count. A glu shape
times the GEMM over both halves and vLLM's SiluAndMul after it.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from benchmark.kernels.timing import time_graph

FP8 = torch.float8_e4m3fn
ROTATION_BYTES = 256 << 20
BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)


def _weights(rows: int, k: int, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    copies = max(2, min(16, ROTATION_BYTES // (rows * k)))
    torch.manual_seed(0)
    out = []
    for _ in range(copies):
        w = torch.randn(rows, k, device=device) * 0.02
        tiles = w.reshape(rows // 128, 128, k // 128, 128)
        s = (tiles.abs().amax(dim=(1, 3)) / torch.finfo(FP8).max).clamp_min(1e-12)
        q = (tiles / s[:, None, :, None]).to(FP8).reshape(rows, k)
        out.append((q.contiguous(), s.contiguous()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shape", action="append", required=True, help="N=..,K=..,glu=0|1 (N per half for glu)")
    p.add_argument("--buckets", default=",".join(map(str, BUCKETS)))
    p.add_argument("--json", help="write every timing here")
    args = p.parse_args()

    import vllm._custom_ops  # noqa: F401 - loads torch.ops._C (silu_and_mul)
    from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
        _dynamic_flashinfer_deepgemm_blockscale_gemm_impl as vllm_dynamic,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
        flashinfer_fp8_blockscale_gemm,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
        per_token_group_quant_fp8,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt, is_deep_gemm_e8m0_used

    e8m0 = is_deep_gemm_e8m0_used()
    device = torch.device("cuda")

    def act(h):  # vLLM's SiluAndMul kernel
        out = torch.empty(h.shape[0], h.shape[1] // 2, dtype=h.dtype, device=h.device)
        torch.ops._C.silu_and_mul(out, h)
        return out

    def deepgemm(x, w, s):
        q, qs = per_token_group_quant_fp8(x, 128, column_major_scales=True, use_ue8m0=e8m0)
        out = torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)
        fp8_gemm_nt((q, qs), (w, s), out, is_deep_gemm_e8m0_used=e8m0)
        return out

    def flashinfer(x, w, s):
        return flashinfer_fp8_blockscale_gemm(input=x, weight=w, weight_scale=s, out_dtype=torch.bfloat16)

    impls = {"vllm": lambda x, w, s: vllm_dynamic(x, w, s, 128, e8m0), "deepgemm": deepgemm,
             "flashinfer": flashinfer}
    report = []
    print(f"# vLLM fp8 block linear on {torch.cuda.get_device_name()}, DeepGEMM e8m0 {e8m0}", flush=True)
    for text in args.shape:
        shape = {k: int(v) for k, v in (part.split("=") for part in text.split(","))}
        n, k, glu = shape["N"], shape["K"], bool(shape["glu"])
        rows = 2 * n if glu else n
        weights = []
        for q, s in _weights(rows, k, device):
            q, s = deepgemm_post_process_fp8_weight_block(wq=q, ws=s, quant_block_shape=(128, 128), use_e8m0=e8m0)
            weights.append((q, s))
        ref_w = (weights[0][0].float() * weights[0][1].float().repeat_interleave(128, 0)[:rows]
                 .repeat_interleave(128, 1)[:, :k]).bfloat16()
        print(f"## N={n},K={k},glu={int(glu)} ({len(weights)} weight copies)", flush=True)
        for m in (int(b) for b in args.buckets.split(",")):
            x = torch.randn(m, k, device=device).bfloat16()
            want = F.linear(x, ref_w).float()
            if glu:
                g, u = want.chunk(2, dim=-1)
                want = F.silu(g) * u
            line = {"N": n, "K": k, "glu": int(glu), "M": m}
            for name, fn in impls.items():
                def call(q, s, fn=fn, x=x, glu=glu):
                    h = fn(x, q, s)
                    return act(h) if glu else h
                try:
                    out = call(*weights[0]).float()
                    err = float((out - want).norm() / want.norm())
                    us = time_graph(lambda call=call, weights=weights: [call(q, s) for q, s in weights],
                                    replays=10) / len(weights)
                except Exception as exc:  # noqa: BLE001 - a kernel that refuses this shape
                    print(f"   {name} M={m}: {exc!r}"[:200], flush=True)
                    err, us = float("nan"), float("nan")
                line[f"{name}_us"], line[f"{name}_err"] = round(us, 2), round(err, 4)
            report.append(line)
            print(f"{m:>6} " + " ".join(f"{name} {line[f'{name}_us']:>8.1f}" for name in impls), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
