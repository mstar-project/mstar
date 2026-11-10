"""vLLM's block-fp8 linear at the same shapes and token counts as ``bench_linears``, the reference column for
``fp8_block``. Runs in a vLLM environment (vLLM is not an M* dependency; nothing of M* is imported):

    PYTHONPATH=<this tree> <vllm venv>/bin/python -m benchmark.quantization.bench_vllm_linears \\
        --shapes qwen3_5_27b --tp 1 --tokens 1,8,16,32,64,256 --out linears.jsonl

What vLLM 0.33 selects for a block-fp8 checkpoint on sm_90 (``FlashInferFp8DeepGEMMDynamicBlockScaledKernel``,
"Selected ... for Fp8LinearMethod" in its log): FlashInfer's ``fp8_blockscale_gemm_sm90`` (swapAB, bf16 in)
below 32 tokens, DeepGEMM's ``fp8_gemm_nt`` (per-token-group 1x128 fp8 activations) from 32, with the weight
scales post-processed for DeepGEMM (UE8M0 where vLLM enables it). Rows: ``vllm[auto]`` (that dispatch, the
custom op vLLM runs) and ``vllm[deepgemm]`` (DeepGEMM at every count), timed like ``bench_linears`` (CUDA-graph
replay) against the same bf16 F.linear.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from benchmark.quantization.bench_linears import SHAPES, _time

FP8 = torch.float8_e4m3fn


def _block_quant(w: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    tiles = w.float().reshape(n // block, block, k // block, block)
    scale = (tiles.abs().amax(dim=(1, 3)) / torch.finfo(FP8).max).clamp_min(1e-12)
    q = (tiles / scale[:, None, :, None]).clamp(-448, 448).to(FP8).reshape(n, k)
    return q, scale.contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", default="qwen3_5_27b", choices=sorted(SHAPES))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--tokens", default="1,8,16,32,64,256,1024")
    parser.add_argument("--layers", default="", help="only these of the shape set's layers (comma-separated)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import vllm
    import vllm.model_executor.kernels.linear.scaled_mm.flashinfer  # noqa: F401  (registers the custom ops)
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
        per_token_group_quant_fp8,
    )
    from vllm.utils.deep_gemm import fp8_gemm_nt, is_deep_gemm_e8m0_used

    e8m0 = is_deep_gemm_e8m0_used()
    device = torch.device("cuda")
    torch.manual_seed(0)
    rows = []
    only = {name for name in args.layers.split(",") if name}
    for layer, (n, k, kind) in SHAPES[args.shapes].items():
        if only and layer not in only:
            continue
        n, k = (n // args.tp, k) if kind == "col" else (n, k // args.tp)
        weight = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
        wq, ws = _block_quant(weight)
        dg_w, dg_s = deepgemm_post_process_fp8_weight_block(wq=wq, ws=ws, quant_block_shape=(128, 128), use_e8m0=e8m0)

        def auto(x):
            return torch.ops.vllm.dynamic_flashinfer_deepgemm_blockscale_gemm(x, dg_w, dg_s, 128, e8m0)  # noqa: B023

        def deepgemm(x):
            q, s = per_token_group_quant_fp8(x, 128, column_major_scales=True, use_ue8m0=e8m0)
            out = torch.empty(x.shape[0], n, dtype=torch.bfloat16, device=device)  # noqa: B023
            fp8_gemm_nt((q, s), (dg_w, dg_s), out, is_deep_gemm_e8m0_used=e8m0)  # noqa: B023
            return out

        for m in (int(t) for t in args.tokens.split(",")):
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            ref = F.linear(x, weight)
            base_us = _time(lambda: F.linear(x, weight), args.iters)  # noqa: B023
            rows.append({"layer": layer, "n": n, "k": k, "tokens": m, "format": "bf16", "us": round(base_us, 2),
                         "engine": "vllm"})
            for name, fn in (("vllm[auto]", auto), ("vllm[deepgemm]", deepgemm)):
                row = {"layer": layer, "n": n, "k": k, "tokens": m, "format": name, "engine": "vllm"}
                try:
                    out = fn(x)
                    us = _time(lambda: fn(x), args.iters)  # noqa: B023
                except Exception as e:  # noqa: BLE001
                    rows.append({**row, "skipped": f"{type(e).__name__}: {e}"[:200]})
                    print(json.dumps(rows[-1]), flush=True)
                    continue
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                rows.append({**row, "us": round(us, 2), "speedup": round(base_us / us, 3), "rel_l2": round(err, 4)})
                print(json.dumps(rows[-1]), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps({"gpu": torch.cuda.get_device_name(), "shapes": args.shapes, "tp": args.tp,
                                    "vllm": vllm.__version__, "e8m0": e8m0, **row}) + "\n")


if __name__ == "__main__":
    main()
