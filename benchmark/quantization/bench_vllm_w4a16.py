"""vLLM's W4A16 GEMMs at the same shapes and token counts as ``bench_w4a16``, the reference column for ``w4a16``.
Runs in a vLLM environment (vLLM is not an M* dependency; nothing of M* is imported):

    PYTHONPATH=<this tree> <vllm venv>/bin/python -m benchmark.quantization.bench_vllm_w4a16 \\
        --shapes qwen3_5_27b --tp 1 --tokens 1,2,4,8,16,32,64,128,256,512,1024 --out w4a16.jsonl

What vLLM 0.33 runs for an int4 checkpoint on sm_90: Machete (``MacheteLinearKernel``, "Using MacheteLinearKernel
for AutoAWQMarlinLinearMethod" in its log for QuantTrio/Qwen3.5-27B-AWQ; group sizes divisible by 64), Marlin where
Machete can't (group 32, Kimi K2.x). Rows per (layer, tokens): ``vllm[machete]`` symmetric g128 (``uint4b8``),
``vllm[machete-zp]`` g128 with zero points (``uint4``, AWQ's), ``vllm[marlin]`` symmetric g128, ``vllm[marlin-g32]``
symmetric g32; timed like ``bench_linears`` (CUDA-graph replay) against the same bf16 F.linear.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from benchmark.quantization.bench_linears import SHAPES, _time


def _quant(w: torch.Tensor, group: int, zero: bool):
    """Round-to-nearest int4 of ``w [N, K]``: unsigned codes ``[N, K]``, scales ``[K/g, N]``, zero points or None, and
    the bf16 weight they stand for."""
    n, k = w.shape
    g = w.float().view(n, k // group, group)
    s = (g.abs().amax(-1) / 7.5).clamp_min(1e-8).to(torch.bfloat16)
    z = torch.randint(4, 12, s.shape, device=w.device) if zero else torch.full(s.shape, 8, device=w.device)
    u = (torch.round(g / s.float()[..., None]) + z[..., None]).clamp(0, 15).to(torch.int64)
    ref = ((u - z[..., None]).float() * s.float()[..., None]).view(n, k).to(torch.bfloat16)
    return u.view(n, k), s.t().contiguous(), (z.t().contiguous() if zero else None), ref


def _gptq_pack(u: torch.Tensor) -> torch.Tensor:
    """Codes ``[N, K]`` as GPTQ's ``[K/8, N]`` int32, eight consecutive K values a word, the first in the low bits."""
    v = u.t().reshape(-1, 8, u.shape[0])
    words = sum(v[:, j] << (4 * j) for j in range(8))
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", default="qwen3_5_27b", choices=sorted(SHAPES))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--tokens", default="1,2,4,8,16,32,64,128,256,512,1024")
    parser.add_argument("--layers", default="", help="only these of the shape set's layers (comma-separated)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
        marlin_make_empty,
        marlin_permute_scales,
    )
    from vllm.scalar_type import scalar_types

    dev = torch.device("cuda")
    torch.manual_seed(0)
    only = {name for name in args.layers.split(",") if name}
    rows = []
    for layer, (n, k, kind) in SHAPES[args.shapes].items():
        if only and layer not in only:
            continue
        n, k = (n // args.tp, k) if kind == "col" else (n, k // args.tp)
        w = (torch.randn(n, k, device=dev) * 0.02).to(torch.bfloat16)
        impls = {}
        for name, group, zero in (("machete", 128, False), ("machete-zp", 128, True), ("marlin", 128, False),
                                  ("marlin-g32", 32, False)):
            u, s, z, ref_w = _quant(w, group, zero)
            qw = _gptq_pack(u)
            try:
                if name.startswith("machete"):
                    btype = scalar_types.uint4 if zero else scalar_types.uint4b8
                    bq = ops.machete_prepack_B(qw.t().contiguous().t(), a_type=torch.bfloat16, b_type=btype,
                                               group_scales_type=torch.bfloat16)
                    zeros = None if z is None else (-1.0 * s.float() * z.float()).to(torch.bfloat16).contiguous()
                    def fn(x, bq=bq, bt=btype, s=s, zz=zeros, g=group):
                        return ops.machete_mm(a=x, b_q=bq, b_type=bt, b_group_scales=s, b_group_zeros=zz,
                                              b_group_size=g)
                else:
                    mq = ops.gptq_marlin_repack(qw, size_k=k, size_n=n, num_bits=4)
                    ms = marlin_permute_scales(s, size_k=k, size_n=n, group_size=group)
                    zp = marlin_make_empty(dev)
                    def fn(x, mq=mq, ms=ms, zp=zp, n=n, k=k):
                        return apply_gptq_marlin_linear(
                            input=x, weight=mq, weight_scale=ms, weight_zp=zp, workspace=None,
                            wtype=scalar_types.uint4b8, output_size_per_partition=n, input_size_per_partition=k,
                            input_dtype=torch.bfloat16)
                impls[name] = (fn, ref_w)
            except Exception as e:  # noqa: BLE001
                rows.append({"layer": layer, "n": n, "k": k, "format": f"vllm[{name}]", "engine": "vllm",
                             "skipped": f"{type(e).__name__}: {e}"[:200]})
                print(json.dumps(rows[-1]), flush=True)
        for m in (int(t) for t in args.tokens.split(",")):
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            base_us = _time(lambda: F.linear(x, w), args.iters)  # noqa: B023
            rows.append({"layer": layer, "n": n, "k": k, "tokens": m, "format": "bf16", "us": round(base_us, 2),
                         "engine": "vllm"})
            for name, (fn, ref_w) in impls.items():
                row = {"layer": layer, "n": n, "k": k, "tokens": m, "format": f"vllm[{name}]", "engine": "vllm"}
                try:
                    out = fn(x)
                    us = _time(lambda fn=fn: fn(x), args.iters)  # noqa: B023
                except Exception as e:  # noqa: BLE001
                    rows.append({**row, "skipped": f"{type(e).__name__}: {e}"[:200]})
                    print(json.dumps(rows[-1]), flush=True)
                    continue
                ref = F.linear(x, ref_w)
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                rows.append({**row, "us": round(us, 2), "speedup": round(base_us / us, 3), "rel_l2": round(err, 4)})
                print(json.dumps(rows[-1]), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps({"gpu": torch.cuda.get_device_name(), "shapes": args.shapes, "tp": args.tp,
                                    "vllm": vllm.__version__, **row}) + "\n")


if __name__ == "__main__":
    main()
