"""Linear formats against bf16 at a model's per-rank GEMM shapes, each call replayed in a CUDA graph.

    python -m benchmark.quantization.bench_linears --formats fp8_tensor,fp8_channel,fp8_block \\
        --shapes qwen3_5_27b --tp 1 --tokens 1,16,64,256,1024,4096 --out linears.jsonl

One row per (format, shape, tokens): median microseconds of a replay over ``--iters`` calls, the speedup over
F.linear in bf16, and the output's relative L2 distance from it. A format the GPU can't run is skipped with
the reason. Times are the GEMM and the activation quantize together, as a served linear runs them.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

from mstar.model.components.quant_linear import as_container, linear_format

# per-rank [N, K] at TP 1 (divide N for column-parallel, K for row-parallel by --tp)
SHAPES = {
    # hidden 5120, q 24 heads x 256 x 2 (gated), kv 4 x 256, inter 17408, delta-net value 6144
    "qwen3_5_27b": {"qkv": (14336, 5120, "col"), "o": (5120, 6144, "row"), "gate_up": (34816, 5120, "col"),
                    "down": (5120, 17408, "row"), "gdn_out": (5120, 6144, "row")},
    "qwen3_5_9b": {"qkv": (6144, 4096, "col"), "o": (4096, 4096, "row"), "gate_up": (24576, 4096, "col"),
                   "down": (4096, 12288, "row")},
    # Qwen3-Omni thinker: attention only (its experts run fused_experts_fp8)
    "qwen3_omni_thinker": {"qkv": (5120, 2048, "col"), "o": (2048, 4096, "row")},
}


def _time(fn, iters: int, repeats: int = 5) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iters)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--formats", default="fp8_tensor,fp8_channel,fp8_block")
    parser.add_argument("--shapes", default="qwen3_5_27b", choices=sorted(SHAPES))
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--tokens", default="1,16,64,256,1024,4096")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    rows = []
    for layer, (n, k, kind) in SHAPES[args.shapes].items():
        n, k = (n // args.tp, k) if kind == "col" else (n, k // args.tp)
        weight = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
        for m in (int(t) for t in args.tokens.split(",")):
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            ref = F.linear(x, weight)
            base_us = _time(lambda: F.linear(x, weight), args.iters)  # noqa: B023
            rows.append({"layer": layer, "n": n, "k": k, "tokens": m, "format": "bf16", "us": round(base_us, 2)})
            for name in args.formats.split(","):
                fmt = linear_format(name)
                reason = fmt.unavailable_reason(device)
                try:
                    fmt.check(n, k, torch.bfloat16)
                except ValueError as e:
                    reason = reason or str(e)
                if reason:
                    rows.append({"layer": layer, "n": n, "k": k, "tokens": m, "format": name, "skipped": reason})
                    continue
                fmt.prepare_device(device)
                buffers = fmt.quantize_weight(weight, None)
                # as a served layer holds them: integer containers, viewed back
                stored = [as_container(buffers[b]).view(buffers[b].dtype) for b in fmt.buffers]
                out = fmt.linear(x, *stored)
                us = _time(lambda: fmt.linear(x, *stored), args.iters)  # noqa: B023
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                rows.append({"layer": layer, "n": n, "k": k, "tokens": m, "format": name, "us": round(us, 2),
                             "speedup": round(base_us / us, 3), "rel_l2": round(err, 4)})
            for row in rows[-(1 + len(args.formats.split(","))):]:
                print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps({"gpu": torch.cuda.get_device_name(), "shapes": args.shapes, "tp": args.tp,
                                    **row}) + "\n")


if __name__ == "__main__":
    main()
