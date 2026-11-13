"""``moe_align_block_size`` (the CUDA op) per call at K3.1's routing (896 experts, top-16), timed from a CUDA
graph of 20 calls, and whether 20 calls on the same ids agree.

    python benchmark/kimi_k3/moe_align.py --tokens 1,64,1024,8192 --experts 896 --block-size 16
"""
import argparse
import json

import torch

from mstar.utils.fused_moe.align import _cuda_op_available, moe_align_block_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="1,64,1024,8192")
    p.add_argument("--experts", type=int, default=896)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--replays", type=int, default=50)
    args = p.parse_args()
    assert _cuda_op_available()
    for T in map(int, args.tokens.split(",")):
        g = torch.Generator().manual_seed(T)
        ids = torch.stack([torch.randperm(args.experts, generator=g)[:args.top_k] for _ in range(T)])
        ids = ids.to(torch.int32).cuda()
        first = moe_align_block_size(ids, args.block_size, args.experts)
        same = all(all(torch.equal(a, b) for a, b in zip(moe_align_block_size(ids, args.block_size, args.experts),
                                                          first, strict=True)) for _ in range(args.calls))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            moe_align_block_size(ids, args.block_size, args.experts)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(args.calls):
                moe_align_block_size(ids, args.block_size, args.experts)
        graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / (args.replays * args.calls)
        print(json.dumps({"gpu": torch.cuda.get_device_name(), "tokens": T, "experts": args.experts,
                          "top_k": args.top_k, "block_size": args.block_size, "us_per_call": round(us, 2),
                          "same_on_every_call": same}), flush=True)


if __name__ == "__main__":
    main()
