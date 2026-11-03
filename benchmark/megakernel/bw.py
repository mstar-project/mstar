"""Copy bandwidth and plain GEMV times on one GPU, sustained.

copy: a 1 GiB and a 2 GiB bf16 tensor copied with `copy_` (bytes = read + write), back to back for ~2 s,
tail median. read: a sum over the same tensors (bytes = read). GEMV: torch.mm at the Qwen3.5 0.8B/2B
decode shapes, 1/2/4 rows, each weight distinct per call (a ring of copies bigger than L2) so nothing
is served from L2, inside a CUDA graph of 50 calls.

usage: python bw.py [--seconds 2]
"""
# Each closure below is timed inside the loop iteration that builds it.
# ruff: noqa: B023
import argparse
import json
import statistics as st

import torch


def timed(fn, seconds, inner=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    total = 0.0
    while total < seconds * 1000:
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        e.synchronize()
        ms = s.elapsed_time(e)
        total += ms
        out.append(ms / inner)
    tail = out[len(out) // 2:]
    return st.median(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(dev)
    res = {"device": p.name, "sms": p.multi_processor_count, "l2_mib": p.L2_cache_size / 2**20}
    print(res)
    for gib in (1, 2):
        n = gib * 2**30 // 2
        a = torch.randn(n, device=dev, dtype=torch.bfloat16)
        b = torch.empty_like(a)
        ms = timed(lambda: b.copy_(a), args.seconds, inner=5)
        res[f"copy_{gib}g_gbs"] = 2 * a.nbytes / ms / 1e6
        ms = timed(lambda: torch.sum(a, dtype=torch.float32), args.seconds, inner=5)
        res[f"read_{gib}g_gbs"] = a.nbytes / ms / 1e6
        print(f"{gib} GiB: copy {res[f'copy_{gib}g_gbs']:.0f} GB/s (r+w), read(sum) {res[f'read_{gib}g_gbs']:.0f} GB/s")
    torch.cuda.empty_cache()
    shapes = {
        "0.8b gate_up 7168x1024": (7168, 1024), "0.8b down 1024x3584": (1024, 3584),
        "0.8b gdn in 8224x1024": (8224, 1024), "0.8b gdn out 1024x2048": (1024, 2048),
        "0.8b attn qkv 5120x1024": (5120, 1024), "0.8b lm_head 248320x1024": (248320, 1024),
        "2b gate_up 12288x2048": (12288, 2048), "2b down 2048x6144": (2048, 6144),
    }
    for name, (n_out, k) in shapes.items():
        nbytes = n_out * k * 2
        ring = max(2, int(4 * p.L2_cache_size // nbytes) + 1)
        ring = min(ring, 64)
        ws = [torch.randn(n_out, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(ring)]
        for rows in (1, 2, 4):
            x = torch.randn(rows, k, device=dev, dtype=torch.bfloat16)
            torch.mm(x, ws[0].t())
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            calls = 50
            with torch.cuda.graph(g):
                for i in range(calls):
                    torch.mm(x, ws[i % ring].t())
            ms = timed(g.replay, args.seconds, inner=2) / calls
            us = ms * 1000
            gbs = nbytes / (us * 1e-6) / 1e9
            res[f"gemv {name} r{rows}"] = {"us": us, "gbs": gbs}
            print(f"{name:28s} rows {rows}: {us:7.2f} us  {gbs:6.0f} GB/s (ring {ring})")
        del ws
        torch.cuda.empty_cache()
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
