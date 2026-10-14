#!/usr/bin/env python3
"""Micro-benchmark of the attention kernels available for the klein DiT's joint attention.

Shapes are the 1024x1024 text-to-image step (512 text + 4096 image tokens, 24 heads x 128)
at a few batch sizes. Compares torch SDPA (flash / cuDNN / efficient backends) with the
FlashInfer ragged prefill wrapper (fa2 / fa3 on Hopper) that the engine's ragged
attention resource runs. Reports median kernel time and max-abs difference against the
SDPA math backend in bf16.

    python benchmark/flux2_klein/attention_microbench.py --tokens 4608 --heads 24 --batch 1 4
"""

from __future__ import annotations

import argparse
import functools
import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _sdpa(qt, kt, vt):
    return F.scaled_dot_product_attention(qt, kt, vt)


def _time(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4608)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    args = ap.parse_args()
    device = torch.device("cuda")
    for bsz in args.batch:
        q, k, v = (torch.randn(bsz, args.tokens, args.heads, args.head_dim, device=device, dtype=torch.bfloat16)
                   for _ in range(3))
        qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(qt.float(), kt.float(), vt.float()).transpose(1, 2)
        print(f"== batch {bsz} x {args.tokens} tokens x {args.heads}h x {args.head_dim}d")
        for name, backend in (("sdpa/flash", SDPBackend.FLASH_ATTENTION), ("sdpa/cudnn", SDPBackend.CUDNN_ATTENTION),
                              ("sdpa/efficient", SDPBackend.EFFICIENT_ATTENTION)):
            try:
                with sdpa_kernel(backend):
                    out = _sdpa(qt, kt, vt).transpose(1, 2)
                    ms = _time(functools.partial(_sdpa, qt, kt, vt))
                print(f"  {name:16s} {ms:8.3f} ms   max_abs vs fp32 math {(out.float() - ref).abs().max().item():.3e}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:16s} unavailable: {type(exc).__name__}: {str(exc)[:80]}")
        try:
            import flashinfer

            workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            cu = torch.arange(0, (bsz + 1) * args.tokens, args.tokens, dtype=torch.int32, device=device)
            for backend in ("fa2", "fa3"):
                try:
                    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD", backend=backend)
                    wrapper.plan(cu, cu, args.heads, args.heads, args.head_dim, causal=False,
                                 q_data_type=torch.bfloat16)
                    packed = tuple(t.reshape(bsz * args.tokens, args.heads, args.head_dim) for t in (q, k, v))
                    out = wrapper.run(*packed).view(bsz, args.tokens, args.heads, args.head_dim)
                    ms = _time(functools.partial(wrapper.run, *packed))
                    diff = (out.float() - ref).abs().max().item()
                    print(f"  flashinfer/{backend:5s} {ms:8.3f} ms   max_abs vs fp32 math {diff:.3e}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  flashinfer/{backend:5s} unavailable: {type(exc).__name__}: {str(exc)[:80]}")
        except ImportError:
            print("  flashinfer not installed")


if __name__ == "__main__":
    main()
