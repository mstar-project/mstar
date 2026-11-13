"""FastConformer's dense-bias attention: cuDNN against the other SDPA backends.

Two numbers per shape, for Parakeet-TDT-0.6B's attention (8 heads of 128, bf16) by default:

- ``graph``: microseconds per call replayed from a CUDA graph (``time_graph``), at every
  (mel bucket, batch size) the encoder node captures. This is the cost of each backend
  inside a captured bucket.
- ``first call``: host milliseconds for the first eager call at a length not seen before,
  which is what an eager clip past every bucket pays per layer.

``default`` is unpinned SDPA, which dispatches a dense bf16 bias to cuDNN where it can;
``no_cudnn`` is the same call under ``no_cudnn_sdpa()``.

    python -m benchmark.fastconformer_sdpa_backends [--heads 8 --head-dim 128] [--replays 200]
"""
from __future__ import annotations

import argparse
import contextlib
import statistics
import time

import torch
import torch.nn.functional as F

from benchmark.kernels.timing import time_graph
from mstar.model.components.fastconformer_submodule import MEL_BUCKETS, _batch_sizes
from mstar.model.components.sdpa_backends import no_cudnn_sdpa

SUBSAMPLING = 8
BACKENDS = {"default": contextlib.nullcontext, "no_cudnn": no_cudnn_sdpa}


def _inputs(bs: int, t: int, heads: int, head_dim: int, dtype: torch.dtype):
    q, k, v = (torch.randn(bs, heads, t, head_dim, device="cuda", dtype=dtype) for _ in range(3))
    bias = torch.randn(bs, heads, t, t, device="cuda", dtype=dtype)
    bias[0, :, :, t - t // 4:] = torch.finfo(dtype).min  # a padded row, as in a captured bucket
    return q, k, v, bias


def _graph_us(backend: str, bs: int, t: int, args, dtype) -> float:
    q, k, v, bias = _inputs(bs, t, args.heads, args.head_dim, dtype)

    def launch():
        with BACKENDS[backend]():
            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    return statistics.median(
        time_graph(launch, replays=args.replays, warmup=args.warmup) for _ in range(args.repeats)
    )


def _first_call_ms(backend: str, t: int, args, dtype) -> float:
    q, k, v, bias = _inputs(1, t, args.heads, args.head_dim, dtype)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with BACKENDS[backend]():
        F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--replays", type=int, default=200, help="graph replays per timing")
    parser.add_argument("--warmup", type=int, default=10, help="eager calls before capture")
    parser.add_argument("--repeats", type=int, default=5, help="timings per shape; the median is reported")
    parser.add_argument("--new-lengths", type=int, nargs="*", default=[600, 750, 900, 1100],
                        help="encoder lengths (mel / 8) for the first-call timing; each must be new to the process")
    args = parser.parse_args()
    dtype = torch.bfloat16
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, {args.heads}x{args.head_dim} bf16")

    # first calls run before any graph timing, so each backend sees every length fresh
    print("\nfirst eager call at a new length, host ms (bs=1)")
    print(f"{'T':>6} {'default':>10} {'no_cudnn':>10}")
    for t in args.new_lengths:
        # offset per backend, so neither inherits the other's build for the same shape
        d, n = _first_call_ms("default", t, args, dtype), _first_call_ms("no_cudnn", t + 1, args, dtype)
        print(f"{t:>6} {d:>10.2f} {n:>10.2f}")

    print(f"\ncaptured, us per call (median of {args.repeats} x {args.replays} replays)")
    print(f"{'mel':>6} {'T':>5} {'bs':>4} {'default':>10} {'no_cudnn':>10} {'faster':>9}")
    for bucket in MEL_BUCKETS:
        t = bucket // SUBSAMPLING
        for bs in _batch_sizes(bucket):
            d, n = (_graph_us(b, bs, t, args, dtype) for b in BACKENDS)
            print(f"{bucket:>6} {t:>5} {bs:>4} {d:>10.1f} {n:>10.1f} {'default' if d < n else 'no_cudnn':>9}")


if __name__ == "__main__":
    main()
