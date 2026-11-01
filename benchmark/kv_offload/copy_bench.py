"""KV offload/reload copy microbenchmark, through ``CPUPagePool`` itself.

For each copier that loads here (``cudaMemcpy2DAsync`` per page through
``cuda.bindings`` or through torch's libcudart, ``hipMemcpy2DAsync`` through
torch's libamdhip64 on ROCm, and the per-page ``copy_`` that is the fallback and
was the only path before), direction and page count:

- host: wall time from issuing the copies to ``sync()`` returning, with a
  ``--queued-ms`` kernel already queued on the compute stream. A path that
  waits for that kernel reports at least ``--queued-ms``.
- copy: CUDA-event time of the copies alone on the pool's stream, and the
  throughput that implies.

Defaults are one BAGEL-7B page per TP2 rank: 28 layers, 2 KV heads of 128,
128 tokens a page, bf16 (3.7 MB). Median of ``--repeats``.

    python benchmark/kv_offload/copy_bench.py
"""

import argparse
import statistics
import time

import torch

from mstar.engine.resources.kv import cpu_page_pool, page_copier
from mstar.engine.resources.kv.arena import PageArena
from mstar.engine.resources.kv.cache import KVCache, PageAllocator
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.cpu_page_pool import PageCopies


def sleep_cycles_per_ms() -> float:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda._sleep(100_000_000)
    e.record()
    torch.cuda.synchronize()
    return 100_000_000 / s.elapsed_time(e)


def measure(pool, kv, n, direction, queued_cycles):
    gpu = kv.tensor
    src, dst, host = list(range(1, n + 1)), list(range(n + 1, 2 * n + 1)), list(range(n))

    def stage():
        if direction == "reload":
            pool.copy_out(gpu, PageCopies(gpu=src, cpu=host))
            pool.sync()
        torch.cuda.synchronize()

    def run():
        if direction == "offload":
            pool.copy_out(gpu, PageCopies(gpu=src, cpu=host))
        else:
            pool.copy_in(gpu, PageCopies(gpu=dst, cpu=host))

    stage()
    torch.cuda._sleep(queued_cycles)
    start = time.perf_counter()
    run()
    pool.sync()
    host_ms = (time.perf_counter() - start) * 1e3
    torch.cuda.synchronize()

    stage()
    stream = pool._get_stream()
    stream.wait_stream(torch.cuda.current_stream())
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(stream)
    run()
    e.record(stream)
    torch.cuda.synchronize()
    return host_ms, s.elapsed_time(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--pages", default="1,4,16,48")
    ap.add_argument("--queued-ms", type=float, default=50.0)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    counts = [int(n) for n in args.pages.split(",")]
    cfg = PagedKVConfig(
        num_layers=args.layers, num_kv_heads=args.kv_heads, head_dim=args.head_dim,
        num_qo_heads=args.kv_heads, page_size=args.page_size,
        max_num_pages=2 * max(counts) + 1, max_seq_len=args.page_size * 2 * max(counts),
        cpu_offload_pages=max(counts),
    )
    kv = KVCache(cfg, torch.device("cuda"), getattr(torch, args.dtype))
    kv.tensor.normal_()
    pool = cpu_page_pool.CPUPagePool(cfg, kv, PageArena(kv_cache=None, allocator=PageAllocator(max(counts))))
    page_bytes = kv.tensor[:, 0].numel() * kv.tensor.element_size()
    queued = int(args.queued_ms * sleep_cycles_per_ms())

    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, "
          f"page {page_bytes / 1e6:.2f} MB ({args.layers} layers), "
          f"{args.queued_ms:g} ms queued ahead")
    copiers = []
    for cls in (page_copier.CudaBindingsCopier, page_copier.CudartCopier,
                page_copier.HipCopier, page_copier.TorchCopier):
        try:
            copiers.append(cls())
        except (ImportError, OSError, AttributeError, RuntimeError) as e:
            print(f"{cls.name}: unavailable ({e})")
    print(f"{'copier':14s} {'dir':7s} {'pages':>5s} {'host ms':>8s} {'copy ms':>8s} {'GB/s':>6s}")
    for copier in copiers:
        pool._copier = copier
        for direction in ("offload", "reload"):
            measure(pool, kv, counts[0], direction, 0)  # warm up
            for n in counts:
                runs = [measure(pool, kv, n, direction, queued) for _ in range(args.repeats)]
                host = statistics.median(r[0] for r in runs)
                copy = statistics.median(r[1] for r in runs)
                print(f"{copier.name:14s} {direction:7s} {n:5d} {host:8.2f} {copy:8.2f} "
                      f"{n * page_bytes / copy / 1e6:6.1f}")


if __name__ == "__main__":
    main()
