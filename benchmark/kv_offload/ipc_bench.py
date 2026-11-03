"""CUDA IPC KV transfer microbenchmark: one request's KV read from another
process's cache, through ``CudaIpcKVTransferEngine._do_read``.

A writer process holds a KV cache on ``--src-device`` and publishes its IPC
descriptor; the reader, on ``--dst-device``, pulls ``--tokens`` tokens of it
(every layer, the last page partial) the way a prefill -> decode handoff does.
For each copier that loads here, and for ``main``'s one ``copy_`` per (layer,
page) as the baseline, it reports the time from the read starting to its pages
being written, optionally with a GEMM loop running on the reader's GPU
(``--busy``), which a copy kernel has to share SMs with and a copy-engine
memcpy does not.

Defaults are BAGEL-7B's LLM at TP1: 28 layers, 4 KV heads of 128, 128 tokens a
page, bf16 (7.3 MB a page). Median of ``--repeats``.

    python benchmark/kv_offload/ipc_bench.py --tokens 512,2500,8000
"""

import argparse
import statistics
import threading
import time

import torch
import torch.multiprocessing as mp

from mstar.engine.resources.kv import page_copier
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.transfer import CudaIpcKVTransferEngine, KVReadInfo


def kv_config(args, max_tokens):
    pages = -(-max_tokens // args.page_size) + 1
    return PagedKVConfig(
        num_layers=args.layers, num_kv_heads=args.kv_heads, head_dim=args.head_dim,
        max_seq_len=pages * args.page_size, max_num_pages=pages, page_size=args.page_size,
    )


def read_infos(cache, tokens):
    """What ``KVTransferManager.start_async_retrieve`` asks for: tokens [0,
    ``tokens``) of pages 0.. on both sides, every layer."""
    page_size, out = cache.page_size, []
    for page in range(-(-tokens // page_size)):
        end = min(page_size, tokens - page * page_size)
        out += [
            KVReadInfo(layer_idx=layer, local_page_idx=page, remote_page_idx=page,
                       token_start=0, token_end=end)
            for layer in range(cache.num_layers)
        ]
    return out


def writer(args, max_tokens, queue, done):
    device = torch.device(args.src_device)
    torch.cuda.set_device(device)
    cache = KVCache(kv_config(args, max_tokens), device)
    cache.tensor.normal_()
    engine = CudaIpcKVTransferEngine(cache)
    torch.cuda.synchronize()
    queue.put((engine.get_kv_transfer_info(), cache.tensor.float().sum(dim=(0, 2, 3, 4, 5)).cpu()))
    done.wait()
    engine.shutdown()


def baseline_read(engine, info, reads):
    """``main``'s loop: one ``copy_`` per (layer, page) on the thread's current
    stream, then wait for them."""
    tensor = _rebuild(info)  # as the engine does, once a read
    for r in reads:
        src = engine._kv_cache.chunk_view(r.layer_idx, r.remote_page_idx, r.token_start, r.token_end, tensor=tensor)
        dst = engine._kv_cache.chunk_view(r.layer_idx, r.local_page_idx, r.token_start, r.token_end)
        dst.copy_(src)
    torch.cuda.current_stream().synchronize()


def busy_loop(device, stop):
    torch.cuda.set_device(device)
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    with torch.cuda.stream(torch.cuda.Stream(device)):
        while not stop.is_set():
            for _ in range(8):
                a = (a @ a).clamp_(-1, 1)
            torch.cuda.current_stream().synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="512,2500,8000")
    p.add_argument("--layers", type=int, default=28)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=128)
    p.add_argument("--src-device", default="cuda:0")
    p.add_argument("--dst-device", default="cuda:1")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--busy", action="store_true", help="run a GEMM loop on the reader's GPU")
    args = p.parse_args()
    token_counts = [int(t) for t in args.tokens.split(",")]

    ctx = mp.get_context("spawn")
    queue, done = ctx.Queue(), ctx.Event()
    proc = ctx.Process(target=writer, args=(args, max(token_counts), queue, done))
    proc.start()
    info, page_sums = queue.get()

    device = torch.device(args.dst_device)
    torch.cuda.set_device(device)
    cache = KVCache(kv_config(args, max(token_counts)), device)
    engine = CudaIpcKVTransferEngine(cache)

    copiers = {"main (copy_ per layer, page)": None}
    for cls in (page_copier.CudaBindingsCopier, page_copier.CudartCopier, page_copier.TorchCopier):
        try:
            copiers[cls.name] = cls()
        except (ImportError, OSError, AttributeError, RuntimeError) as e:
            print(f"{cls.name}: unavailable ({e})")

    stop = threading.Event()
    busy = threading.Thread(target=busy_loop, args=(device, stop), daemon=True)
    if args.busy:
        busy.start()
        time.sleep(1.0)

    page_bytes = cache.tensor[:, 0].numel() * cache.tensor.element_size()
    print(f"{args.layers} layers, {page_bytes / 1e6:.1f} MB a page, "
          f"{args.src_device} -> {args.dst_device}{', GEMM loop on the reader' if args.busy else ''}")
    for tokens in token_counts:
        reads = read_infos(cache, tokens)
        nbytes = sum(r.token_end - r.token_start for r in reads) * page_bytes // (args.layers * args.page_size)
        for name, copier in copiers.items():
            times = []
            for _ in range(args.repeats):
                cache.tensor.zero_()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if copier is None:
                    baseline_read(engine, info, reads)
                else:
                    engine._copier = copier
                    event = torch.cuda.current_stream().record_event()
                    engine._do_read(info, reads, event)
                times.append(time.perf_counter() - t0)
            full = tokens // args.page_size
            got = cache.tensor.float().sum(dim=(0, 2, 3, 4, 5)).cpu()
            assert torch.allclose(got[:full], page_sums[:full], rtol=1e-3), name
            ms = statistics.median(times) * 1e3
            print(f"  {tokens:6d} tokens  {name:30s} {ms:8.2f} ms  {nbytes / ms / 1e6:6.1f} GB/s")
    stop.set()
    done.set()
    proc.join()


def _rebuild(info):
    from torch.multiprocessing.reductions import rebuild_cuda_tensor
    dtype = getattr(torch, info.dtype.split(".")[-1])
    return rebuild_cuda_tensor(
        torch.Tensor, info.size, info.stride, info.offset, torch.UntypedStorage,
        dtype, *info.cuda_share[:4], info.requires_grad, *info.cuda_share[4:],
    )


if __name__ == "__main__":
    main()
