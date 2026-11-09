"""Kimi K3.1's attention-residual read, per read, on one GPU: the torch chain the port ran
(token-major stack, ``torch.cat`` + fp32 copy + norm + score + softmax + einsum + casts, the
score weight cast per read) against the Triton kernels (``attn_res_kernels``), with and without
the residual add ahead of the read.

Timed as decode runs it: ``--reads`` reads (each its own site weights, one stack) captured in one
CUDA graph, replayed; microseconds per read. Sweeps the kernels' slice width and warps per path.

``--site``: a whole read site instead (``attn_res_read_norm``: the read and the RMSNorm after it, over the token-major
stack, with the residual add), the Triton read + norm against FlashInfer's one-launch kimi_k3_attn_res (SM100 /
SM103; PDL on as the model launches it).

    python benchmark/kimi_k3/attn_res_read.py --tokens 1,8,32,64,256,2048 --depths 1,4,8 --out res.jsonl
    python benchmark/kimi_k3/attn_res_read.py --site --tokens 1,2,4,8,16,32,64,512,3072 --depths 0,4,8
"""
from __future__ import annotations

import argparse
import itertools
import json

import torch

from mstar.model.kimi_k3.components import attn_res_kernels as K
from mstar.model.kimi_k3.components.attn_res import score_weight

D, EPS = 7168, 1e-6


def chain_read(prefix, blocks_tm, norm_w, proj_w):
    """The read on k31-marlin (7ae0ded88): ``(T, m, D)`` stack, per-read score weight."""
    sw = score_weight(norm_w, proj_w)
    with torch.autocast(device_type="cuda", enabled=False):
        v = torch.cat((blocks_tm, prefix.unsqueeze(1)), dim=1).float()
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
        probs = (k * sw).sum(-1).softmax(-1)
        mixed = torch.einsum("tm,tmd->td", probs, v)
    return mixed.to(prefix.dtype)


def time_reads(read, reads: int, replays: int) -> float:
    """Microseconds per read: ``reads`` calls of ``read(i)`` in one graph, replayed."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(reads):
            read(i)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(reads):
            read(i)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (replays * reads)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="1,8,32,64,256,2048")
    p.add_argument("--depths", default="1,4,8")
    p.add_argument("--reads", type=int, default=20)
    p.add_argument("--replays", type=int, default=50)
    p.add_argument("--out", default=None)
    p.add_argument("--auto-only", action="store_true", help="the chain and the default launch only")
    p.add_argument("--site", action="store_true", help="read + norm: the Triton read against FlashInfer's")
    args = p.parse_args()
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    if args.site:
        return bench_sites(args, g)
    print(f"# {torch.cuda.get_device_name()} torch {torch.__version__} PDL {K._pdl(dev)}")
    sink = open(args.out, "a") if args.out else None
    for T, m in itertools.product(map(int, args.tokens.split(",")), map(int, args.depths.split(","))):
        row = bench_shape(T, m, args.reads if T <= 256 else max(2, args.reads // 8),
                          args.replays if T <= 256 else 10, g, args.auto_only)
        print(f"T={T:5d} m={m}: chain {row['chain']:8.2f} (+add {row['chain+add']:8.2f}) | "
              f"split {row.get('split', 0):7.2f} [{row.get('split_cfg')}] +add {row.get('split+add', 0):7.2f} "
              f"[{row.get('split+add_cfg')}] | fused {row.get('fused', 0):7.2f} [{row.get('fused_cfg')}] "
              f"+add {row.get('fused+add', 0):7.2f} [{row.get('fused+add_cfg')}] | auto {row['auto']:.2f} "
              f"+add {row['auto+add']:.2f} us/read", flush=True)
        if sink:
            sink.write(json.dumps(row) + "\n")
            sink.flush()


SPLIT_GRID = list(itertools.product((256, 512, 1024), (2, 4, 8)))
FUSED_GRID = list(itertools.product((512, 1024, 2048), (4, 8, 16)))


def bench_shape(T: int, m: int, reads: int, replays: int, g, auto_only: bool = False) -> dict:
    dev = torch.device("cuda")
    stack = torch.randn(8, T, D, device=dev, generator=g).bfloat16()
    blocks_tm = stack[:m].transpose(0, 1).contiguous()  # the chain's token-major stack
    prefix = (3 * torch.randn(T, D, device=dev, generator=g)).bfloat16()
    addend = torch.randn(T, D, device=dev, generator=g).bfloat16()
    nws = [(1 + 0.2 * torch.randn(D, device=dev, generator=g)).bfloat16() for _ in range(reads)]
    pws = [(0.05 * torch.randn(1, D, device=dev, generator=g)).bfloat16() for _ in range(reads)]
    want = chain_read(prefix + addend, blocks_tm, nws[0], pws[0])
    row = {"gpu": torch.cuda.get_device_name(), "T": T, "m": m}
    row["chain"] = round(time_reads(
        lambda i: chain_read(prefix, blocks_tm, nws[i], pws[i]), reads, replays), 2)
    row["chain+add"] = round(time_reads(
        lambda i: chain_read(prefix + addend, blocks_tm, nws[i], pws[i]), reads, replays), 2)

    def kernel(add, **kw):
        return lambda i: K.attn_res_read_triton(stack, m, prefix, nws[i], pws[i], EPS,
                                                addend if add else None, **kw)

    for path, grid in () if auto_only else (("split", SPLIT_GRID), ("fused", FUSED_GRID)):
        for add in (False, True):
            key, best = f"{path}{'+add' if add else ''}", None
            for ch, warps in grid:
                read = kernel(add, path=path, ch=ch, num_warps=warps)
                try:
                    out, stream = read(0)
                    if add:
                        assert torch.equal(stream, prefix + addend)
                        err = (out.float() - want.float()).abs().max().item()
                        assert err <= 2 ** -6 * want.float().abs().max().item(), err
                    us = time_reads(read, reads, replays)
                except Exception as e:  # a config that does not compile or launch
                    print(f"  skip {key} ch={ch} warps={warps}: {type(e).__name__}: {str(e)[:120]}")
                    continue
                row[f"{key}/{ch}x{warps}"] = round(us, 2)
                if best is None or us < best[0]:
                    best = (us, ch, warps)
            if best:
                row[key], row[key + "_cfg"] = round(best[0], 2), f"{best[1]}x{best[2]}"
    for add in (False, True):
        key = "auto" + ("+add" if add else "")
        row[key] = round(time_reads(kernel(add), reads, replays), 2)
        row[key + "_cfg"] = "x".join(map(str, K.launch_config(T)))
    return row


def bench_sites(args, g) -> None:
    """``--site``: microseconds per read site (read + norm, the residual add in it), Triton against FlashInfer."""
    from mstar.model.kimi_k3.components import attn_res

    dev = torch.device("cuda")
    fi = attn_res._fi(dev) is not None
    print(f"# {torch.cuda.get_device_name()} torch {torch.__version__} FlashInfer kimi_k3_attn_res: {fi}")
    sink = open(args.out, "a") if args.out else None
    for T, m in itertools.product(map(int, args.tokens.split(",")), map(int, args.depths.split(","))):
        reads, replays = (args.reads, args.replays) if T <= 256 else (max(2, args.reads // 8), 10)
        row = site_row(T, m, reads, replays, g, fi)
        print(f"T={T:5d} m={m}: triton read + norm {row['site_triton+add']:7.2f} | flashinfer "
              f"{row.get('site_flashinfer+add', float('nan')):7.2f} us/site", flush=True)
        if sink:
            sink.write(json.dumps(row) + "\n")
            sink.flush()


def site_row(T: int, m: int, reads: int, replays: int, g, fi: bool) -> dict:
    from mstar.model.components.norm import RMSNorm
    from mstar.model.kimi_k3.components import attn_res

    dev = torch.device("cuda")
    stack = torch.randn(T, 8, D, device=dev, generator=g).bfloat16()
    prefix = (3 * torch.randn(T, D, device=dev, generator=g)).bfloat16()
    addend = (0.01 * torch.randn(T, D, device=dev, generator=g)).bfloat16()
    nws = [(1 + 0.2 * torch.randn(D, device=dev, generator=g)).bfloat16() for _ in range(reads)]
    pws = [(0.05 * torch.randn(1, D, device=dev, generator=g)).bfloat16() for _ in range(reads)]
    norms = [RMSNorm(D, eps=EPS).to(dev).bfloat16() for _ in range(reads)]
    row = {"gpu": torch.cuda.get_device_name(), "T": T, "m": m}
    for name, use_fi in (("triton", False), ("flashinfer", True)):
        if use_fi and not fi:
            continue

        def read(i, use_fi=use_fi):
            return attn_res.attn_res_read_norm(stack, m, prefix, nws[i], pws[i], EPS, norms[i], addend=addend,
                                               flashinfer=use_fi)

        row[f"site_{name}+add"] = round(time_reads(read, reads, replays), 2)
    return row


if __name__ == "__main__":
    main()
