"""Routed experts per token count, bf16 against block fp8, at a MoE's per-rank shapes: which fp8 path each count
should take. CUDA-graph replays of one call per routing in a rotation (each its own experts, so weights stream
from HBM as in a decode step); one JSON line per (shape, tokens, path) with microseconds, the ratio to bf16, and
the relative L2 distance from the fp32 output of the dequantized experts.

Paths: ``bf16`` (fused_experts: per-pair GEMVs up to decode.ROUTED_MAX_TOKENS, grouped GEMMs above, as the bf16
model runs them), ``fp8`` (fused_experts_fp8, the W8A8 grouped runner the fp8 experts run at every count),
``fp8_pair`` (decode.routed_experts_fp8: per-pair GEMVs over the fp8 experts at the GPU's built-in decode tiles),
and with ``--sweep`` ``fp8_pair_best`` (the fastest per-pair tiles up to 16 tokens). ``--profile 1,8`` prints the
GPU time per kernel of the bf16, fp8 and fp8_pair calls at those counts.

    python -m benchmark.quantization.bench_experts --shapes omni_thinker --tokens 1,2,4,8,16,32,64 --sweep \\
        --profile 1,8 --out experts.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from dataclasses import replace

import torch
import torch.nn.functional as F

# (experts, hidden, per-rank intermediate, top-k)
SHAPES = {
    "omni_thinker": (128, 2048, 768, 8),
    "omni_talker": (128, 1024, 384, 6),
    "glm52_tp8": (256, 6144, 256, 8),
}
ROTATIONS = 8
BLOCK = (128, 128)


def _block_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w (E, N, K)`` as e4m3 bytes (uint8) and fp32 scales per 128 x 128 block."""
    e, n, k = w.shape
    tiles = w.float().reshape(e, n // 128, 128, k // 128, 128)
    scale = (tiles.abs().amax(dim=(2, 4)) / 448.0).clamp_min(1e-12)
    q = (tiles / scale[:, :, None, :, None]).to(torch.float8_e4m3fn).reshape(e, n, k)
    return q.view(torch.uint8).contiguous(), scale.contiguous()


def _dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.view(torch.float8_e4m3fn).float() * s.repeat_interleave(128, 1).repeat_interleave(128, 2)


def experts(e, hidden, inter, dev, seed=0):
    """Block-fp8 experts ``(w13, s13, w2, s2)`` and their dequantized bf16 twins ``(w13, w2)``."""
    torch.manual_seed(seed)
    fp8, bf16 = [], []
    for n, k in ((2 * inter, hidden), (hidden, inter)):
        q, s = zip(*(_block_fp8(torch.randn(16, n, k, device=dev) * 0.02) for _ in range(e // 16)), strict=True)
        q, s = torch.cat(q), torch.cat(s)
        fp8 += [q, s]
        bf16.append(torch.cat([_dequant(q[i:i + 16], s[i:i + 16]).bfloat16() for i in range(0, e, 16)]))
    return tuple(fp8), tuple(bf16)


def reference(x, w13, w2, tw, ids):
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in ids.unique().tolist():
        tok, slot = torch.where(ids == e)
        g, u = (x[tok].float() @ w13[e].float().T).chunk(2, dim=-1)
        out.index_add_(0, tok, ((F.silu(g) * u) @ w2[e].float().T) * tw[tok, slot, None])
    return out


def rel_l2(got, want):
    got = got.float()
    return float("inf") if not torch.isfinite(got).all() else ((got - want).norm() / want.norm()).item()


def pair_candidates():
    gate_up = [dict(bj=bj, bk=bk, warps=w)
               for bj, bk, w in itertools.product((4, 8, 16, 32), (256, 512, 1024), (2, 4, 8))]
    down = [dict(bn=bn, bc=bc, warps=w) for bn, bc, w in itertools.product((16, 32, 64), (128, 256), (4, 8))]
    return gate_up, down


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", default="omni_thinker")
    p.add_argument("--tokens", default="1,2,4,8,16,32,64")
    p.add_argument("--sweep", action="store_true", help="per-pair tile sweep up to 16 tokens")
    p.add_argument("--profile", default="", help="token counts to print a per-kernel GPU time breakdown for")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = torch.device("cuda", torch.cuda.current_device())
    gpu = torch.cuda.get_device_name(dev)
    out = open(args.out, "a") if args.out else None  # noqa: SIM115

    def log(row):
        row = {"gpu": gpu, **row}
        print(json.dumps(row), flush=True)
        if out:
            out.write(json.dumps(row) + "\n")
            out.flush()

    profile_at = {int(t) for t in args.profile.split(",") if t}
    for shape in args.shapes.split(","):
        e, hidden, inter, top_k = SHAPES[shape]
        fp8, bf16 = experts(e, hidden, inter, dev)
        for t in (int(v) for v in args.tokens.split(",")):
            head = {"shape": shape, "E": e, "H": hidden, "I": inter, "K": top_k, "tokens": t}
            _one_count(log, head, fp8, bf16, dev, sweep=args.sweep and t <= 16, profile=t in profile_at)


def _one_count(log, head, fp8, bf16, dev, *, sweep, profile):
    from benchmark.kernels.timing import time_graph
    from mstar.utils.fused_moe import decode, fused_experts, fused_experts_fp8

    (w13, s13, w2, s2), (b13, b2) = fp8, bf16
    e, hidden, inter, top_k, t = (head[k] for k in ("E", "H", "I", "K", "tokens"))
    builtin = decode._tuning(dev, (hidden, e, inter, top_k))
    gen = torch.Generator(device=dev).manual_seed(t)
    x = torch.randn(t, hidden, device=dev, generator=gen).bfloat16()
    routes = []
    for _ in range(ROTATIONS):
        ids = torch.rand(t, e, device=dev, generator=gen).topk(top_k, dim=-1).indices
        tw = torch.rand(t, top_k, device=dev, generator=gen)
        routes.append((tw / tw.sum(-1, keepdim=True), ids))
    want = reference(x, b13, b2, *routes[0])

    def pair(tw, ids, tuning=builtin):
        return decode.routed_experts_fp8(x, w13, s13, w2, s2, tw, ids, block_size=BLOCK, tuning=tuning)

    paths = {
        "bf16": lambda tw, ids: fused_experts(x, b13, b2, tw, ids),
        "fp8": lambda tw, ids: fused_experts_fp8(x, w13, w2, s13, s2, tw, ids, block_size=BLOCK),
        "fp8_pair": pair,
    }

    def timed(fn):
        return time_graph(lambda: [fn(*r) for r in routes], replays=20) / ROTATIONS

    base = None
    for name, fn in paths.items():
        try:
            err = rel_l2(fn(*routes[0]), want)
            us = timed(fn)
        except Exception as exc:  # noqa: BLE001 - a path that can't take this count
            log({**head, "path": name, "skipped": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        base = us if name == "bf16" else base
        log({**head, "path": name, "us": round(us, 2), "vs_bf16": round(base / us, 3) if base else None,
             "rel_l2": round(err, 4)})
    if sweep:
        gu0, dn0 = decode._tiles(t, builtin)
        best = (gu0, dn0, float("inf"))
        gate_ups, downs = pair_candidates()
        # gate/up tiles at the built-in down tiles, then down tiles at the best gate/up
        for stage, options in (("gate_up", gate_ups), ("down", downs)):
            for cand in options:
                gu, dn = (cand, best[1]) if stage == "gate_up" else (best[0], cand)
                tuning = replace(builtin, pair_max_tokens=64, gate_up=((None, gu),), down=((None, dn),))
                try:
                    if rel_l2(pair(*routes[0], tuning=tuning), want) > 0.02:
                        continue
                    us = timed(lambda tw, ids, tuning=tuning: pair(tw, ids, tuning))
                except Exception:  # noqa: BLE001 - a tile the kernel's asserts or Triton reject
                    continue
                if us < best[2]:
                    best = (gu, dn, us)
        if best[2] < float("inf"):
            log({**head, "path": "fp8_pair_best", "us": round(best[2], 2),
                 "vs_bf16": round(base / best[2], 3) if base else None, "gate_up": best[0], "down": best[1],
                 "builtin": {"gate_up": gu0, "down": dn0}})
    if profile:
        for name, fn in paths.items():
            try:
                _profile(log, head, name, lambda fn=fn: fn(*routes[0]))
            except Exception as exc:  # noqa: BLE001 - the timings above stand without the breakdown
                log({**head, "profile": name, "error": f"{type(exc).__name__}: {exc}"[:200]})


def _profile(log, head, name, fn, iters=20):
    """GPU microseconds per call of each kernel ``fn`` launches (eager, so launch gaps are not counted)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per_kernel = defaultdict(float)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            per_kernel[ev.name[:60]] += ev.device_time / iters
    log({**head, "profile": name, "kernels": len(per_kernel), "gpu_us": round(sum(per_kernel.values()), 2),
         "per_kernel_us": {k: round(v, 2) for k, v in sorted(per_kernel.items(), key=lambda kv: -kv[1])}})


if __name__ == "__main__":
    main()
