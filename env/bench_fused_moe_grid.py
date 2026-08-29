#!/usr/bin/env python3
"""ONE GPU, seconds: the fp8 fused-MoE Triton GEMMs at GLM-5.2's decode shape,
grid sized from the worst-case padded slot count (today) vs clamped.

Today ``invoke_fused_moe_kernel_fp8_w8a8`` sizes its grid from
``sorted_token_ids.shape[0]`` = ``tokens*top_k + E*(BLOCK_M-1)`` = 32 + 256*15 =
3872 slots at k=3 decode, i.e. 242 M-tiles: 3,872 CTAs for the gate/up GEMM
and 46,464 for the down GEMM (N=6144, BLOCK_N=32), of which at most 32 M-tiles
carry a token. Every other CTA loads ``num_tokens_post_padded`` and exits —
but it still has to be scheduled, 75 layers per step. vLLM clamps ``EM`` for
small batches; the bound that is always valid is ``topk_ids.numel() * BLOCK_M``
(each (token, expert) slot opens at most one partial tile), here 512.

This emulates the clamp by slicing ``sorted_token_ids``/``expert_ids`` — the
kernel, the valid slots and the outputs are identical — and times both
launches eagerly and inside a CUDA graph (where they actually run). Run it on
any idle GPU:

    CUDA_VISIBLE_DEVICES=<idle> .venv/bin/python env/bench_fused_moe_grid.py
"""
import argparse

import torch
import triton

from mstar.utils.fused_moe import runner as R


def bench(fn, iters=200, graph=False):
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                fn()
        run = g.replay
        per = 20
    else:
        run = fn
        per = 1
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        run()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / (iters * per)  # us per call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4, help="rows in the trunk (k+1)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--inter", type=int, default=256, help="moe_intermediate / TP")
    ap.add_argument("--layers", type=int, default=75, help="only to scale the per-step line")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    E, H, I, T, K = a.experts, a.hidden, a.inter, a.tokens, a.top_k
    bk = bn = 128
    fp8 = R.FP8_DTYPE
    w1 = (torch.randn(E, 2 * I, H, device=dev) * 0.05).to(fp8)
    w2 = (torch.randn(E, H, I, device=dev) * 0.05).to(fp8)
    w1s = torch.rand(E, -(-2 * I // bn), H // bk, device=dev) * 0.01 + 0.01
    w2s = torch.rand(E, -(-H // bn), I // bk, device=dev) * 0.01 + 0.01
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(T, E, device=dev)
    topk_w, topk_ids = torch.topk(logits.sigmoid(), K, dim=-1)
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(torch.bfloat16)
    topk_ids = topk_ids.to(torch.int32).contiguous()

    cfg = R.get_default_config(M=T, E=E, N=2 * I, K=H, top_k=K)
    cfg["BLOCK_SIZE_K"] = bk
    bm = cfg["BLOCK_SIZE_M"]
    sorted_ids, expert_ids, n_post = R.moe_align_block_size(topk_ids, bm, E)
    n_valid = int(n_post.item())
    em_full = sorted_ids.shape[0]
    em_clamp = min(em_full, topk_ids.numel() * bm)
    assert n_valid <= em_clamp, (n_valid, em_clamp)
    ct = R._tl_compute_type(x.dtype)

    a_q, a_s = R.per_token_group_quant_fp8(x, bk)
    c1 = torch.empty(T * K, 2 * I, device=dev, dtype=x.dtype)
    c2 = torch.empty(T * K, I, device=dev, dtype=x.dtype)
    c3 = torch.empty(T, K, H, device=dev, dtype=x.dtype)
    R.act_and_mul_triton(c1.zero_(), c2, activation="silu")
    a2_q, a2_s = R.per_token_group_quant_fp8(c2, bk)

    def up(em):
        R.invoke_fused_moe_kernel_fp8_w8a8(
            A=a_q, B=w1, C=c1, A_scale=a_s, B_scale=w1s, topk_weights=topk_w,
            topk_ids=topk_ids, sorted_token_ids=sorted_ids[:em],
            expert_ids=expert_ids[: triton.cdiv(em, bm)], num_tokens_post_padded=n_post,
            mul_routed_weight=False, top_k=K, config=cfg, compute_type=ct, block_shape=(bn, bk))

    def down(em):
        R.invoke_fused_moe_kernel_fp8_w8a8(
            A=a2_q, B=w2, C=c3.view(T * K, H), A_scale=a2_s, B_scale=w2s, topk_weights=topk_w,
            topk_ids=topk_ids, sorted_token_ids=sorted_ids[:em],
            expert_ids=expert_ids[: triton.cdiv(em, bm)], num_tokens_post_padded=n_post,
            mul_routed_weight=True, top_k=1, config=cfg, compute_type=ct, block_shape=(bn, bk))

    # correctness of the emulated clamp: identical outputs
    up(em_full); ref1 = c1.clone(); up(em_clamp); assert torch.equal(ref1, c1)
    down(em_full); ref3 = c3.clone(); down(em_clamp); assert torch.equal(ref3, c3)

    ctas = lambda em, n: triton.cdiv(em, bm) * triton.cdiv(n, cfg["BLOCK_SIZE_N"])
    print(f"shape: tokens={T} top_k={K} E={E} H={H} I/rank={I} | BLOCK_M={bm} BLOCK_N={cfg['BLOCK_SIZE_N']} "
          f"| valid slots {n_valid} | EM full {em_full} -> clamp {em_clamp}")
    print(f"{'launch':<12}{'EM':>6}{'CTAs':>8}{'eager us':>10}{'graph us':>10}")
    total = {}
    for name, fn, n in (("up (gate)", up, 2 * I), ("down", down, H)):
        for tag, em in (("full", em_full), ("clamp", em_clamp)):
            e = bench(lambda: fn(em)); g = bench(lambda: fn(em), graph=True)
            total[tag] = total.get(tag, 0.0) + g
            print(f"{name:<12}{em:>6}{ctas(em, n):>8}{e:>10.1f}{g:>10.1f}")
    d = total["full"] - total["clamp"]
    print(f"\nper layer (graph): full {total['full']:.1f} us, clamp {total['clamp']:.1f} us, delta {d:.1f} us"
          f" -> x{a.layers} layers = {d*a.layers/1000:.2f} ms per decode step")


if __name__ == "__main__":
    main()
