#!/usr/bin/env python3
"""Sweep Triton tile configs for the fp8 fused-MoE GEMMs at GLM-5.2's decode
shape and report the fastest per launch.  One GPU, minutes.

Companion to ``bench_fused_moe_grid.py``, which measures the grid-size clamp
at a fixed config; this script fixes the grid and sweeps the compile-time
tiles of ``fused_moe_kernel_fp8_w8a8`` -- ``BLOCK_SIZE_N``, ``GROUP_SIZE_M``,
``num_warps``, ``num_stages`` -- for the gate/up and down GEMMs separately.
``fused_experts_fp8`` now tunes gate/up and down separately at decode shapes;
this script validates the split and finds the per-launch optimum.

``BLOCK_SIZE_M`` can also be swept (``--block-m``).  Each value needs its own
``moe_align_block_size`` pass (the padded slot layout is block-size-specific),
so the sweep runs once per value.  ``BLOCK_SIZE_K`` stays at 128: the kernel
static-asserts it equals the weight's fp8 quant-block K.

Every swept config is checked bit-identical against the default's output
before it competes on speed.

    CUDA_VISIBLE_DEVICES=<idle> python benchmark/glm52/bench_fused_moe_config.py

    # smoke-test the plumbing first
    ... --block-n 32 64 --group-m 1 --num-warps 4 --num-stages 3 --iters 20
"""

from __future__ import annotations

import argparse
import itertools
import time

import torch
import triton

from mstar.utils.fused_moe import runner as R


def bench_graph(fn, capture_n: int, iters: int, warmup: int = 10) -> float:
    """Time `fn` (a zero-arg closure) inside a CUDA graph.

    `capture_n` calls of `fn` are captured into one graph; the graph is then
    replayed `iters` times and timed with CUDA events. Returns microseconds
    per single call of `fn` (i.e. per launch, not per replay) -- this is how
    the kernel actually runs in production (captured decode step).
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(capture_n):
            fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / (iters * capture_n)  # us per launch


def _context_alive(dev: torch.device) -> bool:
    """Canary run after any exception.

    A config that Triton rejects at compile time (e.g. out-of-shared-memory)
    fails cleanly before any kernel executes. A config that instead triggers
    a genuine device-side error (illegal memory access) can poison the whole
    CUDA context, silently turning every later config into a spurious skip.
    This tells the two apart so a poisoned run is reported as such.
    """
    try:
        (torch.zeros(1, device=dev) + 1).item()
        return True
    except Exception:
        return False


def fmt_cfg(bn, gm, nw, ns) -> str:
    return f"BN={bn:<4} GM={gm:<2} W={nw:<2} S={ns:<2}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=4, help="rows in the trunk (k+1)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--inter", type=int, default=256, help="moe_intermediate / TP")
    ap.add_argument("--layers", type=int, default=75, help="only to scale the per-step projection")
    ap.add_argument("--block-m", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--block-n", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--group-m", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--num-warps", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--num-stages", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--capture-n", type=int, default=20, help="launches captured per CUDA graph")
    ap.add_argument("--iters", type=int, default=200, help="graph replays timed")
    ap.add_argument("--top", type=int, default=10, help="rows printed per launch")
    ap.add_argument(
        "--time-budget-s",
        type=float,
        default=300.0,
        help="soft wall-clock cap on the whole sweep; stop starting new configs past this and report whatever finished",
    )
    a = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU (CUDA_VISIBLE_DEVICES=<idle>)"
    torch.manual_seed(0)
    dev = torch.device("cuda")
    print(
        f"torch {torch.__version__} | triton {triton.__version__} | cuda {torch.version.cuda} "
        f"| device {torch.cuda.get_device_name(0)}"
    )

    E, H, I, T, K = a.experts, a.hidden, a.inter, a.tokens, a.top_k
    QBLOCK = 128
    fp8 = R.FP8_DTYPE

    w1 = (torch.randn(E, 2 * I, H, device=dev) * 0.05).to(fp8)
    w2 = (torch.randn(E, H, I, device=dev) * 0.05).to(fp8)
    w1s = torch.rand(E, -(-2 * I // QBLOCK), H // QBLOCK, device=dev) * 0.01 + 0.01
    w2s = torch.rand(E, -(-H // QBLOCK), I // QBLOCK, device=dev) * 0.01 + 0.01
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(T, E, device=dev)
    topk_w, topk_ids = torch.topk(logits.sigmoid(), K, dim=-1)
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).to(torch.bfloat16)
    topk_ids = topk_ids.to(torch.int32).contiguous()

    base_cfg = R.get_default_config(M=T, E=E, N=2 * I, K=H, top_k=K)
    raw_block_k = base_cfg["BLOCK_SIZE_K"]
    base_cfg["BLOCK_SIZE_K"] = QBLOCK

    a_q, a_s = R.per_token_group_quant_fp8(x, QBLOCK)
    ct = R._tl_compute_type(x.dtype)
    c1 = torch.empty(T * K, 2 * I, device=dev, dtype=x.dtype)
    c2 = torch.empty(T * K, I, device=dev, dtype=x.dtype)
    c3 = torch.empty(T, K, H, device=dev, dtype=x.dtype)

    grid = list(itertools.product(a.block_n, a.group_m, a.num_warps, a.num_stages))
    launch_names = [
        "gate_up (N=512, K=6144, top_k=8)",
        "down (N=6144, K=256, top_k=1, mul_routed_weight=True)",
    ]
    print(f"\nshape: tokens={T} top_k={K} E={E} H={H} I/rank={I} | BLOCK_SIZE_K={QBLOCK} (fixed, = fp8 quant group)")
    print(f"BASE CONFIG (from get_default_config): {base_cfg}")
    print(
        f"  get_default_config alone picked BLOCK_SIZE_K={raw_block_k}; fused_experts_fp8 always "
        f"overrides it to the fp8 quant block ({QBLOCK}) before launching"
    )
    print(
        "  num_warps / num_stages: not set by get_default_config -> Triton's nvidia-backend "
        f"compiler default applies (num_warps=4, num_stages=3 in triton {triton.__version__}'s "
        "CUDAOptions)"
    )
    print(
        f"sweep grid: BLOCK_SIZE_M={a.block_m} BLOCK_SIZE_N={a.block_n} "
        f"GROUP_SIZE_M={a.group_m} num_warps={a.num_warps} "
        f"num_stages={a.num_stages} -> {len(grid)} configs/launch/block_m, "
        f"{len(grid) * len(launch_names) * len(a.block_m)} compiles total, "
        f"time budget {a.time_budget_s:.0f}s\n"
    )

    t_start = time.time()
    budget_exceeded = False
    cross_best: dict[tuple[int, str], tuple] = {}
    cross_default: dict[tuple[int, str], float] = {}

    for BLOCK_M in a.block_m:
        print(f"\n{'=' * 72}")
        print(f"BLOCK_SIZE_M = {BLOCK_M}")
        print(f"{'=' * 72}")

        sorted_ids, expert_ids, n_post = R.moe_align_block_size(topk_ids, BLOCK_M, E)
        default_cfg = dict(base_cfg, BLOCK_SIZE_M=BLOCK_M)

        R.invoke_fused_moe_kernel_fp8_w8a8(
            A=a_q,
            B=w1,
            C=c1,
            A_scale=a_s,
            B_scale=w1s,
            topk_weights=topk_w,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=n_post,
            mul_routed_weight=False,
            top_k=K,
            config=default_cfg,
            compute_type=ct,
            block_shape=(QBLOCK, QBLOCK),
        )
        R.act_and_mul_triton(c1, c2, activation="silu")
        a2_q, a2_s = R.per_token_group_quant_fp8(c2, QBLOCK)

        def up(cfg, _si=sorted_ids, _ei=expert_ids, _np=n_post):
            R.invoke_fused_moe_kernel_fp8_w8a8(
                A=a_q,
                B=w1,
                C=c1,
                A_scale=a_s,
                B_scale=w1s,
                topk_weights=topk_w,
                topk_ids=topk_ids,
                sorted_token_ids=_si,
                expert_ids=_ei,
                num_tokens_post_padded=_np,
                mul_routed_weight=False,
                top_k=K,
                config=cfg,
                compute_type=ct,
                block_shape=(QBLOCK, QBLOCK),
            )

        def down(cfg, _si=sorted_ids, _ei=expert_ids, _np=n_post, _aq=a2_q, _as=a2_s):
            R.invoke_fused_moe_kernel_fp8_w8a8(
                A=_aq,
                B=w2,
                C=c3.view(T * K, H),
                A_scale=_as,
                B_scale=w2s,
                topk_weights=topk_w,
                topk_ids=topk_ids,
                sorted_token_ids=_si,
                expert_ids=_ei,
                num_tokens_post_padded=_np,
                mul_routed_weight=True,
                top_k=1,
                config=cfg,
                compute_type=ct,
                block_shape=(QBLOCK, QBLOCK),
            )

        launches = [
            (launch_names[0], up, c1),
            (launch_names[1], down, c3.view(T * K, H)),
        ]

        default_us: dict[str, float] = {}
        all_rows: dict[str, list] = {}
        all_skips: dict[str, list] = {}

        for name, fn, out_buf in launches:
            print(f"\n--- {name} ---")
            out_buf.zero_()
            fn(default_cfg)
            torch.cuda.synchronize()
            ref_out = out_buf.clone()
            default_us[name] = bench_graph(lambda fn=fn, dc=default_cfg: fn(dc), a.capture_n, a.iters)
            cross_default[(BLOCK_M, name)] = default_us[name]
            print(
                f"  default: BN={default_cfg['BLOCK_SIZE_N']} "
                f"GM={default_cfg['GROUP_SIZE_M']} W=auto S=auto "
                f"-> {default_us[name]:8.2f} us/launch (in-graph; reference for bit-identity)"
            )

            rows: list = []
            skips: list = []
            for i, (bn, gm, nw, ns) in enumerate(grid):
                if not budget_exceeded and time.time() - t_start > a.time_budget_s:
                    budget_exceeded = True
                    print(
                        f"  time budget ({a.time_budget_s:.0f}s) reached at config {i}/{len(grid)}; "
                        "not starting any more configs"
                    )
                if budget_exceeded:
                    skips.append((bn, gm, nw, ns, "not attempted (time budget)"))
                    continue

                cfg = dict(
                    BLOCK_SIZE_M=BLOCK_M,
                    BLOCK_SIZE_N=bn,
                    BLOCK_SIZE_K=QBLOCK,
                    GROUP_SIZE_M=gm,
                    num_warps=nw,
                    num_stages=ns,
                )
                try:
                    out_buf.zero_()
                    fn(cfg)
                    torch.cuda.synchronize()
                except Exception as e:
                    reason = str(e).splitlines()[0][:160]
                    skips.append((bn, gm, nw, ns, reason))
                    print(f"  {i + 1:>3}/{len(grid)} {fmt_cfg(bn, gm, nw, ns)} -> SKIP ({reason})")
                    if not _context_alive(dev):
                        print("  !! CUDA context dead -- aborting sweep early")
                        budget_exceeded = True
                    continue

                ok = torch.equal(out_buf, ref_out)
                try:
                    us = bench_graph(lambda cfg=cfg, fn=fn: fn(cfg), a.capture_n, a.iters)
                except Exception as e:
                    reason = f"graph capture: {str(e).splitlines()[0][:150]}"
                    skips.append((bn, gm, nw, ns, reason))
                    print(f"  {i + 1:>3}/{len(grid)} {fmt_cfg(bn, gm, nw, ns)} -> SKIP ({reason})")
                    if not _context_alive(dev):
                        print("  !! CUDA context dead -- aborting sweep early")
                        budget_exceeded = True
                    continue

                rows.append((bn, gm, nw, ns, us, ok))
                flag = "OK" if ok else "DIFFERS FROM DEFAULT"
                print(f"  {i + 1:>3}/{len(grid)} {fmt_cfg(bn, gm, nw, ns)} -> {us:8.2f} us  {flag}")

            rows.sort(key=lambda r: r[4])
            all_rows[name] = rows
            all_skips[name] = skips

        print(f"\n=== BLOCK_SIZE_M={BLOCK_M} summary ===")
        for name, _, _ in launches:
            rows = all_rows[name]
            skips = all_skips[name]
            print(f"\n  {name}: top {min(a.top, len(rows))} of {len(rows)} ok / {len(skips)} skipped")
            print(
                f"  {'rank':<5}{'BLOCK_N':>8}{'GROUP_M':>8}{'warps':>7}{'stages':>7}"
                f"{'us/launch':>12}{'bit-id':>8}{'vs default':>12}"
            )
            for rank, (bn, gm, nw, ns, us, ok) in enumerate(rows[: a.top], 1):
                delta = 100.0 * (us - default_us[name]) / default_us[name]
                print(f"  {rank:<5}{bn:>8}{gm:>8}{nw:>7}{ns:>7}{us:>12.2f}{str(ok):>8}{delta:>+11.1f}%")
            print(
                f"  {'--':<5}{'--':>8}{'--':>8}{'--':>7}{'--':>7}"
                f"{default_us[name]:>12.2f}{'ref':>8}{'0.0%':>12}   <- current default"
            )

            bit_id_rows = [r for r in rows if r[5]]
            if bit_id_rows:
                cross_best[(BLOCK_M, name)] = min(bit_id_rows, key=lambda r: r[4])
            elif rows:
                print("  ! no bit-identical config; picking fastest anyway, FLAGGED")
                cross_best[(BLOCK_M, name)] = min(rows, key=lambda r: r[4])

            if skips:
                uniq: dict[str, list] = {}
                for bn, gm, nw, ns, reason in skips:
                    uniq.setdefault(reason, []).append((bn, gm, nw, ns))
                print(f"  skip reasons ({len(skips)} configs):")
                for reason, cfgs in list(uniq.items())[:6]:
                    bn0, gm0, nw0, ns0 = cfgs[0]
                    print(f"    x{len(cfgs):<3} e.g. BN={bn0} GM={gm0} W={nw0} S={ns0}: {reason}")

    elapsed = time.time() - t_start
    print(f"\ntotal sweep time: {elapsed:.1f} s")

    if len(a.block_m) > 1:
        print(f"\n{'=' * 72}")
        print("CROSS BLOCK_SIZE_M COMPARISON")
        print(f"{'=' * 72}")
        for name in launch_names:
            print(f"\n  {name}:")
            print(f"  {'BM':>4}  {'best config':>28}  {'us/launch':>10}  {'bit-id':>7}  {'vs BM=16 default':>17}")
            for bm in a.block_m:
                key = (bm, name)
                if key in cross_best:
                    bn, gm, nw, ns, us, ok = cross_best[key]
                    ref = cross_default.get((16, name), cross_default.get((a.block_m[0], name), us))
                    delta = 100.0 * (us - ref) / ref
                    print(f"  {bm:>4}  {fmt_cfg(bn, gm, nw, ns):>28}  {us:>10.2f}  {str(ok):>7}  {delta:>+16.1f}%")
                else:
                    print(f"  {bm:>4}  {'(no usable configs)':>28}")

        print(f"\n  per-step projection ({a.layers} layers):")
        for bm in a.block_m:
            total = sum(cross_best.get((bm, n), (0, 0, 0, 0, 999, False))[4] for n in launch_names)
            ref_total = sum(cross_default.get((16, n), cross_default.get((a.block_m[0], n), 0)) for n in launch_names)
            delta = total - ref_total
            proj_ms = delta * a.layers / 1000.0
            print(
                f"    BM={bm}: {total:.2f} us/layer ({delta:+.2f} vs BM=16 default), "
                f"x {a.layers} layers = {proj_ms:+.3f} ms/step"
            )
    else:
        BLOCK_M = a.block_m[0]
        if all((BLOCK_M, n) in cross_best for n in launch_names):
            name_gu, name_dn = launch_names
            bn_gu, gm_gu, nw_gu, ns_gu, us_gu, ok_gu = cross_best[(BLOCK_M, name_gu)]
            bn_dn, gm_dn, nw_dn, ns_dn, us_dn, ok_dn = cross_best[(BLOCK_M, name_dn)]
            ref_gu = cross_default[(BLOCK_M, name_gu)]
            ref_dn = cross_default[(BLOCK_M, name_dn)]
            d_gu = ref_gu - us_gu
            d_dn = ref_dn - us_dn
            d_layer = d_gu + d_dn
            proj_ms = d_layer * a.layers / 1000.0
            print("\n=== best pair vs default ===")
            print(
                f"  gate_up: default {ref_gu:.2f} us -> best {us_gu:.2f} us "
                f"(BN={bn_gu} GM={gm_gu} W={nw_gu} S={ns_gu}, bit-identical={ok_gu}), "
                f"delta {d_gu:+.2f} us/layer"
            )
            print(
                f"  down:    default {ref_dn:.2f} us -> best {us_dn:.2f} us "
                f"(BN={bn_dn} GM={gm_dn} W={nw_dn} S={ns_dn}, bit-identical={ok_dn}), "
                f"delta {d_dn:+.2f} us/layer"
            )
            print(
                f"  PROJECTION: {d_layer:.2f} us/layer saved "
                f"x {a.layers} layers = {proj_ms:.3f} ms/decode step "
                f"(T={T}-row trunk, top_k={K})"
            )
        else:
            print("could not compute a best pair -- one or both launches had no usable configs")


if __name__ == "__main__":
    main()
