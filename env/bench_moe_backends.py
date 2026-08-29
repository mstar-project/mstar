#!/usr/bin/env python3
"""ONE GPU, kernel-only: M*'s Triton fp8 fused-MoE vs FlashInfer's CUTLASS
``cutlass_fused_moe`` (DeepSeek-style fp8 block scaling, Hopper/sm90), at
GLM-5.2's decode shape.

Both paths compute the *whole* routed-expert block from the same logical
weights: gate/up GEMM -> SwiGLU -> down GEMM -> top-k reduce, e4m3 weights
with 128x128 block scales and bf16 hidden states.

Layout facts established by reading the sources (see the report / comments):

* ``fc1_expert_weights`` is ``[E, 2I, H]`` e4m3 row-major, ``fc2_expert_weights``
  ``[E, H, I]`` -- identical to M*'s ``w1``/``w2``.  No CUTLASS-side interleave
  or BlockMajorK shuffle: ``reorder_rows_for_gated_act_gemm`` /
  ``_shuffle_deepseek_fp8_moe_weights`` are applied by vLLM only for the
  *TRT-LLM-gen* backend (``is_deepseek_fp8 and is_trtllm``), never for CUTLASS.
* FlashInfer wants the **W31** half order (up || gate); M* stores **W13**
  (gate || up, see ``act_and_mul_kernel``: ``silu(first half) * second half``).
  So both ``w1`` and ``w1_scale_inv`` need their two halves flipped.  This is
  what vLLM's ``swap_w13_to_w31`` does.  ``--order`` = ``both`` probes it
  empirically and reports which one matches the Triton reference.
* DeepSeek block scale: ``quant_scales = [w1_scale, w2_scale]`` (the fp32
  block-scale tensors themselves, ``[E, 2I/128, H/128]`` and ``[E, H/128,
  I/128]``), ``input_sf=None``, and ``input`` stays **bf16** -- the kernel does
  the per-token-group activation quant itself.
* ``min_latency_mode=True`` is rejected unconditionally by the public wrapper
  in 0.6.16 (``raise NotImplementedError`` before the arch dispatch), so it is
  reached here through the raw module op, and its output is the *unreduced*
  ``[T*E, H]` scratch -- timed, not compared.

Run (GPU 7 only):

    CUDA_VISIBLE_DEVICES=7 python env/bench_moe_backends.py --tokens 4
"""

import argparse
import json
import os
import tempfile

import torch

from mstar.utils.fused_moe import runner as R

BLOCK = 128


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def bench(fn, iters=200, graph=False, capture=20):
    """us per call.  ``graph`` captures ``capture`` launches and replays."""
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(capture):
                fn()
        run, per = g.replay, capture
    else:
        run, per = fn, 1
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        run()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / (iters * per)


def count_kernels(fn, tag):
    """(kernels, memcpy/memset) launched by one call, from a chrome trace."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    path = os.path.join(tempfile.gettempdir(), f"moe_trace_{tag}.json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        tr = json.load(f)
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    k = sum(1 for e in evs if e.get("cat") == "kernel")
    m = sum(1 for e in evs if e.get("cat") in ("gpu_memcpy", "gpu_memset"))
    os.remove(path)
    return k, m


def compare(ref, out):
    r = ref.float().flatten()
    o = out.float().flatten()
    d = (r - o).abs()
    denom = r.abs().clamp_min(1e-3)
    return dict(
        max_abs=d.max().item(),
        max_rel=(d / denom).max().item(),
        cos=torch.nn.functional.cosine_similarity(r, o, dim=0).item(),
        ref_absmax=r.abs().max().item(),
    )


def swap_halves(x):
    """W13 (gate||up) -> W31 (up||gate) along dim 1.  vLLM's swap_w13_to_w31.

    Done through a uint8 view for fp8 tensors -- ``cat``/``flip`` are not
    guaranteed for float8 dtypes on every torch build.
    """
    n = x.shape[1]
    assert n % 2 == 0, x.shape
    if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        v = x.view(torch.uint8)
        return torch.cat([v[:, n // 2 :], v[:, : n // 2]], dim=1).contiguous().view(x.dtype)
    return torch.cat([x[:, n // 2 :], x[:, : n // 2]], dim=1).contiguous()


# ---------------------------------------------------------------------------


def build(T, K, E, H, I, dev, seed=0):
    torch.manual_seed(seed)
    fp8 = R.FP8_DTYPE
    w1 = (torch.randn(E, 2 * I, H, device=dev) * 0.05).to(fp8)
    w2 = (torch.randn(E, H, I, device=dev) * 0.05).to(fp8)
    w1s = torch.rand(E, -(-2 * I // BLOCK), H // BLOCK, device=dev) * 0.01 + 0.01
    w2s = torch.rand(E, -(-H // BLOCK), I // BLOCK, device=dev) * 0.01 + 0.01
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(T, E, device=dev)
    tw, tid = torch.topk(logits.sigmoid(), K, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(torch.bfloat16)
    tid = tid.to(torch.int32).contiguous()
    return x, w1, w2, w1s, w2s, tw, tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--inter", type=int, default=256, help="moe_intermediate / TP")
    ap.add_argument("--layers", type=int, default=75)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--capture", type=int, default=20)
    ap.add_argument("--order", choices=["w13", "w31", "both"], default="both")
    ap.add_argument("--min-latency", action="store_true", help="also time the raw min_latency_mode op")
    a = ap.parse_args()

    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import cutlass_fused_moe, cutlass_fused_moe_workspace_size

    dev = torch.device("cuda")
    cc = torch.cuda.get_device_capability()
    print(f"device: {torch.cuda.get_device_name()} sm{cc[0]}{cc[1]}  torch {torch.__version__}")
    import flashinfer

    print(f"flashinfer {flashinfer.__version__}")

    E, H, I, K = a.experts, a.hidden, a.inter, a.top_k
    rows = []

    for T in a.tokens:
        print(f"\n=== T={T}  K={K}  E={E}  H={H}  I/rank={I}  block={BLOCK}x{BLOCK} ===")
        x, w1, w2, w1s, w2s, tw, tid = build(T, K, E, H, I, dev)

        # ---- reference: M*'s Triton path -------------------------------
        ref = R.fused_experts_fp8(x, w1, w2, w1s, w2s, tw, tid, block_size=(BLOCK, BLOCK))
        assert ref.shape == (T, H) and ref.dtype == torch.bfloat16

        def mstar():
            R.fused_experts_fp8(x, w1, w2, w1s, w2s, tw, tid, block_size=(BLOCK, BLOCK))

        # ---- FlashInfer ------------------------------------------------
        tid_fi = tid.to(torch.int32).contiguous()
        tw_fi = tw.float().contiguous()
        out_buf = torch.empty(T, H, device=dev, dtype=torch.bfloat16)
        ws_bytes = cutlass_fused_moe_workspace_size(
            max(T, 8), H, I, E, K,
            x_dtype=torch.bfloat16, weight_dtype=R.FP8_DTYPE, output_dtype=torch.bfloat16,
            use_deepseek_fp8_block_scale=True, device=dev,
        )
        ws = torch.empty(ws_bytes, dtype=torch.uint8, device=dev)
        print(f"flashinfer workspace: {ws_bytes/2**20:.1f} MiB")

        orders = ["w13", "w31"] if a.order == "both" else [a.order]
        variants = {}
        for name in orders:
            fc1 = w1 if name == "w13" else swap_halves(w1)
            fc1s = w1s if name == "w13" else swap_halves(w1s)
            variants[name] = (fc1.contiguous(), fc1s.contiguous())

        def fi_call(fc1, fc1s, out):
            r = cutlass_fused_moe(
                input=x,
                token_selected_experts=tid_fi,
                token_final_scales=tw_fi,
                fc1_expert_weights=fc1,
                fc2_expert_weights=w2,
                output_dtype=torch.bfloat16,
                quant_scales=[fc1s, w2s],
                input_sf=None,
                output=out,
                use_deepseek_fp8_block_scale=True,
                tune_max_num_tokens=max(T, 8),
                workspace_buffer=ws,
            )
            return r

        best, results = None, {}
        for name, (fc1, fc1s) in variants.items():
            try:
                with autotune(True, tuning_buckets=(max(T, 1),)):
                    fi_call(fc1, fc1s, out_buf)
                torch.cuda.synchronize()
                m = compare(ref, out_buf)
                results[name] = m
                print(f"  half order {name}: max|d|={m['max_abs']:.4g}  maxrel={m['max_rel']:.4g}  cos={m['cos']:.6f}")
                if best is None or m["max_abs"] < results[best]["max_abs"]:
                    best = name
            except Exception as exc:  # noqa: BLE001
                print(f"  half order {name}: FAILED {type(exc).__name__}: {exc}")
        if best is None:
            print("  flashinfer path unusable at this shape; skipping timings")
            continue
        print(f"  -> matching layout: {best}  (ref |max|={results[best]['ref_absmax']:.4g})")
        fc1, fc1s = variants[best]

        def fi():
            fi_call(fc1, fc1s, out_buf)

        # ---- timings ---------------------------------------------------
        with autotune(False):
            kt = count_kernels(mstar, f"mstar_{T}")
            kf = count_kernels(fi, f"fi_{T}")
            e_m = bench(mstar, a.iters)
            e_f = bench(fi, a.iters)
            try:
                g_m = bench(mstar, a.iters, graph=True, capture=a.capture)
            except Exception as exc:  # noqa: BLE001
                print(f"  triton graph capture failed: {type(exc).__name__}: {exc}")
                g_m = float("nan")
            try:
                g_f = bench(fi, a.iters, graph=True, capture=a.capture)
            except Exception as exc:  # noqa: BLE001
                print(f"  flashinfer graph capture failed: {type(exc).__name__}: {exc}")
                g_f = float("nan")
        rows.append((T, "triton fp8 (M*)", kt, e_m, g_m))
        rows.append((T, f"flashinfer cutlass ({best})", kf, e_f, g_f))

        if a.min_latency:
            try:
                from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

                mod = get_cutlass_fused_moe_module("90")
                ml_out = torch.empty(T * E, H, device=dev, dtype=torch.bfloat16)

                def fi_ml():
                    mod.cutlass_fused_moe(
                        ml_out, x, tid_fi, tw_fi, fc1, None, w2, None,
                        torch.bfloat16, [fc1s, w2s], None, None, None, None, True,
                        1, 0, 1, 0, 1, 0,
                        enable_alltoall=False,
                        use_deepseek_fp8_block_scale=True,
                        min_latency_mode=True,
                        tune_max_num_tokens=max(T, 8),
                    )

                with autotune(True, tuning_buckets=(max(T, 1),)):
                    fi_ml()
                torch.cuda.synchronize()
                with autotune(False):
                    kml = count_kernels(fi_ml, f"fiml_{T}")
                    e_ml = bench(fi_ml, a.iters)
                    try:
                        g_ml = bench(fi_ml, a.iters, graph=True, capture=a.capture)
                    except Exception:  # noqa: BLE001
                        g_ml = float("nan")
                rows.append((T, "flashinfer min_latency (unreduced)", kml, e_ml, g_ml))
            except Exception as exc:  # noqa: BLE001
                print(f"  min_latency_mode: FAILED {type(exc).__name__}: {exc}")

        del ws, out_buf, variants, fc1, fc1s
        torch.cuda.empty_cache()

    print(f"\n{'T':>3} {'path':<36}{'kern':>6}{'memops':>8}{'eager us':>10}{'graph us':>10}"
          f"{'x' + str(a.layers) + ' ms':>12}")
    for T, name, (k, m), e, g in rows:
        proj = g * a.layers / 1000.0
        print(f"{T:>3} {name:<36}{k:>6}{m:>8}{e:>10.1f}{g:>10.1f}{proj:>12.2f}")
    print(f"\nlast column: PROJECTION only -- per-layer graph us x {a.layers} layers, "
          f"routed-expert block alone (no attention, no shared expert, no comms).")


if __name__ == "__main__":
    main()
