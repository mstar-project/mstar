"""Stage a: one layer's MLP block, graph arm vs the persistent kernel, Qwen3.5 real weights.

Graph arm: the model's own modules (the post-attention RMSNorm's forward_residual, ParallelGatedMLP,
the next layer's input norm's forward_residual), the 24 layers chained, compiled as the engine
compiles a captured decode bucket at -O2 (max-autotune-no-cudagraphs, dynamic=False), captured into
one CUDA graph and replayed. Kernel arm: the same 24-layer chain of MlpMegakernel launches, captured
the same way. Every layer has its own weights (24 x 22 MB at 0.8B, well past L2), as in a step.

Parity: one layer, bf16 inputs from a real-looking residual stream; graph arm (compiled) and kernel
vs an fp32 eager reference, and kernel vs graph.

usage: python mb_mlp.py [--size 0.8B] [--rows 1 2 4] [--seconds 1.0] [--out file.json]
"""
# Each closure below is captured and timed inside the loop iteration that builds it.
# ruff: noqa: B023
import argparse
import json

import torch

from benchmark.megakernel.common import Checkpoint, capture, text_config, timed_us  # noqa: E402
from mstar.kernels.megakernel.mlp import MlpMegakernel  # noqa: E402


def build_graph_arm(ck, cfg, device):
    from mstar.model.components.distributed.mlp import ParallelGatedMLP
    from mstar.model.components.norm import RMSNorm

    H, I, L, eps = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["rms_norm_eps"]
    mlps, post, nxt = torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList()
    raw = []
    for i in range(L):
        m = ParallelGatedMLP(hidden_size=H, intermediate_size=I, activation="silu")
        m.down_proj.reduce_results = False
        gate = ck.layer(i, "mlp.gate_proj.weight")
        up = ck.layer(i, "mlp.up_proj.weight")
        down = ck.layer(i, "mlp.down_proj.weight")
        m.gate_up_proj.weight.data = torch.cat([gate, up], 0)
        m.down_proj.weight.data = down
        pn = RMSNorm(H, eps=eps, gemma_mode=True)
        pn.weight.data = ck.layer(i, "post_attention_layernorm.weight")
        nn_ = RMSNorm(H, eps=eps, gemma_mode=True)
        nn_.weight.data = (ck.layer(i + 1, "input_layernorm.weight") if i + 1 < L
                           else ck.get("model.language_model.norm.weight"))
        mlps.append(m)
        post.append(pn)
        nxt.append(nn_)
        raw.append(dict(w_gu=m.gate_up_proj.weight.data, w_dt=down.t().contiguous(),
                        w_post=pn.weight.data, w_next=nn_.weight.data))
    for mod in (mlps, post, nxt):
        mod.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()
    return mlps, post, nxt, raw


def reference(h, res, w_post, w_gu, w_dt, w_next, eps, I):
    """fp32 eager, rounding where the graph rounds (each kernel's bf16 output)."""
    def norm(x, w):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + w.float())
    r1 = h.float() + res.float()
    x = norm(r1, w_post).bfloat16()
    gu = (x.float() @ w_gu.float().t()).bfloat16().float()
    g, u = gu[:, :I], gu[:, I:]
    a = (torch.nn.functional.silu(g) * u).bfloat16().float()
    d = (a @ w_dt.float()).bfloat16().float()
    r2 = r1 + d
    return norm(r2, w_next).bfloat16(), r2.bfloat16()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="0.8B")
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--configs", default="8:3:8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-graph-arm", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    cfg = text_config(args.size)
    H, I, L, eps = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["rms_norm_eps"]
    ck = Checkpoint(args.size)
    mlps, post, nxt, raw = build_graph_arm(ck, cfg, device)
    wbytes = sum(r["w_gu"].nbytes + r["w_dt"].nbytes for r in raw) / L
    results = {"size": args.size, "layer_weight_mb": wbytes / 1e6, "rows": {}}

    def chain(h, res):
        for i in range(L):
            h, res = post[i].forward_residual(h, res)
            h = mlps[i](h)
            h, res = nxt[i].forward_residual(h, res)
        return h, res

    compiled = torch.compile(chain, mode="max-autotune-no-cudagraphs", dynamic=False, fullgraph=False)

    for B in args.rows:
        row = {}
        # a residual stream of realistic scale: the embedding of random tokens, pushed through
        # the real stack's MLPs once
        emb = ck.get("model.language_model.embed_tokens.weight")[torch.randint(0, 248320, (B,))]
        h0 = (torch.randn(B, H, device=device) * 0.05).bfloat16()
        r0 = emb.to(device, torch.bfloat16).clone()

        # parity on layer 3 (one stage)
        i = 3
        rw = raw[i]
        ref_h, ref_r = reference(h0, r0, rw["w_post"], rw["w_gu"], rw["w_dt"], rw["w_next"], eps, I)
        gh, gr = h0.clone(), r0.clone()

        def one(h, res):
            h, res = post[i].forward_residual(h, res)
            h = mlps[i](h)
            return nxt[i].forward_residual(h, res)

        g_one = torch.compile(one, mode="max-autotune-no-cudagraphs", dynamic=False)
        gh, gr = g_one(gh, gr)
        for cfgstr in args.configs.split(","):
            rb, ns, nw = map(int, cfgstr.split(":"))
            mk = MlpMegakernel(H, I, max_rows=4, rb=rb, ns=ns, num_warps=nw)
            kh, kr = h0.clone(), r0.clone()
            mk(kh, kr, rw["w_post"], rw["w_gu"], rw["w_dt"], rw["w_next"], eps)
            torch.cuda.synchronize()
            par = {
                "kernel_vs_ref_h": (kh.float() - ref_h.float()).abs().max().item(),
                "kernel_vs_ref_res": (kr.float() - ref_r.float()).abs().max().item(),
                "graph_vs_ref_h": (gh.float() - ref_h.float()).abs().max().item(),
                "kernel_vs_graph_h": (kh.float() - gh.float()).abs().max().item(),
                "ref_h_absmax": ref_h.float().abs().max().item(),
            }
            # timing: the 24-layer chain in one graph
            hs, rs = h0.clone(), r0.clone()

            def kchain():
                hs.copy_(h0)
                rs.copy_(r0)
                for j in range(L):
                    w = raw[j]
                    mk(hs, rs, w["w_post"], w["w_gu"], w["w_dt"], w["w_next"], eps)

            g = capture(kchain)
            k_us = timed_us(g.replay, args.seconds, inner=5) / L
            row[f"kernel {cfgstr}"] = {"us_per_layer": k_us,
                                       "parity": par}
            print(f"[{args.size} B={B}] kernel {cfgstr}: {k_us:.2f} us/layer  parity {par}", flush=True)
        if not args.no_graph_arm:
            hs2, rs2 = h0.clone(), r0.clone()

            def gchain():
                hs2.copy_(h0)
                rs2.copy_(r0)
                compiled(hs2, rs2)

            gg = capture(gchain)
            g_us = timed_us(gg.replay, args.seconds, inner=5) / L
            row["graph"] = {"us_per_layer": g_us}
            print(f"[{args.size} B={B}] graph arm: {g_us:.2f} us/layer", flush=True)
        # bytes the stage must move: gate/up + down weights (the transposed copy replaces down)
        row["weight_mb"] = (raw[0]["w_gu"].nbytes + raw[0]["w_dt"].nbytes) / 1e6
        results["rows"][B] = row
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
