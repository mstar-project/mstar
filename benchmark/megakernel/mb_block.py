"""Stage d: Qwen3.5-0.8B's 4-layer blocks (3 GDN + 1 attention, each with its MLP), three arms:

  graph:  the model's modules in run_layers' order (mixer, post-attention add+norm, MLP, next add+norm,
          the residual deferred into the next norm), the engine's GDN and attention resources (pool,
          FlashInfer GDN decode, conv update, KV scatter + trtllm-gen fmha), compiled as -O2 compiles a
          decode bucket, captured.
  stages: the pilot's separate kernels back to back (GdnMegakernel / AttnMegakernel, MlpMegakernel).
  block:  BlockMegakernel, one launch per 4-layer block.
All 24 layers (6 blocks, every layer its own weights) in one CUDA graph per arm; times are per block.

Parity: block 0 alone from identical states, each arm vs the eager graph arm: the block's output (the
next layer's normed input and the residual), the three GDN states and conv windows, the new KV entry.

usage: python mb_block.py [--rows 1 2 4] [--kv 512 2048] [--seconds 1.0] [--out file.json]
"""
# Each closure below is captured and timed inside the loop iteration that builds it.
# ruff: noqa: B023
import argparse
import json

import torch

from benchmark.megakernel.common import Checkpoint, capture, text_config, timed_us
from benchmark.megakernel.mb_attn import PAGE
from benchmark.megakernel.mb_attn import build as build_attn
from benchmark.megakernel.mb_gdn import build_layers, build_resources, plan_decode
from benchmark.megakernel.mb_mlp import build_graph_arm
from mstar.kernels.megakernel.attn import AttnMegakernel
from mstar.kernels.megakernel.block import BlockMegakernel
from mstar.kernels.megakernel.gdn import GdnMegakernel
from mstar.kernels.megakernel.mlp import MlpMegakernel


def setup(ck, cfg, device, B, L, max_len):
    from mstar.engine.resources.step import BucketKey, SlotLease
    from mstar.model.components.norm import RMSNorm

    nlin = sum(t == "linear_attention" for t in cfg["layer_types"])
    pool, manager = build_resources(cfg, nlin, device)
    manager.build_accelerator_graph_buffers([], max_bs=B, max_seq_len=B)
    gdn = build_layers(ck, cfg, pool, manager, device)
    rids = [f"r{i}" for i in range(B)]
    for r in rids:
        pool.ingest_request(r)
    plan_decode(pool, manager, rids, SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=B, num_tokens=B)))
    slots = pool.addressing("main").slot_indices[:B]
    pool._blocks["state"].normal_(0, 0.02)
    pool._blocks["conv"].normal_(0, 0.5)
    A = build_attn(ck, cfg, device, B, L, max_len)
    mlps, post, nxt, raw = build_graph_arm(ck, cfg, device)
    n0 = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"], gemma_mode=True)
    n0 = n0.to(device, torch.bfloat16).requires_grad_(False)
    n0.weight.data.copy_(ck.layer(0, "input_layernorm.weight", dtype=torch.bfloat16))
    return dict(pool=pool, manager=manager, gdn=gdn, slots=slots, A=A, mlps=mlps, post=post, nxt=nxt, raw=raw, n0=n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--kv", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--no-graph-arm", action="store_true")
    ap.add_argument("--no-stages", action="store_true")
    ap.add_argument("--bcfg", default="8:3:4:3:4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")
    cfg = text_config("0.8B")
    ck = Checkpoint("0.8B")
    eps = cfg["rms_norm_eps"]
    types = cfg["layer_types"]
    NL = len(types)
    nblocks = NL // 4
    max_len = 4096
    results = {}
    for L in args.kv:
        for B in args.rows:
            torch.manual_seed(0)
            S = setup(ck, cfg, device, B, L, max_len)
            pool, gdn, A, slots = S["pool"], S["gdn"], S["A"], S["slots"]
            attn, kv = A["mods"], A["kv"]
            st0, cv0, kv0 = pool._blocks["state"].clone(), pool._blocks["conv"].clone(), kv.clone()

            def reset():
                pool._blocks["state"].copy_(st0)
                pool._blocks["conv"].copy_(cv0)
                kv.copy_(kv0)

            H = cfg["hidden_size"]
            emb = ck.get("model.language_model.embed_tokens.weight")[torch.randint(0, cfg["vocab_size"], (B,))]
            res0 = emb.to(device, torch.bfloat16).contiguous()
            h0 = S["n0"](res0).detach().contiguous()

            def graph_layers(h, res, first, last):
                gi = sum(t == "linear_attention" for t in types[:first])
                ai = sum(t == "full_attention" for t in types[:first])
                for i in range(first, last):
                    if types[i] == "linear_attention":
                        m = gdn[gi]
                        m.mix.bind_step("main")
                        m.mix.set_layer_idx(gi)
                        gi += 1
                    else:
                        m = attn[ai]
                        ai += 1
                    h = m(h)
                    h, res = S["post"][i].forward_residual(h, res)
                    h = S["mlps"][i](h)
                    h, res = S["nxt"][i].forward_residual(h, res)
                return h, res

            blocks = []
            knobs = dict(zip(("rb", "ns", "rb_m", "ns_m", "num_warps"), map(int, args.bcfg.split(":")), strict=True))
            for k in range(nblocks):
                lay = list(range(4 * k, 4 * k + 4))
                assert [types[i] for i in lay] == ["linear_attention"] * 3 + ["full_attention"], lay
                blocks.append(BlockMegakernel(cfg, gdn[3 * k:3 * k + 3], attn[k], [S["mlps"][i] for i in lay],
                                              [S["post"][i] for i in lay], [S["nxt"][i] for i in lay], max_len,
                                              device=device, **knobs))
            hb = [torch.empty(B, H, device=device, dtype=torch.bfloat16) for _ in range(2)]
            rb_ = [torch.empty(B, H, device=device, dtype=torch.bfloat16) for _ in range(2)]

            def block_chain(nb=nblocks):
                hb[0].copy_(h0)
                rb_[0].copy_(res0)
                for k in range(nb):
                    blocks[k](hb[k % 2], rb_[k % 2], rb_[(k + 1) % 2], hb[(k + 1) % 2], pool._blocks["conv"],
                              pool._blocks["state"], 3 * k, slots, A["cos"], A["sin"], kv[k], A["table"], A["lens"],
                              eps)
                return hb[nb % 2], rb_[nb % 2]

            mk_g = GdnMegakernel(H, gdn[0].num_v_heads, gdn[0].head_k_dim, gdn[0].head_v_dim, gdn[0].num_k_heads,
                                 gdn[0].in_proj_fused.weight.shape[0], rb=8, ns=3, vb=16, ro=128, num_warps=4)
            mk_a = AttnMegakernel(H, attn[0].num_heads, attn[0].num_kv_heads, attn[0].head_dim, A["rot"], PAGE,
                                  max_len, rb=8, ns=3, ch=32 if L <= 512 else 64, ro=64, num_warps=4)
            mk_m = MlpMegakernel(H, cfg["intermediate_size"], rb=4, ns=3, num_warps=4)
            sx = torch.empty(B, H, device=device, dtype=torch.bfloat16)
            sres = torch.empty(B, H, device=device, dtype=torch.bfloat16)
            sout = torch.empty(B, H, device=device, dtype=torch.bfloat16)

            def stage_chain(nl=NL):
                sx.copy_(h0)
                sres.copy_(res0)
                gi = ai = 0
                for i in range(nl):
                    if types[i] == "linear_attention":
                        m = gdn[gi]
                        mk_g(sx, m.in_proj_fused.weight, m.conv1d.weight.squeeze(1), pool.block("conv", gi), slots,
                             m.A_log, m.dt_bias, pool.block("state", gi), m.norm.weight, m.out_proj.weight, sout, eps)
                        gi += 1
                    else:
                        m = attn[ai]
                        mk_a(sx, m.qkv_proj.weight, m.q_norm.weight, m.k_norm.weight, A["cos"], A["sin"], kv[ai],
                             A["table"], A["lens"], m.o_proj.weight, sout, eps)
                        ai += 1
                    w = S["raw"][i]
                    mk_m(sout, sres, w["w_post"], w["w_gu"], w["w_dt"], w["w_next"], eps)
                    sx.copy_(sout)
                return sx, sres

            # ---- parity on block 0, from identical states
            row = {}
            reset()
            ref_h, ref_r = graph_layers(h0.clone(), res0.clone(), 0, 4)
            ref_state = pool._blocks["state"][:3].clone()
            ref_conv = pool._blocks["conv"][:3].clone()
            ref_kv = kv[0].clone()
            torch.cuda.synchronize()

            def cmp(h, r, tag):
                used = slots.long()
                d = {
                    "h_maxabs": (h.float() - ref_h.float()).abs().max().item(),
                    "h_ref_absmax": ref_h.float().abs().max().item(),
                    "h_rel_l2": ((h.float() - ref_h.float()).norm() / ref_h.float().norm()).item(),
                    "res_maxabs": (r.float() - ref_r.float()).abs().max().item(),
                    "res_rel_l2": ((r.float() - ref_r.float()).norm() / ref_r.float().norm()).item(),
                    "state_maxabs": (pool._blocks["state"][:3, used] - ref_state[:, used]).abs().max().item(),
                    "conv_maxabs": (pool._blocks["conv"][:3, used].float()
                                    - ref_conv[:, used].float()).abs().max().item(),
                    "kv_maxabs": (kv[0].float() - ref_kv.float()).abs().max().item(),
                }
                print(f"[KV={L} B={B}] parity {tag}: {d}", flush=True)
                return d

            reset()
            h, r = block_chain(1)
            torch.cuda.synchronize()
            row["parity block"] = cmp(h, r, "block")
            reset()
            if not args.no_stages:
                h, r = stage_chain(4)
                torch.cuda.synchronize()
                row["parity stages"] = cmp(h, r, "stages")
            # ---- timing, all 6 blocks
            reset()
            g = capture(block_chain)
            us = timed_us(g.replay, args.seconds) / nblocks
            row["block"] = us
            print(f"[KV={L} B={B}] block kernel: {us:.2f} us/block", flush=True)
            if not args.no_stages:
                reset()
                g = capture(stage_chain)
                us = timed_us(g.replay, args.seconds) / nblocks
                row["stages"] = us
                print(f"[KV={L} B={B}] stage kernels back to back: {us:.2f} us/block", flush=True)
            if not args.no_graph_arm:
                reset()
                compiled = torch.compile(lambda h, r: graph_layers(h, r, 0, NL), mode="max-autotune-no-cudagraphs",
                                         dynamic=False, fullgraph=False)
                gh, gr = h0.clone(), res0.clone()

                def gstep():
                    gh.copy_(h0)
                    gr.copy_(res0)
                    compiled(gh, gr)

                gg = capture(gstep)
                us = timed_us(gg.replay, args.seconds) / nblocks
                row["graph"] = us
                print(f"[KV={L} B={B}] graph arm: {us:.2f} us/block", flush=True)
            results[f"kv{L}_b{B}"] = row
            torch.cuda.empty_cache()
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
