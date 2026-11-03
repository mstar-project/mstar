"""Stage b: one gated-delta-net mixer, graph arm vs the persistent kernel, Qwen3.5 real weights.

Both arms run against the engine's RecurrentStatePool (fp32 [HV, V, K] state and bf16 conv window per
slot, layer-major) planned for a decode step of B requests, as the runner plans one. Graph arm: the
model's ParallelGatedDeltaNet layers (the 18 of the checkpoint, chained), with the engine's
LinearAttnManager (FlashInfer's pretranspose decode, the vendored conv update), compiled as -O2 compiles
a captured decode bucket and captured into one CUDA graph. Kernel arm: the same chain of GdnMegakernel
launches on the same pool views and slot indices.

Parity: the 18-layer chain run once by each arm from identical states (output, every layer's state and
conv window), plus a 4-step run per arm (state carried) for drift.

usage: python mb_gdn.py [--rows 1 2 4] [--seconds 1.0] [--out file.json]
"""
# Each closure below is captured and timed inside the loop iteration that builds it.
# ruff: noqa: B023
import argparse
import json

import torch

from benchmark.megakernel.common import Checkpoint, capture, text_config, timed_us  # noqa: E402
from mstar.kernels.megakernel.gdn import GdnMegakernel  # noqa: E402

GDN_STATE, LINEAR_ATTN = "gdn_state", "linear_attn"


def build_resources(cfg, num_layers, device, max_slots=9):
    from mstar.engine.resources.base import EngineResourceInfo
    from mstar.engine.resources.linear_attn import LinearAttnConfig, LinearAttnManager, LinearAttnSpec
    from mstar.engine.resources.linear_attn.config import LinearAttnVariant
    from mstar.engine.resources.recurrent import (
        DeltaNetGeometry,
        RecurrentStateConfig,
        RecurrentStatePool,
        RecurrentStateSpec,
    )

    geo = DeltaNetGeometry(cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"],
                           cfg["linear_value_head_dim"], cfg["linear_conv_kernel_dim"])
    pool_spec = RecurrentStateSpec(GDN_STATE, {"llm"}, RecurrentStateConfig(
        num_layers=num_layers, blocks=geo.to_blocks(), max_slots=max_slots))
    pool = RecurrentStatePool.build(pool_spec, EngineResourceInfo(device=device))
    manager = LinearAttnManager.build(
        LinearAttnSpec(LINEAR_ATTN, {"llm"}, LinearAttnConfig(recurrent_state=GDN_STATE,
                                                              variant=LinearAttnVariant.GDN)),
        EngineResourceInfo(device=device, dependencies={GDN_STATE: pool_spec}),
    )
    return pool, manager


def build_layers(ck, cfg, pool, manager, device):
    from mstar.model.components.distributed.linear_attn import ParallelGatedDeltaNet
    from mstar.model.components.linear_attn import GDNProjLayout

    lin = [i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention"]
    layers = torch.nn.ModuleList()
    for i in lin:
        m = ParallelGatedDeltaNet(
            hidden_size=cfg["hidden_size"], num_k_heads=cfg["linear_num_key_heads"],
            num_v_heads=cfg["linear_num_value_heads"], head_k_dim=cfg["linear_key_head_dim"],
            head_v_dim=cfg["linear_value_head_dim"], conv_kernel_size=cfg["linear_conv_kernel_dim"],
            layout=GDNProjLayout.SPLIT, rms_norm_eps=cfg["rms_norm_eps"],
            linear_attn_key=LINEAR_ATTN, state_key=GDN_STATE,
        )
        m.to(device=device, dtype=torch.bfloat16)
        p = "linear_attn."
        w = m.in_proj_fused.weight
        for name, shard in (("in_proj_qkv", "qkv"), ("in_proj_z", "z"), ("in_proj_a", "a"), ("in_proj_b", "b")):
            w.weight_loader(w, ck.layer(i, p + name + ".weight", dtype=torch.bfloat16), shard)
        m.conv1d.weight.weight_loader(m.conv1d.weight, ck.layer(i, p + "conv1d.weight", dtype=torch.bfloat16))
        for name in ("A_log", "dt_bias"):
            prm = getattr(m, name)
            prm.weight_loader(prm, ck.layer(i, p + name))
        m.norm.weight.data.copy_(ck.layer(i, p + "norm.weight").float())
        m.out_proj.weight.data.copy_(ck.layer(i, p + "out_proj.weight", dtype=torch.bfloat16))
        m.requires_grad_(False).eval()
        m.bind_resources({GDN_STATE: pool, LINEAR_ATTN: manager})
        layers.append(m)
    return layers


def plan_decode(pool, manager, rids, lease):
    from mstar.engine.resources.linear_attn import LinearAttnStep
    from mstar.engine.resources.recurrent import RecurrentStep
    from mstar.engine.resources.step import Segment, StepContext

    ctx = StepContext(request_ids=list(rids), graph_walk="decode", slot=0, capture=False, slot_lease=lease)
    segs = tuple(Segment(r, "main", 1) for r in rids)
    rstep = RecurrentStep(segments=segs)
    assert pool.admit(rstep, ctx).ok
    ctx.plan_results[GDN_STATE] = pool.plan(rstep, ctx)
    manager.plan(LinearAttnStep(segments=segs), ctx)
    return rstep, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--configs", default="8:3:16:128:4")
    ap.add_argument("--no-graph-arm", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from mstar.engine.resources.step import BucketKey, SlotLease

    device = torch.device("cuda")
    cfg = text_config("0.8B")
    ck = Checkpoint("0.8B")
    eps = cfg["rms_norm_eps"]
    nlin = sum(t == "linear_attention" for t in cfg["layer_types"])
    results = {"rows": {}}
    for B in args.rows:
        torch.manual_seed(0)
        pool, manager = build_resources(cfg, nlin, device)
        manager.build_accelerator_graph_buffers([], max_bs=B, max_seq_len=B)
        layers = build_layers(ck, cfg, pool, manager, device)
        H = cfg["hidden_size"]
        rids = [f"r{i}" for i in range(B)]
        for r in rids:
            pool.ingest_request(r)
        lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=B, num_tokens=B))
        plan_decode(pool, manager, rids, lease)
        slots = pool.addressing("main").slot_indices[:B]
        # realistic, nonzero recurrent state: small random state and conv window in every used slot
        st_all = pool._blocks["state"]
        cv_all = pool._blocks["conv"]
        st_all.normal_(0, 0.02)
        cv_all.normal_(0, 0.5)
        st0, cv0 = st_all.clone(), cv_all.clone()
        x0 = (torch.randn(B, H, device=device) * 1.0).bfloat16()

        def chain_graph(x):
            # every layer reads the same input (no residual here, so a chain would shrink to nothing)
            outs = []
            for i, m in enumerate(layers):
                m.mix.bind_step("main")
                m.mix.set_layer_idx(i)
                outs.append(m(x))
            return torch.stack(outs)

        compiled = torch.compile(chain_graph, mode="max-autotune-no-cudagraphs", dynamic=False, fullgraph=False)
        row = {}
        # parity: eager graph-arm chain vs kernel chain, same initial state
        ref_out = chain_graph(x0.clone())
        ref_state, ref_conv = st_all.clone(), cv_all.clone()
        torch.cuda.synchronize()
        for cfgstr in args.configs.split(","):
            rb, ns, vb, ro, nw = map(int, cfgstr.split(":"))
            mk = GdnMegakernel(H, cfg["linear_num_value_heads"], cfg["linear_key_head_dim"],
                               cfg["linear_value_head_dim"], cfg["linear_num_key_heads"],
                               layers[0].in_proj_fused.weight.shape[0], rb=rb, ns=ns, vb=vb, ro=ro, num_warps=nw)
            out_bufs = [torch.empty(B, H, device=device, dtype=torch.bfloat16) for _ in range(len(layers) + 1)]

            def chain_kernel():
                x = out_bufs[0]
                for i, m in enumerate(layers):
                    mk(x, m.in_proj_fused.weight, m.conv1d.weight.squeeze(1), pool.block("conv", i), slots,
                       m.A_log, m.dt_bias, pool.block("state", i), m.norm.weight, m.out_proj.weight,
                       out_bufs[i + 1], eps)
                return torch.stack(out_bufs[1:])

            st_all.copy_(st0)
            cv_all.copy_(cv0)
            out_bufs[0].copy_(x0)
            kout = chain_kernel()
            torch.cuda.synchronize()
            used = slots.long()
            par = {
                "out_maxabs_diff": (kout.float() - ref_out.float()).abs().max().item(),
                "out_ref_absmax": ref_out.float().abs().max().item(),
                "out_rel_l2": ((kout.float() - ref_out.float()).norm() / ref_out.float().norm()).item(),
                "state_maxabs_diff": (st_all[:, used] - ref_state[:, used]).abs().max().item(),
                "state_ref_absmax": ref_state[:, used].abs().max().item(),
                "conv_maxabs_diff": (cv_all[:, used].float() - ref_conv[:, used].float()).abs().max().item(),
                "layer0_out_diff": None,
            }
            # one layer alone (no chain amplification)
            st_all.copy_(st0)
            cv_all.copy_(cv0)
            m = layers[0]
            m.mix.bind_step("main")
            m.mix.set_layer_idx(0)
            r1 = m(x0.clone())
            st_all.copy_(st0)
            cv_all.copy_(cv0)
            o1 = torch.empty_like(x0)
            mk(x0, m.in_proj_fused.weight, m.conv1d.weight.squeeze(1), pool.block("conv", 0), slots, m.A_log,
               m.dt_bias, pool.block("state", 0), m.norm.weight, m.out_proj.weight, o1, eps)
            torch.cuda.synchronize()
            par["layer0_out_diff"] = (o1.float() - r1.float()).abs().max().item()
            par["layer0_ref_absmax"] = r1.float().abs().max().item()
            st_all.copy_(st0)
            cv_all.copy_(cv0)

            def kstep():
                out_bufs[0].copy_(x0)
                chain_kernel()

            g = capture(kstep)
            k_us = timed_us(g.replay, args.seconds) / len(layers)
            row[f"kernel {cfgstr}"] = {"us_per_layer": k_us, "parity": par}
            print(f"[B={B}] kernel {cfgstr}: {k_us:.2f} us/layer  parity {par}", flush=True)
        if not args.no_graph_arm:
            st_all.copy_(st0)
            cv_all.copy_(cv0)
            xs = x0.clone()

            def gstep():
                xs.copy_(x0)
                compiled(xs)

            gg = capture(gstep)
            g_us = timed_us(gg.replay, args.seconds) / len(layers)
            row["graph"] = {"us_per_layer": g_us}
            print(f"[B={B}] graph arm: {g_us:.2f} us/layer", flush=True)
        results["rows"][B] = row
        torch.cuda.empty_cache()
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
