"""Stage c: one full-attention mixer, graph arm vs the persistent kernel, Qwen3.5-0.8B real weights.

Graph arm: the model's Qwen3_5Attention (fused qkv, gemma q/k norm, partial mRoPE from the step's cos/sin,
output gate, o_proj) with its `attend` doing what AttentionCallable does for this model on Blackwell:
the engine's NHD scatter of the new K/V (`_kv_scatter_nhd_eager`) then FmhaDecodeWrapper.run (trtllm-gen
fmha over the page table). The 6 attention layers, compiled as -O2 compiles a decode bucket, captured.
Kernel arm: AttnMegakernel on the same cache, table and lengths.

usage: python mb_attn.py [--rows 1 2 4] [--kv 512 2048] [--seconds 1.0] [--out file.json]
"""
# Each closure below is captured and timed inside the loop iteration that builds it.
# ruff: noqa: B023
import argparse
import json

import torch

from benchmark.megakernel.common import Checkpoint, capture, text_config, timed_us  # noqa: E402
from mstar.kernels.megakernel.attn import AttnMegakernel  # noqa: E402

PAGE = 128


class Attend:
    """AttentionCallable's write-then-attend for one layer, on fixed buffers."""

    label = "main"

    def __init__(self, kv_layer, wrapper, page_idx, cache_idx):
        self.kv_layer, self.wrapper, self.page_idx, self.cache_idx = kv_layer, wrapper, page_idx, cache_idx

    @torch.compiler.disable
    def __call__(self, q, k, v):
        from mstar.engine.resources.kv.cache import _kv_scatter_nhd_eager

        _kv_scatter_nhd_eager(self.kv_layer, k, v, self.page_idx, self.cache_idx)
        return self.wrapper.run(q, self.kv_layer)


def build(ck, cfg, device, B, L, max_len):
    from mstar.engine.resources.attn.workspace import Workspace
    from mstar.engine.resources.attn.wrappers import FmhaDecodeWrapper
    from mstar.model.qwen3_5.components.language_model import Qwen3_5Attention, RopeCache
    from mstar.model.qwen3_5.components.rope import compute_3d_cos_sin, compute_inv_freq

    att = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
    H, D, QH, KVH = cfg["hidden_size"], cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    rope = cfg.get("rope_parameters") or cfg["rope_scaling"]
    rot = int(D * rope["partial_rotary_factor"])
    pages_per_row = -(-max_len // PAGE)
    npages = 1 + B * pages_per_row
    kv = torch.randn(len(att), npages, 2, PAGE, KVH, D, device=device).bfloat16()
    perm = torch.randperm(npages - 1) + 1
    table = torch.zeros(B, pages_per_row + 1, dtype=torch.int32, device=device)
    for b in range(B):
        table[b, :pages_per_row] = perm[b * pages_per_row:(b + 1) * pages_per_row].to(torch.int32)
    lens = torch.full((B,), L, dtype=torch.int32, device=device)
    page_idx = table[torch.arange(B), (lens - 1) // PAGE].contiguous()
    cache_idx = ((lens - 1) % PAGE).contiguous()
    ws = Workspace(torch.zeros(256 << 20, dtype=torch.uint8, device=device), "mb_attn")
    wrapper = FmhaDecodeWrapper(workspace_buffer=ws, num_qo_heads=QH, num_kv_heads=KVH, head_dim=D,
                                page_size=PAGE, batch_size=B, max_seq_len=max_len, device=device)
    n = -(-L // PAGE)
    tab_h = table.cpu()
    indptr = torch.arange(0, (B + 1) * n, n, dtype=torch.int32)
    indices = torch.cat([tab_h[b, :n] for b in range(B)])
    last = torch.full((B,), (L - 1) % PAGE + 1, dtype=torch.int32)
    wrapper.plan(paged_kv_indptr=indptr, paged_kv_indices=indices, paged_kv_last_page_len=last,
                 kv_lens=torch.full((B,), L, dtype=torch.int32))
    inv_freq = compute_inv_freq(rot, rope["rope_theta"]).to(device)
    pos = torch.full((3, B), L - 1, device=device)
    cos, sin = compute_3d_cos_sin(pos, inv_freq, rope["mrope_section"], target_dtype=torch.bfloat16)
    rc = RopeCache()
    rc.set(cos, sin)
    mods = torch.nn.ModuleList()
    for li, i in enumerate(att):
        m = Qwen3_5Attention(output_gate=True, rope_cache=rc, hidden_size=H, num_heads=QH, num_kv_heads=KVH,
                             head_dim=D, qk_norm=True, rms_norm_eps=cfg["rms_norm_eps"],
                             rope_theta=rope["rope_theta"], attn_key="attn", kv_key="kv", pos_key="rope")
        m.to(device=device, dtype=torch.bfloat16)
        w = m.qkv_proj.weight
        for name, shard in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
            w.weight_loader(w, ck.layer(i, f"self_attn.{name}.weight", dtype=torch.bfloat16), shard)
        m.q_norm.weight.data.copy_(ck.layer(i, "self_attn.q_norm.weight", dtype=torch.bfloat16))
        m.k_norm.weight.data.copy_(ck.layer(i, "self_attn.k_norm.weight", dtype=torch.bfloat16))
        m.o_proj.weight.data.copy_(ck.layer(i, "self_attn.o_proj.weight", dtype=torch.bfloat16))
        m.o_proj.reduce_results = False
        m.requires_grad_(False).eval()
        m.attend = Attend(kv[li], wrapper, page_idx, cache_idx)
        mods.append(m)
    return dict(mods=mods, kv=kv, table=table, lens=lens, cos=cos, sin=sin, rot=rot, page_idx=page_idx,
                cache_idx=cache_idx, H=H, D=D, QH=QH, KVH=KVH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--kv", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--configs", default="8:3:32:64:4,8:3:64:64:4")
    ap.add_argument("--no-graph-arm", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")
    cfg = text_config("0.8B")
    ck = Checkpoint("0.8B")
    eps = cfg["rms_norm_eps"]
    max_len = 4096
    results = {}
    for L in args.kv:
        for B in args.rows:
            torch.manual_seed(0)
            S = build(ck, cfg, device, B, L, max_len)
            mods, kv = S["mods"], S["kv"]
            kv0 = kv.clone()
            x0 = torch.randn(B, S["H"], device=device).bfloat16()

            def chain_graph(x):
                return torch.stack([m(x) for m in mods])

            ref = chain_graph(x0)
            kv_ref = kv.clone()
            row = {}
            for cfgstr in args.configs.split(","):
                rb, ns, ch, ro, nw = map(int, cfgstr.split(":"))
                mk = AttnMegakernel(S["H"], S["QH"], S["KVH"], S["D"], S["rot"], PAGE, max_len, rb=rb, ns=ns,
                                    ch=ch, ro=ro, num_warps=nw)
                outs = [torch.empty(B, S["H"], device=device, dtype=torch.bfloat16) for _ in mods]
                xin = x0.clone()

                def chain_kernel():
                    for li, m in enumerate(mods):
                        mk(xin, m.qkv_proj.weight, m.q_norm.weight, m.k_norm.weight, S["cos"], S["sin"], kv[li],
                           S["table"], S["lens"], m.o_proj.weight, outs[li], eps)
                    return torch.stack(outs)

                kv.copy_(kv0)
                kout = chain_kernel()
                torch.cuda.synchronize()
                pi, ci = S["page_idx"].long(), S["cache_idx"].long()
                par = {
                    "out_maxabs_diff": (kout.float() - ref.float()).abs().max().item(),
                    "out_ref_absmax": ref.float().abs().max().item(),
                    "out_rel_l2": ((kout.float() - ref.float()).norm() / ref.float().norm()).item(),
                    "kv_new_maxabs_diff": max(
                        (kv[li, pi[b], :, ci[b]].float() - kv_ref[li, pi[b], :, ci[b]].float()).abs().max().item()
                        for li in range(len(mods)) for b in range(B)),
                    "kv_all_maxabs_diff": (kv.float() - kv_ref.float()).abs().max().item(),
                }
                g = capture(chain_kernel)
                k_us = timed_us(g.replay, args.seconds) / len(mods)
                row[f"kernel {cfgstr}"] = {"us_per_layer": k_us, "parity": par}
                print(f"[KV={L} B={B}] kernel {cfgstr}: {k_us:.2f} us/layer parity {par}", flush=True)
            if not args.no_graph_arm:
                kv.copy_(kv0)
                compiled = torch.compile(chain_graph, mode="max-autotune-no-cudagraphs", dynamic=False,
                                         fullgraph=False)
                xs = x0.clone()
                gg = capture(lambda: compiled(xs))
                g_us = timed_us(gg.replay, args.seconds) / len(mods)
                cref = compiled(x0)
                row["graph"] = {"us_per_layer": g_us,
                                "compiled_vs_eager_maxabs": (cref.float() - ref.float()).abs().max().item()}
                print(f"[KV={L} B={B}] graph arm: {g_us:.2f} us/layer", flush=True)
            results[f"kv{L}_b{B}"] = row
            torch.cuda.empty_cache()
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
