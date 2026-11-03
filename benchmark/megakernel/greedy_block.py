"""Greedy tokens with block 0 (layers 0-3) on BlockMegakernel vs on the model's own modules.

Both arms run every token (prompt and reply) as a one-row decode step through the same engine
resources (GDN pool slots, the paged KV cache, FlashInfer's GDN decode and trtllm-gen fmha in the
module arm), eager, and share layers 4-23, the final norm and the tied LM head. Arm `graph`: all 24
layers on the model's modules. Arm `block`: layers 0-3 on the fused kernel, the rest on the modules.
32 prompts from prompts_512.txt, the first --prompt-len tokens of each, --gen greedy tokens.

usage: python greedy_block.py [--prompts 32] [--prompt-len 64] [--gen 32] [--out file.json]
"""
# Each closure below is captured and timed inside the loop iteration that builds it.
# ruff: noqa: B023
import argparse
import json
import os

import torch

HERE = os.path.dirname(__file__)
from benchmark.megakernel.common import Checkpoint, snapshot, text_config  # noqa: E402
from benchmark.megakernel.mb_block import setup  # noqa: E402
from mstar.kernels.megakernel.block import BlockMegakernel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", nargs="+", default=["graph", "block"])
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from mstar.model.qwen3_5.components.rope import compute_3d_cos_sin, compute_inv_freq

    device = torch.device("cuda")
    cfg = text_config("0.8B")
    ck = Checkpoint("0.8B")
    eps = cfg["rms_norm_eps"]
    types = cfg["layer_types"]
    tok = AutoTokenizer.from_pretrained(snapshot("0.8B"))
    lines = [ln.strip() for ln in open("/root/work/runs/llm1/data/prompts_512.txt") if ln.strip()][: args.prompts]
    max_len = 4096
    torch.manual_seed(0)
    S = setup(ck, cfg, device, 1, args.prompt_len + args.gen + 1, max_len)
    pool, gdn, A, slots = S["pool"], S["gdn"], S["A"], S["slots"]
    attn, kv = A["mods"], A["kv"]
    W = ck.get("model.language_model.embed_tokens.weight", dtype=torch.bfloat16)
    rope = cfg.get("rope_parameters") or cfg["rope_scaling"]
    inv_freq = compute_inv_freq(A["rot"], rope["rope_theta"]).to(device)
    blk = BlockMegakernel(cfg, gdn[0:3], attn[0], S["mlps"][0:4], S["post"][0:4], S["nxt"][0:4], max_len, device=device)
    wrapper = attn[0].attend.wrapper
    table = A["table"]
    PAGE = kv.shape[3]
    cos_buf, sin_buf = A["cos"], A["sin"]
    hb, rb = torch.empty(1, cfg["hidden_size"], device=device, dtype=torch.bfloat16), \
        torch.empty(1, cfg["hidden_size"], device=device, dtype=torch.bfloat16)

    def set_pos(t):
        """Step at position t: cos/sin, lengths, the fmha plan and the KV write target."""
        L = t + 1
        pos = torch.full((3, 1), t, device=device)
        c, s = compute_3d_cos_sin(pos, inv_freq, rope["mrope_section"], target_dtype=torch.bfloat16)
        cos_buf.copy_(c)
        sin_buf.copy_(s)
        A["lens"].fill_(L)
        n = -(-L // PAGE)
        tab = table.cpu()
        wrapper.plan(paged_kv_indptr=torch.tensor([0, n], dtype=torch.int32), paged_kv_indices=tab[0, :n].clone(),
                     paged_kv_last_page_len=torch.tensor([(L - 1) % PAGE + 1], dtype=torch.int32),
                     kv_lens=torch.tensor([L], dtype=torch.int32))
        A["page_idx"].copy_(table[0, (L - 1) // PAGE].reshape(1))
        A["cache_idx"].fill_((L - 1) % PAGE)

    def layers(h, res, first):
        gi = sum(t == "linear_attention" for t in types[:first])
        ai = sum(t == "full_attention" for t in types[:first])
        for i in range(first, len(types)):
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
        return h

    compiled_layers = torch.compile(layers, mode="max-autotune-no-cudagraphs", dynamic=False, fullgraph=False)

    @torch.no_grad()
    def step(token, t, arm):
        set_pos(t)
        res = W[token].reshape(1, -1).clone()
        h = S["n0"](res).detach()
        if arm == "block":
            blk(h.contiguous(), res.contiguous(), rb, hb, pool._blocks["conv"], pool._blocks["state"], 0, slots,
                cos_buf, sin_buf, kv[0], table, A["lens"], eps)
            h = layers(hb.clone(), rb.clone(), 4)
        elif arm == "compiled":
            h = compiled_layers(h, res, 0)
        else:
            h = layers(h, res, 0)
        return int(torch.argmax(h.float() @ W.float().t(), dim=-1).item())

    out = {"prompts": [], "args": vars(args)}
    for pi, line in enumerate(lines):
        ids = tok(line, add_special_tokens=False)["input_ids"][: args.prompt_len]
        gens = {}
        for arm in args.arms:
            pool._blocks["state"].zero_()
            pool._blocks["conv"].zero_()
            kv.zero_()
            nxt = None
            for t, tk in enumerate(ids):
                nxt = step(tk, t, arm)
            gen = [nxt]
            for j in range(args.gen - 1):
                gen.append(step(gen[-1], len(ids) + j, arm))
            gens[arm] = gen
        rec = {"i": pi, **gens}
        for a in args.arms:
            for b in args.arms:
                if a < b:
                    first = next((i for i, (x, y) in enumerate(zip(gens[a], gens[b], strict=True)) if x != y), None)
                    rec[f"{a}~{b}"] = first
        out["prompts"].append(rec)
        print(f"prompt {pi}: " + " ".join(f"{k}={v}" for k, v in rec.items() if "~" in k), flush=True)
    for a in args.arms:
        for b in args.arms:
            if a < b:
                k = f"{a}~{b}"
                n = sum(p[k] is None for p in out["prompts"])
                out[k] = n
                print(f"identical greedy replies {k}: {n}/{len(out['prompts'])}, first splits "
                      f"{sorted(p[k] for p in out['prompts'] if p[k] is not None)}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
