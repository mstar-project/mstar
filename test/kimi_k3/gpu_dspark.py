"""Kimi K3.1's DSpark drafter (``Inferact/Kimi-K3.1-Pro-DSpark-preview``) with its real weights on
a GPU, beside the target's embedding and ``lm_head`` from a cut of the release (``truncate.py``).

Checks: every release tensor but the confidence head loads, strictly; the absorbed fp32 drafter
matches the unabsorbed float64 reference (``tests/test_dspark._reference``) at the release's
geometry, and both drafters' states are compared with it layer by layer (``bisect``, at 512 and
8192 context rows: the context cache rows, after each attention and MLP, the final norm); bf16
against fp32 over rows of long contexts (logit error over the logits' spread, how many drafts
agree); context rows appended in chunks equal them appended at once. The target's
tapped states (layers 75-91) are not in a 4-layer cut, so the context states are synthetic: this
checks the drafter's weights and numerics, not its acceptance, which needs the whole target
(``--cpu`` runs it on a tiny drafter, for a dry run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn


def _target_head(target: str, device, dtype) -> tuple[nn.Embedding, nn.Linear]:
    from safetensors import safe_open

    want = {"language_model.model.embed_tokens.weight": None, "language_model.lm_head.weight": None}
    for shard in sorted(Path(target).glob("*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for key in want:
                if key in f.keys():
                    want[key] = f.get_tensor(key)
    missing = [k for k, v in want.items() if v is None]
    if missing:
        raise SystemExit(f"{target}: no {missing}")
    emb, head = want.values()
    embed = nn.Embedding(*emb.shape).to(device=device, dtype=dtype)
    lm_head = nn.Linear(head.shape[1], head.shape[0], bias=False).to(device=device, dtype=dtype)
    with torch.no_grad():
        embed.weight.copy_(emb)
        lm_head.weight.copy_(head)
    return embed, lm_head


def _build(args, device, dtype):
    from mstar.model.kimi_k3.dspark.config import DSparkConfig
    from mstar.model.kimi_k3.dspark.model import DSparkDraft
    from mstar.model.loader.iterators import iter_safetensors_shards

    cfg = DSparkConfig.from_dir(args.draft)
    target = json.loads((Path(args.target) / "config.json").read_text())
    text = target.get("text_config", target)
    if (text["hidden_size"], text["vocab_size"]) != (cfg.target_hidden_size, cfg.vocab_size):
        raise SystemExit(f"target {text['hidden_size']} / {text['vocab_size']} is not the drafter's")
    embed, lm_head = _target_head(args.target, device, dtype)
    with torch.device(device):
        draft = DSparkDraft(cfg, embed, lm_head).to(dtype)
    loaded = draft.load_weights(iter_safetensors_shards(args.draft), strict=True)
    return draft.eval(), cfg, len(loaded)


def drafter_trace(draft, states: torch.Tensor, anchor: int) -> dict:
    """The drafter's block run layer by layer at the points ``_reference(trace=...)`` records."""
    cfg = draft.cfg
    dev = states.device
    k, tc = cfg.block_size, states.shape[0]
    contexts = draft.context_latents(states.to(next(draft.parameters()).dtype), torch.arange(tc, device=dev))
    positions = torch.arange(tc, tc + k, device=dev)
    x = draft.embed_tokens(draft.block_ids(torch.tensor([anchor], device=dev), k).reshape(-1))
    out = {}
    for i, layer in enumerate(draft.layers):
        out["ctx", i] = contexts[i]
        x = x + layer.self_attn.forward_block(layer.input_layernorm(x), positions, [contexts[i]])
        out["attn", i] = x
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        out["mlp", i] = x
    out["final"] = draft.final_norm(x)
    return out


def bisect(drafts: dict, cfg, states: torch.Tensor, anchor: int, label: str) -> list[dict]:
    """Each drafter (``name -> module``) against the float64 reference at every recorded point:
    the max error and the median one over the reference's std there. The reference rounds its rope
    angles as fp32 does (vLLM's and HF's tables, and the drafter's): exact angles drift from them by
    position x 3e-8 rad, which job 1473 saw as a 20x larger error at 8192 rows than at 512. The
    exact-angle reference's rows are printed too (``.../exact_angles``), ungated."""
    from mstar.model.kimi_k3.tests.test_dspark import _reference

    first = next(iter(drafts.values()))
    ref: dict = {}
    _reference(first, cfg, states.cpu(), anchor, trace=ref, fp32_angles=True)
    exact: dict = {}
    _reference(first, cfg, states.cpu(), anchor, trace=exact, fp32_angles=False)
    rows = []
    for name, draft in drafts.items():
        got = drafter_trace(draft, states, anchor)
        for point in (("ctx", 0), "final"):
            diff = (got[point].double().cpu() - exact[point]).abs()
            print(json.dumps({"case": f"bisect/{label}/{name}/exact_angles", "point": str(point),
                              "err_max": diff.max().item(),
                              "err_median_over_std": (diff.median() / exact[point].std()).item()}), flush=True)
        for point, want in ref.items():
            diff = (got[point].double().cpu() - want).abs()
            where = point if isinstance(point, str) else f"{point[0]}{point[1]}"
            row = {"case": f"bisect/{label}/{name}", "point": where,
                   "err_max": diff.max().item(), "err_median_over_std": (diff.median() / want.std()).item()}
            print(json.dumps(row), flush=True)
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, help="the drafter's dir (config.json, model.safetensors)")
    ap.add_argument("--target", required=True, help="a cut of the target (its embedding and lm_head)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ctx", type=int, default=4096, help="context rows of the longest request")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--ref-ctx", type=int, default=256, help="context of the float64 reference's request")
    ap.add_argument("--cpu", action="store_true", help="a dry run on the CPU")
    ap.add_argument("--bisect-ctx", type=int, nargs="*", default=[512, 8192],
                    help="contexts at which each layer's states are compared with the float64 reference")
    args = ap.parse_args()
    if args.cpu:
        args.device = "cpu"
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mstar.model.kimi_k3.dspark.model import DSparkContextStore
    from mstar.model.kimi_k3.tests.test_dspark import _reference

    torch.manual_seed(0)
    dev = torch.device(args.device)
    # an fp32 oracle in fp32: no TF32 GEMMs (the engine sets precision per node; this script has none)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    failures = []
    draft32, cfg, n32 = _build(args, dev, torch.float32)
    draft16, _, n16 = _build(args, dev, torch.bfloat16)
    params = {name for name, _ in draft32.named_parameters()}
    print(json.dumps({"case": "load", "params_loaded": n32, "params": len(params), "bf16_loaded": n16,
                      "heads": cfg.num_attention_heads, "markov_rank": cfg.markov_rank, "block": cfg.block_size,
                      "causal": cfg.causal, "taps": cfg.target_layer_ids}), flush=True)
    if n32 != len(params) or n16 != len(params):
        failures.append("load: not every parameter got a tensor")

    g = torch.Generator().manual_seed(1)

    def states(n: int) -> torch.Tensor:  # synthetic tapped states, combined as the target's would be
        aux = [torch.randn(n, cfg.target_hidden_size, generator=g).to(dev) for _ in cfg.target_layer_ids]
        acc = draft32.context_accumulator()
        for j, a in enumerate(aux):
            acc(j, a)
        return acc.finish()

    with torch.inference_mode():
        # 1. fp32 absorbed against the float64 unabsorbed reference
        s = states(args.ref_ctx)
        anchor = int(torch.randint(0, cfg.vocab_size, (1,), generator=g))
        drafts, logits = draft32.draft_dense(s, anchor)
        ref_drafts, ref_logits = _reference(draft32, cfg, s.cpu(), anchor)
        err = (logits.double().cpu() - ref_logits).abs().max().item()
        spread = ref_logits.std().item()
        row = {"case": "fp32_vs_float64_ref", "ctx": args.ref_ctx, "logits_max_err": err, "logits_std": spread,
               "drafts_equal": bool(torch.equal(drafts.cpu(), ref_drafts))}
        print(json.dumps(row), flush=True)
        if err > 1e-3 * max(spread, 1.0) or not row["drafts_equal"]:
            failures.append(f"fp32 drafter vs reference: {row}")

        # 1b. layer by layer, both drafters, short and long contexts: where an error starts
        for n in args.bisect_ctx:
            for row in bisect({"fp32": draft32, "bf16": draft16}, cfg, states(n), anchor, f"ctx{n}"):
                # fp32 should sit at fp32 rounding everywhere; bf16 is reported (gated on logits below)
                if row["case"].endswith("/fp32") and row["err_median_over_std"] > 1e-4:
                    failures.append(f"fp32 drafter off the reference at {row['point']}: {row}")

        # 2. bf16 against fp32 over rows of long contexts, rows batched; chunked appends
        lengths = [max(1, args.ctx - 97 * r) for r in range(args.rows)]
        anchors = torch.randint(0, cfg.vocab_size, (args.rows,), generator=g).to(dev)
        ctx_states = [states(n) for n in lengths]
        store32, store16 = DSparkContextStore(draft32), DSparkContextStore(draft16)
        chunked = DSparkContextStore(draft16)
        rids = [f"r{i}" for i in range(args.rows)]
        for rid, st in zip(rids, ctx_states, strict=True):
            store32.append(rid, st, 0)
            store16.append(rid, st.bfloat16(), 0)
            half = st.shape[0] // 2
            chunked.append(rid, st[:half].bfloat16(), 0)
            chunked.append(rid, st[half:].bfloat16(), half)
        d32, l32 = draft32.draft(anchors, store32.contexts(rids))
        d16, l16 = draft16.draft(anchors, store16.contexts(rids))
        dc, lc = draft16.draft(anchors, chunked.contexts(rids))
        err16 = (l16.float() - l32).abs()
        row = {"case": "bf16_vs_fp32", "ctx": lengths, "logits_err_median": err16.median().item(),
               "logits_err_max": err16.max().item(), "logits_std": l32.std().item(),
               "first_draft_agree": int((d16[:, 0] == d32[:, 0]).sum()), "drafts_agree": int((d16 == d32).sum()),
               "drafts": int(d32.numel()), "finite": bool(torch.isfinite(l16).all())}
        print(json.dumps(row), flush=True)
        if not row["finite"] or row["logits_err_median"] > 0.1 * row["logits_std"]:
            failures.append(f"bf16 drafter vs fp32: {row}")
        row = {"case": "chunked_vs_whole_bf16", "logits_max_diff": (lc.float() - l16.float()).abs().max().item(),
               "drafts_equal": bool(torch.equal(dc, d16))}
        print(json.dumps(row), flush=True)
        if row["logits_max_diff"] > 0.05 * l32.std().item():
            failures.append(f"chunked context rows: {row}")
    for f in failures:
        print("FAIL:", f, flush=True)
    print("PASS" if not failures else f"{len(failures)} failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
