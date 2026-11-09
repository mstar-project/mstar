#!/usr/bin/env python3
"""Teacher-force Ripple-65B's served replies through the checkpoint's own modeling code.

The oracle never imports mstar. It runs in its own venv (transformers 4.57.6, the version
config.json names) and imports a read copy of ``modeling_ricursive.py`` and
``configuration_ricursive.py`` from ``--ref-dir``, not through ``trust_remote_code``.

Every ``<tag>.json`` that ``run_parity.py`` wrote into ``--runs`` holds rows of ``messages``,
the server's ``prompt_ids`` and its reply ``text``. For each row this records, at every
position of the reply and at the stop, the served token's rank and the oracle's top-5 and
top-1/top-2 logit gap. ``run_parity.py gate`` reads the result; nothing is decided here.

Variants, all in one session of one loaded model:

    base        the reference as shipped, every tag
    no_gate     attention output gate off, ``--control-tag`` only: the gate's control
    no_valres   value residual off, ``--control-tag`` only: the gate's control

    python test/ripple/record_oracle.py --ref-dir ref/ripple_ref --ckpt <snapshot> \
        --runs out/runs --out out/oracle_bf16.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import torch

TOP_K = 5
NEAR_TIE = 0.25  # as run_parity.py
MAX_REALIGN = 32
STOP_IDS = (100265, 100257)  # <|im_end|>, <|endoftext|>


def load_reference(ref_dir: Path, ckpt: str, dtype: torch.dtype):
    assert "mstar" not in sys.modules, "the oracle owes nothing to the code it judges"
    sys.path.insert(0, str(ref_dir.resolve().parent))
    module = importlib.import_module(f"{ref_dir.resolve().name}.modeling_ricursive")
    model, info = module.RicursiveForCausalLM.from_pretrained(
        ckpt, dtype=dtype, device_map="auto", output_loading_info=True,
    )
    bad = {k: info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys") if info[k]}
    if bad:
        raise SystemExit(f"the reference did not load the checkpoint cleanly: {bad}")
    return model.eval()


@torch.inference_mode()
def forced_logits(model, prompt_ids: list[int], ids: list[int]) -> torch.Tensor:
    """fp32 logits at each reply position, plus one row after the last token."""
    device = model.model.embed_tokens.weight.device
    seq = torch.tensor([prompt_ids + ids], device=device)
    hidden = model.model(input_ids=seq, use_cache=False).last_hidden_state
    return model.lm_head(hidden[0, len(prompt_ids) - 1:]).float()


def retokenize(model, tok, prompt_ids: list[int], text: str, cap: int) -> tuple[list[int], torch.Tensor, int]:
    """The reply's ids and their logits. The chat API returns text, and re-tokenizing can split
    it differently from how it was sampled, so where the oracle's argmax spells the same text,
    take its split and force again: only a disagreement that changes the text remains."""
    def enc(s: str) -> list[int]:
        return tok(s, add_special_tokens=False)["input_ids"]

    ids, artifacts = enc(text)[:cap], 0
    for _ in range(MAX_REALIGN):
        logits = forced_logits(model, prompt_ids, ids)
        top = logits[:len(ids)].argmax(-1).tolist()
        for k, (served, best) in enumerate(zip(ids, top, strict=True)):
            if served == best:
                continue
            near = logits[k].topk(TOP_K)
            if served in near.indices.tolist() and float(near.values[0] - near.values[1]) <= NEAR_TIE:
                continue  # a near tie the server may really have sampled: taking the other split
                          # would force a sequence it never saw (`.` then `\n\n` for `.\n\n`)
            alt = ids[:k] + [best]
            spelled = tok.decode(alt)
            if spelled != tok.decode(ids[:k]) and (text.startswith(spelled) or spelled.startswith(text)):
                ids, artifacts = (alt + enc(text[len(spelled):]))[:cap], artifacts + 1
                break
        else:
            return ids, logits, artifacts
    return ids, forced_logits(model, prompt_ids, ids), artifacts


def score(logits: torch.Tensor, ids: list[int], stopped: bool) -> dict:
    """Per position: rank, the oracle's top-k, its top-1/top-2 gap, the gap to the served token."""
    n = len(ids)
    served = torch.tensor(ids, device=logits.device)
    rows = logits[:n]
    if stopped:  # the reply ended: the stop token is a served token too, whichever of the two
        last = logits[n]
        rows = logits[:n + 1]
        stop = torch.tensor(STOP_IDS, device=logits.device)
        served = torch.cat([served, stop[last[stop].argmax()].reshape(1)])
    picked = rows.gather(1, served[:, None])
    top = rows.topk(TOP_K, dim=-1)
    return {
        "served": served.tolist(),
        "rank": (rows > picked).sum(-1).tolist(),
        "top5": top.indices.tolist(),
        "top5_logits": [[round(x, 3) for x in r] for r in top.values.tolist()],
        "gap12": (top.values[:, 0] - top.values[:, 1]).tolist(),
        "gap_served": (top.values[:, 0] - picked[:, 0]).tolist(),
        "stopped": stopped,
    }


def run_tag(model, tok, rows: list[dict], fixed: list[dict] | None = None) -> list[dict]:
    """Score a tag's rows; ``fixed`` reuses the base variant's ids instead of aligning again."""
    out = []
    for i, row in enumerate(rows):
        cap = row["max_tokens"]
        text = row["text"].rstrip("�")  # a reply cut by max_tokens can end inside a character
        if fixed is not None:
            ids, artifacts = fixed[i]["ids"], fixed[i]["retokenized"]
            logits = forced_logits(model, row["prompt_ids"], ids)
        else:
            ids, logits, artifacts = retokenize(model, tok, row["prompt_ids"], text, cap)
        # the re-tokenized length can trail the served one by a little: only a clear stop counts
        scored = score(logits, ids, stopped=len(ids) < cap - 2)
        out.append({"ids": ids, "prompt_tokens": len(row["prompt_ids"]), "retokenized": artifacts, **scored})
        print(f"  [{i}] prompt {len(row['prompt_ids'])} reply {len(ids)} artifacts {artifacts} "
              f"max_rank {max(scored['rank'], default=0)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", type=Path, required=True, help="directory of the two reference files")
    ap.add_argument("--ckpt", required=True, help="local snapshot of the checkpoint")
    ap.add_argument("--runs", type=Path, required=True, help="the <tag>.json files of run_parity.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--control-tag", default="tp2")
    a = ap.parse_args()

    from transformers import AutoTokenizer  # after argparse: --help needs no transformers

    torch.set_float32_matmul_precision("highest")
    tok = AutoTokenizer.from_pretrained(a.ckpt, trust_remote_code=False)
    t0 = time.time()
    model = load_reference(a.ref_dir, a.ckpt, torch.bfloat16)
    print(f"reference loaded in {time.time() - t0:.0f}s on {sorted({str(p.device) for p in model.parameters()})}",
          flush=True)
    runs = {f.stem: json.loads(f.read_text()) for f in sorted(a.runs.glob("*.json"))}
    report = {"base": {}, "no_gate": {}, "no_valres": {}}

    for tag, rows in runs.items():
        for row in rows:
            ours = tok.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True,
                                           return_dict=False)
            if ours != row["prompt_ids"]:
                raise SystemExit(f"{tag}: mstar's prompt ids differ from the template's: {row['messages']}")
        print(f"base / {tag}", flush=True)
        report["base"][tag] = run_tag(model, tok, rows)
        a.out.write_text(json.dumps(report))

    base = report["base"].get(a.control_tag)
    if base is not None:
        rows = runs[a.control_tag]
        for name, field in (("no_gate", "use_gated_attention"), ("no_valres", "use_valres")):
            setattr(model.config, field, False)
            print(f"{name} / {a.control_tag}", flush=True)
            report[name][a.control_tag] = run_tag(model, tok, rows, fixed=base)
            setattr(model.config, field, True)
    a.out.write_text(json.dumps(report))
    print(f"wrote {a.out} after {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
