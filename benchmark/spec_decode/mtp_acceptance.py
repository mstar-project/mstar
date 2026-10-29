"""Acceptance rate of the Qwen3.5 MTP head against its own base model, measured without a server.

    python -m benchmark.spec_decode.mtp_acceptance --model Qwen/Qwen3.5-9B --calibrate --out acceptance.json

Loads the base model and the ``mtp.*`` head from the cached snapshot (offline), builds the chat and long-prompt sets
the sweep uses (benchmark/assets, chat template with thinking off), then runs real speculative decoding for every
(k, sampling mode) cell: k drafts chained from the head, verified token by token against the base model (spec_loop).
Reports per-position acceptance, mean accepted drafts and tokens per step, and writes everything to a JSON file
after each cell so a cut-off run still leaves its finished cells behind.

``--calibrate`` first compares the four input conventions the head could have been trained with (hidden state before
or after the base's final norm, embedding or hidden state first in the fc input) on a few prompts and keeps the one
with the highest greedy acceptance; a wrong convention shows up as near-zero acceptance, not as an error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from benchmark.spec_decode.qwen3_5_mtp import HFDraft, HFTarget, MtpHead, load_base, load_tokenizer
from benchmark.spec_decode.spec_loop import RunStats, run_speculative
from benchmark.spec_decode.verify import SamplingParams

REPO = Path(__file__).resolve().parents[2]
CHAT_PROMPTS = REPO / "benchmark/assets/simple_text_queries.txt"
LONG_PROMPTS = REPO / "benchmark/assets/long_text_queries_1k.txt"


def read_prompts(path: Path, n: int) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[:n]


def chat_ids(tokenizer, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def stop_ids_of(tokenizer, config) -> frozenset[int]:
    """The chat turn's end and the document end, as the serving stack stops on both."""
    ids = set()
    for value in (tokenizer.eos_token_id, getattr(config, "eos_token_id", None)):
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple)):
            ids.update(int(v) for v in value)
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    return frozenset(ids)


def fmt_rates(rates: list[float]) -> str:
    return "[" + " ".join(f"{r:.3f}" for r in rates) + "]"


def run_set(
    target: HFTarget,
    draft: HFDraft,
    prompts: list[list[int]],
    k: int,
    params: SamplingParams,
    max_new: int,
    stop_ids: frozenset[int],
    seed: int,
    deadline: float | None,
) -> tuple[RunStats, list[list[int]], float, bool]:
    """Run one prompt set. Returns the merged stats, the generations, the wall time and whether the deadline cut it."""
    stats = RunStats(k=k)
    outputs: list[list[int]] = []
    t0 = time.perf_counter()
    cut = False
    for i, prompt in enumerate(prompts):
        if deadline is not None and time.time() > deadline:
            cut = True
            break
        generator = torch.Generator(device=target.base.device)
        generator.manual_seed(seed * 1000 + i)
        tokens, run_stats = run_speculative(target, draft, prompt, k, params, max_new, stop_ids, generator)
        stats.merge(run_stats)
        outputs.append(tokens)
    return stats, outputs, time.perf_counter() - t0, cut


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chat-prompts", default=str(CHAT_PROMPTS))
    ap.add_argument("--long-prompts", default=str(LONG_PROMPTS))
    ap.add_argument("--n-chat", type=int, default=32)
    ap.add_argument("--n-long", type=int, default=16)
    ap.add_argument("--max-new-chat", type=int, default=256)
    ap.add_argument("--max-new-long", type=int, default=128)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--modes", nargs="+", default=["greedy", "sampling"], choices=["greedy", "sampling"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--hidden", default="post", choices=["post", "pre"], help="which base hidden state the head sees")
    ap.add_argument("--fc-order", default="emb_first", choices=["emb_first", "hidden_first"])
    ap.add_argument("--calibrate", action="store_true", help="pick --hidden/--fc-order by greedy acceptance first")
    ap.add_argument("--calibrate-prompts", type=int, default=6)
    ap.add_argument("--calibrate-tokens", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds; stop starting prompts after it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    base, snapshot = load_base(args.model, args.device)
    tokenizer = load_tokenizer(snapshot)
    head = MtpHead.from_snapshot(snapshot, base.config, base.device, base.dtype)
    load_s = time.perf_counter() - t0
    lm_head_bytes = base.lm_head.weight.numel() * base.lm_head.weight.element_size()
    print(
        f"{args.model}: snapshot {snapshot.name}, {type(base.model).__name__}, {base.dtype}, "
        f"{base.config.num_hidden_layers} layers, hidden {base.config.hidden_size}, vocab {base.config.vocab_size}; "
        f"mtp head {head.weight_bytes() / 1e6:.1f} MB (+ lm_head {lm_head_bytes / 1e6:.1f} MB shared); "
        f"loaded in {load_s:.0f} s",
        flush=True,
    )
    stop_ids = stop_ids_of(tokenizer, base.config)

    chat_prompts = read_prompts(Path(args.chat_prompts), args.n_chat)
    long_prompts = read_prompts(Path(args.long_prompts), args.n_long)
    sets = {
        "chat": ([chat_ids(tokenizer, p) for p in chat_prompts], args.max_new_chat),
        "long": ([chat_ids(tokenizer, p) for p in long_prompts], args.max_new_long),
    }
    for name, (prompts, max_new) in sets.items():
        lengths = [len(p) for p in prompts]
        print(f"{name}: {len(prompts)} prompts, {min(lengths)}-{max(lengths)} tokens, up to {max_new} new", flush=True)

    result = {
        "model": args.model,
        "snapshot": str(snapshot),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": {
            "num_hidden_layers": base.config.num_hidden_layers,
            "hidden_size": base.config.hidden_size,
            "vocab_size": base.config.vocab_size,
            "mtp_num_hidden_layers": getattr(base.config, "mtp_num_hidden_layers", None),
        },
        "mtp_weight_bytes": head.weight_bytes(),
        "lm_head_bytes": lm_head_bytes,
        "prompts": {
            name: {"file": args.chat_prompts if name == "chat" else args.long_prompts, "n": len(prompts),
                   "max_new": max_new, "token_counts": [len(p) for p in prompts]}
            for name, (prompts, max_new) in sets.items()
        },
        "sampling": {"temperature": args.temperature, "top_p": args.top_p},
        "calibration": [],
        "cells": [],
        "samples": [],
    }
    out = Path(args.out)

    def save() -> None:
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    import transformers

    result["transformers"] = transformers.__version__

    hidden, fc_order = args.hidden, args.fc_order
    if args.calibrate:
        cal_prompts = sets["chat"][0][: args.calibrate_prompts]
        best = None
        for cand_hidden in ("post", "pre"):
            for cand_order in ("emb_first", "hidden_first"):
                target = HFTarget(base, cand_hidden)
                draft = HFDraft(head, base, cand_order)
                stats, _, wall, _ = run_set(
                    target, draft, cal_prompts, 4, SamplingParams(), args.calibrate_tokens, stop_ids, args.seed, None
                )
                target.close()
                row = {"hidden": cand_hidden, "fc_order": cand_order, "mean_accepted": stats.mean_accepted,
                       "position_rates": stats.position_rates(), "steps": stats.steps}
                result["calibration"].append(row)
                print(
                    f"calibrate hidden={cand_hidden} fc_order={cand_order}: accepted/step {stats.mean_accepted:.3f} "
                    f"p={fmt_rates(stats.position_rates())} ({stats.steps} steps, {wall:.0f} s)",
                    flush=True,
                )
                if best is None or stats.mean_accepted > best[0]:
                    best = (stats.mean_accepted, cand_hidden, cand_order)
        _, hidden, fc_order = best
        print(f"calibrate: using hidden={hidden} fc_order={fc_order}", flush=True)
    result["chosen"] = {"hidden": hidden, "fc_order": fc_order}
    save()

    target = HFTarget(base, hidden)
    draft = HFDraft(head, base, fc_order)
    greedy_outputs: dict[tuple[str, int], list[list[int]]] = {}
    cut = False
    for mode in args.modes:
        params = SamplingParams() if mode == "greedy" else SamplingParams(args.temperature, args.top_p)
        for k in args.k:
            if cut:
                break
            cell = {"mode": mode, "temperature": params.temperature, "top_p": params.top_p, "k": k, "sets": {}}
            total = RunStats(k=k)
            wall_total = 0.0
            for name, (prompts, max_new) in sets.items():
                stats, outputs, wall, cut = run_set(
                    target, draft, prompts, k, params, max_new, stop_ids, args.seed, args.deadline
                )
                wall_total += wall
                summary = stats.summary()
                summary["wall_s"] = wall
                summary["hf_ms_per_token"] = 1e3 * wall / max(stats.tokens, 1)
                summary["complete"] = not cut
                cell["sets"][name] = summary
                total.merge(stats)
                if mode == "greedy":
                    greedy_outputs[(name, k)] = outputs
                if outputs and k == args.k[0]:
                    for i in range(min(2, len(outputs))):
                        result["samples"].append({
                            "set": name, "index": i, "mode": mode, "k": k,
                            "text": tokenizer.decode(outputs[i], skip_special_tokens=False)[:600],
                        })
                print(
                    f"{args.model} {mode} k={k} {name}: {stats.prompts} prompts, {stats.steps} steps, "
                    f"{stats.tokens} tokens, accepted/step {stats.mean_accepted:.3f}, tokens/step "
                    f"{stats.tokens_per_step:.3f}, p={fmt_rates(stats.position_rates())} "
                    f"E[p]={fmt_rates(stats.position_expected_rates())} reach={fmt_rates(stats.reach_rates())} "
                    f"({wall:.0f} s, {summary['hf_ms_per_token']:.1f} ms/token){' CUT' if cut else ''}",
                    flush=True,
                )
                if cut:
                    break
            cell["sets"]["all"] = total.summary()
            cell["wall_s"] = wall_total
            result["cells"].append(cell)
            save()
        if cut:
            print("deadline reached; remaining cells skipped", flush=True)
            break

    # Greedy speculative decoding must reproduce greedy decoding whatever k is: the target is driven one token at
    # a time either way, so the continuations should agree across k token for token.
    consistency = {"prompts_compared": 0, "prompts_differing": 0}
    for name in sets:
        runs = [greedy_outputs[(name, k)] for k in args.k if (name, k) in greedy_outputs]
        if len(runs) < 2:
            continue
        for i in range(min(len(r) for r in runs)):
            consistency["prompts_compared"] += 1
            if any(r[i] != runs[0][i] for r in runs[1:]):
                consistency["prompts_differing"] += 1
    result["greedy_consistency"] = consistency
    save()

    print(f"\n{args.model} (hidden={hidden}, fc_order={fc_order}); greedy outputs differing across k: "
          f"{consistency['prompts_differing']}/{consistency['prompts_compared']}")
    print(f"{'mode':9s} {'k':>2s} {'set':5s} {'steps':>6s} {'tok/step':>8s} {'acc/step':>8s} {'p1..pk':30s} E[p1..pk]")
    for cell in result["cells"]:
        for name in ("chat", "long", "all"):
            s = cell["sets"].get(name)
            if s is None:
                continue
            print(
                f"{cell['mode']:9s} {cell['k']:2d} {name:5s} {s['steps']:6d} {s['tokens_per_step']:8.3f} "
                f"{s['mean_accepted']:8.3f} {fmt_rates(s['position_rates']):30s} "
                f"{fmt_rates(s['position_expected_rates']):30s}"
            )
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
