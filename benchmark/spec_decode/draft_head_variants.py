"""What a cheaper lm_head for the draft costs in acceptance, measured without a server.

    python -m benchmark.spec_decode.draft_head_variants --model Qwen/Qwen3.5-9B --k 3 --out variants.json

Runs greedy speculative decoding once with the full bf16 head and, at every draft position, also reads the head's
output through each variant (a quantized copy, or a subset of the vocabulary). A variant's accepted run in a step is
the longest prefix of its tokens equal to what the target emitted. The chain itself follows the bf16 drafts, so a
variant that is right where bf16 was wrong stops counting there: the figure is a slight lower bound on what the variant
reaches on its own chain. ``bf16`` is the control: its figure is the run's own tokens per step, give or take the last
step of each prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.spec_decode.mtp_acceptance import (
    CHAT_PROMPTS,
    LONG_PROMPTS,
    chat_ids,
    read_prompts,
    stop_ids_of,
)
from benchmark.spec_decode.qwen3_5_mtp import HFDraft, HFTarget, MtpHead, load_base, load_tokenizer
from benchmark.spec_decode.spec_loop import run_speculative
from benchmark.spec_decode.verify import SamplingParams


def fp8_round(x: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    """``x`` rounded through float8_e4m3fnuz at ``scale`` (the format holds up to 240), back in the original units."""
    q = (x.float() * scale).to(torch.float8_e4m3fnuz)
    return q.float() / scale


def int_round(x: torch.Tensor, scale: torch.Tensor | float, bits: int) -> torch.Tensor:
    """``x`` rounded to a signed ``bits``-bit integer at ``scale``, clipped, back in the original units."""
    top = 2 ** (bits - 1) - 1
    return ((x.float() * scale).round().clamp(-top, top)) / scale


@dataclass
class Variant:
    """One way of reading the draft head: ``weight`` [ids, hidden] with ``ids`` naming its rows (None: every row), and
    ``x_round`` applied to the head's input first."""
    name: str
    weight: torch.Tensor
    ids: torch.Tensor | None = None
    x_round: object = None
    agree: int = 0
    total: int = 0
    accepted: int = 0
    per_position: list[int] = field(default_factory=list)

    def token(self, hidden: torch.Tensor) -> int:
        x = hidden.float()
        if self.x_round is not None:
            x = self.x_round(x)
        logits = F.linear(x, self.weight.float())
        best = int(logits.argmax())
        return best if self.ids is None else int(self.ids[best])


class RecordingDraft(HFDraft):
    """The MTP head, keeping per step the bf16 draft tokens and each variant's: the steps' lists line up with the
    positions they predict, which the generated sequence settles afterwards."""

    def __init__(self, head: MtpHead, base, variants: list[Variant]):
        super().__init__(head, base)
        self.variants = variants
        # one entry per draft position: (predicted position, {variant name: token})
        self.steps: list[list[tuple[int, dict[str, int]]]] = []

    def reset(self) -> None:
        super().reset()
        self.steps = []

    def extend(self, hiddens: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logits = super().extend(hiddens, tokens)
        self.steps.append([self._record()])
        return logits

    def chain(self, token: int) -> torch.Tensor:
        logits = super().chain(token)
        self.steps[-1].append(self._record())
        return logits

    def _record(self) -> tuple[int, dict[str, int]]:
        # position p of the draft predicts the token at p + 2; its cache now holds positions up to length - 1
        return self.length + 1, {v.name: v.token(self.last_hidden[None]) for v in self.variants}


def score(variants: list[Variant], steps, prompt: list[int], generated: list[int], k: int) -> None:
    seq = prompt + generated
    for step in steps:
        if len(step) < k:
            continue  # a step the generation ended before verifying
        if step[0][0] >= len(seq):
            continue
        for v in variants:
            run = 0
            alive = True
            for j, (position, tokens) in enumerate(step):
                if position >= len(seq):
                    break
                if alive and tokens[v.name] == seq[position]:
                    run += 1
                    v.per_position[j] += 1
                else:
                    alive = False
                v.agree += tokens[v.name] == tokens["bf16"]
                v.total += 1
            v.accepted += run


def build_variants(names: list[str], weight: torch.Tensor, stop_ids: frozenset[int], k: int) -> list[Variant]:
    device = weight.device
    w = weight.float()
    variants = [Variant("bf16", w)]
    row_max = w.abs().amax(dim=1, keepdim=True)
    row_norm = w.norm(dim=1)
    specials = torch.tensor(sorted(stop_ids), device=device)
    for name in names:
        if name == "fp8":  # one scale for the head: argmax needs no per-channel scale back
            variants.append(Variant(name, fp8_round(w, 240.0 / float(row_max.max()))))
        elif name == "fp8-rowscale":
            variants.append(Variant(name, fp8_round(w, 240.0 / row_max)))
        elif name == "fp8-x":  # the input too, as an fp8 x fp8 kernel reads it
            variants.append(Variant(name, fp8_round(w, 240.0 / float(row_max.max())),
                                    x_round=lambda x: fp8_round(x, 240.0 / x.abs().amax(dim=-1, keepdim=True))))
        elif name.startswith("int"):
            # three of the head's standard deviations reach full scale: the far tail clips, the bulk keeps its bits
            bits, _, rowscale = name[3:].partition("-")
            top = 2 ** (int(bits) - 1) - 1
            scale = top / (3 * w.std(dim=1, keepdim=True)) if rowscale else top / (3 * float(w.std()))
            variants.append(Variant(name, int_round(w, scale, int(bits))))
        elif name.startswith("prefix-"):
            n = int(name.split("-")[1])
            ids = torch.cat((torch.arange(n, device=device), specials[specials >= n]))
            variants.append(Variant(name, w[ids], ids=ids))
        elif name.startswith("norm-"):
            n = int(name.split("-")[1])
            ids = torch.unique(torch.cat((row_norm.topk(n).indices, specials)))
            variants.append(Variant(name, w[ids], ids=ids))
        else:
            raise SystemExit(f"unknown variant {name}")
    for v in variants:
        v.per_position = [0] * k
    return variants


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-chat", type=int, default=32)
    ap.add_argument("--n-long", type=int, default=16)
    ap.add_argument("--max-new-chat", type=int, default=256)
    ap.add_argument("--max-new-long", type=int, default=128)
    ap.add_argument("--extra-prompts", help="a file of further prompts, one a line, scored as their own set")
    ap.add_argument("--variants", nargs="+", default=[
        "fp8", "fp8-rowscale", "fp8-x", "int8", "int8-rowscale", "int6-rowscale", "int4-rowscale",
        "prefix-64000", "prefix-96000", "prefix-128000", "prefix-151000",
        "norm-64000", "norm-124160",
    ])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    base, snapshot = load_base(args.model, args.device)
    tokenizer = load_tokenizer(snapshot)
    head = MtpHead.from_snapshot(snapshot, base.config, base.device, base.dtype)
    stop_ids = stop_ids_of(tokenizer, base.config)
    variants = build_variants(args.variants, base.lm_head.weight.detach(), stop_ids, args.k)
    target = HFTarget(base, "post")
    draft = RecordingDraft(head, base, variants)
    sets = {
        "chat": ([chat_ids(tokenizer, p) for p in read_prompts(CHAT_PROMPTS, args.n_chat)], args.max_new_chat),
        "long": ([chat_ids(tokenizer, p) for p in read_prompts(LONG_PROMPTS, args.n_long)], args.max_new_long),
    }
    if args.extra_prompts:
        extra = read_prompts(Path(args.extra_prompts), 1 << 30)
        sets["extra"] = ([chat_ids(tokenizer, p) for p in extra], args.max_new_chat)
    out: dict = {"model": args.model, "k": args.k, "sets": {}}
    for name, (prompts, max_new) in sets.items():
        for v in variants:
            v.agree = v.total = v.accepted = 0
            v.per_position = [0] * args.k
        run_steps = run_tokens = accepted = 0
        t0 = time.perf_counter()
        for i, prompt in enumerate(prompts):
            generator = torch.Generator(device=base.device)
            generator.manual_seed(args.seed * 1000 + i)
            generated, stats = run_speculative(
                target, draft, prompt, args.k, SamplingParams(), max_new, stop_ids, generator,
            )
            run_steps += stats.steps
            run_tokens += stats.tokens
            accepted += stats.accepted_total
            score(variants, draft.steps, prompt, generated, args.k)
        wall = time.perf_counter() - t0
        rows = {}
        control = max(next(v for v in variants if v.name == "bf16").accepted, 1)
        for v in variants:
            rows[v.name] = {
                "tokens_per_step": 1.0 + v.accepted / max(run_steps, 1),
                "accepted_vs_bf16": v.accepted / control,
                "agree_with_bf16": v.agree / max(v.total, 1),
                "accepted_per_position": [c / max(run_steps, 1) for c in v.per_position],
            }
            print(f"{name} {v.name:16s} tok/step {rows[v.name]['tokens_per_step']:.3f} accepted/bf16 "
                  f"{rows[v.name]['accepted_vs_bf16']:.4f} agree {rows[v.name]['agree_with_bf16']:.4f}", flush=True)
        out["sets"][name] = {"steps": run_steps, "tokens": run_tokens, "wall_s": wall, "variants": rows,
                             "run_tokens_per_step": 1 + accepted / max(run_steps, 1)}
        print(f"{name}: {len(prompts)} prompts, {run_steps} steps, {run_tokens} tokens, run tok/step "
              f"{1 + accepted / max(run_steps, 1):.3f}, {wall:.0f} s", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
