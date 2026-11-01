"""What each way of coupling a draft to the target would accept, on the same real distributions, without a server.

    python -m benchmark.spec_decode.coupling_stats --model Qwen/Qwen3.5-9B --temperature 0.7 --top-p 0.9 --top-k 40 \\
        --k 2 --out stats.json

Runs speculative sampling (spec_loop) and, at every judged position, takes the filtered draft distribution p and
target distribution q and computes in closed form what a verify step would accept there under each scheme. Each
one emits a token distributed exactly as q; they differ in how often that token equals the draft:

- ``leviathan``: rejection sampling on (p, q), sum min(p, q), the most any coupling can reach. Needs p(d) and q(d)
  and, on rejection, a draw from (q - p)+.
- ``unfiltered_draft``: the same with the draft at the temperature alone, no top-k or top-p, which is what filtering
  the draft buys: any distribution the draft really came from is valid.
- ``argmax``: a deterministic draft, the head's argmax, against an independent draw from q: q(argmax).
- ``cdf``: inverse CDF in token-id order with one shared uniform.
- ``gumbel``: argmax of log + noise with shared per-token noise. The chance both pick x is 1 / sum_y max(p_y / p_x,
  q_y / q_x).

The cdf and gumbel schemes need neither p(d) nor the residual, which is why they are worth pricing. Also reports how
many tokens the filters keep and how they cluster in the vocabulary, which is what sizes a kernel that holds a row's
kept tokens in registers.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

from benchmark.spec_decode.mtp_acceptance import CHAT_PROMPTS, LONG_PROMPTS, chat_ids, read_prompts, stop_ids_of
from benchmark.spec_decode.qwen3_5_mtp import HFDraft, HFTarget, MtpHead, load_base, load_tokenizer
from benchmark.spec_decode.spec_loop import run_speculative
from benchmark.spec_decode.verify import SamplingParams


def leviathan(p: torch.Tensor, q: torch.Tensor) -> float:
    return float(torch.minimum(p, q).sum())


def cdf_overlap(p: torch.Tensor, q: torch.Tensor) -> float:
    """The chance a shared uniform lands in the same token's interval of both CDFs, in token-id order."""
    ids = torch.nonzero((p > 0) | (q > 0)).reshape(-1)
    pp, qq = p[ids].double(), q[ids].double()
    pp, qq = pp / pp.sum(), qq / qq.sum()
    hi_p, hi_q = pp.cumsum(0), qq.cumsum(0)
    lo_p, lo_q = hi_p - pp, hi_q - qq
    return float((torch.minimum(hi_p, hi_q) - torch.maximum(lo_p, lo_q)).clamp_min(0).sum())


def gumbel_match(p: torch.Tensor, q: torch.Tensor) -> float:
    ids = torch.nonzero((p > 0) | (q > 0)).reshape(-1)
    pp, qq = p[ids].double(), q[ids].double()
    both = (pp > 0) & (qq > 0)
    px, qx = pp[both][:, None], qq[both][:, None]
    denom = torch.maximum(pp[None, :] / px, qq[None, :] / qx).sum(dim=1)
    return float((1.0 / denom).sum())


def tile_peak(kept: torch.Tensor, tile: int) -> int:
    return int(torch.bincount(torch.div(kept, tile, rounding_mode="floor")).max())


class Observer:
    def __init__(self, k: int, temperature: float):
        self.temperature = temperature
        self.sums = [defaultdict(float) for _ in range(k)]
        self.counts = [0] * k
        self.kept: list[int] = []
        self.peak_8192: list[int] = []
        self.peak_2048: list[int] = []

    def __call__(self, position: int, p: torch.Tensor, q: torch.Tensor, draft_logits: torch.Tensor) -> None:
        s = self.sums[position]
        self.counts[position] += 1
        s["leviathan"] += leviathan(p, q)
        s["unfiltered_draft"] += leviathan(torch.softmax(draft_logits.float() / self.temperature, -1), q)
        s["argmax"] += float(q[draft_logits.argmax()])
        s["cdf"] += cdf_overlap(p, q)
        s["gumbel"] += gumbel_match(p, q)
        kept = torch.nonzero(q > 0).reshape(-1)
        self.kept.append(kept.numel())
        self.peak_8192.append(tile_peak(kept, 8192))
        self.peak_2048.append(tile_peak(kept, 2048))


def quantiles(values: list[int]) -> dict:
    x = torch.tensor(values, dtype=torch.float64)
    return {f"p{int(q * 100)}": float(x.quantile(q)) for q in (0.5, 0.9, 0.99)} | {"max": float(x.max())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--n-chat", type=int, default=16)
    ap.add_argument("--n-long", type=int, default=8)
    ap.add_argument("--max-new-chat", type=int, default=128)
    ap.add_argument("--max-new-long", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    base, snapshot = load_base(args.model, "cuda")
    tokenizer = load_tokenizer(snapshot)
    head = MtpHead.from_snapshot(snapshot, base.config, base.device, base.dtype)
    target = HFTarget(base, "post")
    draft = HFDraft(head, base, "emb_first")
    stop_ids = stop_ids_of(tokenizer, base.config)
    params = SamplingParams(args.temperature, args.top_p, args.top_k)
    sets = [
        (read_prompts(CHAT_PROMPTS, args.n_chat), args.max_new_chat),
        (read_prompts(LONG_PROMPTS, args.n_long), args.max_new_long),
    ]
    observer = Observer(args.k, args.temperature)
    tokens = steps = i = 0
    for prompts, max_new in sets:
        for text in prompts:
            generator = torch.Generator(device=base.device).manual_seed(args.seed * 1000 + i)
            i += 1
            out, stats = run_speculative(target, draft, chat_ids(tokenizer, text), args.k, params, max_new, stop_ids,
                                         generator, observe=observer)
            tokens += len(out)
            steps += stats.steps
    schemes = {
        name: [s[name] / max(n, 1) for s, n in zip(observer.sums, observer.counts, strict=True)]
        for name in ("leviathan", "unfiltered_draft", "argmax", "cdf", "gumbel")
    }
    # tokens a step emits when each position accepts at its mean rate: 1 + a1 + a1 a2 + ...
    per_step = {}
    for name, rates in schemes.items():
        run, total = 1.0, 1.0
        for r in rates:
            run *= r
            total += run
        per_step[name] = total
    result = {
        "model": args.model, "k": args.k, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "tokens": tokens, "steps": steps, "judged": observer.counts, "accept_per_position": schemes,
        "tokens_per_step": per_step, "kept_tokens": quantiles(observer.kept),
        "tile_8192_peak": quantiles(observer.peak_8192), "tile_2048_peak": quantiles(observer.peak_2048),
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
