"""Whether a server samples text from the same distribution as another, for one that cannot reproduce the other's
draws at a seed (speculative decoding consumes its random numbers differently from plain decoding).

    python -m benchmark.spec_decode.sampled_dist save --url http://127.0.0.1:8100 --out spec.json
    python -m benchmark.spec_decode.sampled_dist compare base.json spec.json [--tokenizer Qwen/Qwen3.5-0.8B]

``save`` sends each of the first ``--prompts`` of T2's 48 prompts ``--samples`` times (one seed per sample) at
temperature 0.7, top-p 0.9 and top-k 40 for ``--max-tokens`` tokens. ``compare`` re-tokenizes every completion and, per
prompt and position 1..N-1 (position 0 is the prompt's own token, sampled the same way on both paths), tests the two
token marginals with a permutation test of their total-variation distance, and the same for the whole prefix up to that
position, which also sees how positions depend on each other (a draw reused between positions leaves the marginals
alone). Under the null each p-value is uniform, so the fraction below 0.01 should be near 1%. The same test between two
halves of the first run (``--null``) shows the level the harness reaches on identical servers.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import random
import sys
from pathlib import Path

import requests

from benchmark.spec_decode.serve_parity import prompt_set


def one(url: str, model: str, prompt: str, seed: int, args) -> str:
    body = {
        "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens,
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "seed": seed,
        "stream": False, "enable_thinking": False,
    }
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=args.timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def save(args) -> int:
    prompts = [p for _, p, _ in prompt_set()][: args.prompts]
    jobs = [(i, s) for i in range(len(prompts)) for s in range(args.samples)]
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        texts = list(pool.map(lambda js: one(args.url, args.model, prompts[js[0]], 1000 * js[0] + js[1], args), jobs))
    out = {"settings": {k: getattr(args, k) for k in ("temperature", "top_p", "top_k", "max_tokens", "samples")},
           "texts": [[texts[i * args.samples + s] for s in range(args.samples)] for i in range(len(prompts))]}
    Path(args.out).write_text(json.dumps(out))
    print(f"{len(jobs)} samples over {len(prompts)} prompts")
    return 0


def tv(a: list, b: list) -> float:
    ca, cb = collections.Counter(a), collections.Counter(b)
    return 0.5 * sum(abs(ca[t] / len(a) - cb[t] / len(b)) for t in set(ca) | set(cb))


def perm_p(a: list, b: list, rounds: int, rng: random.Random) -> float:
    observed, pooled, n = tv(a, b), a + b, len(a)
    hits = 0
    for _ in range(rounds):
        rng.shuffle(pooled)
        hits += tv(pooled[:n], pooled[n:]) >= observed - 1e-12
    return (hits + 1) / (rounds + 1)


def tests(xs: list[list[list[int]]], ys: list[list[list[int]]], positions: int, rounds: int, seed: int,
          joint: bool = False):
    """One row per (prompt, position): the two runs' tokens there, or with ``joint`` their prefixes through it."""
    rng = random.Random(seed)
    rows = []
    pick = (lambda s, p: tuple(s[1:p + 1])) if joint else (lambda s, p: s[p])
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        for p in range(1, positions):
            a = [pick(s, p) for s in x if len(s) > p]
            b = [pick(s, p) for s in y if len(s) > p]
            if len(a) < 10 or len(b) < 10:
                continue
            rows.append((i, p, tv(a, b), perm_p(a, b, rounds, rng)))
    return rows


def fisher(ps: list[float]) -> float:
    """Fisher's combined p-value: -2 sum(ln p) is chi-squared with 2m degrees of freedom."""
    import math

    if not ps:
        return 1.0
    x, m = -2 * sum(math.log(p) for p in ps), len(ps)
    # chi-squared survival at even degrees of freedom, in closed form
    term = total = math.exp(-x / 2)
    for i in range(1, m):
        term *= x / (2 * i)
        total += term
    return min(total, 1.0)


def summary(name: str, rows) -> str:
    ps = [r[3] for r in rows]
    small = sum(p < 0.01 for p in ps)
    share = 100 * small / max(len(rows), 1)
    return (f"{name}: {len(rows)} (prompt, position) tests, p<0.01 in {small} ({share:.1f}%), "
            f"p<0.05 in {sum(p < 0.05 for p in ps)}, mean TV {sum(r[2] for r in rows) / max(len(rows), 1):.3f}, "
            f"min p {min(ps, default=1):.4f}, Fisher combined p {fisher(ps):.3g}")


def compare(args) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    a, b = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    enc = [[[tok.encode(t, add_special_tokens=False) for t in prompt] for prompt in run["texts"]] for run in (a, b)]
    for name, joint in (("marginal", False), ("prefix", True)):
        rows = tests(enc[0], enc[1], args.positions, args.rounds, 0, joint)
        print(summary(f"{name} a vs b", rows))
        for r in rows:
            if r[3] < 0.01:
                print(f"  prompt {r[0]} position {r[1]}: TV {r[2]:.3f} p {r[3]:.4f}")
        if args.null:
            half = len(enc[0][0]) // 2
            print(summary(f"{name} a vs a (halves)", tests([p[:half] for p in enc[0]], [p[half:] for p in enc[0]],
                                                           args.positions, args.rounds, 1, joint)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("--url", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="qwen3_5")
    s.add_argument("--prompts", type=int, default=48)
    s.add_argument("--samples", type=int, default=200)
    s.add_argument("--max-tokens", type=int, default=6)
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--top-p", type=float, default=0.9)
    s.add_argument("--top-k", type=int, default=40)
    s.add_argument("--concurrency", type=int, default=32)
    s.add_argument("--timeout", type=float, default=300)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--tokenizer", required=True)
    c.add_argument("--positions", type=int, default=5)
    c.add_argument("--rounds", type=int, default=2000)
    c.add_argument("--null", action="store_true")
    args = ap.parse_args(argv)
    return save(args) if args.cmd == "save" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
