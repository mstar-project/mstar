"""Drive a running Zonos2 server to check that request batching works.

Prereq: server up (see test/zonos2/README.md), e.g.
    mstar serve zonos2 --gpus 0
For the reproducibility test, the model must sample with a *fixed* seed
(TTSSamplingParams(seed=...) in zonos2_model.py) — otherwise identical
requests intentionally diverge.

Usage:
    python test/zonos2/test_batching.py --url http://localhost:20002 -n 4
"""
from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

from mstar import MStarClient


def _h(pcm: bytes) -> str:
    return hashlib.sha256(pcm).hexdigest()[:12]


def fire_concurrent(url: str, texts: list[str]) -> list:
    """Submit all texts at once so they co-schedule into one batch."""
    client = MStarClient(url)
    with ThreadPoolExecutor(max_workers=len(texts)) as ex:
        # small stagger-free launch: gather futures first, then results
        futs = [ex.submit(client.tts, t) for t in texts]
        return [f.result() for f in futs]


def test_concurrency_correctness(url: str, n: int) -> None:
    """Different texts, fired together: each returns valid, DISTINCT audio.
    Catches per-request state bleed / cross-talk under batching."""
    texts = [f"This is test sentence number {i}, spoken clearly." for i in range(n)]
    outs = fire_concurrent(url, texts)
    lens = [len(o.pcm) for o in outs]
    hashes = [_h(o.pcm) for o in outs]
    print(f"[correctness] n={n}  bytes={lens}")
    assert all(l > 0 for l in lens), "a request returned empty audio"
    assert len(set(hashes)) == n, f"distinct texts produced duplicate audio: {hashes}"
    print("[correctness] PASS — all non-empty and distinct")


def test_reproducibility(url: str, n: int) -> None:
    """IDENTICAL requests, fired together, must yield byte-identical audio.
    This is the acid test for the change: it fails if batching corrupts
    per-request state, or if sampling is position-dependent. Requires a
    fixed model seed."""
    text = "The quick brown fox jumps over the lazy dog."
    outs = fire_concurrent(url, [text] * n)
    hashes = [_h(o.pcm) for o in outs]
    print(f"[reproducibility] n={n} identical reqs -> hashes={hashes}")
    assert len(set(hashes)) == 1, (
        "identical co-batched requests diverged — either non-fixed seed, "
        "state bleed, or position-dependent sampling"
    )
    # Run the whole batch again: must reproduce.
    again = {_h(o.pcm) for o in fire_concurrent(url, [text] * n)}
    assert again == set(hashes), "batch not reproducible across runs"
    print("[reproducibility] PASS — identical within batch and across runs")


def test_throughput(url: str, n: int) -> None:
    """Wall-clock for 1 request vs n concurrent. Batched serving should make
    n-concurrent far cheaper than n x single (until compute-bound)."""
    text = "Benchmarking the batched decode path end to end."
    t0 = time.time(); MStarClient(url).tts(text); solo = time.time() - t0
    t0 = time.time(); fire_concurrent(url, [text] * n); conc = time.time() - t0
    print(f"[throughput] solo={solo:.2f}s  {n}-concurrent={conc:.2f}s  "
          f"speedup vs {n}x-serial={n * solo / conc:.1f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:20002")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--repro", action="store_true", help="run reproducibility test (needs fixed seed)")
    args = ap.parse_args()

    assert MStarClient(args.url).health(), f"server not healthy at {args.url}"
    test_concurrency_correctness(args.url, args.n)
    test_throughput(args.url, args.n)
    if args.repro:
        test_reproducibility(args.url, args.n)
