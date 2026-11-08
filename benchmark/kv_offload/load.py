"""Session-free text load for ``tight_pool.yaml``.

Each request is a ~0.3k-2.5k token prompt (a number to remember, filler, then
the question) with a short reply, so a few running requests fill the 64-page
pool and admissions keep evicting each other to the host. Prints the outcome
counts and the wall time; the eviction count is in the server log
(``grep -c "Offloaded request"``).

    python benchmark/kv_offload/load.py --url http://localhost:8000 --concurrency 32
"""

import argparse
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

from mstar import MStarClient

STATE = "Remember this number: {number}. Reply with just the word OK."
ASK = "What was the number I asked you to remember? Answer with digits only."
FILLER = (
    "Here is some unrelated text to read and then ignore: the lighthouse keeper "
    "logged the tides, the ferry ran late, and the gulls argued over the pier. "
)


def one(args, i, tally):
    rng = random.Random(args.seed * 7919 + i)
    turns = rng.randint(1, args.turns - 1)
    text = STATE.format(number=rng.randint(10**6, 10**7 - 1)) + (
        FILLER * args.filler + " ") * turns + ASK
    try:
        MStarClient(args.url, timeout=args.request_timeout).generate(
            text=text, max_output_tokens=args.max_output_tokens, top_k=1)
        tally["ok"] += 1
    except requests.Timeout:
        tally["timeout"] += 1
    except requests.HTTPError as e:
        tally[f"http {e.response.status_code}"] += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--requests", type=int, default=384)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--turns", type=int, default=6, help="filler blocks, up to turns - 1")
    ap.add_argument("--filler", type=int, default=12, help="filler repeats a block (~35 tokens each)")
    ap.add_argument("--max-output-tokens", type=int, default=16)
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    tally = Counter()
    start = time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        for i in range(args.requests):
            pool.submit(one, args, i, tally)
    print(f"concurrency {args.concurrency}: {dict(tally)} in {time.time() - start:.1f}s", flush=True)
    return 0 if set(tally) == {"ok"} else 1


if __name__ == "__main__":
    sys.exit(main())
