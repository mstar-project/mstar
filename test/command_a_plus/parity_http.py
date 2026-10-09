"""Record greedy outputs and decode speed from a running Command A+ server.

Run once against an eager server and once against a CUDA-graph server, then
compare the two JSON files with ``--compare``.

    python -m test.command_a_plus.parity_http --url http://127.0.0.1:8000 --output eager.json
    python -m test.command_a_plus.parity_http --url http://127.0.0.1:8000 --output graphs.json \
      --compare eager.json
"""

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests

PROMPTS = [
    "What is 2 + 2? Answer briefly.",
    "Complete this sentence: The capital of France is",
    "Write one short sentence greeting a new colleague.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 50.",
    "Translate to Spanish: I would like a cup of coffee, please.",
]


def generate(url, prompt, max_tokens, repetition_penalty=None):
    model_kwargs = {"max_output_tokens": max_tokens, "temperature": 0}
    if repetition_penalty is not None:
        model_kwargs["repetition_penalty"] = repetition_penalty
    response = requests.post(url.rstrip("/") + "/generate", data={
        "text": prompt, "streaming": "false", "output_modalities": "text",
        "model_kwargs": json.dumps(model_kwargs),
    }, timeout=600)
    response.raise_for_status()
    chunks = response.json()["outputs"].get("text", [])
    return b"".join(base64.b64decode(chunk["data"]) for chunk in chunks).decode("utf-8"), len(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    args = parser.parse_args()

    def run(prompt):
        return generate(args.url, prompt, args.max_tokens, args.repetition_penalty)

    generate(args.url, "Warm up.", 8)
    sequential = []
    for prompt in PROMPTS:
        started = time.perf_counter()
        text, tokens = run(prompt)
        seconds = time.perf_counter() - started
        sequential.append(dict(prompt=prompt, text=text, tokens=tokens, seconds=seconds))
        print(f"[{tokens:3d} tok, {tokens / seconds:5.1f} tok/s] {prompt!r} -> {text[-80:]!r}", flush=True)

    started = time.perf_counter()
    with ThreadPoolExecutor(len(PROMPTS)) as pool:
        concurrent = list(pool.map(run, PROMPTS))
    seconds = time.perf_counter() - started
    total = sum(tokens for _, tokens in concurrent)
    print(f"concurrent x{len(PROMPTS)}: {total} tokens in {seconds:.1f}s -> {total / seconds:.1f} tok/s total",
          flush=True)

    result = dict(sequential=sequential, concurrent=[dict(prompt=p, text=t, tokens=n)
                                                    for p, (t, n) in zip(PROMPTS, concurrent, strict=True)])
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    if args.compare:
        with open(args.compare) as f:
            reference = json.load(f)
        for kind in ("sequential", "concurrent"):
            same = sum(a["text"] == b["text"] for a, b in zip(result[kind], reference[kind], strict=True))
            print(f"{kind}: {same}/{len(PROMPTS)} outputs identical to {args.compare}", flush=True)
            for a, b in zip(result[kind], reference[kind], strict=True):
                if a["text"] != b["text"]:
                    prefix = next((i for i, (x, y) in enumerate(zip(a["text"], b["text"], strict=False)) if x != y),
                                  min(len(a["text"]), len(b["text"])))
                    print(f"  differs at char {prefix}: {a['prompt']!r}\n    new: {a['text'][prefix:prefix + 60]!r}"
                          f"\n    ref: {b['text'][prefix:prefix + 60]!r}", flush=True)


if __name__ == "__main__":
    main()
