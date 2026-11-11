"""Long-context decode speed and passkey retrieval against a running server, by prompt length.

Each prompt is filler with a five-digit pass key in it and a question at its end, sized with the
model's tokenizer, and streams greedily from ``/v1/chat/completions``. Each rep of a length
draws its own key (``--seed``) and puts it at the next of ``--depths`` (fractions of the
filler), so a length's reps are different prompts, not one greedy reply repeated. Per request:
time to first token, decode time per token ((last chunk - first chunk) / (tokens - 1)), the
median chunk gap, token counts, the key's depth and whether the reply names the key, or a
refusal's status.

``--ignore-eos`` decodes all of ``--max-tokens`` so decode times compare across lengths;
without it the reply ends where the model stops, which is how retrieval is judged. Pointed at
vLLM or SGLang (``--url``, ``--model``) it gives numbers to compare. For example::

    mstar-serve --config configs/glm5_next_tp8_longctx.yaml
    python -m benchmark.glm.longctx_probe --tokenizer zai-org/GLM-5.3-Flash \\
        --lengths 4000,32000,128000 --reps 3 --max-tokens 256 --ignore-eos --output probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def passkey_text(repeats: int, key: str, depth: float = 0.5) -> str:
    """``repeats`` filler lines with the key between two of them, ``depth`` of the way in."""
    before = round(repeats * depth)
    return (FILLER * before + f"The pass key is {key}. Remember it. {key} is the pass key. "
            + FILLER * (repeats - before) + "\nWhat is the pass key? Answer with the number only.")


def filler_repeats(tokenizer, target_tokens: int) -> int:
    """Filler lines for a prompt of about ``target_tokens`` tokens (within 2%)."""
    repeats = max(1, target_tokens // 20)
    for _ in range(4):
        count = len(tokenizer.encode(passkey_text(repeats, "00000")))
        if abs(count / target_tokens - 1) < 0.02:
            break
        repeats = max(1, round(repeats * target_tokens / count))
    return repeats


def rep_key(seed: int, length: int, rep: int) -> str:
    """Rep ``rep`` of length ``length``'s five-digit key, the same on every run of a seed."""
    return str(random.Random(f"{seed}-{length}-{rep}").randrange(10000, 100000))


def served_model(url: str) -> str:
    with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def stream_one(url: str, model: str, text: str, max_tokens: int, ignore_eos: bool, usage: bool,
               timeout: float) -> dict:
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
            "messages": [{"role": "user", "content": text}]}
    if ignore_eos:
        body["ignore_eos"] = True
    if usage:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, pieces, reported = [], [], {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[len("data: "):])
                reported = event.get("usage") or reported
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = ((delta.get("reasoning_content") or delta.get("reasoning") or "")
                             + (delta.get("content") or ""))
                    if piece:
                        stamps.append(time.perf_counter())
                        pieces.append(piece)
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode(errors="replace")[:500]}
    if not stamps:
        return {"status": 200, "error": "no tokens streamed"}
    # a server that streams several tokens a chunk reports them in its usage
    tokens = reported.get("completion_tokens") or len(stamps)
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    return {
        "status": 200,
        "ttft_s": round(stamps[0] - t0, 3),
        "decode_ms_per_token": round((stamps[-1] - stamps[0]) * 1000 / max(1, tokens - 1), 2),
        "chunk_gap_p50_ms": round(statistics.median(gaps) * 1000, 2) if gaps else None,
        "chunks": len(stamps),
        "completion_tokens": tokens,
        "prompt_tokens": reported.get("prompt_tokens"),
        "text": "".join(pieces),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", help="the served model name (default: the server's first)")
    p.add_argument("--tokenizer", required=True, help="Hugging Face id or path, to size the prompts")
    p.add_argument("--lengths", default="4000,32000,128000", help="prompt tokens, comma separated")
    p.add_argument("--reps", type=int, default=3, help="requests per length")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--ignore-eos", action="store_true", help="decode all of --max-tokens")
    p.add_argument("--depths", default="0.1,0.5,0.9",
                   help="where the key sits, as fractions of the filler; rep i takes the i-th, cycling")
    p.add_argument("--seed", type=int, default=0, help="draws each rep's key")
    p.add_argument("--no-usage", action="store_true", help="don't ask for stream_options.include_usage")
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--output", help="append one JSON line per request here")
    a = p.parse_args()
    depths = [float(d) for d in a.depths.split(",")]
    if not all(0 <= d <= 1 for d in depths):
        p.error(f"--depths are fractions in [0, 1], not {a.depths}")

    from transformers import AutoTokenizer

    url = a.url.rstrip("/")
    model = a.model or served_model(url)
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    print(f"{'length':>8} {'rep':>3} {'status':>6} {'ttft s':>8} {'ms/token':>9} {'gap p50':>8} "
          f"{'prompt':>8} {'tokens':>6} {'depth':>5}  passkey")
    for length in (int(n) for n in a.lengths.split(",")):
        repeats = filler_repeats(tokenizer, length)
        for rep in range(a.reps):
            key, depth = rep_key(a.seed, length, rep), depths[rep % len(depths)]
            text = passkey_text(repeats, key, depth)
            counted = len(tokenizer.encode(text))
            row = stream_one(url, model, text, a.max_tokens, a.ignore_eos, not a.no_usage, a.timeout)
            row.update(length=length, rep=rep, prompt_tokens_counted=counted, ignore_eos=a.ignore_eos,
                       key=key, depth=depth, passkey_found=key in row.get("text", ""))
            if row["status"] == 200 and "error" not in row:
                print(f"{length:>8} {rep:>3} {row['status']:>6} {row['ttft_s']:>8} "
                      f"{row['decode_ms_per_token']:>9} {row['chunk_gap_p50_ms']!s:>8} "
                      f"{row['prompt_tokens'] or counted:>8} {row['completion_tokens']:>6} "
                      f"{depth:>5}  {row['passkey_found']}")
            else:
                print(f"{length:>8} {rep:>3} {row['status']:>6}  {row['error'][:120]}")
            if a.output:
                with open(a.output, "a") as f:
                    f.write(json.dumps({"url": url, "model": model, **row}) + "\n")


if __name__ == "__main__":
    main()
