"""Quality gates for a quantized deployment against its bf16 one, over any OpenAI-compatible server (M*, vLLM,
SGLang), greedy and with thinking off, so two engines' arms differ only by what they serve.

    python -m benchmark.quantization.qa_gate gsm8k --url http://127.0.0.1:8000 --model Qwen/Qwen3.5-27B \\
        --limit 500 --label mstar-fp8 --out res/
    python -m benchmark.quantization.qa_gate mmlu_pro ...   # TIGER-Lab/MMLU-Pro test split, a seeded sample
    python -m benchmark.quantization.qa_gate agree --ref res/agree_mstar-bf16.jsonl ...   # greedy text vs an arm

``gsm8k``: 0-shot, "put the final answer after ####", exact match on the last number. ``mmlu_pro``: 0-shot CoT,
"The answer is (X)". Both write one JSON line per item (``<task>_<label>.jsonl``) and print a summary line with
accuracy and its 95% interval; ``compare`` takes two arms' item files and prints the paired difference with its
discordant counts (the McNemar view), which is what a format gate reads: a paired test needs far fewer items
than two unpaired accuracies to see a 1-2 point drop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import time
from pathlib import Path

import aiohttp

GSM8K_PROMPT = "{question}\nPlease reason step by step, and put your final answer after '####'."
MMLU_PROMPT = (
    "The following is a multiple choice question. Think step by step and then finish your answer with "
    "'The answer is (X)' where X is the correct letter.\n\nQuestion: {question}\nOptions:\n{options}"
)
LETTERS = "ABCDEFGHIJ"
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _hf_parquet(repo: str, pattern: str) -> list[Path]:
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=[pattern]))
    return sorted(root.glob(pattern))


def load_gsm8k(limit: int | None) -> list[dict]:
    import pandas as pd

    frame = pd.concat(pd.read_parquet(p) for p in _hf_parquet("openai/gsm8k", "main/test-*.parquet"))
    items = [{"id": i, "question": q, "gold": a.split("####")[-1].strip().replace(",", "")}
             for i, (q, a) in enumerate(zip(frame["question"], frame["answer"], strict=True))]
    return items[:limit] if limit else items


def load_mmlu_pro(limit: int | None, seed: int) -> list[dict]:
    import pandas as pd

    frame = pd.concat(pd.read_parquet(p) for p in _hf_parquet("TIGER-Lab/MMLU-Pro", "data/test-*.parquet"))
    items = [{"id": int(qid), "question": q, "options": list(opts), "gold": ans, "category": cat}
             for qid, q, opts, ans, cat in zip(frame["question_id"], frame["question"], frame["options"],
                                               frame["answer"], frame["category"], strict=True)]
    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items


def gsm8k_answer(text: str) -> str | None:
    tail = text.rsplit("####", maxsplit=1)[-1] if "####" in text else text
    nums = _NUM.findall(tail)
    if not nums:
        return None
    value = nums[-1 if "####" not in text else 0].replace(",", "").rstrip(".")
    try:
        return str(int(float(value))) if float(value).is_integer() else value
    except ValueError:
        return value


def mmlu_answer(text: str) -> str | None:
    found = re.findall(r"answer is \(?([A-J])\)?", text)
    if found:
        return found[-1]
    found = re.findall(r"\b([A-J])\b", text[-40:])
    return found[-1] if found else None


async def _chat(session, url, model, prompt, max_tokens, timeout) -> tuple[str, int]:
    body = {
        "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
        "temperature": 0.0, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(f"{url}/v1/chat/completions", json=body,
                            timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        data = await resp.json()
    usage = data.get("usage") or {}
    return data["choices"][0]["message"]["content"] or "", int(usage.get("completion_tokens", 0))


async def _run(items, make_prompt, args) -> list[dict]:
    sem = asyncio.Semaphore(args.concurrency)
    out: list[dict] = []
    async with aiohttp.ClientSession() as session:
        async def one(item):
            async with sem:
                try:
                    text, tokens = await _chat(session, args.url, args.model, make_prompt(item), args.max_tokens,
                                               args.timeout)
                except Exception as e:  # noqa: BLE001 - recorded per item, counted as wrong
                    text, tokens = f"<error {type(e).__name__}: {e}>", 0
                out.append({**{k: v for k, v in item.items() if k != "options"}, "text": text, "tokens": tokens})
        await asyncio.gather(*(one(item) for item in items))
    return sorted(out, key=lambda r: r["id"])


def _ci(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval."""
    if n == 0:
        return 0.0, 0.0
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _write(rows, task, args, extra) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{task}_{args.label}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {"task": task, "label": args.label, "n": len(rows), **extra}
    with open(out / "summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)


def run_task(task: str, args) -> None:
    t0 = time.monotonic()
    if task == "gsm8k":
        items = load_gsm8k(args.limit)
        rows = asyncio.run(_run(items, lambda it: GSM8K_PROMPT.format(question=it["question"]), args))
        for row in rows:
            row["pred"] = gsm8k_answer(row["text"])
            row["correct"] = row["pred"] == row["gold"]
    elif task == "mmlu_pro":
        items = load_mmlu_pro(args.limit, args.seed)
        def prompt(it):
            options = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it["options"]))
            return MMLU_PROMPT.format(question=it["question"], options=options)
        rows = asyncio.run(_run(items, prompt, args))
        for row in rows:
            row["pred"] = mmlu_answer(row["text"])
            row["correct"] = row["pred"] == row["gold"]
    else:  # agree: greedy text of the first `limit` GSM8K questions against a reference arm's
        items = load_gsm8k(args.limit)
        rows = asyncio.run(_run(items, lambda it: GSM8K_PROMPT.format(question=it["question"]), args))
        if args.ref:
            ref = {r["id"]: r["text"] for r in map(json.loads, open(args.ref))}
            for row in rows:
                other = ref.get(row["id"], "")
                row["same"] = row["text"] == other
                row["common_prefix"] = next((i for i, (a, b) in enumerate(zip(row["text"], other, strict=False))
                                             if a != b), min(len(row["text"]), len(other)))
        n = len(rows)
        same = sum(r.get("same", False) for r in rows)
        prefix = sorted(r.get("common_prefix", 0) for r in rows)
        _write(rows, task, args, {"identical": same, "median_common_prefix_chars": prefix[n // 2] if n else 0,
                                  "seconds": round(time.monotonic() - t0, 1)})
        return
    k = sum(r["correct"] for r in rows)
    lo, hi = _ci(k, len(rows))
    errors = sum(r["text"].startswith("<error") for r in rows)
    n = max(len(rows), 1)
    _write(rows, task, args, {"accuracy": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)], "errors": errors,
                              "mean_tokens": round(sum(r["tokens"] for r in rows) / n),
                              "seconds": round(time.monotonic() - t0, 1)})


def compare(a_path: str, b_path: str) -> None:
    a = {r["id"]: r["correct"] for r in map(json.loads, open(a_path))}
    b = {r["id"]: r["correct"] for r in map(json.loads, open(b_path))}
    ids = sorted(set(a) & set(b))
    only_a = sum(a[i] and not b[i] for i in ids)
    only_b = sum(b[i] and not a[i] for i in ids)
    n = len(ids)
    # exact two-sided McNemar on the discordant pairs
    d = only_a + only_b
    tail = sum(math.comb(d, i) for i in range(min(only_a, only_b) + 1)) / 2 ** d if d else 1.0
    print(json.dumps({"a": a_path, "b": b_path, "n": n, "acc_a": round(sum(a[i] for i in ids) / n, 4),
                      "acc_b": round(sum(b[i] for i in ids) / n, 4), "only_a": only_a, "only_b": only_b,
                      "mcnemar_p": round(min(1.0, 2 * tail), 4)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=("gsm8k", "mmlu_pro", "agree", "compare"))
    parser.add_argument("files", nargs="*", help="compare: two item files")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="default")
    parser.add_argument("--label", default="arm")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--ref", default=None, help="agree: the reference arm's agree_*.jsonl")
    parser.add_argument("--out", default="qa-gate")
    args = parser.parse_args()
    if args.task == "compare":
        compare(*args.files)
    else:
        run_task(args.task, args)


if __name__ == "__main__":
    main()
