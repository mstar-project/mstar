"""GSM8K accuracy of a chat model behind an OpenAI-compatible route (M*, vLLM, SGLang).

Greedy, zero-shot with an instruction to end on "The answer is N."; a reply is right when the
last number in it equals the reference answer. A gate of RFC #1186 ("Correctness") compares Qwen3-8B with
and without the DSpark drafter: speculative decoding must not move accuracy beyond noise.

    python -m benchmark.qwen3.gsm8k --url http://127.0.0.1:8000 --model qwen3_8b --limit 500 --out gsm8k.json
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

INSTRUCTION = "Solve the problem. Think step by step, then end with: The answer is N."
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> str | None:
    found = NUMBER.findall(text)
    return found[-1].replace(",", "").rstrip(".") if found else None


def ask(url: str, model: str, question: str, max_tokens: int) -> dict:
    body = {
        "model": model, "temperature": 0.0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": f"{INSTRUCTION}\n\n{question}"}],
        # the chat template answers directly: as OpenAI clients send it (Qwen3), and flat (M*'s
        # Qwen3.5 reads that one)
        "chat_template_kwargs": {"enable_thinking": False},
        "enable_thinking": False,
    }
    r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=3600)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    return {
        "reply": choice["message"]["content"] or "", "finish_reason": choice.get("finish_reason"),
        "tokens": (data.get("usage") or {}).get("completion_tokens"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from datasets import load_dataset

    rows = list(load_dataset("openai/gsm8k", "main", split="test"))[: args.limit]
    with ThreadPoolExecutor(args.concurrency) as pool:
        replies = list(pool.map(lambda row: ask(args.url, args.model, row["question"], args.max_tokens), rows))
    graded = [
        {"answer": row["answer"].split("####")[-1].strip().replace(",", ""), **reply}
        for row, reply in zip(rows, replies, strict=True)
    ]
    for g in graded:
        g["correct"] = last_number(g["reply"]) == g["answer"]
    accuracy = sum(g["correct"] for g in graded) / len(graded)
    # a reply cut at max_tokens, or one that opened a thinking block, says the score isn't the answers'
    lengths = sorted(g["tokens"] for g in graded if g["tokens"] is not None)
    summary = {
        "accuracy": accuracy, "n": len(graded),
        "cut_at_max_tokens": sum(g["finish_reason"] == "length" for g in graded),
        "thinking": sum("<think>" in g["reply"] for g in graded),
        "median_tokens": lengths[len(lengths) // 2] if lengths else None,
    }
    Path(args.out).write_text(json.dumps({"args": vars(args), **summary, "rows": graded}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
