"""Closed-loop chat-completions benchmark for draft verify (speculative decoding).

The same requests to any OpenAI-compatible server (M*, vLLM, SGLang), streamed: TTFT, TPOT and
output tokens per second, and the drafts each server accepted. Output tokens are the model
tokenizer's count of the streamed text, so every server is counted the same way (M*'s chat
endpoint reports no usage).

Acceptance: from vLLM's ``/metrics`` counters (``--metrics-url``), or from the M* server log's
draft-verify line (``--server-log``), read before and after the measured requests.

    python -m benchmark.draft_verify.bench --url http://127.0.0.1:8100 \\
        --prompts out/code_edit.jsonl --tokenizer Qwen/Qwen3.5-0.8B --concurrency 8 \\
        --num-requests 64 --out out/run.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

import aiohttp

_DV_LINE = re.compile(r"draft-verify(?:, final)?: (\d+) verify rows, ([\d.]+) drafts accepted per row")
_VLLM_COUNTERS = ("spec_decode_num_accepted_tokens", "spec_decode_num_draft_tokens", "spec_decode_num_drafts")


def _load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


async def _one(session, url, model, row, args) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": row["prompt"]}],
        "max_tokens": args.max_tokens or row.get("max_tokens", 512),
        "temperature": args.temperature,
        "stream": True,
        # M* reads the flat key, vLLM and SGLang the template kwargs
        "enable_thinking": args.thinking,
        "chat_template_kwargs": {"enable_thinking": args.thinking},
    }
    if args.ignore_eos:
        body["ignore_eos"] = True
    start = time.perf_counter()
    first = None
    parts: list[str] = []
    error = None
    try:
        async with session.post(f"{url}/v1/chat/completions", json=body) as resp:
            if resp.status != 200:
                error = f"HTTP {resp.status}: {(await resp.text())[:300]}"
            else:
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    for choice in chunk.get("choices") or ():
                        text = (choice.get("delta") or {}).get("content") or ""
                        if text:
                            if first is None:
                                first = time.perf_counter()
                            parts.append(text)
    except Exception as exc:  # noqa: BLE001 - counted as a failed request
        error = repr(exc)
    end = time.perf_counter()
    return {"id": row["id"], "start": start, "first": first, "end": end, "text": "".join(parts), "error": error}


async def _run(args, rows: list[dict], n: int) -> tuple[list[dict], float]:
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def go(i):
            async with sem:
                return await _one(session, args.url, args.served_model, rows[i % len(rows)], args)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(go(i) for i in range(n)))
        return results, time.perf_counter() - t0


def _metrics(url: str) -> dict[str, float]:
    import urllib.request

    text = urllib.request.urlopen(f"{url}/metrics", timeout=10).read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for name in _VLLM_COUNTERS:
            if line.startswith(f"vllm:{name}_total"):
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def _dv_log(path: str) -> tuple[int, float] | None:
    last = None
    for match in _DV_LINE.finditer(Path(path).read_text(errors="replace")):
        last = (int(match.group(1)), float(match.group(2)))
    return last


def _summary(results: list[dict], wall: float, tokenizer) -> dict:
    ok = [r for r in results if r["error"] is None and r["first"] is not None]
    for r in ok:
        r["tokens"] = len(tokenizer.encode(r["text"], add_special_tokens=False))
        r["ttft"] = r["first"] - r["start"]
        r["tpot"] = (r["end"] - r["first"]) / max(r["tokens"] - 1, 1)
    tokens = sum(r["tokens"] for r in ok)

    def stats(key):
        values = sorted(r[key] for r in ok)
        if not values:
            return {}
        return {"mean": statistics.fmean(values), "p50": values[len(values) // 2],
                "p90": values[min(len(values) - 1, int(0.9 * len(values)))]}

    return {
        "completed": len(ok), "failed": len(results) - len(ok), "wall_s": wall,
        "output_tokens": tokens, "output_tok_s": tokens / wall if wall else 0.0,
        "mean_output_tokens": tokens / len(ok) if ok else 0.0,
        "ttft_s": stats("ttft"), "tpot_s": stats("tpot"),
        "errors": sorted({r["error"] for r in results if r["error"]})[:5],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", required=True)
    p.add_argument("--served-model", default="default", help="the request's model field (vLLM checks it)")
    p.add_argument("--prompts", required=True, help="JSONL from benchmark.draft_verify.prompts")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--num-requests", type=int, default=None, help="default: one per prompt")
    p.add_argument("--warmup-waves", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=None, help="default: each prompt's own")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--metrics-url", default=None, help="vLLM: read its spec-decode counters")
    p.add_argument("--server-log", default=None, help="M*: read its draft-verify log line")
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = _load(args.prompts)
    n = args.num_requests or len(rows)
    if args.warmup_waves:
        # other prompts than the measured ones' first, so no prefix cache hit carries over
        warm = rows[::-1]
        asyncio.run(_run(args, warm, args.concurrency * args.warmup_waves))
    before_m = _metrics(args.metrics_url) if args.metrics_url else None
    before_l = _dv_log(args.server_log) if args.server_log else None
    results, wall = asyncio.run(_run(args, rows, n))
    summary = _summary(results, wall, tokenizer)
    if args.metrics_url:
        after = _metrics(args.metrics_url)
        delta = {k: after.get(k, 0.0) - before_m.get(k, 0.0) for k in _VLLM_COUNTERS}
        drafts = delta["spec_decode_num_drafts"]
        summary["acceptance"] = {
            **delta,
            "accepted_per_step": delta["spec_decode_num_accepted_tokens"] / drafts if drafts else None,
            "acceptance_rate": (delta["spec_decode_num_accepted_tokens"] / delta["spec_decode_num_draft_tokens"]
                                if delta["spec_decode_num_draft_tokens"] else None),
        }
    if args.server_log:
        after = _dv_log(args.server_log)
        if after is not None:
            rows0, mean0 = before_l or (0, 0.0)
            verified = after[0] - rows0
            accepted = after[0] * after[1] - rows0 * mean0
            summary["acceptance"] = {
                "verify_rows": verified,
                "accepted_per_step": accepted / verified if verified else None,
            }
    summary.update({"tag": args.tag, "concurrency": args.concurrency, "num_requests": n, "prompts": args.prompts})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "requests": results}, indent=1))
    keep = {k: summary[k] for k in ("tag", "concurrency", "completed", "failed", "output_tok_s", "mean_output_tokens")}
    keep["ttft_p50"] = summary["ttft_s"].get("p50")
    keep["tpot_p50"] = summary["tpot_s"].get("p50")
    keep["acceptance"] = summary.get("acceptance", {}).get("accepted_per_step")
    print(json.dumps(keep))


if __name__ == "__main__":
    main()
