"""Phi-4-multimodal over OpenAI chat: one client for M* and vLLM alike.

Closed loop at a fixed concurrency: each worker sends its next request when its last
one finishes. Every request is greedy with a fixed output length (``ignore_eos``), so
both engines generate the same number of tokens; a run whose mean output falls short
of it is flagged (``short_output``), since an engine can stop on ids its
``ignore_eos`` doesn't cover. Output tokens come from the engine's ``usage`` where it
reports one (vLLM: a LoRA reply run past its end samples special tokens the streamed
text leaves out), else from the streamed chunks, which M* sends one per token.
Re-tokenizing the text reads a few percent high; it is kept as a cross-check
(``mean_retokenized``).

Workloads:
- ``text``: short questions (benchmark/assets/simple_text_queries.txt), 128 tokens out
- ``image``: a Food-101 image (as stored, ~512 px) and a question, 128 tokens out
- ``asr``: a LibriSpeech test-clean utterance to transcribe
- ``mixed``: the three in turn

vLLM serves the checkpoint's modality LoRAs as named adapters; ``--engine vllm`` names
the one each prompt needs in ``model`` (vLLM's ``--default-mm-loras`` picks it too).

usage: python benchmark/phi4mm/bench.py --url http://127.0.0.1:8310 --engine mstar \
    --workload image --concurrency 32 --requests 160 --out results.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import io
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field

import aiohttp

HUB = os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Result:
    ttft: float
    latency: float
    tokens: int
    itl: float | None
    ok: bool
    error: str = ""
    chunks: int = 0  # streamed deltas that carried text
    retokenized: int = 0


@dataclass
class Summary:
    engine: str
    workload: str
    concurrency: int
    requests: int
    failed: int
    wall_s: float
    output_tok_s: float
    req_s: float
    ttft_ms: dict = field(default_factory=dict)
    itl_ms: dict = field(default_factory=dict)
    latency_s: dict = field(default_factory=dict)
    mean_output_tokens: float = 0.0
    mean_chunks: float = 0.0
    mean_retokenized: float = 0.0
    # the mean output fell short of the length asked for: tok/s doesn't compare
    short_output: bool = False


def _stats(xs: list[float], scale: float = 1.0) -> dict:
    if not xs:
        return {}
    xs = sorted(x * scale for x in xs)
    pick = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    return {"mean": round(statistics.fmean(xs), 2), "p50": round(pick(0.5), 2), "p90": round(pick(0.9), 2),
            "p99": round(pick(0.99), 2)}


def _texts(n: int) -> list[str]:
    path = os.path.join(REPO, "benchmark/assets/simple_text_queries.txt")
    lines = [line.strip() for line in open(path) if line.strip()]
    return [lines[i % len(lines)] for i in range(n)]


def _images(n: int, offset: int) -> list[str]:
    import pyarrow.parquet as pq

    path = glob.glob(f"{HUB}/datasets--ethz--food101/snapshots/*/data/train-00000-of-00008.parquet")[0]
    rows = pq.read_table(path, columns=["image"]).slice(offset, n).to_pylist()
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} images from offset {offset}; every request needs its own")
    return [base64.b64encode(r["image"]["bytes"]).decode() for r in rows]


def _clips(n: int, offset: int) -> list[str]:
    import pyarrow.parquet as pq
    import soundfile as sf

    path = glob.glob(f"{HUB}/datasets--openslr--librispeech_asr/snapshots/*/clean/test/0000.parquet")[0]
    rows = pq.read_table(path, columns=["audio"]).slice(offset, n).to_pylist()
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} clips from offset {offset}; every request needs its own")
    out = []
    for r in rows:
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]))
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def build(workload: str, n: int, output_len: int, offset: int) -> list[tuple[str, list[dict], int]]:
    """``(adapter, content, max_tokens)`` per request. Every image and clip is its own,
    from ``offset`` on: engines that cache encoder outputs or prefixes across requests
    (vLLM does) would otherwise serve repeats from the cache."""
    kinds = ["text", "image", "asr"] if workload == "mixed" else [workload]
    count = {k: sum(1 for i in range(n) if kinds[i % len(kinds)] == k) for k in kinds}
    texts = _texts(n)
    images = _images(count["image"], offset) if "image" in count else []
    clips = _clips(count["asr"], offset) if "asr" in count else []
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        if kind == "text":
            out.append(("base", [{"type": "text", "text": texts[i]}], output_len))
        elif kind == "image":
            image = images.pop(0)
            out.append(("vision", [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
                                   {"type": "text", "text": "Describe this image in detail."}], output_len))
        else:
            clip = clips.pop(0)
            out.append(("speech", [{"type": "input_audio", "input_audio": {"data": clip, "format": "wav"}},
                                   {"type": "text", "text": "Transcribe the audio clip into text."}], output_len))
    return out


async def one(session, url, engine, model, adapter, content, max_tokens, tok) -> Result:
    body = {"model": model if engine == "mstar" or adapter == "base" else adapter,
            "messages": [{"role": "user", "content": content}], "temperature": 0.0, "stream": True,
            "max_tokens": max_tokens, "ignore_eos": True}
    if engine == "vllm":
        # its own count: the special tokens it samples past the reply are dropped from the text
        body["stream_options"] = {"include_usage": True}
    start = time.perf_counter()
    first, text, reported, chunks = None, "", None, 0
    try:
        async with session.post(f"{url}/v1/chat/completions", json=body) as r:
            if r.status != 200:
                return Result(0, 0, 0, None, False, f"{r.status}: {(await r.text())[:200]}")
            async for raw in r.content:
                line = raw.strip()
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    reported = chunk["usage"].get("completion_tokens") or reported
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {}).get("content") or ""
                    if delta:
                        if first is None:
                            first = time.perf_counter()
                        text += delta
                        chunks += 1
    except Exception as e:  # noqa: BLE001
        return Result(0, 0, 0, None, False, repr(e))
    end = time.perf_counter()
    tokens = reported if reported is not None else chunks
    first = first or end
    itl = (end - first) / (tokens - 1) if tokens > 1 else None
    return Result(first - start, end - start, tokens, itl, True, chunks=chunks, retokenized=len(tok.encode(text)))


async def run(args) -> Summary:
    from transformers import GPT2TokenizerFast

    snapshot = glob.glob(f"{HUB}/models--microsoft--Phi-4-multimodal-instruct/snapshots/*/")[0]
    tok = GPT2TokenizerFast.from_pretrained(snapshot)
    work = build(args.workload, args.requests + args.warmup, args.output_len, args.offset)
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        async def drain(items):
            queue = list(items)
            results: list[Result] = []

            async def worker():
                while queue:
                    adapter, content, max_tokens = queue.pop(0)
                    results.append(await one(
                        session, args.url, args.engine, args.model, adapter, content, max_tokens, tok,
                    ))
            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
            return results

        await drain(work[: args.warmup])
        start = time.perf_counter()
        results = await drain(work[args.warmup:])
        wall = time.perf_counter() - start
    ok = [r for r in results if r.ok]
    errors = sorted({r.error for r in results if not r.ok})
    if errors:
        print("errors:", errors[:3])
    return Summary(
        engine=args.engine, workload=args.workload, concurrency=args.concurrency, requests=len(results),
        failed=len(results) - len(ok), wall_s=round(wall, 2),
        output_tok_s=round(sum(r.tokens for r in ok) / wall, 1), req_s=round(len(ok) / wall, 3),
        ttft_ms=_stats([r.ttft for r in ok], 1000), itl_ms=_stats([r.itl for r in ok if r.itl], 1000),
        latency_s=_stats([r.latency for r in ok]), mean_output_tokens=round(statistics.fmean(r.tokens for r in ok), 1),
        mean_chunks=round(statistics.fmean(r.chunks for r in ok), 1),
        mean_retokenized=round(statistics.fmean(r.retokenized for r in ok), 1),
        short_output=statistics.fmean(r.tokens for r in ok) < args.output_len,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--engine", choices=["mstar", "vllm"], required=True)
    p.add_argument("--model", default="phi4mm", help="the served base model's name")
    p.add_argument("--workload", choices=["text", "image", "asr", "mixed"], required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--warmup", type=int, default=16)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--offset", type=int, default=0, help="the first image / clip; give each run its own range")
    p.add_argument("--out")
    args = p.parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(asdict(summary)))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(asdict(summary)) + "\n")


if __name__ == "__main__":
    main()
