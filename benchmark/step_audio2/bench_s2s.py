#!/usr/bin/env python3
"""Step-Audio-2-mini speech-to-speech load test, one client for M*, VoxServe and vllm-omni.

Each request sends one spoken question (a clip from Step-Audio2's assets) and streams back a
spoken reply. Per request: time to first audio (TTFA), end-to-end time, audio seconds
received. Per run: request and audio throughput, TTFA and end-to-end percentiles, and the
real-time factor each stream saw (audio seconds per wall second after its first audio).

Usage:
  bench_s2s.py --backend {mstar,voxserve,omni} --port N --assets DIR --concurrency 1 8 32 \
      --requests-per-level 32 --max-tokens 1024 --out results.jsonl

``--assets`` is the ``assets/`` directory of a stepfun-ai/Step-Audio2 checkout, which holds the four clips.

Sampling is the reference's (temperature 0.7, top-p 0.9, repetition penalty 1.05) where the
backend takes it per request. VoxServe takes sampling at launch: start it with
``--temperature 0.7 --top_p 0.9``.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import aiohttp

CLIPS = [
    "give_me_a_brief_introduction_to_the_great_wall.wav",
    "帮我查一下今天上证指数的开盘价是多少.wav",
    "multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
    "multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
]
SAMPLE_RATE = 24000
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.05}


@dataclass
class Result:
    ok: bool
    ttfa: float | None
    e2e: float
    audio_s: float
    error: str = ""


def pcm_seconds(nbytes: int) -> float:
    return nbytes / 2 / SAMPLE_RATE


async def mstar(session, base, clip: Path, max_tokens: int, seed: int) -> Result:
    form = aiohttp.FormData()
    form.add_field("output_modalities", "text,audio")
    form.add_field("input_modalities", "audio")
    form.add_field("streaming", "true")
    form.add_field("model_kwargs", json.dumps({"max_output_tokens": max_tokens, "seed": seed, **SAMPLING}))
    form.add_field("files", clip.read_bytes(), filename=f"q{CLIPS.index(clip.name)}.wav")
    t0 = time.perf_counter()
    ttfa, audio = None, 0
    async with session.post(f"{base}/generate", data=form) as r:
        if r.status != 200:
            return Result(False, None, time.perf_counter() - t0, 0, f"{r.status} {await r.text()}"[:300])
        async for line in r.content:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("modality") == "audio":
                n = len(base64.b64decode(chunk["data"]))
                if n and ttfa is None:
                    ttfa = time.perf_counter() - t0
                audio += n
    return Result(audio > 0, ttfa, time.perf_counter() - t0, pcm_seconds(audio))


async def voxserve(session, base, clip: Path, max_tokens: int, seed: int) -> Result:
    form = aiohttp.FormData()
    form.add_field("text", "")
    form.add_field("streaming", "true")
    form.add_field("audio", clip.read_bytes(), filename=f"q{CLIPS.index(clip.name)}.wav")
    t0 = time.perf_counter()
    ttfa, audio, header = None, 0, 44
    async with session.post(f"{base}/generate", data=form) as r:
        if r.status != 200:
            return Result(False, None, time.perf_counter() - t0, 0, f"{r.status} {await r.text()}"[:300])
        async for data in r.content.iter_any():
            skip = min(header, len(data))
            header -= skip
            n = len(data) - skip
            if n and ttfa is None:
                ttfa = time.perf_counter() - t0
            audio += n
    return Result(audio > 0, ttfa, time.perf_counter() - t0, pcm_seconds(audio))


def wav_seconds(b64: str) -> float:
    raw = base64.b64decode(b64)
    try:
        with wave.open(io.BytesIO(raw)) as w:
            return w.getnframes() / w.getframerate()
    except wave.Error:
        return pcm_seconds(len(raw))


async def omni(session, base, clip: Path, max_tokens: int, seed: int) -> Result:
    audio_b64 = base64.b64encode(clip.read_bytes()).decode()
    body = {
        "model": "stepfun-ai/Step-Audio-2-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}]},
            # A spoken reply opens with <tts_start>, as the reference's examples prefill it; M* and VoxServe add it
            # themselves, vllm-omni's chat endpoint takes it as a continued assistant turn.
            {"role": "assistant", "content": "<tts_start>"},
        ],
        "add_generation_prompt": False, "continue_final_message": True,
        "modalities": ["text", "audio"], "stream": True, "max_tokens": max_tokens, "seed": seed, **SAMPLING,
    }
    t0 = time.perf_counter()
    ttfa, audio = None, 0.0
    async with session.post(f"{base}/v1/chat/completions", json=body) as r:
        if r.status != 200:
            return Result(False, None, time.perf_counter() - t0, 0, f"{r.status} {await r.text()}"[:300])
        async for line in r.content:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            for choice in chunk.get("choices", []):
                if choice.get("audio_metadata") is None or not (choice.get("delta") or {}).get("content"):
                    continue
                s = wav_seconds(choice["delta"]["content"])
                if s and ttfa is None:
                    ttfa = time.perf_counter() - t0
                audio += s
    return Result(audio > 0, ttfa, time.perf_counter() - t0, audio)


BACKENDS = {"mstar": mstar, "voxserve": voxserve, "omni": omni}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


async def level(backend, base, assets, concurrency, n, max_tokens, seed0) -> dict:
    fn = BACKENDS[backend]
    clips = [assets / c for c in CLIPS]
    queue = list(range(n))
    results: list[Result] = []
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def worker():
            while queue:
                i = queue.pop(0)
                try:
                    results.append(await fn(session, base, clips[i % len(clips)], max_tokens, seed0 + i))
                except Exception as e:  # noqa: BLE001 - a failed request is a result
                    results.append(Result(False, None, 0.0, 0.0, repr(e)[:300]))
        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t0
    ok = [r for r in results if r.ok]
    ttfa = [r.ttfa for r in ok if r.ttfa is not None]
    rtf = [(r.e2e - r.ttfa) / r.audio_s for r in ok if r.ttfa is not None and r.audio_s > 0]
    return {
        "backend": backend, "concurrency": concurrency, "requests": n, "ok": len(ok), "wall_s": wall,
        "req_per_s": len(ok) / wall, "audio_s_per_s": sum(r.audio_s for r in ok) / wall,
        "mean_audio_s": statistics.mean(r.audio_s for r in ok) if ok else None,
        "ttfa_p50": pct(ttfa, 0.5), "ttfa_p90": pct(ttfa, 0.9),
        "e2e_p50": pct([r.e2e for r in ok], 0.5), "e2e_p90": pct([r.e2e for r in ok], 0.9),
        "stream_rtf_p50": pct(rtf, 0.5), "stream_rtf_p90": pct(rtf, 0.9),
        "errors": [r.error for r in results if not r.ok][:3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--requests-per-level", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base = f"http://localhost:{args.port}"
    asyncio.run(level(args.backend, base, args.assets, min(4, args.warmup), args.warmup, args.max_tokens, 10_000))
    with args.out.open("a") as f:
        for c in args.concurrency:
            n = max(args.requests_per_level, c)
            row = asyncio.run(level(args.backend, base, args.assets, c, n, args.max_tokens, 0))
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
