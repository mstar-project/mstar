#!/usr/bin/env python3
"""Closed-loop load against a running ``mstar serve step_audio2``, for worker-phase runs.

    python benchmark/step_audio2/load.py --port 8100 --assets <Step-Audio2 checkout>/assets \\
        --workload s2s --concurrency 8 --requests 40

Workloads (``POST /generate``, streamed):

* ``t2t``: a text question, a text reply of exactly ``--text-tokens`` (``ignore_eos``).
* ``s2t``: a spoken question (the reference repo's clips, in turn), a text reply of exactly
  ``--text-tokens``.
* ``s2s``: a spoken question, a spoken reply, sampled with a per-request seed and ended by
  the model, so its length varies: compare steps/s and audio s/s, not req/s.
* ``t2s``: a text question, a spoken reply, as ``s2s``.

``--unique-lengths`` trims each request's clip to its own length (seeded by the request), as
real traffic's clips are: a shape-keyed cost (a kernel plan per input length) then lands on
every request rather than once per clip.

A streamed text chunk is one LLM step (a speech id's is empty), so steps/s is the LLM's
decode rate as the client sees it. Prints one JSON line of aggregates; ``--out`` keeps the
per-request records.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path

import aiohttp

CLIPS = [
    "give_me_a_brief_introduction_to_the_great_wall.wav",
    "帮我查一下今天上证指数的开盘价是多少.wav",
    "multi-turn-round1-听说荡口古镇从下个月开始取消门票了，你知道这事吗。.wav",
    "multi-turn-round2-新闻说九月十九号就免费开放了。好像整个古镇都升级改造了，现在变成开放式街区了。.wav",
]
QUESTIONS = [
    "Give me a brief introduction to the Great Wall.",
    "Explain how a rainbow forms.",
    "What should I pack for a weekend hiking trip?",
    "Describe the water cycle in a few sentences.",
]
SAMPLE_RATE = 24000


def trimmed(clip: bytes, i: int) -> bytes:
    """``clip`` cut to a length of its own for request ``i``: between 1.5 s and all of it."""
    import io
    import random

    import soundfile as sf

    data, rate = sf.read(io.BytesIO(clip), dtype="float32")
    keep = random.Random(i).randint(int(1.5 * rate), len(data))
    out = io.BytesIO()
    sf.write(out, data[:keep], rate, format="WAV")
    return out.getvalue()


async def one(session: aiohttp.ClientSession, url: str, i: int, args, clips: list[bytes]) -> dict:
    spoken = args.workload in ("s2s", "t2s")
    audio_in = args.workload in ("s2t", "s2s")
    kwargs: dict = {"seed": i}
    if spoken:
        kwargs["max_output_tokens"] = args.speech_tokens
    else:
        kwargs.update(max_output_tokens=args.text_tokens, ignore_eos=True)
    form = aiohttp.FormData()
    form.add_field("output_modalities", "text,audio" if spoken else "text")
    form.add_field("input_modalities", "audio" if audio_in else "text")
    form.add_field("streaming", "true")
    form.add_field("model_kwargs", json.dumps(kwargs))
    if audio_in:
        k = i % len(clips)
        clip = trimmed(clips[k], i) if args.unique_lengths else clips[k]
        form.add_field("files", clip, filename=f"clip{k}.wav")
    else:
        form.add_field("text", QUESTIONS[i % len(QUESTIONS)])
    t0 = time.perf_counter()
    ttft = ttfa = None
    steps = audio_bytes = 0
    error = None
    async with session.post(url, data=form) as r:
        if r.status != 200:
            return {"i": i, "error": f"{r.status}: {await r.text()}"}
        buf = b""
        async for piece in r.content.iter_any():
            buf += piece
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if not line.strip():
                    continue
                c = json.loads(line)
                now = time.perf_counter() - t0
                if c.get("modality") == "text":
                    steps += 1
                    ttft = now if ttft is None else ttft
                elif c.get("modality") == "audio":
                    audio_bytes += len(base64.b64decode(c["data"]))
                    ttfa = now if ttfa is None else ttfa
                else:
                    error = c
    return {
        "i": i, "jct": time.perf_counter() - t0, "ttft": ttft, "ttfa": ttfa, "steps": steps,
        "audio_s": audio_bytes / 2 / SAMPLE_RATE, "error": error,
    }


async def run(args) -> dict:
    url = f"http://127.0.0.1:{args.port}/generate"
    clips = [(args.assets / c).read_bytes() for c in CLIPS]
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def bounded(i):
            async with sem:
                return await one(session, url, i, args, clips)

        if args.warmup:
            await asyncio.gather(*(bounded(10_000 + i) for i in range(args.warmup)))
        t0 = time.perf_counter()
        recs = await asyncio.gather(*(bounded(i) for i in range(args.requests)))
        wall = time.perf_counter() - t0
    ok = [r for r in recs if not r.get("error")]

    def p(xs, q):
        xs = sorted(x for x in xs if x is not None)
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4) if xs else None

    out = {
        "workload": args.workload, "concurrency": args.concurrency, "requests": args.requests,
        "failed": len(recs) - len(ok), "wall_s": round(wall, 3),
        "steps_per_s": round(sum(r["steps"] for r in ok) / wall, 1),
        "audio_s_per_s": round(sum(r["audio_s"] for r in ok) / wall, 3),
        "steps_mean": round(statistics.mean(r["steps"] for r in ok), 1) if ok else None,
        "audio_s_total": round(sum(r["audio_s"] for r in ok), 2),
        "ttft_p50": p([r["ttft"] for r in ok], 0.5), "ttft_p90": p([r["ttft"] for r in ok], 0.9),
        "ttfa_p50": p([r["ttfa"] for r in ok], 0.5), "ttfa_p90": p([r["ttfa"] for r in ok], 0.9),
        "jct_p50": p([r["jct"] for r in ok], 0.5),
        # per-request decode rate past the first token
        "itl_ms_p50": p([1000 * (r["jct"] - r["ttft"]) / max(1, r["steps"] - 1) for r in ok if r["ttft"]], 0.5),
    }
    if args.out:
        args.out.write_text(json.dumps({"summary": out, "records": recs}, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--assets", type=Path, required=True, help="a stepfun-ai/Step-Audio2 checkout's assets/ directory")
    ap.add_argument("--workload", choices=["t2t", "s2t", "s2s", "t2s"], required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=0, help="requests sent (at the concurrency) before timing")
    ap.add_argument("--text-tokens", type=int, default=256)
    ap.add_argument("--speech-tokens", type=int, default=1024)
    ap.add_argument("--unique-lengths", action="store_true", help="trim each request's clip to its own length")
    ap.add_argument("--out", type=Path, default=None)
    print(json.dumps(asyncio.run(run(ap.parse_args()))), flush=True)


if __name__ == "__main__":
    main()
