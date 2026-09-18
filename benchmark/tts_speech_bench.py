#!/usr/bin/env python3
"""Streaming ``/v1/audio/speech`` benchmark shared by M*, vLLM-Omni and SGLang-Omni.

One client, one metric definition, every engine (BENCHMARK_PROTOCOL.md, TTS row):

* time-to-first-audio (TTFA): request start to the first PCM byte (WAV header
  excluded), p50 / p95 over the requests of a repeat;
* end-to-end latency and RTF = wall time / seconds of audio produced;
* audio-seconds generated per wall-clock second at the given concurrency;
* the PCM of every request can be written out for a WER check
  (``benchmark/tts_wer.py``).

Requests are closed-loop: ``--concurrency`` requests in flight at all times until
every sentence of the input file has been synthesized once per repeat. Warmup
requests are excluded. Repeats are reported individually and as the median.

Engine specifics are limited to how audio is streamed:

* ``mstar``:       ``response_format=wav``, one open-ended WAV (44-byte header, then PCM16);
* ``vllm-omni``:   ``response_format=pcm`` + ``stream_format=audio`` -> raw PCM16;
* ``sglang-omni``: ``response_format=pcm`` -> raw PCM16.

Example (same node, back to back, 200 sentences, 3 repeats)::

    python -m benchmark.tts_speech_bench --engine mstar --url http://127.0.0.1:8000 \\
        --model qwen3_tts_1p7b --sentences $BENCH/tts/sentences_200.txt \\
        --voice vivian --language English --concurrency 8 --repeats 3 \\
        --out results/mstar_c8.json --save-audio-dir results/mstar_c8_wav
    python -m benchmark.tts_speech_bench --engine vllm-omni --url http://127.0.0.1:8002 \\
        --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

WAV_HEADER_BYTES = 44
BYTES_PER_SAMPLE = 2  # PCM16 mono


@dataclass
class RequestResult:
    sentence_id: int
    words: int
    ttfa_s: float | None
    e2e_s: float | None
    audio_s: float
    pcm_bytes: int
    error: str | None = None

    @property
    def rtf(self) -> float | None:
        if self.e2e_s is None or self.audio_s <= 0:
            return None
        return self.e2e_s / self.audio_s


@dataclass
class RepeatSummary:
    repeat: int
    requests: int
    errors: int
    wall_s: float
    ttfa_p50_ms: float
    ttfa_p95_ms: float
    e2e_p50_s: float
    e2e_p95_s: float
    rtf_mean: float
    rtf_p95: float
    audio_s_total: float
    audio_s_per_wall_s: float
    results: list[dict[str, Any]] = field(default_factory=list)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return ordered[idx]


def _payload(args: argparse.Namespace, text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": args.model, "input": text, "stream": True}
    if args.engine == "mstar":
        payload["response_format"] = "wav"
    else:
        payload["response_format"] = "pcm"
        if args.engine == "vllm-omni":
            payload["stream_format"] = "audio"
    if args.voice:
        payload["voice"] = args.voice
    if args.language:
        payload["language"] = args.language
    if args.instructions:
        payload["instructions"] = args.instructions
    if args.seed is not None:
        payload["seed"] = args.seed
    for item in args.extra:
        key, _, raw = item.partition("=")
        try:
            payload[key] = json.loads(raw)
        except json.JSONDecodeError:
            payload[key] = raw
    return payload


def _write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1, sample_rate,
        sample_rate * BYTES_PER_SAMPLE, BYTES_PER_SAMPLE, 8 * BYTES_PER_SAMPLE,
        b"data", len(pcm),
    )
    path.write_bytes(header + pcm)


async def _one_request(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    sentence_id: int,
    text: str,
    save_dir: Path | None,
) -> RequestResult:
    payload = _payload(args, text)
    skip = WAV_HEADER_BYTES if args.engine == "mstar" else 0
    pcm = bytearray()
    ttfa = None
    start = time.perf_counter()
    try:
        async with session.post(
            f"{args.url}/v1/audio/speech",
            json=payload,
            headers={"Authorization": "Bearer EMPTY"},
            timeout=aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(sentence_id, len(text.split()), None, None, 0.0, 0,
                                     error=f"HTTP {resp.status}: {body[:200]}")
            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue
                if skip:
                    drop = min(skip, len(chunk))
                    chunk = chunk[drop:]
                    skip -= drop
                    if not chunk:
                        continue
                if ttfa is None:
                    ttfa = time.perf_counter() - start
                pcm.extend(chunk)
    except Exception as exc:  # noqa: BLE001 - every failure is a benchmark error row
        return RequestResult(sentence_id, len(text.split()), None, None, 0.0, 0, error=str(exc)[:200])
    e2e = time.perf_counter() - start
    if not pcm:
        return RequestResult(sentence_id, len(text.split()), ttfa, e2e, 0.0, 0, error="empty audio")
    if save_dir is not None:
        _write_wav(save_dir / f"{sentence_id:04d}.wav", bytes(pcm), args.sample_rate)
    audio_s = len(pcm) / (BYTES_PER_SAMPLE * args.sample_rate)
    return RequestResult(sentence_id, len(text.split()), ttfa, e2e, audio_s, len(pcm))


async def _run_repeat(
    args: argparse.Namespace, sentences: list[tuple[int, str]], repeat: int, save_dir: Path | None,
) -> RepeatSummary:
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=0)

    async def limited(session, sid, text):
        async with semaphore:
            return await _one_request(session, args, sid, text, save_dir)

    async with aiohttp.ClientSession(connector=connector) as session:
        wall_start = time.perf_counter()
        results = await asyncio.gather(*(limited(session, sid, text) for sid, text in sentences))
        wall = time.perf_counter() - wall_start

    ok = [r for r in results if r.error is None]
    ttfa = [r.ttfa_s for r in ok if r.ttfa_s is not None]
    e2e = [r.e2e_s for r in ok if r.e2e_s is not None]
    rtf = [r.rtf for r in ok if r.rtf is not None]
    audio_total = sum(r.audio_s for r in ok)
    return RepeatSummary(
        repeat=repeat,
        requests=len(results),
        errors=len(results) - len(ok),
        wall_s=wall,
        ttfa_p50_ms=1000 * _percentile(ttfa, 0.50),
        ttfa_p95_ms=1000 * _percentile(ttfa, 0.95),
        e2e_p50_s=_percentile(e2e, 0.50),
        e2e_p95_s=_percentile(e2e, 0.95),
        rtf_mean=statistics.fmean(rtf) if rtf else float("nan"),
        rtf_p95=_percentile(rtf, 0.95),
        audio_s_total=audio_total,
        audio_s_per_wall_s=audio_total / wall if wall > 0 else float("nan"),
        results=[asdict(r) | {"rtf": r.rtf} for r in results],
    )


def _gpu_info() -> dict[str, str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip().splitlines()
    except Exception as exc:  # noqa: BLE001 - informational only
        return {"error": str(exc)}
    return {"gpus": out}


def _median_summary(repeats: list[RepeatSummary]) -> dict[str, float]:
    keys = ("ttfa_p50_ms", "ttfa_p95_ms", "e2e_p50_s", "e2e_p95_s", "rtf_mean", "rtf_p95",
            "audio_s_per_wall_s", "wall_s")
    return {key: statistics.median(getattr(r, key) for r in repeats) for key in keys}


def _markdown_row(args: argparse.Namespace, med: dict[str, float], errors: int) -> str:
    return (
        f"| {args.label or args.engine} | c={args.concurrency} | "
        f"TTFA p50 {med['ttfa_p50_ms']:.0f} ms / p95 {med['ttfa_p95_ms']:.0f} ms | "
        f"RTF {med['rtf_mean']:.3f} | {med['audio_s_per_wall_s']:.1f} audio-s/s | "
        f"e2e p50 {med['e2e_p50_s']:.2f} s | errors {errors} |"
    )


async def _main_async(args: argparse.Namespace) -> None:
    lines = [ln.strip() for ln in Path(args.sentences).read_text(encoding="utf-8").splitlines()]
    sentences = [(i + 1, ln) for i, ln in enumerate(lines) if ln]
    if args.num_sentences:
        sentences = sentences[: args.num_sentences]
    if not sentences:
        sys.exit("no sentences to synthesize")

    save_dir = Path(args.save_audio_dir) if args.save_audio_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.warmup:
        warm = [sentences[i % len(sentences)] for i in range(args.warmup)]
        print(f"warmup: {len(warm)} requests", file=sys.stderr)
        await _run_repeat(args, warm, repeat=-1, save_dir=None)

    repeats: list[RepeatSummary] = []
    for rep in range(args.repeats):
        # Only the last repeat keeps audio: identical inputs, and one copy is
        # all the WER check needs.
        summary = await _run_repeat(
            args, sentences, rep, save_dir if rep == args.repeats - 1 else None,
        )
        repeats.append(summary)
        print(
            f"repeat {rep}: {summary.requests} req, {summary.errors} errors, "
            f"TTFA p50 {summary.ttfa_p50_ms:.0f} ms p95 {summary.ttfa_p95_ms:.0f} ms, "
            f"RTF {summary.rtf_mean:.3f}, {summary.audio_s_per_wall_s:.1f} audio-s/s, "
            f"wall {summary.wall_s:.1f} s",
            file=sys.stderr,
        )

    med = _median_summary(repeats)
    errors = sum(r.errors for r in repeats)
    report = {
        "engine": args.engine,
        "label": args.label,
        "url": args.url,
        "model": args.model,
        "engine_version": args.engine_version,
        "sentences_file": str(Path(args.sentences).resolve()),
        "num_sentences": len(sentences),
        "concurrency": args.concurrency,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "request_fields": {k: v for k, v in _payload(args, "").items() if k != "input"},
        "sample_rate": args.sample_rate,
        "gpu": _gpu_info(),
        "started_at": args.started_at,
        "median_over_repeats": med,
        "errors_total": errors,
        "repeats_detail": [asdict(r) for r in repeats],
        "markdown_row": _markdown_row(args, med, errors),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["markdown_row"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=("mstar", "vllm-omni", "sglang-omni"), required=True)
    parser.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="model name sent in the request")
    parser.add_argument("--sentences", required=True, help="text file, one sentence per line")
    parser.add_argument("--num-sentences", type=int, default=0, help="use only the first N sentences")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--instructions", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--extra", action="append", default=[], metavar="KEY=JSON",
                        help="extra request field, e.g. --extra non_streaming_mode=false")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--label", default=None, help="row label for the markdown table")
    parser.add_argument("--engine-version", default=None, help="recorded verbatim in the report")
    parser.add_argument("--out", default=None, help="JSON report path")
    parser.add_argument("--save-audio-dir", default=None, help="write <id>.wav per request (last repeat)")
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.repeats < 1:
        parser.error("concurrency and repeats must be positive")
    args.started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return args


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_main_async(parse_args(argv)))


if __name__ == "__main__":
    main()
