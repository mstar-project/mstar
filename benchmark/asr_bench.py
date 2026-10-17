"""ASR benchmark per the workspace protocol: RTFx, latency and WER vs. any
OpenAI-compatible transcription server.

One client measures every engine (mstar, vLLM, TensorRT-LLM, SGLang-Omni,
faster-whisper) through ``POST /v1/audio/transcriptions`` on the same files,
built once by ``benchmark/asr_data.py``. For each concurrency level the whole
set is sent closed-loop (at most ``c`` requests in flight), ``--repeats``
times after a warmup pass, and the run reports:

    RTFx          audio seconds transcribed per wall-clock second (median over repeats)
    latency       per-request end-to-end p50 / p95 (ms) of the median run
    WER           on the transcripts of the last run, whisper-normalized
    long-form     wall time and RTFx of the ~10 min file at concurrency 1

Results go to ``--output-json`` with the engine label, git SHA and GPU
clocks, and a markdown row for the PR table is printed.

    python -m benchmark.asr_bench --url http://localhost:8000 --model whisper_large_v3_turbo \\
        --system "M* <sha>" --data-dir $MSTAR_WS_ROOT/commons/bench/data/asr \\
        --concurrency 1 8 32 --long-form --output-json results/asr_mstar.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp


@dataclass
class Utterance:
    uid: str
    path: Path
    duration: float
    reference: str
    audio: bytes = field(default_factory=bytes, repr=False)


def load_set(directory: Path) -> list[Utterance]:
    refs = {}
    for line in (directory / "references.tsv").read_text().splitlines():
        if line.strip():
            uid, text = line.split("\t", 1)
            refs[uid] = text
    items: list[Utterance] = []
    for uid, text in refs.items():
        path = directory / f"{uid}.wav"
        with wave.open(str(path), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        items.append(Utterance(uid, path, duration, text, path.read_bytes()))
    return items


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------


async def transcribe(
    session: aiohttp.ClientSession, base_url: str, model: str, utt: Utterance,
    language: str | None, extra: dict[str, str],
) -> tuple[str | None, float, str | None]:
    """One ``/v1/audio/transcriptions`` call: (text, seconds, error)."""
    form = aiohttp.FormData()
    form.add_field("model", model)
    form.add_field("response_format", "json")
    form.add_field("temperature", "0")
    if language:
        form.add_field("language", language)
    for key, value in extra.items():
        form.add_field(key, value)
    form.add_field("file", utt.audio, filename=utt.path.name, content_type="audio/wav")
    t0 = time.perf_counter()
    try:
        async with session.post(f"{base_url}/v1/audio/transcriptions", data=form) as resp:
            body = await resp.read()
            if resp.status != 200:
                return None, time.perf_counter() - t0, f"HTTP {resp.status}: {body[:200]!r}"
            text = json.loads(body).get("text", "")
            return text, time.perf_counter() - t0, None
    except Exception as exc:  # noqa: BLE001 — recorded per request
        return None, time.perf_counter() - t0, repr(exc)


async def run_set(
    base_url: str, model: str, items: list[Utterance], concurrency: int,
    language: str | None, extra: dict[str, str], timeout_s: float,
) -> dict:
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, tuple[str | None, float, str | None]] = {}

    async def one(session, utt):
        async with sem:
            results[utt.uid] = await transcribe(session, base_url, model, utt, language, extra)

    connector = aiohttp.TCPConnector(limit=max(64, concurrency + 8))
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout_s), connector=connector,
    ) as session:
        t0 = time.perf_counter()
        await asyncio.gather(*(one(session, utt) for utt in items))
        wall = time.perf_counter() - t0

    latencies = sorted(r[1] for r in results.values())
    errors = {uid: r[2] for uid, r in results.items() if r[2]}
    audio_seconds = sum(u.duration for u in items if u.uid not in errors)
    return {
        "wall_s": wall,
        "audio_s": audio_seconds,
        "rtfx": audio_seconds / wall if wall > 0 else 0.0,
        "p50_ms": 1000 * statistics.median(latencies) if latencies else None,
        "p95_ms": 1000 * _percentile(latencies, 0.95) if latencies else None,
        "errors": errors,
        "hypotheses": {uid: (r[0] or "") for uid, r in results.items()},
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


# --------------------------------------------------------------------------
# WER
# --------------------------------------------------------------------------


def make_normalizer(kind: str):
    """``whisper``: the EnglishTextNormalizer every Whisper WER in the
    literature uses (numbers, spellings, punctuation); ``basic``: uppercase,
    strip punctuation (``benchmark/asr_eval.py``'s choice)."""
    if kind == "whisper":
        try:
            from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

            spelling = {}
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download("openai/whisper-large-v3-turbo", "normalizer.json", local_files_only=True)
                spelling = json.loads(Path(path).read_text())
            except Exception:  # noqa: BLE001 — spelling map is optional
                pass
            return EnglishTextNormalizer(spelling)
        except ImportError:
            print("transformers unavailable; falling back to the basic normalizer", file=sys.stderr)
    import re

    def basic(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", s.upper())).strip()

    return basic


def word_error_rate(references: list[str], hypotheses: list[str], normalize) -> float:
    import jiwer

    refs = [normalize(r) for r in references]
    hyps = [normalize(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps, strict=True) if r]
    return jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])


# --------------------------------------------------------------------------
# environment record
# --------------------------------------------------------------------------


def _sh(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def environment_record() -> dict:
    return {
        "git_sha": _sh(["git", "rev-parse", "--short", "HEAD"]),
        "hostname": _sh(["hostname"]),
        "gpu": _sh(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.sm,memory.total",
                    "--format=csv,noheader"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASR protocol benchmark over /v1/audio/transcriptions")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True, help="model id sent in the request")
    p.add_argument("--system", required=True, help='engine label for the table, e.g. "vLLM 0.29.0"')
    p.add_argument("--data-dir", default=os.path.join(os.environ.get("MSTAR_WS_ROOT", "."), "commons/bench/data/asr"))
    p.add_argument("--short-set", default="test_clean_200")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=8, help="requests sent (and discarded) before timing")
    p.add_argument("--language", default="en", help="ISO-639-1 code; '' to let the engine detect")
    p.add_argument("--extra", action="append", default=[], help="extra form field key=value (repeatable)")
    p.add_argument("--long-form", action="store_true", help="also time the long_form file at concurrency 1")
    p.add_argument("--normalizer", choices=["whisper", "basic"], default="whisper")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--output-json", default=None)
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    items = load_set(data_dir / args.short_set)
    language = args.language or None
    extra = dict(kv.split("=", 1) for kv in args.extra)
    normalize = make_normalizer(args.normalizer)
    record: dict = {
        "system": args.system, "model": args.model, "url": args.url, "data_dir": str(data_dir),
        "num_utterances": len(items), "audio_seconds": sum(u.duration for u in items),
        "language": language, "normalizer": args.normalizer, "env": environment_record(),
        "concurrency": {},
    }

    if args.warmup:
        warm = (items * ((args.warmup // len(items)) + 1))[:args.warmup]
        await run_set(args.url, args.model, warm, min(8, max(args.concurrency)), language, extra, args.timeout)

    for c in args.concurrency:
        runs = []
        for _ in range(args.repeats):
            runs.append(await run_set(args.url, args.model, items, c, language, extra, args.timeout))
        rtfx = sorted(r["rtfx"] for r in runs)
        median_run = runs[[r["rtfx"] for r in runs].index(rtfx[len(rtfx) // 2])]
        last = runs[-1]
        wer = word_error_rate(
            [u.reference for u in items if u.uid not in last["errors"]],
            [last["hypotheses"][u.uid] for u in items if u.uid not in last["errors"]],
            normalize,
        ) if len(last["errors"]) < len(items) else None
        record["concurrency"][str(c)] = {
            "rtfx_median": rtfx[len(rtfx) // 2],
            "rtfx_runs": rtfx,
            "wall_s_median": median_run["wall_s"],
            "p50_ms": median_run["p50_ms"],
            "p95_ms": median_run["p95_ms"],
            "wer": wer,
            "errors": len(last["errors"]),
            "error_samples": list(last["errors"].values())[:3],
        }
        print(f"[c={c:>3}] RTFx {rtfx[len(rtfx) // 2]:8.1f}  p50 {median_run['p50_ms']:8.1f} ms  "
              f"p95 {median_run['p95_ms']:8.1f} ms  WER {100 * wer if wer is not None else float('nan'):5.2f}%  "
              f"errors {len(last['errors'])}", flush=True)
        if c == args.concurrency[-1]:
            record["hypotheses"] = last["hypotheses"]

    if args.long_form:
        long_items = load_set(data_dir / "long_form")
        runs = [await run_set(args.url, args.model, long_items, 1, language, extra, args.timeout)
                for _ in range(args.repeats)]
        walls = sorted(r["wall_s"] for r in runs)
        last = runs[-1]
        wer = word_error_rate([u.reference for u in long_items],
                              [last["hypotheses"][u.uid] for u in long_items], normalize) \
            if not last["errors"] else None
        record["long_form"] = {
            "audio_s": last["audio_s"], "wall_s_median": walls[len(walls) // 2],
            "rtfx_median": last["audio_s"] / walls[len(walls) // 2] if walls[len(walls) // 2] else None,
            "wer": wer, "errors": list(last["errors"].values())[:3],
            "hypothesis": last["hypotheses"],
        }
        print(f"[long-form] {last['audio_s']:.0f} s audio in {walls[len(walls) // 2]:.2f} s "
              f"(RTFx {record['long_form']['rtfx_median'] or float('nan'):.1f})  WER "
              f"{100 * wer if wer is not None else float('nan'):.2f}%", flush=True)

    cells = [args.system]
    for c in args.concurrency:
        r = record["concurrency"][str(c)]
        cells.append(f"{r['rtfx_median']:.0f}")
    cells.append(f"{100 * record['concurrency'][str(args.concurrency[0])]['wer']:.2f}%"
                 if record["concurrency"][str(args.concurrency[0])]["wer"] is not None else "n/a")
    cells.append(f"{record['long_form']['rtfx_median']:.0f}" if record.get("long_form") and
                 record["long_form"]["rtfx_median"] else "n/a")
    print("| " + " | ".join(cells) + " |  <- system | RTFx@" +
          " | RTFx@".join(map(str, args.concurrency)) + " | WER | long-form RTFx")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        Path(args.output_json).write_text(json.dumps(record, indent=2))
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
