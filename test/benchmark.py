#!/usr/bin/env python3
"""Unified benchmark harness for mminf serving.

Supports BAGEL, Qwen3-Omni, and Orpheus. Measures TTFT, ITL, E2E latency,
and throughput across tasks and batch sizes. Output is a JSON file compatible
with the paper's evaluation plan.

Supported tasks per model:
  bagel:       T2T  T2I  I2T  I2I
  qwen3_omni:  T2T  I2T  A2T  V2T  T2A  I2A
  orpheus:     T2A

Usage examples:
  # Single-request baseline, bagel I2T
  python test/benchmark.py --model bagel --task I2T \\
      --image test/bagel/bagel.png --num-requests 10 --output results.json

  # Batch sweep: send 16 concurrent Qwen3-Omni A2T requests
  python test/benchmark.py --model qwen3_omni --task A2T \\
      --audio test/qwen3-omni/audio.wav --batch-size 16 --num-requests 64

  # Orpheus TTS streaming viability
  python test/benchmark.py --model orpheus --task T2A --num-requests 20

  # Mix of tasks (round-robin)
  python test/benchmark.py --model bagel --task mixture \\
      --image test/bagel/bagel.png --num-requests 30
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import aiohttp

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

# (input_modalities, output_modality)
TASK_SPEC: dict[str, tuple[list[str], str]] = {
    "T2T": (["text"], "text"),
    "T2I": (["text"], "image"),
    "T2A": (["text"], "audio"),
    "I2T": (["image", "text"], "text"),
    "I2I": (["image", "text"], "image"),
    "I2A": (["image", "text"], "audio"),
    "A2T": (["audio", "text"], "text"),
    "V2T": (["video", "text"], "text"),
}

MODEL_TASKS: dict[str, list[str]] = {
    "bagel": ["T2T", "T2I", "I2T", "I2I"],
    "qwen3_omni": ["T2T", "I2T", "A2T", "V2T", "T2A", "I2A"],
    "orpheus": ["T2A"],
}

DEFAULT_PROMPTS: dict[str, str] = {
    "T2T": "What is the 7th digit after the decimal point in pi?",
    "T2I": "A golden retriever wearing a graduation cap",
    "T2A": "Hello, welcome to the multimodal inference benchmark.",
    "I2T": "Describe what you see in this image in detail.",
    "I2I": "Transform this image into an oil painting style.",
    "I2A": "Describe what you see and say it aloud.",
    "A2T": "Please transcribe and translate this audio to English.",
    "V2T": "Describe this video in detail.",
}

# ---------------------------------------------------------------------------
# Per-request result
# ---------------------------------------------------------------------------

@dataclass
class TokenEvent:
    """Timestamp of one streaming chunk from the server."""
    t: float          # monotonic time
    modality: str     # "text" | "image" | "audio"
    nbytes: int       # payload size in bytes


@dataclass
class RequestResult:
    request_id: int
    task: str
    status: Literal["success", "failed"]
    error: str = ""

    t_send: float = 0.0          # time request was dispatched
    t_first_chunk: float = 0.0   # time of first non-empty chunk (TTFT proxy)
    t_done: float = 0.0          # time stream closed

    token_events: list[TokenEvent] = field(default_factory=list)

    # Derived metrics (populated by compute_metrics())
    ttft: float | None = None    # seconds to first chunk
    e2e: float = 0.0             # total elapsed seconds
    itl_values: list[float] = field(default_factory=list)   # per-gap ITL (seconds)
    output_tokens: int = 0       # number of text chunks received (proxy for tokens)
    output_bytes: int = 0        # total payload bytes

    def compute_metrics(self) -> None:
        if self.status != "success":
            self.e2e = self.t_done - self.t_send
            return

        self.e2e = self.t_done - self.t_send

        if self.t_first_chunk > 0:
            self.ttft = self.t_first_chunk - self.t_send

        # ITL: gaps between consecutive same-modality chunks
        # For text output, each chunk ≈ 1 token; for audio, chunks are fixed-size frames.
        text_events = [e for e in self.token_events if e.modality == "text"]
        if len(text_events) >= 2:
            self.itl_values = [
                text_events[i].t - text_events[i - 1].t
                for i in range(1, len(text_events))
            ]
            self.output_tokens = len(text_events)

        audio_events = [e for e in self.token_events if e.modality == "audio"]
        if audio_events and not text_events:
            # For audio-only output (T2A/orpheus) use chunk gaps as ITL
            if len(audio_events) >= 2:
                self.itl_values = [
                    audio_events[i].t - audio_events[i - 1].t
                    for i in range(1, len(audio_events))
                ]
            self.output_tokens = len(audio_events)  # chunks, not linguistic tokens

        self.output_bytes = sum(e.nbytes for e in self.token_events)


# ---------------------------------------------------------------------------
# Request builders  (model × task → aiohttp form data / files)
# ---------------------------------------------------------------------------

def _b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _build_form(
    task: str,
    text: str,
    model_kwargs: dict,
    image_path: Path | None,
    audio_path: Path | None,
    video_path: Path | None,
) -> tuple[aiohttp.FormData, list]:
    """Return (FormData, open_file_handles).

    Callers must close the file handles after the request completes.
    """
    in_mods, out_mod = TASK_SPEC[task]
    form = aiohttp.FormData()
    form.add_field("text", text)

    if out_mod != "text":
        form.add_field("output_modalities", out_mod)

    if model_kwargs:
        form.add_field("model_kwargs", json.dumps(model_kwargs))

    handles = []
    for mod in in_mods:
        if mod == "image" and image_path is not None:
            fh = open(image_path, "rb")
            handles.append(fh)
            form.add_field("files", fh, filename=image_path.name, content_type="application/octet-stream")
        elif mod == "audio" and audio_path is not None:
            fh = open(audio_path, "rb")
            handles.append(fh)
            form.add_field("files", fh, filename=audio_path.name, content_type="application/octet-stream")
        elif mod == "video" and video_path is not None:
            fh = open(video_path, "rb")
            handles.append(fh)
            form.add_field("files", fh, filename=video_path.name, content_type="application/octet-stream")

    return form, handles


# ---------------------------------------------------------------------------
# Core async send-and-measure
# ---------------------------------------------------------------------------

async def send_one(
    session: aiohttp.ClientSession,
    url: str,
    request_id: int,
    task: str,
    text: str,
    model_kwargs: dict,
    image_path: Path | None,
    audio_path: Path | None,
    video_path: Path | None,
) -> RequestResult:
    result = RequestResult(request_id=request_id, task=task, status="success")
    result.t_send = time.monotonic()

    form, handles = _build_form(task, text, model_kwargs, image_path, audio_path, video_path)
    try:
        async with session.post(url, data=form) as resp:
            resp.raise_for_status()
            async for raw_line in resp.content:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                data_b64 = msg.get("data", "")
                if not data_b64:
                    continue

                modality = msg.get("modality", "text")
                nbytes = len(base64.b64decode(data_b64))
                now = time.monotonic()

                if result.t_first_chunk == 0.0:
                    result.t_first_chunk = now

                result.token_events.append(TokenEvent(t=now, modality=modality, nbytes=nbytes))

    except Exception as exc:
        result.status = "failed"
        result.error = str(exc)
    finally:
        for fh in handles:
            fh.close()

    result.t_done = time.monotonic()
    result.compute_metrics()
    return result


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _pct(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (k - lo) * (s[hi] - s[lo])


def _safe_mean(data: list[float]) -> float:
    return statistics.mean(data) if data else float("nan")


def _safe_median(data: list[float]) -> float:
    return statistics.median(data) if data else float("nan")


def compute_summary(results: list[RequestResult], wall_time: float, batch_size: int) -> dict:
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "failed"]

    ttft_vals = [r.ttft for r in successful if r.ttft is not None]
    e2e_vals = [r.e2e for r in successful]
    itl_flat = [v for r in successful for v in r.itl_values]
    tok_counts = [r.output_tokens for r in successful if r.output_tokens > 0]

    # Throughput: total output tokens / wall time
    total_tokens = sum(r.output_tokens for r in successful)
    tok_per_sec = total_tokens / wall_time if wall_time > 0 and total_tokens > 0 else float("nan")
    req_per_sec = len(successful) / wall_time if wall_time > 0 else float("nan")

    return {
        "n_total": len(results),
        "n_success": len(successful),
        "n_failed": len(failed),
        "batch_size": batch_size,
        "wall_time_s": round(wall_time, 3),
        "throughput_req_s": round(req_per_sec, 4),
        "throughput_tok_s": round(tok_per_sec, 4),
        "ttft": {
            "mean": round(_safe_mean(ttft_vals), 4),
            "median": round(_safe_median(ttft_vals), 4),
            "p95": round(_pct(ttft_vals, 95), 4),
            "p99": round(_pct(ttft_vals, 99), 4),
            "min": round(min(ttft_vals), 4) if ttft_vals else float("nan"),
            "max": round(max(ttft_vals), 4) if ttft_vals else float("nan"),
        },
        "e2e": {
            "mean": round(_safe_mean(e2e_vals), 4),
            "median": round(_safe_median(e2e_vals), 4),
            "p95": round(_pct(e2e_vals, 95), 4),
            "p99": round(_pct(e2e_vals, 99), 4),
            "min": round(min(e2e_vals), 4) if e2e_vals else float("nan"),
            "max": round(max(e2e_vals), 4) if e2e_vals else float("nan"),
        },
        "itl": {
            "mean": round(_safe_mean(itl_flat), 4),
            "median": round(_safe_median(itl_flat), 4),
            "p95": round(_pct(itl_flat, 95), 4),
            "p99": round(_pct(itl_flat, 99), 4),
            "min": round(min(itl_flat), 4) if itl_flat else float("nan"),
            "max": round(max(itl_flat), 4) if itl_flat else float("nan"),
        },
        "output_tokens": {
            "mean": round(_safe_mean(tok_counts), 2),
            "total": total_tokens,
        },
        "errors": [{"id": r.request_id, "msg": r.error} for r in failed],
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _fmt(v: float, unit: str = "s", decimals: int = 3) -> str:
    if v != v:  # NaN
        return "  N/A  "
    return f"{v:.{decimals}f}{unit}"


def print_summary(summary: dict, model: str, task: str) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {model.upper()}  task={task}  batch={summary['batch_size']}")
    print(sep)
    print(f"  Requests : {summary['n_success']}/{summary['n_total']} succeeded   "
          f"({summary['n_failed']} failed)")
    print(f"  Wall time: {_fmt(summary['wall_time_s'])}   "
          f"Tput: {_fmt(summary['throughput_req_s'], 'req/s', 2)}  "
          f"{_fmt(summary['throughput_tok_s'], 'tok/s', 1)}")
    print(sep)
    t = summary["ttft"]
    print(f"  TTFT (s) : mean={_fmt(t['mean'])}  p50={_fmt(t['median'])}  "
          f"p95={_fmt(t['p95'])}  p99={_fmt(t['p99'])}")
    e = summary["e2e"]
    print(f"  E2E  (s) : mean={_fmt(e['mean'])}  p50={_fmt(e['median'])}  "
          f"p95={_fmt(e['p95'])}  p99={_fmt(e['p99'])}")
    i = summary["itl"]
    print(f"  ITL  (s) : mean={_fmt(i['mean'])}  p50={_fmt(i['median'])}  "
          f"p95={_fmt(i['p95'])}  p99={_fmt(i['p99'])}")
    print(f"  Out tokens: {summary['output_tokens']['total']} total  "
          f"({_fmt(summary['output_tokens']['mean'], '', 1)} mean/req)")
    if summary["errors"]:
        print(f"\n  Errors:")
        for e in summary["errors"][:5]:
            print(f"    #{e['id']}: {e['msg']}")
        if len(summary["errors"]) > 5:
            print(f"    ... and {len(summary['errors']) - 5} more")
    print(sep)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(args: argparse.Namespace) -> dict:
    """Run all requests and return the full result dict."""
    # Resolve task list
    if args.task == "mixture":
        valid = MODEL_TASKS[args.model]
        task_list = [valid[i % len(valid)] for i in range(args.num_requests)]
    else:
        task_list = [args.task] * args.num_requests

    image_path = Path(args.image) if args.image else None
    audio_path = Path(args.audio) if args.audio else None
    video_path = Path(args.video) if args.video else None

    model_kwargs: dict = {}
    if args.model_kwargs:
        model_kwargs = json.loads(args.model_kwargs)

    connector = aiohttp.TCPConnector(limit=0)  # no connection cap
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    all_results: list[RequestResult] = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        wall_start = time.monotonic()

        # Emit requests in waves of batch_size, each wave fired concurrently.
        # This matches the "batch size sweep" semantics: batch_size simultaneous
        # requests in-flight, repeated until num_requests are done.
        for wave_start in range(0, args.num_requests, args.batch_size):
            wave_ids = range(wave_start, min(wave_start + args.batch_size, args.num_requests))
            tasks_coros = [
                send_one(
                    session=session,
                    url=args.url,
                    request_id=i,
                    task=task_list[i],
                    text=args.prompt or DEFAULT_PROMPTS[task_list[i]],
                    model_kwargs=model_kwargs,
                    image_path=image_path,
                    audio_path=audio_path,
                    video_path=video_path,
                )
                for i in wave_ids
            ]
            wave_results = await asyncio.gather(*tasks_coros)
            all_results.extend(wave_results)

            # Optional inter-wave delay (0 by default = back-to-back waves)
            if args.wave_delay > 0 and wave_start + args.batch_size < args.num_requests:
                await asyncio.sleep(args.wave_delay)

        wall_time = time.monotonic() - wall_start

    summary = compute_summary(all_results, wall_time, batch_size=args.batch_size)

    # Build canonical output document
    output = {
        "system": args.system,
        "model": args.model,
        "task": args.task,
        "batch_size": args.batch_size,
        "num_requests": args.num_requests,
        "url": args.url,
        "summary": summary,
        "per_request": [
            {
                "id": r.request_id,
                "task": r.task,
                "status": r.status,
                "ttft": round(r.ttft, 4) if r.ttft is not None else None,
                "e2e": round(r.e2e, 4),
                "itl_mean": round(_safe_mean(r.itl_values), 4) if r.itl_values else None,
                "output_tokens": r.output_tokens,
                "output_bytes": r.output_bytes,
                "error": r.error or None,
            }
            for r in all_results
        ],
    }

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_url() -> str:
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    return f"http://{host}:{port}/generate"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified mminf benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Identity
    p.add_argument("--system", default="mminf",
                   help="System label in output JSON (e.g. mminf, vllm, sglang)")
    p.add_argument("--model", required=True, choices=list(MODEL_TASKS),
                   help="Model being served")
    p.add_argument("--task", required=True,
                   choices=list(TASK_SPEC) + ["mixture"],
                   help="Task type, or 'mixture' to round-robin valid tasks for --model")

    # Server
    p.add_argument("--url", default=None,
                   help="Server endpoint URL (default: http://$HOST:$PORT/generate)")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Per-request HTTP timeout in seconds (default: 300)")

    # Load shape
    p.add_argument("--num-requests", type=int, default=10,
                   help="Total number of requests to issue")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Number of requests sent concurrently per wave (default: 1)")
    p.add_argument("--wave-delay", type=float, default=0.0,
                   help="Seconds to wait between waves (default: 0 = back-to-back)")

    # Prompt / media
    p.add_argument("--prompt", default=None,
                   help="Text prompt override (default: per-task default)")
    p.add_argument("--image", default=None, metavar="PATH",
                   help="Image file for tasks that need image input")
    p.add_argument("--audio", default=None, metavar="PATH",
                   help="Audio file for tasks that need audio input")
    p.add_argument("--video", default=None, metavar="PATH",
                   help="Video file for tasks that need video input")
    p.add_argument("--model-kwargs", default=None, metavar="JSON",
                   help='Extra JSON kwargs forwarded to the model (e.g. \'{"voice":"tara"}\')')

    # Output
    p.add_argument("--output", default=None, metavar="PATH",
                   help="Write full JSON results to this file")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-run console summary")

    args = p.parse_args()

    if args.url is None:
        args.url = _default_url()

    # Validate task vs model
    if args.task != "mixture" and args.task not in MODEL_TASKS[args.model]:
        p.error(f"Task '{args.task}' is not supported by model '{args.model}'. "
                f"Valid tasks: {MODEL_TASKS[args.model]}")

    # Warn about missing media
    if args.task != "mixture":
        in_mods, _ = TASK_SPEC[args.task]
        if "image" in in_mods and not args.image:
            p.error(f"Task '{args.task}' requires --image")
        if "audio" in in_mods and not args.audio:
            p.error(f"Task '{args.task}' requires --audio")
        if "video" in in_mods and not args.video:
            p.error(f"Task '{args.task}' requires --video")

    return args


def main() -> None:
    args = parse_args()
    output = asyncio.run(run_benchmark(args))

    if not args.quiet:
        print_summary(output["summary"], args.model, args.task)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
