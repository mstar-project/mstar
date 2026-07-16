"""Whisper-large-v3 eval/throughput A/B: sglang-omni vs mstar.

Reuses mstar's LibriSpeechEvalDataset (same samples/order) and _compute_wer
(same jiwer normalization) so both engines are measured by the SAME client with
identical data, scoring, and closed-loop concurrency semantics. Only the engine
differs.

    --backend sglang  -> POST /v1/audio/transcriptions (OpenAI-style)
    --backend mstar   -> POST /generate (mstar A2T form + NDJSON base64 stream)

Run in the mstar venv (has datasets/jiwer/torchcodec/aiohttp):
    python -m benchmark.sglang_whisper_eval \
        --url http://localhost:18830 --backend sglang \
        --num-requests 100 --max-concurrency 8 --output-json out.json
"""

import argparse
import asyncio
import base64
import json
import os
import statistics
import time

import aiohttp

from benchmark.asr_eval import _compute_wer
from benchmark.base import Model, ModelType, RequestType
from benchmark.dataset import LibriSpeechEvalDataset


async def _transcribe_sglang(session, audio_path, prompt, base_url, model_name, mstar_model, rid):
    """One /v1/audio/transcriptions request; return (rid, text, e2e_s, err)."""
    try:
        with open(audio_path, "rb") as f:
            audio = f.read()
        form = aiohttp.FormData()
        form.add_field("model", model_name)
        form.add_field("language", "en")
        form.add_field("response_format", "json")
        form.add_field("temperature", "0")
        form.add_field(
            "file", audio,
            filename=os.path.basename(audio_path),
            content_type="audio/wav",
        )
        t0 = time.monotonic()
        async with session.post(f"{base_url}/v1/audio/transcriptions", data=form) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        e2e = time.monotonic() - t0
        text = (payload.get("text") or "").strip()
        return rid, text, e2e, None
    except Exception as exc:  # noqa: BLE001
        return rid, None, None, str(exc)


async def _transcribe_mstar(session, audio_path, prompt, base_url, model_name, mstar_model, rid):
    """One /generate A2T request (mstar form + NDJSON base64 text stream)."""
    try:
        model_kwargs = json.dumps(mstar_model.get_model_kwargs(RequestType.A2T))
        form = aiohttp.FormData()
        form.add_field("text", prompt)
        form.add_field("model_kwargs", model_kwargs)
        form.add_field("output_modalities", "text")
        form.add_field("input_modalities", "audio,text")
        with open(audio_path, "rb") as f:
            audio = f.read()
        form.add_field(
            "files", audio,
            filename=os.path.basename(audio_path),
            content_type="application/octet-stream",
        )
        chunks: list[bytes] = []
        t0 = time.monotonic()
        async with session.post(f"{base_url}/generate", data=form) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("data") and msg.get("modality") == "text":
                    chunks.append(base64.b64decode(msg["data"]))
        e2e = time.monotonic() - t0
        text = b"".join(chunks).decode("utf-8", errors="replace").strip()
        return rid, text, e2e, None
    except Exception as exc:  # noqa: BLE001
        return rid, None, None, str(exc)


_BACKENDS = {"sglang": _transcribe_sglang, "mstar": _transcribe_mstar}


async def _run(base_url, model_name, dataset, max_concurrency, num_warmup, backend, mstar_model):
    items = dataset.get_requests()
    audio_paths = [it.audio_path for it in items]
    prompts = [it.prompt for it in items]
    references = dataset.references
    transcribe = _BACKENDS[backend]

    sem = asyncio.Semaphore(max_concurrency)

    async def _limited(session, i, path):
        async with sem:
            return await transcribe(session, path, prompts[i % len(prompts)], base_url, model_name, mstar_model, i)

    connector_limit = max(100, max_concurrency + 10)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300),
        connector=aiohttp.TCPConnector(limit=connector_limit),
    ) as session:
        if num_warmup > 0 and audio_paths:
            print(f"Warming up ({num_warmup} request(s))...")
            await asyncio.gather(*[
                _limited(session, -(i + 1), audio_paths[i % len(audio_paths)])
                for i in range(num_warmup)
            ])
            print("Warmup done.")

        print(f"Running on {len(audio_paths)} sample(s), concurrency={max_concurrency}...")
        wall_start = time.monotonic()
        results = await asyncio.gather(*[
            _limited(session, i, p) for i, p in enumerate(audio_paths)
        ])
        wall_time = time.monotonic() - wall_start

    hyps = [""] * len(audio_paths)
    lat = [0.0] * len(audio_paths)
    errors = []
    for rid, text, e2e, err in results:
        if rid < 0:
            continue
        if err:
            errors.append(f"[{rid}] {err}")
        else:
            hyps[rid] = text or ""
            lat[rid] = e2e or 0.0
    return references, hyps, lat, wall_time, errors


def _pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--backend", choices=list(_BACKENDS), default="sglang")
    ap.add_argument("--model-name", default="openai/whisper-large-v3")
    ap.add_argument("--mstar-model", default="whisper_large",
                    help="mstar ModelType value (used only for --backend mstar)")
    ap.add_argument("--num-requests", type=int, default=50)
    ap.add_argument("--split", default="test.clean")
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--num-warmup", type=int, default=3)
    ap.add_argument("--local-cache", default="/tmp/mstar-eval-cache")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    mstar_model = ModelType(args.mstar_model).inst() if args.backend == "mstar" else None

    print(f"Loading LibriSpeech '{args.split}' ({args.num_requests} samples) from {args.local_cache}...")
    dataset = LibriSpeechEvalDataset(
        local_file_dir=os.path.join(args.local_cache, "librispeech_eval"),
        num_requests=args.num_requests,
        split=args.split,
    )
    print(f"Loaded {len(dataset)} samples.")

    references, hyps, lat, wall_time, errors = asyncio.run(
        _run(args.url, args.model_name, dataset, args.max_concurrency,
             args.num_warmup, args.backend, mstar_model)
    )

    wer_results = _compute_wer(references, hyps)
    ok_lat = [x for x in lat if x > 0]
    n = len(hyps)

    summary = {
        "engine": "sglang-omni" if args.backend == "sglang" else "mstar",
        "backend": args.backend,
        "model": args.model_name,
        "split": args.split,
        "num_requests": n,
        "max_concurrency": args.max_concurrency,
        "wall_time_s": wall_time,
        "throughput_req_s": n / wall_time if wall_time else 0.0,
        "wer": wer_results["wer"],
        "e2e_mean_s": statistics.mean(ok_lat) if ok_lat else 0.0,
        "e2e_p50_s": _pct(ok_lat, 0.50),
        "e2e_p95_s": _pct(ok_lat, 0.95),
        "e2e_p99_s": _pct(ok_lat, 0.99),
        "errors": errors,
    }

    print(f"\n{'=' * 56}")
    print(f"  {summary['engine']}  {args.model_name}  (backend={args.backend})")
    print(f"{'=' * 56}")
    print(f"  Samples     : {n}   (errors: {len(errors)})")
    print(f"  WER         : {summary['wer'] * 100:.2f}%")
    print(f"  Wall time   : {wall_time:.2f}s")
    print(f"  Throughput  : {summary['throughput_req_s']:.2f} req/s")
    print(f"  E2E  mean   : {summary['e2e_mean_s']:.3f}s  p50={summary['e2e_p50_s']:.3f}s  "
          f"p95={summary['e2e_p95_s']:.3f}s  p99={summary['e2e_p99_s']:.3f}s")
    if errors:
        print(f"  first errors: {errors[:3]}")
    print(f"{'=' * 56}\n")

    if args.output_json:
        summary["per_sample"] = wer_results["per_sample"]
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
