"""ASR evaluation script: compute Word Error Rate (WER) on LibriSpeech test-clean.

Runs inference against a running mstar server and compares transcription output
against ground-truth transcripts.  Supports Whisper-large-v3, Higgs Audio, and
Voxtral.

Usage:
    python -m benchmark.asr_eval \\
        --url http://localhost:8000 \\
        --model whisper_large \\
        --num-requests 100 \\
        --local-cache ./mstar-eval-cache

Requires:
    pip install jiwer
"""

import argparse
import asyncio
import base64
import json
import os
import time

import aiohttp

from benchmark.base import Model, ModelType, RequestType
from benchmark.dataset import LibriSpeechEvalDataset
from benchmark.request import RequestInput

_ASR_MODELS = {ModelType.WHISPER_LARGE, ModelType.HIGGS_AUDIO, ModelType.VOXTRAL}


async def _transcribe(
    session: aiohttp.ClientSession,
    req_input: RequestInput,
    base_url: str,
    request_id: int,
    model: Model,
) -> tuple[int, str | None, str | None]:
    """Send one A2T request to /generate; return (id, transcript, error)."""
    model_kwargs = json.dumps(
        {**model.get_model_kwargs(RequestType.A2T), **req_input.model_kwargs}
    )

    try:
        form = aiohttp.FormData()
        form.add_field("text", req_input.prompt)
        form.add_field("model_kwargs", model_kwargs)
        form.add_field("output_modalities", "text")
        form.add_field("input_modalities", "audio,text")

        audio_bytes = req_input.get_bytes("audio")
        if audio_bytes:
            form.add_field(
                "files",
                audio_bytes,
                filename=req_input.get_filename("audio"),
                content_type="application/octet-stream",
            )

        text_chunks: list[bytes] = []
        async with session.post(f"{base_url}/generate", data=form) as resp:
            resp.raise_for_status()
            async for raw_line in resp.content:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not msg.get("data"):
                    continue
                if msg.get("modality") == "text":
                    text_chunks.append(base64.b64decode(msg["data"]))

        transcript = b"".join(text_chunks).decode("utf-8", errors="replace").strip()
        return request_id, transcript, None

    except Exception as exc:
        return request_id, None, str(exc)


async def _run_eval(
    base_url: str,
    model: Model,
    dataset: LibriSpeechEvalDataset,
    max_concurrency: int,
    num_warmup: int,
) -> dict:
    requests = dataset.get_requests()
    references = dataset.references

    sem = asyncio.Semaphore(max_concurrency)

    async def _limited(i: int, req: RequestInput):
        async with sem:
            return await _transcribe(session, req, base_url, i, model)

    connector_limit = max(100, max_concurrency + 10)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300),
        connector=aiohttp.TCPConnector(limit=connector_limit),
    ) as session:
        # Warmup
        if num_warmup > 0 and requests:
            print(f"Warming up ({num_warmup} request(s))...")
            warmup_tasks = [
                asyncio.create_task(_limited(-(i + 1), requests[i % len(requests)]))
                for i in range(num_warmup)
            ]
            await asyncio.gather(*warmup_tasks)
            print("Warmup done.")

        print(f"Running eval on {len(requests)} sample(s)...")
        wall_start = time.monotonic()
        tasks = [asyncio.create_task(_limited(i, req)) for i, req in enumerate(requests)]
        results = await asyncio.gather(*tasks)
        wall_time = time.monotonic() - wall_start

    hypotheses: list[str] = [""] * len(requests)
    errors: list[str] = []
    for req_id, transcript, err in results:
        if req_id >= 0:
            if err:
                errors.append(f"[{req_id}] {err}")
            else:
                hypotheses[req_id] = transcript or ""

    return {
        "references": references,
        "hypotheses": hypotheses,
        "errors": errors,
        "wall_time": wall_time,
        "num_requests": len(requests),
    }


def _compute_wer(references: list[str], hypotheses: list[str]) -> dict:
    try:
        import jiwer
    except ImportError as exc:
        raise ImportError("pip install jiwer") from exc

    transform = jiwer.Compose([
        jiwer.ToUpperCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
    ])

    refs_clean = [transform(r) for r in references]
    hyps_clean = [transform(h) for h in hypotheses]

    overall = jiwer.wer(refs_clean, hyps_clean)

    per_sample: list[dict] = []
    for i, (ref, hyp) in enumerate(zip(refs_clean, hyps_clean, strict=True)):
        sample_wer = jiwer.wer([ref], [hyp]) if ref else 0.0
        per_sample.append({"index": i, "wer": sample_wer, "ref": ref, "hyp": hyp})

    return {"wer": overall, "per_sample": per_sample}


def _print_report(
    model_name: str,
    wer_results: dict,
    wall_time: float,
    num_requests: int,
    errors: list[str],
) -> None:
    wer = wer_results["wer"]
    per_sample = wer_results["per_sample"]

    worst = sorted(per_sample, key=lambda x: x["wer"], reverse=True)[:5]

    print(f"\n{'=' * 60}")
    print(f"  ASR Evaluation — {model_name}")
    print(f"{'=' * 60}")
    print(f"  Samples   : {num_requests}")
    print(f"  Errors    : {len(errors)}")
    print(f"  Wall time : {wall_time:.2f}s")
    print(f"  Throughput: {num_requests / wall_time:.2f} req/s")
    print(f"\n  WER (overall): {wer * 100:.2f}%")
    print("\n  Worst samples (top-5):")
    for s in worst:
        print(f"    [{s['index']:4d}] WER={s['wer'] * 100:.1f}%")
        print(f"           REF: {s['ref'][:80]}")
        print(f"           HYP: {s['hyp'][:80]}")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")
    print(f"{'=' * 60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR WER evaluation on LibriSpeech")
    parser.add_argument("--url", required=True, help="mstar server base URL")
    parser.add_argument(
        "--model",
        required=True,
        choices=[m.value for m in _ASR_MODELS],
        help="ASR model to evaluate",
    )
    parser.add_argument(
        "--num-requests", type=int, default=100, help="Number of audio samples to evaluate"
    )
    parser.add_argument(
        "--split",
        default="test.clean",
        help="LibriSpeech split (default: test.clean)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max in-flight requests (default: 4)",
    )
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--local-cache", default="./mstar-eval-cache", type=str)
    parser.add_argument("--hf-cache", default=None, help="HuggingFace cache dir")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write full results JSON to this path",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    model_type = ModelType(args.model)
    model = model_type.inst()

    print(f"Loading LibriSpeech '{args.split}' split ({args.num_requests} samples)...")
    dataset = LibriSpeechEvalDataset(
        local_file_dir=os.path.join(args.local_cache, "librispeech_eval"),
        num_requests=args.num_requests,
        split=args.split,
        cache_dir=args.hf_cache,
    )
    print(f"Loaded {len(dataset)} samples.")

    result = await _run_eval(
        base_url=args.url,
        model=model,
        dataset=dataset,
        max_concurrency=args.max_concurrency,
        num_warmup=args.num_warmup,
    )

    wer_results = _compute_wer(result["references"], result["hypotheses"])

    _print_report(
        model_name=args.model,
        wer_results=wer_results,
        wall_time=result["wall_time"],
        num_requests=result["num_requests"],
        errors=result["errors"],
    )

    if args.output_json:
        payload = {
            "model": args.model,
            "split": args.split,
            "num_requests": result["num_requests"],
            "wall_time_s": result["wall_time"],
            "wer": wer_results["wer"],
            "errors": result["errors"],
            "per_sample": wer_results["per_sample"],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
