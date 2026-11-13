"""Run the repository benchmark on an explicit, reusable Ministral request manifest.

The JSONL manifest contains ``prompt`` and optional ``image_path`` (relative to
the manifest directory). Every arm uses the same OpenAI HTTP client. Preserve
raw client observations, including failures, alongside runner aggregates. This
is a performance harness; correctness and checkpoint equivalence are separate
gates. See README.rst for measurement limitations.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from benchmark.base import Ministral3, RequestType, Status
from benchmark.request import RequestInput
from benchmark.runner import Benchmark, BenchmarkConfig, DatasetType, InferenceSystemType, ProfilingType


class ManifestBenchmark(Benchmark):
    def __init__(self, config, requests):
        super().__init__(config)
        self.requests = requests

    def _get_dataset(self):
        return SimpleNamespace(get_requests=lambda: self.requests)


async def run(args):
    manifest = args.manifest.resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if args.num_requests is not None:
        if args.num_requests > len(rows):
            raise ValueError("The manifest has fewer requests than requested; implicit repetition is disabled")
        rows = rows[:args.num_requests]
    if len(rows) < args.concurrency or args.concurrency < 1:
        raise ValueError("Require at least one request per concurrency slot")
    requests, images = [], {}
    for row in rows:
        image = (manifest.parent / row["image_path"]).resolve() if row.get("image_path") else None
        if image is not None:
            images[str(image)] = hashlib.sha256(image.read_bytes()).hexdigest()
        requests.append(RequestInput(
            req_type=RequestType.I2T if image else RequestType.T2T,
            prompt=row["prompt"], image_path=str(image) if image else None,
        ))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "engine": args.engine, "url": args.url,
        "checkpoint": "mistralai/Ministral-3-3B-Instruct-2512",
        "checkpoint_revision": "b35d4dfe56c142746f54dbd64f579faab2744308",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "requests": len(rows), "concurrency": args.concurrency,
        "output_tokens": args.output_tokens, "warmup_waves": args.warmup,
        "image_sha256": images,
        "system_message": Ministral3().get_openai_system_message(),
        "precision_equivalence": "must be established separately from timing",
    }
    (args.output_dir / "run-manifest.json").write_text(json.dumps(metadata, indent=2))
    (args.output_dir / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = BenchmarkConfig(
        url=args.url, model=Ministral3(), dataset=DatasetType.TEXT,
        num_requests=len(requests), request_type=RequestType.T2T,
        local_cache_dir=str(args.output_dir / "cache"), num_warmup=args.warmup,
        profiling_type=ProfilingType.CLOSED_LOOP,
        inference_system=InferenceSystemType.OURS_OPENAI,
        max_concurrency=args.concurrency, output_dir=str(args.output_dir),
        ignore_eos=True, output_len_min=args.output_tokens, output_len_max=args.output_tokens,
        verbose=True,
    )
    metrics, _ = await ManifestBenchmark(config, requests).run()
    tokenizer = config.model.get_tokenizer()
    records = []
    for m in metrics:
        output_text = b"".join(m._text_chunks).decode("utf-8", errors="replace")
        records.append({
            "request_id": m.request_id, "type": m.type.value, "status": m.status.value,
            "error": m.error, "e2e_s": m.e2e_latency, "ttft_s": m.ttft,
            "output_tokens": m.output_text_tokens, "input_tokens": m.input_tokens,
            "output_count_source": "server_usage" if m.server_completion_tokens is not None else "chunk_estimate",
            "server_completion_tokens": m.server_completion_tokens,
            "output_text": output_text,
            "decoded_text_tokens": len(tokenizer.encode(output_text, add_special_tokens=False)),
            "chunks": m.response_chunks, "output_bytes": m.output_bytes,
            "chunk_arrivals_s": {
                key: [t - m.start_time for t in times] for key, times in m.chunk_arrivals.items()
            },
        })
    (args.output_dir / "raw-requests.json").write_text(json.dumps(records, indent=2))
    # Keep the adapter label as well as the actual engine being measured.
    path = args.output_dir / "results.json"
    result = json.loads(path.read_text())
    result["engine"] = args.engine
    result["token_counts_verified_by_server"] = all(m.server_completion_tokens is not None for m in metrics)
    result["measurement_note"] = (
        "Token throughput and token-normalized latency are estimates when server usage is absent. "
        "Chunk counts do not prove fixed generation length. Decoded text is re-tokenized after timing; "
        "this checks the returned text length, not the original generated token IDs."
    )
    path.write_text(json.dumps(result, indent=2))
    failures = [m for m in metrics if m.status != Status.SUCCESS]
    wrong_lengths = [r for r in records if r["status"] == Status.SUCCESS.value and
                     (r["server_completion_tokens"] if r["server_completion_tokens"] is not None
                      else r["decoded_text_tokens"]) != args.output_tokens]
    if failures or wrong_lengths:
        raise RuntimeError(f"Invalid measurement: {len(failures)} failures, {len(wrong_lengths)} wrong output lengths")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--engine", required=True, choices=["mstar", "vllm", "sglang"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
