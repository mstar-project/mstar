#!/usr/bin/env python3
"""HuggingFace transformers baseline for V-JEPA 2 (action-conditioned) on DROID.

Mirrors the methodology of the vLLM-Omni paper's HF transformers baseline
(see ``vllm-omni/benchmarks/qwen3-omni/transformers/qwen3_omni_moe_transformers.py``):
synchronous in-process Python script, concurrency=1, warmup + timed loop,
JCT (Job Completion Time) as the primary metric. No FastAPI shim — the
small HTTP / multipart overhead asymmetry vs our serving system is
footnoted, not engineered around.

The script loads ``facebook/vjepa2-ac-vitg-256`` (the action-conditioned
V-JEPA 2 checkpoint that mstar's ``VJepa2AC`` model class also targets —
see ``benchmark/base.py``). If that checkpoint isn't reachable on
HuggingFace Hub, swap to the plain world-model checkpoint via
``--hf-model facebook/vjepa2-vitl-fpc64-256`` and note in the paper that
the HF baseline is plain V-JEPA 2 (no action conditioning) since HF
transformers doesn't expose AC rollout out of the box.

Usage
-----
    # 5-episode smoke test
    python benchmark/hf_vjepa2_ac.py --num-requests 5 --output-dir /tmp/hf_vjepa2

    # Match runner.py CLI pattern: same dataset, same warmup, same device
    python benchmark/hf_vjepa2_ac.py \
        --num-requests 100 --num-warmup 3 --rollout-horizon 4 \
        --output-dir results/hf_vjepa2 --hf-cache /data/hf_cache

Outputs
-------
- ``<output-dir>/req_NN_latents.npy``  per-request predictor latents
  (same filename our system produces, so ``validate_latents.py`` works).
- ``<output-dir>/results.json``        aggregate stats + per-request JCT.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Allow `from benchmark.dataset import DROIDDataset` when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_MODEL_ID = "facebook/vjepa2-ac-vitg-256"


@dataclass
class PerRequestResult:
    request_id: int
    jct_ms: float
    n_rollout_steps: int
    output_shape: list[int]
    finite: bool


@dataclass
class BenchmarkResult:
    system: str = "hf_transformers"
    model: str = "vjepa2_ac"
    hf_model_id: str = DEFAULT_MODEL_ID
    num_requests: int = 0
    num_warmup: int = 0
    completed: int = 0
    failed: int = 0
    rollout_horizon: int = 4
    # JCT (E2E latency) stats (ms)
    jct_mean_ms: float = 0.0
    jct_median_ms: float = 0.0
    jct_std_ms: float = 0.0
    jct_p90_ms: float = 0.0
    jct_p95_ms: float = 0.0
    jct_p99_ms: float = 0.0
    # Throughput
    rollout_steps_per_sec: float = 0.0
    request_throughput: float = 0.0
    per_request: list[PerRequestResult] = field(default_factory=list)


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile, matches numpy default."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _load_video_tensor(video_path: str, num_frames: int = 16):
    """Decode an mp4 to a tensor sized for the VJEPA2 video processor.

    Returns a list of PIL frames; the processor will handle resize / crop /
    normalize. We sample ``num_frames`` evenly across the clip (the AC
    checkpoint expects 64 in mstar, but the predictor only consumes a fixed
    grid_depth so the processor + model handle that internally — for this
    baseline we just pass the raw frames and let the processor decide).
    """
    from PIL import Image
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(video_path)
    n_total = dec.metadata.num_frames if dec.metadata.num_frames else len(dec)
    indices = np.linspace(0, max(0, n_total - 1), num=num_frames).astype(int).tolist()
    batch = dec.get_frames_at(indices=indices)
    tensors = batch.data  # [T, C, H, W] uint8
    frames = [Image.fromarray(t.permute(1, 2, 0).numpy()) for t in tensors]
    return frames


def _run_one(model, processor, video_path: str, device: str, dtype):
    """Run a single forward pass and return predictor latents + step count."""
    import torch

    frames = _load_video_tensor(video_path, num_frames=16)
    inputs = processor(videos=[frames], return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device)
              for k, v in inputs.items()}

    with torch.inference_mode():
        out = model(**inputs)

    # VJEPA2WithMaskedInputModelOutput.predictor_output.last_hidden_state
    # has shape [B=1, N_tokens, hidden_size]. We treat each "rollout step"
    # as a single forward (mstar's rollout calls the predictor H times; HF's
    # plain VJEPA2 produces one prediction per call). For an AC checkpoint
    # that internally rolls out, this is one full prediction — the JCT
    # captures the full inference cost per request, which is what we want.
    pred = out.predictor_output
    if pred is None:
        # Some configs return only encoder output; fall back to that.
        latents = out.last_hidden_state
    else:
        latents = pred.last_hidden_state
    return latents.float().cpu().numpy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF transformers baseline for V-JEPA 2 on DROID")
    p.add_argument("--num-requests", type=int, default=10)
    p.add_argument("--num-warmup", type=int, default=3,
                   help="Warmup requests, identical cadence to runner.py defaults.")
    p.add_argument("--rollout-horizon", type=int, default=4,
                   help="Passed to DROIDDataset; also recorded in result JSON.")
    p.add_argument("--hf-model", default=DEFAULT_MODEL_ID,
                   help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID}).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--output-dir", default="/tmp/hf_vjepa2")
    p.add_argument("--local-cache", default="./mstar-benchmark-cache/",
                   help="Same default as runner.py so DROIDDataset reuses extracted videos.")
    p.add_argument("--hf-cache", default=None,
                   help="HuggingFace cache directory (for lerobot/droid_100 + model weights).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.local_cache, exist_ok=True)

    try:
        import torch
        from transformers import AutoVideoProcessor, VJEPA2Model
    except ImportError as e:
        sys.exit(
            f"ERROR: {e}\n"
            "transformers / torch not installed. Run:\n"
            "  pip install -U transformers torch torchcodec\n"
        )

    print(f"=== HF V-JEPA 2 baseline ===")
    print(f"  model_id    : {args.hf_model}")
    print(f"  device      : {args.device}")
    print(f"  dtype       : {args.dtype}")
    print(f"  num_requests: {args.num_requests}")
    print(f"  num_warmup  : {args.num_warmup}")

    # ------------------------------------------------------------------
    # Load model + processor
    # ------------------------------------------------------------------
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"\nLoading model {args.hf_model}...")
    t0 = time.perf_counter()
    model = VJEPA2Model.from_pretrained(args.hf_model, dtype=dtype).to(args.device)
    model.eval()
    processor = AutoVideoProcessor.from_pretrained(args.hf_model)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Build dataset (same DROIDDataset our HTTP harness uses)
    # ------------------------------------------------------------------
    from benchmark.dataset import DROIDDataset

    print(f"\nBuilding DROIDDataset (task=vjepa2_ac, n={args.num_requests})...")
    dataset = DROIDDataset(
        local_file_dir=args.local_cache,
        num_requests=args.num_requests,
        task="vjepa2_ac",
        rollout_horizon=args.rollout_horizon,
        cache_dir=args.hf_cache,
    )
    requests = dataset.get_requests()
    if not requests:
        sys.exit("ERROR: DROIDDataset returned 0 requests")
    print(f"  loaded {len(requests)} episodes")

    # ------------------------------------------------------------------
    # Warmup (same cadence as runner.py's _warmup with bs=1)
    # ------------------------------------------------------------------
    if args.num_warmup > 0:
        print(f"\nWarmup ({args.num_warmup} requests)...")
        for w in range(args.num_warmup):
            _run_one(model, processor, requests[0].video_path, args.device, dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("  done")

    # ------------------------------------------------------------------
    # Timed loop
    # ------------------------------------------------------------------
    print(f"\nRunning benchmark ({len(requests)} requests, sequential)...")
    per_request: list[PerRequestResult] = []
    failed = 0
    wall_start = time.monotonic()
    for i, req in enumerate(requests):
        try:
            t0 = time.perf_counter()
            latents = _run_one(model, processor, req.video_path, args.device, dtype)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            jct_ms = (time.perf_counter() - t0) * 1000.0

            np.save(os.path.join(args.output_dir, f"req_{i:02d}_latents.npy"), latents)
            per_request.append(PerRequestResult(
                request_id=i,
                jct_ms=jct_ms,
                n_rollout_steps=args.rollout_horizon,
                output_shape=list(latents.shape),
                finite=bool(np.isfinite(latents).all()),
            ))
            print(f"  req {i:02d}: jct={jct_ms:.1f} ms  shape={latents.shape}  "
                  f"finite={per_request[-1].finite}")
        except Exception as e:
            failed += 1
            print(f"  req {i:02d}: FAILED — {e}")
    wall_time = time.monotonic() - wall_start

    # ------------------------------------------------------------------
    # Aggregate + write JSON
    # ------------------------------------------------------------------
    jcts = [r.jct_ms for r in per_request]
    result = BenchmarkResult(
        hf_model_id=args.hf_model,
        num_requests=len(requests),
        num_warmup=args.num_warmup,
        completed=len(per_request),
        failed=failed,
        rollout_horizon=args.rollout_horizon,
        per_request=per_request,
    )
    if jcts:
        result.jct_mean_ms = statistics.mean(jcts)
        result.jct_median_ms = statistics.median(jcts)
        result.jct_std_ms = statistics.stdev(jcts) if len(jcts) > 1 else 0.0
        result.jct_p90_ms = _percentile(jcts, 90)
        result.jct_p95_ms = _percentile(jcts, 95)
        result.jct_p99_ms = _percentile(jcts, 99)
        result.rollout_steps_per_sec = (
            sum(r.n_rollout_steps for r in per_request) / (sum(jcts) / 1000.0)
        )
        result.request_throughput = len(per_request) / wall_time if wall_time else 0.0

    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump({**asdict(result),
                   "per_request": [asdict(r) for r in per_request]},
                  f, indent=2)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n=== Results (wall {wall_time:.1f}s, {len(per_request)}/{len(requests)} ok) ===")
    if jcts:
        print(f"  JCT mean   : {result.jct_mean_ms:.1f} ms")
        print(f"  JCT median : {result.jct_median_ms:.1f} ms")
        print(f"  JCT p95    : {result.jct_p95_ms:.1f} ms")
        print(f"  JCT p99    : {result.jct_p99_ms:.1f} ms")
        print(f"  Throughput : {result.request_throughput:.2f} req/s, "
              f"{result.rollout_steps_per_sec:.2f} rollout-steps/s")
    print(f"  Outputs    : {args.output_dir}/")
    print(f"  Results    : {out_json}")


if __name__ == "__main__":
    main()
