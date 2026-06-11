#!/usr/bin/env python3
"""openpi (PyTorch) baseline for π₀.₅ on DROID.

Mirrors the methodology of the vLLM-Omni paper's HF transformers baseline
pattern (see ``vllm-omni/benchmarks/qwen3-omni/transformers/qwen3_omni_moe_transformers.py``):
synchronous in-process Python script, concurrency=1, warmup + timed loop,
JCT (Job Completion Time) as the primary metric. We use openpi as the
baseline because HuggingFace transformers ships ``pi0`` (Oct 2024) but
not ``pi0.5`` — and openpi is Physical Intelligence's official
implementation for both, mirroring vLLM-Omni's *"For BAGEL and MiMo-Audio,
we adopt its original implementation as our baseline"* approach for
models without an HF release.

Env requirement
---------------
This script must run inside the **openpi** conda env (or any env with
``openpi`` installed via ``uv pip install -e .``). openpi pins
``torch==2.7.1`` and ``transformers==4.53.2`` with vendored patches, which
will conflict with our mstar environment — keep them separate.

    conda activate openpi
    python benchmark/openpi_pi05.py --num-requests 5 --output-dir /tmp/openpi_pi05

The script does an import check at startup and prints the activation
command if openpi isn't reachable.

Outputs
-------
- ``<output-dir>/req_NN_actions.npy``  per-request action chunk [H, 8]
  (same filename our system produces, so ``validate_actions.py`` works).
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

# Allow `from benchmark.dataset import DROIDDataset` when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_CONFIG = "pi05_droid"
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_droid"


@dataclass
class PerRequestResult:
    request_id: int
    jct_ms: float
    openpi_internal_infer_ms: float  # from policy.infer()'s own policy_timing
    n_actions: int
    action_dim: int
    finite: bool


@dataclass
class BenchmarkResult:
    system: str = "openpi"
    model: str = "pi05"
    openpi_config: str = DEFAULT_CONFIG
    checkpoint: str = DEFAULT_CHECKPOINT
    num_requests: int = 0
    num_warmup: int = 0
    completed: int = 0
    failed: int = 0
    # JCT (E2E latency, externally timed) stats (ms)
    jct_mean_ms: float = 0.0
    jct_median_ms: float = 0.0
    jct_std_ms: float = 0.0
    jct_p90_ms: float = 0.0
    jct_p95_ms: float = 0.0
    jct_p99_ms: float = 0.0
    # openpi's own internal timing (excludes our wrapping overhead)
    openpi_internal_mean_ms: float = 0.0
    # Throughput
    actions_per_sec: float = 0.0
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


def _check_env_or_exit():
    """Verify openpi is importable; bail with a friendly conda hint if not."""
    try:
        import openpi  # noqa: F401
        from openpi.policies import policy_config  # noqa: F401
        from openpi.shared import download  # noqa: F401
        from openpi.training import config as _config  # noqa: F401
    except ImportError as e:
        sys.exit(
            f"\n[ERROR] openpi is not importable in this environment ({e}).\n\n"
            "This script requires the dedicated openpi conda env. To activate it:\n\n"
            "    conda activate openpi\n"
            "    python benchmark/openpi_pi05.py ...\n\n"
            "If openpi isn't installed yet, follow openpi/README.md:\n"
            "    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git\n"
            "    cd openpi\n"
            "    GIT_LFS_SKIP_SMUDGE=1 uv sync\n"
            "    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .\n\n"
            "Note: openpi pins torch==2.7.1 and transformers==4.53.2 with vendored\n"
            "patches, so it must live in its own env separate from mstar.\n"
        )


def _load_image_uint8(path: str) -> np.ndarray:
    """Read a PNG and return (H, W, 3) uint8 — DroidInputs's expected format."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _build_obs(req_input) -> dict:
    """Map our DROIDDataset RequestInput → openpi DroidInputs dict.

    DROIDDataset emits:
      - image_path           → first camera (mapped to exterior_image_1_left)
      - extra_image_paths[0] → second camera (mapped to wrist_image_left)
      - extra_image_paths[1] → third camera (unused by openpi droid policy)
      - model_kwargs["robot_state"] → 32-dim padded state vector
      - prompt → language instruction

    openpi DroidInputs (in droid_policy.py:make_droid_example) expects:
      observation/exterior_image_1_left : (H,W,3) uint8
      observation/wrist_image_left      : (H,W,3) uint8
      observation/joint_position        : (7,) float
      observation/gripper_position      : (1,) or scalar float
      prompt                            : str

    The DROID 32-dim state convention: first 7 dims are joint positions,
    dim 7 is gripper. Anything beyond is per-platform padding we ignore.
    """
    state = np.asarray(req_input.model_kwargs.get("robot_state", []), dtype=np.float32)
    if state.size < 8:
        state = np.pad(state, (0, 8 - state.size))
    joint_pos = state[:7].astype(np.float32)
    gripper_pos = state[7:8].astype(np.float32)

    base_img = _load_image_uint8(req_input.image_path)
    wrist_path = req_input.extra_image_paths[0] if req_input.extra_image_paths else req_input.image_path
    wrist_img = _load_image_uint8(wrist_path)

    return {
        "observation/exterior_image_1_left": base_img,
        "observation/wrist_image_left": wrist_img,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": gripper_pos,
        "prompt": req_input.prompt or "manipulate the object",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="openpi pi0.5 baseline on DROID")
    p.add_argument("--num-requests", type=int, default=10)
    p.add_argument("--num-warmup", type=int, default=3,
                   help="Warmup requests, identical cadence to runner.py defaults.")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"openpi config name (default: {DEFAULT_CONFIG}).")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help=f"GCS / local path to checkpoint (default: {DEFAULT_CHECKPOINT}).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="/tmp/openpi_pi05")
    p.add_argument("--local-cache", default="./mstar-benchmark-cache/",
                   help="Same default as runner.py so DROIDDataset reuses extracted videos.")
    p.add_argument("--hf-cache", default=None,
                   help="HuggingFace cache directory for lerobot/droid_100.")
    return p.parse_args()


def main():
    # argparse first so `--help` works even when openpi isn't installed.
    args = parse_args()

    _check_env_or_exit()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.local_cache, exist_ok=True)

    # Imports deferred until after env check so the friendly message fires
    # before any heavy library loads.
    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as _config

    print(f"=== openpi pi0.5 baseline ===")
    print(f"  config       : {args.config}")
    print(f"  checkpoint   : {args.checkpoint}")
    print(f"  device       : {args.device}")
    print(f"  num_requests : {args.num_requests}")
    print(f"  num_warmup   : {args.num_warmup}")

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    print(f"\nDownloading checkpoint (cached at ~/.cache/openpi)...")
    t0 = time.perf_counter()
    config = _config.get_config(args.config)
    ckpt_dir = download.maybe_download(args.checkpoint)
    print(f"  ckpt at {ckpt_dir} ({time.perf_counter() - t0:.1f}s)")

    print(f"\nCreating policy...")
    t0 = time.perf_counter()
    policy = policy_config.create_trained_policy(
        config, ckpt_dir, pytorch_device=args.device
    )
    print(f"  policy ready ({time.perf_counter() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # Build dataset (same DROIDDataset as our HTTP harness)
    # ------------------------------------------------------------------
    from benchmark.dataset import DROIDDataset

    print(f"\nBuilding DROIDDataset (task=pi05, n={args.num_requests})...")
    dataset = DROIDDataset(
        local_file_dir=args.local_cache,
        num_requests=args.num_requests,
        task="pi05",
        cache_dir=args.hf_cache,
    )
    requests = dataset.get_requests()
    if not requests:
        sys.exit("ERROR: DROIDDataset returned 0 requests")
    print(f"  loaded {len(requests)} episodes")

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    if args.num_warmup > 0:
        print(f"\nWarmup ({args.num_warmup} requests)...")
        warmup_obs = _build_obs(requests[0])
        for _ in range(args.num_warmup):
            policy.infer(warmup_obs)
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
            obs = _build_obs(req)
            t0 = time.perf_counter()
            result = policy.infer(obs)
            jct_ms = (time.perf_counter() - t0) * 1000.0

            actions = np.asarray(result["actions"])
            internal_ms = float(result.get("policy_timing", {}).get("infer_ms", 0.0))

            np.save(os.path.join(args.output_dir, f"req_{i:02d}_actions.npy"), actions)
            per_request.append(PerRequestResult(
                request_id=i,
                jct_ms=jct_ms,
                openpi_internal_infer_ms=internal_ms,
                n_actions=int(actions.shape[0]),
                action_dim=int(actions.shape[1]) if actions.ndim >= 2 else 0,
                finite=bool(np.isfinite(actions).all()),
            ))
            print(f"  req {i:02d}: jct={jct_ms:.1f} ms (openpi internal={internal_ms:.1f} ms)  "
                  f"shape={actions.shape}  finite={per_request[-1].finite}")
        except Exception as e:
            failed += 1
            print(f"  req {i:02d}: FAILED — {e}")
    wall_time = time.monotonic() - wall_start

    # ------------------------------------------------------------------
    # Aggregate + write JSON
    # ------------------------------------------------------------------
    jcts = [r.jct_ms for r in per_request]
    internal = [r.openpi_internal_infer_ms for r in per_request]
    result = BenchmarkResult(
        openpi_config=args.config,
        checkpoint=args.checkpoint,
        num_requests=len(requests),
        num_warmup=args.num_warmup,
        completed=len(per_request),
        failed=failed,
        per_request=per_request,
    )
    if jcts:
        result.jct_mean_ms = statistics.mean(jcts)
        result.jct_median_ms = statistics.median(jcts)
        result.jct_std_ms = statistics.stdev(jcts) if len(jcts) > 1 else 0.0
        result.jct_p90_ms = _percentile(jcts, 90)
        result.jct_p95_ms = _percentile(jcts, 95)
        result.jct_p99_ms = _percentile(jcts, 99)
        result.openpi_internal_mean_ms = statistics.mean(internal) if internal else 0.0
        total_actions = sum(r.n_actions for r in per_request)
        result.actions_per_sec = (total_actions / (sum(jcts) / 1000.0)) if sum(jcts) else 0.0
        result.request_throughput = (len(per_request) / wall_time) if wall_time else 0.0

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
        print(f"  JCT mean      : {result.jct_mean_ms:.1f} ms  "
              f"(openpi internal mean: {result.openpi_internal_mean_ms:.1f} ms)")
        print(f"  JCT median    : {result.jct_median_ms:.1f} ms")
        print(f"  JCT p95       : {result.jct_p95_ms:.1f} ms")
        print(f"  JCT p99       : {result.jct_p99_ms:.1f} ms")
        print(f"  Throughput    : {result.request_throughput:.2f} req/s, "
              f"{result.actions_per_sec:.2f} actions/s")
    print(f"  Outputs       : {args.output_dir}/")
    print(f"  Results       : {out_json}")


if __name__ == "__main__":
    main()
