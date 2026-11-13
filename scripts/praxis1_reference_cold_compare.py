"""Compare one saved M* cold DROID action reply with the pinned Praxis pipeline.

Run after the diagnostic M* server has stopped so both models never occupy
the GPU together. The same generated observation, prompt and seed are used.
This is a numerical diagnostic; the WAN VAE implementations differ slightly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from praxis.model import SamplerSettings
from praxis.pipeline import PraxisPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mstar-actions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae-pth", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--mstar-revision", required=True)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.03)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "comparison needs CUDA"
    for path in (args.mstar_actions, args.vae_pth, args.qwen / "config.json", args.checkpoint / "config.json"):
        assert path.is_file(), f"missing input: {path}"
    actual = np.load(args.mstar_actions, allow_pickle=False)
    assert actual.shape == (21, 10) and actual.dtype == np.float32
    random = np.random.default_rng(71)
    exterior = random.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    wrist = random.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    observation = {
        "image.exterior": exterior,
        "image.wrist": wrist,
        "robot_state.ee_pose": np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "grip": np.array([0.0], dtype=np.float32),
    }
    settings = SamplerSettings(num_steps=2, t_shift=12.5, cfg_scale=1.0, cfg_video_scale=4.0)
    started = time.perf_counter()
    pipeline = PraxisPipeline(
        args.checkpoint, args.vae_pth, args.qwen, device="cuda", sampler=settings,
        encoder_attention_backend="triton", decoder_attention_backend="triton",
        action_cuda_graphs=False, nvfp4_moe_cuda_graphs=False,
    )
    session = pipeline.session(
        "pick up the marker", seed=17, max_context_chunks=0,
        control_mode="none", exec_horizon=21, sampler=settings, skip_last_video=True,
    )
    with torch.inference_mode():
        result = session.step(observation)
    torch.cuda.synchronize()
    expected = np.asarray(result.action_rel, dtype=np.float32)
    assert expected.shape == actual.shape, (expected.shape, actual.shape)
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    diff = np.abs(actual - expected)
    close = np.isclose(actual, expected, rtol=args.rtol, atol=args.atol)
    print(json.dumps({
        "mstar_revision": args.mstar_revision,
        "reference_revision": "68f4d7a254b4bf41951d2e97f44b07f6cac2bf69",
        "checkpoint": str(args.checkpoint), "shape": list(actual.shape),
        "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
        "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-12)),
        "fraction_close": float(close.mean()), "all_close": bool(close.all()),
        "per_dim_max_abs": [float(x) for x in diff.max(axis=0)],
        "reference_seconds_including_load": round(time.perf_counter() - started, 2),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
    }), flush=True)


if __name__ == "__main__":
    main()
