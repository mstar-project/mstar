#!/usr/bin/env python3
"""Upstream vjepa2 (Meta FAIR) baseline for AC rollout on DROID.

Mirrors the methodology of vllm-omni's HF transformers baseline and our own
``benchmark/openpi_pi05.py``: synchronous in-process Python script,
sequential concurrency=1, warmup + timed loop, JCT (Job Completion Time)
as the primary metric.

Why upstream and not HF
-----------------------
HuggingFace transformers ships base V-JEPA 2 only (encoder + masked
predictor); it has no ``VisionTransformerPredictorAC`` class, no
actions/states inputs, and no AC checkpoint on Hub. The upstream repo at
``facebookresearch/vjepa2`` ships the AC predictor (``src/models/
ac_predictor.py``), the bundled encoder+predictor checkpoint
(``vjepa2-ac-vitg.pt`` on FAIR S3 — the same ``.pt`` mminf loads), and a
canonical AC rollout pattern in ``notebooks/energy_landscape_example.ipynb``
Cell 5 / Cell 6.

What this script benchmarks
---------------------------
Every model-logic line in this file is either:
  (a) an ``import`` from the upstream vjepa2 repo,
  (b) a call into upstream's exported ``torch.hub`` entrypoint
      ``vjepa2_ac_vit_giant``, or
  (c) a byte-for-byte verbatim copy of a named cell in
      ``energy_landscape_example.ipynb``, fenced with explicit
      ``BEGIN/END VERBATIM COPY`` markers and source citation.

The only non-upstream code is benchmarking scaffolding (argparse, timing,
JSON serialization, env-check, video decode wrapper).

Env requirement
---------------
This script requires the dedicated ``vjepa2_repo`` conda env (separate
from mminf — torch and torchcodec versions diverge). See the
implementation plan at ``vjepa2_ac_bench_implementation_plan.md`` for the
full env setup. Quick start:

    conda activate vjepa2_repo
    export VJEPA2_REPO=$HOME/vjepa2
    python benchmark/vjepa2_repo_ac.py --num-requests 5 --output-dir /tmp/vjepa2_repo_ac

The script does an import check at startup and prints the activation
command if upstream isn't reachable.

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
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Allow ``from benchmark.dataset import DROIDDataset`` when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Architecture constants. Both come from upstream's
# ``_make_vjepa2_ac_model`` defaults (img_size=256, num_frames=64,
# tubelet_size=2). ``MAX_T`` is the predictor's grid_depth — the attention
# mask is sized for this many frames, so the rollout cannot exceed it.
#
# DEFAULT_NUM_FRAMES=8 matches upstream's published DROID training config
# (configs/train/vitg16/droid-256px-8f.yaml: dataset_fpcs: 8). Each frame is
# encoded independently via the self-tubelet replication trick (matches the
# `forward_target` pattern in app/vjepa_droid/train.py:408-415 and notebook
# Cell 5). Override via the --num-frames CLI flag if you want a different
# encoder workload (e.g. --num-frames 64 reproduces the energy_landscape
# notebook's default).
CROP_SIZE = 256
DEFAULT_NUM_FRAMES = 8
TUBELET_SIZE = 2
MAX_T = 32     # = predictor grid_depth (architecture default num_frames=64 // tubelet=2)
               # rollout_horizon must be < MAX_T


@dataclass
class PerRequestResult:
    request_id: int
    jct_ms: float
    encoder_ms: float            # diagnostic — encoder forward only
    rollout_ms: float            # diagnostic — H predictor forwards
    n_rollout_steps: int
    output_shape: list[int]
    finite: bool


@dataclass
class BenchmarkResult:
    system: str = "vjepa2_repo"
    model: str = "vjepa2_ac"
    checkpoint: str = "vjepa2-ac-vitg"   # bundled encoder+predictor; FAIR S3
    repo_path: str = ""
    upstream_commit: str = ""            # git SHA of vjepa2 repo at run time
    num_requests: int = 0
    num_warmup: int = 0
    completed: int = 0
    failed: int = 0
    rollout_horizon: int = 4
    # JCT (E2E latency, externally timed) stats (ms)
    jct_mean_ms: float = 0.0
    jct_median_ms: float = 0.0
    jct_std_ms: float = 0.0
    jct_p90_ms: float = 0.0
    jct_p95_ms: float = 0.0
    jct_p99_ms: float = 0.0
    # Per-stage diagnostics
    encoder_mean_ms: float = 0.0
    rollout_mean_ms: float = 0.0
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


def _check_env_or_exit(repo_path: str) -> None:
    """Verify upstream vjepa2 imports work; bail with a friendly conda hint."""
    if not os.path.isdir(repo_path):
        sys.exit(
            f"\n[ERROR] vjepa2 repo path does not exist: {repo_path}\n\n"
            f"Set VJEPA2_REPO or pass --vjepa2-repo to the upstream clone.\n"
            f"To clone upstream: git clone https://github.com/facebookresearch/vjepa2.git\n"
        )
    sys.path.insert(0, repo_path)
    try:
        import torch  # noqa: F401
        from app.vjepa_droid.transforms import make_transforms  # noqa: F401
        from src.models.ac_predictor import VisionTransformerPredictorAC  # noqa: F401
        from notebooks.utils.mpc_utils import compute_new_pose  # noqa: F401
    except ImportError as e:
        sys.exit(
            f"\n[ERROR] vjepa2 upstream imports failed ({e}).\n\n"
            f"This script requires the dedicated vjepa2_repo conda env:\n\n"
            f"    conda activate vjepa2_repo\n"
            f"    python benchmark/vjepa2_repo_ac.py --vjepa2-repo {repo_path} ...\n\n"
            f"If the env isn't set up yet, see the implementation plan in\n"
            f"multimodal_inference/vjepa2_ac_bench_implementation_plan.md.\n"
        )


def _load_video_clip(video_path: str, num_frames: int = DEFAULT_NUM_FRAMES) -> np.ndarray:
    """Decode the mp4 to a [num_frames, H, W, 3] uint8 numpy array.

    DROIDDataset(task='vjepa2_ac') produces an episode-clipped mp4 with
    ``num_video_frames`` evenly-sampled frames per episode (default 8 for the
    DROID-config-aligned comparison; see ``benchmark/dataset.py:_make_vjepa2_ac``).
    We resample to exactly ``num_frames`` here so the encoder always sees a
    deterministic frame count regardless of decoder rounding.
    """
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(video_path)
    n_total = dec.metadata.num_frames or len(dec)
    if n_total <= 0:
        raise RuntimeError(f"VideoDecoder reported 0 frames for {video_path}")
    # Evenly sample ``num_frames`` indices, repeating the last frame if the
    # episode is too short. Matches the spirit of DROIDDataset's encode-side
    # sampling but enforced again here so the encoder sees exactly 64 frames.
    indices = np.linspace(0, n_total - 1, num=num_frames).astype(int).tolist()
    batch = dec.get_frames_at(indices=indices)
    arr = batch.data.permute(0, 2, 3, 1).numpy()  # [T, H, W, C] uint8
    return arr


def _capture_upstream_commit(repo_path: str, cli_override: str | None) -> str:
    """Return the upstream git SHA for paper provenance."""
    if cli_override:
        return cli_override
    try:
        return subprocess.check_output(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upstream vjepa2 AC rollout baseline on DROID")
    p.add_argument("--num-requests", type=int, default=10)
    p.add_argument("--num-warmup", type=int, default=3,
                   help="Warmup requests, identical cadence to runner.py defaults.")
    p.add_argument("--rollout-horizon", type=int, default=4,
                   help=f"Number of AC rollout steps. Must be < {MAX_T} "
                        f"(predictor attention mask is sized for {MAX_T} frames).")
    p.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                   help=f"Number of video frames per request to feed the encoder via "
                        f"self-tubelet replication. Default {DEFAULT_NUM_FRAMES} matches "
                        f"upstream's published DROID training config "
                        f"(configs/train/vitg16/droid-256px-8f.yaml: dataset_fpcs: 8).")
    p.add_argument("--vjepa2-repo",
                   default=os.environ.get("VJEPA2_REPO", str(Path.home() / "vjepa2")),
                   help="Path to the upstream vjepa2 repo clone (default $VJEPA2_REPO or ~/vjepa2).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Inference dtype. Upstream notebook defaults to fp32; "
                        "we default to bf16 to match deployment.")
    p.add_argument("--output-dir", default="/tmp/vjepa2_repo_ac")
    p.add_argument("--local-cache", default="./mminf-benchmark-cache/",
                   help="Same default as runner.py so DROIDDataset reuses extracted clips.")
    p.add_argument("--hf-cache", default=None,
                   help="HuggingFace cache directory for lerobot/droid_100.")
    p.add_argument("--normalize-reps", type=int, default=1,
                   help="Apply LayerNorm after each predictor output (matches notebook Cell 5).")
    p.add_argument("--upstream-commit", default=None,
                   help="Optional: git SHA of the vjepa2 repo at run time. "
                        "Logged into results.json for paper provenance.")
    return p.parse_args()


def main() -> None:
    # argparse first so --help works even before env-check.
    args = parse_args()

    # Hard-fail early on the architecture constraint instead of producing
    # a confusing torch shape error deep in the rollout loop.
    if args.rollout_horizon < 1 or args.rollout_horizon >= MAX_T:
        sys.exit(
            f"\n[ERROR] --rollout-horizon must be in [1, {MAX_T - 1}]; "
            f"got {args.rollout_horizon}. The AC predictor's attention mask "
            f"is sized for grid_depth={MAX_T} (architecture default num_frames=64, "
            f"tubelet_size={TUBELET_SIZE}).\n"
        )
    if args.num_frames < 1:
        sys.exit(f"\n[ERROR] --num-frames must be >= 1; got {args.num_frames}.\n")

    _check_env_or_exit(args.vjepa2_repo)
    upstream_sha = _capture_upstream_commit(args.vjepa2_repo, args.upstream_commit)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.local_cache, exist_ok=True)

    # Heavy imports deferred until after env-check so the friendly message
    # fires before any model loads. Upstream repo is already on sys.path
    # via _check_env_or_exit().
    import torch
    import torch.nn.functional as F
    from app.vjepa_droid.transforms import make_transforms
    from notebooks.utils.mpc_utils import compute_new_pose
    from benchmark.dataset import DROIDDataset

    print("=== upstream vjepa2 AC baseline ===")
    print(f"  repo         : {args.vjepa2_repo}")
    print(f"  upstream SHA : {upstream_sha or '(unknown)'}")
    print(f"  device       : {args.device}")
    print(f"  dtype        : {args.dtype}")
    print(f"  num_requests : {args.num_requests}")
    print(f"  num_warmup   : {args.num_warmup}")
    print(f"  rollout_H    : {args.rollout_horizon}")
    print(f"  num_frames   : {args.num_frames}  (encoder workload per request)")

    # ------------------------------------------------------------------
    # Model + transform load (mirrors notebook Cell 2)
    # ------------------------------------------------------------------
    print("\nLoading vjepa2_ac_vit_giant via torch.hub "
          "(downloads ~12GB on first run)...")
    t0 = time.perf_counter()
    encoder, predictor = torch.hub.load(
        args.vjepa2_repo, "vjepa2_ac_vit_giant", source="local"
    )
    encoder = encoder.to(args.device).eval()
    predictor = predictor.to(args.device).eval()
    tokens_per_frame = int((CROP_SIZE // encoder.patch_size) ** 2)
    print(f"  loaded encoder + AC predictor ({time.perf_counter() - t0:.1f}s)")
    print(f"  tokens_per_frame = {tokens_per_frame}")

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1., 1.),
        random_resize_scale=(1., 1.),
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=CROP_SIZE,
    )

    # =========================================================================
    # === BEGIN VERBATIM COPY ===
    # Source: vjepa2/notebooks/energy_landscape_example.ipynb, Cell 5.
    # Module-level helpers `forward_target` (encoder pass) and
    # `step_predictor` (one rollout step) are reproduced byte-for-byte from
    # the notebook so this benchmark exercises upstream's published pattern
    # without modification. `normalize_reps` is captured from the CLI flag
    # (default 1, matches notebook Cell 5).
    # `encoder`, `predictor`, `tokens_per_frame`, `compute_new_pose` are
    # imported/loaded above.
    # =========================================================================
    normalize_reps = bool(args.normalize_reps)

    @torch.inference_mode()
    def forward_target(c, normalize_reps=normalize_reps):
        B, C, T, H, W = c.size()
        c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = encoder(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    @torch.inference_mode()
    def step_predictor(_z, _a, _s):
        _z = predictor(_z, _a, _s)[:, -tokens_per_frame:]
        if normalize_reps:
            _z = F.layer_norm(_z, (_z.size(-1),))
        _s = compute_new_pose(_s[:, -1:], _a[:, -1:])
        return _z, _s
    # === END VERBATIM COPY ===

    # ------------------------------------------------------------------
    # Dataset (same DROIDDataset as our HTTP harness)
    # ------------------------------------------------------------------
    print(f"\nBuilding DROIDDataset (task=vjepa2_ac, n={args.num_requests}, "
          f"num_video_frames={args.num_frames})...")
    dataset = DROIDDataset(
        local_file_dir=args.local_cache,
        num_requests=args.num_requests,
        task="vjepa2_ac",
        rollout_horizon=args.rollout_horizon,
        cache_dir=args.hf_cache,
        num_video_frames=args.num_frames,
    )
    requests = dataset.get_requests() if hasattr(dataset, "get_requests") else list(dataset)
    if not requests:
        sys.exit("ERROR: DROIDDataset returned 0 requests")
    print(f"  loaded {len(requests)} episodes")

    # =========================================================================
    # === BEGIN VERBATIM COPY (driver pattern) ===
    # Source: vjepa2/notebooks/energy_landscape_example.ipynb, Cell 3 (input
    # prep) + Cell 6 (encode call) + Cell 5 `forward_actions` (rollout body).
    # Only difference vs. upstream: per-step DROID action via slicing
    # `actions_t[:, n+1:n+2]` instead of appending a constant
    # `action_samples` (the notebook holds the action constant since it
    # sweeps a grid; we have a real action trajectory from DROIDDataset).
    # The for-loop structure, the history-growing
    # `a_hat = torch.cat([a_hat, ...], dim=1)` pattern, and the tensor
    # concatenation order are unchanged from the notebook.
    # The growing-history pattern is *required* by
    # `vjepa2/src/models/ac_predictor.py:VisionTransformerPredictorAC.forward`
    # — that forward derives T from x.size(1) and concatenates [a, s, x]
    # along dim=2, so a/s must have the same T as x at every step.
    # =========================================================================
    def _prepare_request_inputs(req):
        np_clip = _load_video_clip(req.video_path, num_frames=args.num_frames)
        # Cell 3 verbatim: transform(np_clip).unsqueeze(0) → [1, C, T, H, W]
        clip_t = transform(np_clip).unsqueeze(0).to(args.device, non_blocking=True)
        actions_t = torch.tensor(
            req.model_kwargs["actions"], dtype=torch.float32, device=args.device,
        ).unsqueeze(0)
        states_t = torch.tensor(
            req.model_kwargs["states"], dtype=torch.float32, device=args.device,
        ).unsqueeze(0)
        # Sanity: the dataset produces n_actions = t_ctx + H - 1 = 31 + H
        # actions per episode (see _make_vjepa2_ac). The upstream baseline
        # only consumes the first ``rollout_horizon`` of them (Cell 5's
        # forward_actions starts with a single initial action and appends
        # one per iteration). Excess entries are intentionally unused —
        # mminf may consume the full sequence on its side.
        if actions_t.size(1) < args.rollout_horizon:
            raise RuntimeError(
                f"req has only {actions_t.size(1)} actions; need >= "
                f"{args.rollout_horizon} for the rollout horizon."
            )
        return clip_t, actions_t, states_t

    def _run_rollout(clip_t, actions_t, states_t):
        """Cell 6 verbatim: h = forward_target(clips) → rollout body from Cell 5."""
        h = forward_target(clip_t)
        z_hat = h[:, :tokens_per_frame]      # 1 frame of context (notebook convention)
        s_hat = states_t[:, :1]
        a_hat = actions_t[:, :1]
        preds = []
        for n in range(args.rollout_horizon):
            _z, _s = step_predictor(z_hat, a_hat, s_hat)
            z_hat = torch.cat([z_hat, _z], dim=1)
            s_hat = torch.cat([s_hat, _s], dim=1)
            preds.append(_z)
            if n + 1 < args.rollout_horizon:
                a_hat = torch.cat([a_hat, actions_t[:, n + 1:n + 2]], dim=1)
        return h, preds
    # === END VERBATIM COPY ===

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    print(f"\nWarmup ({args.num_warmup} requests)...")
    if args.num_warmup > 0 and requests:
        for _ in range(args.num_warmup):
            clip_t, actions_t, states_t = _prepare_request_inputs(requests[0])
            _run_rollout(clip_t, actions_t, states_t)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    print("  done")

    # ------------------------------------------------------------------
    # Timed loop. Same body as _run_rollout() inlined here so timing is
    # exact (no helper-call overhead between perf_counter calls).
    # ------------------------------------------------------------------
    print(f"\nRunning benchmark ({len(requests)} requests, sequential)...")
    per_request: list[PerRequestResult] = []
    failed = 0
    wall_start = time.monotonic()
    for i, req in enumerate(requests):
        try:
            clip_t, actions_t, states_t = _prepare_request_inputs(req)

            t0 = time.perf_counter()
            h = forward_target(clip_t)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_enc_done = time.perf_counter()

            # Cell 5 `forward_actions` rollout body, inlined for timing precision.
            z_hat = h[:, :tokens_per_frame]
            s_hat = states_t[:, :1]
            a_hat = actions_t[:, :1]
            preds = []
            for n in range(args.rollout_horizon):
                _z, _s = step_predictor(z_hat, a_hat, s_hat)
                z_hat = torch.cat([z_hat, _z], dim=1)
                s_hat = torch.cat([s_hat, _s], dim=1)
                preds.append(_z)
                if n + 1 < args.rollout_horizon:
                    a_hat = torch.cat([a_hat, actions_t[:, n + 1:n + 2]], dim=1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_done = time.perf_counter()

            jct_ms = (t_done - t0) * 1000.0
            encoder_ms = (t_enc_done - t0) * 1000.0
            rollout_ms = (t_done - t_enc_done) * 1000.0

            preds_arr = torch.cat(preds, dim=1).float().cpu().numpy()
            np.save(
                os.path.join(args.output_dir, f"req_{i:02d}_latents.npy"),
                preds_arr,
            )

            per_request.append(PerRequestResult(
                request_id=i, jct_ms=jct_ms,
                encoder_ms=encoder_ms, rollout_ms=rollout_ms,
                n_rollout_steps=args.rollout_horizon,
                output_shape=list(preds_arr.shape),
                finite=bool(np.isfinite(preds_arr).all()),
            ))
            print(f"  req {i:02d}: jct={jct_ms:.1f} ms "
                  f"(enc={encoder_ms:.1f}, rollout={rollout_ms:.1f}) "
                  f"shape={preds_arr.shape} finite={per_request[-1].finite}")
        except Exception as e:
            failed += 1
            print(f"  req {i:02d}: FAILED — {e}")
    wall_time = time.monotonic() - wall_start

    # ------------------------------------------------------------------
    # Aggregate + write JSON (schema matches benchmark/openpi_pi05.py)
    # ------------------------------------------------------------------
    jcts = [r.jct_ms for r in per_request]
    result = BenchmarkResult(
        repo_path=args.vjepa2_repo,
        upstream_commit=upstream_sha,
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
        result.encoder_mean_ms = statistics.mean(r.encoder_ms for r in per_request)
        result.rollout_mean_ms = statistics.mean(r.rollout_ms for r in per_request)
        total_steps = sum(r.n_rollout_steps for r in per_request)
        total_secs = sum(jcts) / 1000.0
        result.rollout_steps_per_sec = (total_steps / total_secs) if total_secs else 0.0
        result.request_throughput = (len(per_request) / wall_time) if wall_time else 0.0

    out_json = os.path.join(args.output_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump({**asdict(result),
                   "per_request": [asdict(r) for r in per_request]},
                  f, indent=2)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print(f"\n=== Results (wall {wall_time:.1f}s, "
          f"{len(per_request)}/{len(requests)} ok) ===")
    if jcts:
        print(f"  JCT mean      : {result.jct_mean_ms:.1f} ms")
        print(f"  JCT median    : {result.jct_median_ms:.1f} ms")
        print(f"  JCT p95       : {result.jct_p95_ms:.1f} ms")
        print(f"  JCT p99       : {result.jct_p99_ms:.1f} ms")
        print(f"  encoder mean  : {result.encoder_mean_ms:.1f} ms")
        print(f"  rollout mean  : {result.rollout_mean_ms:.1f} ms")
        print(f"  Throughput    : {result.request_throughput:.2f} req/s, "
              f"{result.rollout_steps_per_sec:.2f} rollout-steps/s")
    print(f"  Outputs       : {args.output_dir}/")
    print(f"  Results       : {out_json}")


if __name__ == "__main__":
    main()
