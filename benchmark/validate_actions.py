#!/usr/bin/env python3
"""Validate and plot pi0.5 action trajectories saved by run_robotics.py.

Each *_actions.npy file contains a [T, action_dim] float32 array of predicted
joint / end-effector deltas over the action horizon.

Usage:
    python benchmark/validate_actions.py /tmp/robotics_benchmark/req_00_actions.npy
    python benchmark/validate_actions.py /tmp/robotics_benchmark/   # all files
    python benchmark/validate_actions.py /tmp/robotics_benchmark/ --plot  # save PNGs
"""

from __future__ import annotations
import sys
import os
import numpy as np


def validate_one(path: str) -> dict:
    arr = np.load(path)                  # [T, action_dim]
    if arr.ndim == 1:
        # flat bytes → try to reshape to (T, 32)
        action_dim = 32
        arr = arr[: (arr.size // action_dim) * action_dim].reshape(-1, action_dim)
    T, D = arr.shape

    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())

    abs_max  = float(np.abs(arr).max())
    mean_abs = float(np.abs(arr).mean())

    # Smoothness: mean absolute first-difference across time
    diffs   = np.abs(np.diff(arr, axis=0))        # [T-1, D]
    smooth  = float(diffs.mean())

    # Per-dimension range (useful for spotting stuck / saturated dims)
    dim_range = (arr.max(axis=0) - arr.min(axis=0)).tolist()

    return {
        "shape":     (T, D),
        "finite":    n_nan == 0 and n_inf == 0,
        "n_nan":     n_nan,
        "n_inf":     n_inf,
        "abs_max":   abs_max,
        "mean_abs":  mean_abs,
        "smoothness": smooth,   # lower = smoother trajectory
        "dim_range": dim_range,
        "arr":       arr,
    }


def report(path: str, r: dict) -> None:
    ok   = "✓" if r["finite"] else "✗"
    name = os.path.basename(path)
    print(f"\n{name}  {r['shape']}")
    print(f"  finite      {ok}  (nan={r['n_nan']}, inf={r['n_inf']})")
    print(f"  abs_max     {r['abs_max']:.4f}")
    print(f"  mean_abs    {r['mean_abs']:.4f}")
    print(f"  smoothness  {r['smoothness']:.6f}  (mean |Δ| per step; lower = smoother)")

    active_dims = [i for i, rng in enumerate(r["dim_range"]) if rng > 1e-4]
    zero_dims   = [i for i, rng in enumerate(r["dim_range"]) if rng <= 1e-4]
    print(f"  active dims {active_dims}  (range > 1e-4)")
    if zero_dims:
        print(f"  zero dims   {zero_dims}  (constant throughout — likely padding)")

    issues = []
    if not r["finite"]:
        issues.append("NaN/Inf values")
    if r["abs_max"] > 100:
        issues.append(f"very large actions (abs_max={r['abs_max']:.1f}) — check normalisation")
    if r["abs_max"] < 1e-5:
        issues.append("near-zero actions — model may have collapsed")
    if r["smoothness"] > 1.0:
        issues.append(f"very jerky trajectory (mean |Δ|={r['smoothness']:.3f})")
    if not active_dims:
        issues.append("all dimensions are zero — no motion predicted")

    if issues:
        print(f"  WARNINGS: {'; '.join(issues)}")
    else:
        print(f"  PASS — actions look plausible")


def plot_one(path: str, r: dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib not installed")
        return

    arr = r["arr"]           # [T, D]
    T, D = arr.shape

    # Only plot dims with non-trivial range
    active = [i for i, rng in enumerate(r["dim_range"]) if rng > 1e-4]
    if not active:
        active = list(range(min(8, D)))

    fig, axes = plt.subplots(len(active), 1,
                              figsize=(10, 1.5 * len(active)),
                              sharex=True)
    if len(active) == 1:
        axes = [axes]

    t = np.arange(T)
    for ax, dim in zip(axes, active):
        ax.plot(t, arr[:, dim], linewidth=1.2)
        ax.set_ylabel(f"dim {dim}", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timestep")
    fig.suptitle(os.path.basename(path), fontsize=10)
    fig.tight_layout()

    stem = os.path.splitext(os.path.basename(path))[0]
    out  = os.path.join(out_dir, f"{stem}_plot.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  plot → {out}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    do_plot = "--plot" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = args[0]

    paths = []
    if os.path.isdir(target):
        paths = sorted(
            os.path.join(target, f)
            for f in os.listdir(target)
            if f.endswith("_actions.npy")
        )
        if not paths:
            print(f"No *_actions.npy files found in {target}")
            sys.exit(1)
    else:
        paths = [target]

    for path in paths:
        try:
            r = validate_one(path)
            report(path, r)
            if do_plot:
                plot_one(path, r, os.path.dirname(path) or ".")
        except Exception as e:
            print(f"\n{os.path.basename(path)}: ERROR — {e}")

    print()


if __name__ == "__main__":
    main()
