#!/usr/bin/env python3
"""Validate VJepa2-AC rollout latents saved by run_robotics.py.

Usage:
    python benchmark/validate_latents.py /tmp/robotics_benchmark/req_00_latents.npy
    python benchmark/validate_latents.py /tmp/robotics_benchmark/  # all .npy files
"""

from __future__ import annotations
import sys
import os
import numpy as np


def validate_one(path: str) -> dict:
    arr = np.load(path)                          # [H, N, D]
    H, N, D = arr.shape

    # 1. Finiteness
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())

    # 2. Per-step token-mean L2 norm
    norms = np.linalg.norm(arr, axis=-1).mean(axis=-1)   # [H] — mean over N tokens

    # 3. Cross-step cosine similarity between consecutive steps
    flat = arr.reshape(H, -1)                             # [H, N*D]
    flat_norm = flat / (np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-8)
    cosine_sims = [(flat_norm[t] * flat_norm[t + 1]).sum()
                   for t in range(H - 1)]

    # 4. Variance across steps (how much the prediction changes)
    step_var = arr.var(axis=0).mean()                    # mean token-wise variance across steps

    return {
        "shape":        (H, N, D),
        "finite":       n_nan == 0 and n_inf == 0,
        "n_nan":        n_nan,
        "n_inf":        n_inf,
        "norms":        norms.tolist(),
        "cosine_sims":  [round(float(c), 4) for c in cosine_sims],
        "step_variance": float(step_var),
    }


def report(path: str, r: dict) -> None:
    ok   = "✓" if r["finite"] else "✗"
    name = os.path.basename(path)
    print(f"\n{name}  {r['shape']}")
    print(f"  finite          {ok}  (nan={r['n_nan']}, inf={r['n_inf']})")
    print(f"  per-step norms  {[round(n, 3) for n in r['norms']]}")
    print(f"  cosine(t→t+1)   {r['cosine_sims']}  (expect ~0.90+ for slow scene)")
    print(f"  step variance   {r['step_variance']:.6f}  (>0 means predictor is changing state)")

    # Heuristic pass/fail
    issues = []
    if not r["finite"]:
        issues.append("NaN/Inf values")
    if any(n < 0.1 or n > 1000 for n in r["norms"]):
        issues.append(f"unusual norms {[round(n,2) for n in r['norms']]}")
    if r["cosine_sims"] and max(r["cosine_sims"]) < 0.3:
        issues.append(f"very low cross-step cosine sim {r['cosine_sims']} — latents look random")
    if r["step_variance"] < 1e-6:
        issues.append("zero step variance — predictor output is constant (degenerate)")

    if issues:
        print(f"  WARNINGS: {'; '.join(issues)}")
    else:
        print(f"  PASS — latents look plausible")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    paths = []
    if os.path.isdir(target):
        paths = sorted(
            os.path.join(target, f)
            for f in os.listdir(target)
            if f.endswith("_latents.npy")
        )
        if not paths:
            print(f"No *_latents.npy files found in {target}")
            sys.exit(1)
    else:
        paths = [target]

    for path in paths:
        try:
            r = validate_one(path)
            report(path, r)
        except Exception as e:
            print(f"\n{os.path.basename(path)}: ERROR — {e}")

    print()


if __name__ == "__main__":
    main()
