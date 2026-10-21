"""Audit saved reference comparisons. A failed comparison exits nonzero."""

import argparse
import json
from pathlib import Path

import numpy as np


def audit(directory):
    root = Path(directory)
    summary = json.loads((root / "summary.json").read_text())
    with np.load(root / "pipeline.npz") as arrays:
        ours, reference = arrays["mstar_agent"], arrays["nvidia_agent"]
        frames = np.flatnonzero(np.any(ours != reference, axis=(0, 1))).tolist()
        text_frames = np.flatnonzero(np.any(arrays["mstar_text"] != arrays["nvidia_text"], axis=1)).tolist()
        emitted = sorted(set(frames) | set(text_frames))
        counter_matches = emitted == summary["checks"]["output_frame_mismatches"]
        difference = arrays["mstar_pcm"].astype(np.float64) - arrays["nvidia_pcm"].astype(np.float64)
        finite = bool(np.isfinite(arrays["mstar_pcm"]).all() and np.isfinite(arrays["nvidia_pcm"]).all())
        return {
            "status": "PASS"
            if counter_matches and not emitted and finite and summary["checks"]["user_codes"]["different"] == 0
            else "FAIL",
            "agent_code_differences": int(np.count_nonzero(ours != reference)),
            "agent_frames": frames,
            "agent_frame_count": len(frames),
            "text_frames": text_frames,
            "event_frame_count": len(summary["checks"]["output_frame_mismatches"]),
            "event_counter_matches_arrays": counter_matches,
            "finite": finite,
            "float_audio": {
                "max_abs": float(np.abs(difference).max()),
                "mse": float(np.mean(difference**2)),
                "rmse": float(np.sqrt(np.mean(difference**2))),
            },
            "first_decision": summary["checks"].get("first_decision"),
            "scope": "Exact token trajectory check; float audio metrics are descriptive, not quality acceptance.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
