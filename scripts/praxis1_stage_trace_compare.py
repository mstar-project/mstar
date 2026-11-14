"""Find the first numerical divergence in a paired cold Praxis request.

Both inputs are local diagnostic artifacts from the same image, prompt and
seed. The script reports stage differences and exits nonzero when a required
stage is absent; it does not define a policy-quality threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    if actual.shape != expected.shape:
        return {"stage": name, "actual_shape": list(actual.shape),
                "reference_shape": list(expected.shape), "shape_match": False}
    diff = (actual - expected).abs()
    return {"stage": name, "shape": list(actual.shape),
            "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "relative_l2": float(torch.linalg.vector_norm(diff) /
                                 torch.linalg.vector_norm(expected).clamp_min(1e-12)),
            "fraction_close_3pct": float(torch.isclose(actual, expected, rtol=.03, atol=.03).float().mean()),
            "exact": bool(torch.equal(actual, expected))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mstar-trace-dir", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    args = parser.parse_args()
    artifacts = [torch.load(path, map_location="cpu", weights_only=False)
                 for path in sorted(args.mstar_trace_dir.glob("*.pt"))]
    by_stage = {}
    for item in artifacts:
        by_stage.setdefault(item["stage"], []).append(item)
    for stage, count in (
        ("Praxis1TextEncoderSubmodule", 1),
        ("Praxis1VaeEncoderSubmodule", 1),
        ("Praxis1BackboneSubmodule", 2),
        ("Praxis1ActionExpertSubmodule", 2),
        ("Praxis1ActionHeadSubmodule", 1),
    ):
        if len(by_stage.get(stage, [])) != count:
            raise SystemExit(f"expected {count} {stage} calls; got {len(by_stage.get(stage, []))}")
    ref = torch.load(args.reference_trace, map_location="cpu", weights_only=False)
    events = {}
    for name, value in ref["events"]:
        events.setdefault(name, []).append(value)
    text = by_stage["Praxis1TextEncoderSubmodule"][0]
    vae = by_stage["Praxis1VaeEncoderSubmodule"][0]
    backbones = sorted(by_stage["Praxis1BackboneSubmodule"],
                       key=lambda item: item["inputs"].get("step_index", 0))
    experts = sorted(by_stage["Praxis1ActionExpertSubmodule"],
                     key=lambda item: item["inputs"].get("step_index", 0))
    head = by_stage["Praxis1ActionHeadSubmodule"][0]
    checks = [
        ("text_task", text["outputs"]["text_embeds"][0], ref["inputs"]["text_embedding"]),
        ("vae_anchor", vae["outputs"]["obs_latents"][0], ref["inputs"]["anchor"]),
        ("expert_proprio", experts[0]["inputs"]["proprio"], ref["inputs"]["proprio"]),
        ("video_noise", backbones[0]["inputs"]["video_latents"], events["noise.salt0"][0]),
        ("action_noise", backbones[0]["inputs"]["action_latents"],
         events["noise.salt1"][0].unsqueeze(0)),
        ("video_after_step0", backbones[0]["outputs"]["expert_video_latents"][0],
         events["velocity.in.xt_v"][2]),
        ("action_after_step0", experts[0]["outputs"]["action_latents"][0],
         events["velocity.in.xt_a"][2].unsqueeze(0)),
        ("video_loopback", backbones[1]["inputs"]["video_latents"],
         backbones[0]["outputs"]["expert_video_latents"][0]),
        ("action_loopback", backbones[1]["inputs"]["action_latents"],
         experts[0]["outputs"]["action_latents"][0]),
        ("action_after_step1", head["inputs"]["action_latents"], ref["pred_action_norm"]),
        ("relative_actions", head["outputs"]["actions"][0],
         torch.as_tensor(np.asarray(ref["action_rel"]))),
    ]
    for name, actual, expected in checks:
        print(json.dumps(compare(name, actual, expected)), flush=True)


if __name__ == "__main__":
    main()
