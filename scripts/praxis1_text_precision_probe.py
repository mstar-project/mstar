"""Isolate the live M* versus reference Qwen text embedding difference.

Load Qwen once from the same local weights, reuse the exact M* token IDs,
and sweep only autocast and float32 matmul precision. This keeps the DROID
backbone, VAE and simulator out of the probe.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from praxis.text_encoder import TextEncoder


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, b = actual.float().cpu(), expected.float().cpu()
    if a.shape != b.shape:
        return {"same_shape": False, "actual_shape": list(a.shape), "expected_shape": list(b.shape)}
    delta = (a - b).abs()
    return {"same_shape": True, "max_abs": float(delta.max()),
            "relative_l2": float(torch.linalg.vector_norm(delta) /
                                 torch.linalg.vector_norm(b).clamp_min(1e-12)),
            "exact": bool(torch.equal(a, b))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--mstar-stage-dir", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "this probe needs the same GPU as the live calls"
    paths = list(args.mstar_stage_dir.glob("*-Praxis1TextEncoderSubmodule.pt"))
    if len(paths) != 1:
        raise ValueError(f"expected one M* text trace, got {len(paths)}")
    stage = torch.load(paths[0], map_location="cpu", weights_only=False)
    ref = torch.load(args.reference_trace, map_location="cpu", weights_only=False)
    ids = stage["inputs"]["text_ids"].to("cuda")[None]
    mask = stage["inputs"]["text_mask"].to("cuda")[None]
    mstar = stage["outputs"]["text_embeds"][0]
    reference = ref["inputs"]["text_embedding"]
    print(json.dumps({"phase": "environment", "torch": torch.__version__,
                      "transformers": importlib.metadata.version("transformers"),
                      "fla": _version_or_none("flash-linear-attention"),
                      "causal_conv1d": _version_or_none("causal-conv1d"),
                      "mstar_vs_reference": metrics(mstar, reference)}), flush=True)
    encoder = TextEncoder(args.qwen, device="cuda")
    print(json.dumps({"phase": "model", "dtype": str(encoder.model.dtype),
                      "kernel_configs": encoder.kernel_configs}), flush=True)
    original_precision = torch.get_float32_matmul_precision()
    try:
        for precision in ("highest", "high"):
            torch.set_float32_matmul_precision(precision)
            for autocast in (False, True):
                with torch.inference_mode(), torch.autocast("cuda", enabled=autocast,
                                                            dtype=torch.bfloat16):
                    out = encoder.model(
                        input_ids=ids, attention_mask=mask,
                        output_hidden_states=True, use_cache=False,
                    ).hidden_states[-3][:, :int(mask.sum())]
                torch.cuda.synchronize()
                print(json.dumps({"phase": "text_forward", "precision": precision,
                                  "autocast": autocast, "dtype": str(out.dtype),
                                  "vs_mstar": metrics(out, mstar),
                                  "vs_reference": metrics(out, reference)}), flush=True)
    finally:
        torch.set_float32_matmul_precision(original_precision)


def _version_or_none(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
