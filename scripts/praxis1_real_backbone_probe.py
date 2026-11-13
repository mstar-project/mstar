"""Run the published DROID encoder and decoder on one small cold chunk.

This is a bounded real-weight graph probe, not a serving or latency benchmark.
The two-token video grid exercises all blocks while keeping attention small.
The reference and port run sequentially so their weights do not coexist on GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from praxis.checkpoint import load_config, load_state_dict_from_dir, strict_load
from praxis.model import PraxisPolicy

from mstar.model.praxis1.components.chunk_mask import encoder_codes, visible
from mstar.model.praxis1.components.encoder import dense_chunk_causal_mask
from mstar.model.praxis1.config import Praxis1Config
from mstar.model.praxis1.native_towers import build_backbone
from mstar.model.praxis1.weight_loader import load_component


def _inputs(config: Praxis1Config) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(109)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            device="cuda", dtype=torch.bfloat16
        )

    return (
        rand(1, 3, config.text_hidden_dim),
        rand(1, config.latent_channels * 4, 1, 1, 2),
        rand(1, config.latent_channels * 4, config.latents_per_chunk, 1, 2),
        torch.full((1, 1), 0.5, device="cuda", dtype=torch.bfloat16),
    )


def _run_reference(root: Path, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.device("meta"):
        policy = PraxisPolicy.from_config(load_config(root))
    for name in ("encoder", "decoder"):
        module = getattr(policy.backbone, name)
        weights = load_state_dict_from_dir(root, prefix=f"backbone.{name}.", device="cpu")
        strict_load(module, weights, assign=True)
        del weights
    backbone = policy.backbone.eval().requires_grad_(False).to("cuda")
    del policy
    started = time.perf_counter()
    with torch.inference_mode():
        text, anchor, noised, timestep = inputs
        out = backbone.forward_chunk_causal(text, anchor, None, noised, timestep, decode=True)
    torch.cuda.synchronize()
    print(json.dumps({"phase": "reference", "seconds": round(time.perf_counter() - started, 1),
                      "velocity_shape": list(out.velocity.shape)}), flush=True)
    cond = out.velocity.detach().cpu()
    tap = out.taps.taps[-1].detach().cpu()
    del out, backbone
    gc.collect()
    torch.cuda.empty_cache()
    return cond, tap


def _run_native(
    root: Path, config: Praxis1Config, inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.device("meta"):
        backbone = build_backbone(config)
    for name in ("encoder", "decoder"):
        load_component(getattr(backbone, name), root, f"backbone.{name}.", assign=True)
    backbone = backbone.eval().requires_grad_(False).to("cuda")
    text, anchor, noised, timestep = inputs
    # Four register tokens, two anchor tokens, three text tokens, then the
    # six noised frames with two spatial tokens each.
    enc_mask = dense_chunk_causal_mask(
        config.num_registers, 2, text.shape[1], config.latents_per_chunk * 2, 1, 0
    ).to("cuda")
    dec_mask = visible(encoder_codes(
        config.num_registers, 0, 0,
        config.latents_per_chunk * 2 * config.patch_size[1] * config.patch_size[2], 1, 0,
    )).to("cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        cond, taps = backbone.encoder.forward_chunk_causal(
            text, anchor, None, noised, timestep, mask=enc_mask,
        )
        velocity = backbone.decoder.forward_chunk_causal(
            noised, cond, timestep, mask=dec_mask,
        )
    torch.cuda.synchronize()
    print(json.dumps({"phase": "native", "seconds": round(time.perf_counter() - started, 1),
                      "velocity_shape": list(velocity.shape),
                      "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 2)}), flush=True)
    return velocity.detach().cpu(), taps.taps[-1].detach().cpu()


def _compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert actual.isfinite().all() and expected.isfinite().all(), (name, "non-finite output")
    diff = (actual.float() - expected.float()).abs()
    relative_l2 = (diff.square().sum() / expected.float().square().sum().clamp_min(1e-12)).sqrt()
    close = torch.allclose(actual, expected, rtol=0.03, atol=0.03)
    print(json.dumps({"phase": "parity", "tensor": name, "shape": list(actual.shape),
                      "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                      "relative_l2": float(relative_l2), "close_3pct": bool(close)}), flush=True)
    assert close, f"{name} differs from the pinned reference beyond rtol/atol 0.03"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mstar-revision", required=True)
    parser.add_argument("--reference-revision", default="68f4d7a254b4bf41951d2e97f44b07f6cac2bf69")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "this probe needs CUDA"
    root = args.checkpoint.resolve()
    config = Praxis1Config.from_checkpoint(root)
    assert config.embodiment == "single_arm", "this bounded probe expects DROID"
    print(json.dumps({"phase": "source", "mstar_revision": args.mstar_revision,
                      "reference_revision": args.reference_revision, "checkpoint": str(root)}), flush=True)
    inputs = _inputs(config)
    torch.cuda.reset_peak_memory_stats()
    expected_velocity, expected_tap = _run_reference(root, inputs)
    actual_velocity, actual_tap = _run_native(root, config, inputs)
    _compare("velocity", actual_velocity, expected_velocity)
    _compare("last_tap", actual_tap, expected_tap)


if __name__ == "__main__":
    main()
