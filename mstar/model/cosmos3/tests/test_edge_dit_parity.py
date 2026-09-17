"""CPU numerics parity of the Cosmos3-Edge DiT against diffusers 0.40's
``Cosmos3OmniTransformer`` (the first Edge-aware release).

The reference runs out-of-process (``notes/ref_dump_edge_dit_step.py`` in the
workspace, diffusers >= 0.40 env) and dumps one fp32 forward on CPU: the
prompt ids, a seeded latent at the 256p tier, the joint mRoPE position ids
and the predicted velocity. This test packs the same prompt with M*'s own
helpers, runs the fused M* forward with real weights in fp32, and compares
positions (exact) and velocity (fp32 tolerance) — the check that the relu2
MLPs, the Nemotron norms and ``k_norm_und_for_gen`` are wired the way the
reference has them.

Needs ``COSMOS3_EDGE_DIT_REF`` (the dump) and the snapshot; skipped otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mstar.model.cosmos3.tests.test_edge import EDGE_DIR

REF = os.environ.get("COSMOS3_EDGE_DIT_REF")
needs_ref = pytest.mark.skipif(
    EDGE_DIR is None or not REF or not Path(REF).exists(),
    reason="set COSMOS3_EDGE_DIT_REF to the diffusers DiT dump and COSMOS3_EDGE_DIR to the snapshot",
)


@needs_ref
def test_edge_dit_step_matches_diffusers() -> None:
    from mstar.model.cosmos3.components.packing import build_static_inputs
    from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
    from mstar.model.cosmos3.config import Cosmos3Config
    from mstar.model.cosmos3.loader import load_transformer_weights

    rec = torch.load(REF)
    cfg = Cosmos3Config.from_pretrained(EDGE_DIR)
    ids = rec["input_ids"].tolist()
    latents = rec["latents"].float()
    static = build_static_inputs(
        ids, tuple(latents.shape), cfg, cfg.vae.scale_factor_temporal, float(rec["fps"]), "cpu",
        has_image_condition=False,
    )
    assert torch.equal(static["position_ids"].to(rec["position_ids"].dtype), rec["position_ids"])

    with torch.device("meta"):
        model = Cosmos3OmniTransformer(cfg)
    model = model.to_empty(device="cpu").float()
    load_transformer_weights(model, EDGE_DIR, device="cpu")
    model.eval()
    fields = (
        "input_ids", "text_indexes", "position_ids", "und_len", "sequence_length", "vision_token_shapes",
        "vision_sequence_indexes", "vision_mse_loss_indexes", "vision_noisy_frame_indexes",
    )
    ts = torch.full((static["num_noisy_vision_tokens"],), float(rec["timestep"]))
    with torch.no_grad():
        preds, _ = model(vision_tokens=[latents], vision_timesteps=ts, **{k: static[k] for k in fields})
    velocity = preds[0]
    ref = rec["velocity"]
    assert velocity.shape == ref.shape
    err = (velocity - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err <= 1e-3 * max(scale, 1.0), f"velocity max abs diff {err:.3e} (ref scale {scale:.2f})"
