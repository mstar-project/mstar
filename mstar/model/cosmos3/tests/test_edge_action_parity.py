"""CPU numerics parity of the Cosmos3-Edge *action* pathway against diffusers
0.40's ``Cosmos3OmniTransformer``: one policy-mode denoise forward (video
latents with the observation frame clean, an all-noisy action chunk, the
``droid_lerobot`` domain).

The reference runs out-of-process (``notes/ref_dump_edge_action_step.py`` in
the workspace, diffusers >= 0.40 env) and dumps the prompt ids, the seeded
latents and action chunk, the joint mRoPE ids and both predicted velocities.
This test packs the same request with M*'s ``build_action_static_inputs``,
runs the fused forward with real weights in fp32 and compares positions
(exact), the video velocity and the action velocity (fp32 tolerance) — the
check that ``action_proj_in`` / the domain-aware head, the action mRoPE band
and the domain embedding are wired the way the reference has them.
Needs ``COSMOS3_EDGE_ACTION_REF`` (the dump) and the snapshot; skipped otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mstar.model.cosmos3.tests.test_edge import EDGE_DIR

REF = os.environ.get("COSMOS3_EDGE_ACTION_REF")
needs_ref = pytest.mark.skipif(
    EDGE_DIR is None or not REF or not Path(REF).exists(),
    reason="set COSMOS3_EDGE_ACTION_REF to the diffusers action-step dump and COSMOS3_EDGE_DIR to the snapshot",
)

FIELDS = (
    "input_ids", "text_indexes", "position_ids", "und_len", "sequence_length", "vision_token_shapes",
    "vision_sequence_indexes", "vision_mse_loss_indexes", "vision_noisy_frame_indexes", "action_token_shapes",
    "action_sequence_indexes", "action_mse_loss_indexes", "action_noisy_frame_indexes",
)


@needs_ref
def test_edge_policy_step_matches_diffusers() -> None:
    from mstar.model.cosmos3.components.packing import build_action_static_inputs, resolve_action_domain_id
    from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
    from mstar.model.cosmos3.config import Cosmos3Config
    from mstar.model.cosmos3.loader import load_transformer_weights

    rec = torch.load(REF)
    cfg = Cosmos3Config.from_pretrained(EDGE_DIR)
    ids = rec["input_ids"].tolist()
    latents = rec["latents"].float()
    action = rec["action_latents"].float()  # [chunk, action_dim]
    chunk, fps = int(rec["chunk"]), float(rec["fps"])
    static = build_action_static_inputs(
        ids, tuple(latents.shape), chunk, "policy", cfg, cfg.vae.scale_factor_temporal,
        fps=fps, action_fps=fps, action_start_offset=1, device="cpu",
    )
    assert torch.equal(static["position_ids"].to(rec["position_ids"].dtype), rec["position_ids"])
    assert static["num_noisy_action_tokens"] == chunk
    domain = torch.tensor([resolve_action_domain_id(None, rec["domain_name"])], dtype=torch.long)
    assert int(domain) == int(rec["domain_id"])

    with torch.device("meta"):
        model = Cosmos3OmniTransformer(cfg)
    model = model.to_empty(device="cpu").float()
    load_transformer_weights(model, EDGE_DIR, device="cpu")
    model.eval()
    vts = torch.full((static["num_noisy_vision_tokens"],), float(rec["timestep"]))
    ats = torch.full((static["num_noisy_action_tokens"],), float(rec["timestep"]))
    with torch.no_grad():
        pv, pa, _ = model(
            vision_tokens=[latents], vision_timesteps=vts,
            action_tokens=action.unsqueeze(0), action_timesteps=ats, action_domain_id=domain,
            **{k: static[k] for k in FIELDS},
        )
    for name, ours, ref in (
        ("video velocity", pv[0], rec["velocity"]),
        ("action velocity", pa[0].reshape(rec["action_velocity"].shape), rec["action_velocity"]),
    ):
        assert ours.shape == ref.shape, (name, ours.shape, ref.shape)
        err = (ours - ref).abs().max().item()
        scale = ref.abs().max().item()
        assert err <= 1e-3 * max(scale, 1.0), f"{name} max abs diff {err:.3e} (ref scale {scale:.2f})"
