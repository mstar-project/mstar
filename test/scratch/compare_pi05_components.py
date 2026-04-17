"""Compare individual mminf Pi0.5 components against the matching lerobot
PI05Pytorch submodules with real production weights from lerobot/pi05_base.

Tests:
  1. Pi05TimeMLP vs lerobot's time_mlp_in / time_mlp_out chain
  2. Pi05AdaRMSNorm vs lerobot's GemmaRMSNorm (adaRMS path) for one layer
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lerobot.policies.pi05 import PI05Policy

from mminf.model.pi05.components.action_expert import (
    Pi05AdaRMSNorm,
    Pi05TimeMLP,
)
from mminf.model.pi05.components.flow_matching import sincos_timestep_embedding
from mminf.model.pi05.config import Pi05Config

DEVICE = torch.device("cuda")

torch.manual_seed(0)


def main():
    print("Loading lerobot/pi05_base ...")
    policy = PI05Policy.from_pretrained("lerobot/pi05_base").to(DEVICE).eval()
    model = policy.model
    config = policy.config

    action_hidden = model.action_in_proj.out_features  # 1024 for gemma_300m
    print(f"action_expert hidden size: {action_hidden}")

    # ----- Test 1: Pi05TimeMLP vs lerobot time_mlp_in/out -----
    # lerobot's pi05 path: sincos -> Linear(time_mlp_in) -> silu -> Linear(time_mlp_out) -> silu
    timestep = torch.tensor([0.7], device=DEVICE, dtype=torch.float32)

    # Reference path (read directly from lerobot model parameters)
    ref_time_emb = sincos_timestep_embedding(
        timestep, dim=action_hidden, min_period=config.min_period, max_period=config.max_period
    ).squeeze(0)
    ref_h = F.silu(model.time_mlp_in(ref_time_emb.to(model.time_mlp_in.weight.dtype)))
    ref_h = model.time_mlp_out(ref_h)
    ref_adarms_cond = F.silu(ref_h)

    # mminf path
    ours = Pi05TimeMLP(action_hidden).to(DEVICE, dtype=model.time_mlp_in.weight.dtype)
    with torch.no_grad():
        ours.linear_in.weight.copy_(model.time_mlp_in.weight)
        ours.linear_in.bias.copy_(model.time_mlp_in.bias)
        ours.linear_out.weight.copy_(model.time_mlp_out.weight)
        ours.linear_out.bias.copy_(model.time_mlp_out.bias)
    ours_adarms_cond = ours(ref_time_emb.to(ours.linear_in.weight.dtype))

    delta = (ours_adarms_cond.float() - ref_adarms_cond.float()).abs().max().item()
    print("\n[Test 1] Pi05TimeMLP vs lerobot time_mlp chain")
    print(f"  ref abs max: {ref_adarms_cond.abs().max().item():.4f}")
    print(f"  max delta: {delta:.6e}")
    print(f"  status: {'PASS' if delta < 1e-4 else 'FAIL'}")

    # ----- Test 2: Pi05AdaRMSNorm vs lerobot's adaRMS -----
    # Extract layer 0 of the gemma_expert and compare its input_layernorm.
    layer0 = model.paligemma_with_expert.gemma_expert.model.layers[0]
    ref_norm = layer0.input_layernorm
    print("\n[Test 2] adaRMS norm")
    print(f"  ref norm type: {type(ref_norm).__name__}")
    print(f"  ref dense.weight shape: {ref_norm.dense.weight.shape}")
    print(f"  has weight param: {hasattr(ref_norm, 'weight') and ref_norm.weight is not None}")

    # Build mine with the same dims and copy the dense weights.
    cfg = Pi05Config(hidden_size=action_hidden)
    ours_norm = Pi05AdaRMSNorm(
        hidden_size=action_hidden, cond_dim=action_hidden, eps=ref_norm.eps
    ).to(DEVICE, dtype=ref_norm.dense.weight.dtype)
    with torch.no_grad():
        ours_norm.dense.weight.copy_(ref_norm.dense.weight)
        ours_norm.dense.bias.copy_(ref_norm.dense.bias)

    seq_len = 50
    x = torch.randn(seq_len, action_hidden, device=DEVICE, dtype=ref_norm.dense.weight.dtype)
    cond = ref_adarms_cond.to(ref_norm.dense.weight.dtype)

    # Reference: lerobot RMSNorm forward
    ref_out, ref_gate = ref_norm(x, cond)

    ours_out, ours_gate = ours_norm(x, cond)

    delta = (ours_out.float() - ref_out.float()).abs().max().item()
    delta_gate = (ours_gate.float() - ref_gate.float()).abs().max().item()
    print(f"  ref out abs max: {ref_out.abs().max().item():.4f}")
    print(f"  ref gate abs max: {ref_gate.abs().max().item():.4f}")
    print(f"  out max delta: {delta:.6e}")
    print(f"  gate max delta: {delta_gate:.6e}")
    print(f"  status: {'PASS' if (delta < 1e-3 and delta_gate < 1e-3) else 'FAIL'}")


if __name__ == "__main__":
    main()
