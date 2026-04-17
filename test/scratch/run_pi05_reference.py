"""Run lerobot's PI05Pytorch reference inference on a fixed deterministic
input and dump the inputs / intermediates / outputs to disk for later
numerical comparison against mminf.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lerobot.policies.pi05 import PI05Policy

OUT_DIR = Path(__file__).parent / "pi05_reference_dump"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda")
SEED = 0
torch.manual_seed(SEED)


def main():
    policy = PI05Policy.from_pretrained("lerobot/pi05_base").to(DEVICE).eval()
    config = policy.config
    print(f"Loaded PI05 with chunk_size={config.chunk_size}, max_action_dim={config.max_action_dim}")
    print(f"Inference steps: {config.num_inference_steps}")

    model = policy.model
    print(f"Model: {type(model).__name__}")

    # Build deterministic inputs ----------------------------------------
    bsize = 1
    horizon = config.chunk_size  # 50
    action_dim = config.max_action_dim  # 32

    # Three 224x224 RGB images, normalized to [-1, 1] (matches preprocessing).
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    images = [
        torch.rand(bsize, 3, 224, 224, device=DEVICE, generator=g) * 2 - 1
        for _ in range(3)
    ]
    img_masks = [torch.ones(bsize, dtype=torch.bool, device=DEVICE) for _ in range(3)]

    # Tokenized prompt: 4 random PaliGemma tokens (just for testing - the
    # tokens themselves are arbitrary; we only care about deterministic numerics).
    tok_len = 4
    tokens = torch.randint(0, 200, (bsize, tok_len), device=DEVICE, generator=g)
    masks = torch.ones(bsize, tok_len, dtype=torch.bool, device=DEVICE)

    noise = torch.randn(bsize, horizon, action_dim, device=DEVICE, generator=g, dtype=torch.float32)

    # Cast everything to the model's parameter dtype where appropriate.
    param_dtype = next(model.parameters()).dtype
    print(f"Model param dtype: {param_dtype}")

    images_cast = [img.to(dtype=param_dtype) for img in images]

    # Run sample_actions ------------------------------------------------
    actions = model.sample_actions(
        images=images_cast,
        img_masks=img_masks,
        tokens=tokens,
        masks=masks,
        noise=noise,
        num_steps=config.num_inference_steps,
    )
    print(f"Action output shape: {actions.shape}, dtype: {actions.dtype}")
    print(f"  abs max: {actions.abs().max().item():.4f}")
    print(f"  first row: {actions[0, 0].cpu().tolist()[:8]}")

    # Dump everything we'll need for the mminf comparison ---------------
    payload = {
        "images": [img.detach().cpu() for img in images],
        "img_masks": [m.detach().cpu() for m in img_masks],
        "tokens": tokens.detach().cpu(),
        "masks": masks.detach().cpu(),
        "noise": noise.detach().cpu(),
        "actions": actions.detach().cpu(),
        "config": {
            "chunk_size": config.chunk_size,
            "max_action_dim": config.max_action_dim,
            "max_state_dim": config.max_state_dim,
            "num_inference_steps": config.num_inference_steps,
            "min_period": config.min_period,
            "max_period": config.max_period,
            "paligemma_variant": config.paligemma_variant,
            "action_expert_variant": config.action_expert_variant,
            "image_resolution": config.image_resolution,
        },
        "param_dtype": str(param_dtype),
        "seed": SEED,
    }
    out_path = OUT_DIR / "reference_io.pt"
    torch.save(payload, out_path)
    print(f"Saved reference dump to {out_path}")


if __name__ == "__main__":
    main()
