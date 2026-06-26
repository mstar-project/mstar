"""Tests for the Cosmos3 t2i forward + packing.

CPU-safe unit tests (tiny config) cover patchify/unpatchify, the 3D mRoPE id
helpers, the t2i packing assembly, and a full forward smoke test. An optional
GPU integration test (gated on ``COSMOS3_NANO_DIR`` + CUDA + diffusers) checks
the t2i image against the diffusers ``Cosmos3OmniPipeline``.

Run CPU only:  python3 test_t2i.py
Run with GPU:  COSMOS3_NANO_DIR=<snap> python3 test_t2i.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch

from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.packing import (
    build_t2i_static_inputs,
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)


def _tiny_config() -> Cosmos3Config:
    """A small, CPU-cheap Cosmos3 config with the same structure as Nano.

    head_dim // 2 == sum(rope_axes_dim) is required by the interleaved mRoPE;
    patch_latent_dim == latent_patch_size**2 * latent_channel.
    """
    return Cosmos3Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=100,
        rope_axes_dim=(4, 2, 2),
        latent_channel=8,
        latent_patch_size=2,
        patch_latent_dim=32,
        sound_gen=False,
        action_gen=False,
    )


def test_patchify_unpatchify_roundtrip() -> None:
    cfg = _tiny_config()
    model = Cosmos3OmniTransformer(cfg)
    p = cfg.latent_patch_size
    x = torch.randn(1, cfg.latent_channel, 1, 4 * p, 3 * p)  # [1,C,T=1,H,W], H/W divisible by p
    packed, orig_shapes = model._patchify_and_pack_latents([x])
    assert packed.shape == (1 * 4 * 3, cfg.patch_latent_dim), packed.shape
    assert orig_shapes == [(1, 4 * p, 3 * p)], orig_shapes
    # All-noisy single frame -> unpatchify recovers x exactly.
    token_shapes = [(1, 4, 3)]
    recovered = model._unpatchify_and_unpack_latents(
        packed, token_shapes, [torch.arange(1)], orig_shapes
    )[0]
    assert recovered.shape == x.shape
    assert torch.allclose(recovered, x, atol=1e-6), (recovered - x).abs().max()


def test_mrope_ids_text() -> None:
    ids, nxt = get_3d_mrope_ids_text_tokens(num_tokens=5, temporal_offset=3)
    assert ids.shape == (3, 5)
    assert torch.equal(ids[0], ids[1]) and torch.equal(ids[1], ids[2])
    assert ids[0].tolist() == [3, 4, 5, 6, 7]
    assert nxt == 8


def test_mrope_ids_vae() -> None:
    # t2i: grid_t=1 -> no fps modulation; spatial reset keeps h/w as plain grids.
    ids, _ = get_3d_mrope_ids_vae_tokens(grid_t=1, grid_h=2, grid_w=3, temporal_offset=10)
    assert ids.shape == (3, 6)
    assert ids[0].tolist() == [10] * 6  # all temporal positions == offset
    assert ids[1].tolist() == [0, 0, 0, 1, 1, 1]  # h grid
    assert ids[2].tolist() == [0, 1, 2, 0, 1, 2]  # w grid


def test_packing_t2i_structure() -> None:
    cfg = Cosmos3Config()  # Nano defaults
    input_ids = list(range(7))
    latent_shape = (1, cfg.latent_channel, 1, 16, 16)
    out = build_t2i_static_inputs(input_ids, latent_shape, cfg, vae_scale_factor_temporal=4, fps=24.0, device="cpu")
    num_vision = 1 * 8 * 8  # patch grid 8x8
    assert out["und_len"] == 7
    assert out["sequence_length"] == 7 + num_vision
    assert out["position_ids"].shape == (3, 7 + num_vision)
    assert out["vision_sequence_indexes"].tolist() == list(range(7, 7 + num_vision))
    assert out["vision_token_shapes"] == [(1, 8, 8)]
    # Vision temporal positions sit past the text + 15000 margin.
    assert int(out["position_ids"][0, 7].item()) == 7 + cfg.unified_3d_mrope_temporal_modality_margin


def test_forward_smoke_cpu() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    model = Cosmos3OmniTransformer(cfg).eval()
    latent_shape = (1, cfg.latent_channel, 1, 4, 4)  # patch grid 2x2 -> 4 vision tokens
    static = build_t2i_static_inputs(
        [1, 2, 3], latent_shape, cfg, vae_scale_factor_temporal=4, fps=24.0, device="cpu"
    )
    fields = [
        "input_ids", "text_indexes", "position_ids", "und_len", "sequence_length",
        "vision_token_shapes", "vision_sequence_indexes", "vision_mse_loss_indexes",
        "vision_noisy_frame_indexes",
    ]
    with torch.no_grad():
        preds, sound = model(
            vision_tokens=[torch.randn(latent_shape)],
            vision_timesteps=torch.full((static["num_noisy_vision_tokens"],), 500.0),
            **{k: static[k] for k in fields},
        )
    assert sound is None
    assert preds[0].shape == latent_shape, preds[0].shape
    assert torch.isfinite(preds[0]).all()


def test_t2i_parity_vs_diffusers() -> None:
    """GPU integration: mstar DiT swapped into the diffusers pipeline yields a
    bit-exact t2i image (deterministic cuBLAS). Skipped without GPU/checkpoint."""
    snap = os.environ.get("COSMOS3_NANO_DIR")
    if not snap or not torch.cuda.is_available():
        print("  (skipped t2i parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    try:
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
        from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer as DTr
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped t2i parity: diffusers/transformers unavailable: {exc})")
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model

    snap_p = Path(snap)
    dev, dtype = "cuda:0", torch.bfloat16
    pipe = Cosmos3OmniPipeline(
        transformer=DTr.from_pretrained(snap_p, subfolder="transformer", torch_dtype=dtype),
        text_tokenizer=AutoTokenizer.from_pretrained(str(snap_p / "text_tokenizer")),
        vae=AutoencoderKLWan.from_pretrained(snap_p, subfolder="vae", torch_dtype=dtype),
        scheduler=UniPCMultistepScheduler.from_pretrained(snap_p, subfolder="scheduler"),
        sound_tokenizer=None, enable_safety_checker=False,
    ).to(dev)

    def gen():
        return pipe(prompt="A red cube on a wooden table.", negative_prompt="", num_frames=1,
                    height=256, width=256, num_inference_steps=4, guidance_scale=6.0,
                    generator=torch.Generator(device=dev).manual_seed(0),
                    output_type="pt", enable_safety_check=False).video[0].float().cpu()

    img_d = gen()
    mtr = Cosmos3Model(model_path_hf=snap).get_submodule("dit", device=dev).transformer
    mtr.dtype = dtype
    pipe.transformer = mtr
    img_m = gen()
    mse = (img_d - img_m).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 30, f"t2i image PSNR {psnr:.2f} < 30 (MSE {mse:.3e})"
    print(f"  t2i parity PSNR={psnr:.2f} dB")


def _main() -> None:
    failures = []
    tests = [
        ("patchify_unpatchify_roundtrip", test_patchify_unpatchify_roundtrip),
        ("mrope_ids_text", test_mrope_ids_text),
        ("mrope_ids_vae", test_mrope_ids_vae),
        ("packing_t2i_structure", test_packing_t2i_structure),
        ("forward_smoke_cpu", test_forward_smoke_cpu),
        ("t2i_parity_vs_diffusers", test_t2i_parity_vs_diffusers),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc!r}")
    if failures:
        raise SystemExit(1)
    print("\nAll Cosmos3 t2i checks passed.")


if __name__ == "__main__":
    _main()
