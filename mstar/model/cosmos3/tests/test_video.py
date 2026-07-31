"""Tests for the Cosmos3 t2v / i2v path (video packing + conditioning).

CPU-safe unit tests (tiny config / stub tokenizer) cover the video prompt
templates, fps-modulated temporal mRoPE, the conditioned (image-to-video) vs
all-noisy (text-to-video) frame layout, and a multi-frame forward smoke test. An
optional GPU integration test (gated on ``COSMOS3_NANO_DIR`` + CUDA + diffusers)
checks the fused t2v / i2v output against the diffusers ``Cosmos3OmniPipeline``.

Run CPU only:  python3 test_video.py
Run with GPU:  COSMOS3_NANO_DIR=<snap> python3 test_video.py
"""

from __future__ import annotations

import math
import os

import torch

from mstar.model.cosmos3.components.packing import (
    build_static_inputs,
    get_3d_mrope_ids_vae_tokens,
    tokenize_prompt,
)
from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
from mstar.model.cosmos3.config import Cosmos3Config


def _tiny_config() -> Cosmos3Config:
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


class _StubTokenizer:
    """Records the chat-template messages so the metadata templates can be asserted."""

    eos_token_id = 99

    def __init__(self):
        self.seen: list[list[dict]] = []

    def convert_tokens_to_ids(self, _tok):
        return 98  # stand-in for <|vision_start|>

    def apply_chat_template(self, conversations, **_kw):
        self.seen.append(conversations)
        return {"input_ids": [1, 2, 3]}


def test_video_prompt_templates() -> None:
    tok = _StubTokenizer()
    cond, uncond = tokenize_prompt(tok, "a cat", "bad", num_frames=48, height=720, width=1280, fps=24.0)
    # Special tokens appended (eos, start-of-generation).
    assert cond[-2:] == [99, 98] and uncond[-2:] == [99, 98]
    # System prompt is the video one; positive prompt carries duration + video resolution.
    sys_msg = tok.seen[0][0]
    assert sys_msg["role"] == "system" and "videos" in sys_msg["content"]
    pos_user = tok.seen[0][1]["content"]
    assert "2.0 seconds long" in pos_user and "24 FPS" in pos_user
    assert "This video is of 720x1280 resolution." in pos_user
    # Negative prompt uses the inverse templates.
    neg_user = tok.seen[1][1]["content"]
    assert "is not 2.0 seconds long" in neg_user and "This video is not of" in neg_user


def test_image_prompt_has_no_duration() -> None:
    tok = _StubTokenizer()
    tokenize_prompt(tok, "a cat", "", num_frames=1, height=256, width=256)
    sys_msg, user_msg = tok.seen[0][0], tok.seen[0][1]["content"]
    assert "images" in sys_msg["content"]
    assert "seconds long" not in user_msg
    assert "This image is of 256x256 resolution." in user_msg


def test_video_mrope_fps_modulation() -> None:
    # grid_t > 1 with fps enables float, fps-scaled temporal positions; halving the
    # fps relative to base doubles the temporal spacing.
    ids12, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=3, grid_h=1, grid_w=1, temporal_offset=100, fps=12.0, base_fps=24.0, temporal_compression_factor=4
    )
    ids24, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=3, grid_h=1, grid_w=1, temporal_offset=100, fps=24.0, base_fps=24.0, temporal_compression_factor=4
    )
    assert ids12.dtype == torch.float32
    assert ids12[0].tolist() == [100.0, 102.0, 104.0]
    assert ids24[0].tolist() == [100.0, 101.0, 102.0]
    # A single frame disables fps modulation (image mode) -> integer positions.
    ids1, _ = get_3d_mrope_ids_vae_tokens(grid_t=1, grid_h=2, grid_w=2, temporal_offset=5, fps=24.0)
    assert ids1.dtype == torch.long and ids1[0].tolist() == [5, 5, 5, 5]


def test_video_packing_t2v_vs_i2v() -> None:
    cfg = Cosmos3Config()  # Nano defaults
    input_ids = list(range(7))
    latent_shape = (1, cfg.latent_channel, 3, 16, 16)  # T_lat=3, patch grid 8x8
    per_frame = 8 * 8

    t2v = build_static_inputs(input_ids, latent_shape, cfg, 4, 24.0, "cpu", has_image_condition=False)
    assert t2v["num_vision_tokens"] == 3 * per_frame
    assert t2v["num_noisy_vision_tokens"] == 3 * per_frame  # all frames noisy
    assert t2v["vision_noisy_frame_indexes"][0].tolist() == [0, 1, 2]
    assert t2v["position_ids"].dtype == torch.float32  # fps modulation -> float positions
    # Vision temporal positions sit past the text + margin.
    assert int(t2v["position_ids"][0, 7].item()) == 7 + cfg.unified_3d_mrope_temporal_modality_margin

    i2v = build_static_inputs(input_ids, latent_shape, cfg, 4, 24.0, "cpu", has_image_condition=True)
    assert i2v["num_vision_tokens"] == 3 * per_frame  # frame 0 stays in the sequence
    assert i2v["num_noisy_vision_tokens"] == 2 * per_frame  # frame 0 anchored, frames 1-2 noisy
    assert i2v["vision_noisy_frame_indexes"][0].tolist() == [1, 2]
    # mse indexes skip frame 0 (first noisy token is und_len + one frame stride).
    assert int(i2v["vision_mse_loss_indexes"][0]) == 7 + per_frame


def test_video_forward_smoke_cpu() -> None:
    cfg = _tiny_config()
    torch.manual_seed(0)
    model = Cosmos3OmniTransformer(cfg).eval()
    latent_shape = (1, cfg.latent_channel, 3, 4, 4)  # T_lat=3, patch grid 2x2 -> 12 vision tokens
    fields = [
        "input_ids", "text_indexes", "position_ids", "und_len", "sequence_length",
        "vision_token_shapes", "vision_sequence_indexes", "vision_mse_loss_indexes",
        "vision_noisy_frame_indexes",
    ]
    for has_cond in (False, True):
        static = build_static_inputs([1, 2, 3], latent_shape, cfg, 4, 24.0, "cpu", has_image_condition=has_cond)
        with torch.no_grad():
            preds, sound = model(
                vision_tokens=[torch.randn(latent_shape)],
                vision_timesteps=torch.full((static["num_noisy_vision_tokens"],), 500.0),
                **{k: static[k] for k in fields},
            )
        assert sound is None
        assert preds[0].shape == latent_shape, preds[0].shape
        assert torch.isfinite(preds[0]).all()
        if has_cond:
            # The conditioning frame is anchored: the model predicts no velocity for it.
            assert torch.count_nonzero(preds[0][:, :, 0]) == 0


# ---------------------------------------------------------------------------
# GPU parity (gated on COSMOS3_NANO_DIR + CUDA + diffusers).
# ---------------------------------------------------------------------------

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
_GPU_CACHE: dict = {}
_V_FRAMES, _V_RES, _V_STEPS, _V_GS = 17, 256, 15, 6.0


def _gpu_setup():
    if "ctx" in _GPU_CACHE:
        return _GPU_CACHE["ctx"]
    snap = os.environ.get("COSMOS3_NANO_DIR")
    if not snap or not torch.cuda.is_available():
        _GPU_CACHE["ctx"] = None
        return None
    try:
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
        from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer as DTr
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped video parity: diffusers/transformers unavailable: {exc})")
        _GPU_CACHE["ctx"] = None
        return None
    torch.use_deterministic_algorithms(True, warn_only=True)
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
    from mstar.model.cosmos3.tests.pipeline import Cosmos3Pipeline

    dev, dtype = "cuda:0", torch.bfloat16
    dpipe = Cosmos3OmniPipeline(
        transformer=DTr.from_pretrained(snap, subfolder="transformer", torch_dtype=dtype),
        text_tokenizer=AutoTokenizer.from_pretrained(os.path.join(snap, "text_tokenizer")),
        vae=AutoencoderKLWan.from_pretrained(snap, subfolder="vae", torch_dtype=dtype),
        scheduler=UniPCMultistepScheduler.from_pretrained(snap, subfolder="scheduler"),
        sound_tokenizer=None, enable_safety_checker=False,
    ).to(dev)
    mpipe = Cosmos3Pipeline.from_model(Cosmos3Model(model_path_hf=snap), device=dev, dtype=dtype)
    _GPU_CACHE["ctx"] = dict(dpipe=dpipe, mpipe=mpipe, snap=snap, device=dev, dtype=dtype)
    return _GPU_CACHE["ctx"]


def _video_parity(mode: str) -> None:
    ctx = _gpu_setup()
    if ctx is None:
        print(f"  (skipped {mode} parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    import json

    from PIL import Image

    dpipe, mpipe, snap = ctx["dpipe"], ctx["mpipe"], ctx["snap"]
    dev, dtype = ctx["device"], ctx["dtype"]
    is_i2v = mode == "i2v"
    asset = os.path.join(snap, "assets", "example_i2v_prompt.json" if is_i2v else "example_t2v_prompt.json")
    with open(asset) as f:
        prompt = json.load(f)["temporal_caption"]
    image = (
        Image.open(os.path.join(snap, "assets", "example_i2v_input.jpg")).convert("RGB") if is_i2v else None
    )
    gen = torch.Generator(device=dev).manual_seed(0)
    init, _ = mpipe._prepare_latents(image, _V_FRAMES, _V_RES, _V_RES, gen, None, dev, dtype)
    common = dict(prompt=prompt, negative_prompt="", num_frames=_V_FRAMES, height=_V_RES, width=_V_RES,
                  num_inference_steps=_V_STEPS, guidance_scale=_V_GS, fps=24.0)
    lat_d = dpipe(image=image, latents=init.clone(), output_type="latent", enable_safety_check=False, **common)[0]
    lat_m = mpipe(image=image, latents=init.clone(), decode=False, **common)
    img_d = mpipe._decode(lat_d.reshape(lat_m.shape).to(dtype)).squeeze().float().cpu()
    img_m = mpipe._decode(lat_m).squeeze().float().cpu()
    mse = (img_d - img_m).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 30, f"{mode} video PSNR {psnr:.2f} < 30 (MSE {mse:.3e})"
    print(f"  {mode} parity PSNR={psnr:.2f} dB")


def test_t2v_parity_vs_diffusers() -> None:
    _video_parity("t2v")


def test_i2v_parity_vs_diffusers() -> None:
    _video_parity("i2v")


def _main() -> None:
    failures = []
    tests = [
        ("video_prompt_templates", test_video_prompt_templates),
        ("image_prompt_has_no_duration", test_image_prompt_has_no_duration),
        ("video_mrope_fps_modulation", test_video_mrope_fps_modulation),
        ("video_packing_t2v_vs_i2v", test_video_packing_t2v_vs_i2v),
        ("video_forward_smoke_cpu", test_video_forward_smoke_cpu),
        ("t2v_parity_vs_diffusers", test_t2v_parity_vs_diffusers),
        ("i2v_parity_vs_diffusers", test_i2v_parity_vs_diffusers),
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
    print("\nAll Cosmos3 video checks passed.")


if __name__ == "__main__":
    _main()
