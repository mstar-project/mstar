"""Tests for the Cosmos3 opt-in sound generation path.

CPU-safe unit tests cover the sound segment packing (mrope band, indexes,
position ids), the duration -> latent-frame math, the request-level validation
and walk selection, a tiny AVAE decoder shape check, and the decoder-subset
checkpoint load. GPU tests (gated on ``COSMOS3_NANO_DIR`` + CUDA) check the
engine cache-once [video | sound] loop against the fused reference pipeline —
bit-tight with the sdpa handle, PSNR-level with FlashInfer — and smoke the
real checkpoint sound tokenizer.

Run CPU only:  python3 test_sound.py
Run with GPU:  COSMOS3_NANO_DIR=<snap> python3 test_sound.py
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F  # noqa: F401 — keeps parity harness imports uniform

from mstar.model.cosmos3.components.packing import build_static_inputs
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.constants import VIDEO_SOUND_GEN_WALK
from mstar.model.cosmos3.submodules import Cosmos3DiTSubmodule
from mstar.model.cosmos3.tests.test_engine_cache import (
    _flashinfer_cache,
    _load,
    _SdpaCacheHandle,
)

# ---------------------------------------------------------------------------
# CPU-safe unit tests
# ---------------------------------------------------------------------------


def test_sound_segment_layout() -> None:
    cfg = Cosmos3Config()  # Nano defaults (fps modulation on, sound fps 25)
    input_ids = list(range(7))
    latent_shape = (1, cfg.latent_channel, 3, 16, 16)  # patch grid 8x8
    per_frame = 8 * 8
    t_s = 18

    st = build_static_inputs(input_ids, latent_shape, cfg, 4, 24.0, "cpu", sound_latent_frames=t_s)
    n_vis = 3 * per_frame
    assert st["num_sound_tokens"] == t_s
    assert st["sequence_length"] == 7 + n_vis + t_s
    # Sound tokens sit after the vision band; all of them are noisy/predicted.
    assert st["sound_sequence_indexes"].tolist() == list(range(7 + n_vis, 7 + n_vis + t_s))
    assert st["sound_mse_loss_indexes"].tolist() == st["sound_sequence_indexes"].tolist()
    assert st["sound_noisy_frame_indexes"][0].tolist() == list(range(t_s))
    assert st["sound_token_shapes"] == [(t_s, 1, 1)]
    # Position ids cover text + vision + sound; the sound band shares the media
    # temporal offset and advances at base_fps / sound_latent_fps (24/25) per
    # latent frame, with zero spatial ids ((T, 1, 1) grid).
    assert st["position_ids"].shape == (3, 7 + n_vis + t_s)
    media_off = 7 + cfg.unified_3d_mrope_temporal_modality_margin
    s_t = st["position_ids"][0, 7 + n_vis:]
    assert abs(s_t[0].item() - media_off) < 1e-6
    # float32 positions at the ~15k media offset quantize the 0.96 spacing to
    # ~4e-4; assert against the same-precision computation's tolerance.
    assert abs((s_t[1] - s_t[0]).item() - 24.0 / 25.0) < 5e-3
    assert st["position_ids"][1, 7 + n_vis:].abs().max().item() == 0
    assert st["position_ids"][2, 7 + n_vis:].abs().max().item() == 0
    # Without sound the layout is unchanged.
    st0 = build_static_inputs(input_ids, latent_shape, cfg, 4, 24.0, "cpu")
    assert st0["sequence_length"] == 7 + n_vis and "sound_sequence_indexes" not in st0


def test_sound_frame_math() -> None:
    dit = Cosmos3DiTSubmodule(transformer=None, config=Cosmos3Config())
    # Video-duration default: 93 frames @ 16 fps = 5.8125 s -> 279000 samples
    # -> ceil(279000 / 1920) = 146 latent frames.
    target, frames = dit._resolve_sound_frames({"num_frames": 93, "fps": 16.0})
    assert (target, frames) == (279000, 146)
    # 17 frames @ 24 fps = 0.7083 s -> 34000 samples -> 18 latent frames.
    target, frames = dit._resolve_sound_frames({"num_frames": 17, "fps": 24.0})
    assert (target, frames) == (34000, 18)
    # Explicit sound_duration wins; sub-frame durations clamp to one video frame.
    target, frames = dit._resolve_sound_frames({"num_frames": 17, "fps": 24.0, "sound_duration": 2.0})
    assert (target, frames) == (96000, 50)
    target, frames = dit._resolve_sound_frames({"num_frames": 17, "fps": 24.0, "sound_duration": 0.0})
    assert target == round(48000 / 24.0) and frames == 2


def test_sound_request_validation() -> None:
    from mstar.model.cosmos3.cosmos3_model import AUDIO_DECODER_NODE, Cosmos3Model

    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    # Sound rides the video walk; t2i and action requests reject it.
    p = model._resolve_gen_params({"generate_sound": True, "num_frames": 17}, [], ["video"])
    assert p["generate_sound"] is True
    for bad in (
        {"generate_sound": True},  # single-frame (image) default
        {"generate_sound": True, "num_frames": 17, "action_mode": "policy"},
    ):
        try:
            model._resolve_gen_params(bad, [], ["video"] if "num_frames" in bad else [])
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    # The sound walk + audio decoder node are registered when sound is enabled...
    assert VIDEO_SOUND_GEN_WALK in model.get_graph_walk_graphs()
    assert model.get_node_engine_types()[AUDIO_DECODER_NODE] is not None
    # ...and disappear when the serving knob is off.
    model.config.enable_sound = False
    assert VIDEO_SOUND_GEN_WALK not in model.get_graph_walk_graphs()
    assert AUDIO_DECODER_NODE not in model.get_node_engine_types()
    try:
        model._resolve_gen_params({"generate_sound": True, "num_frames": 17}, [], ["video"])
        raise AssertionError("expected ValueError with sound serving disabled")
    except ValueError:
        pass


def test_sound_forward_smoke_cpu() -> None:
    from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer

    cfg = Cosmos3Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=100, rope_axes_dim=(4, 2, 2),
        latent_channel=8, latent_patch_size=2, patch_latent_dim=32,
        sound_gen=True, sound_dim=6, action_gen=False,
    )
    model = Cosmos3OmniTransformer(cfg).eval()
    # The parallel linears allocate uninitialized storage (production overwrites
    # it with checkpoint weights); give every parameter small deterministic
    # values so this smoke is not at the mercy of allocator garbage.
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
    latent_shape = (1, cfg.latent_channel, 3, 4, 4)
    t_s = 5
    static = build_static_inputs([1, 2, 3], latent_shape, cfg, 4, 24.0, "cpu", sound_latent_frames=t_s)
    fields = [
        "input_ids", "text_indexes", "position_ids", "und_len", "sequence_length",
        "vision_token_shapes", "vision_sequence_indexes", "vision_mse_loss_indexes",
        "vision_noisy_frame_indexes", "sound_token_shapes", "sound_sequence_indexes",
        "sound_mse_loss_indexes", "sound_noisy_frame_indexes",
    ]
    with torch.no_grad():
        preds, sound = model(
            vision_tokens=[torch.randn(latent_shape)],
            vision_timesteps=torch.full((static["num_noisy_vision_tokens"],), 500.0),
            sound_tokens=[torch.randn(cfg.sound_dim, t_s)],
            sound_timesteps=torch.full((t_s,), 500.0),
            **{k: static[k] for k in fields},
        )
    assert preds[0].shape == latent_shape and torch.isfinite(preds[0]).all()
    assert sound[0].shape == (cfg.sound_dim, t_s) and torch.isfinite(sound[0]).all()
    # The cached denoise step decodes the same band via _embed_sound/_decode_sound;
    # its scatter/gather helpers must round-trip all-noisy sound frames.
    emb = model._embed_sound(
        torch.randn(1, cfg.sound_dim, t_s), torch.full((t_s,), 500.0),
        static["sound_token_shapes"], static["sound_noisy_frame_indexes"], torch.float32,
    )
    assert emb.shape == (t_s, cfg.hidden_size)
    dec = model._decode_sound(
        torch.randn(t_s, cfg.hidden_size), static["sound_token_shapes"], static["sound_noisy_frame_indexes"]
    )
    assert dec.shape == (1, cfg.sound_dim, t_s)


def test_sound_tokenizer_tiny_decode() -> None:
    from mstar.model.cosmos3.components.sound_tokenizer import Cosmos3SoundTokenizer

    tok = Cosmos3SoundTokenizer({
        "sampling_rate": 8, "hop_size": 4, "dec_dim": 4, "dec_c_mults": [1, 2],
        "dec_strides": [2, 2], "dec_out_channels": 2, "vocoder_input_dim": 3,
    })
    tok = tok.to(torch.float32)
    assert tok.latent_fps == 2.0 and tok.get_audio_num_samples(5) == 20
    with torch.no_grad():
        audio = tok.decode(torch.randn(1, 3, 5))
    assert audio.shape == (1, 2, 20)
    assert audio.abs().max().item() <= 1.0


def test_sound_tokenizer_load_ignores_encoder_keys() -> None:
    import json
    import tempfile

    from safetensors.torch import save_file

    from mstar.model.cosmos3.components.sound_tokenizer import Cosmos3SoundTokenizer

    config = {
        "sampling_rate": 8, "hop_size": 4, "dec_dim": 4, "dec_c_mults": [1, 2],
        "dec_strides": [2, 2], "dec_out_channels": 2, "vocoder_input_dim": 3,
    }
    decoder_sd = {k: v.clone() for k, v in Cosmos3SoundTokenizer(config).state_dict().items()}
    # Full-AVAE checkpoints carry encoder tensors on top of the decoder ones.
    encoder_sd = {
        "encoder.layers.0.weight_g": torch.randn(4, 1, 1),
        "encoder.layers.0.weight_v": torch.randn(4, 3, 7),
        "encoder.layers.1.act.alpha": torch.randn(1, 4, 1),
    }
    with tempfile.TemporaryDirectory() as tmp:
        tdir = os.path.join(tmp, "sound_tokenizer")
        os.makedirs(tdir)
        with open(os.path.join(tdir, Cosmos3SoundTokenizer.CONFIG_NAME), "w") as f:
            json.dump(config, f)
        weights_path = os.path.join(tdir, Cosmos3SoundTokenizer.WEIGHTS_NAME)
        save_file({**decoder_sd, **encoder_sd}, weights_path)
        tok = Cosmos3SoundTokenizer.from_pretrained(tmp, dtype=torch.float32)
        # The encoder tensors are dropped; every decoder weight loads verbatim.
        loaded = tok.state_dict()
        assert set(loaded) == set(decoder_sd)
        assert all((loaded[k] == decoder_sd[k]).all() for k in decoder_sd)
        # A genuinely missing decoder key must still fail the load.
        short_sd = dict(decoder_sd)
        dropped = sorted(short_sd)[0]
        short_sd.pop(dropped)
        save_file({**short_sd, **encoder_sd}, weights_path)
        try:
            Cosmos3SoundTokenizer.from_pretrained(tmp, dtype=torch.float32)
            raise AssertionError("expected KeyError for a missing decoder key")
        except KeyError as exc:
            assert dropped in str(exc)


# ---------------------------------------------------------------------------
# GPU parity (gated on COSMOS3_NANO_DIR + CUDA). Reuses the engine-cache test
# harness: same prompt/seed discipline, sdpa handle for bit-tight bounds,
# FlashInfer for the served path.
# ---------------------------------------------------------------------------

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
PROMPT = "A drummer plays a fast roll on a snare drum in a small room."
H = W = 256
FRAMES, STEPS, GS, SEED = 17, 10, 6.0, 7
_SOUND_CACHE: dict = {}


@torch.no_grad()
def _run_cache_once_sound(model, dit, cm, init, sound_init, cond_ids, uncond_ids, device):
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.model.submodule_base import ModelInputsFromEngine

    rid = "rs0"
    md = {"height": H, "width": W, "num_frames": FRAMES, "fps": 24.0,
          "guidance_scale": GS, "num_inference_steps": STEPS, "generate_sound": True}
    fwd = CurrentForwardPassInfo(
        request_id=rid, graph_walk="prefill", requires_cfg=True,
        fwd_index=0, random_seed=SEED, max_tokens=0, sampling_config={}, step_metadata=md,
    )
    ei = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: fwd}, cache_manager=cm)
    text_inputs = [
        torch.tensor(cond_ids, dtype=torch.long, device=device),
        torch.tensor(uncond_ids, dtype=torch.long, device=device),
    ]
    ni = dit.prepare_inputs("prefill", fwd, {"text_inputs": text_inputs})
    dit.forward("prefill", ei, **dit.preprocess("prefill", ei, [ni]))

    latents, sound_latents = init.clone(), sound_init.clone()
    time_index = torch.zeros(1, dtype=torch.long, device=device)
    fwd.graph_walk = VIDEO_SOUND_GEN_WALK
    for _ in range(STEPS):
        ni = dit.prepare_inputs(VIDEO_SOUND_GEN_WALK, fwd, {
            "latents": [latents], "sound_latents": [sound_latents], "time_index": [time_index],
        })
        out = dit.forward(VIDEO_SOUND_GEN_WALK, ei, **dit.preprocess(VIDEO_SOUND_GEN_WALK, ei, [ni]))
        latents = out["latents"][0]
        sound_latents = out["sound_latents"][0]
        time_index = out["time_index"][0]
    dit.cleanup_request(rid)
    return latents, sound_latents


def _sound_scenario():
    if "ctx" in _SOUND_CACHE:
        return _SOUND_CACHE["ctx"]
    base = _load()
    if base is None:
        _SOUND_CACHE["ctx"] = None
        return None
    from mstar.model.cosmos3.components.packing import tokenize_prompt

    device, dtype, mpipe, model = base["device"], base["dtype"], base["mpipe"], base["model"]
    cond_ids, uncond_ids = tokenize_prompt(model.tokenizer, PROMPT, "", num_frames=FRAMES, height=H, width=W)
    lat_t = 1 + (FRAMES - 1) // mpipe.vae_scale_temporal
    _, t_s = model.get_submodule("dit")._resolve_sound_frames({"num_frames": FRAMES, "fps": 24.0})
    gen = torch.Generator(device=device).manual_seed(SEED)
    init = torch.randn((1, 48, lat_t, H // 16, W // 16), generator=gen, device=device, dtype=dtype)
    # The fused pipeline draws its sound noise from `generator` after the video
    # latents; with explicit `latents` that is the generator's first draw, so a
    # same-state generator here reproduces it for the engine loop.
    sgen = torch.Generator(device=device).manual_seed(SEED + 1)
    from diffusers.utils.torch_utils import randn_tensor

    sound_init = randn_tensor((1, model.config.sound_dim, t_s), generator=sgen, device=device, dtype=dtype)
    sgen2 = torch.Generator(device=device).manual_seed(SEED + 1)
    lat_fused, sound_fused = mpipe(
        prompt=PROMPT, negative_prompt="", num_frames=FRAMES, height=H, width=W,
        num_inference_steps=STEPS, guidance_scale=GS, latents=init.clone(), decode=False,
        generate_sound=True, generator=sgen2,
    )
    ctx = dict(
        cond=cond_ids, uncond=uncond_ids, init=init, sound_init=sound_init,
        lat_fused=lat_fused, sound_fused=sound_fused, t_s=t_s, **base,
    )
    _SOUND_CACHE["ctx"] = ctx
    return ctx


def test_sound_cache_once_matches_fused_exact() -> None:
    ctx = _sound_scenario()
    if ctx is None:
        print("  (skipped sound cache-once parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    dit = ctx["dit"]
    prev = dit.batched_cfg
    dit.batched_cfg = False
    try:
        lat, snd = _run_cache_once_sound(
            ctx["model"], dit, _SdpaCacheHandle(), ctx["init"], ctx["sound_init"],
            ctx["cond"], ctx["uncond"], ctx["device"],
        )
    finally:
        dit.batched_cfg = prev
    vdiff = (ctx["lat_fused"].float() - lat.reshape(ctx["lat_fused"].shape).float()).abs().max().item()
    sdiff = (ctx["sound_fused"].float() - snd.reshape(ctx["sound_fused"].shape).float()).abs().max().item()
    assert vdiff <= 1e-3, f"sound-walk video latents differ from fused by {vdiff:.3e} (> 1e-3)"
    assert sdiff <= 1e-3, f"sound latents differ from fused by {sdiff:.3e} (> 1e-3)"
    print(f"  sound cache-once (sdpa) abs-max diff: video={vdiff:.3e} sound={sdiff:.3e}")


def test_sound_engine_path_flashinfer() -> None:
    ctx = _sound_scenario()
    if ctx is None:
        print("  (skipped sound engine parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    try:
        cm = _flashinfer_cache(ctx["model"], "rs0", ctx["device"], ctx["dtype"])
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped sound engine parity: FlashInfer unavailable: {exc})")
        return
    lat, snd = _run_cache_once_sound(
        ctx["model"], ctx["dit"], cm, ctx["init"], ctx["sound_init"],
        ctx["cond"], ctx["uncond"], ctx["device"],
    )
    img_fused = ctx["mpipe"]._decode(ctx["lat_fused"]).squeeze().float().cpu()
    img_engine = ctx["mpipe"]._decode(lat.reshape(ctx["lat_fused"].shape)).squeeze().float().cpu()
    mse = (img_fused - img_engine).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    s_ref = ctx["sound_fused"].float()
    s_mse = (s_ref - snd.reshape(s_ref.shape).float()).pow(2).mean().item()
    s_snr = float("inf") if s_mse == 0 else 10 * math.log10(s_ref.pow(2).mean().item() / s_mse)
    assert psnr >= 30, f"sound-walk video PSNR {psnr:.2f} < 30 (MSE {mse:.3e})"
    assert s_snr >= 30, f"sound latent SNR {s_snr:.2f} < 30 dB (MSE {s_mse:.3e})"
    print(f"  sound engine path (flashinfer): video PSNR={psnr:.2f} dB, sound latent SNR={s_snr:.2f} dB")


def test_sound_tokenizer_decode_real() -> None:
    ctx = _sound_scenario()
    if ctx is None:
        print("  (skipped sound tokenizer decode: needs COSMOS3_NANO_DIR + CUDA)")
        return
    from mstar.model.cosmos3.components.sound_tokenizer import Cosmos3SoundTokenizer

    snap = os.environ.get("COSMOS3_NANO_DIR")
    if not (os.path.isdir(os.path.join(snap, "sound_tokenizer"))):
        print("  (skipped sound tokenizer decode: checkpoint has no sound_tokenizer/)")
        return
    tok = Cosmos3SoundTokenizer.from_pretrained(snap, device=ctx["device"], dtype=torch.bfloat16)
    assert tok.sample_rate == 48000 and tok.hop_size == 1920 and tok.latent_ch == 64
    audio = tok.decode(ctx["sound_fused"].to(torch.bfloat16))
    assert audio.shape == (1, tok.audio_channels, ctx["t_s"] * tok.hop_size)
    audio = audio.float()
    assert torch.isfinite(audio).all() and audio.abs().max().item() <= 1.0
    # Denoised latents must decode to a live waveform, not (near-)silence or rail
    # clipping (a wrong-band or unloaded-weight failure shows up here).
    rms = audio.pow(2).mean().sqrt().item()
    assert 1e-4 < rms < 0.9, f"decoded sound RMS {rms:.5f} out of range"
    print(f"  sound tokenizer decode: shape={tuple(audio.shape)}, rms={rms:.4f}")


def _main() -> None:
    failures = []
    tests = [
        ("sound_segment_layout", test_sound_segment_layout),
        ("sound_frame_math", test_sound_frame_math),
        ("sound_request_validation", test_sound_request_validation),
        ("sound_forward_smoke_cpu", test_sound_forward_smoke_cpu),
        ("sound_tokenizer_tiny_decode", test_sound_tokenizer_tiny_decode),
        ("sound_tokenizer_load_ignores_encoder_keys", test_sound_tokenizer_load_ignores_encoder_keys),
        ("sound_cache_once_matches_fused_exact", test_sound_cache_once_matches_fused_exact),
        ("sound_engine_path_flashinfer", test_sound_engine_path_flashinfer),
        ("sound_tokenizer_decode_real", test_sound_tokenizer_decode_real),
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
    print("\nAll Cosmos3 sound checks passed.")


if __name__ == "__main__":
    _main()
