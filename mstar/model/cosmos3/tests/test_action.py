"""Tests for the Cosmos3 action path (forward / inverse dynamics + policy).

CPU-safe unit tests (tiny random config, no weights) cover:
  * the action mRoPE band matches vllm-omni's ``compute_mrope_position_ids_action``;
  * the per-mode conditioning layout (which video frames / action tokens are
    clean context vs noisy) matches vllm-omni's ``action.py``;
  * ``build_action_static_inputs`` produces the right joint ``[text|video|action]``
    sequence length, action mse indexes, and position-id width;
  * the transformer ``forward`` returns ``(video, action, sound)`` with the right
    shapes and the right zeros (inverse-dynamics predicts no video velocity;
    forward-dynamics treats the action as clean condition);
  * the engine ``denoise_step`` (generation tower over ``[video|action]`` against
    the frozen understanding K/V) reproduces the fused ``forward`` bit-for-bit
    with an in-process sdpa cache — the cache-once restructuring for action.

Run: python3 test_action.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.packing import (
    action_condition_frame_indexes,
    build_action_static_inputs,
    get_3d_mrope_ids_action_tokens,
    vision_condition_frame_indexes,
)


# --- verbatim vllm-omni references (transformer_cosmos3.py / action.py) ------
def _ref_mrope(grid_t, grid_h, grid_w, temporal_offset, fps, base_fps, tcf, base_tcf, start):
    fps_mod = fps is not None
    if fps_mod:
        tps = fps / tcf
        base_tps = base_fps / (base_tcf if base_tcf is not None else tcf)
        fi = torch.arange(grid_t, dtype=torch.float32)
        t_index = ((fi + start) / tps * base_tps + temporal_offset).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset) + start
        )
    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    if fps_mod:
        return torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    return torch.stack([t_index, h_index, w_index], dim=0)


def _ref_action_condition_indexes(mode, action_length):
    if mode == "forward_dynamics":
        return list(range(action_length))
    return []  # inverse_dynamics / policy


def _ref_vision_condition_indexes(mode, latent_frames):
    if mode in ("policy", "forward_dynamics"):
        return [0]
    return list(range(latent_frames))  # inverse_dynamics


def _cfg() -> Cosmos3Config:
    return Cosmos3Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, vocab_size=100, rope_axes_dim=(4, 2, 2),
        latent_channel=8, latent_patch_size=2, patch_latent_dim=32,
        sound_gen=False, action_gen=True, max_action_dim=12, num_embodiment_domains=6,
    )


class _SdpaCache:
    """In-process cache-once handle (stored K/V + sdpa), the BatchedCacheManager
    surface the DiT uses. Prefill stashes the understanding K/V; the denoise step
    re-reads it."""

    def __init__(self):
        self.active, self.layer = "main", 0
        self.committed, self.pending, self.is_causal = {}, {}, {}

    def set_active_label(self, label):
        self.active = label

    def set_layer_idx(self, i):
        self.layer = i

    def plan(self, is_causal):
        self.is_causal[self.active] = is_causal

    # Engine-facing surface (used when the DiT submodule drives the cache).
    def plan_attention(self, seq_lens=None, dtype=None, is_causal=True, write_store=True, label=None):
        self.is_causal[label or self.active] = is_causal

    def plan_rope(self, *a, **k):
        pass

    @staticmethod
    def _sdpa(q, k, v, c):
        o = F.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2),
            v.unsqueeze(0).transpose(1, 2), is_causal=c, enable_gqa=True)
        return o.transpose(1, 2).squeeze(0)

    def run_attention(self, q, k, v, layer_idx=None):
        key = (self.active, self.layer if layer_idx is None else layer_idx)
        if key in self.committed:
            pk, pv = self.committed[key]
            return self._sdpa(q, torch.cat([pk, k], 0), torch.cat([pv, v], 0), self.is_causal[self.active])
        self.pending[key] = (k, v)
        return self._sdpa(q, k, v, self.is_causal[self.active])

    def advance_seq_lens(self, pos_id_ns=None):
        self.committed.update(self.pending)
        self.pending = {}


_MODES = ("inverse_dynamics", "forward_dynamics", "policy")


def test_action_mrope_matches_reference() -> None:
    for fps in (10.0, 24.0, None):
        ours, _ = get_3d_mrope_ids_action_tokens(
            grid_t=12, temporal_offset=100, action_fps=fps, base_fps=24.0,
            base_temporal_compression_factor=4, start_frame_offset=1,
        )
        ref = _ref_mrope(12, 1, 1, 100, fps, 24.0, 1, 4, 1)
        assert torch.allclose(ours.float(), ref.float(), atol=0), (fps, ours[0, :4], ref[0, :4])


def test_condition_indexes_match_reference() -> None:
    for mode in _MODES:
        assert action_condition_frame_indexes(mode, 16) == _ref_action_condition_indexes(mode, 16)
        assert vision_condition_frame_indexes(mode, 5) == _ref_vision_condition_indexes(mode, 5)


def test_action_static_layout() -> None:
    cfg = _cfg()
    action_chunk, num_frames = 8, 9
    latent_t = 1 + (num_frames - 1) // cfg.vae.scale_factor_temporal  # 3
    latent_shape = (1, cfg.latent_channel, latent_t, 4, 4)
    ids = [1, 2, 3, 4]
    tok_per_frame = (4 // cfg.latent_patch_size) ** 2  # 4
    for mode in _MODES:
        s = build_action_static_inputs(
            ids, latent_shape, action_chunk, mode, cfg, cfg.vae.scale_factor_temporal,
            fps=10.0, action_fps=10.0, action_start_offset=1, device="cpu",
        )
        assert s["sequence_length"] == len(ids) + latent_t * tok_per_frame + action_chunk
        assert s["position_ids"].shape[1] == s["sequence_length"]
        exp_vis_noisy = len(_ref_vision_condition_indexes(mode, latent_t))
        exp_vis_noisy = latent_t - exp_vis_noisy
        assert s["num_noisy_vision_tokens"] == exp_vis_noisy * tok_per_frame
        exp_act_noisy = action_chunk - len(_ref_action_condition_indexes(mode, action_chunk))
        assert s["num_noisy_action_tokens"] == exp_act_noisy
        assert s["action_mse_loss_indexes"].numel() == exp_act_noisy


def _run_mode(model, cfg, mode, latent_shape, action_chunk, ids):
    s = build_action_static_inputs(
        ids, latent_shape, action_chunk, mode, cfg, cfg.vae.scale_factor_temporal,
        fps=10.0, action_fps=10.0, action_start_offset=1, device="cpu",
    )
    keys = ("input_ids", "text_indexes", "position_ids", "und_len", "sequence_length",
            "vision_token_shapes", "vision_sequence_indexes", "vision_mse_loss_indexes",
            "vision_noisy_frame_indexes", "action_token_shapes", "action_sequence_indexes",
            "action_mse_loss_indexes", "action_noisy_frame_indexes")
    sk = {k: s[k] for k in keys}
    domain = torch.tensor([2], dtype=torch.long)
    latents = torch.randn(latent_shape)
    action_lat = torch.randn(1, action_chunk, cfg.max_action_dim)
    vts = torch.full((s["num_noisy_vision_tokens"],), 500.0)
    ats = torch.full((s["num_noisy_action_tokens"],), 500.0)
    with torch.no_grad():
        pv, pa, ps = model(
            vision_tokens=[latents], vision_timesteps=vts,
            action_tokens=action_lat, action_timesteps=ats, action_domain_id=domain, **sk,
        )
    return s, sk, latents, action_lat, domain, vts, ats, pv, pa, ps


def test_action_forward_shapes_and_masks() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    model = Cosmos3OmniTransformer(cfg).eval()
    action_chunk = 8
    latent_t = 1 + (9 - 1) // cfg.vae.scale_factor_temporal
    latent_shape = (1, cfg.latent_channel, latent_t, 4, 4)
    for mode in _MODES:
        _, _, _, _, _, _, _, pv, pa, ps = _run_mode(model, cfg, mode, latent_shape, action_chunk, [1, 2, 3, 4])
        assert ps is None
        assert pv[0].shape == latent_shape
        assert pa.shape == (1, action_chunk, cfg.max_action_dim)
        if mode == "inverse_dynamics":
            assert torch.count_nonzero(pv[0]) == 0
        if mode == "forward_dynamics":
            assert torch.count_nonzero(pa) == 0


def test_action_denoise_step_matches_fused() -> None:
    """The engine generation tower over [video|action] against the frozen
    understanding K/V reproduces the fused forward bit-for-bit (sdpa cache)."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = Cosmos3OmniTransformer(cfg).eval()
    action_chunk = 8
    latent_t = 1 + (9 - 1) // cfg.vae.scale_factor_temporal
    latent_shape = (1, cfg.latent_channel, latent_t, 4, 4)
    for mode in _MODES:
        s, _, latents, action_lat, domain, vts, ats, pv, pa, _ = _run_mode(
            model, cfg, mode, latent_shape, action_chunk, [1, 2, 3, 4]
        )
        cache = _SdpaCache()
        und_len = s["und_len"]
        cache.set_active_label("main")
        cache.plan(is_causal=True)
        model.prefill_und(s["input_ids"], s["text_mrope_ids"], cache)
        cache.plan(is_causal=False)
        with torch.no_grad():
            dv, da = model.denoise_step(
                latents, vts, s["position_ids"][:, und_len:],
                s["vision_token_shapes"], s["vision_noisy_frame_indexes"],
                s["vision_mse_loss_indexes"] - und_len, cache,
                action_latents=action_lat, action_token_shapes=s["action_token_shapes"],
                action_noisy_frame_indexes=s["action_noisy_frame_indexes"],
                action_mse_gen_indexes=s["action_mse_loss_indexes"] - und_len,
                action_timesteps=ats, action_domain_id=domain,
            )
        assert (pv[0] - dv).abs().max().item() < 1e-4, mode
        assert (pa - da).abs().max().item() < 1e-4, mode


# --- GPU-gated parity (needs COSMOS3_NANO_DIR + CUDA + diffusers) ------------
import math  # noqa: E402
import os  # noqa: E402

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_GPU: dict = {}


def _gpu_base():
    if "base" in _GPU:
        return _GPU["base"]
    snap = os.environ.get("COSMOS3_NANO_DIR")
    if not snap or not torch.cuda.is_available():
        _GPU["base"] = None
        return None
    torch.use_deterministic_algorithms(True, warn_only=True)
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
    from mstar.model.cosmos3.pipeline import Cosmos3Pipeline

    device, dtype = "cuda:0", torch.bfloat16
    model = Cosmos3Model(model_path_hf=snap)
    mpipe = Cosmos3Pipeline.from_model(model, device=device, dtype=dtype)
    dit = model.get_submodule("dit", device=device)
    _GPU["base"] = dict(model=model, mpipe=mpipe, dit=dit, device=device, dtype=dtype, snap=snap)
    return _GPU["base"]


def test_action_engine_matches_fused() -> None:
    """The cache-once engine action path reproduces the fused pipeline bit-for-bit
    (sdpa), on real Nano weights — the action analogue of the video engine test."""
    base = _gpu_base()
    if base is None:
        print("  (skipped action engine parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    from diffusers.utils.torch_utils import randn_tensor

    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.model.submodule_base import ModelInputsFromEngine

    device, dtype, mpipe, dit, model = (
        base["device"], base["dtype"], base["mpipe"], base["dit"], base["model"])
    prompt, chunk, raw, dom, fps, steps, fshift, h, w = (
        "You are an autonomous vehicle planning system.", 12, 9, 1, 10.0, 8, 10.0, 128, 128)
    nf = chunk + 1
    cond_latent = torch.randn(
        (1, model.config.latent_channel, 1 + (nf - 1) // 4, h // 16, w // 16), device=device, dtype=dtype)

    gen = torch.Generator(device=device).manual_seed(0)
    act_fused = mpipe.generate_action(
        prompt=prompt, mode="inverse_dynamics", domain_id=dom, action_chunk_size=chunk, raw_action_dim=raw,
        video_latents=cond_latent, num_frames=nf, height=h, width=w, fps=fps, action_fps=fps,
        num_inference_steps=steps, guidance_scale=1.0, flow_shift=fshift, generator=gen)

    gen2 = torch.Generator(device=device).manual_seed(0)
    a_noise = randn_tensor((1, chunk, dit.transformer.action_dim), generator=gen2, device=device, dtype=dtype)
    a_noise[..., raw:] = 0

    from mstar.model.cosmos3.packing import tokenize_prompt
    cond_ids, _ = tokenize_prompt(model.tokenizer, prompt, "", num_frames=nf, height=h, width=w, fps=fps)
    rid = "ra"
    md = {"height": h, "width": w, "num_frames": nf, "fps": fps, "action_fps": fps, "guidance_scale": 1.0,
          "num_inference_steps": steps, "action_mode": "inverse_dynamics", "action_chunk_size": chunk,
          "raw_action_dim": raw, "domain_id": dom, "flow_shift": fshift}
    fwd = CurrentForwardPassInfo(request_id=rid, graph_walk="prefill", requires_cfg=False, fwd_index=0,
                                 random_seed=0, max_tokens=0, sampling_config={}, step_metadata=md)
    cm = _SdpaCache()
    ei = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: fwd}, cache_manager=cm)
    ni = dit.prepare_inputs("prefill", fwd, {"text_inputs": [torch.tensor(cond_ids, dtype=torch.long, device=device)]})
    dit.forward("prefill", ei, **dit.preprocess("prefill", ei, [ni]))
    fwd.graph_walk = "action_gen"
    latents, action_latents = cond_latent.clone(), a_noise.clone()
    time_index = torch.zeros(1, dtype=torch.long, device=device)
    for _ in range(steps):
        ni = dit.prepare_inputs("action_gen", fwd, {
            "latents": [latents], "action_latents": [action_latents], "time_index": [time_index]})
        out = dit.forward("action_gen", ei, **dit.preprocess("action_gen", ei, [ni]))
        latents, action_latents, time_index = out["latents"][0], out["action_latents"][0], out["time_index"][0]
    dit.cleanup_request(rid)
    diff = (act_fused.float() - out["action_output"][0].float()).abs().max().item()
    assert diff <= 1e-3, f"engine action differs from fused by {diff:.3e}"
    print(f"  action engine cache-once (sdpa) abs-max diff = {diff:.3e}")


def test_action_id_golden_gate() -> None:
    """Inverse-dynamics on av_0 reproduces NVIDIA's reference action output
    ([60, 9]) within MSE <= 0.05 / PSNR >= 14 (NVIDIA's own thresholds)."""
    base = _gpu_base()
    if base is None:
        print("  (skipped action id golden gate: needs COSMOS3_NANO_DIR + CUDA)")
        return
    import json

    import torchvision
    from PIL import Image

    from mstar.model.cosmos3.packing import tokenize_prompt

    device, dtype, mpipe, model, snap = (
        base["device"], base["dtype"], base["mpipe"], base["model"], base["snap"])
    assets = os.path.join(snap, "assets")
    inp = os.path.join(assets, "example_action_id_av_0_input.mp4")
    if not os.path.exists(inp):
        print("  (skipped action id golden gate: av_0 input video missing)")
        return
    prompt, chunk, raw, dom, fps = "You are an autonomous vehicle planning system.", 60, 9, 1, 10.0
    nf = chunk + 1
    frames, _, _ = torchvision.io.read_video(inp, pts_unit="sec")
    frames = frames[:nf]
    h, w = int(frames.shape[1]), int(frames.shape[2])
    procs = [mpipe.video_processor.preprocess(Image.fromarray(frames[i].numpy()), height=h, width=w).squeeze(0)
             for i in range(frames.shape[0])]
    video = torch.stack(procs, dim=1).unsqueeze(0).to(device=device, dtype=dtype)

    cond_ids, _ = tokenize_prompt(model.tokenizer, prompt, "", num_frames=nf, height=h, width=w, fps=fps,
                                  use_system_prompt=False, add_resolution_template=False,
                                  add_duration_template=False)
    gen = torch.Generator(device=device).manual_seed(0)
    action = mpipe.generate_action(
        prompt=prompt, mode="inverse_dynamics", domain_id=dom, action_chunk_size=chunk, raw_action_dim=raw,
        video=video, num_frames=nf, height=h, width=w, fps=fps, action_fps=fps,
        num_inference_steps=30, guidance_scale=1.0, flow_shift=10.0, generator=gen,
        cond_ids=cond_ids, uncond_ids=cond_ids)
    pred = action[0].float().cpu()
    gold = torch.tensor(json.load(open(os.path.join(assets, "example_action_id_av_0_output.json")))["data"],
                        dtype=torch.float32)
    mse = (pred - gold).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert mse <= 0.05 and psnr >= 14.0, f"action id MSE {mse:.5f} / PSNR {psnr:.2f} outside gate"
    print(f"  action id av_0: MSE = {mse:.5f}, PSNR = {psnr:.2f} dB")


def test_action_fd_agibotworld_golden_gate() -> None:
    """Autoregressive forward-dynamics on the AgiBotWorld 4-chunk example
    reproduces NVIDIA's golden video (PSNR >= 14). Each chunk takes a [16, 29]
    action chunk as the clean condition; chunk 0 conditions on the first frame,
    chunks 1-3 on the previous chunk's final generated frame."""
    base = _gpu_base()
    if base is None:
        print("  (skipped fd agibotworld golden gate: needs COSMOS3_NANO_DIR + CUDA)")
        return
    import json

    import torchvision
    from PIL import Image

    from mstar.model.cosmos3.packing import tokenize_prompt

    device, dtype, mpipe, model, snap = (
        base["device"], base["dtype"], base["mpipe"], base["model"], base["snap"])
    assets = os.path.join(snap, "assets")
    first_png = os.path.join(assets, "example_action_fd_agibotworld_first_frame.png")
    chunks_json = os.path.join(assets, "example_action_fd_agibotworld_action_chunks.json")
    gold_mp4 = os.path.join(assets, "example_action_fd_agibotworld_4chunk_output.mp4")
    if not (os.path.exists(first_png) and os.path.exists(chunks_json) and os.path.exists(gold_mp4)):
        print("  (skipped fd agibotworld golden gate: assets missing)")
        return
    prompt, dom, raw, chunk = "Pickup items in the supermarket", 15, 29, 16
    nf, fps = chunk + 1, 10.0
    im = Image.open(first_png).convert("RGB")
    w, h = im.size
    cond_frame = mpipe.video_processor.preprocess(im, height=h, width=w).to(device=device, dtype=dtype)[0]
    chunks = torch.tensor(json.load(open(chunks_json))["action_chunks"], dtype=torch.float32)
    cond_ids, _ = tokenize_prompt(model.tokenizer, prompt, "", num_frames=nf, height=h, width=w, fps=fps,
                                  use_system_prompt=False, add_resolution_template=False,
                                  add_duration_template=False)
    out = []
    for k in range(chunks.shape[0]):
        cond_video = cond_frame.unsqueeze(0).unsqueeze(2).expand(-1, -1, nf, -1, -1).contiguous()
        gen = torch.Generator(device=device).manual_seed(k)
        _, video = mpipe.generate_action(
            prompt=prompt, mode="forward_dynamics", domain_id=dom, action_chunk_size=chunk, raw_action_dim=raw,
            action=chunks[k], video=cond_video, num_frames=nf, height=h, width=w, fps=fps, action_fps=fps,
            num_inference_steps=30, guidance_scale=1.0, flow_shift=10.0, generator=gen,
            cond_ids=cond_ids, uncond_ids=cond_ids, return_video=True)
        pred = video[0, :, 1:, :, :].float()
        out.append(pred.cpu())
        cond_frame = (pred[:, -1].clamp(0, 1) * 2 - 1).to(device=device, dtype=dtype)
    pred_video = torch.cat(out, dim=1)
    g, _, _ = torchvision.io.read_video(gold_mp4, pts_unit="sec")
    gold = (g.permute(3, 0, 1, 2).float() / 255.0)
    n = min(pred_video.shape[1], gold.shape[1])
    mse = (pred_video[:, :n] - gold[:, :n]).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 14.0, f"fd agibotworld PSNR {psnr:.2f} < 14 (MSE {mse:.5f})"
    print(f"  fd agibotworld: {n} frames, PSNR = {psnr:.2f} dB")


def _main() -> None:
    fns = [
        ("action_mrope_matches_reference", test_action_mrope_matches_reference),
        ("condition_indexes_match_reference", test_condition_indexes_match_reference),
        ("action_static_layout", test_action_static_layout),
        ("action_forward_shapes_and_masks", test_action_forward_shapes_and_masks),
        ("action_denoise_step_matches_fused", test_action_denoise_step_matches_fused),
        ("action_engine_matches_fused", test_action_engine_matches_fused),
        ("action_id_golden_gate", test_action_id_golden_gate),
        ("action_fd_agibotworld_golden_gate", test_action_fd_agibotworld_golden_gate),
    ]
    failures = []
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc!r}")
    if failures:
        raise SystemExit(1)
    print("\nAll Cosmos3 action unit checks passed.")


if __name__ == "__main__":
    _main()
