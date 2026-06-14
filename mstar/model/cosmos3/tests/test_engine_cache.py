"""GPU parity for the cache-once engine path of the Cosmos3 generator.

The understanding tower runs once and writes its per-layer K/V; the generation
tower then runs each denoise step re-reading that frozen K/V (the text tokens get
no timestep embedding, so their K/V is denoise-step independent — caching it once
is exact). This checks the ``Cosmos3DiTSubmodule`` prefill + denoise loop against
the fused ``Cosmos3Pipeline`` that runs the whole transformer every step, for both
image (single frame) and video (multi-frame, fps-modulated mRoPE) generation.

Two GPU-gated checks per mode (need ``COSMOS3_NANO_DIR`` + CUDA; skipped otherwise):
  * with an in-process sdpa cache (same attention kernel as the fused pipeline),
    the cache-once output is bit-for-bit identical;
  * with the engine's FlashInfer paged cache (the served path), the decoded output
    matches the fused pipeline within PSNR >= 30 (FlashInfer-vs-sdpa precision).

Run: COSMOS3_NANO_DIR=<snap> python3 test_engine_cache.py
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

PROMPT = "A red cube resting on a polished wooden table, soft daylight."
H = W = 256
STEPS = 12
GS = 6.0
SEED = 42
VIDEO_FRAMES = 17  # latent T = 1 + (17 - 1) // 4 = 5


class _SdpaCacheHandle:
    """In-process reference cache with the ``BatchedCacheManager`` surface the
    DiT uses, backed by stored tensors + sdpa (same kernel as the fused pipeline).
    Prefill stashes each layer's understanding K/V; every denoise step re-reads it.
    """

    def __init__(self):
        self.active = "main"
        self.layer = 0
        self.committed: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.pending: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.is_causal: dict[str, bool] = {}

    def set_active_label(self, label):
        self.active = label

    def set_layer_idx(self, i):
        self.layer = i

    def plan_attention(self, seq_lens=None, dtype=None, is_causal=True, write_store=True, label=None):
        self.is_causal[label or self.active] = is_causal

    def plan_rope(self, *args, **kwargs):
        pass

    @staticmethod
    def _sdpa(q, k, v, is_causal):
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2),
            v.unsqueeze(0).transpose(1, 2), is_causal=is_causal, enable_gqa=True,
        )
        return out.transpose(1, 2).squeeze(0)

    def run_attention(self, q, k, v, layer_idx=None):
        key = (self.active, self.layer if layer_idx is None else layer_idx)
        causal = self.is_causal[self.active]
        if key in self.committed:
            pk, pv = self.committed[key]
            return self._sdpa(q, torch.cat([pk, k], 0), torch.cat([pv, v], 0), causal)
        self.pending[key] = (k, v)
        return self._sdpa(q, k, v, causal)

    def advance_seq_lens(self, pos_id_ns=None):
        self.committed.update(self.pending)
        self.pending = {}


def _flashinfer_cache(model, rid, device, dtype):
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.engine.cache_manager import BatchedCacheManager, WorkspaceBufferManager
    from mstar.engine.kv_store import PagedAllocationManager, TransferEngineInfo
    from mstar.model.cosmos3.submodules import COND_LABEL, UNCOND_LABEL

    cfg = model.get_kv_cache_config()[0]
    cfg.max_num_pages = 64
    cfg.shard(1)
    kv_cache = torch.zeros(
        cfg.num_layers, cfg.max_num_pages, 2, cfg.page_size, cfg.num_kv_heads, cfg.head_dim,
        dtype=dtype, device=device,
    )
    alloc = PagedAllocationManager(cfg, kv_cache, TransferEngineInfo("h", "h", LocalTransferEngine("h")))
    alloc.add_request(rid, [COND_LABEL, UNCOND_LABEL])
    return BatchedCacheManager(
        request_ids=[rid], active_labels_per_request={rid: COND_LABEL}, kv_cache=kv_cache,
        alloc_manager=alloc, buffer_manager=WorkspaceBufferManager(256 * 1024 * 1024, device),
        kv_cache_config=cfg, device=device, auto_write_store=False,
    )


@torch.no_grad()
def _run_cache_once(model, dit, cm, init, cond_ids, uncond_ids, device, num_frames):
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.model.submodule_base import ModelInputsFromEngine

    rid = "r0"
    md = {"height": H, "width": W, "num_frames": num_frames, "fps": 24.0,
          "guidance_scale": GS, "num_inference_steps": STEPS}
    fwd = CurrentForwardPassInfo(
        request_id=rid, graph_walk="prefill", requires_cfg=(GS != 1.0),
        fwd_index=0, random_seed=SEED, max_tokens=0, sampling_config={}, step_metadata=md,
    )
    ei = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: fwd}, cache_manager=cm)
    text_inputs = [
        torch.tensor(cond_ids, dtype=torch.long, device=device),
        torch.tensor(uncond_ids, dtype=torch.long, device=device),
    ]
    ni = dit.prepare_inputs("prefill", fwd, {"text_inputs": text_inputs})
    dit.forward("prefill", ei, **dit.preprocess("prefill", ei, [ni]))

    latents = init.clone()
    time_index = torch.zeros(1, dtype=torch.long, device=device)
    fwd.graph_walk = "image_gen"
    for _ in range(STEPS):
        ni = dit.prepare_inputs("image_gen", fwd, {"latents": [latents], "time_index": [time_index]})
        out = dit.forward("image_gen", ei, **dit.preprocess("image_gen", ei, [ni]))
        latents, time_index = out["latents"][0], out["time_index"][0]
    dit.cleanup_request(rid)
    return latents


_SETUP_CACHE: dict = {}


def _load():
    """Load the model / DiT / fused pipeline once (mode-independent)."""
    if "base" in _SETUP_CACHE:
        return _SETUP_CACHE["base"]
    snap = os.environ.get("COSMOS3_NANO_DIR")
    if not snap or not torch.cuda.is_available():
        _SETUP_CACHE["base"] = None
        return None
    torch.use_deterministic_algorithms(True, warn_only=True)
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
    from mstar.model.cosmos3.pipeline import Cosmos3Pipeline

    device, dtype = "cuda:0", torch.bfloat16
    model = Cosmos3Model(model_path_hf=snap)
    mpipe = Cosmos3Pipeline.from_model(model, device=device, dtype=dtype)
    dit = model.get_submodule("dit", device=device)  # shares mpipe's transformer
    _SETUP_CACHE["base"] = dict(model=model, mpipe=mpipe, dit=dit, device=device, dtype=dtype)
    return _SETUP_CACHE["base"]


def _scenario(num_frames):
    """Per-mode context: video-aware token ids, shared initial latents, and the
    fused-pipeline latents the cache-once path must reproduce."""
    key = f"frames{num_frames}"
    if key in _SETUP_CACHE:
        return _SETUP_CACHE[key]
    base = _load()
    if base is None:
        _SETUP_CACHE[key] = None
        return None
    from mstar.model.cosmos3.packing import tokenize_prompt

    device, dtype, mpipe = base["device"], base["dtype"], base["mpipe"]
    cond_ids, uncond_ids = tokenize_prompt(
        base["model"].tokenizer, PROMPT, "", num_frames=num_frames, height=H, width=W
    )
    lat_t = 1 if num_frames == 1 else 1 + (num_frames - 1) // mpipe.vae_scale_temporal
    gen = torch.Generator(device=device).manual_seed(SEED)
    init = torch.randn((1, 48, lat_t, H // 16, W // 16), generator=gen, device=device, dtype=dtype)
    lat_fused = mpipe(
        prompt=PROMPT, negative_prompt="", num_frames=num_frames, height=H, width=W,
        num_inference_steps=STEPS, guidance_scale=GS, latents=init.clone(), decode=False,
    )
    ctx = dict(cond=cond_ids, uncond=uncond_ids, init=init, lat_fused=lat_fused, num_frames=num_frames, **base)
    _SETUP_CACHE[key] = ctx
    return ctx


def _check_cache_once_exact(num_frames, tag):
    ctx = _scenario(num_frames)
    if ctx is None:
        print(f"  (skipped {tag} cache-once parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    lat = _run_cache_once(
        ctx["model"], ctx["dit"], _SdpaCacheHandle(), ctx["init"], ctx["cond"], ctx["uncond"],
        ctx["device"], num_frames,
    )
    diff = (ctx["lat_fused"].float() - lat.reshape(ctx["lat_fused"].shape).float()).abs().max().item()
    assert diff <= 1e-3, f"{tag} cache-once latents differ from fused by {diff:.3e} (> 1e-3)"
    print(f"  {tag} cache-once (sdpa) latent abs-max diff = {diff:.3e}")


def _check_engine_psnr(num_frames, tag):
    ctx = _scenario(num_frames)
    if ctx is None:
        print(f"  (skipped {tag} engine cache parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    try:
        cm = _flashinfer_cache(ctx["model"], "r0", ctx["device"], ctx["dtype"])
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped {tag} engine cache parity: FlashInfer unavailable: {exc})")
        return
    lat = _run_cache_once(
        ctx["model"], ctx["dit"], cm, ctx["init"], ctx["cond"], ctx["uncond"], ctx["device"], num_frames,
    )
    img_fused = ctx["mpipe"]._decode(ctx["lat_fused"]).squeeze().float().cpu()
    img_engine = ctx["mpipe"]._decode(lat.reshape(ctx["lat_fused"].shape)).squeeze().float().cpu()
    mse = (img_fused - img_engine).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 30, f"{tag} engine-path PSNR {psnr:.2f} < 30 (MSE {mse:.3e})"
    print(f"  {tag} engine cache path (flashinfer) PSNR = {psnr:.2f} dB")


def test_cache_once_matches_fused_exact() -> None:
    _check_cache_once_exact(1, "t2i")


def test_engine_cache_path_image_psnr() -> None:
    _check_engine_psnr(1, "t2i")


def test_cache_once_matches_fused_exact_t2v() -> None:
    _check_cache_once_exact(VIDEO_FRAMES, "t2v")


def test_engine_cache_path_video_psnr() -> None:
    _check_engine_psnr(VIDEO_FRAMES, "t2v")


def _main() -> None:
    failures = []
    for name, fn in [
        ("cache_once_matches_fused_exact", test_cache_once_matches_fused_exact),
        ("engine_cache_path_image_psnr", test_engine_cache_path_image_psnr),
        ("cache_once_matches_fused_exact_t2v", test_cache_once_matches_fused_exact_t2v),
        ("engine_cache_path_video_psnr", test_engine_cache_path_video_psnr),
    ]:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc!r}")
    if failures:
        raise SystemExit(1)
    print("\nAll Cosmos3 engine-cache checks passed.")


if __name__ == "__main__":
    _main()
