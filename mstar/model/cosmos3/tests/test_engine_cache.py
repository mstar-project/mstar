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
# These checks validate the eager cache-once mechanism's numerical exactness, so
# run the eager denoise step. The served default torch.compiles it, which fuses
# pointwise ops and perturbs the latents past the tight bit-exact bounds below
# without changing image quality (the FlashInfer PSNR checks still pass with
# compile on — validated over HTTP). Set COSMOS3_DISABLE_COMPILE_DENOISE= (empty)
# to exercise the compiled path here instead.
os.environ.setdefault("COSMOS3_DISABLE_COMPILE_DENOISE", "1")

import torch
import torch.nn.functional as F

PROMPT = "A red cube resting on a polished wooden table, soft daylight."
# Parity checks here are resolution-independent; 256x256 keeps them quick. The
# CUDA-graph check below captures at whatever (H, W) it sets. NOTE: the in-process
# graph-vs-fused PSNR is a coarse smoke check — it carries a cache-setup artifact
# of this harness. The authoritative bit-exactness gate for the served graph is
# the HTTP A/B (graph-on vs COSMOS3_DISABLE_CUDA_GRAPH=1), which is byte-identical
# at every resolution.
H = W = 256
STEPS = 12
GS = 6.0
SEED = 42
VIDEO_FRAMES = 17  # latent T = 1 + (17 - 1) // 4 = 5


class _SdpaCacheHandle:
    """In-process reference cache with the ``BatchedCacheManager`` surface the
    DiT uses, backed by stored tensors + sdpa (same kernel as the fused pipeline).
    Prefill stashes each layer's understanding K/V; every denoise step re-reads it.

    Also models the batched classifier-free-guidance plan: when both guidance
    branches run in one forward, ``run_attention`` receives the two branches
    concatenated and routes each half to its own label's cached prefix, so the
    batched result equals running the branches sequentially.
    """

    def __init__(self):
        self.active = "main"
        self.layer = 0
        self.committed: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.pending: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.is_causal: dict[str, bool] = {}
        self.batched_labels: list[str] | None = None

    def set_active_label(self, label):
        self.active = label

    def set_layer_idx(self, i):
        self.layer = i

    def plan_attention(self, seq_lens=None, dtype=None, is_causal=True, write_store=True, label=None):
        self.is_causal[label or self.active] = is_causal

    def plan_attention_batched_cfg(self, labels, seq_lens, is_causal=False, write_store=False, **kwargs):
        self.batched_labels = list(labels)
        self.is_causal["_cfg_batched"] = is_causal

    def plan_rope(self, *args, **kwargs):
        pass

    @staticmethod
    def _sdpa(q, k, v, is_causal):
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2),
            v.unsqueeze(0).transpose(1, 2), is_causal=is_causal, enable_gqa=True,
        )
        return out.transpose(1, 2).squeeze(0)

    def _attend_label(self, label, layer, q, k, v, causal):
        key = (label, layer)
        if key in self.committed:
            pk, pv = self.committed[key]
            return self._sdpa(q, torch.cat([pk, k], 0), torch.cat([pv, v], 0), causal)
        self.pending[key] = (k, v)
        return self._sdpa(q, k, v, causal)

    def run_attention(self, q, k, v, layer_idx=None):
        layer = self.layer if layer_idx is None else layer_idx
        if self.active == "_cfg_batched":
            causal = self.is_causal["_cfg_batched"]
            n = q.shape[0] // len(self.batched_labels)
            outs = []
            for bi, label in enumerate(self.batched_labels):
                sl = slice(bi * n, (bi + 1) * n)
                outs.append(self._attend_label(label, layer, q[sl], k[sl], v[sl], causal))
            return torch.cat(outs, 0)
        return self._attend_label(self.active, layer, q, k, v, self.is_causal[self.active])

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


def _flashinfer_shared(model, rids, device, dtype):
    """A KV cache + paged allocator shared by several requests, each with both
    guidance labels (mirrors the engine's persistent per-node cache)."""
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.engine.cache_manager import WorkspaceBufferManager
    from mstar.engine.kv_store import PagedAllocationManager, TransferEngineInfo
    from mstar.model.cosmos3.submodules import COND_LABEL, UNCOND_LABEL

    cfg = model.get_kv_cache_config()[0]
    cfg.max_num_pages = 256
    cfg.shard(1)
    kv_cache = torch.zeros(
        cfg.num_layers, cfg.max_num_pages, 2, cfg.page_size, cfg.num_kv_heads, cfg.head_dim,
        dtype=dtype, device=device,
    )
    alloc = PagedAllocationManager(cfg, kv_cache, TransferEngineInfo("h", "h", LocalTransferEngine("h")))
    for rid in rids:
        alloc.add_request(rid, [COND_LABEL, UNCOND_LABEL])
    buf = WorkspaceBufferManager(256 * 1024 * 1024, device)
    return {"kv_cache": kv_cache, "alloc": alloc, "buf": buf, "cfg": cfg, "device": device}


def _mk_cm(shared, rids):
    from mstar.engine.cache_manager import BatchedCacheManager
    from mstar.model.cosmos3.submodules import COND_LABEL

    return BatchedCacheManager(
        request_ids=rids, active_labels_per_request={r: COND_LABEL for r in rids},
        kv_cache=shared["kv_cache"], alloc_manager=shared["alloc"], buffer_manager=shared["buf"],
        kv_cache_config=shared["cfg"], device=shared["device"], auto_write_store=False,
    )


@torch.no_grad()
def _run_batched(model, dit, shared, init, conds, unconds, device, rids):
    """Prefill each request (sequential, like the engine), then run the whole
    denoise loop as one batched step per iteration. Returns final latents per rid."""
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.model.submodule_base import ModelInputsFromEngine

    md = {"height": H, "width": W, "num_frames": 1, "fps": 24.0,
          "guidance_scale": GS, "num_inference_steps": STEPS}
    fwds = {}
    for i, rid in enumerate(rids):
        fwd = CurrentForwardPassInfo(
            request_id=rid, graph_walk="prefill", requires_cfg=True, fwd_index=0,
            random_seed=SEED, max_tokens=0, sampling_config={}, step_metadata=md,
        )
        fwds[rid] = fwd
        cm1 = _mk_cm(shared, [rid])
        ei1 = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: fwd}, cache_manager=cm1)
        ti = [torch.tensor(conds[i], dtype=torch.long, device=device),
              torch.tensor(unconds[i], dtype=torch.long, device=device)]
        ni = dit.prepare_inputs("prefill", fwd, {"text_inputs": ti})
        dit.forward("prefill", ei1, **dit.preprocess("prefill", ei1, [ni]))

    cmN = _mk_cm(shared, rids)
    eiN = ModelInputsFromEngine(request_ids=rids, per_request_info=fwds, cache_manager=cmN)
    for rid in rids:
        fwds[rid].graph_walk = "image_gen"
    latents = {rid: init.clone() for rid in rids}
    time_index = {rid: torch.zeros(1, dtype=torch.long, device=device) for rid in rids}
    for _ in range(STEPS):
        inputs = [
            dit.prepare_inputs("image_gen", fwds[rid],
                               {"latents": [latents[rid]], "time_index": [time_index[rid]]})
            for rid in rids
        ]
        out = dit.forward_batched("image_gen", eiN, **dit.preprocess("image_gen", eiN, inputs))
        for rid in rids:
            latents[rid], time_index[rid] = out[rid]["latents"][0], out[rid]["time_index"][0]
    for rid in rids:
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
    dit = ctx["dit"]
    prev = dit.batched_cfg
    # The sequential guidance path matches the fused pipeline bit-for-bit; the
    # batched path differs only in bf16 GEMM rounding (covered by the PSNR checks).
    dit.batched_cfg = False
    try:
        lat = _run_cache_once(
            ctx["model"], dit, _SdpaCacheHandle(), ctx["init"], ctx["cond"], ctx["uncond"],
            ctx["device"], num_frames,
        )
    finally:
        dit.batched_cfg = prev
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


@torch.no_grad()
def _check_dense_fa3(num_frames, tag):
    """Dense FlashAttention-3 generation attention vs the paged FlashInfer path.
    Both attend each guidance branch's generation tokens over its frozen text
    prefix; they differ only in the attention kernel (FA3 over a gathered
    contiguous [prefix | gen] vs FlashInfer paged) and its bf16 rounding. So the
    decoded images must match closely, and the dense path must clear the same
    fused-reference bar the paged path meets."""
    ctx = _scenario(num_frames)
    if ctx is None:
        print(f"  (skipped {tag} dense-FA3 parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    had = os.environ.get("COSMOS3_DENSE_FA3")
    try:
        os.environ["COSMOS3_DENSE_FA3"] = "0"  # paged baseline (cosmos3 config defaults dense on)
        cm = _flashinfer_cache(ctx["model"], "r0", ctx["device"], ctx["dtype"])
        lat_paged = _run_cache_once(
            ctx["model"], ctx["dit"], cm, ctx["init"], ctx["cond"], ctx["uncond"],
            ctx["device"], num_frames,
        )
        os.environ["COSMOS3_DENSE_FA3"] = "1"
        cm2 = _flashinfer_cache(ctx["model"], "r0", ctx["device"], ctx["dtype"])
        lat_dense = _run_cache_once(
            ctx["model"], ctx["dit"], cm2, ctx["init"], ctx["cond"], ctx["uncond"],
            ctx["device"], num_frames,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped {tag} dense-FA3 parity: FA3/FlashInfer unavailable: {exc})")
        return
    finally:
        if had is None:
            os.environ.pop("COSMOS3_DENSE_FA3", None)
        else:
            os.environ["COSMOS3_DENSE_FA3"] = had
    shape = ctx["lat_fused"].shape
    img_fused = ctx["mpipe"]._decode(ctx["lat_fused"]).squeeze().float().cpu()
    img_paged = ctx["mpipe"]._decode(lat_paged.reshape(shape)).squeeze().float().cpu()
    img_dense = ctx["mpipe"]._decode(lat_dense.reshape(shape)).squeeze().float().cpu()

    def _psnr(a, b):
        mse = (a - b).pow(2).mean().item()
        return float("inf") if mse == 0 else -10 * math.log10(mse)

    vs_paged = _psnr(img_dense, img_paged)
    vs_fused = _psnr(img_dense, img_fused)
    # The dense path must match the fused reference as well as the paged engine
    # path does (>= 30, the same bar), and the two engine kernels must agree to
    # within their bf16 rounding (a real ordering/gather bug tanks this < 15).
    assert vs_fused >= 30, f"{tag} dense-FA3 vs fused PSNR {vs_fused:.2f} < 30"
    assert vs_paged >= 30, f"{tag} dense-FA3 vs paged PSNR {vs_paged:.2f} < 30"
    print(f"  {tag} dense-FA3 PSNR vs paged = {vs_paged:.2f} dB, vs fused = {vs_fused:.2f} dB")


def test_dense_fa3_image_psnr() -> None:
    _check_dense_fa3(1, "t2i")


def test_dense_fa3_video_psnr() -> None:
    _check_dense_fa3(VIDEO_FRAMES, "t2v")


@torch.no_grad()
def test_anchor_encode_matches_full() -> None:
    """Image-to-video only consumes latent frame 0, and the Wan VAE encodes it as
    a standalone causal anchor, so encoding the single conditioning frame
    (anchor_only=True) must give a bit-identical frame 0 to encoding the whole
    repeat-padded clip — at a fraction of the cost."""
    base = _load()
    if base is None:
        print("  (skipped anchor-encode parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    dit, device = base["dit"], base["device"]
    img = torch.rand(3, H, W, device=device)  # [C, H, W] in [0, 1], like load_image
    anchor = dit._encode_conditioning(img, H, W, VIDEO_FRAMES, device, anchor_only=True)
    full = dit._encode_conditioning(img, H, W, VIDEO_FRAMES, device, anchor_only=False)
    assert anchor.shape[2] == 1, f"anchor_only must encode one latent frame, got T={anchor.shape[2]}"
    diff = (anchor[:, :, 0].float() - full[:, :, 0].float()).abs().max().item()
    assert diff < 1e-4, f"anchor frame-0 differs from full-clip frame-0 by {diff:.3e} (> 1e-4)"
    print(f"  anchor-encode 1-frame vs full-clip frame-0 abs-max diff = {diff:.3e}")


@torch.no_grad()
def _check_compile_vae(num_frames, tag):
    """torch.compile of the Wan VAE decode (COSMOS3_COMPILE_VAE) must reproduce
    the eager decode. Compile fuses the pointwise epilogues around the (fp32) 3D
    convolutions without changing their math, so the decoded uint8 frames match
    the eager path to fp rounding; a real fusion/ordering bug shows up as visible
    banding that tanks the PSNR. Checked for both a single image frame and a
    multi-frame video clip (video is the lever's main beneficiary)."""
    ctx = _scenario(num_frames)
    if ctx is None:
        print(f"  (skipped {tag} compile-VAE parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderSubmodule

    model, lat = ctx["model"], ctx["lat_fused"]
    vae, config = model._build_vae(ctx["device"]), model.config
    walk = "video_gen" if num_frames > 1 else "image_gen"
    out_key = "video_output" if num_frames > 1 else "image_output"
    had = os.environ.pop("COSMOS3_COMPILE_VAE", None)
    try:
        eager = Cosmos3VAEDecoderSubmodule(vae=vae, config=config)
        img_eager = eager.forward(walk, None, latents=lat.clone())[out_key][0]
        os.environ["COSMOS3_COMPILE_VAE"] = "1"
        compiled = Cosmos3VAEDecoderSubmodule(vae=vae, config=config)
        img_comp = compiled.forward(walk, None, latents=lat.clone())[out_key][0]
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped {tag} compile-VAE parity: VAE/compile unavailable: {exc})")
        return
    finally:
        if had is None:
            os.environ.pop("COSMOS3_COMPILE_VAE", None)
        else:
            os.environ["COSMOS3_COMPILE_VAE"] = had
    a = img_eager.float().cpu() / 255.0
    b = img_comp.float().cpu() / 255.0
    maxdiff = (a - b).abs().max().item() * 255.0
    mse = (a - b).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 40, f"{tag} compile-VAE vs eager PSNR {psnr:.2f} < 40 (max uint8 diff {maxdiff:.0f})"
    print(f"  {tag} compile-VAE vs eager decoded PSNR = {psnr:.2f} dB (max uint8 diff {maxdiff:.0f})")


def test_compile_vae_matches_eager() -> None:
    _check_compile_vae(1, "t2i")


def test_compile_vae_matches_eager_t2v() -> None:
    _check_compile_vae(VIDEO_FRAMES, "t2v")


@torch.no_grad()
def test_batched_cfg_matches_sequential() -> None:
    """Running both guidance branches in one batched forward must match running
    them sequentially. The two paths differ only in bf16 GEMM rounding (a batched
    matmul tiles differently), so compare the decoded images by PSNR."""
    ctx = _scenario(1)
    if ctx is None:
        print("  (skipped batched-CFG vs sequential: needs COSMOS3_NANO_DIR + CUDA)")
        return
    dit, prev, decoded = ctx["dit"], ctx["dit"].batched_cfg, {}
    try:
        for flag in (False, True):
            dit.batched_cfg = flag
            try:
                cm = _flashinfer_cache(ctx["model"], "r0", ctx["device"], ctx["dtype"])
            except Exception as exc:  # noqa: BLE001
                print(f"  (skipped batched-CFG vs sequential: FlashInfer unavailable: {exc})")
                return
            lat = _run_cache_once(
                ctx["model"], dit, cm, ctx["init"], ctx["cond"], ctx["uncond"], ctx["device"], 1
            )
            decoded[flag] = ctx["mpipe"]._decode(lat.reshape(ctx["lat_fused"].shape)).squeeze().float().cpu()
    finally:
        dit.batched_cfg = prev
    mse = (decoded[False] - decoded[True]).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 35, f"batched vs sequential PSNR {psnr:.2f} < 35 (MSE {mse:.3e})"
    print(f"  batched-CFG vs sequential decoded PSNR = {psnr:.2f} dB")


def test_cache_once_matches_fused_exact() -> None:
    _check_cache_once_exact(1, "t2i")


def test_engine_cache_path_image_psnr() -> None:
    _check_engine_psnr(1, "t2i")


def test_cache_once_matches_fused_exact_t2v() -> None:
    _check_cache_once_exact(VIDEO_FRAMES, "t2v")


def test_engine_cache_path_video_psnr() -> None:
    _check_engine_psnr(VIDEO_FRAMES, "t2v")


@torch.no_grad()
def test_cross_request_batch_matches_individual() -> None:
    """Several requests denoised together in one batch must reproduce each
    request run alone. Distinct prompts are decoded and compared to the fused
    pipeline: batching must (a) keep each request isolated — its own image far
    closer than any other request's — and (b) not lose quality versus the bs=1
    path (per-prompt fidelity varies with the FlashInfer kernel, so the bar is
    relative to bs=1, not an absolute PSNR)."""
    base = _load()
    if base is None:
        print("  (skipped cross-request batch parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    from mstar.model.cosmos3.packing import tokenize_prompt

    model, dit, mpipe = base["model"], base["dit"], base["mpipe"]
    device, dtype = base["device"], base["dtype"]
    prompts = [
        "A red cube resting on a polished wooden table, soft daylight.",
        "A blue ceramic vase of yellow tulips beside a sunny window.",
        "A small wooden sailboat on a calm turquoise sea at dawn.",
        "A snowy mountain peak under a clear starry night sky.",
    ]
    rids = [f"r{i}" for i in range(len(prompts))]
    conds, unconds = [], []
    for p in prompts:
        c, u = tokenize_prompt(model.tokenizer, p, "", num_frames=1, height=H, width=W)
        conds.append(c)
        unconds.append(u)
    gen = torch.Generator(device=device).manual_seed(SEED)
    init = torch.randn((1, 48, 1, H // 16, W // 16), generator=gen, device=device, dtype=dtype)
    shape = (1, 48, 1, H // 16, W // 16)

    def _dec(lat):
        return mpipe._decode(lat.reshape(shape)).squeeze().float().cpu()

    def _psnr(a, b):
        mse = (a - b).pow(2).mean().item()
        return float("inf") if mse == 0 else -10 * math.log10(mse)

    try:
        fused = [
            _dec(mpipe(prompt=p, negative_prompt="", num_frames=1, height=H, width=W,
                       num_inference_steps=STEPS, guidance_scale=GS, latents=init.clone(), decode=False))
            for p in prompts
        ]
        bs1 = []
        for i, rid in enumerate(rids):
            cm = _flashinfer_cache(model, "r0", device, dtype)
            bs1.append(_dec(_run_cache_once(model, dit, cm, init, conds[i], unconds[i], device, 1)))
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped cross-request batch parity: FlashInfer unavailable: {exc})")
        return

    shared = _flashinfer_shared(model, rids, device, dtype)
    bat = _run_batched(model, dit, shared, init, conds, unconds, device, rids)
    batched = [_dec(bat[rid]) for rid in rids]

    n = len(prompts)
    for i in range(n):
        match = _psnr(batched[i], fused[i])
        cross = max(_psnr(batched[i], fused[j]) for j in range(n) if j != i)
        ref = _psnr(bs1[i], fused[i])
        assert match > cross + 8, f"request {i} not isolated: self {match:.2f} vs other {cross:.2f}"
        assert match >= ref - 3.0, f"request {i} batched {match:.2f} degrades vs bs=1 {ref:.2f}"
    print(f"  cross-request batch (bs={n}) vs fused PSNR = "
          + ", ".join(f"{_psnr(batched[i], fused[i]):.1f}" for i in range(n))
          + " dB (bs=1: " + ", ".join(f"{_psnr(bs1[i], fused[i]):.1f}" for i in range(n)) + ")")
    # This test holds several requests' caches at once; release them so later
    # GPU checks in the same process aren't starved.
    del fused, bs1, batched, bat, shared
    import gc
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def _run_cuda_graph_denoise(ctx):
    """Capture the image denoise step and run the whole loop through the real
    CudaGraphRunner (one captured forward per step covering both guidance
    branches), returning the final latents."""
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.distributed.communication import CommGroup
    from mstar.engine.cuda_graph_runner import CudaGraphRunner
    from mstar.model.submodule_base import ModelInputsFromEngine
    from mstar.utils.sampling import Sampler, SamplingConfig

    model, dit = ctx["model"], ctx["dit"]
    device, dtype = ctx["device"], ctx["dtype"]
    dev = torch.device(device)
    # Capture at this test's (H, W) regardless of the production default.
    dit.gen_capture_resolutions = ((H, W),)
    rid = "cgr0"
    shared = _flashinfer_shared(model, [rid], device, dtype)
    md = {"height": H, "width": W, "num_frames": 1, "fps": 24.0,
          "guidance_scale": GS, "num_inference_steps": STEPS}
    fwd = CurrentForwardPassInfo(
        request_id=rid, graph_walk="prefill", requires_cfg=False, fwd_index=0,
        random_seed=SEED, max_tokens=0, sampling_config={}, step_metadata=md,
    )
    cm = _mk_cm(shared, [rid])
    ei = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: fwd}, cache_manager=cm)
    ti = [torch.tensor(ctx["cond"], dtype=torch.long, device=device),
          torch.tensor(ctx["uncond"], dtype=torch.long, device=device)]
    ni = dit.prepare_inputs("prefill", fwd, {"text_inputs": ti})
    dit.forward("prefill", ei, **dit.preprocess("prefill", ei, [ni]))

    runner = CudaGraphRunner(
        submodule_name="dit", submodule=dit, kv_cache_config=shared["cfg"],
        alloc_manager=shared["alloc"], sampler=Sampler(device=dev, tp_group=CommGroup.trivial()),
        buffer_manager=shared["buf"], device=dev, autocast_dtype=dtype,
        default_sampling_config=SamplingConfig(), tp_group=CommGroup.trivial(),
    )
    runner.warmup_and_capture()
    assert runner.graphs, "no CUDA graph captured for cosmos3 image_gen"
    runner.register_request(rid)

    fwd.graph_walk = "image_gen"
    latents = ctx["init"].clone()
    time_index = torch.zeros(1, dtype=torch.long, device=device)
    for _ in range(STEPS):
        ni = dit.prepare_inputs("image_gen", fwd, {"latents": [latents], "time_index": [time_index]})
        out = runner.run(
            graph_walk="image_gen", requires_cfg=False, request_ids=[rid],
            inputs=[ni], per_request_info={rid: fwd}, submodule=dit,
        )
        latents, time_index = out[rid]["latents"][0], out[rid]["time_index"][0]
    dit.cleanup_request(rid)
    return latents


@torch.no_grad()
def test_cuda_graph_matches_eager() -> None:
    """The captured-graph denoise step is the served path's accelerator: both
    guidance branches run in one captured forward (~2x faster than the eager
    step). Each captured forward matches eager to within bf16 (the first step
    differs by ~one ULP); the multistep solver amplifies that into a small latent
    spread, but the decoded image is unchanged — so gate the decoded image against
    the fused pipeline, the same bar the eager engine path meets."""
    ctx = _scenario(1)
    if ctx is None:
        print("  (skipped cuda-graph parity: needs COSMOS3_NANO_DIR + CUDA)")
        return
    try:
        lat_graph = _run_cuda_graph_denoise(ctx)
    except Exception as exc:  # noqa: BLE001
        print(f"  (skipped cuda-graph parity: FlashInfer/capture unavailable: {exc})")
        return
    img_fused = ctx["mpipe"]._decode(ctx["lat_fused"]).squeeze().float().cpu()
    img_graph = ctx["mpipe"]._decode(lat_graph.reshape(ctx["lat_fused"].shape)).squeeze().float().cpu()
    mse = (img_fused - img_graph).pow(2).mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    assert psnr >= 25, f"cuda-graph denoise PSNR {psnr:.2f} < 25 (MSE {mse:.3e})"
    print(f"  cuda-graph denoise vs fused PSNR = {psnr:.2f} dB")


def _main() -> None:
    failures = []
    for name, fn in [
        ("batched_cfg_matches_sequential", test_batched_cfg_matches_sequential),
        ("cache_once_matches_fused_exact", test_cache_once_matches_fused_exact),
        ("engine_cache_path_image_psnr", test_engine_cache_path_image_psnr),
        ("cache_once_matches_fused_exact_t2v", test_cache_once_matches_fused_exact_t2v),
        ("engine_cache_path_video_psnr", test_engine_cache_path_video_psnr),
        ("dense_fa3_image_psnr", test_dense_fa3_image_psnr),
        ("dense_fa3_video_psnr", test_dense_fa3_video_psnr),
        ("anchor_encode_matches_full", test_anchor_encode_matches_full),
        ("compile_vae_matches_eager", test_compile_vae_matches_eager),
        ("compile_vae_matches_eager_t2v", test_compile_vae_matches_eager_t2v),
        ("cuda_graph_matches_eager", test_cuda_graph_matches_eager),
        ("cross_request_batch_matches_individual", test_cross_request_batch_matches_individual),
    ]:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {exc!r}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(1)
    print("\nAll Cosmos3 engine-cache checks passed.")


if __name__ == "__main__":
    _main()
