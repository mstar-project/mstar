"""Parity for the opt-in GPU image preprocess (MSTAR_GPU_IMAGE_PREPROCESS=1) vs
HF's ``Qwen2VLImageProcessor``.

When enabled, ``_gpu_image_preprocess`` (qwen3_omni_model.py) does resize +
rescale + normalize + patchify entirely on-device, replacing the CPU round-trip
through HF's image processor. The native vision encoder is parity-tested against
HF, so the ``pixel_values`` / ``image_grid_thw`` it consumes must match HF's. This
pins that:

  * ``image_grid_thw`` is BIT-EXACT (it drives token counts / positional layout —
    any drift desyncs the whole multimodal prompt), across resolutions including
    non-multiple-of-patch sizes, a small (<min_pixels) image, and a large (~3000px)
    image.
  * ``pixel_values`` matches HF within a cosine threshold. The GPU path uses the
    same torchvision bicubic+antialias kernel as HF's *fast* image-processor
    backend; the residual is CPU-vs-GPU bicubic rounding (a few uint8 levels). We
    compare against the fast backend and require cos > 0.999 (the module docstring
    targets ~0.9999; 0.999 is the safe regression gate that tolerates the
    documented few-level rounding).

Requires CUDA + the Qwen3-Omni checkpoint (for the real image-processor config);
skips otherwise. Point at a checkpoint with MSTAR_QWEN3_OMNI_DIR.
"""
import os

import numpy as np
import pytest
import torch


def _resolve_checkpoint():
    d = os.environ.get("MSTAR_QWEN3_OMNI_DIR")
    if d and os.path.isdir(d):
        return d
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download("Qwen/Qwen3-Omni-30B-A3B-Instruct",
                                 allow_patterns=["*.json", "*.txt"],
                                 local_files_only=True)
    except Exception:
        return None


CKPT = _resolve_checkpoint()
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(CKPT is None, reason="Qwen3-Omni checkpoint not available"),
]

PIXEL_COS_MIN = 0.999      # primary gate (module targets ~0.9999; see docstring)
PIXEL_MAXABS_MAX = 0.2     # normalized units; ~3 uint8 levels / (255*std) ~= 0.024


def _img_proc(use_fast: bool):
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(CKPT, trust_remote_code=True, use_fast=use_fast)


def _proc_params(ip):
    """Read patch / merge / pixel-bounds robustly (attrs differ across versions)."""
    size = getattr(ip, "size", None) or {}
    min_pixels = getattr(ip, "min_pixels", None) or size.get("shortest_edge")
    max_pixels = getattr(ip, "max_pixels", None) or size.get("longest_edge")
    return dict(
        patch_size=ip.patch_size,
        temporal_patch_size=ip.temporal_patch_size,
        merge_size=ip.merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_mean=ip.image_mean,
        image_std=ip.image_std,
    )


# (H, W): non-multiple-of-patch (factor = patch*merge = 32), small (< min_pixels),
# large (~3000px), plus a couple of awkward aspect ratios.
SIZES = [
    (512, 512),     # clean square
    (500, 333),     # non-multiple of 32, non-square
    (50, 40),       # below min_pixels -> upscale path
    (3000, 2000),   # large
    (777, 1023),    # odd, non-multiple
    (1080, 1920),   # HD aspect
]


@pytest.mark.parametrize("H,W", SIZES, ids=[f"{h}x{w}" for h, w in SIZES])
def test_gpu_image_preprocess_matches_hf(H, W):
    from mstar.model.qwen3_omni.qwen3_omni_model import _gpu_image_preprocess

    # Identical pixel data fed to both paths: uint8 HWC for HF, uint8 CHW on GPU.
    rng = np.random.default_rng(H * 100003 + W)
    img_hwc = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)

    ip = _img_proc(use_fast=True)
    params = _proc_params(ip)

    # HF fast backend reference (numpy HWC in -> pixel_values + image_grid_thw).
    hf = ip(images=[img_hwc], return_tensors="pt")
    ref_pv = hf["pixel_values"].float()
    ref_grid = hf["image_grid_thw"]
    if isinstance(ref_grid, list):
        ref_grid = torch.stack([torch.as_tensor(g) for g in ref_grid])
    ref_grid = ref_grid.cpu().to(torch.long)

    # GPU path under test (CHW uint8 on cuda).
    img_chw = torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous().to("cuda")
    pv, grid = _gpu_image_preprocess(img_chw, **params)
    pv = pv.float().cpu()
    grid = grid.cpu().to(torch.long)

    # grid_thw must be BIT-EXACT (token count / positional layout).
    assert torch.equal(grid, ref_grid), f"grid_thw {grid.tolist()} != HF {ref_grid.tolist()}"
    assert pv.shape == ref_pv.shape, f"pixel_values shape {tuple(pv.shape)} != {tuple(ref_pv.shape)}"

    a, b = pv.flatten(), ref_pv.flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    maxabs = (a - b).abs().max().item()
    assert cos > PIXEL_COS_MIN and maxabs < PIXEL_MAXABS_MAX, (
        f"{H}x{W}: cos={cos:.6f} maxabs={maxabs:.4f} "
        f"(grid={grid.tolist()})")
