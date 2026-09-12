"""Parity for the GPU image preprocess vs HF's ``Qwen2VLImageProcessor``.

``_image_preprocess`` (qwen3_omni_model.py) does resize + rescale +
normalize + patchify entirely on-device, replacing the CPU round-trip
through HF's image processor. The native vision encoder is parity-tested against
HF, so the ``pixel_values`` / ``image_grid_thw`` it consumes must match HF's. This
pins that:

  * ``image_grid_thw`` is BIT-EXACT (it drives token counts / positional layout —
    any drift desyncs the whole multimodal prompt), across resolutions including
    non-multiple-of-patch sizes, a small (<min_pixels) image, and a large (~3000px)
    image.
  * ``pixel_values`` matches HF within a cosine threshold. Same torchvision
    bicubic+antialias kernel in the same order (resize on uint8, then
    rescale+normalize), so the residual is CPU-vs-CUDA rounding.

Gates are set from measurement at 3000x2000 (35.8M values):

    content            max      p99.99    mean       cos(float64)
    uniform noise      0.2039   0.1098    0.0021     0.9999293
    smooth gradient    0.0078   0.0078    0.00033    0.9999950

Both content types run: noise alone lets a regression hide in its wide tolerance,
smooth alone never exercises the rounding tail. Cosine is float64 (in float32 it
exceeds 1.0 at this size and compares nothing), and the magnitude gate is the
99.99th percentile, since max-abs over 36M values is set by a single worst pixel.

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

PIXEL_COS_MIN = 0.9999     # primary gate, float64; worst measured 0.9999293
# Normalized units (post rescale+normalize, so 1.0 ~= one standard deviation).
# Per-content bounds from the table in the module docstring, with ~1.4x headroom.
PIXEL_BOUNDS = {
    # content: (p99.99 gate, max ceiling)
    "noise": (0.15, 0.28),
    "smooth": (0.02, 0.02),
}


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


def _make_image(H, W, content):
    """uint8 HWC test image. ``noise`` is the adversarial case for a resampling
    kernel; ``smooth`` stands in for real photographic content, where the tight
    bound applies."""
    if content == "noise":
        rng = np.random.default_rng(H * 100003 + W)
        return rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    grad = ((np.sin(xx / 97.0) + np.cos(yy / 131.0)) * 0.25 + 0.5) * 255.0
    return grad.astype(np.uint8)[:, :, None].repeat(3, axis=2)


@pytest.mark.parametrize("content", ["noise", "smooth"])
@pytest.mark.parametrize("H,W", SIZES, ids=[f"{h}x{w}" for h, w in SIZES])
def test_image_preprocess_matches_hf(H, W, content):
    from mstar.model.qwen3_omni.qwen3_omni_model import _image_preprocess

    # Identical pixel data fed to both paths: uint8 HWC for HF, uint8 CHW on GPU.
    img_hwc = _make_image(H, W, content)

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
    pv, grid = _image_preprocess(img_chw, **params)
    pv = pv.float().cpu()
    grid = grid.cpu().to(torch.long)

    # grid_thw must be BIT-EXACT (token count / positional layout).
    assert torch.equal(grid, ref_grid), f"grid_thw {grid.tolist()} != HF {ref_grid.tolist()}"
    assert pv.shape == ref_pv.shape, f"pixel_values shape {tuple(pv.shape)} != {tuple(ref_pv.shape)}"

    # float64: over tens of millions of float32 values the cosine accumulation
    # error is large enough to exceed 1.0, which is not comparable to a threshold.
    a, b = pv.flatten().double(), ref_pv.flatten().double()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    diff = (a - b).abs()
    maxabs = diff.max().item()
    # torch.quantile refuses inputs above ~16M elements, and these tensors run to
    # ~36M, so take the 99.99th percentile as the k-th largest value directly.
    k = max(1, int(diff.numel() * 1e-4))
    p9999 = diff.topk(k).values[-1].item()
    p_gate, max_gate = PIXEL_BOUNDS[content]
    stats = (f"{H}x{W}/{content}: cos={cos:.7f} p99.99={p9999:.4f} "
             f"maxabs={maxabs:.4f} (grid={grid.tolist()})")
    assert cos > PIXEL_COS_MIN, stats
    assert p9999 < p_gate, stats
    assert maxabs < max_gate, stats
