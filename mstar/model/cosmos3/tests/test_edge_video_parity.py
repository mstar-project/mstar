"""GPU parity of Cosmos3-Edge generation against the diffusers 0.40
``Cosmos3OmniPipeline`` reference, for t2i / t2v / i2v.

The reference runs out-of-process (``notes/ref_dump_edge_video.py``, diffusers
>= 0.40 env, same GPU) with a fixed seed and dumps the final latents and the
decoded frames. Here the M* fused pipeline regenerates the same request from
the same seed (same RNG order: the reference draws its initial noise from a
freshly seeded generator over the latent shape) and the outputs are compared
as PSNR over decoded pixels (>= 30 dB, the bar the Nano tests use) plus a
per-frame PSNR report.

Needs CUDA, the snapshot (``COSMOS3_EDGE_DIR``) and ``COSMOS3_EDGE_VIDEO_REF``
(a glob or directory of dumps); skipped otherwise.
"""

from __future__ import annotations

import glob
import math
import os
from pathlib import Path

import pytest
import torch

from mstar.model.cosmos3.tests.test_edge import EDGE_DIR

REF = os.environ.get("COSMOS3_EDGE_VIDEO_REF")


def _refs() -> list[str]:
    if not REF:
        return []
    if os.path.isdir(REF):
        return sorted(glob.glob(os.path.join(REF, "edge_*_*x*_f*_s*.pt")))
    return sorted(glob.glob(REF))


REFS = _refs()
needs_gpu_refs = pytest.mark.skipif(
    EDGE_DIR is None or not REFS or not torch.cuda.is_available(),
    reason="needs CUDA, COSMOS3_EDGE_DIR and COSMOS3_EDGE_VIDEO_REF dumps",
)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    return float("inf") if mse == 0 else -10 * math.log10(mse)


@pytest.fixture(scope="module")
def mpipe():
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
    from mstar.model.cosmos3.tests.pipeline import Cosmos3Pipeline

    model = Cosmos3Model(model_path_hf=str(EDGE_DIR), compile_denoise=False, enable_reasoner=False)
    return Cosmos3Pipeline.from_model(model, device="cuda")


@needs_gpu_refs
@pytest.mark.parametrize("ref_path", REFS, ids=[Path(p).stem for p in REFS])
def test_edge_generation_matches_diffusers(ref_path, mpipe) -> None:
    from PIL import Image

    rec = torch.load(ref_path)
    image = Image.open(rec["image_path"]).convert("RGB") if rec.get("image_path") else None
    gen = torch.Generator(device="cuda").manual_seed(int(rec["seed"]))
    init, _ = mpipe._prepare_latents(
        image, int(rec["frames"]) if isinstance(rec["frames"], int) else rec["final_latents"].shape[2] * 4 - 3,
        int(rec["height"]), int(rec["width"]), gen, None, "cuda", torch.bfloat16,
    )
    num_frames = 1 if rec["mode"] == "t2i" else 1 + (rec["final_latents"].shape[2] - 1) * 4
    lat = mpipe(
        prompt=rec["prompt"], negative_prompt=rec["negative_prompt"], image=image, num_frames=num_frames,
        height=int(rec["height"]), width=int(rec["width"]), num_inference_steps=int(rec["steps"]),
        guidance_scale=float(rec["guidance"]), fps=float(rec["fps"]), latents=init, decode=False,
    )
    ref_lat = rec["final_latents"].to(lat.device, lat.dtype).reshape(lat.shape)
    px_m = mpipe._decode(lat).squeeze(0).float().cpu()          # [3, T, H, W] in [0, 1]
    px_r = ((rec["frames"].squeeze(0).float() / 2) + 0.5).clamp(0, 1)
    psnr = _psnr(px_m, px_r)
    per_frame = [round(_psnr(px_m[:, t], px_r[:, t]), 2) for t in range(px_m.shape[1])]
    lat_err = (lat.float() - ref_lat.float()).abs().max().item()
    print(f"  {Path(ref_path).stem}: PSNR={psnr:.2f} dB per-frame={per_frame} latent max-abs-diff={lat_err:.3e}")
    assert psnr >= 30, f"{Path(ref_path).stem}: PSNR {psnr:.2f} dB < 30"
