"""GPU parity of Cosmos3-Edge generation against the diffusers 0.40
``Cosmos3OmniPipeline`` reference, for t2i / t2v / i2v.

The reference runs out-of-process (``notes/ref_dump_edge_video.py``, diffusers
>= 0.40 env, same GPU) with a fixed seed and dumps the final latents and the
decoded frames. Here the M* fused pipeline regenerates the same request from
the same seed (same RNG order: the reference draws its initial noise from a
freshly seeded generator over the latent shape) and the outputs are compared
as PSNR over decoded pixels plus a per-frame PSNR report.

The bar depends on the dump's precision. An fp32 dump is the parity proof:
M* runs the loop in fp32 too and must land within 30 dB (the bar the Nano
tests use; measured 56.8 dB i2v, 39.0 dB t2v). A bf16 dump against the bf16
loop measures kernel drift over the multistep loop, not implementation
fidelity (measured 40.9 dB t2i, 29.7 dB i2v, 25.1 dB t2v at 192x320, 29.8 dB
t2i at 640x640), so those cases get a floor a few dB under the measured
value, enough to catch a real break (a wrong layout lands under 15 dB).

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
# bf16-vs-bf16 drift floors by mode (dB); fp32 dumps use the 30 dB parity bar
BF16_FLOOR = {"t2i": 25.0, "i2v": 24.0, "t2v": 22.0}
needs_gpu_refs = pytest.mark.skipif(
    EDGE_DIR is None or not REFS or not torch.cuda.is_available(),
    reason="needs CUDA, COSMOS3_EDGE_DIR and COSMOS3_EDGE_VIDEO_REF dumps",
)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    return float("inf") if mse == 0 else -10 * math.log10(mse)


_PIPES: dict = {}


def _pipe(dtype: torch.dtype):
    """One pipeline per precision, built on first use (fp32 only when an fp32
    dump asks for it), so the parity cases run the reference's precision."""
    if dtype not in _PIPES:
        from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
        from mstar.model.cosmos3.tests.pipeline import Cosmos3Pipeline

        for other in list(_PIPES):
            del _PIPES[other]
        torch.cuda.empty_cache()
        model = Cosmos3Model(model_path_hf=str(EDGE_DIR), compile_denoise=False, enable_reasoner=False)
        pipe = Cosmos3Pipeline.from_model(model, device="cuda", dtype=dtype)
        if dtype == torch.float32:
            # the checkpoint loads in bf16; the parity loop runs the weights in fp32
            pipe.transformer.float()
            pipe.vae.float()
        _PIPES[dtype] = pipe
    return _PIPES[dtype]


def _ref_dtype(rec: dict) -> torch.dtype:
    return torch.float32 if "float32" in str(rec.get("dtype", "")) else torch.bfloat16


@needs_gpu_refs
@pytest.mark.parametrize("ref_path", REFS, ids=[Path(p).stem for p in REFS])
def test_edge_generation_matches_diffusers(ref_path) -> None:
    from PIL import Image

    rec = torch.load(ref_path, map_location="cpu")
    dtype = _ref_dtype(rec)
    mpipe = _pipe(dtype)
    image = Image.open(rec["image_path"]).convert("RGB") if rec.get("image_path") else None
    gen = torch.Generator(device="cuda").manual_seed(int(rec["seed"]))
    init, _ = mpipe._prepare_latents(
        image, int(rec["frames"]) if isinstance(rec["frames"], int) else rec["final_latents"].shape[2] * 4 - 3,
        int(rec["height"]), int(rec["width"]), gen, None, "cuda", dtype,
    )
    num_frames = 1 if rec["mode"] == "t2i" else 1 + (rec["final_latents"].shape[2] - 1) * 4
    lat = mpipe(
        prompt=rec["prompt"], negative_prompt=rec["negative_prompt"], image=image, num_frames=num_frames,
        height=int(rec["height"]), width=int(rec["width"]), num_inference_steps=int(rec["steps"]),
        guidance_scale=float(rec["guidance"]), fps=float(rec["fps"]), latents=init, decode=False,
        flow_shift=float(rec["flow_shift"]),
    )
    ref_lat = rec["final_latents"].to(lat.device, lat.dtype).reshape(lat.shape)
    # The whole-clip fp32 decode of a 121-frame 480p clip does not fit next
    # to the model; the served decoder runs bf16, so long clips decode there.
    if lat.shape[2] * int(rec["height"]) * int(rec["width"]) > 17 * 192 * 320 * 4:
        mpipe.vae.to(torch.bfloat16)
    px_m = mpipe._decode(lat).squeeze(0).float().cpu()          # [3, T, H, W] in [0, 1]
    mpipe.vae.to(dtype)
    px_r = ((rec["frames"].squeeze(0).float() / 2) + 0.5).clamp(0, 1)
    psnr = _psnr(px_m, px_r)
    per_frame = [round(_psnr(px_m[:, t], px_r[:, t]), 2) for t in range(px_m.shape[1])]
    lat_err = (lat.float() - ref_lat.float()).abs().max().item()
    bar = 30.0 if dtype == torch.float32 else BF16_FLOOR[rec["mode"]]
    kind = "parity" if dtype == torch.float32 else "bf16 drift"
    print(f"  {Path(ref_path).stem} [{kind}, bar {bar} dB]: PSNR={psnr:.2f} dB per-frame={per_frame} "
          f"latent max-abs-diff={lat_err:.3e}")
    assert psnr >= bar, f"{Path(ref_path).stem}: PSNR {psnr:.2f} dB < {bar} ({kind})"
    del lat, px_m, px_r, ref_lat
    torch.cuda.empty_cache()
