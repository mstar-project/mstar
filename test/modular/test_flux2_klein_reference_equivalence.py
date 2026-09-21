"""Reference-equivalence tests: mstar's FLUX.2 [klein] vs the diffusers oracle, on real weights.

The reference is a recorded run of the stock diffusers ``Flux2KleinPipeline`` on
``black-forest-labs/FLUX.2-klein-4B`` (``test/flux2_klein/record_oracle.py``): prompt
embeddings, the seeded initial latents, the packed latents after every Euler step, and
the final PNG, for a text-to-image request and a single-reference edit.

Gates (never loosen one to make a run pass — a regression is a finding):

* text encoder taps: bit-exact when the same attention kernel is used; a bf16-noise
  bound otherwise (``TEXT_MAX_ABS``)
* initial noise: bit-exact (same CPU generator, same dtype, same packing)
* per-step latents: ``STEP_MAX_ABS`` — SDPA-backend runs are expected at 0.0; the
  FlashInfer backend prices its own kernel's rounding
* final image: PSNR >= ``MIN_PSNR_DB`` against the oracle PNG

The suite skips without CUDA, without the checkpoint in the HF cache, or without
``FLUX2_KLEIN_ORACLE_DIR``. Record the oracle and run the suite in the same environment
with ``NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.model.flux2_klein.config import DENOISE_LOOP, Flux2KleinConfig, resolve_snapshot_dir  # noqa: E402
from mstar.model.flux2_klein.flux2_klein_model import IMAGE_EDIT_WALK, IMAGE_GEN_WALK, Flux2KleinModel  # noqa: E402
from mstar.model.flux2_klein.submodules import (  # noqa: E402
    LATENTS,
    REF_LATENTS,
    TEXT_EMBEDS,
    KleinDenoiseSubmodule,
    KleinTextEncoderSubmodule,
    KleinVaeDecoderSubmodule,
    KleinVaeEncoderSubmodule,
)
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402

MODEL_REPO = os.environ.get("FLUX2_KLEIN_REPO", "black-forest-labs/FLUX.2-klein-4B")
_ORACLE_ENV = os.environ.get("FLUX2_KLEIN_ORACLE_DIR", "")
ORACLE_DIR = Path(_ORACLE_ENV) if _ORACLE_ENV else None
CUDA_AVAILABLE = torch.cuda.is_available()

TEXT_MAX_ABS = 0.0
NOISE_MAX_ABS = 0.0
STEP_MAX_ABS = 0.0
MIN_PSNR_DB = 40.0


def _hf_cache_has_checkpoint() -> bool:
    dirname = f"models--{MODEL_REPO.replace('/', '--')}"
    for env in ("HF_HUB_CACHE", "HF_HOME"):
        root = os.environ.get(env)
        if not root:
            continue
        base = Path(root) if env == "HF_HUB_CACHE" else Path(root) / "hub"
        if (base / dirname).exists():
            return True
    return (Path.home() / ".cache" / "huggingface" / "hub" / dirname).exists()


pytestmark = [
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA"),
    pytest.mark.skipif(not _hf_cache_has_checkpoint(), reason=f"{MODEL_REPO} not in the local HF cache"),
    pytest.mark.skipif(ORACLE_DIR is None, reason="set FLUX2_KLEIN_ORACLE_DIR (see test/flux2_klein/record_oracle.py)"),
]

DEVICE = torch.device("cuda")


@pytest.fixture(scope="module")
def meta() -> dict:
    with open(ORACLE_DIR / "metadata.json") as f:
        meta = json.load(f)
    # The oracle must have been recorded from the checkpoint under test (FLUX2_KLEIN_REPO
    # selects 4B or 9B); comparing against the other model's trajectory is a setup error.
    assert meta["repo"] == MODEL_REPO, (
        f"oracle {ORACLE_DIR} was recorded from {meta['repo']!r} but FLUX2_KLEIN_REPO={MODEL_REPO!r}")
    return meta


@pytest.fixture(scope="module")
def model() -> Flux2KleinModel:
    # SDPA backend: the reference kernel, so the per-step gates can be bit-exact.
    m = Flux2KleinModel(model_path_hf=MODEL_REPO, attention_backend="sdpa", compile=False, cuda_graph=False)
    m.set_config(Flux2KleinConfig.from_snapshot(resolve_snapshot_dir(MODEL_REPO)))
    return m


def _fwd_info(meta: dict, k: int, seed: int, ref_grids=(), walk: str = IMAGE_GEN_WALK) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="oracle", graph_walk=walk, fwd_index=k, random_seed=seed, max_tokens=0,
        step_metadata={"height": meta["height"], "width": meta["width"], "num_inference_steps": meta["steps"],
                       "ref_grids": [list(g) for g in ref_grids]},
        dynamic_loop_iter_counts={DENOISE_LOOP: k},
    )


def _engine_inputs() -> ModelInputsFromEngine:
    return ModelInputsFromEngine(request_ids=["oracle"], per_request_info={})


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)


def _load_png(path: Path) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)


def test_text_encoder_matches_oracle(model, meta):
    sub: KleinTextEncoderSubmodule = model.get_submodule("text_encoder", device=DEVICE)
    ids, mask = model.tokenize(meta["prompt"])
    with torch.no_grad():
        out = sub.forward(IMAGE_GEN_WALK, _engine_inputs(), text_inputs=ids[None], text_mask=mask[None])
    embeds = out[TEXT_EMBEDS][0].cpu()
    expected = torch.load(ORACLE_DIR / "t2i" / "prompt_embeds.pt")
    assert embeds.shape == expected.shape and embeds.dtype == expected.dtype
    diff = (embeds.float() - expected.float()).abs().max().item()
    print(f"text embeds max_abs={diff:.3e}")
    assert diff <= TEXT_MAX_ABS


def test_initial_noise_matches_oracle(model, meta):
    dit: KleinDenoiseSubmodule = model.get_submodule("dit", device=DEVICE)
    node_inputs = dit.prepare_inputs(IMAGE_GEN_WALK, _fwd_info(meta, 0, meta["seed"]),
                                     {TEXT_EMBEDS: [torch.zeros(1, 512, model.config.transformer.joint_attention_dim)]})
    expected = torch.load(ORACLE_DIR / "t2i" / "latents_init.pt")[0]
    diff = (node_inputs.tensor_inputs[LATENTS].cpu().float() - expected.float()).abs().max().item()
    print(f"initial noise max_abs={diff:.3e}")
    assert diff <= NOISE_MAX_ABS
    dit.cleanup_request("oracle")


def test_denoise_trajectory_and_image_match_oracle(model, meta):
    text: KleinTextEncoderSubmodule = model.get_submodule("text_encoder", device=DEVICE)
    dit: KleinDenoiseSubmodule = model.get_submodule("dit", device=DEVICE)
    decoder: KleinVaeDecoderSubmodule = model.get_submodule("vae_decoder", device=DEVICE)
    ids, mask = model.tokenize(meta["prompt"])
    with torch.no_grad():
        out = text.forward(IMAGE_GEN_WALK, _engine_inputs(), text_inputs=ids[None], text_mask=mask[None])
        embeds = out[TEXT_EMBEDS][0]
        inputs = {TEXT_EMBEDS: [embeds]}
        latents = None
        worst = 0.0
        for k in range(meta["steps"]):
            node_inputs = dit.prepare_inputs(IMAGE_GEN_WALK, _fwd_info(meta, k, meta["seed"]),
                                             {**inputs, **({LATENTS: [latents]} if latents is not None else {})})
            kwargs = dit.preprocess(IMAGE_GEN_WALK, _engine_inputs(), [node_inputs])
            latents = dit.forward(IMAGE_GEN_WALK, _engine_inputs(), **kwargs)[LATENTS][0]
            expected = torch.load(ORACLE_DIR / "t2i" / f"latents_step_{k:03d}.pt")[0]
            diff = (latents.cpu().float() - expected.float()).abs().max().item()
            worst = max(worst, diff)
            print(f"step {k}: latents max_abs={diff:.3e}")
        assert worst <= STEP_MAX_ABS, f"per-step latents diverge from the oracle (max {worst:.3e})"
        image = decoder.forward(
            IMAGE_GEN_WALK, _engine_inputs(), latents=latents[None],
            grid=model.config.latent_grid(meta["height"], meta["width"]),
        )["image_output"][0][0].cpu()
    expected_image = _load_png(ORACLE_DIR / "t2i" / "image.png")
    psnr = _psnr(image, expected_image)
    print(f"final image PSNR={psnr:.2f} dB")
    assert psnr >= MIN_PSNR_DB
    dit.cleanup_request("oracle")


def test_edit_trajectory_and_image_match_oracle(model, meta):
    """Single-reference edit: the t2i image is the reference, the edit prompt and seed come from the
    oracle; reference latents, text embeddings, every Euler step and the final image must match."""
    if "edit_prompt" not in meta:
        pytest.skip("oracle recorded with --skip-edit")
    text: KleinTextEncoderSubmodule = model.get_submodule("text_encoder", device=DEVICE)
    encoder: KleinVaeEncoderSubmodule = model.get_submodule("vae_encoder", device=DEVICE)
    dit: KleinDenoiseSubmodule = model.get_submodule("dit", device=DEVICE)
    decoder: KleinVaeDecoderSubmodule = model.get_submodule("vae_decoder", device=DEVICE)
    reference = model.load_image(str(ORACLE_DIR / meta["edit_reference"]), "cpu").data  # [3, H, W] in [0, 1]
    ref_grids = [model.config.latent_grid(reference.shape[1], reference.shape[2])]
    ids, mask = model.tokenize(meta["edit_prompt"])
    with torch.no_grad():
        embeds = text.forward(
            IMAGE_EDIT_WALK, _engine_inputs(), text_inputs=ids[None], text_mask=mask[None],
        )[TEXT_EMBEDS][0]
        expected_embeds = torch.load(ORACLE_DIR / "edit" / "prompt_embeds.pt")
        diff = (embeds.cpu().float() - expected_embeds.float()).abs().max().item()
        print(f"edit text embeds max_abs={diff:.3e}")
        assert diff <= TEXT_MAX_ABS
        ref_latents = encoder.forward(IMAGE_EDIT_WALK, _engine_inputs(), image_0=reference)[REF_LATENTS][0]
        assert ref_latents.shape == (1, ref_grids[0][0] * ref_grids[0][1], model.config.transformer.in_channels)
        inputs = {TEXT_EMBEDS: [embeds], REF_LATENTS: [ref_latents]}
        latents, worst = None, 0.0
        for k in range(meta["steps"]):
            info = _fwd_info(meta, k, meta["edit_seed"], ref_grids=ref_grids, walk=IMAGE_EDIT_WALK)
            node_inputs = dit.prepare_inputs(IMAGE_EDIT_WALK, info,
                                             {**inputs, **({LATENTS: [latents]} if latents is not None else {})})
            kwargs = dit.preprocess(IMAGE_EDIT_WALK, _engine_inputs(), [node_inputs])
            latents = dit.forward(IMAGE_EDIT_WALK, _engine_inputs(), **kwargs)[LATENTS][0]
            expected = torch.load(ORACLE_DIR / "edit" / f"latents_step_{k:03d}.pt")[0]
            diff = (latents.cpu().float() - expected.float()).abs().max().item()
            worst = max(worst, diff)
            print(f"edit step {k}: latents max_abs={diff:.3e}")
        assert worst <= STEP_MAX_ABS, f"edit per-step latents diverge from the oracle (max {worst:.3e})"
        image = decoder.forward(
            IMAGE_EDIT_WALK, _engine_inputs(), latents=latents[None],
            grid=model.config.latent_grid(meta["height"], meta["width"]),
        )["image_output"][0][0].cpu()
    dit.cleanup_request("oracle")
    psnr = _psnr(image, _load_png(ORACLE_DIR / "edit" / "image.png"))
    print(f"edit image PSNR={psnr:.2f} dB")
    assert psnr >= MIN_PSNR_DB
