"""Reference-equivalence tests: mstar's Z-Image-Turbo vs the diffusers oracle on real weights
(``test/z_image/record_oracle.py``). Skips without CUDA, the checkpoint, or
``Z_IMAGE_ORACLE_DIR``. Gates: caption embeddings and per-step fp32 latents bit-exact on the
SDPA backend, final image PSNR >= 40 dB. Never loosen a bound to pass."""

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
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402
from mstar.model.z_image.config import DENOISE_LOOP, ZImageConfig, resolve_snapshot_dir  # noqa: E402
from mstar.model.z_image.submodules import LATENTS, TEXT_EMBEDS, padded_length  # noqa: E402
from mstar.model.z_image.z_image_model import IMAGE_GEN_WALK, ZImageModel  # noqa: E402

MODEL_REPO = "Tongyi-MAI/Z-Image-Turbo"
_ORACLE_ENV = os.environ.get("Z_IMAGE_ORACLE_DIR", "")
ORACLE_DIR = Path(_ORACLE_ENV) if _ORACLE_ENV else None
TEXT_MAX_ABS = 0.0
NOISE_MAX_ABS = 0.0
STEP_MAX_ABS = 0.0
MIN_PSNR_DB = 40.0


def _cached() -> bool:
    dirname = f"models--{MODEL_REPO.replace('/', '--')}"
    for env in ("HF_HUB_CACHE", "HF_HOME"):
        root = os.environ.get(env)
        if root and ((Path(root) if env == "HF_HUB_CACHE" else Path(root) / "hub") / dirname).exists():
            return True
    return False


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not _cached(), reason=f"{MODEL_REPO} not in the local HF cache"),
    pytest.mark.skipif(ORACLE_DIR is None, reason="set Z_IMAGE_ORACLE_DIR (see test/z_image/record_oracle.py)"),
]
DEVICE = torch.device("cuda")


@pytest.fixture(scope="module")
def meta():
    with open(ORACLE_DIR / "metadata.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model():
    m = ZImageModel(model_path_hf=MODEL_REPO, attention_backend="sdpa", compile=False, cuda_graph=False)
    m.set_config(ZImageConfig.from_snapshot(resolve_snapshot_dir(MODEL_REPO)))
    return m


def _info(meta, k, text_len):
    return CurrentForwardPassInfo(
        request_id="oracle", graph_walk=IMAGE_GEN_WALK, fwd_index=k, random_seed=meta["seed"], max_tokens=0,
        step_metadata={"height": meta["height"], "width": meta["width"], "num_inference_steps": meta["steps"],
                       "text_len": text_len, "cap_len": padded_length(text_len)},
        dynamic_loop_iter_counts={DENOISE_LOOP: k},
    )


def _psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)


def test_trajectory_and_image_match_oracle(model, meta):
    engine_inputs = ModelInputsFromEngine(request_ids=["oracle"], per_request_info={})
    text = model.get_submodule("text_encoder", device=DEVICE)
    dit = model.get_submodule("dit", device=DEVICE)
    decoder = model.get_submodule("vae_decoder", device=DEVICE)
    ids = model.tokenize(meta["prompt"])
    text_len = int(ids.shape[0])
    with torch.no_grad():
        node_inputs = text.prepare_inputs(IMAGE_GEN_WALK, None, {"text_inputs": [ids]})
        kwargs = text.preprocess(IMAGE_GEN_WALK, engine_inputs, [node_inputs])
        embeds = text.forward(IMAGE_GEN_WALK, engine_inputs, **kwargs)[TEXT_EMBEDS][0]
        expected_embeds = torch.load(ORACLE_DIR / "t2i" / "prompt_embeds.pt")
        diff = (embeds[0, :text_len].cpu().float() - expected_embeds.float()).abs().max().item()
        print(f"caption embeds max_abs={diff:.3e}")
        assert diff <= TEXT_MAX_ABS
        latents = None
        worst = 0.0
        for k in range(meta["steps"]):
            inputs = {TEXT_EMBEDS: [embeds], **({LATENTS: [latents]} if latents is not None else {})}
            node_inputs = dit.prepare_inputs(IMAGE_GEN_WALK, _info(meta, k, text_len), inputs)
            if k == 0:
                init = torch.load(ORACLE_DIR / "t2i" / "latents_init.pt")[0]
                assert (node_inputs.tensor_inputs[LATENTS].cpu() - init).abs().max().item() <= NOISE_MAX_ABS
            step_kwargs = dit.preprocess(IMAGE_GEN_WALK, engine_inputs, [node_inputs])
            latents = dit.forward(IMAGE_GEN_WALK, engine_inputs, **step_kwargs)[LATENTS][0]
            expected = torch.load(ORACLE_DIR / "t2i" / f"latents_step_{k:03d}.pt")[0]
            diff = (latents.cpu() - expected).abs().max().item()
            worst = max(worst, diff)
            print(f"step {k}: latents max_abs={diff:.3e}")
        assert worst <= STEP_MAX_ABS
        image = decoder.forward(IMAGE_GEN_WALK, engine_inputs, latents=latents[None])["image_output"][0][0].cpu()
    import numpy as np
    from PIL import Image

    ref = torch.from_numpy(np.array(Image.open(ORACLE_DIR / "t2i" / "image.png").convert("RGB"))).permute(2, 0, 1)
    psnr = _psnr(image, ref)
    print(f"final image PSNR={psnr:.2f} dB")
    assert psnr >= MIN_PSNR_DB
    dit.cleanup_request("oracle")
