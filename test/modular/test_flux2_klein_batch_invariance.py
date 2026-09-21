"""Batch invariance of the FLUX.2 [klein] nodes on real weights (GPU only).

The served path batches concurrent requests at the same shape in every node: the text encoder
stacks prompts into ``[B, 512]``, the denoise loop steps ``[B, L, C]``, the VAE decodes
``[B, C, H, W]``. A request's row inside such a batch must equal the same request alone, or a
row-mixing bug / padding leak would silently change images under load. Rows here carry different
prompts and seeds. The SDPA/eager path is used, so the comparison is between kernels that see
each row independently; the printed numbers (max abs per stage, PSNR per image) are the
measurement, the assertions are the contract (embeddings within bf16 rounding, images >= 40 dB).

Skips without CUDA or without the checkpoint (``FLUX2_KLEIN_REPO``, default 4B) in the HF cache.
Run with ``NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8`` for repeatable kernels.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.flux2_klein.config import DENOISE_LOOP, Flux2KleinConfig, resolve_snapshot_dir
from mstar.model.flux2_klein.flux2_klein_model import IMAGE_GEN_WALK, Flux2KleinModel
from mstar.model.flux2_klein.submodules import LATENTS, TEXT_EMBEDS
from mstar.model.submodule_base import ModelInputsFromEngine

MODEL_REPO = os.environ.get("FLUX2_KLEIN_REPO", "black-forest-labs/FLUX.2-klein-4B")
CUDA_AVAILABLE = torch.cuda.is_available()
PROMPTS = (
    "A cat holding a sign that says hello world, studio lighting, detailed fur",
    "An astronaut riding a horse on Mars, cinematic wide shot",
    "A bowl of ramen on a wooden table, soft window light, steam",
    "A watercolor painting of a lighthouse in a storm",
)
SEEDS = (0, 1, 2, 3)
HEIGHT = WIDTH = 1024
STEPS = 4
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
]

DEVICE = torch.device("cuda")


@pytest.fixture(scope="module")
def model() -> Flux2KleinModel:
    m = Flux2KleinModel(model_path_hf=MODEL_REPO, attention_backend="sdpa", compile=False, cuda_graph=False)
    m.set_config(Flux2KleinConfig.from_snapshot(resolve_snapshot_dir(MODEL_REPO)))
    return m


@pytest.fixture(scope="module")
def subs(model) -> dict:
    return {name: model.get_submodule(name, device=DEVICE) for name in ("text_encoder", "dit", "vae_decoder")}


def _eng(rids) -> ModelInputsFromEngine:
    return ModelInputsFromEngine(request_ids=list(rids), per_request_info={})


def _info(rid: str, k: int, seed: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=IMAGE_GEN_WALK, fwd_index=k, random_seed=seed, max_tokens=0,
        step_metadata={"height": HEIGHT, "width": WIDTH, "num_inference_steps": STEPS, "ref_grids": []},
        dynamic_loop_iter_counts={DENOISE_LOOP: k},
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    return math.inf if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)


def _encode(model, subs, batched: bool) -> list[torch.Tensor]:
    """Per-prompt ``[1, 512, joint_dim]`` embeddings, from one stacked call or one call per prompt."""
    text = subs["text_encoder"]
    ids, masks = zip(*(model.tokenize(p) for p in PROMPTS), strict=True)
    with torch.inference_mode():
        if batched:
            out = text.forward(IMAGE_GEN_WALK, _eng(f"b{i}" for i in range(len(PROMPTS))),
                               text_inputs=torch.stack(ids), text_mask=torch.stack(masks))[TEXT_EMBEDS][0]
            return [out[i:i + 1] for i in range(len(PROMPTS))]
        return [text.forward(IMAGE_GEN_WALK, _eng([f"s{i}"]), text_inputs=i_ids[None], text_mask=i_mask[None])
                [TEXT_EMBEDS][0] for i, (i_ids, i_mask) in enumerate(zip(ids, masks, strict=True))]


def _trajectory(subs, embeds: list[torch.Tensor], prefix: str, batched: bool) -> list[list[torch.Tensor]]:
    """Latents after every step for each row: ``[step][row]``; one stacked forward per step or one per row."""
    dit = subs["dit"]
    rids = [f"{prefix}{i}" for i in range(len(embeds))]
    latents: list[torch.Tensor | None] = [None] * len(embeds)
    steps: list[list[torch.Tensor]] = []
    with torch.inference_mode():
        for k in range(STEPS):
            rows = []
            for i, rid in enumerate(rids):
                inputs = {TEXT_EMBEDS: [embeds[i]]}
                if latents[i] is not None:
                    inputs[LATENTS] = [latents[i]]
                rows.append(dit.prepare_inputs(IMAGE_GEN_WALK, _info(rid, k, SEEDS[i]), inputs))
            if batched:
                kwargs = dit.preprocess(IMAGE_GEN_WALK, _eng(rids), rows)
                out = dit.forward_batched(IMAGE_GEN_WALK, _eng(rids), **kwargs)
                latents = [out[rid][LATENTS][0] for rid in rids]
            else:
                latents = []
                for rid, row in zip(rids, rows, strict=True):
                    kwargs = dit.preprocess(IMAGE_GEN_WALK, _eng([rid]), [row])
                    latents.append(dit.forward(IMAGE_GEN_WALK, _eng([rid]), **kwargs)[LATENTS][0])
            steps.append(list(latents))
    for rid in rids:
        dit.cleanup_request(rid)
    return steps


def _decode(model, subs, latents: list[torch.Tensor], batched: bool) -> list[torch.Tensor]:
    """uint8 ``[3, H, W]`` per row, from one stacked decode or one decode per row."""
    decoder = subs["vae_decoder"]
    grid = model.config.latent_grid(HEIGHT, WIDTH)
    rids = [f"d{i}" for i in range(len(latents))]

    def rows(out) -> list[torch.Tensor]:
        images = out["image_output"]
        stacked = images[0] if len(images) == 1 and images[0].ndim == 4 else torch.stack(list(images))
        return [img for img in stacked]

    with torch.inference_mode():
        if batched:
            return rows(decoder.forward(IMAGE_GEN_WALK, _eng(rids), latents=torch.stack(latents), grid=grid))
        return [rows(decoder.forward(IMAGE_GEN_WALK, _eng([rid]), latents=lat[None], grid=grid))[0]
                for rid, lat in zip(rids, latents, strict=True)]


@pytest.fixture(scope="module")
def single(model, subs) -> dict:
    embeds = _encode(model, subs, batched=False)
    steps = _trajectory(subs, embeds, "s", batched=False)
    return {"embeds": embeds, "steps": steps, "images": _decode(model, subs, steps[-1], batched=False)}


def test_text_encoder_rows_match_single_requests(model, subs, single):
    batched = _encode(model, subs, batched=True)
    worst = max(_max_abs(b, s) for b, s in zip(batched, single["embeds"], strict=True))
    print(f"text embeds, stacked [{len(PROMPTS)}, 512] vs alone: max_abs={worst:.3e}")
    for b, s in zip(batched, single["embeds"], strict=True):
        torch.testing.assert_close(b, s, rtol=1e-2, atol=1e-2)


def test_denoise_rows_match_single_requests(model, subs, single):
    steps = _trajectory(subs, single["embeds"], "b", batched=True)
    for k, (batched_rows, single_rows) in enumerate(zip(steps, single["steps"], strict=True)):
        worst = max(_max_abs(b, s) for b, s in zip(batched_rows, single_rows, strict=True))
        print(f"dit step {k}, batch of {len(PROMPTS)} vs alone: max_abs={worst:.3e}")
    images = _decode(model, subs, steps[-1], batched=False)
    for i, (img, ref) in enumerate(zip(images, single["images"], strict=True)):
        psnr = _psnr(img, ref)
        print(f"row {i} (seed {SEEDS[i]}): image PSNR batched-trajectory vs alone = {psnr:.2f} dB")
        assert psnr >= MIN_PSNR_DB, f"row {i}: {psnr:.2f} dB < {MIN_PSNR_DB}"


def test_vae_decode_rows_match_single_requests(model, subs, single):
    images = _decode(model, subs, single["steps"][-1], batched=True)
    for i, (img, ref) in enumerate(zip(images, single["images"], strict=True)):
        psnr = _psnr(img, ref)
        print(f"row {i}: VAE decode in a batch of {len(PROMPTS)} vs alone = {psnr:.2f} dB, "
              f"max_abs={_max_abs(img, ref):.0f}")
        assert psnr >= MIN_PSNR_DB, f"row {i}: {psnr:.2f} dB < {MIN_PSNR_DB}"
