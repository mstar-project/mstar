"""Batch invariance of the Z-Image nodes on real weights (GPU only).

Mirror of ``test_flux2_klein_batch_invariance.py`` for the second model on the scaffold: the text
encoder pads every caption to 512 and batches captions of any length; the denoise loop batches
requests with the same (grid, padded caption length) shape key over three attention spans, so a
row-mixing bug or a leaking pad row shows up here; the VAE decodes ``[B, C, H, W]``. Rows carry
different prompts and seeds; the SDPA/eager path is used.

Two contracts. (1) Row independence: the same rows in another order, at the same batch size, must
reproduce every row bit for bit (same kernels: any difference is a leak — diagnosed 2026-09-21,
`notes/diag_zbatch.py`: permuted rows 0.0, replicated rows identical). (2) Batch size: B=2 equals
B=1 bit for bit, B=4 does not (cuBLAS / cuDNN pick other algorithms for that M: step-0 max_abs
4.8e-2, images ~39 dB after 8 steps, content-independent), so a row in a batch of four vs alone
is reported per stage and per image and asserted only as a gross-error bound (>= 30 dB).

Skips without CUDA or without ``Tongyi-MAI/Z-Image-Turbo`` in the HF cache. Run with
``NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8`` for repeatable kernels.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.z_image.config import DENOISE_LOOP, ZImageConfig, resolve_snapshot_dir
from mstar.model.z_image.submodules import CAP_PAD_MASK, LATENTS, TEXT_EMBEDS, TEXT_INPUTS, padded_length
from mstar.model.z_image.z_image_model import IMAGE_GEN_WALK, ZImageModel

MODEL_REPO = "Tongyi-MAI/Z-Image-Turbo"
CANDIDATES = (
    "A cat holding a sign that says hello world, studio lighting, detailed fur",
    "An astronaut riding a horse on Mars, cinematic wide shot, dust in the air",
    "A bowl of ramen on a wooden table, soft window light, steam rising",
    "A watercolor painting of a lighthouse in a storm, dramatic waves",
    "A red bicycle leaning against a yellow wall in Lisbon, afternoon sun",
    "Macro photograph of a dew-covered spider web at dawn",
    "A cozy cabin in a snowy forest at night, warm light in the windows",
    "An oil painting of a market street in Marrakech, crowded and colorful",
)
SEEDS = (0, 1, 2, 3)
HEIGHT = WIDTH = 1024
STEPS = 8
MIN_PSNR_DB = 40.0        # exactness contracts (row independence, VAE)
MIN_BATCH_PSNR_DB = 30.0  # gross-error bound for the batch-size comparison (see the docstring)


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
]
DEVICE = torch.device("cuda")


@pytest.fixture(scope="module")
def model() -> ZImageModel:
    m = ZImageModel(model_path_hf=MODEL_REPO, attention_backend="sdpa", compile=False, cuda_graph=False)
    m.set_config(ZImageConfig.from_snapshot(resolve_snapshot_dir(MODEL_REPO)))
    return m


@pytest.fixture(scope="module")
def subs(model) -> dict:
    return {name: model.get_submodule(name, device=DEVICE) for name in ("text_encoder", "dit", "vae_decoder")}


@pytest.fixture(scope="module")
def prompts(model) -> list[tuple[str, torch.Tensor]]:
    """Four captions with the same padded length (the served batching condition), as (prompt, ids)."""
    by_len: dict[int, list[tuple[str, torch.Tensor]]] = {}
    for prompt in CANDIDATES:
        ids = model.tokenize(prompt)
        by_len.setdefault(padded_length(int(ids.shape[0])), []).append((prompt, ids))
    group = max(by_len.values(), key=len)
    if len(group) < 4:
        pytest.skip(f"no 4 candidate prompts share a padded caption length: {sorted(by_len)}")
    return group[:4]


def _eng(rids) -> ModelInputsFromEngine:
    return ModelInputsFromEngine(request_ids=list(rids), per_request_info={})


def _info(rid: str, k: int, seed: int, text_len: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=IMAGE_GEN_WALK, fwd_index=k, random_seed=seed, max_tokens=0,
        step_metadata={"height": HEIGHT, "width": WIDTH, "num_inference_steps": STEPS,
                       "text_len": text_len, "cap_len": padded_length(text_len)},
        dynamic_loop_iter_counts={DENOISE_LOOP: k},
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.float() - b.float()).pow(2).mean().item()
    return math.inf if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)


def _encode(subs, prompts, batched: bool) -> list[torch.Tensor]:
    """Per-caption ``[1, cap_len, 2560]`` features, from one stacked call or one call per caption."""
    text = subs["text_encoder"]
    rows = [text.prepare_inputs(IMAGE_GEN_WALK, _info(f"t{i}", 0, 0, int(ids.shape[0])), {TEXT_INPUTS: [ids]})
            for i, (_, ids) in enumerate(prompts)]
    rids = [f"t{i}" for i in range(len(prompts))]
    with torch.inference_mode():
        if batched:
            out = text.forward_batched(IMAGE_GEN_WALK, _eng(rids), **text.preprocess(IMAGE_GEN_WALK, _eng(rids), rows))
            return [out[rid][TEXT_EMBEDS][0] for rid in rids]
        return [text.forward(IMAGE_GEN_WALK, _eng([rid]), **text.preprocess(IMAGE_GEN_WALK, _eng([rid]), [row]))
                [TEXT_EMBEDS][0] for rid, row in zip(rids, rows, strict=True)]


def _trajectory(subs, prompts, embeds, prefix: str, batched: bool, seeds=SEEDS) -> list[list[torch.Tensor]]:
    dit = subs["dit"]
    rids = [f"{prefix}{i}" for i in range(len(embeds))]
    text_lens = [int(ids.shape[0]) for _, ids in prompts]
    latents: list[torch.Tensor | None] = [None] * len(embeds)
    steps: list[list[torch.Tensor]] = []
    with torch.inference_mode():
        for k in range(STEPS):
            rows = []
            for i, rid in enumerate(rids):
                inputs = {TEXT_EMBEDS: [embeds[i]]}
                if latents[i] is not None:
                    inputs[LATENTS] = [latents[i]]
                rows.append(dit.prepare_inputs(IMAGE_GEN_WALK, _info(rid, k, seeds[i], text_lens[i]), inputs))
            assert CAP_PAD_MASK in rows[0].tensor_inputs
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


def _decode(subs, latents: list[torch.Tensor], batched: bool) -> list[torch.Tensor]:
    decoder = subs["vae_decoder"]
    rids = [f"d{i}" for i in range(len(latents))]
    with torch.inference_mode():
        if batched:
            out = decoder.forward_batched(IMAGE_GEN_WALK, _eng(rids), latents=torch.stack(latents))
            return [out[rid]["image_output"][0][0] for rid in rids]
        return [decoder.forward(IMAGE_GEN_WALK, _eng([rid]), latents=lat[None])["image_output"][0][0]
                for rid, lat in zip(rids, latents, strict=True)]


@pytest.fixture(scope="module")
def single(subs, prompts) -> dict:
    embeds = _encode(subs, prompts, batched=False)
    steps = _trajectory(subs, prompts, embeds, "s", batched=False)
    return {"embeds": embeds, "steps": steps, "images": _decode(subs, steps[-1], batched=False)}


def test_text_encoder_rows_match_single_requests(subs, prompts, single):
    batched = _encode(subs, prompts, batched=True)
    worst = max(_max_abs(b, s) for b, s in zip(batched, single["embeds"], strict=True))
    print(f"caption features, stacked [{len(prompts)}, 512] vs alone: max_abs={worst:.3e}")
    for b, s in zip(batched, single["embeds"], strict=True):
        torch.testing.assert_close(b, s, rtol=1e-2, atol=1e-2)


def test_denoise_rows_are_independent_of_their_neighbours(subs, prompts, single):
    """Same batch size, rows in another order: every row must come back bit for bit."""
    perm = [3, 1, 0, 2]
    embeds = single["embeds"]
    straight = _trajectory(subs, prompts, embeds, "o", batched=True)
    shuffled = _trajectory(subs, [prompts[j] for j in perm], [embeds[j] for j in perm], "p", batched=True,
                           seeds=[SEEDS[j] for j in perm])
    for k, (a, b) in enumerate(zip(straight, shuffled, strict=True)):
        worst = max(_max_abs(a[j], b[perm.index(j)]) for j in range(len(perm)))
        print(f"dit step {k}, permuted batch vs batch: max_abs={worst:.3e}")
        for j in range(len(perm)):
            assert torch.equal(a[j], b[perm.index(j)]), f"step {k}: row {j} depends on its neighbours"


def test_denoise_rows_match_single_requests(subs, prompts, single):
    steps = _trajectory(subs, prompts, single["embeds"], "b", batched=True)
    for k, (batched_rows, single_rows) in enumerate(zip(steps, single["steps"], strict=True)):
        per_row = [f"{_max_abs(b, s):.1e}" for b, s in zip(batched_rows, single_rows, strict=True)]
        print(f"dit step {k}, batch of {len(prompts)} vs alone, max_abs per row: {per_row}")
    images = _decode(subs, steps[-1], batched=False)
    psnrs = [_psnr(img, ref) for img, ref in zip(images, single["images"], strict=True)]
    for i, psnr in enumerate(psnrs):
        print(f"row {i} (seed {SEEDS[i]}): image PSNR batched-trajectory vs alone = {psnr:.2f} dB")
    assert min(psnrs) >= MIN_BATCH_PSNR_DB, f"batch-size numerics beyond a kernel-selection effect: {psnrs}"


def test_vae_decode_rows_match_single_requests(subs, prompts, single):
    images = _decode(subs, single["steps"][-1], batched=True)
    for i, (img, ref) in enumerate(zip(images, single["images"], strict=True)):
        psnr = _psnr(img, ref)
        print(f"row {i}: VAE decode in a batch of {len(prompts)} vs alone = {psnr:.2f} dB, "
              f"max_abs={_max_abs(img, ref):.0f}")
        assert psnr >= MIN_PSNR_DB, f"row {i}: {psnr:.2f} dB < {MIN_PSNR_DB}"
