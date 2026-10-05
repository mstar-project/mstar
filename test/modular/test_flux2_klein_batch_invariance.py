"""Batch invariance of the FLUX.2 [klein] nodes on real weights (GPU only).

The served path batches concurrent requests at the same shape in every node: the text encoder
stacks prompts into ``[B, 512]``, the denoise loop steps ``[B, L, C]``, the VAE decodes
``[B, C, H, W]``. A request's row inside such a batch must equal the same request alone, or a
row-mixing bug / padding leak would silently change images under load. Rows here carry different
prompts and seeds; the SDPA/eager path is used.

Two contracts. (1) Row independence: the same rows in another order, at the same batch size,
must reproduce every row bit for bit — the same kernels run, so any difference is a cross-row
leak (padding rows, spans, stacked scalars). (2) Batch size: a row inside a batch of four vs the
row alone is reported per stage and per image; cuBLAS / cuDNN may pick other algorithms for
another M (klein-4B measured bit-exact; Z-Image's dit differs from step 0 and lands near 39 dB
after 8 steps with no leak: rows measured 39.3 / 30.7 / 37.4 / 33.1 dB), so this one is asserted
only as a gross-error bound (>= 25 dB) and the printed numbers are the measurement.

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
from mstar.model.flux2_klein.config import DENOISE_LOOP, Flux2KleinConfig
from mstar.model.flux2_klein.flux2_klein_model import IMAGE_EDIT_WALK, IMAGE_GEN_WALK, Flux2KleinModel
from mstar.model.flux2_klein.submodules import IMAGE_INPUTS, LATENTS, REF_LATENTS, TEXT_EMBEDS
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.utils.hf_snapshot import resolve_snapshot_dir

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
MIN_PSNR_DB = 40.0        # exactness contracts (row independence, VAE)
MIN_BATCH_PSNR_DB = 25.0  # gross-error bound for the batch-size comparison (see the docstring)


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
    return {name: model.get_submodule(name, device=DEVICE)
            for name in ("text_encoder", "dit", "vae_encoder", "vae_decoder")}


def _eng(rids) -> ModelInputsFromEngine:
    return ModelInputsFromEngine(request_ids=list(rids), per_request_info={})


def _info(rid: str, k: int, seed: int, walk: str = IMAGE_GEN_WALK, ref_grids=()) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=walk, fwd_index=k, random_seed=seed, max_tokens=0,
        step_metadata={"height": HEIGHT, "width": WIDTH, "num_inference_steps": STEPS,
                       "ref_grids": [list(g) for g in ref_grids]},
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


def _trajectory(subs, embeds: list[torch.Tensor], prefix: str, batched: bool, seeds=SEEDS,
                walk: str = IMAGE_GEN_WALK, refs: list[torch.Tensor] | None = None,
                ref_grids=()) -> list[list[torch.Tensor]]:
    """Latents after every step for each row: ``[step][row]``; one stacked forward per step or one per row.

    ``refs`` supplies each row's reference latents for the edit walk, where the denoise
    step runs over ``[txt | img | ref]`` tokens and a leak would cross rows through the
    reference block as readily as through the image one."""
    dit = subs["dit"]
    rids = [f"{prefix}{i}" for i in range(len(embeds))]
    latents: list[torch.Tensor | None] = [None] * len(embeds)
    steps: list[list[torch.Tensor]] = []
    with torch.inference_mode():
        for k in range(STEPS):
            rows = []
            for i, rid in enumerate(rids):
                inputs = {TEXT_EMBEDS: [embeds[i]]}
                if refs is not None:
                    inputs[REF_LATENTS] = [refs[i]]
                if latents[i] is not None:
                    inputs[LATENTS] = [latents[i]]
                rows.append(dit.prepare_inputs(
                    walk, _info(rid, k, seeds[i], walk=walk, ref_grids=ref_grids), inputs))
            if batched:
                kwargs = dit.preprocess(walk, _eng(rids), rows)
                out = dit.forward_batched(walk, _eng(rids), **kwargs)
                latents = [out[rid][LATENTS][0] for rid in rids]
            else:
                latents = []
                for rid, row in zip(rids, rows, strict=True):
                    kwargs = dit.preprocess(walk, _eng([rid]), [row])
                    latents.append(dit.forward(walk, _eng([rid]), **kwargs)[LATENTS][0])
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


def test_denoise_rows_are_independent_of_their_neighbours(subs, single):
    """Same batch size, rows in another order: every row must come back bit for bit."""
    perm = [3, 1, 0, 2]
    embeds = single["embeds"]
    straight = _trajectory(subs, embeds, "o", batched=True)
    shuffled = _trajectory(subs, [embeds[j] for j in perm], "p", batched=True, seeds=[SEEDS[j] for j in perm])
    for k, (a, b) in enumerate(zip(straight, shuffled, strict=True)):
        worst = max(_max_abs(a[j], b[perm.index(j)]) for j in range(len(perm)))
        print(f"dit step {k}, permuted batch vs batch: max_abs={worst:.3e}")
        for j in range(len(perm)):
            assert torch.equal(a[j], b[perm.index(j)]), f"step {k}: row {j} depends on its neighbours"


def test_denoise_rows_match_single_requests(model, subs, single):
    steps = _trajectory(subs, single["embeds"], "b", batched=True)
    for k, (batched_rows, single_rows) in enumerate(zip(steps, single["steps"], strict=True)):
        per_row = [f"{_max_abs(b, s):.1e}" for b, s in zip(batched_rows, single_rows, strict=True)]
        print(f"dit step {k}, batch of {len(PROMPTS)} vs alone, max_abs per row: {per_row}")
    images = _decode(model, subs, steps[-1], batched=False)
    psnrs = [_psnr(img, ref) for img, ref in zip(images, single["images"], strict=True)]
    for i, psnr in enumerate(psnrs):
        print(f"row {i} (seed {SEEDS[i]}): image PSNR batched-trajectory vs alone = {psnr:.2f} dB")
    assert min(psnrs) >= MIN_BATCH_PSNR_DB, f"batch-size numerics beyond a kernel-selection effect: {psnrs}"


def test_vae_decode_rows_match_single_requests(model, subs, single):
    images = _decode(model, subs, single["steps"][-1], batched=True)
    for i, (img, ref) in enumerate(zip(images, single["images"], strict=True)):
        psnr = _psnr(img, ref)
        print(f"row {i}: VAE decode in a batch of {len(PROMPTS)} vs alone = {psnr:.2f} dB, "
              f"max_abs={_max_abs(img, ref):.0f}")
        assert psnr >= MIN_PSNR_DB, f"row {i}: {psnr:.2f} dB < {MIN_PSNR_DB}"


# ----------------------------------------------------------------------
# Edit walk
#
# The denoise step for an edit runs over ``[txt | img | ref]`` tokens, so a
# batched edit has a conditioning block the text-to-image walk does not: a row
# could pick up a neighbour's reference as readily as its latents. The rows here
# share a reference GRID (they must, to batch at all — ``KleinShape`` keys on it)
# but carry different reference CONTENT, which is what makes a leak visible.
# ----------------------------------------------------------------------


def _reference_latents(model, subs, count: int) -> tuple[list[torch.Tensor], tuple]:
    """One distinct reference per row, encoded the way the edit walk encodes them."""
    encoder = subs["vae_encoder"]
    grid = model.config.latent_grid(HEIGHT, WIDTH)
    gen = torch.Generator().manual_seed(1234)
    refs = []
    with torch.inference_mode():
        for i in range(count):
            # float [3, H, W] in [0, 1] -- what Flux2KleinModel.load_image returns
            pixels = torch.rand(3, HEIGHT, WIDTH, generator=gen)
            rid = f"ref{i}"
            node = encoder.prepare_inputs(
                IMAGE_EDIT_WALK, _info(rid, 0, i, walk=IMAGE_EDIT_WALK, ref_grids=(grid,)),
                {IMAGE_INPUTS: [pixels]},
            )
            out = encoder.forward(IMAGE_EDIT_WALK, _eng([rid]), **node.tensor_inputs)
            refs.append(out[REF_LATENTS][0])
    assert len({r.shape for r in refs}) == 1, "rows must share a reference grid to batch"
    return refs, (grid,)


@pytest.fixture(scope="module")
def edit_single(model, subs) -> dict:
    """Each edit row run alone: the baseline a batched row must reproduce."""
    embeds = _encode(model, subs, batched=False)
    refs, ref_grids = _reference_latents(model, subs, len(PROMPTS))
    steps = _trajectory(subs, embeds, "es", batched=False,
                        walk=IMAGE_EDIT_WALK, refs=refs, ref_grids=ref_grids)
    return {"embeds": embeds, "refs": refs, "ref_grids": ref_grids, "steps": steps}


def test_edit_denoise_rows_are_independent_of_their_neighbours(subs, edit_single):
    """Same batch size, rows in another order: every row must come back bit for bit.

    Permuting the references as well as the prompts is the point — if a row read a
    neighbour's reference block, reordering the batch would change its output."""
    perm = [3, 1, 0, 2]
    embeds, refs, grids = edit_single["embeds"], edit_single["refs"], edit_single["ref_grids"]
    straight = _trajectory(subs, embeds, "eo", batched=True,
                           walk=IMAGE_EDIT_WALK, refs=refs, ref_grids=grids)
    shuffled = _trajectory(subs, [embeds[j] for j in perm], "ep", batched=True,
                           seeds=[SEEDS[j] for j in perm], walk=IMAGE_EDIT_WALK,
                           refs=[refs[j] for j in perm], ref_grids=grids)
    for k, (a, b) in enumerate(zip(straight, shuffled, strict=True)):
        worst = max(_max_abs(a[j], b[perm.index(j)]) for j in range(len(perm)))
        print(f"edit dit step {k}, permuted batch vs batch: max_abs={worst:.3e}")
        for j in range(len(perm)):
            assert torch.equal(a[j], b[perm.index(j)]), \
                f"edit step {k}: row {j} depends on its neighbours"


def test_edit_denoise_rows_match_single_requests(model, subs, edit_single):
    """A row inside a batch of four vs the same request alone, through to the image.

    This is the contract the served edit probe measures end to end; it broke in the
    engine's loop-counter handling without any single-request test noticing."""
    steps = _trajectory(subs, edit_single["embeds"], "eb", batched=True,
                        walk=IMAGE_EDIT_WALK, refs=edit_single["refs"],
                        ref_grids=edit_single["ref_grids"])
    for k, (batched_rows, single_rows) in enumerate(zip(steps, edit_single["steps"], strict=True)):
        per_row = [f"{_max_abs(b, s):.1e}" for b, s in zip(batched_rows, single_rows, strict=True)]
        print(f"edit dit step {k}, batch of {len(PROMPTS)} vs alone, max_abs per row: {per_row}")
    images = _decode(model, subs, steps[-1], batched=False)
    singles = _decode(model, subs, edit_single["steps"][-1], batched=False)
    psnrs = [_psnr(img, ref) for img, ref in zip(images, singles, strict=True)]
    for i, psnr in enumerate(psnrs):
        print(f"edit row {i} (seed {SEEDS[i]}): image PSNR batched-trajectory vs alone = {psnr:.2f} dB")
    assert min(psnrs) >= MIN_BATCH_PSNR_DB, f"edit batch-size numerics beyond kernel selection: {psnrs}"
