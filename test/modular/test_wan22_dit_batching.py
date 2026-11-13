"""Several requests a forward on Wan2.2's dit (``dit_max_batch_size`` above 1), on the CPU.

Requests at one latent grid share a forward whatever denoise step each is on, so a request's batch changes under it:
others join mid-loop, finish first or are cancelled. The contract is that none of it shows: after every step a request
holds the latents it gets alone, bit for bit. The DiT is the real one at a tiny size in float32, under FastWan's DMD
sampler (no CFG) and under UniPC with guidance (two branches a request).
"""

from __future__ import annotations

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.submodule_base import ModelInputsFromEngine
from mstar.model.wan22.components.dit import Wan22DiT
from mstar.model.wan22.config import SAMPLER_DMD, SAMPLER_UNIPC, Wan22Config
from mstar.model.wan22.submodules import DENOISE_LOOP_NAME, Wan22DitSubmodule, Wan22DmdDitSubmodule
from mstar.model.wan22.wan22_model import Wan22Model

WALK = Wan22Model.VIDEO_GEN_WALK
_TINY = dict(
    num_attention_heads=2, attention_head_dim=24, num_layers=2, ffn_dim=96, in_channels=4, out_channels=4,
    text_dim=8, freq_dim=16, rope_max_seq_len=16, compile_dit=False, dit_max_batch_size=4,
)
# 9 frames x 256x128 pixels: a latent grid of (3, 8, 16), 96 tokens a request
_SIZE = {"height": 128, "width": 256, "num_frames": 9}
_TEXT_TOKENS = 16
# what each sampler's request carries, and its step count
_REQUEST = {
    SAMPLER_DMD: {"num_inference_steps": 3, "guidance_scale": 1.0},
    SAMPLER_UNIPC: {"num_inference_steps": 4, "guidance_scale": 5.0},
}
_SEEDS = {"a": 11, "b": 12, "c": 13, "d": 14}
SAMPLERS = pytest.mark.parametrize("sampler", [SAMPLER_DMD, SAMPLER_UNIPC])


@pytest.fixture(autouse=True)
def _threads():
    # the comparisons are bitwise, and a BLAS may round differently per thread count
    torch.set_num_threads(2)


@pytest.fixture(scope="module")
def dit() -> Wan22DiT:
    torch.manual_seed(0)
    dit = Wan22DiT(Wan22Config(**_TINY))
    for param in dit.parameters():
        torch.nn.init.normal_(param, std=0.2)
    return dit.eval()


def _sub(dit: Wan22DiT, sampler: str) -> Wan22DitSubmodule:
    cls = Wan22DmdDitSubmodule if sampler == SAMPLER_DMD else Wan22DitSubmodule
    sub = cls(dit, Wan22Config(**_TINY, sampler=sampler))
    sub.get_device = lambda: torch.device("cpu")
    return sub


def _info(sampler: str, rid: str, step: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=WALK, fwd_index=step, random_seed=_SEEDS[rid], max_tokens=0,
        step_metadata={**_REQUEST[sampler], **_SIZE}, dynamic_loop_iter_counts={DENOISE_LOOP_NAME: step},
    )


def _text(rid: str) -> dict:
    gen = torch.Generator().manual_seed(100 + _SEEDS[rid])
    return {
        name: [torch.randn(1, _TEXT_TOKENS, _TINY["text_dim"], generator=gen)]
        for name in ("text_embeds_pos", "text_embeds_neg")
    }


def _serve(sub: Wan22DitSubmodule, sampler: str, forwards: list[tuple[str, ...]]) -> dict[str, list[torch.Tensor]]:
    """Run ``forwards``, each the requests sharing that step in row order, through the engine contract
    (prepare_inputs, preprocess, forward), routing the loop-back edges by hand. A request's step is the number of
    forwards it has had. Returns each request's latents after each of its steps."""
    carried: dict[str, dict] = {}
    latents: dict[str, list[torch.Tensor]] = {}
    for rids in forwards:
        infos = {rid: _info(sampler, rid, len(latents.get(rid, ()))) for rid in rids}
        inputs = [sub.prepare_inputs(WALK, infos[rid], {**_text(rid), **carried.get(rid, {})}) for rid in rids]
        engine_inputs = ModelInputsFromEngine(request_ids=list(rids), per_request_info=infos)
        kwargs = sub.preprocess(WALK, engine_inputs, inputs)
        if len(rids) == 1:
            outs = {rids[0]: sub.forward(WALK, engine_inputs, **kwargs)}
        else:
            assert sub.can_batch(None, inputs)
            outs = sub.forward_batched(WALK, engine_inputs, **kwargs)
        for rid in rids:
            carried[rid] = {name: outs[rid][name] for name in sub.loop_back_names}
            latents.setdefault(rid, []).append(outs[rid][sub.LATENTS][0])
    return latents


def _steps(sampler: str) -> int:
    return _REQUEST[sampler]["num_inference_steps"]


def _assert_as_alone(dit: Wan22DiT, sampler: str, served: dict[str, list[torch.Tensor]]) -> None:
    """Every step ``served`` holds for a request is that step of the request served alone."""
    for rid, got in served.items():
        alone = _serve(_sub(dit, sampler), sampler, [(rid,)] * len(got))[rid]
        for step, (x, y) in enumerate(zip(got, alone, strict=True)):
            assert torch.equal(x, y), f"request {rid}, step {step}: max |diff| {(x - y).abs().max().item():.3g}"


@SAMPLERS
def test_requests_sharing_every_forward_step_as_they_do_alone(dit, sampler):
    served = _serve(_sub(dit, sampler), sampler, [("a", "b", "c", "d")] * _steps(sampler))
    assert all(len(steps) == _steps(sampler) for steps in served.values())
    _assert_as_alone(dit, sampler, served)


@SAMPLERS
def test_requests_that_join_and_finish_mid_loop_step_as_they_do_alone(dit, sampler):
    """One request arrives per forward, so each forward holds rows at different steps, and a row moves up as the
    requests ahead of it finish."""
    steps, starts = _steps(sampler), {"a": 0, "b": 1, "c": 2, "d": 4}
    forwards = [
        tuple(rid for rid, start in starts.items() if start <= at < start + steps)
        for at in range(max(starts.values()) + steps)
    ]
    assert {len(rids) for rids in forwards} >= {1, 2, 3}
    served = _serve(_sub(dit, sampler), sampler, forwards)
    assert all(len(got) == steps for got in served.values())
    _assert_as_alone(dit, sampler, served)


@SAMPLERS
def test_a_cancelled_request_leaves_the_others_steps_as_they_are_alone(dit, sampler):
    """``b`` is cancelled after its first step and ``c`` after its second: neither rides a later forward."""
    steps = _steps(sampler)
    forwards = [("a", "b", "c"), ("a", "c"), *[("a",)] * (steps - 2)]
    served = _serve(_sub(dit, sampler), sampler, forwards)
    assert [len(served[rid]) for rid in "abc"] == [steps, 1, 2]
    _assert_as_alone(dit, sampler, served)


def test_the_dit_runs_the_patchify_and_the_embedder_once_per_request(dit):
    """Their kernels are picked by row count, so the dit calls them on the rows a request brings alone: its two
    branches under guidance, ``requests`` rows apart."""
    patch_rows, text_rows = [], []
    hooks = [
        dit.patch_embedding.register_forward_hook(lambda _m, args, _out: patch_rows.append(args[0].shape[0])),
        dit.condition_embedder.register_forward_hook(lambda _m, args, _out: text_rows.append(args[1].clone())),
    ]
    gen = torch.Generator().manual_seed(5)
    requests, branches = 3, 2
    hidden = torch.randn(branches * requests, 4, 3, 8, 16, generator=gen)
    timestep = torch.rand(branches * requests, 3, generator=gen) * 999
    text = torch.randn(branches * requests, _TEXT_TOKENS, 8, generator=gen)
    try:
        with torch.no_grad():
            out = dit(hidden, timestep, text, requests=requests)
            assert patch_rows == [branches] * requests
            assert all(torch.equal(rows, text[r::requests]) for r, rows in enumerate(text_rows))
            # a request's rows come back where they went in
            for r in range(requests):
                rows = slice(r, None, requests)
                assert torch.equal(out[rows], dit(hidden[rows], timestep[rows], text[rows]))
    finally:
        for hook in hooks:
            hook.remove()
