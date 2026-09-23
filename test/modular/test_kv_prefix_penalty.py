"""A cache hit must not change which tokens the repetition penalty has seen.

A model's ``declare_step`` builds its tracked tokens out of the inputs it is
handed, and by then the engine has already cut the matched prefix out of them.
So a request served from the cache would seed its seen-token mask with the tail
of its prompt alone, and the penalty would happily let the model repeat
everything before it: the same prompt, the same seed, a different answer,
depending only on whether the cache happened to be warm. Nothing raises. These
tests compare a cut run's mask against the mask of the same prompt run whole.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import torch

from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import StepContext
from mstar.model.submodule_base import ARNodeInputs

RID = "r0"
NODE = "LLM"
WALK = "prefill"
PROMPT = torch.arange(100)
MATCHED = 96


def _sampler(penalty: float = 1.2) -> SamplerResource:
    resource = SamplerResource(
        vocab_size=256, enable_repetion_penalty=True,
        device=torch.device("cpu"), comm_group=None,
    )
    resource.ingest_request(RID, SamplingReqConfig(repetition_penalty=penalty))
    return resource


def _ctx() -> StepContext:
    return StepContext(
        request_ids=(RID,), graph_walk=WALK, slot=0, capture=False,
    )


def _prefill(resource: SamplerResource, tracked: torch.Tensor) -> None:
    """Plan the one prefill step, tracking ``tracked`` as its prompt."""
    resource.plan(SamplerStep(prefill_tracked_tokens={RID: tracked}), _ctx())


def _seen(resource: SamplerResource) -> list[int]:
    mask = resource._sampler.get_token_mask(RID)._seen_token_mask
    return mask.nonzero().flatten().tolist()


# ── the mask after a hit ────────────────────────────────────────────────


def test_a_cut_prefill_still_ends_with_the_whole_prompt_in_the_mask():
    resource = _sampler()
    inputs = ARNodeInputs(input_ids=PROMPT, input_seq_len=len(PROMPT))

    resource.apply_cached_prefix(RID, NODE, WALK, inputs, MATCHED)
    _prefill(resource, PROMPT[MATCHED:])

    assert _seen(resource) == PROMPT.tolist(), (
        "the mask holds only the tokens that were not served from the cache"
    )


def test_the_skipped_tokens_are_folded_in_once():
    resource = _sampler()
    resource.apply_cached_prefix(
        RID, NODE, WALK,
        ARNodeInputs(input_ids=PROMPT, input_seq_len=len(PROMPT)), MATCHED,
    )
    _prefill(resource, PROMPT[MATCHED:])

    resource.plan(SamplerStep(prefill_tracked_tokens={}), _ctx())

    assert RID not in resource._cached_prefix, (
        "a decode step would re-scatter the prefix on every step of the request"
    )


# ── when there is nothing to fold ───────────────────────────────────────


def test_a_miss_keeps_nothing():
    resource = _sampler()

    resource.apply_cached_prefix(
        RID, NODE, WALK,
        ARNodeInputs(input_ids=PROMPT, input_seq_len=len(PROMPT)), 0,
    )

    assert RID not in resource._cached_prefix, (
        "a request that matched nothing left tokens for the mask to fold in"
    )


def test_a_walk_without_token_ids_keeps_nothing():
    resource = _sampler()

    resource.apply_cached_prefix(
        RID, NODE, WALK,
        ARNodeInputs(input_embeds=torch.zeros(100, 8), input_seq_len=100),
        MATCHED,
    )

    assert RID not in resource._cached_prefix, (
        "a walk with no ids to keep kept something anyway"
    )


def test_an_inert_penalty_leaves_the_mask_alone():
    resource = _sampler(penalty=1.0)
    resource.apply_cached_prefix(
        RID, NODE, WALK,
        ARNodeInputs(input_ids=PROMPT, input_seq_len=len(PROMPT)), MATCHED,
    )

    _prefill(resource, PROMPT[MATCHED:])

    assert _seen(resource) == [], (
        "a request nobody penalises paid for the mask traffic anyway"
    )


def test_a_preplan_folds_nothing_in():
    resource = _sampler()
    resource.apply_cached_prefix(
        RID, NODE, WALK,
        ARNodeInputs(input_ids=PROMPT, input_seq_len=len(PROMPT)), MATCHED,
    )

    resource.plan(
        SamplerStep(prefill_tracked_tokens={RID: PROMPT[MATCHED:]}),
        StepContext(
            request_ids=(RID,), graph_walk=WALK, slot=0, capture=False,
            is_preplan=True,
        ),
    )

    assert RID in resource._cached_prefix, (
        "a staged step consumed the prefix the real step still needs"
    )


def test_removing_the_request_drops_what_was_kept_for_it():
    resource = _sampler()
    resource.apply_cached_prefix(
        RID, NODE, WALK,
        ARNodeInputs(input_ids=PROMPT, input_seq_len=len(PROMPT)), MATCHED,
    )

    resource.remove_request(RID)

    assert RID not in resource._cached_prefix, (
        "the skipped prompt outlived the request it was kept for"
    )
