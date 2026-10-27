"""A captured prefill replay's padding rows stay out of the seen-token masks.

The engine declares a leased step over the padded batch, so a model that tracks
every prefill row's ids (Orpheus, Qwen3.5, Qwen3-Omni) hands the sampler an
entry for each padding row too. Those rows are capture rows, which hold no mask,
so folding their ids in logs "add_tokens on an uninitialized SeenTokenMask" for
each one and buries the case the warning is for: a real request whose prompt
would go unpenalized.
"""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")

import torch

from mstar.engine.accelerator_graph_runner import dummy_rid_handle
from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import StepContext

VOCAB = 64
REAL = (0, 1)
PROMPTS = {0: torch.tensor([3, 5, 7]), 1: torch.tensor([11, 13])}
PAD = tuple(dummy_rid_handle(f"__cg_test_pad_{i}__") for i in range(2))
UNINITIALIZED = "uninitialized SeenTokenMask"


def _resource() -> SamplerResource:
    res = SamplerResource(
        vocab_size=VOCAB, enable_repetition_penalty=True, device=torch.device("cpu"),
    )
    for rid in REAL:
        res.ingest_request(rid, SamplingReqConfig(repetition_penalty=1.3))
    for rid in PAD:
        res.ingest_capture_row(rid)
    return res


def _padded_prefill(res: SamplerResource, real=REAL) -> None:
    """Plan a prefill replay padded to a bs=4 bucket, tracked the way the
    models declare it: every row's ids, the padding rows' empty."""
    tracked = {rid: PROMPTS[rid] for rid in real}
    tracked.update({rid: torch.empty(0, dtype=torch.long) for rid in PAD})
    ctx = StepContext(request_ids=tuple(real), graph_walk="prefill", slot=0, capture=False)
    ctx.set_padded_rids((*real, *PAD))
    res.plan(SamplerStep(prefill_tracked_tokens=tracked), ctx)


def _seen(res: SamplerResource, rid) -> list[int]:
    return res._sampler.get_token_mask(rid)._seen_token_mask.nonzero().flatten().tolist()


def test_a_padded_prefill_logs_nothing_for_its_padding_rows(caplog):
    res = _resource()

    with caplog.at_level(logging.WARNING):
        _padded_prefill(res)

    assert UNINITIALIZED not in caplog.text, (
        "the padding rows' empty tracked ids reached their unsized masks"
    )


def test_the_real_rows_still_hold_their_prompts():
    res = _resource()

    _padded_prefill(res)

    assert _seen(res, 0) == PROMPTS[0].tolist()
    assert _seen(res, 1) == PROMPTS[1].tolist()
    assert all(
        res._sampler.get_token_mask(rid)._seen_token_mask is None for rid in PAD
    ), "a padding row was given a vocab-sized mask"


def test_a_real_request_with_an_unsized_mask_still_warns(caplog):
    # an unsized real mask would drop its prompt from the penalty
    res = _resource()
    res._sampler._seen_token_mask[1]._seen_token_mask = None

    with caplog.at_level(logging.WARNING):
        _padded_prefill(res)

    assert caplog.text.count(UNINITIALIZED) == 1
