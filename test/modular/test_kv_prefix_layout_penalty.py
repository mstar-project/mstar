"""Every walk the cache skips keeps its tokens for the penalty mask.

A prompt laid out text, image, text can have its first text walk served whole
and its last one trimmed. The served walk runs no plan, so the ids it skipped
wait for the walk that does; kept in one slot per request, the trimmed walk's
own skipped ids replaced them, and the mask never saw the text before the
image.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import torch

from mstar.engine.resources.base import CachedPrefix
from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import StepContext
from mstar.model.submodule_base import ARNodeInputs

RID = "r0"
NODE = "LLM"
TEXT = torch.arange(10, 30)
TAIL = torch.arange(100, 125)
TRIMMED = 14


def test_skipped_ids_from_two_walks_both_reach_the_penalty_mask():
    sampler = SamplerResource(vocab_size=256, enable_repetion_penalty=True, device=torch.device("cpu"), comm_group=None)
    sampler.ingest_request(RID, SamplingReqConfig(repetition_penalty=1.2))
    # text served whole, a one-position image served whole, then 14 of the tail
    for walk, inputs, prefix in (
        ("prefill", ARNodeInputs(input_ids=TEXT, input_seq_len=len(TEXT)), CachedPrefix("main", len(TEXT), len(TEXT))),
        ("prefill_image", ARNodeInputs(input_seq_len=30), CachedPrefix("main", 30, len(TEXT) + 1)),
        ("prefill", ARNodeInputs(input_ids=TAIL, input_seq_len=len(TAIL)), CachedPrefix("main", TRIMMED, 35)),
    ):
        sampler.apply_cached_prefix(RID, NODE, walk, inputs, prefix)

    sampler.plan(
        SamplerStep(prefill_tracked_tokens={RID: TAIL[TRIMMED:]}),
        StepContext(request_ids=(RID,), graph_walk="prefill", slot=0, capture=False),
    )

    seen = sampler._sampler.get_token_mask(RID)._seen_token_mask.nonzero().flatten().tolist()
    assert seen == sorted(TEXT.tolist() + TAIL.tolist()), (
        "the text before the image never reached the mask, so the penalty "
        "lets the model repeat it only when the cache was warm"
    )
