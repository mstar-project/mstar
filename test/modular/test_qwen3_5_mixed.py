"""Qwen3.5 text prefill and decode rows ride in one step: the model declares
the combined walk, a mixed step tracks only its prefill rows' prompt tokens,
and the GDN kernel choice follows a packed capture rather than the spans."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.linear_attn.gdn import GDNManager
from mstar.model.qwen3_5.config import LLM_MIXED, SAMPLER
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.qwen3_5.submodules import LLMSubmodule
from mstar.model.submodule_base import ARNodeInputs


def test_text_prefill_and_decode_share_the_mixed_walk():
    assert Qwen3_5DenseModel.get_combined_graph_walks(None) == {
        "LLM": {LLM_MIXED: {"prefill_text", "decode"}},
    }


def test_a_mixed_step_tracks_only_its_prefill_rows_prompt_tokens():
    llm = object.__new__(LLMSubmodule)
    prompt = torch.tensor([5, 6, 7])
    rows = [
        ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([9]), graph_walk="decode"),
        ARNodeInputs(input_seq_len=3, input_ids=prompt, graph_walk="prefill_text"),
    ]

    step = llm.declare_step(LLM_MIXED, [0, 1], rows)

    tracked = step.steps[SAMPLER].prefill_tracked_tokens
    assert list(tracked) == [1] and tracked[1] is prompt


class _Recorder(GDNManager):
    def __init__(self):
        self.kinds = []

    def _get_wrapper(self, label, is_decode, lease=None):
        self.kinds.append("decode" if is_decode else "prefill")
        return SimpleNamespace(plan=lambda *args: None)


@pytest.mark.parametrize("force_prefill,spans,kind", [
    (False, [1, 1], "decode"),
    (False, [1, 3], "prefill"),
    (True, [1, 1], "prefill"),  # a packed capture replays mixed steps
])
def test_the_kernel_follows_a_packed_capture_not_the_spans(force_prefill, spans, kind):
    gdn = _Recorder()
    segments = [SimpleNamespace(span=s) for s in spans]
    addressing = SimpleNamespace(slot_indices=torch.zeros(4, dtype=torch.long),
                                 has_state=torch.zeros(4, dtype=torch.bool))
    ctx = SimpleNamespace(force_prefill=force_prefill, slot_lease=None)

    gdn._build_plan("main", segments, addressing, ctx)

    assert gdn.kinds == [kind]


@pytest.mark.parametrize("walk,inputs,length", [
    ("decode", {}, 1),
    ("prefill_text", {"text_inputs": [torch.zeros(7, dtype=torch.long)]}, 7),
    ("prefill_vision", {"text_inputs": [torch.zeros(7, dtype=torch.long)]}, None),
])
def test_text_and_decode_rows_report_their_length(walk, inputs, length):
    info = LLMSubmodule.get_input_sequence_len(None, walk, None, inputs)

    assert (info.seq_len if info is not None else None) == length


def test_text_prefill_chunks_under_a_budget_that_fits_the_captures():
    llm = object.__new__(LLMSubmodule)

    assert llm.supports_chunked_prefill("prefill_text")
    assert not llm.supports_chunked_prefill("prefill_vision")
    assert llm.max_batch_tokens(LLM_MIXED) == llm.MAX_BATCH_TOKENS
    largest = max(LLMSubmodule.PREFILL_TOKEN_BUCKETS)
    assert llm.MAX_BATCH_TOKENS + max(LLMSubmodule.DECODE_CAPTURE_BATCH_SIZES) <= largest


def test_a_chunked_prompt_keeps_only_its_final_chunks_token():
    from mstar.model.submodule_base import ARNodeSubmodule

    llm = object.__new__(LLMSubmodule)
    full = ARNodeInputs(input_seq_len=10, input_ids=torch.arange(10))
    info = SimpleNamespace(graph_walk="prefill_text", step_metadata={"last_prefill": True})
    kept = []
    for start, end in ((0, 4), (4, 10)):
        chunk = ARNodeSubmodule.split_inputs(llm, "prefill_text", info, full, start, end)
        chunk.chunk_start, chunk.chunk_total = start, full.input_seq_len  # as the engine stamps them
        outputs = {"new_token": [torch.tensor([1])]}
        llm.postprocess(0, info, outputs, inputs=chunk)
        kept.append("text_inputs" in outputs)

    assert kept == [False, True]
