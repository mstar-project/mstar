"""Qwen3-Omni Talker: an EOS frame never reaches the Code2Wav stream.

The Talker spans ``codec_tokens`` on the walks that sample a frame; a row that
samples codec_eos gets an empty span, which the worker turns into no stream
item (test_output_spans.py). Drives the real ``preprocess`` and forwards on
CPU with an identity backbone and depth decoder, which need the engine's
attention resources otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.model.qwen3_omni.config import (
    CODE_PRED_SAMPLER,
    TALKER_SAMPLER,
    CodePredictorConfig,
    Qwen3OmniModelConfig,
)
from mstar.model.qwen3_omni.submodules import TalkerSubmodule
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine

HIDDEN, GROUPS = 8, 4


class _Backbone(torch.nn.Module):
    """The Talker's heads at their real vocab, around an identity backbone."""

    def __init__(self, vocab: int):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.codec_embedding = torch.nn.Embedding(vocab, HIDDEN)
        self.codec_head = torch.nn.Linear(HIDDEN, vocab, bias=False)

    def forward(self, input_embeds, label):
        return input_embeds


class _DepthDecoder(torch.nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.codec_embedding = torch.nn.ModuleList(
            torch.nn.Embedding(vocab, HIDDEN) for _ in range(GROUPS - 1)
        )
        self.lm_head_weight = torch.nn.Parameter(torch.zeros(GROUPS - 1, vocab, HIDDEN))

    def forward_depth_unrolled(self, inputs_embeds, position_ids, kv_cache, cache_pos):
        return inputs_embeds


@pytest.fixture
def talker():
    config = Qwen3OmniModelConfig(code_predictor=CodePredictorConfig(
        vocab_size=16, hidden_size=HIDDEN, num_hidden_layers=1, num_attention_heads=1,
        num_key_value_heads=1, head_dim=2, num_code_groups=GROUPS,
    ))
    return TalkerSubmodule(_Backbone(config.talker_text.vocab_size), _DepthDecoder(16), config)


def _run(talker, method, graph_walk, tokens, seq_len=1):
    """One step whose rows sample ``tokens`` for codebook 0 (and 0 for the rest)."""
    rids = [f"r{i}" for i in range(len(tokens))]
    sampler = SimpleNamespace(sample=lambda request_ids, logits: torch.tensor(tokens))
    code_sampler = SimpleNamespace(sample=lambda request_ids, logits: torch.zeros(len(tokens), dtype=torch.long))
    engine_inputs = ModelInputsFromEngine(
        request_ids=rids, per_request_info={},
        resources={TALKER_SAMPLER: sampler, CODE_PRED_SAMPLER: code_sampler},
    )
    inputs = [ARNodeInputs(input_embeds=torch.zeros(seq_len, HIDDEN), input_seq_len=seq_len) for _ in rids]
    packed = talker.preprocess(graph_walk, engine_inputs, inputs)
    with torch.no_grad():
        return getattr(talker, method)(graph_walk, engine_inputs, **packed)


@pytest.mark.parametrize("graph_walk, seq_len", [("talker_decode", 1), ("talker_last_prefill", 6)])
def test_an_eos_frame_gets_an_empty_span(talker, graph_walk, seq_len):
    eos = talker.config.talker.codec_eos_token_id
    out = _run(talker, "forward_batched", graph_walk, [5, eos, 7], seq_len)
    assert out.output_spans["codec_tokens"].tolist() == [[0, GROUPS], [0, 0], [0, GROUPS]]
    codes = [out.per_rid_outputs[rid]["codec_tokens"][0] for rid in ("r0", "r1", "r2")]
    assert [frame.shape for frame in codes] == [(GROUPS,)] * 3
    assert [frame[0].item() for frame in codes] == [5, eos, 7]


def test_the_unbatched_forward_keys_its_request_and_spans_its_frame(talker):
    eos = talker.config.talker.codec_eos_token_id
    for token, span in ((5, [[0, GROUPS]]), (eos, [[0, 0]])):
        out = _run(talker, "forward", "talker_decode", [token])
        assert list(out.per_rid_outputs) == ["r0"]
        assert out.per_rid_outputs["r0"]["codec_tokens"][0].shape == (1, GROUPS)
        assert out.output_spans["codec_tokens"].tolist() == span


def test_only_the_walks_that_sample_a_frame_span_it(talker):
    assert talker.spanned_outputs("talker_decode") == talker.spanned_outputs("talker_last_prefill") == {"codec_tokens"}
    # the plain prefill only fills the KV cache
    assert talker.spanned_outputs("talker_prefill") == frozenset()
