"""Whisper decoder hooks that run per step on the host, checked on the CPU with
no weights: the batched stop check agrees with the per-request one, a decode
step declared over the capture template is the step its loop-back rows declare,
and the batch caps per walk."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from mstar.model.submodule_base import ARNodeInputs, HostRows
from mstar.model.whisper.config import CROSS_KV_CACHE, SAMPLER, WhisperModelConfig
from mstar.model.whisper.submodules import WhisperDecoderSubmodule
from mstar.model.whisper.whisper_model import WhisperModel

EOS = WhisperModelConfig().eos_token_id


def _decoder() -> WhisperDecoderSubmodule:
    sub = object.__new__(WhisperDecoderSubmodule)
    torch.nn.Module.__init__(sub)
    sub.config = WhisperModelConfig()
    return sub


def _info(decoded: int, max_tokens: int = 444, ignore_eos: bool = False):
    return SimpleNamespace(
        resource_configs={SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)},
        dynamic_loop_iter_counts={"decode_loop": decoded} if decoded else {},
        max_tokens=max_tokens,
    )


CASES = {
    "plain": (123, _info(5)),
    "eos": (EOS, _info(5)),
    "eos_ignored": (EOS, _info(5, ignore_eos=True)),
    "last_token": (123, _info(443)),
    "first_token_eos": (EOS, _info(0)),
}


@pytest.mark.parametrize("order", ["forward", "shuffled"])
def test_batched_stop_check_agrees_with_per_request(order):
    sub = _decoder()
    rids = list(range(len(CASES)))
    tokens = [token for token, _ in CASES.values()]
    infos = dict(zip(rids, (info for _, info in CASES.values()), strict=True))
    rows = tuple(rids) if order == "forward" else tuple(reversed(rids))
    values = tokens if order == "forward" else list(reversed(tokens))
    host = HostRows(rows, {"new_token": torch.tensor(values)})
    expected = {}
    for rid, token in zip(rids, tokens, strict=True):
        stops = sub.check_stop(rid, infos[rid], {"new_token": [torch.tensor([token])]})
        if stops:
            expected[rid] = stops
    assert sub.check_stop_batched(rids, infos, host, step_walk="decode") == expected
    assert set(expected) == {1, 3, 4}


def test_a_request_without_a_row_stops_nothing():
    sub = _decoder()
    host = HostRows((0,), {"new_token": torch.tensor([EOS])})
    assert sub.check_stop_batched([0, 1], {0: _info(1), 1: _info(1)}, host) == {0: {"decode_loop"}}


def test_decode_template_declares_the_loopback_rows_step():
    sub = _decoder()
    rids = [0, 1, 2]
    template = [ARNodeInputs(input_ids=torch.zeros(1, dtype=torch.long), input_seq_len=1) for _ in rids]
    loopback = [ARNodeInputs(input_seq_len=1) for _ in rids]
    assert sub.step_is_input_free("decode")
    assert not sub.step_is_input_free("prefill")
    assert sub.declare_step("decode", rids, template) == sub.declare_step("decode", rids, loopback)


def test_batch_caps_per_walk():
    sub = _decoder()
    model = object.__new__(WhisperModel)
    model.config = WhisperModelConfig()
    context = next(s.config for s in model.get_node_resources() if s.resource_key == CROSS_KV_CACHE)
    sub.node_resources = {CROSS_KV_CACHE: SimpleNamespace(config=context)}
    assert sub.max_batch_size("prefill") == WhisperModel.MAX_CONCURRENT_REQUESTS
    # decode is bounded by its captured sizes, which run to the same cap
    assert sub.max_batch_size("decode") is None
    assert max(WhisperDecoderSubmodule.DECODE_CAPTURE_BATCH_SIZES) == WhisperModel.MAX_CONCURRENT_REQUESTS
    smaller = dataclasses.replace(context, max_num_pages=1 + 12 * 10)
    sub.node_resources = {CROSS_KV_CACHE: SimpleNamespace(config=smaller)}
    assert sub.max_batch_size("prefill") == 10


def test_prefill_capture_names_the_context_its_rows_declare():
    """The prefill config sizes the context cache by what its template rows
    declare, and a real batch declares that same step, so a pre-plan staged
    over the template is the one the batch promotes."""
    sub = _decoder()
    configs = {c.capture_graph_walk: c for c in sub.get_accelerator_graph_configs(torch.device("cpu"))}
    prefill = configs["prefill"]
    assert prefill.capture_batch_sizes == WhisperDecoderSubmodule.PREFILL_CAPTURE_BATCH_SIZES
    context = sub.config.max_source_positions
    assert prefill.resource_write_tokens == {CROSS_KV_CACHE: context}

    rids = [0, 1, 2, 3]
    template = prefill.get_node_inputs(len(rids), 0)
    step = sub.declare_step("prefill", rids, template)
    assert [seg.span for seg in step.steps[CROSS_KV_CACHE].segments] == [context] * len(rids)
    assert sum(seg.span for seg in step.steps[CROSS_KV_CACHE].segments) == prefill.write_tokens(
        CROSS_KV_CACHE, len(rids),
    )

    prompt = torch.zeros(sub.config.decoder_prompt_len, dtype=torch.long)
    real = [
        ARNodeInputs(
            input_ids=prompt, input_seq_len=len(prompt),
            tensor_inputs={"encoder_states": torch.randn(context, sub.config.d_model)},
        ) for _ in rids
    ]
    assert sub.declare_step("prefill", rids, real) == step
    # the encoder hands over float32 (its last layer norm, under autocast)
    assert template[0].tensor_inputs["encoder_states"].dtype == torch.float32
