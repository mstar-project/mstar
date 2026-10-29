"""The unfiltered draw (``sampler.unfiltered``): its torch reference, which batches
may take it, and how a step finds out it runs it.

A decode batch whose rows filter nothing replays a second capture whose sampler
is one read of the logits. That is only right while no row filters, penalises or
reads a history, and only reachable where the walk captured the draw at the size
the batch pads to; otherwise the batch keeps the ordinary capture.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig, PackedAcceleratorGraphConfig
from mstar.engine.resources.sampler import unfiltered as U
from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sampler.utils import SamplingConfig
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext
from mstar.model.submodule_base import ARNodeInputs


def _u32(*words):
    return [torch.tensor([w], dtype=torch.int64) for w in words]


@pytest.mark.parametrize("ctr, key, expected", [
    # Random123's known answers for Philox-4x32-10
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
])
def test_reference_philox_matches_the_known_answers(ctr, key, expected):
    seed = key[0] | (key[1] << 32)
    out = U.philox4x32(seed, *_u32(*ctr))
    assert [int(w) for w in out] == list(expected)


def _draw(logits, temperature, seed, offset):
    tokens = torch.empty(logits.shape[0], dtype=torch.int64)
    offset = offset.clone()
    U.SAMPLE_UNFILTERED(logits, temperature, seed, offset, tokens, 1)
    return tokens, offset


def test_greedy_rows_take_the_first_index_of_the_max():
    logits = torch.zeros(3, 50)
    logits[0, [7, 20]] = 3.0  # an exact tie
    logits[1, 49] = 1.0
    logits[2] = torch.randn(50)
    tokens, offset = _draw(logits, torch.zeros(3), torch.arange(3), torch.tensor([4, 5, 6]))
    assert tokens.tolist() == [7, 49, int(logits[2].argmax())]
    assert offset.tolist() == [5, 6, 7]


def test_a_draw_repeats_for_its_offset_and_moves_with_it():
    logits = torch.randn(1, 4000)
    t, s = torch.tensor([1.0]), torch.tensor([11])
    a, _ = _draw(logits, t, s, torch.tensor([3]))
    b, _ = _draw(logits, t, s, torch.tensor([3]))
    assert a.tolist() == b.tolist()
    draws = {int(_draw(logits, t, s, torch.tensor([o]))[0]) for o in range(40)}
    assert len(draws) > 10, "consecutive offsets drew from the same noise"


def test_sampled_rows_follow_the_softmax():
    vocab, rows = 6, 3000
    row = torch.tensor([1.5, 0.2, -0.7, 0.9, 0.0, -2.0])
    temperature = 0.8
    logits = row.expand(rows, vocab).contiguous()
    tokens, _ = _draw(logits, torch.full((rows,), temperature), torch.arange(rows) * 7919 + 1,
                      torch.zeros(rows, dtype=torch.int64))
    counts = torch.bincount(tokens, minlength=vocab).double()
    expected = torch.softmax(row.double() / temperature, 0) * rows
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 20.5, f"chi2 {chi2:.1f} on 5 degrees of freedom (p < 0.001)"


def test_slot_masters_hold_the_offset_and_token():
    logits = torch.randn(3, 30)
    slots = torch.tensor([2, 0, 3])
    offset_master = torch.tensor([10, 20, 30, 40])
    token_master = torch.full((4,), -1, dtype=torch.int64)
    offset = torch.zeros(3, dtype=torch.int64)
    tokens = torch.empty(3, dtype=torch.int64)
    U.SAMPLE_UNFILTERED(logits, torch.zeros(3), torch.arange(3), offset, tokens, 1,
                        slots, offset_master, token_master)
    assert offset_master.tolist() == [11, 20, 31, 41]
    assert offset.tolist() == [31, 11, 41]
    assert token_master[slots].tolist() == tokens.tolist() == logits.argmax(-1).tolist()


@pytest.mark.parametrize("kwargs, unfiltered", [
    ({}, True),
    ({"temperature": 0.0}, True),
    ({"temperature": 0.0, "top_k": 5}, True),  # a greedy row ignores its filters
    ({"top_k": 20}, False),
    ({"top_p": 0.95}, False),
    ({"min_p": 0.1}, False),
    ({"repetition_penalty": 1.1}, False),
    ({"repetition_window": 8, "repetition_penalty": 1.1}, False),
    ({"min_tokens": 3}, False),
    ({"top_p_first": True}, False),
])
def test_which_requests_filter_nothing(kwargs, unfiltered):
    assert U.filters_nothing(SamplingConfig(**kwargs)) is unfiltered


def _lease(bs: int, key=None) -> SlotLease:
    return SlotLease(slot=0, bucket=BucketKey("decode", bs, bs, key))


def _plan(res: SamplerResource, rids, lease: SlotLease, capture: bool) -> None:
    ctx = StepContext(request_ids=tuple(rids), graph_walk="decode", slot=0, capture=capture, slot_lease=lease)
    res.plan(SamplerStep(), ctx)


def _captured(sizes=(1, 2, 4, 8), unfiltered=(1, 2, 4, 8)) -> SamplerResource:
    res = SamplerResource(vocab_size=64, enable_repetion_penalty=True, device=torch.device("cpu"))
    res.buckets_captured([BucketKey("decode", bs, bs) for bs in sizes])
    res.buckets_captured([BucketKey("decode", bs, bs, U.UNFILTERED_SAMPLING) for bs in unfiltered])
    return res


def test_a_batch_that_filters_nothing_takes_the_unfiltered_capture():
    res = _captured()
    for rid in (0, 1, 2):
        res.ingest_request(rid, SamplingReqConfig(temperature=0.6))
    assert res.unfiltered_key("decode", [0, 1, 2]) == U.UNFILTERED_SAMPLING
    res.ingest_request(3, SamplingReqConfig(top_p=0.9))
    assert res.unfiltered_key("decode", [0, 1, 2, 3]) is None
    assert res.unfiltered_key("decode", [0, 2]) == U.UNFILTERED_SAMPLING
    res.remove_request(3)
    assert res.unfiltered_key("decode", [0, 1, 2]) == U.UNFILTERED_SAMPLING


def test_only_where_the_walk_captured_it_at_the_padded_size():
    # the unfiltered capture lacks bs=4: a 3- or 4-row batch keeps the ordinary one
    res = _captured(unfiltered=(1, 2, 8))
    for rid in range(8):
        res.ingest_request(rid, SamplingReqConfig())
    assert res.unfiltered_key("decode", [0, 1]) == U.UNFILTERED_SAMPLING
    assert res.unfiltered_key("decode", [0, 1, 2]) is None
    assert res.unfiltered_key("decode", list(range(6))) == U.UNFILTERED_SAMPLING
    assert res.unfiltered_key("prefill_text", [0]) is None
    assert _captured(unfiltered=()).unfiltered_key("decode", [0]) is None


def test_the_leased_bucket_decides_the_draw():
    res = _captured()
    res.ingest_request(0, SamplingReqConfig(repetition_penalty=1.2))
    res.ingest_request(1, SamplingReqConfig())
    _plan(res, [1], _lease(1, U.UNFILTERED_SAMPLING), capture=False)
    assert res._unfiltered_step
    # a penalised request is resident, but the unfiltered step reads no mask
    assert not res._penalty_needed_this_step
    _plan(res, [0, 1], _lease(2), capture=False)
    assert not res._unfiltered_step and res._penalty_needed_this_step
    ctx = StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=False)
    res.plan(SamplerStep(), ctx)
    assert not res._unfiltered_step, "an eager step kept the last replay's draw"


def _configs():
    dummy = ARNodeInputs(input_ids=torch.zeros(1, dtype=torch.long), input_seq_len=1)
    return [
        BatchedAcceleratorGraphConfig(capture_graph_walk="decode", single_request_inputs=dummy,
                                      capture_batch_sizes=[1, 2]),
        BatchedAcceleratorGraphConfig(capture_graph_walk="decode", single_request_inputs=dummy,
                                      additional_key_info="other"),
        PackedAcceleratorGraphConfig(capture_graph_walk="prefill_text", capture_token_lengths=[32],
                                     make_node_input=lambda n: dummy, capture_batch_sizes=[1]),
    ]


def test_configs_gain_an_unfiltered_copy_only_with_an_implementation(monkeypatch):
    configs = _configs()
    if U.SAMPLE_UNFILTERED.unavailable_reason() is not None:
        assert U.with_unfiltered_sampling(configs, walks=("decode",)) == configs
    monkeypatch.setattr(U.SAMPLE_UNFILTERED, "unavailable_reason", lambda: None)
    out = U.with_unfiltered_sampling(configs, walks=("decode",))
    assert out[:3] == configs
    assert len(out) == 4
    extra = out[3]
    assert extra.capture_graph_walk == "decode" and extra.additional_key_info == U.UNFILTERED_SAMPLING
    assert extra.capture_batch_sizes == [1, 2]
    assert configs[0].additional_key_info is None, "the original config was changed"
