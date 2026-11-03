"""The fused filtered draw (``sampler.filtered``): its torch reference, and which sampler takes it.

A code predictor's sampler declares ``fused_draw``; a captured step then draws in one kernel where the
platform has one, and from the same distribution as the full sampler. The kernel itself is checked against
this reference in ``test/rocm/test_sampler_filtered.py``.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.sampler import filtered as F
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sampler.utils import (
    FUSED_FILTER_MAX_VOCAB,
    AcceleratorGraphableSampler,
    SamplerBuffers,
)


def _draw(logits, temperature, top_k, top_p, seed=None, offset=None):
    rows = logits.shape[0]
    tokens = torch.zeros(rows, dtype=torch.int64)
    offset = torch.zeros(rows, dtype=torch.int64) if offset is None else offset
    seed = torch.arange(rows, dtype=torch.int64) * 31 + 7 if seed is None else seed
    F._reference(logits, torch.full((rows,), temperature), torch.full((rows,), top_k, dtype=torch.int32),
                 torch.full((rows,), top_p), seed, offset, tokens, 1)
    return tokens, offset


def test_greedy_rows_take_the_first_index_of_the_max_whatever_the_filters():
    logits = torch.tensor([[0.0, 3.0, 3.0, 1.0], [5.0, 0.0, 5.0, 5.0]])
    tokens, _ = _draw(logits, 0.0, 1, 0.1)
    assert tokens.tolist() == [1, 0]


def test_top_k_keeps_the_k_largest_and_their_ties():
    scaled = torch.tensor([1.0, 4.0, 3.0, 3.0, 0.0])
    assert F.kept_tokens(scaled, 1, 1.0).tolist() == [False, True, False, False, False]
    assert F.kept_tokens(scaled, 2, 1.0).tolist() == [False, True, True, True, False]
    assert F.kept_tokens(scaled, 0, 1.0).all() and F.kept_tokens(scaled, 5, 1.0).all()


def test_top_p_runs_over_what_top_k_kept():
    # probabilities 0.6, 0.3, 0.09, 0.01: top-p 0.8 keeps the first two tokens
    scaled = torch.tensor([0.6, 0.3, 0.09, 0.01]).log()
    assert F.kept_tokens(scaled, 0, 0.8).tolist() == [True, True, False, False]
    # top-k 1 leaves one token, and top-p cannot cut it
    assert F.kept_tokens(scaled, 1, 0.1).tolist() == [True, False, False, False]
    # the top token alone covers less than 0.8 of what top-k 3 keeps
    assert F.kept_tokens(scaled, 3, 0.8).tolist() == [True, True, False, False]


def test_a_draw_repeats_for_its_offset_and_moves_with_it():
    logits = torch.randn(1, 64)
    first, advanced = _draw(logits, 0.9, 0, 1.0, seed=torch.tensor([5]))
    again, _ = _draw(logits, 0.9, 0, 1.0, seed=torch.tensor([5]))
    assert first.tolist() == again.tolist() and advanced.tolist() == [1]
    tokens = {int(_draw(logits, 0.9, 0, 1.0, seed=torch.tensor([5]), offset=torch.tensor([i]))[0]) for i in range(40)}
    assert len(tokens) > 1


def test_rows_draw_independently_of_their_position_in_the_batch():
    logits = torch.randn(1, 64).expand(4, 64).contiguous()
    seed = torch.tensor([9, 9, 9, 9])
    tokens, _ = _draw(logits, 0.9, 0, 1.0, seed=seed)
    assert len(set(tokens.tolist())) == 1


@pytest.mark.parametrize("top_k, top_p", [(5, 1.0), (0, 0.8), (6, 0.9)])
def test_the_draw_follows_the_renormalised_kept_set(top_k, top_p):
    vocab, rows, t = 12, 1 << 13, 0.9
    row = torch.linspace(-2, 2, vocab)
    tokens, _ = _draw(row.expand(rows, vocab).contiguous(), t, top_k, top_p)
    kept = F.kept_tokens(row * (1.0 / torch.tensor([t])), top_k, top_p)
    expected = torch.where(kept, torch.softmax(row.double() / t, 0), torch.zeros(vocab).double())
    expected /= expected.sum()
    counts = torch.bincount(tokens, minlength=vocab).double() / rows
    assert counts[~kept].sum() == 0
    assert 0.5 * (counts - expected).abs().sum() < 0.03


def test_nan_logits_never_win():
    logits = torch.randn(2, 8)
    logits[0] = float("nan")
    logits[1, :4] = float("nan")
    tokens, _ = _draw(logits, 0.7, 3, 1.0)
    assert 0 <= int(tokens[0]) < 8 and int(tokens[1]) >= 4


def _sampler(**fields) -> AcceleratorGraphableSampler:
    rows = 2
    return AcceleratorGraphableSampler(
        temperature_buf=torch.full((rows,), 0.9), top_k_buf=torch.full((rows,), 3, dtype=torch.int32),
        top_p_buf=torch.ones(rows), seed_buf=torch.arange(rows, dtype=torch.int64),
        offset_buf=torch.zeros(rows, dtype=torch.int64), **fields,
    )


def test_a_fused_sampler_takes_the_draw_only_when_it_can(monkeypatch):
    calls = []
    sampler = _sampler(fused_draw=True)
    monkeypatch.setattr(
        AcceleratorGraphableSampler, "_sample_fused", lambda self, logits: calls.append(logits) or "fused",
    )
    monkeypatch.setattr("mstar.engine.resources.sampler.utils.sample_accelerator_graphable_gpu",
                        lambda *args, **kwargs: torch.zeros(2, dtype=torch.int64))
    logits = torch.zeros(2, 2048)
    assert sampler.sample([], logits) == "fused" and len(calls) == 1
    # a step with its filters off, a penalty, or a vocabulary a program can't hold samples in full
    cases = (({"apply_filters": False}, 2048), ({"apply_penalty": True}, 2048), ({}, FUSED_FILTER_MAX_VOCAB + 1))
    for kwargs, vocab in cases:
        sampler.seen_tokens_buf = None
        assert torch.is_tensor(sampler.sample([], torch.zeros(2, vocab), **kwargs))
    assert len(calls) == 1
    assert torch.is_tensor(_sampler().sample([], logits)) and len(calls) == 1


@pytest.mark.parametrize("capability", [
    {"enable_repetion_penalty": True}, {"enable_min_p": True}, {"max_repetition_window": 4},
    {"min_tokens_stop_ids": (1,)}, {"enable_top_p_first": True},
])
def test_fused_draw_refuses_the_capabilities_it_does_not_cover(capability):
    kwargs = {"enable_repetion_penalty": False, **capability}
    with pytest.raises(ValueError, match="fused_draw"):
        SamplerResource(vocab_size=8, device=torch.device("cpu"), fused_draw=True, **kwargs)


def test_the_buffers_hand_their_samplers_the_draw():
    device = torch.device("cpu")
    fused = SamplerBuffers.allocate(max_batch_size=2, device=device, fused_draw=True).sampler_for(2, 0)
    assert fused.fused_draw
    assert not SamplerBuffers.allocate(max_batch_size=2, device=device).sampler_for(2, 0).fused_draw


def test_a_platform_without_the_kernel_samples_in_full(monkeypatch):
    monkeypatch.setattr(F, "available", lambda: False)
    resource = SamplerResource(vocab_size=8, device=torch.device("cpu"), enable_repetion_penalty=False, fused_draw=True)
    resource.build_accelerator_graph_buffers([], max_bs=2, max_seq_len=8)
    assert not resource._cg_buffers.fused_draw


def test_a_node_that_reads_its_tokens_samples_in_full(monkeypatch):
    monkeypatch.setattr(F, "available", lambda: True)
    monkeypatch.setattr("mstar.engine.resources.sampler.resource.device_loopback_enabled", lambda: True)
    resource = SamplerResource(vocab_size=8, device=torch.device("cpu"), enable_repetion_penalty=False, fused_draw=True)
    resource.enable_device_loopback_reader()
    resource.build_accelerator_graph_buffers([], max_bs=2, max_seq_len=8)
    assert not resource._cg_buffers.fused_draw


def test_only_the_code_predictor_samplers_opt_in():
    assert SamplerSpec(resource_key="s", nodes={"n"}, vocab_size=8).fused_draw is False
