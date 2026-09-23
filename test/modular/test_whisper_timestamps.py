"""Whisper timestamp rules against HF's ``WhisperTimeStampLogitsProcessor``.

Random logits and random generated histories (text / timestamps in every
arrangement the rules care about) go through both; the masked logits must
be identical, including the "timestamps beat every text token" rule.
"""

import random

import pytest
import torch

from mstar.model.whisper.components.timestamps import (
    AFTER_PAIR,
    AFTER_SEGMENT_END,
    FIRST_TOKEN,
    IN_TEXT,
    STATE_SIZE,
    TimestampRules,
    inactive_state,
    rule_state,
)
from mstar.model.whisper.config import WhisperModelConfig

# a small vocabulary with the real layout: text < eot < specials < <|notimestamps|> < timestamps
EOT, NOTS = 200, 210
CFG = WhisperModelConfig(
    vocab_size=NOTS + 1 + 60, eos_token_id=EOT, no_timestamps_token_id=NOTS, max_initial_timestamp_index=10,
)
TB = CFG.timestamp_begin


def _hf_processor(begin_index: int):
    logits_process = pytest.importorskip("transformers.generation.logits_process")
    from transformers import GenerationConfig

    gen = GenerationConfig(
        eos_token_id=EOT, no_timestamps_token_id=NOTS, max_initial_timestamp_index=10,
    )
    return logits_process.WhisperTimeStampLogitsProcessor(gen, begin_index=begin_index)


def _histories(rng: random.Random) -> list[list[int]]:
    text = lambda: rng.randrange(0, EOT)  # noqa: E731
    stamp = lambda lo=TB, hi=None: rng.randrange(lo, hi or CFG.vocab_size)  # noqa: E731
    cases = [
        [],
        [stamp(TB, TB + 5)],
        [stamp(TB, TB + 5), text(), text()],
        [stamp(TB, TB + 5), text(), stamp(TB + 5, TB + 20)],
        [stamp(TB, TB + 5), text(), stamp(TB + 7, TB + 8), stamp(TB + 7, TB + 8)],
        [stamp(TB, TB + 5), text(), stamp(TB + 7, TB + 8), stamp(TB + 7, TB + 8), text()],
        [text(), text()],
        [text(), stamp(TB + 30, TB + 40)],
    ]
    for _ in range(12):
        n = rng.randrange(1, 12)
        cases.append([stamp() if rng.random() < 0.4 else text() for _ in range(n)])
    return cases


def test_rule_state_phases():
    assert rule_state([], CFG) == [1, FIRST_TOKEN, TB - 1, TB + 11]
    assert rule_state([5, 6], CFG) == [1, IN_TEXT, TB - 1, CFG.vocab_size]
    assert rule_state([TB + 3], CFG) == [1, AFTER_PAIR, TB + 3, CFG.vocab_size]  # first stamp counts as a pair
    assert rule_state([TB + 3, 7, TB + 9], CFG) == [1, AFTER_SEGMENT_END, TB + 9, CFG.vocab_size]
    assert rule_state([TB + 3, 7, TB + 9, TB + 9], CFG) == [1, AFTER_PAIR, TB + 9, CFG.vocab_size]
    assert rule_state([TB + 3, 7, TB + 9, TB + 9, 8], CFG) == [1, IN_TEXT, TB + 9, CFG.vocab_size]
    assert inactive_state()[0] == 0 and len(inactive_state()) == STATE_SIZE


@pytest.mark.parametrize("seed", range(3))
def test_advance_reproduces_rule_state_token_by_token(seed):
    """Folding ``advance`` over a history from the first-token state lands on
    exactly the row ``rule_state`` builds from that history."""
    rng = random.Random(100 + seed)
    rules = TimestampRules(CFG)
    for history in _histories(rng):
        state = torch.tensor([rule_state([], CFG)])
        for i, tok in enumerate(history):
            state = rules.advance(state, torch.tensor([tok]))
            assert state[0].tolist() == rule_state(history[: i + 1], CFG), (history[: i + 1], state[0].tolist())
    # inactive rows never move; active rows advance independently in a batch
    state = torch.tensor([inactive_state(), rule_state([], CFG), rule_state([TB + 3, 7], CFG)])
    out = rules.advance(state, torch.tensor([TB + 5, TB + 2, TB + 9]))
    assert out[0].tolist() == inactive_state()
    assert out[1].tolist() == rule_state([TB + 2], CFG)
    assert out[2].tolist() == rule_state([TB + 3, 7, TB + 9], CFG)


@pytest.mark.parametrize("seed", range(4))
def test_masked_logits_match_hf(seed):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    rules = TimestampRules(CFG)
    prompt_len = 4
    for history in _histories(rng):
        logits = torch.randn(1, CFG.vocab_size) * 3
        # push probability mass onto the timestamps in some cases so the
        # logprob rule fires on both sides
        if rng.random() < 0.5:
            logits[:, TB:] += 4.0
        input_ids = torch.tensor([[50258] * prompt_len + history])
        expected = _hf_processor(prompt_len)(input_ids, logits.clone())
        state = torch.tensor([rule_state(history, CFG)])
        actual = rules.apply(logits.clone(), state)
        differing = (torch.isinf(actual) ^ torch.isinf(expected)).nonzero().flatten().tolist()
        assert not differing, (history, differing)
        finite = ~torch.isinf(expected)
        assert torch.allclose(actual[finite], expected[finite])


def test_batched_rows_are_independent_and_inactive_rows_pass_through():
    rules = TimestampRules(CFG)
    torch.manual_seed(0)
    logits = torch.randn(3, CFG.vocab_size)
    state = torch.tensor([
        inactive_state(),
        rule_state([], CFG),
        rule_state([TB + 3, 7, TB + 9], CFG),
    ])
    out = rules.apply(logits.clone(), state)
    assert torch.equal(out[0], logits[0])
    assert torch.isinf(out[1, :TB]).all() and torch.isfinite(out[1, TB]) and torch.isinf(out[1, TB + 11])
    # after a segment end: no text, and no timestamp before the one that closed it
    assert torch.isinf(out[2, :EOT]).all()
    assert torch.isinf(out[2, TB:TB + 9]).all() and torch.isfinite(out[2, TB + 9])
