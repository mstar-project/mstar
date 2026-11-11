"""The pieces a drafter that drafts one token a pass is built from (`mstar.model.components.chained_draft`) and
`AttentionManager.prompt_last_rows`, which places a prompt step's per-row values: each against a plain
reference, on the CPU."""
from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.model.components.chained_draft import at_last_rows, chain_drafts, kept_rows

HIDDEN, VOCAB, ROWS = 4, 13, 3


class _Head:
    """A stand-in drafter head over a flattened ``[rows, k + 1]`` block: a kept position's output reads every
    position of its row up to it (as causal attention does) and records what it was given."""

    def __init__(self, k: int):
        gen = torch.Generator().manual_seed(k)
        self.width = k + 1
        self.embed = torch.randn(VOCAB, HIDDEN, generator=gen, dtype=torch.float64)
        self.unembed = torch.randn(HIDDEN, VOCAB, generator=gen, dtype=torch.float64)
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def out(self, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """One row's output from its positions so far."""
        return torch.tanh(hidden.sum(0) + self.embed[ids].sum(0))

    def to_draft(self, out: torch.Tensor) -> torch.Tensor:
        return (out @ self.unembed).argmax(-1)

    def step(self, hidden: torch.Tensor, ids: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        self.calls.append((hidden.clone(), ids.clone(), keep.clone()))
        starts = keep - keep % self.width
        spans = zip(starts.tolist(), keep.tolist(), strict=True)
        return torch.stack([self.out(hidden[s: p + 1], ids[s: p + 1]) for s, p in spans])


def _reference(head: _Head, carry: torch.Tensor, k: int):
    """Each row on its own: draft from the carry, then ``k - 1`` passes, each over the outputs and drafts so far."""
    drafts, prefixes = [], []
    for row in carry:
        hidden, ids = [row], [head.to_draft(row)]
        for _ in range(1, k):
            prefixes.append((torch.stack(hidden), torch.stack(ids)))
            out = head.out(torch.stack(hidden), torch.stack(ids))
            hidden.append(out)
            ids.append(head.to_draft(out))
        drafts.append(torch.stack(ids))
    return torch.stack(drafts), prefixes


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_the_chain_drafts_what_a_sequential_loop_does_each_pass_reading_the_last_passs_output_and_draft(k):
    head = _Head(k)
    carry = torch.randn(ROWS, HIDDEN, generator=torch.Generator().manual_seed(100 + k), dtype=torch.float64)
    want, prefixes = _reference(head, carry, k)
    if k > 2:
        # the inputs vary from pass to pass, so a pass that reads a stale one drafts differently
        assert len(set(want.reshape(-1).tolist())) > 1

    got = chain_drafts(carry, k, head.step, head.to_draft)

    assert got.shape == (ROWS, k) and torch.equal(got, want)
    assert len(head.calls) == k - 1
    for m, (hidden, ids, keep) in enumerate(head.calls, start=1):
        assert hidden.shape == (ROWS * (k + 1), HIDDEN) and ids.shape == (ROWS * (k + 1),)
        # pass m keeps each row's position m - 1, which reads the m outputs and drafts before it
        assert keep.tolist() == [r * (k + 1) + m - 1 for r in range(ROWS)], m
        for r in range(ROWS):
            ref_hidden, ref_ids = prefixes[r * (k - 1) + m - 1]
            at = slice(r * (k + 1), r * (k + 1) + m)
            assert torch.equal(hidden[at], ref_hidden) and torch.equal(ids[at], ref_ids), (m, r)


def test_kept_rows_is_each_rows_last_kept_position_for_every_accepted_count():
    k = 3
    block = torch.arange(ROWS * (k + 1)).view(ROWS, k + 1) * 10
    for accepted in range(k + 1):
        # a different count on each row, so every row sees every count
        counts = (torch.arange(ROWS) + accepted) % (k + 1)
        at = kept_rows(block, counts)
        assert torch.equal(block.reshape(-1)[at], block[torch.arange(ROWS), counts]), accepted


def test_each_prompts_value_lands_at_its_last_row_and_a_padding_rows_goes_to_the_sink():
    values = torch.tensor([11, 22, 99])
    # two prompts ending at rows 2 and 6 of a 7-token step, and a padding row, whose `at` is the sink
    got = at_last_rows(values, torch.tensor([2, 6, 7]), 7)
    assert got.tolist() == [0, 0, 11, 0, 0, 0, 22]


class _Attn:
    def __init__(self, qo):
        self.qo = qo

    def qo_indptr_buf(self, label="main"):
        return self.qo

    # the real one, over this plan's offsets
    prompt_last_rows = AttentionManager.prompt_last_rows


def test_a_padded_prompt_steps_padding_rows_leave_the_real_rows_values():
    """A captured packed step pads with rows of no tokens, whose last index is the last real row's: their
    per-row values go to the sink, not over it."""
    last, at = _Attn(torch.tensor([0, 3, 7, 7, 7])).prompt_last_rows(7, torch.device("cpu"))
    assert last.tolist() == [2, 6, 6, 6] and at.tolist() == [2, 6, 7, 7]
    assert at_last_rows(torch.tensor([11, 22, 99, 98]), at, 7).tolist() == [0, 0, 11, 0, 0, 0, 22]


@pytest.mark.parametrize("qo", [torch.tensor([0, 1, 2, 3]), None], ids=["planned", "no_offsets"])
def test_a_prompt_step_of_one_token_rows_has_every_row_its_own_last(qo):
    """A step whose rows are one token each, planned with offsets or (an eager step) without."""
    last, at = _Attn(qo).prompt_last_rows(3, torch.device("cpu"))
    assert last.tolist() == [0, 1, 2] and at.tolist() == [0, 1, 2]
    assert at_last_rows(torch.tensor([11, 22, 33]), at, 3).tolist() == [11, 22, 33]
