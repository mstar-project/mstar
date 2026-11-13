"""The CSA resource's selection rules."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import torch

from mstar.engine.resources.csa.manager import candidate_blocks, top_scores

NEG = float("-inf")


class TestSelection:
    def test_equal_scores_go_to_the_earlier_group_whatever_the_width(self):
        scores = torch.tensor([[0.0, 2.0, 0.0, 0.0, 1.0, 0.0]])
        wide = torch.cat([scores, torch.full((1, 7), NEG)], dim=1)
        for s in (scores, wide):
            values, indices = top_scores(s, 4)
            assert indices.tolist() == [[1, 4, 0, 2]] and values.tolist() == [[2.0, 1.0, 0.0, 0.0]]

    def test_candidates_keep_the_best_blocks_and_the_newest(self):
        # 10 visible groups in blocks of 4: [0-3] best 5, [4-7] best 9, [8-9] the newest
        scores = torch.tensor([[5.0, 0, 0, 0, 1, 9, 1, 1, -3, -4, NEG, NEG]])
        keep = candidate_blocks(scores, torch.tensor([10]), topk_blocks=2, block_size=4)
        # the newest block is kept though it scores lowest, then the best of the others
        assert keep.tolist() == [[False] * 4 + [True] * 8]

    def test_a_row_with_fewer_blocks_than_topk_keeps_only_what_it_sees(self):
        scores = torch.tensor([[1.0, 2.0, 3.0, NEG, NEG, NEG, NEG, NEG]])
        keep = candidate_blocks(scores, torch.tensor([3]), topk_blocks=2, block_size=4)
        assert keep.tolist() == [[True] * 4 + [False] * 4]

    def test_a_row_that_sees_nothing_keeps_nothing(self):
        keep = candidate_blocks(torch.full((1, 8), NEG), torch.tensor([0]), topk_blocks=2, block_size=4)
        assert not keep.any()
