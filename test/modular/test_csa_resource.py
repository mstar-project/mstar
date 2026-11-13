"""The CSA resource: its selection rules, and the rows its caches advance by."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.csa.config import CSAConfig
from mstar.engine.resources.csa.manager import CSAManager, candidate_blocks, top_scores
from mstar.engine.resources.step import Segment, StepContext

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


class TestTokenClock:
    """A cache of ratio r advances a row every r tokens of a request, wherever a step starts."""

    @staticmethod
    def _manager() -> CSAManager:
        config = CSAConfig(caches={1: "every", 2: "pairs", 4: "fours"}, topk=4, sm_scale=1.0, index_head_dim=8)
        return CSAManager(config, {1: 16, 2: 16, 4: 16}, torch.device("cpu"))

    @staticmethod
    def _commit(manager: CSAManager, segments) -> None:
        from mstar.engine.resources.csa.config import CSAStep

        manager.commit(CSAStep(segments=tuple(segments)), StepContext(
            request_ids=tuple(s.request_id for s in segments), graph_walk="decode", slot=0, capture=False))

    def test_rows_are_the_groups_a_steps_tokens_complete(self):
        manager = self._manager()
        spans = {key: [s.span for s in rows] for key, rows in manager.kv_segments(
            [Segment("a", "main", 7), Segment("b", "main", 1), Segment("c", "main", 0)]).items()}
        assert spans == {"every": [7, 1, 0], "pairs": [3, 0, 0], "fours": [1, 0, 0]}

    def test_a_requests_steps_add_up_to_its_tokens_over_the_ratio(self):
        manager = self._manager()
        total = {"every": 0, "pairs": 0, "fours": 0}
        tokens = 0
        for n in (3, 1, 1, 6, 1, 1, 1, 9):
            step = [Segment("a", "main", n)]
            for key, rows in manager.kv_segments(step).items():
                total[key] += rows[0].span
            self._commit(manager, step)
            tokens += n
            assert total == {"every": tokens, "pairs": tokens // 2, "fours": tokens // 4}

    def test_a_declared_step_moves_nothing_until_it_commits(self):
        manager = self._manager()
        step = [Segment("a", "main", 3)]
        first = manager.kv_segments(step)
        assert manager.kv_segments(step) == first  # declared again, as a retried admit does
        self._commit(manager, step)
        assert manager.kv_segments([Segment("a", "main", 1)])["pairs"][0].span == 1

    def test_a_removed_request_starts_over(self):
        manager = self._manager()
        self._commit(manager, [Segment("a", "main", 3)])
        manager.remove_request("a")
        assert manager.kv_segments([Segment("a", "main", 1)])["pairs"][0].span == 0

    @pytest.mark.parametrize("caches", [{}, {0: "x"}, {1: "x", 2: "x"}])
    def test_a_config_maps_each_ratio_to_its_own_cache(self, caches):
        with pytest.raises(ValueError, match="caches"):
            CSAConfig(caches=caches, topk=4, sm_scale=1.0, index_head_dim=8)
