"""A drafter's vocabulary against its target's (`TokenMap`): an EAGLE-3 d2t maps each draft id to a target id."""
from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.draft.token_map import TokenMap


def test_an_eagle3_d2t_offsets_each_draft_id():
    d2t = torch.tensor([3, 3, 4, 6])  # draft d -> target d + d2t[d]
    tmap = TokenMap.from_offsets(d2t, target_vocab=12)
    assert tmap.to_target(torch.arange(4)).tolist() == [3, 4, 6, 9]
    assert torch.equal(tmap.to_target(torch.tensor([[3, 0], [1, 1]])), torch.tensor([[9, 3], [4, 4]]))


def test_a_map_naming_a_target_id_twice_or_outside_the_vocabulary_is_refused():
    with pytest.raises(ValueError, match="two draft ids map to one target id"):
        TokenMap.from_offsets(torch.tensor([3, 2]), target_vocab=5)
    with pytest.raises(ValueError, match="maps outside the target"):
        TokenMap.from_offsets(torch.tensor([0, 9]), target_vocab=5)
