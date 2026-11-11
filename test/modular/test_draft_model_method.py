"""The ``draft_model`` and ``eagle3`` methods: the config's defaults and limits."""
from __future__ import annotations

import pytest

from mstar.engine.resources.draft.config import DraftConfig


def test_the_methods_default_and_bound_k():
    assert DraftConfig.parse({"method": "draft_model", "model": "m"}).num_draft_tokens == 3
    assert DraftConfig.parse({"method": "eagle3", "model": "m"}).num_draft_tokens == 3
    assert DraftConfig.parse({"method": "draft_model", "model": "m", "num_draft_tokens": 4}).num_draft_tokens == 4
    with pytest.raises(ValueError, match="at most 4 tokens"):
        DraftConfig.parse({"method": "draft_model", "model": "m", "num_draft_tokens": 7})
    with pytest.raises(ValueError, match="num_draft_tokens must be a positive integer"):
        DraftConfig.parse({"method": "dspark", "model": "m"})
    with pytest.raises(ValueError, match="set model"):
        DraftConfig.parse({"method": "draft_model"})
