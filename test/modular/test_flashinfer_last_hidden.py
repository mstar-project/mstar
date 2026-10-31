"""A prefill step planned as a decode (one token per row) still picks each row's last token."""
from types import SimpleNamespace

import torch

from mstar.engine.resources.attn.flashinfer import FlashInferManager


def _manager(plan_state):
    manager = object.__new__(FlashInferManager)
    manager._current_plan_states = {"main": plan_state}
    return manager


def test_a_prefill_plan_selects_each_rows_last_token():
    manager = _manager(SimpleNamespace(_qo_indptr_buf=torch.tensor([0, 3, 4])))
    hidden = torch.arange(4.0).unsqueeze(1)

    assert manager.select_last_hidden(hidden).flatten().tolist() == [2.0, 3.0]


def test_a_decode_plan_keeps_every_token():
    manager = _manager(SimpleNamespace())
    hidden = torch.arange(3.0).unsqueeze(1)

    assert manager.select_last_hidden(hidden).flatten().tolist() == [0.0, 1.0, 2.0]


def test_a_label_with_no_plan_fails_rather_than_keep_every_token():
    import pytest

    manager = _manager(SimpleNamespace())
    with pytest.raises(RuntimeError, match="no plan for label 'other'"):
        manager.select_last_hidden(torch.arange(3.0).unsqueeze(1), label="other")
