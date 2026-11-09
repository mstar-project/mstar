"""A prefill plan picks each row's last token (one-token rows and a missing plan:
test_flashinfer_force_prefill)."""
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
