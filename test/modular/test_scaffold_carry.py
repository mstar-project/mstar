"""The decoder scaffold's prompt step stores the first carry its drafter's ``prefill`` returns, and refuses a
carrying drafter whose ``prefill`` returns none. Run on the CPU Qwen3 harness with DSpark made to declare a carry:
DSpark keeps no state across steps itself, so its ``needs`` and ``prefill`` are patched to give the scaffold one."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qwen3_cpu  # noqa: E402

from mstar.engine.resources.draft.drafter import DRAFT_CARRY, CarrySpec  # noqa: E402
from mstar.model.components.dspark.drafter import DSpark, DSparkDrafter  # noqa: E402

WIDTH = 4
PROMPTS = {1: [5, 9, 2, 7, 11, 3], 2: [17, 4, 8]}


def _carrying(monkeypatch, carry_of):
    """A Qwen3 + DSpark engine whose drafter declares a carry, its ``prefill`` returning ``carry_of(rows)``."""
    qwen3_cpu.patch_cpu(monkeypatch)
    needs = DSpark.needs.fget
    monkeypatch.setattr(DSpark, "needs", property(lambda self: needs(self)._replace(
        carry=CarrySpec(WIDTH, torch.float32))))
    prefill = DSparkDrafter.prefill

    def returning(self, inputs):
        prefill(self, inputs)
        return carry_of(inputs.sampled.reshape(-1).shape[0])

    monkeypatch.setattr(DSparkDrafter, "prefill", returning)


def _prompt_step(tmp_path):
    target, dspark = qwen3_cpu.write_checkpoints(tmp_path)
    model = qwen3_cpu.make_model(target, qwen3_cpu.draft_block(dspark, 3))
    eng = qwen3_cpu.engine(model)
    driver = qwen3_cpu.Driver(eng, model)
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, 8)
    driver.prefill(list(PROMPTS), PROMPTS)
    return eng


def test_the_scaffold_stores_the_first_carry_its_drafters_prefill_returns(tmp_path, monkeypatch):
    def known(rows):
        return torch.arange(rows * WIDTH, dtype=torch.float32).view(rows, WIDTH) + 1

    _carrying(monkeypatch, known)
    eng = _prompt_step(tmp_path)
    stored = eng._resources["sampler"].loopback_state(DRAFT_CARRY, list(PROMPTS))
    assert torch.equal(stored[: len(PROMPTS)].float(), known(len(PROMPTS)))


def test_a_carrying_drafter_whose_prefill_returns_none_is_refused(tmp_path, monkeypatch):
    _carrying(monkeypatch, lambda rows: None)
    with pytest.raises(RuntimeError, match="returned none from prefill"):
        _prompt_step(tmp_path)
