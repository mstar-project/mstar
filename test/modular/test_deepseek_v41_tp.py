"""DeepSeek-V4.1 at TP2 on two gloo ranks through the real ``Engine``, against one rank.

Each rank loads its slices of one random checkpoint (the loader's sharding of every linear, the
experts, the embedding, the head, the sinks and the Engram tables) and runs the collectives a
single rank never does: the row-parallel all-reduces, the index scores summed over the ranks'
heads, the vocabulary all-gather. Both ranks must see the same logits, and those must be the
single-rank ones up to fp32 summation order.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch
import torch.multiprocessing as mp

from mstar.model.deepseek_v41 import _testing
from mstar.model.deepseek_v41._testing import cpu_engine, logit_gap, random_ids, run_request
from mstar.model.deepseek_v41.config import DeepseekV41Config

PROMPT, STEPS = 37, 8
# a rank's slices must hold whole 32-wide scale blocks of the expert width and of the output
# projection's input
TP2 = dict(moe_intermediate_size=64, o_lora_rank=32)
# fp32 on both sides: an all-reduce only reorders sums (measured 1e-6 of the logits' spread)
TOL = 2e-5


def _rank(rank: int, snapshot: str, store: str, out_dir: str, model_kwargs: dict) -> None:
    torch.set_num_threads(1)
    group, groups = _testing.gloo_tp_rank(rank, 2, store)
    try:
        config = DeepseekV41Config.reduced(**TP2)
        engine, sampler, _ = _testing.build_engine(
            snapshot, torch.float32, tp_group=group, parallel_groups=groups, **model_kwargs)
        ids = random_ids(config, PROMPT + STEPS, seed=11)
        torch.save(run_request(engine, sampler, ids, PROMPT), f"{out_dir}/{rank}.pt")
    finally:
        _testing.close_gloo_tp_rank(group)


@pytest.fixture
def cpu(monkeypatch):
    _testing.patch_cpu_engine(monkeypatch)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("model_kwargs", [{}, {"moe_fp8_resident": True}], ids=["dense-experts", "fp8-experts"])
def test_two_ranks_match_one(tmp_path, cpu, model_kwargs):
    snapshot = tmp_path / "snap"
    config, _ = _testing.write_snapshot(snapshot, DeepseekV41Config.reduced(**TP2), seed=5)
    mp.spawn(_rank, args=(str(snapshot), str(tmp_path / "store"), str(tmp_path), model_kwargs), nprocs=2)
    r0, r1 = (torch.load(tmp_path / f"{rank}.pt") for rank in (0, 1))
    assert torch.equal(r0, r1)

    engine, sampler, _ = cpu_engine(snapshot, torch.float32, **model_kwargs)
    want = run_request(engine, sampler, random_ids(config, PROMPT + STEPS, seed=11), PROMPT)
    assert torch.equal(r0.argmax(-1), want.argmax(-1))
    assert logit_gap(r0, want) < TOL, logit_gap(r0, want)
