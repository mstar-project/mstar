"""Thinker TP/EP selection from the yaml ``model_kwargs.thinker_moe`` section.

Uses a shrunken Thinker config on CPU; no checkpoint or GPU needed.
"""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components import ExpertParallelSparseMoeBlock, ParallelSparseMoeBlock
from mstar.model.qwen3_omni.components.thinker import (
    Qwen3OmniThinkerLayer,
    Qwen3OmniThinkerModel,
    ThinkerMoeParallelConfig,
)
from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig

NUM_EXPERTS = 8
_LOGGER = "mstar.model.qwen3_omni.components.thinker"


def _config(num_layers=2):
    cfg = Qwen3OmniModelConfig()
    tc = cfg.thinker_text
    tc.vocab_size, tc.hidden_size, tc.num_hidden_layers = 32, 64, num_layers
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 2, 1, 32
    tc.num_experts, tc.num_experts_per_tok, tc.moe_intermediate_size = NUM_EXPERTS, 2, 16
    return cfg


def _group(rank, world_size):
    return CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(world_size)))


def test_from_yaml_defaults_to_tp():
    assert ThinkerMoeParallelConfig.from_yaml(None) == ThinkerMoeParallelConfig()
    assert ThinkerMoeParallelConfig.from_yaml({}).parallel == "tp"
    assert not ThinkerMoeParallelConfig.from_yaml({}).mask_padding


@pytest.mark.parametrize("parallel", ["tp", "ep"])
def test_mask_padding_reaches_model(parallel):
    moe = ThinkerMoeParallelConfig.from_yaml({"parallel": parallel, "mask_padding": True})
    with torch.device("meta"):
        model = Qwen3OmniThinkerModel(_config(), comm_group=_group(0, 2), moe_parallel=moe)
    assert model.mask_padding
    assert all(layer.is_moe for layer in model.model.layers)


@pytest.mark.parametrize("section, match", [
    ({"parallel": "dp"}, "parallel"),
    ({"parallel": "ep", "ep_group": "sp"}, "ep_group"),
    ({"parallel": "ep", "log_expert_load_every": -1}, ">= 0"),
    ({"debug_check_routing": True}, "need parallel: ep"),
    ({"log_expert_load_every": 10}, "need parallel: ep"),
    ({"parallel": "ep", "typo": 1}, "unknown"),
])
def test_from_yaml_rejects(section, match):
    with pytest.raises(ValueError, match=match):
        ThinkerMoeParallelConfig.from_yaml(section)


@pytest.mark.parametrize("parallel, cls", [("tp", ParallelSparseMoeBlock), ("ep", ExpertParallelSparseMoeBlock)])
def test_layer_picks_moe_block(parallel, cls):
    moe = ThinkerMoeParallelConfig.from_yaml({"parallel": parallel})
    with torch.device("meta"):
        layer = Qwen3OmniThinkerLayer(_config(), 0, comm_group=_group(1, 2), moe_parallel=moe)
    assert type(layer.mlp) is cls
    if parallel == "ep":
        # Rank 1 of 2 owns experts 4..7 at full width.
        assert layer.mlp.expert_start == NUM_EXPERTS // 2
        assert layer.mlp.experts.down_proj.shape == (NUM_EXPERTS // 2, 64, 16)
        assert not layer.mlp.debug_check_routing


@pytest.mark.parametrize("rank, tracked", [(0, True), (1, False)])
def test_only_group_rank_zero_tracks_load(rank, tracked):
    moe = ThinkerMoeParallelConfig(parallel="ep", log_expert_load_every=5, debug_check_routing=True)
    with torch.device("meta"):
        layer = Qwen3OmniThinkerLayer(_config(), 0, comm_group=_group(rank, 2), moe_parallel=moe)
    assert layer.mlp.track_expert_load is tracked
    assert layer.mlp.debug_check_routing


def _load_model(every):
    moe = ThinkerMoeParallelConfig(parallel="ep", log_expert_load_every=every)
    with torch.device("meta"):
        model = Qwen3OmniThinkerModel(_config(num_layers=2), comm_group=_group(0, 2), moe_parallel=moe)
    model.to_empty(device="cpu")
    return model, [layer.mlp for layer in model.model.layers]


def _fake_forward(blocks):
    # Layer 0: all slots on rank 0's experts. Layer 1: evenly split.
    blocks[0].expert_load += torch.tensor([2, 0, 0, 0, 0, 0, 0, 0])
    blocks[1].expert_load += torch.tensor([1, 0, 0, 0, 1, 0, 0, 0])


def test_load_log_every_n_steps(caplog):
    model, blocks = _load_model(every=3)

    def logged():
        return [r for r in caplog.records if r.name == _LOGGER]

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        # The hook runs before each forward, so step 4's call logs steps 1..3.
        for _ in range(3):
            model.maybe_log_expert_load()
            _fake_forward(blocks)
        assert not logged()
        model.maybe_log_expert_load()
    (record,) = logged()
    msg = record.getMessage()
    assert "over 3 steps, 2 MoE layers" in msg
    assert "per-rank slots [9, 3]" in msg
    assert "worst layer 0 [6, 0] (max/mean 2.00)" in msg
    # Logging resets the counters.
    assert all((b.expert_load == 0).all() for b in blocks)


def test_load_log_discards_warmup(caplog):
    model, blocks = _load_model(every=2)
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        for _ in range(5):
            model.maybe_log_expert_load(synthetic=True)
            blocks[0].expert_load += 1000
        for _ in range(2):
            model.maybe_log_expert_load()
            _fake_forward(blocks)
        model.maybe_log_expert_load()
    (record,) = [r for r in caplog.records if r.name == _LOGGER]
    assert "over 2 steps" in record.getMessage()
    assert "per-rank slots [6, 2]" in record.getMessage()
