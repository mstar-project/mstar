"""Unit tests for :mod:`mstar.model.zonos2.config`.

Focus on the per-layer MoE top-k resolution (``special_topk_layers`` /
``get_num_experts_per_tok`` — the routing fix on this branch), the MoE-layer
band predicate, and ``params.json`` parsing / normalization. Pure-CPU.
"""
from __future__ import annotations

import pytest

from mstar.model.zonos2.config import Zonos2Config, load_zonos2_config


# -- get_num_experts_per_tok ------------------------------------------------
def test_default_topk_when_no_special():
    cfg = Zonos2Config(num_experts_per_tok=2, special_topk_layers=None)
    assert cfg.get_num_experts_per_tok(0) == 2
    assert cfg.get_num_experts_per_tok(26) == 2


def test_special_topk_overrides_per_layer_int_key():
    cfg = Zonos2Config(num_experts_per_tok=8, special_topk_layers={26: 2})
    assert cfg.get_num_experts_per_tok(26) == 2   # overridden
    assert cfg.get_num_experts_per_tok(3) == 8    # default elsewhere


def test_special_topk_accepts_str_keys():
    # Checkpoint JSON stores keys as strings; resolution must still match.
    cfg = Zonos2Config(num_experts_per_tok=8, special_topk_layers={"26": 2})
    assert cfg.get_num_experts_per_tok(26) == 2


def test_zero_default_topk_falls_back_to_one():
    cfg = Zonos2Config(num_experts_per_tok=0, special_topk_layers=None)
    assert cfg.get_num_experts_per_tok(0) == 1


def test_invalid_special_topk_raises():
    cfg = Zonos2Config(special_topk_layers={5: 0})
    with pytest.raises(ValueError):
        cfg.get_num_experts_per_tok(5)


# -- is_moe_layer -----------------------------------------------------------
def test_is_moe_layer_band():
    cfg = Zonos2Config(
        num_layers=10, moe_n_experts=8, moe_start_from_layer=2, moe_end_from_layer=2
    )
    # MoE active on [start, num_layers - end) == [2, 8).
    assert not cfg.is_moe_layer(1)
    assert cfg.is_moe_layer(2)
    assert cfg.is_moe_layer(7)
    assert not cfg.is_moe_layer(8)


def test_is_moe_layer_false_when_single_expert():
    cfg = Zonos2Config(num_layers=10, moe_n_experts=1, moe_start_from_layer=0)
    assert not cfg.is_moe_layer(5)


# -- load_zonos2_config -----------------------------------------------------
def test_load_maps_reference_field_names():
    cfg = load_zonos2_config(
        {"dim": 256, "n_layers": 4, "head_dim": 64, "n_heads": 4, "moe_router_topk": 2}
    )
    assert cfg.hidden_size == 256
    assert cfg.num_layers == 4
    assert cfg.num_experts_per_tok == 2


def test_load_normalizes_special_topk_str_keys():
    cfg = load_zonos2_config({"dim": 64, "special_topk_layers": {"26": 2, "27": 1}})
    assert cfg.special_topk_layers == {26: 2, 27: 1}


def test_load_rejects_invalid_special_topk():
    with pytest.raises(ValueError):
        load_zonos2_config({"dim": 64, "special_topk_layers": {"5": 0}})


def test_overrides_win_over_checkpoint():
    cfg = load_zonos2_config({"dim": 64, "n_layers": 4}, num_layers=99)
    assert cfg.num_layers == 99
