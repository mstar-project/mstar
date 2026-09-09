"""The TP fast read path's GLM-5.2 half: the read plan and the shape-driven
expert loaders.

The generic loader has every rank read the full checkpoint and keep its
slice (8x the bytes at TP8). ``build_glm52_read_plan`` excludes never-loaded
keys and hands the iterator ``(dim, start, stop)`` specs so each rank reads
only its shard of the routed-expert tensors; the expert loaders accept the
pre-sliced shards by shape. (The iterator's sliced read and the on-disk
fast-path-vs-generic parity live with the loader's own tests.)
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.model.glm52.components.moe import (  # noqa: E402
    _down_fp8_loader,
    _gate_up_fp8_loader,
)
from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.weight_loader import build_glm52_read_plan  # noqa: E402

BLOCK = (16, 16)


def test_read_plan_excludes_and_slices():
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)  # moe_inter 64, 2 layers
    keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.indexer.wk.weight",       # layer 0 FULL
        "model.layers.1.self_attn.indexer.wk.weight",       # layer 1 SHARED
        "model.layers.1.mlp.experts.2.gate_proj.weight",
        "model.layers.1.mlp.experts.2.gate_proj.weight_scale_inv",
        "model.layers.1.mlp.experts.2.down_proj.weight",
        "model.layers.1.mlp.experts.2.down_proj.weight_scale_inv",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.2.enorm.weight",                      # MTP layer
        "model.layers.2.mlp.experts.0.up_proj.weight",      # MTP expert
    ]
    plan_keys, specs = build_glm52_read_plan(keys, cfg, tp_rank=1, tp_size=2)

    assert "model.layers.2.enorm.weight" not in plan_keys
    assert "model.layers.2.mlp.experts.0.up_proj.weight" not in plan_keys
    assert "model.layers.1.self_attn.indexer.wk.weight" not in plan_keys  # SHARED
    assert "model.layers.0.self_attn.indexer.wk.weight" in plan_keys      # FULL
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in plan_keys
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" not in specs
    assert "model.embed_tokens.weight" in plan_keys

    shard = cfg.moe_intermediate_size // 2  # 32
    assert specs["model.layers.1.mlp.experts.2.gate_proj.weight"] == (0, 32, 64)
    srows = shard // BLOCK[0]  # 2
    assert specs["model.layers.1.mlp.experts.2.gate_proj.weight_scale_inv"] == (
        0, srows, 2 * srows)
    assert specs["model.layers.1.mlp.experts.2.down_proj.weight"] == (1, 32, 64)
    scols = shard // BLOCK[1]
    assert specs["model.layers.1.mlp.experts.2.down_proj.weight_scale_inv"] == (
        1, scols, 2 * scols)


def test_read_plan_flag_off_indexer_and_bf16_experts():
    # load_indexer=False drops every indexer key, FULL layers included; a
    # bf16 (non-fp8-resident) config reads experts whole — no slice specs
    cfg = Glm52ModelConfig.reduced()
    keys = [
        "model.layers.0.self_attn.indexer.wk.weight",
        "model.layers.1.mlp.experts.2.gate_proj.weight",
    ]
    plan_keys, specs = build_glm52_read_plan(keys, cfg, tp_rank=0, tp_size=2, load_indexer=False)
    assert plan_keys == {"model.layers.1.mlp.experts.2.gate_proj.weight"}
    assert specs == {}


def test_expert_loaders_accept_full_and_presliced():
    full_inter, tp = 64, 2
    shard = full_inter // tp
    hidden = 32
    param = torch.nn.Parameter(
        torch.zeros(4, 2 * shard, hidden, dtype=torch.uint8), requires_grad=False)
    full = torch.arange(full_inter * hidden, dtype=torch.uint8).view(full_inter, hidden)

    _gate_up_fp8_loader(1, tp, full_inter, 1, param, full, "gate:3")
    from_full = param.data[3, :shard].clone()
    param.data.zero_()
    _gate_up_fp8_loader(1, tp, full_inter, 1, param, full[shard:], "gate:3")
    assert torch.equal(param.data[3, :shard], from_full)
    assert torch.equal(from_full, full[shard:])

    with pytest.raises(ValueError):
        _gate_up_fp8_loader(1, tp, full_inter, 1, param, full[:10], "gate:3")

    dparam = torch.nn.Parameter(
        torch.zeros(4, hidden, shard, dtype=torch.uint8), requires_grad=False)
    dfull = torch.arange(hidden * full_inter, dtype=torch.uint8).view(hidden, full_inter)
    _down_fp8_loader(1, tp, full_inter, 1, dparam, dfull, "down:2")
    d_from_full = dparam.data[2].clone()
    dparam.data.zero_()
    _down_fp8_loader(1, tp, full_inter, 1, dparam, dfull[:, shard:], "down:2")
    assert torch.equal(dparam.data[2], d_from_full)
    with pytest.raises(ValueError):
        _down_fp8_loader(1, tp, full_inter, 1, dparam, dfull[:, :10], "down:2")


def _mtp_fp8_config():
    """reduced_fp8 with the MTP position landing FULL: 4 trunk layers put
    layer 4 on the IndexShare grid (offset-1 + freq, the same geometry as
    the real 78 = 2 + 19·4), and drafting on builds the mtp submodule."""
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)
    cfg.num_hidden_layers = 4
    cfg.mtp_num_draft_tokens = 2
    return cfg


def test_read_plan_includes_mtp_when_enabled():
    """The 2026-08-09 0.00-acceptance bug: the plan dropped every layer-78
    key regardless of drafting (1569/118629 keys on the real checkpoint),
    so the MTP module served ``to_empty`` memory. ``load_mtp=True`` must
    read the MTP layer like trunk: glue + decoder + FULL indexer, with
    its routed experts sliced by the same specs."""
    cfg = _mtp_fp8_config()
    keys = [
        "model.layers.3.self_attn.q_a_proj.weight",
        "model.layers.4.enorm.weight",
        "model.layers.4.eh_proj.weight",
        "model.layers.4.self_attn.indexer.wk.weight",   # layer 4 FULL
        "model.layers.4.mlp.experts.0.up_proj.weight",
        "model.layers.4.mlp.experts.0.up_proj.weight_scale_inv",
    ]
    # Default stays M1 flag-off behavior: the MTP layer is never read.
    plan_keys, _ = build_glm52_read_plan(keys, cfg, tp_rank=1, tp_size=2)
    assert "model.layers.3.self_attn.q_a_proj.weight" in plan_keys
    assert not any(".layers.4." in k for k in plan_keys)

    plan_keys, specs = build_glm52_read_plan(
        keys, cfg, tp_rank=1, tp_size=2, load_mtp=True)
    assert "model.layers.4.enorm.weight" in plan_keys
    assert "model.layers.4.eh_proj.weight" in plan_keys
    assert "model.layers.4.self_attn.indexer.wk.weight" in plan_keys
    shard = cfg.moe_intermediate_size // 2
    assert specs["model.layers.4.mlp.experts.0.up_proj.weight"] == (
        0, shard, 2 * shard)
    srows = shard // BLOCK[0]
    assert specs["model.layers.4.mlp.experts.0.up_proj.weight_scale_inv"] == (
        0, srows, 2 * srows)
    assert "model.layers.4.enorm.weight" not in specs


def test_read_plan_refuses_shards_that_split_a_scale_block():
    # per-rank intermediate must be a whole number of scale blocks, or the
    # sliced fp8 bytes and sliced scales would misalign
    cfg = Glm52ModelConfig.reduced_fp8(block=BLOCK)  # moe_inter 64
    with pytest.raises(AssertionError, match="scale block"):
        build_glm52_read_plan(["model.embed_tokens.weight"], cfg, tp_rank=0, tp_size=8)
