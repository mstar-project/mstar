"""``Mamba2Geometry``: the recurrent pool's blocks for a Mamba-2 (SSD) layer
stack, and the inverse that a backend uses to read the geometry off a pool.
Nemotron-H (VoiceChat-11B's nano) numbers: 128 heads x 80, state 128, 8 B/C
groups, conv kernel 4 -> conv_dim 12288."""
import pytest
import torch

from mstar.engine.resources.recurrent import (
    DeltaNetGeometry,
    Mamba2Geometry,
    RecurrentStateConfig,
)

NEMOTRON_H = Mamba2Geometry(num_heads=128, head_dim=80, state_size=128, n_groups=8, conv_kernel_size=4)


def test_nemotron_h_blocks():
    blocks = NEMOTRON_H.to_blocks()
    assert NEMOTRON_H.d_inner == 10240 and NEMOTRON_H.conv_dim == 12288
    assert blocks["ssm"].shape == (128, 80, 128) and blocks["ssm"].dtype is torch.float32
    assert blocks["conv"].shape == (12288, 3) and blocks["conv"].dtype is torch.bfloat16
    # per-slot bytes: fp32 ssm (5.0 MiB/layer) + bf16 conv (72 KiB/layer)
    cfg = RecurrentStateConfig(num_layers=27, blocks=blocks, max_slots=64)
    assert cfg.slot_bytes == 27 * (128 * 80 * 128 * 4 + 12288 * 3 * 2)
    assert round(cfg.slot_bytes / 2**20) == 137


def test_blocks_round_trip():
    assert Mamba2Geometry.from_blocks(NEMOTRON_H.to_blocks()) == NEMOTRON_H
    small = Mamba2Geometry(num_heads=4, head_dim=16, state_size=8, n_groups=2, conv_kernel_size=3)
    assert Mamba2Geometry.from_blocks(small.to_blocks(state_dtype=torch.bfloat16)) == small


def test_round_trip_survives_sharding():
    """Heads and conv channels shard on their leading axis; the sharded shapes
    still describe a Mamba-2 layout (groups shard with the heads)."""
    blocks = NEMOTRON_H.to_blocks()
    for block in blocks.values():
        block.shard(8)
    sharded = Mamba2Geometry.from_blocks(blocks)
    assert sharded.num_heads == 16 and sharded.n_groups == 1
    assert sharded.head_dim == 80 and sharded.state_size == 128 and sharded.conv_kernel_size == 4


def test_state_dtype_is_the_models_call():
    blocks = NEMOTRON_H.to_blocks(state_dtype=torch.bfloat16, conv_dtype=torch.float16)
    assert blocks["ssm"].dtype is torch.bfloat16 and blocks["conv"].dtype is torch.float16


def test_rejects_other_families():
    delta = DeltaNetGeometry(num_k_heads=2, num_v_heads=4, head_k_dim=16, head_v_dim=16, conv_kernel_size=4)
    with pytest.raises(ValueError, match="no 'ssm' block"):
        Mamba2Geometry.from_blocks(delta.to_blocks())
    bad = NEMOTRON_H.to_blocks()
    bad["conv"].shape = (12000, 3)  # not H*P + 2*groups*N for any group count
    with pytest.raises(ValueError, match="does not match"):
        Mamba2Geometry.from_blocks(bad)
    with pytest.raises(ValueError, match="no 'conv' block"):
        Mamba2Geometry.from_blocks({"ssm": NEMOTRON_H.to_blocks()["ssm"]})
