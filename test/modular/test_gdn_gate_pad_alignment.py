"""The fused delta-net projection's `b` block must start on a 32-byte line.

FlashInfer's bf16 GDN decode kernel checks q/k/v/a/b for 32-byte alignment on
every call. `a` always lands on one (whole heads before it); `b` sits one
v-head count later, so the pad between them has to make up the difference,
per rank, since tensor parallelism divides the head count.
"""

from __future__ import annotations

import pytest
import torch

from mstar.model.components.linear_attn import GatedDeltaNet, gate_pad

BF16 = 2  # bytes


@pytest.mark.parametrize("num_v_heads", [16, 24, 32, 48, 64])
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_b_block_starts_on_a_32_byte_line(num_v_heads, tp):
    local = num_v_heads // tp
    if local * tp != num_v_heads:
        pytest.skip("head count does not divide")
    mixer = GatedDeltaNet(
        hidden_size=32, num_k_heads=max(local // 2, 1), num_v_heads=local,
        head_k_dim=128, head_v_dim=128, conv_kernel_size=4,
    )
    blocks = mixer.in_proj_blocks
    b_offset = sum(blocks[:6]) * BF16
    a_offset = sum(blocks[:4]) * BF16
    assert a_offset % 32 == 0
    assert b_offset % 32 == 0, f"b at {b_offset} bytes for {local} local heads"
    assert blocks[5] == gate_pad(local)
    # the pad is never loaded, so it must not carry `torch.empty` garbage
    pad = mixer.in_proj_fused.weight[sum(blocks[:5]):sum(blocks[:6])]
    assert torch.equal(pad, torch.zeros_like(pad))
