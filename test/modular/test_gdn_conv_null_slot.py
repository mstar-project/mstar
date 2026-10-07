"""The conv kernels skip rows whose slot is the pool's pad index, not slot 0.

Vendored from vLLM, both kernels early-return for a row whose conv-state index
equals `null_block_id`, which defaults to 0 there because vLLM's block 0 is a
null block. Here slot 0 is the sink only while the pool has one; with the sink
off it belongs to a request, so the wrappers pass the pool's own pad index.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")


def _reference(x_row, state_row, weight):
    window = torch.cat([state_row.float(), x_row.float()[:, None]], dim=1)
    return torch.nn.functional.silu((window * weight.float()).sum(1))


@pytest.mark.parametrize("null_slot_id", [0, -1])
def test_decode_conv_honours_the_pad_index(null_slot_id):
    from mstar.utils.causal_conv1d import causal_conv1d_update

    torch.manual_seed(0)
    dim, width, slots = 64, 4, 4
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16) * 0.3
    state = torch.randn(slots, dim, width - 1, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, dim, device="cuda", dtype=torch.bfloat16)
    before = state.clone()
    out = causal_conv1d_update(
        x=x.clone(), conv_state=state, weight=weight, bias=None, activation="silu",
        conv_state_indices=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        null_block_id=null_slot_id,
    )
    # slot 1 is a live request under either convention
    assert torch.allclose(out[1].float(), _reference(x[1], before[1], weight), atol=2e-2)
    if null_slot_id == 0:
        # slot 0 is the sink: skipped, passthrough, state untouched
        assert torch.equal(out[0], x[0]) and torch.equal(state[0], before[0])
    else:
        # the sink is off, slot 0 is a real request: computed and updated
        assert torch.allclose(out[0].float(), _reference(x[0], before[0], weight), atol=2e-2)
        assert not torch.equal(state[0], before[0])
