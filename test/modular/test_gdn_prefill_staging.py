"""A GDN prefill plan's index buffers, written through pinned staging with the
padding filled on the host: the conv chunk map and its chunk offsets cover
the whole buffer (stale chunks of a wider earlier plan point at the pad slot),
cu_seqlens is the running sum of the spans, and the mask marks exactly the
real tokens. On CPU the stager copies directly; with an accelerator this also
runs the pinned, non-blocking path."""
from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.linear_attn.wrappers import _CONV_BLOCK_M, GDNPrefillWrapper

PAD = 7
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _plan(wrapper, spans, device):
    n = len(spans)
    wrapper.plan(
        spans, slots=torch.zeros(n, dtype=torch.int32, device=device),
        has_state=torch.zeros(n, dtype=torch.bool, device=device),
    )
    if device != "cpu":
        torch.cuda.synchronize()
    return wrapper._plan_state


def _expected_chunks(spans):
    rows, offsets = [], []
    for row, span in enumerate(spans):
        chunks = -(-(span or 1) // _CONV_BLOCK_M)
        rows += [row] * chunks
        offsets += list(range(chunks))
    return rows, offsets


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("captured", [True, False])
def test_plan_buffers_after_a_wider_plan(device, captured):
    wrapper = GDNPrefillWrapper(
        device=torch.device(device), pad_slot_id=PAD, num_tokens=256, bs=4,
        accelerator_graph=captured,
    )
    for spans in ([100, 60, 30, 9], [17, 3, 0]):
        state = _plan(wrapper, spans, device)
        rows, offsets = _expected_chunks(spans)
        batch = wrapper._batch_ptr.cpu().tolist()
        chunk_offsets = wrapper._offset_ptr.cpu().tolist()
        assert batch[:len(rows)] == rows and set(batch[len(rows):]) <= {PAD}
        assert chunk_offsets[:len(offsets)] == offsets and set(chunk_offsets[len(offsets):]) <= {PAD}
        cu = [0]
        for span in spans:
            cu.append(cu[-1] + (span or 1))
        assert state.cu_seqlens.cpu().tolist() == cu
        mask = state.token_mask.cpu().tolist()
        real = sum(spans)
        assert mask[:real] == [1.0] * real and set(mask[real:]) <= {0.0}
