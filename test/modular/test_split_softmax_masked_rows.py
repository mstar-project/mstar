"""The split-vocab sampler prep on rows with -inf logits.

The online softmax keeps a running max that starts at -inf. A chunk whose
leading block is entirely -inf (a masked-out vocabulary region) used to leave
`-inf - -inf` = NaN in its partial sum, which the combine step turned into a
1e30 scale on the whole row. The fused kernel is the reference.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels")


def _run(logits, temperature, force_fused: bool):
    from mstar.engine.resources.sampler import utils as U

    orig = U._split_count
    if force_fused:
        U._split_count = lambda b, v, d: 1
    try:
        return U.fused_temperature_softmax(
            logits.clone(), temperature, include_greedy=True
        ).float()
    finally:
        U._split_count = orig


@pytest.mark.parametrize("masked", [
    (0, 16384),        # a whole leading chunk
    (0, 8192),         # the first block of the first chunk
    (16384, 32768),    # a whole interior chunk
    (100, 200),        # a few tokens mid-block
])
def test_split_matches_fused_and_torch_with_masked_regions(masked):
    torch.manual_seed(0)
    B, V = 3, 248320
    logits = torch.randn(B, V, device="cuda", dtype=torch.bfloat16)
    logits[:, masked[0]:masked[1]] = float("-inf")
    temperature = torch.tensor([1.0, 0.7, 0.0], device="cuda")  # last row greedy
    split = _run(logits, temperature, force_fused=False)
    fused = _run(logits, temperature, force_fused=True)
    ref = torch.softmax(logits[:2].float() / temperature[:2, None], dim=-1)
    assert torch.isfinite(split).all()
    assert torch.allclose(split.sum(-1), torch.ones(B, device="cuda"), atol=1e-3)
    assert torch.allclose(split[:2], ref, atol=1e-6, rtol=1e-3)
    assert torch.allclose(split, fused, atol=1e-6, rtol=1e-3)
    greedy = split[2]
    assert greedy.argmax().item() == logits[2].float().argmax().item()
    assert greedy.max().item() == 1.0


@pytest.mark.parametrize("split", [True, False])
def test_seen_rows_read_the_mask_rows_they_name(split):
    """With ``seen_rows``, logits row i is penalised by mask row ``seen_rows[i]``,
    as if the masks had been gathered into a dense batch first."""
    from mstar.engine.resources.sampler import utils as U

    torch.manual_seed(0)
    B, V, slots = 4, 248320, 7
    logits = torch.randn(B, V, device="cuda", dtype=torch.bfloat16)
    temperature = torch.tensor([1.0, 0.7, 1.3, 0.0], device="cuda")
    penalty = torch.tensor([2.0, 1.5, 1.0, 3.0], device="cuda")
    masks = torch.rand(slots, V, device="cuda") < 0.01
    at = torch.tensor([6, 0, 3, 3], device="cuda", dtype=torch.int32)
    orig = U._split_count
    if not split:
        U._split_count = lambda b, v, d: 1
    try:
        by_slot = U.fused_temperature_softmax(
            logits, temperature, penalty=penalty, seen_mask=masks, include_greedy=True, seen_rows=at,
        )
        dense = U.fused_temperature_softmax(
            logits, temperature, penalty=penalty, seen_mask=masks[at.long()].contiguous(), include_greedy=True,
        )
    finally:
        U._split_count = orig
    assert torch.equal(by_slot, dense)
