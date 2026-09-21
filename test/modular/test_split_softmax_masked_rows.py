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
