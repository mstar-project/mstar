"""moe_align_block_size on an empty step, and the CUDA op's placement: up to 4096 slots every
expert's slots in increasing slot order, as the torch fallback places them, the same on every
call; past that the same slots per expert, in vLLM's atomic order."""
import pytest
import torch

from mstar.utils.fused_moe.align import moe_align_block_size


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_an_empty_step_aligns_to_nothing(device):
    """With E > 64 the CUDA op launched its sort over 0 blocks and never checked: the
    error stayed latched and the next unrelated kernel raised it."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    _, _, padded = moe_align_block_size(torch.empty(0, 8, dtype=torch.int32, device=device),
                                        16, 256)
    assert padded.item() == 0
    torch.ones(4, device=device).sum().item()  # a latched launch error would raise here


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("experts", [128, 896, 112])
@pytest.mark.parametrize("tokens", [1, 4, 64, 128, 256, 1024])
@pytest.mark.parametrize("block_size", [8, 16, 64])
def test_the_cuda_op_places_slots_as_the_torch_fallback_does_on_every_call(experts, tokens, block_size):
    """vLLM's large-batch placement ranked slots by atomicAdd, so the order within an expert
    changed between calls with the same ids and Marlin MoE's output with it (H200 job 1656)."""
    from mstar.utils.fused_moe import align

    if not align._cuda_op_available():
        pytest.skip("the CUDA op did not build")
    g = torch.Generator().manual_seed(tokens * 7 + experts)
    ids = torch.stack([torch.randperm(896 if experts == 112 else experts, generator=g)[:16] for _ in range(tokens)])
    if experts == 112:  # expert parallel: another rank's experts are the sentinel id
        ids = torch.where(ids < experts, ids, torch.full_like(ids, experts))
    ids = ids.to(torch.int32)
    n = ids.numel() + experts * (block_size - 1)
    want = [torch.empty(n, dtype=torch.int32), torch.empty(-(-n // block_size), dtype=torch.int32),
            torch.empty(1, dtype=torch.int32)]
    align._moe_align_block_size_torch(ids, block_size, experts, *want)
    first = align.moe_align_block_size(ids.cuda(), block_size, experts)
    padded = int(want[2])
    assert int(first[2]) == padded
    torch.testing.assert_close(first[1].cpu()[: padded // block_size], want[1][: padded // block_size], rtol=0, atol=0)
    if ids.numel() > 4096:  # vLLM's placement: each expert's slots, in any order
        owner = want[1][: padded // block_size].repeat_interleave(block_size).long() * (ids.numel() + 1)
        torch.testing.assert_close((owner + first[0].cpu()[:padded]).sort().values,
                                   (owner + want[0][:padded]).sort().values, rtol=0, atol=0)
        return
    torch.testing.assert_close(first[0].cpu(), want[0], rtol=0, atol=0)
    for _ in range(20):
        again = align.moe_align_block_size(ids.cuda(), block_size, experts)
        assert all(torch.equal(a, b) for a, b in zip(again, first, strict=True))
