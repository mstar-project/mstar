"""moe_align_block_size on an empty step."""
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
