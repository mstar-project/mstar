"""LinearPatchEmbed computes what the patchifying conv it replaces computes."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.model.components.patch_embed import LinearPatchEmbed


def _randomized(conv: nn.Module) -> nn.Module:
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in conv.parameters():
            p.copy_(torch.randn(p.shape, generator=g))
    return conv.double()


def test_matches_the_qwen3_omni_vision_patch_embed():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionPatchEmbed

    cfg = SimpleNamespace(patch_size=4, temporal_patch_size=2, in_channels=3, hidden_size=24)
    hf = Qwen3OmniMoeVisionPatchEmbed(cfg)
    _randomized(hf.proj)
    # the processor's rows: one flattened (channel, time, row, column) patch each
    patches = torch.randn(37, 3 * 2 * 4 * 4, dtype=torch.float64)
    with torch.no_grad():
        expected = hf(patches)
        got = LinearPatchEmbed(hf.proj)(patches)
    assert got.shape == expected.shape == (37, 24)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_matches_a_2d_patch_conv_and_casts_to_the_weight_dtype():
    conv = _randomized(nn.Conv2d(3, 8, kernel_size=5, stride=5))
    patches = torch.randn(11, 3, 5, 5, dtype=torch.float32)
    with torch.no_grad():
        expected = conv(patches.double()).flatten(1)
        got = LinearPatchEmbed(conv)(patches)
    assert got.dtype == torch.float64
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_without_bias():
    conv = _randomized(nn.Conv3d(2, 6, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False))
    patches = torch.randn(5, 2 * 1 * 2 * 2, dtype=torch.float64)
    with torch.no_grad():
        expected = conv(patches.view(-1, 2, 1, 2, 2)).flatten(1)
        got = LinearPatchEmbed(conv)(patches)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("conv", [
    nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
    nn.Conv2d(1, 4, kernel_size=2, stride=1),
    nn.Conv3d(4, 4, kernel_size=2, stride=2, groups=2),
])
def test_rejects_convs_that_overlap_pad_or_group(conv):
    with pytest.raises(ValueError, match="not a patchifying conv"):
        LinearPatchEmbed(conv)
