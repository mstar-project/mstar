"""The shared BigVGAN generator: weight-norm folding in both spellings, output length and
the checkpoint's parameter names."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mstar.model.components.bigvgan import BigVGANGenerator, fold_weight_norm


@pytest.mark.parametrize("spelling", ["hook", "parametrization"])
def test_fold_weight_norm_gives_the_effective_weight(spelling):
    torch.manual_seed(0)
    conv = nn.ConvTranspose1d(6, 3, 4, 2) if spelling == "hook" else nn.Conv1d(6, 3, 5)
    if spelling == "hook":
        conv = torch.nn.utils.weight_norm(conv)
        conv.weight_g.data.uniform_(0.5, 2.0)
    else:
        conv = torch.nn.utils.parametrizations.weight_norm(conv)
        conv.parametrizations.weight.original0.data.uniform_(0.5, 2.0)
    x = torch.randn(1, 6, 11)
    want = conv(x)  # evaluates the weight from its two halves
    state = {f"block.{k}": v for k, v in conv.state_dict().items()}
    folded = dict(fold_weight_norm(state.items()))
    assert set(folded) == {"block.weight", "block.bias"}
    plain = nn.ConvTranspose1d(6, 3, 4, 2) if spelling == "hook" else nn.Conv1d(6, 3, 5)
    plain.load_state_dict({k.removeprefix("block."): v for k, v in folded.items()})
    torch.testing.assert_close(plain(x), want, rtol=0, atol=0)


def test_generator_upsamples_by_the_product_of_its_rates():
    gen = BigVGANGenerator(8, 16, (2, 3), (4, 5), (3, 5), ((1, 3), (1, 2)), final_bias=False, final_tanh=False)
    assert gen.conv_post.bias is None
    out = gen(torch.randn(2, 8, 7))
    assert out.shape == (2, 1, 7 * 6)
    assert out.abs().max() <= 1.0
    names = {n for n, _ in gen.named_parameters()}
    assert {"ups.0.0.weight", "resblocks.3.convs2.1.weight", "activation_post.upsample.filter",
            "resblocks.0.activations.0.downsample.lowpass.filter", "activation_post.act.beta"} <= names
