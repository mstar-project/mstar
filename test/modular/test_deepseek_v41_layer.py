"""Two-layer CPU golden from DeepSeek V4.1's published Block.forward.

Reference: deepseek-ai/DeepSeek-V4.1-Flash at
314113f8a7dadb388215396ba5fbee2c610a04d6. The published Block and
RMSNorm run with deterministic toy attention/FFN modules and a CPU translation
of its published Sinkhorn kernel. The generation recipe is in data/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.deepseek_v41.layer import DeepSeekV41Layer
from mstar.model.glm5_next.mhc import Glm5NextHyperConnection, update_streams


class ToyAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, hidden_size))

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        return torch.tanh(F.linear(x, self.weight)) + start_pos * 0.001


class ToyFFN(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, hidden_size))

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        image_offset = image_mask.unsqueeze(-1) * 0.01 if image_mask is not None else 0
        return torch.sin(F.linear(x, self.weight)) + image_offset


def _two_layers() -> tuple[list[DeepSeekV41Layer], torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(811)
    layers = [DeepSeekV41Layer(i, 8, ToyAttention(8), ToyFFN(8), norm_eps=1e-20) for i in range(2)]
    with torch.no_grad():
        for layer in layers:
            for name in ("attn", "ffn"):
                getattr(layer, name).weight.copy_(torch.randn((8, 8), generator=generator) * 0.15)
            for name in ("attn_norm", "ffn_norm"):
                getattr(layer, name).weight.copy_(0.8 + torch.randn((8,), generator=generator) * 0.1)
            for name in ("attn", "ffn"):
                site = getattr(layer, f"{name}_hc")
                for suffix in ("fn", "base", "scale"):
                    target = getattr(site, suffix)
                    multiplier = {"fn": 0.1, "base": 0.2, "scale": 0.1}[suffix]
                    value = torch.randn(target.shape, generator=generator) * multiplier
                    if suffix == "scale":
                        value += 1
                    target.copy_(value)
    streams = torch.randn((1, 3, 4, 8), generator=generator) * 0.5
    image_mask = torch.tensor([[False, True, False]])
    initial_pre = torch.zeros((1, 3, 4))
    initial_pre[:, :, 0] = 1
    return layers, streams, initial_pre, image_mask


def test_two_layer_carry_matches_pinned_reference_block() -> None:
    layers, streams, pre, image_mask = _two_layers()
    golden = json.loads((Path(__file__).parent / "data/deepseek_v41_layer_golden.json").read_text())
    with torch.no_grad():
        for index, layer in enumerate(layers):
            streams, pre = layer(streams, 7, pre, image_mask)
            prefix = "golden_first_" if index == 0 else "golden_"
            torch.testing.assert_close(streams, torch.tensor(golden[f"{prefix}streams"]), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(pre, torch.tensor(golden[f"{prefix}pre"]), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            layers[-1]._collapse(streams, pre),
            torch.tensor(golden["golden_final"]),
            rtol=1e-5,
            atol=1e-6,
        )


def test_glm_hyper_connection_forward_keeps_its_original_collapse() -> None:
    site = Glm5NextHyperConnection(8)
    streams = torch.randn(1, 3, 4, 8, generator=torch.Generator().manual_seed(19))
    pre, post, comb = site.mixes(streams)
    post_forward, comb_forward, collapsed = site(streams)
    torch.testing.assert_close(post_forward, post, rtol=0, atol=0)
    torch.testing.assert_close(comb_forward, comb, rtol=0, atol=0)
    torch.testing.assert_close(collapsed, (pre.unsqueeze(-1) * streams).sum(dim=2), rtol=0, atol=0)


def test_layer_refuses_wrong_incoming_pre_shape() -> None:
    layers, streams, _, image_mask = _two_layers()
    with pytest.raises(ValueError, match="one value per stream and token"):
        layers[0](streams, 7, torch.ones(1, 3, 3), image_mask)


def test_layer_keeps_published_hyper_connection_parameters_in_fp32() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        layer = DeepSeekV41Layer(0, 8, ToyAttention(8), ToyFFN(8))
    finally:
        torch.set_default_dtype(previous)
    for site in (layer.attn_hc, layer.ffn_hc):
        assert all(getattr(site, name).dtype == torch.float32 for name in ("fn", "base", "scale"))
    original = {}
    with torch.no_grad():
        for site_name in ("attn_hc", "ffn_hc"):
            site = getattr(layer, site_name)
            for name in ("fn", "base", "scale"):
                param = getattr(site, name)
                param.fill_(0.1234567)
                original[(site_name, name)] = param.detach().clone()
    layer.to(torch.bfloat16)
    for (site_name, name), before in original.items():
        after = getattr(getattr(layer, site_name), name)
        assert after.dtype == torch.float32
        assert torch.equal(after, before)
    assert layer.attn_norm.weight.dtype == torch.bfloat16


def test_bf16_post_keeps_fp32_coefficients_until_final_cast() -> None:
    generator = torch.Generator().manual_seed(824)
    streams = torch.randn((2, 3, 4, 8), generator=generator).to(torch.bfloat16)
    out = torch.randn((2, 3, 8), generator=generator).to(torch.bfloat16)
    post = torch.rand((2, 3, 4), generator=generator)
    comb = torch.softmax(torch.randn((2, 3, 4, 4), generator=generator), dim=-1)
    expected = (
        post.unsqueeze(-1) * out.float().unsqueeze(-2)
        + torch.sum(comb.unsqueeze(-1) * streams.float().unsqueeze(-2), dim=2)
    ).to(out.dtype)
    observed = DeepSeekV41Layer._post(streams, out, post, comb)
    assert torch.equal(observed, expected)
    assert not torch.equal(observed, update_streams(streams, out, post, comb))
