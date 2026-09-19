"""The draft's yarn rope: the tables and rotation match vLLM's DeepseekScalingRotaryEmbedding for
K3DSparkModel (checked bit-exact against vLLM's helpers on 2026-09-17; the numbers below are from
that run), and the attention scale carries the yarn mscale squared."""
import math

import pytest
import torch

from mstar.model.kimi_k3.dspark.config import DSparkConfig, YarnParams
from mstar.model.kimi_k3.dspark.rope import YarnRotary, yarn_inv_freq, yarn_mscale


def test_yarn_frequencies_and_scale_for_the_kimi_k3_draft():
    p = YarnParams()  # the checkpoint's: factor 32, window 32768, theta 50000, beta 32 / 1, mscale 1 / 1
    inv = yarn_inv_freq(64, p)
    assert inv.shape == (32,) and inv[0] == pytest.approx(1.0)  # the fastest dim is pure extrapolation
    assert inv[-1] == pytest.approx(1.0 / (32 * 50000 ** (62 / 64)))  # the slowest is interpolated by the factor
    assert torch.all(inv[1:] < inv[:-1])
    assert yarn_mscale(32.0, 1.0) == pytest.approx(0.1 * math.log(32.0) + 1.0)
    rope = YarnRotary(64, p, 4096)
    assert rope.attn_scale_factor == pytest.approx(1.8132604340394958)
    assert torch.equal(rope.cos[0], torch.ones(64)) and torch.equal(rope.sin[0], torch.zeros(64))


def test_rotation_is_interleaved_and_norm_preserving():
    rope = YarnRotary(64, YarnParams(), 4096)
    torch.manual_seed(0)
    x = torch.randn(5, 3, 64)
    pos = torch.tensor([0, 1, 100, 2047, 4095])
    y = rope.apply(x, pos)
    assert torch.equal(y[0], x[0])  # position 0 is the identity
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)  # pairs rotate, norms stay
    # GPT-J style: dims (2i, 2i+1) form the rotated pairs
    c, s = rope.cos[pos][:, None, ::2], rope.sin[pos][:, None, ::2]
    want_even = x[..., ::2] * c - x[..., 1::2] * s
    assert torch.allclose(y[..., ::2], want_even, atol=1e-5)
    k = torch.randn(5, 64)
    assert rope.apply(k, pos).shape == (5, 64)


def test_config_reads_the_checkpoint_when_present():
    import os
    d = "/scratch/m000137-pm06/atj10/kimi_k3_mstar/ckpt/Kimi-K3-DSpark"
    if not os.path.exists(os.path.join(d, "config.json")):
        pytest.skip("draft checkpoint not present")
    cfg = DSparkConfig.from_dir(d)
    assert cfg.num_hidden_layers == 5 and cfg.target_layer_ids == (2, 23, 47, 71, 89) and cfg.context_width == 35840
    assert cfg.qk_head_dim == 192 and cfg.rope.factor == 32.0 and cfg.mask_token_id == 163837
