"""YaRN declaration, table parity, dispatch, and prefix identity contracts."""

from dataclasses import asdict, replace

import pytest
import torch

from mstar.engine.resources.position.config import LongRoPE, PositionConfig, YaRNConfig
from mstar.engine.resources.position.manager import RopeManager

YARN = YaRNConfig(factor=16, original_max_position_embeddings=16384)


def manager(*, max_seq_len=64, **kwargs):
    config = dict(kv_cache="kv", rotary_dim=128, rope_theta=1e6, yarn=YARN, rope_dtype=torch.float32)
    config.update(kwargs)
    return RopeManager(PositionConfig(**config), torch.device("cpu"), max_seq_len=max_seq_len)


@pytest.mark.parametrize("changes", [{}, {"mscale": 2.0}, {"attention_factor": 1.3}, {"truncate": False}])
def test_table_matches_transformers(changes):
    pytest.importorskip("transformers.models.ministral3.configuration_ministral3")
    from transformers.modeling_rope_utils import _compute_yarn_parameters
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

    yarn = replace(YARN, **changes)
    config = Ministral3Config(
        hidden_size=3072, num_attention_heads=32, head_dim=128,
        rope_parameters=dict(rope_type="yarn", rope_theta=1e6, **asdict(yarn)),
    )
    inv_freq, amplitude = _compute_yarn_parameters(config, torch.device("cpu"))
    positions = torch.tensor([0, 1, 7, 16383, 16384, 16385, 32768, 262143])
    angles = positions.float().unsqueeze(1) * inv_freq
    expected = torch.cat((angles.cos(), angles.sin()), -1) * amplitude
    table = manager(yarn=yarn, max_seq_len=262144).cos_sin_table
    torch.testing.assert_close(table[positions], expected, rtol=0, atol=0)
    # Negative control: plain RoPE disagrees even on a short prompt.
    plain_freq = 1 / (1e6 ** (torch.arange(0, 128, 2).float() / 128))
    assert not torch.allclose(table[7, :64], (7 * plain_freq).cos(), atol=1e-6, rtol=1e-6)


def test_capacity_does_not_change_frequencies_and_rounding_is_last():
    small = manager().cos_sin_table
    large = manager(max_seq_len=128).cos_sin_table
    assert torch.equal(small, large[:64])
    assert torch.equal(manager(rope_table_dtype=torch.bfloat16).cos_sin_table, small.bfloat16().float())


def test_dispatch_and_fusion_guard(monkeypatch):
    pos = manager()
    calls = []
    monkeypatch.setattr(torch.ops.mstar, "rope_apply_qk_inplace", lambda *args, **kwargs: calls.append(args))
    q, k = torch.ones(3, 4, 128), torch.ones(3, 2, 128)
    positions = torch.tensor([7, 1, 3, 0])  # padded position tail
    pos.apply_qk(q, k, pos_ids=positions,
                 low_freq_factor=1.0, high_freq_factor=1.0, old_context_len=8192)
    assert len(calls) == 1 and calls[0][3] is pos.cos_sin_table
    assert torch.equal(calls[0][2], positions[:3])
    assert pos.fusable_rope_theta(128) is None
    assert manager(yarn=None, rope_dtype=None).fusable_rope_theta(128) == 1e6
    with pytest.raises(RuntimeError, match="do not match"):
        pos.apply_qk(q, k, pos_ids=positions, rope_theta=10000)
    with pytest.raises(ValueError, match="runtime Llama"):
        pos.apply_qk(q, k, pos_ids=positions, low_freq_factor=2.0)


@pytest.mark.parametrize("field,value", [
    ("factor", 8), ("original_max_position_embeddings", 8192),
    ("beta_fast", 16), ("beta_slow", 2), ("mscale", 2),
    ("mscale_all_dim", 2), ("attention_factor", 1.2), ("truncate", False),
])
def test_each_yarn_parameter_changes_prefix_identity(field, value):
    assert manager(yarn=replace(YARN, **{field: value})).fingerprint() != manager().fingerprint()
    assert manager(yarn=None).fingerprint() != manager().fingerprint()


@pytest.mark.parametrize("kwargs", [
    dict(rope_scale=2), dict(low_freq_factor=1), dict(rope_theta=1),
    dict(longrope=LongRoPE((1.0,) * 64, (2.0,) * 64, 16384, 262144)),
])
def test_refuse_incompatible_config(kwargs):
    with pytest.raises(ValueError, match="YaRN"):
        manager(**kwargs)


@pytest.mark.parametrize("rotary_dim,interleave,alignment", [(16, False, 32), (8, True, 16)])
def test_yarn_table_rejects_unsafe_cuda_width_before_allocation(rotary_dim, interleave, alignment):
    config = PositionConfig(kv_cache="kv", rotary_dim=rotary_dim, interleave=interleave, yarn=YARN)
    with pytest.raises(ValueError, match=f"multiple of {alignment}"):
        RopeManager(config, torch.device("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and FlashInfer")
@pytest.mark.parametrize("interleave", [False, True])
def test_cuda_rotation_against_transformers(interleave):
    pytest.importorskip("flashinfer")
    pytest.importorskip("transformers.models.ministral3.configuration_ministral3")
    from transformers.modeling_rope_utils import _compute_yarn_parameters
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

    config = Ministral3Config(head_dim=128, rope_parameters=dict(rope_type="yarn", rope_theta=1e6, **asdict(YARN)))
    inv_freq, amplitude = _compute_yarn_parameters(config, torch.device("cuda"))
    positions = torch.tensor([0, 7, 16383, 16384, 262143], device="cuda")
    angles = positions.float().unsqueeze(1) * inv_freq
    cos, sin = (angles.cos() * amplitude)[:, None], (angles.sin() * amplitude)[:, None]

    def reference(x):
        a, b = (x.float()[..., ::2], x.float()[..., 1::2]) if interleave else x.float().chunk(2, -1)
        a, b = a * cos - b * sin, b * cos + a * sin
        return torch.stack((a, b), -1).flatten(-2) if interleave else torch.cat((a, b), -1)

    pos = RopeManager(PositionConfig(kv_cache="kv", rotary_dim=128, rope_theta=1e6,
                                    yarn=YARN, interleave=interleave), torch.device("cuda"), max_seq_len=262144)
    gen = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(5, 4, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn(5, 2, 128, device="cuda", dtype=torch.bfloat16, generator=gen)
    expected = reference(q), reference(k)
    actual = pos.apply_qk(q.clone(), k.clone(), pos_ids=positions)
    for result, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(result.float(), target.bfloat16().float(), atol=0.03125, rtol=0.01)
