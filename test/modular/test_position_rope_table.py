"""``PositionConfig.rope_table_dtype``: CUDA rotates from a rounded cos/sin
table built once from the config, so per-call RoPE arguments the table does
not encode are refused instead of silently ignored."""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager

# What Attention._apply_rope passes on every call unless a model overrides it.
ATTENTION_LLAMA31_DEFAULTS = dict(low_freq_factor=1.0, high_freq_factor=1.0, old_context_len=8192)


def table_manager(**overrides) -> RopeManager:
    config = dict(kv_cache="kv", interleave=True, rope_theta=1e6, rotary_dim=8, rope_table_dtype=torch.bfloat16)
    config.update(overrides)
    manager = RopeManager(PositionConfig(**config), torch.device("cpu"), max_seq_len=64, head_dim=8)
    manager._current_pos_ids["main"] = torch.arange(2)
    return manager


@pytest.fixture
def kernel_calls(monkeypatch):
    calls = []

    def record(q, k, pos_ids, cos_sin_cache, *args, **kwargs):
        calls.append(cos_sin_cache)

    monkeypatch.setattr(torch.ops.mstar, "rope_apply_qk_inplace", record)
    return calls


def qk():
    q = torch.ones((2, 1, 8), dtype=torch.bfloat16)
    return q, q.clone()


def test_rope_table_dtype_rounds_the_cos_sin_table():
    manager = table_manager()
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    angles = torch.outer(torch.arange(64, dtype=torch.float32), inv_freq)
    expected = torch.cat([angles.cos(), angles.sin()], dim=-1).to(torch.bfloat16).float()
    assert manager._rope_table.dtype == torch.float32
    assert torch.equal(manager._rope_table, expected)
    exact = RopeManager(PositionConfig(kv_cache="kv", rotary_dim=8), torch.device("cpu"), max_seq_len=64, head_dim=8)
    assert exact._rope_table is None


def test_matching_arguments_rotate_from_the_table(kernel_calls):
    manager = table_manager()
    manager.apply_qk(*qk(), "main")
    manager.apply_qk(*qk(), "main", rope_theta=1e6, rope_scale=1.0, **ATTENTION_LLAMA31_DEFAULTS)
    assert len(kernel_calls) == 2
    assert all(cache is manager._rope_table for cache in kernel_calls)


def test_rotary_dim_defaults_to_the_head_dim():
    manager = table_manager(rotary_dim=None)
    assert manager._rope_table.shape == (64, 8)
    manager._check_rope_table_args(8, 1.0, 1e6, {})


@pytest.mark.parametrize(
    "call_args",
    [
        dict(rope_theta=1e4),
        dict(rope_scale=2.0),
        dict(rotary_dim=4),
    ],
)
def test_a_per_call_override_the_table_ignores_is_refused(kernel_calls, call_args):
    with pytest.raises(RuntimeError, match="do not match the rope_table_dtype table"):
        table_manager().apply_qk(*qk(), "main", **call_args)
    assert kernel_calls == []


def test_llama31_scaling_is_refused_at_runtime_unless_it_is_a_no_op(kernel_calls):
    scaled = table_manager(rope_scale=8.0)
    scaled.apply_qk(*qk(), "main")
    with pytest.raises(NotImplementedError, match="Llama 3.1"):
        scaled.apply_qk(*qk(), "main", **ATTENTION_LLAMA31_DEFAULTS)
    assert len(kernel_calls) == 1


def test_llama31_config_is_refused_at_build():
    with pytest.raises(NotImplementedError, match="Llama 3.1"):
        table_manager(rope_scale=8.0, **ATTENTION_LLAMA31_DEFAULTS)


def xpu_manager(monkeypatch, kv_dtype=torch.bfloat16, **overrides) -> RopeManager:
    """A manager on an XPU device whose tables are built on the CPU (no XPU here)."""
    build = RopeManager._build_rope_cache

    def build_on_cpu(self, *args):
        return build(SimpleNamespace(_max_seq_len=self._max_seq_len, _device=torch.device("cpu")), *args)

    monkeypatch.setattr(RopeManager, "_build_rope_cache", build_on_cpu)
    config = dict(kv_cache="kv", interleave=True, rope_theta=1e6, rotary_dim=8, rope_table_dtype=torch.bfloat16)
    config.update(overrides)
    return RopeManager(PositionConfig(**config), torch.device("xpu"), max_seq_len=64, head_dim=8, dtype=kv_dtype)


def test_xpu_refuses_a_table_dtype_its_cache_does_not_round_to(monkeypatch):
    with pytest.raises(NotImplementedError, match=r"rope_table_dtype=torch.bfloat16 is not implemented for XPU"):
        xpu_manager(monkeypatch, kv_dtype=torch.float16)
    with pytest.raises(NotImplementedError, match=r"rotates from a torch.float32 cos/sin table"):
        xpu_manager(monkeypatch, rope_dtype=torch.float32)


def test_xpu_cache_in_the_table_dtype_holds_the_cuda_table(monkeypatch):
    xpu = xpu_manager(monkeypatch)
    assert xpu._rope_table is None
    assert torch.equal(xpu._rope_cache.float(), table_manager()._rope_table)
