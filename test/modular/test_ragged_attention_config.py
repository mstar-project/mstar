"""CPU tests for the ragged attention resource's activation dtype.

A cacheless attention has no KV cache to inherit a dtype from, so a node without a
KV-backed attention (a DiT) declares it on the config; the engine's KV dtype stays the
fallback for nodes that have one, and neither being set is an error, not an fp32 kernel.
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources import RaggedAttentionConfig, RaggedAttentionSpec  # noqa: E402
from mstar.engine.resources.attn.ragged.base import RaggedAttnManager  # noqa: E402
from mstar.engine.resources.base import EngineResourceInfo  # noqa: E402

CPU = torch.device("cpu")


def _spec(dtype=None) -> RaggedAttentionSpec:
    return RaggedAttentionSpec(
        resource_key="dit_attn", nodes={"dit"},
        config=RaggedAttentionConfig(num_qo_heads=2, num_kv_heads=2, head_dim=128, dtype=dtype),
    )


def test_config_dtype_reaches_the_kernel_without_a_kv_dtype():
    manager = RaggedAttnManager.build(_spec(torch.bfloat16), EngineResourceInfo(device=CPU, kv_dtype=None))
    assert manager._kwargs["q_data_type"] is torch.bfloat16


def test_config_dtype_wins_over_the_engine_kv_dtype():
    manager = RaggedAttnManager.build(_spec(torch.float16), EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16))
    assert manager._kwargs["q_data_type"] is torch.float16


def test_engine_kv_dtype_is_the_fallback():
    manager = RaggedAttnManager.build(_spec(None), EngineResourceInfo(device=CPU, kv_dtype=torch.bfloat16))
    assert manager._kwargs["q_data_type"] is torch.bfloat16


def test_no_dtype_anywhere_is_an_error():
    with pytest.raises(ValueError, match="no activation dtype"):
        RaggedAttnManager.build(_spec(None), EngineResourceInfo(device=CPU, kv_dtype=None))
