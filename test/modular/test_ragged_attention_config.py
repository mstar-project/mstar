"""CPU tests for the ragged attention resource's config: its activation dtype
and its deployment overrides.

A cacheless attention has no KV cache to inherit a dtype from, so a node without a
dtype of its own (a DiT run without autocast) declares it on the config; the nodes'
resolved dtype stays the fallback for nodes that have one, and neither being set is
an error, not an fp32 kernel.
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


def test_config_dtype_reaches_the_kernel_without_a_node_dtype():
    manager = RaggedAttnManager.build(_spec(torch.bfloat16), EngineResourceInfo(device=CPU, compute_dtype=None))
    assert manager._kwargs["q_data_type"] is torch.bfloat16


def test_config_dtype_wins_over_the_node_dtype():
    manager = RaggedAttnManager.build(
        _spec(torch.float16), EngineResourceInfo(device=CPU, compute_dtype=torch.bfloat16),
    )
    assert manager._kwargs["q_data_type"] is torch.float16


def test_node_dtype_is_the_fallback():
    manager = RaggedAttnManager.build(_spec(None), EngineResourceInfo(device=CPU, compute_dtype=torch.bfloat16))
    assert manager._kwargs["q_data_type"] is torch.bfloat16


def test_no_dtype_anywhere_is_an_error():
    with pytest.raises(ValueError, match="'dit_attn' declares no dtype"):
        RaggedAttnManager.build(
            _spec(None), EngineResourceInfo(device=CPU, node_dtypes={"dit": None}),
        )


@pytest.mark.parametrize("declared, override", [
    (64, {"max_segments_per_request": 4}),
    (None, {"max_segments_per_request": 4, "max_segments_per_step": 64}),
])
def test_a_per_request_ceiling_on_a_per_step_resource_is_refused(declared, override):
    """A resource sized per step never reads the per-request ceiling, so a
    deployment's override of it is refused rather than ignored."""
    spec = _spec()
    spec.config.max_segments_per_step = declared
    with pytest.raises(ValueError, match="max_segments_per_step"):
        spec.apply_yaml_overrides(**override)
