"""Praxis nodes keep reference autocast and float32 precision settings."""

import pytest
import torch

from mstar.engine.accelerator_graph_runner import autocast_scope
from mstar.engine.fp32_precision import fp32_precision_scope, node_fp32_precision
from mstar.engine.node_dtype import resolve_node_dtype
from mstar.model.praxis1.submodules import (
    Praxis1ActionExpertSubmodule,
    Praxis1BackboneSubmodule,
    Praxis1TextEncoderSubmodule,
)


@pytest.mark.parametrize(
    ("name", "submodule"),
    [
        ("text_encoder", Praxis1TextEncoderSubmodule),
        ("backbone", Praxis1BackboneSubmodule),
        ("action_expert", Praxis1ActionExpertSubmodule),
    ],
)
def test_trunk_float32_matmuls_use_reference_precision(name, submodule):
    """The engine's per-node scope overrides and restores its TF32 default."""
    original = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        precision = node_fp32_precision(name, submodule)
        assert precision is not None
        with fp32_precision_scope(precision):
            assert torch.get_float32_matmul_precision() == "highest"
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(original)


def test_text_encoder_keeps_reference_math_outside_engine_autocast():
    """The engine's BF16 model default must not wrap Qwen's own forward."""
    resolved = resolve_node_dtype(Praxis1TextEncoderSubmodule, torch.bfloat16)
    assert resolved is None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        with autocast_scope(resolved, "cpu", "text_encoder"):
            assert not torch.is_autocast_enabled("cpu")
        assert torch.is_autocast_enabled("cpu")
