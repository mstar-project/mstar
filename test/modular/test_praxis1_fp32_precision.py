"""Praxis trunk nodes keep their float32 matmuls at reference precision."""

import pytest
import torch

from mstar.engine.fp32_precision import fp32_precision_scope, node_fp32_precision
from mstar.model.praxis1.submodules import (
    Praxis1ActionExpertSubmodule,
    Praxis1BackboneSubmodule,
)


@pytest.mark.parametrize(
    ("name", "submodule"),
    [
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
