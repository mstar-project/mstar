"""Cold Praxis-1 action head, checked against the pinned reference on CPU."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from mstar.engine.node_dtype import resolve_node_dtype
from mstar.model.praxis1.config import Praxis1Config
from mstar.model.praxis1.praxis1_model import Praxis1Model
from mstar.model.praxis1.submodules import Praxis1ActionHeadSubmodule


def _head(rig: str, *, norm: dict | None = None) -> Praxis1ActionHeadSubmodule:
    config = Praxis1Config.for_rig(rig)
    dim = config.action_dim
    if norm is None:
        norm = {
            "action_q01": np.linspace(-1.0, -0.2, dim).tolist(),
            "action_q99": np.linspace(0.7, 1.9, dim).tolist(),
        }
    return Praxis1ActionHeadSubmodule(replace(config, normalization=norm))


@pytest.mark.parametrize("rig", ["single_arm", "bimanual"])
def test_cold_head_returns_21_relative_rows_for_openpi(rig: str) -> None:
    head = _head(rig)
    config = head.config
    tokens = torch.randn(1, config.action_tokens, config.action_patch_dim)
    got = head.forward("chunk", None, tokens)["actions"][0]
    assert got.shape == (21, config.action_dim)
    assert got.dtype == torch.float32
    assert torch.isfinite(got).all()
    assert head.prepare_inputs("chunk", None, {"action_latents": [tokens]}).tensor_inputs["action_latents"] is tokens


@pytest.mark.parametrize("rig,custom_dims", [
    ("single_arm", None), ("single_arm", [0, 9]), ("bimanual", None),
])
def test_cold_head_matches_reference_relative_rows(rig: str, custom_dims: list[int] | None) -> None:
    pytest.importorskip("praxis", reason="reference checkout is not on PYTHONPATH")
    from praxis import bimanual, robot_io, single_arm

    reference = single_arm if rig == "single_arm" else bimanual
    norm = None
    if custom_dims is not None:
        norm = {
            "action_q01": [-1.0] * 10,
            "action_q99": [1.0] * 10,
            "q99_normed_dims": custom_dims,
        }
    head = _head(rig, norm=norm)
    config = head.config
    stats = robot_io.NormStats(
        action_q01=head.action_q01.numpy(),
        action_q99=head.action_q99.numpy(),
        proprio_mean=np.zeros(config.action_dim, np.float32),
        proprio_std=np.ones(config.action_dim, np.float32),
    )
    generator = torch.Generator().manual_seed(2026)
    tokens = torch.randn(1, config.action_tokens, config.action_patch_dim, generator=generator)
    _normalized, expected = reference.decode_actions(
        tokens, stats, depth=0, dims=head.q99_dims.numpy(),
    )
    got = head.forward("chunk", None, tokens)["actions"][0]
    assert np.array_equal(got.numpy(), expected)
    assert expected.shape[0] == config.action_horizon == 21


@pytest.mark.parametrize("bad", [
    {"action_q01": [0.0] * 10},
    {"action_q01": [0.0] * 10, "action_q99": [float("nan")] * 10},
    {"action_q01": [0.0] * 10, "action_q99": [0.0] * 10},
    {"action_q01": [0.0] * 10, "action_q99": [1.0] * 10, "q99_normed_dims": [10]},
    {"action_q01": [0.0] * 10, "action_q99": [1.0] * 10, "q99_normed_dims": [0.5]},
])
def test_head_refuses_bad_checkpoint_normalization(bad: dict) -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        _head("single_arm", norm=bad)


def test_head_refuses_an_action_window_with_wrong_token_count() -> None:
    head = _head("single_arm")
    tokens = torch.zeros(1, head.config.action_tokens - 1, head.config.action_patch_dim)
    with pytest.raises(ValueError, match="expected 21 rows"):
        head.forward("chunk", None, tokens)


def test_model_builds_its_action_head_from_checkpoint_stats() -> None:
    model = Praxis1Model("runwayml/praxis-1-droid", skip_weight_loading=True)
    model.config = _head("single_arm").config
    model.skip_weight_loading = False
    head = model._create_submodule("action_head", "cpu")
    assert isinstance(head, Praxis1ActionHeadSubmodule)
    assert head.config is model.config
    dtype = resolve_node_dtype(head, torch.bfloat16)
    assert dtype is None
    head.to(device="cpu", dtype=dtype)
    assert head.action_q01.dtype == head.action_q99.dtype == torch.float32
