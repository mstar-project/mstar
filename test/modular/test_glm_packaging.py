"""``mstar serve`` and ``pip install`` know the GLM models."""

from pathlib import Path

import pytest

from mstar.cli.main import DEFAULT_CONFIGS, _resolve_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("model,config", [("glm52", "glm52_tp8.yaml")])
def test_mstar_serve_resolves_the_default_config(model, config):
    assert DEFAULT_CONFIGS[model] == config
    assert Path(_resolve_config(model, None)).resolve() == (ROOT / "configs" / config).resolve()
