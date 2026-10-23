"""``mstar serve`` and ``pip install`` know the GLM models."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from mstar.cli.main import DEFAULT_CONFIGS, _resolve_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("model,config", [("glm52", "glm52_tp8.yaml"), ("glm5_next", "glm5_next_tp8.yaml")])
def test_mstar_serve_resolves_the_default_config(model, config):
    assert DEFAULT_CONFIGS[model] == config
    assert Path(_resolve_config(model, None)).resolve() == (ROOT / "configs" / config).resolve()


def _extras(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)["project"]["optional-dependencies"]


def test_the_glm_extra_installs_what_the_models_import():
    glm = {Requirement(raw).name: Requirement(raw) for raw in _extras(ROOT / "pyproject.toml")["glm"]}
    assert {"flashinfer-python", "huggingface-hub", "safetensors", "transformers", "triton"} <= set(glm)
    # check_flashinfer's floor, which long context refuses below
    assert str(glm["flashinfer-python"].specifier) == ">=0.6.18"
    for alias in ("mstar-serve", "mstar-project"):
        assert _extras(ROOT / "packaging" / "aliases" / alias / "pyproject.toml")["glm"] == ["mstar-ai[glm]"]
