"""Packaging contracts that keep Waypoint usable from every install name."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - the project requires Python 3.12
    import tomli as tomllib

import pytest
from packaging.requirements import Requirement

from mstar.cli.main import DEFAULT_CONFIGS, _resolve_config
from mstar.model.waypoint.checkpoint import (
    TAEHV_UPSTREAM_ARCHIVE,
    TAEHV_UPSTREAM_REVISION,
    require_taehv_runtime,
)
from mstar.model.waypoint.components.taehv import load_taehv

ROOT = Path(__file__).resolve().parents[2]
ROOT_PYPROJECT = ROOT / "pyproject.toml"
ALIAS_PYPROJECTS = (
    ROOT / "packaging" / "aliases" / "mstar-ai" / "pyproject.toml",
    ROOT / "packaging" / "aliases" / "mstar-project" / "pyproject.toml",
)


def _project(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)["project"]


def test_published_metadata_contains_no_direct_url_dependencies():
    project = _project(ROOT_PYPROJECT)
    grouped = {"base": project["dependencies"], **project["optional-dependencies"]}
    direct = [
        (group, raw)
        for group, requirements in grouped.items()
        for raw in requirements
        if Requirement(raw).url is not None
    ]
    assert direct == [], "PyPI rejects distributions that declare direct-URL dependencies"


def test_alias_packages_forward_every_root_extra():
    root_extras = _project(ROOT_PYPROJECT)["optional-dependencies"]
    for path in ALIAS_PYPROJECTS:
        alias_extras = _project(path)["optional-dependencies"]
        assert set(alias_extras) == set(root_extras)
        for extra in root_extras:
            assert alias_extras[extra] == [f"m-star[{extra}]"]


def test_waypoint_extra_is_index_safe_and_taehv_pin_is_runtime_contract():
    requirements = {
        Requirement(raw).name
        for raw in _project(ROOT_PYPROJECT)["optional-dependencies"]["waypoint"]
    }
    assert {"huggingface-hub", "safetensors", "tensordict"} <= requirements
    assert "taehv" not in requirements
    assert TAEHV_UPSTREAM_REVISION in TAEHV_UPSTREAM_ARCHIVE


def test_waypoint_has_a_resolvable_packaged_default_config():
    assert DEFAULT_CONFIGS["waypoint"] == "waypoint.yaml"
    assert (ROOT / "configs" / DEFAULT_CONFIGS["waypoint"]).is_file()
    assert Path(_resolve_config("waypoint", None)).resolve() == (
        ROOT / "configs" / "waypoint.yaml"
    ).resolve()


@pytest.mark.parametrize("installed", [None, object()])
def test_missing_or_invalid_taehv_names_the_exact_separate_install(installed, monkeypatch):
    monkeypatch.setitem(sys.modules, "taehv", installed)
    with pytest.raises(RuntimeError) as error:
        require_taehv_runtime()
    message = str(error.value)
    assert ".[waypoint]" in message
    assert "--no-deps" in message
    assert f"taehv @ {TAEHV_UPSTREAM_ARCHIVE}" in message
    assert "uv>=0.4.0 or pip>=24.3" in message


def test_low_level_taehv_loader_names_the_exact_separate_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "taehv", None)
    with pytest.raises(RuntimeError) as error:
        load_taehv("unused")
    message = str(error.value)
    assert ".[waypoint]" in message
    assert "--no-deps" in message
    assert f"taehv @ {TAEHV_UPSTREAM_ARCHIVE}" in message
    assert "uv>=0.4.0 or pip>=24.3" in message
