"""How Chatterbox's watermark loads ``resemble-perth`` (no weights, CPU only)."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from mstar.model.chatterbox.components.watermark import PerthWatermarker

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_a_missing_package_serves_unwatermarked(monkeypatch):
    monkeypatch.setitem(sys.modules, "perth", None)  # import raises ImportError
    assert PerthWatermarker.build("cpu") is None


def test_a_package_that_hid_its_watermarker_fails_with_the_fix(monkeypatch):
    """resemble-perth swallows the ImportError of its watermarker module (it
    needs pkg_resources) and leaves ``PerthImplicitWatermarker = None``."""
    fake = ModuleType("perth")
    fake.PerthImplicitWatermarker = None
    monkeypatch.setitem(sys.modules, "perth", fake)
    with pytest.raises(RuntimeError, match=r"setuptools<82"):
        PerthWatermarker.build("cpu")


def test_the_chatterbox_extra_keeps_pkg_resources():
    extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert "resemble-perth" in extras["chatterbox"]
    # setuptools 82 removed pkg_resources, which resemble-perth 1.0.1 imports
    assert "setuptools<82" in extras["chatterbox"]
