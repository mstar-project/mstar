"""Every extra that installs Mooncake requires a release with the TCP data
port fix, and a worker on an older one says so.

Before 0.3.12.post1, any host that reached Mooncake's TCP data port could read
or write the process's memory (CVE-2026-103764). The extras list Mooncake
twelve times, so an unpinned one added later would quietly allow it again.
"""
import importlib.metadata
import logging
import tomllib
from pathlib import Path

import pytest

from mstar.communication import tensors

pytest.importorskip("packaging")
from packaging.requirements import Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402
from packaging.version import Version  # noqa: E402

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
FLOORS = (">=", ">", "==", "~=")


def _mooncake_requirements(pyproject: str):
    project = tomllib.loads(pyproject)["project"]
    groups = {"dependencies": project.get("dependencies", [])}
    groups.update(project.get("optional-dependencies", {}))
    for group, reqs in groups.items():
        for line in reqs:
            req = Requirement(line)
            if canonicalize_name(req.name).startswith("mooncake-transfer-engine"):
                yield group, req


def _problems(pyproject: str) -> list[str]:
    patched = Version(tensors.MOONCAKE_PATCHED_VERSION)
    problems = []
    for group, req in _mooncake_requirements(pyproject):
        floors = [Version(s.version) for s in req.specifier if s.operator in FLOORS]
        if not floors or max(floors) < patched:
            problems.append(f"[{group}] {req} allows Mooncake releases before {patched}")
        if not req.specifier.contains("0.3.13.post1"):
            problems.append(f"[{group}] {req} excludes 0.3.13.post1, the release CI tests")
    return problems


def test_every_mooncake_requirement_excludes_unpatched_releases():
    text = PYPROJECT.read_text()
    assert list(_mooncake_requirements(text)), "no extra installs Mooncake any more; drop this test"
    assert _problems(text) == []


def test_the_check_catches_an_unpinned_extra_in_any_spelling():
    text = """
[project]
dependencies = []
[project.optional-dependencies]
new_model = ["Mooncake_Transfer.Engine", "mooncake-transfer-engine-cuda13>=0.3.10"]
"""
    assert len(_problems(text)) == 2


@pytest.fixture
def installed(monkeypatch):
    """Pretend ``mooncake-transfer-engine`` is installed at ``version``, plus
    any other distributions given as ``name=version``."""
    def install(version, **others):
        dists = {"mooncake-transfer-engine": version}
        dists.update({f"mooncake-transfer-engine-{k}": v for k, v in others.items()})

        def fake_version(dist):
            if dist in dists:
                return dists[dist]
            raise importlib.metadata.PackageNotFoundError(dist)
        monkeypatch.setattr(importlib.metadata, "version", fake_version)
        monkeypatch.setattr(tensors, "_mooncake_version_checked", False)
    return install


def test_an_unpatched_mooncake_warns_once(installed, caplog):
    installed("0.3.11.post1")
    with caplog.at_level(logging.WARNING, logger=tensors.logger.name):
        tensors._warn_if_mooncake_unpatched()
        tensors._warn_if_mooncake_unpatched()
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert "mooncake-transfer-engine 0.3.11.post1" in warnings[0]
    assert "CVE-2026-103764" in warnings[0]


@pytest.mark.parametrize("version", ["0.3.12.post1", "0.3.13.post1", "0.3.14rc1", "not-a-version"])
def test_a_patched_or_unreadable_version_is_quiet(installed, caplog, version):
    installed(version)
    with caplog.at_level(logging.WARNING, logger=tensors.logger.name):
        tensors._warn_if_mooncake_unpatched()
    assert caplog.records == []


def test_a_patched_distribution_beside_stale_metadata_is_quiet(installed, caplog):
    """Swapping the default wheel for the CUDA 13 one can leave the old
    metadata behind; the ``mooncake`` package on disk is the patched one."""
    installed("0.3.11.post1", cuda13="0.3.13.post1")
    with caplog.at_level(logging.WARNING, logger=tensors.logger.name):
        tensors._warn_if_mooncake_unpatched()
    assert caplog.records == []

