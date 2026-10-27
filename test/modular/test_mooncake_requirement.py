"""Every extra that installs Mooncake requires a release with the TCP data
port fix, and a worker on an older one says so.

Before 0.3.12.post1, any host that reached Mooncake's TCP data port could read
or write the process's memory (CVE-2026-103764). The extras list Mooncake
twelve times, so an unpinned one added later would quietly allow it again.
"""
import importlib.metadata
import logging
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

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
        def no_record(dist):
            raise importlib.metadata.PackageNotFoundError(dist)
        monkeypatch.setattr(importlib.metadata, "version", fake_version)
        # the real distributions' RECORDs would decide between the fake ones
        monkeypatch.setattr(importlib.metadata, "distribution", no_record)
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


@pytest.mark.parametrize("broken", [None, UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")])
def test_a_version_that_cannot_be_read_is_quiet(installed, caplog, monkeypatch, broken):
    """Stale or corrupt metadata beside a working engine must not stop it
    from starting: no Version field, or a METADATA that isn't UTF-8."""
    installed("0.3.11.post1")

    def version(dist):
        if isinstance(broken, Exception):
            raise broken
        return broken

    monkeypatch.setattr(importlib.metadata, "version", version)
    with caplog.at_level(logging.WARNING, logger=tensors.logger.name):
        tensors._warn_if_mooncake_unpatched()
    assert caplog.records == []


OLD, NEW = b"engine 0.3.11.post1", b"engine 0.3.13.post1"


def _engine_record(root, dist, content: bytes, on_disk: bytes):
    """A distribution under ``root`` whose RECORD hashes ``content`` for the
    engine, while the engine on disk holds ``on_disk``."""
    import base64
    import hashlib
    engine = root / "mooncake" / "engine.so"
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(on_disk)
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
    info = root / f"{dist.replace('-', '_')}-0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: {dist}\nVersion: 0\n")
    (info / "RECORD").write_text(f"mooncake/engine.so,sha256={digest},{len(content)}\n")
    return importlib.metadata.PathDistribution(info)


def _warn_with_engine(monkeypatch, caplog, dists, imported):
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: dists[name])
    monkeypatch.setitem(sys.modules, "mooncake.engine", SimpleNamespace(__file__=str(imported)))
    with caplog.at_level(logging.WARNING, logger=tensors.logger.name):
        tensors._warn_if_mooncake_unpatched()
    return [r.getMessage() for r in caplog.records]


def test_the_imported_engine_decides_between_installed_distributions(installed, caplog, monkeypatch, tmp_path):
    """A patched default wheel, then an unpatched non-CUDA one over it: both
    report a version, but the engine on disk is the old one."""
    installed("0.3.13.post1", **{"non-cuda": "0.3.11.post1"})
    dists = {
        "mooncake-transfer-engine": _engine_record(tmp_path, "mooncake-transfer-engine", NEW, OLD),
        "mooncake-transfer-engine-non-cuda": _engine_record(tmp_path, "mooncake-transfer-engine-non-cuda", OLD, OLD),
    }
    warnings = _warn_with_engine(monkeypatch, caplog, dists, tmp_path / "mooncake" / "engine.so")
    assert len(warnings) == 1
    assert "mooncake-transfer-engine-non-cuda 0.3.11.post1" in warnings[0]


def test_a_patched_engine_that_is_not_imported_does_not_count(installed, caplog, monkeypatch, tmp_path):
    """Each in its own site directory, both engines match their RECORDs; the
    old one comes first on the path, so it is the one running."""
    installed("0.3.13.post1", **{"non-cuda": "0.3.11.post1"})
    dists = {
        "mooncake-transfer-engine": _engine_record(tmp_path / "late", "mooncake-transfer-engine", NEW, NEW),
        "mooncake-transfer-engine-non-cuda": _engine_record(
            tmp_path / "early", "mooncake-transfer-engine-non-cuda", OLD, OLD),
    }
    warnings = _warn_with_engine(monkeypatch, caplog, dists, tmp_path / "early" / "mooncake" / "engine.so")
    assert len(warnings) == 1
    assert "mooncake-transfer-engine-non-cuda 0.3.11.post1" in warnings[0]


class _UnreadableRecord:
    @property
    def files(self):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")


def test_unreadable_records_warn_about_any_unpatched_one(installed, caplog, monkeypatch, tmp_path):
    """A corrupt RECORD must not stop the engine from starting; with no hash
    to compare, the unpatched one might be running, so it still warns."""
    installed("0.3.13.post1", **{"non-cuda": "0.3.11.post1"})
    engine = tmp_path / "engine.so"
    engine.write_bytes(OLD)
    dists = {name: _UnreadableRecord() for name in ("mooncake-transfer-engine", "mooncake-transfer-engine-non-cuda")}
    warnings = _warn_with_engine(monkeypatch, caplog, dists, engine)
    assert len(warnings) == 1
    assert "none can be matched to the imported engine" in warnings[0]
    assert "mooncake-transfer-engine-non-cuda 0.3.11.post1" in warnings[0]
    assert "CVE-2026-103764" in warnings[0]


def test_two_patched_distributions_with_no_record_match_are_quiet(installed, caplog):
    installed("0.3.12.post1", cuda13="0.3.13.post1")
    with caplog.at_level(logging.WARNING, logger=tensors.logger.name):
        tensors._warn_if_mooncake_unpatched()
    assert caplog.records == []


def test_a_patched_distribution_beside_stale_metadata_is_quiet(installed, caplog, monkeypatch, tmp_path):
    """Swapping the default wheel for the CUDA 13 one can leave the old
    metadata behind; the engine on disk matches the CUDA 13 RECORD."""
    installed("0.3.11.post1", cuda13="0.3.13.post1")
    dists = {
        "mooncake-transfer-engine": _engine_record(tmp_path, "mooncake-transfer-engine", OLD, NEW),
        "mooncake-transfer-engine-cuda13": _engine_record(tmp_path, "mooncake-transfer-engine-cuda13", NEW, NEW),
    }
    assert _warn_with_engine(monkeypatch, caplog, dists, tmp_path / "mooncake" / "engine.so") == []


def test_stale_metadata_with_no_record_match_warns(installed, caplog):
    """The same swap with no RECORD to compare: the old metadata could be
    the engine in use, so it warns rather than trusting the patched one."""
    installed("0.3.11.post1", cuda13="0.3.13.post1")
    with caplog.at_level(logging.WARNING, logger=tensors.logger.name):
        tensors._warn_if_mooncake_unpatched()
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert "mooncake-transfer-engine 0.3.11.post1" in warnings[0]

