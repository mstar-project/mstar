"""Dependency contracts for CUDA compatibility and a resolvable XPU install."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]


def _project(path=ROOT / "pyproject.toml"):
    with path.open("rb") as file:
        return tomllib.load(file)["project"]


def _requirements(extra):
    project = _project()
    pending = [extra]
    visited = set()
    requirements = [Requirement(raw) for raw in project["dependencies"]]
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for raw in project["optional-dependencies"][name]:
            requirement = Requirement(raw)
            if requirement.name == project["name"]:
                pending.extend(requirement.extras)
            else:
                requirements.append(requirement)
    return requirements


def test_xpu_backend_pins_intel_builds_compatible_with_base_and_bagel():
    backend = [Requirement(raw) for raw in _project()["optional-dependencies"]["xpu"]]
    combined = _requirements("bagel_xpu")
    for package in ("torch", "torchvision"):
        pin = next(requirement for requirement in backend if requirement.name == package)
        version = next(spec.version for spec in pin.specifier if spec.operator == "==")
        assert Version(version).local == "xpu", "select Intel wheels explicitly"
        assert all(
            Version(version) in requirement.specifier
            for requirement in combined if requirement.name == package
        ), f"{package}'s XPU build conflicts with another dependency"


def test_xpu_bagel_does_not_pull_cuda_providers_or_unused_audio_codecs():
    names = {canonicalize_name(requirement.name) for requirement in _requirements("bagel_xpu")}
    assert "vllm-xpu-kernels" in names
    assert not names & {"flashinfer-python", "fa3-fwd", "triton", "torchaudio", "torchcodec"}


@pytest.mark.parametrize("extra", ["bagel", "all", "qwen3_omni", "waypoint", "orpheus", "wan22"])
def test_existing_model_extras_keep_pytorch_bounds_and_media_dependencies(extra):
    requirements = _requirements(extra)
    torch_requirements = [requirement for requirement in requirements if requirement.name == "torch"]
    assert all(Version("2.9.1") in requirement.specifier for requirement in torch_requirements)
    assert all(Version("2.12.0") in requirement.specifier for requirement in torch_requirements)
    assert not all(Version("2.13.0") in requirement.specifier for requirement in torch_requirements)
    assert {"torchvision", "torchaudio"} <= {requirement.name for requirement in requirements}
    assert "vllm-xpu-kernels" not in {requirement.name for requirement in requirements}


@pytest.mark.parametrize("alias", ["mstar-project", "mstar-serve"])
def test_aliases_forward_backend_and_xpu_model_extras(alias):
    extras = _project(ROOT / "packaging" / "aliases" / alias / "pyproject.toml")["optional-dependencies"]
    for extra in ("cuda", "xpu", "bagel_xpu"):
        assert extras[extra] == [f"mstar-ai[{extra}]"]
