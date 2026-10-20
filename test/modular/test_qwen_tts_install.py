"""qwen-tts stays out of the extras and is installed with ``--no-deps``.

qwen-tts 0.1.1 pins ``transformers==4.57.3``, and through it
``huggingface-hub<1.0``. diffusers 0.40 and later need ``huggingface-hub>=1.23``
and OmniVoice needs ``transformers>=5.3``, so an extra that lists qwen-tts either
pins every model in the environment to that set or, installed after them,
downgrades their dependencies. mstar reads only qwen-tts' speech-tokenizer
source files and runs them under transformers 4.57.3 or 5.x.
"""
import importlib.metadata
import tomllib
from pathlib import Path

import pytest

from mstar.model.qwen3_tts import qwen3_tts_model

pytest.importorskip("packaging")
from packaging.requirements import Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

#: Packages whose own metadata pins transformers to one release.
PINS_TRANSFORMERS = {"qwen-tts": "transformers==4.57.3"}


def _offending(pyproject: str) -> list[str]:
    project = tomllib.loads(pyproject)["project"]
    groups = {"dependencies": project.get("dependencies", [])}
    groups.update(project.get("optional-dependencies", {}))
    return [
        f"[{group}] {line} (it pins {PINS_TRANSFORMERS[canonicalize_name(Requirement(line).name)]})"
        for group, reqs in groups.items()
        for line in reqs
        if canonicalize_name(Requirement(line).name) in PINS_TRANSFORMERS
    ]


def test_no_extra_installs_a_package_that_pins_transformers():
    assert _offending((ROOT / "pyproject.toml").read_text()) == []


def test_the_check_sees_any_spelling():
    text = """
[project]
dependencies = []
[project.optional-dependencies]
tts = ["Qwen_TTS==0.1.1"]
"""
    assert len(_offending(text)) == 1


@pytest.fixture
def no_qwen_tts(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    qwen3_tts_model._load_qwen3_tts_codec_classes.cache_clear()
    yield
    qwen3_tts_model._load_qwen3_tts_codec_classes.cache_clear()


def test_a_missing_qwen_tts_names_the_install_command(no_qwen_tts):
    with pytest.raises(ImportError) as err:
        qwen3_tts_model._load_qwen3_tts_codec_classes()
    assert qwen3_tts_model.QWEN_TTS_INSTALL in str(err.value)


def test_the_docs_show_the_same_command():
    assert qwen3_tts_model.QWEN_TTS_INSTALL in (ROOT / "docs" / "installation.rst").read_text()


@pytest.mark.parametrize("name", ["Dockerfile", "Dockerfile.dynamo-base"])
def test_the_docker_images_install_qwen_tts(name):
    """Both images install ``.[all]``, which does not carry qwen-tts."""
    dockerfile = (ROOT / "mstar" / "integrations" / "dynamo" / "docker" / name).read_text()
    assert '--no-deps "qwen-tts==0.1.1"' in dockerfile


def test_all_keeps_the_transformers_floor_qwen3_tts_needs():
    """qwen-tts is installed --no-deps, so nothing else holds transformers at
    the 4.57.3 its speech tokenizer needs; an older one fails to load it."""
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]

    def floor(extra):
        (req,) = [Requirement(r) for r in extras[extra] if canonicalize_name(Requirement(r).name) == "transformers"]
        return req.specifier

    for extra in ("all", "qwen3_tts"):
        assert not floor(extra).contains("4.57.2"), extra
        assert floor(extra).contains("4.57.3") and floor(extra).contains("5.19.0"), extra
