"""The ``zonos2`` extra must cover what ``mstar serve zonos2`` imports.

The default config builds the voice-clone speaker encoder, so a package it
imports but the extra omits stops the server at startup. This scans every
import in ``mstar/model/zonos2`` (lazy ones included) and checks each against
the core dependencies, the extra, or a documented exception.
"""
from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ZONOS2 = REPO / "mstar" / "model" / "zonos2"

# Import name -> distribution name, where they differ.
DIST = {"huggingface_hub": "huggingface-hub", "flashinfer": "flashinfer-python"}

# Imported, but deliberately not in the extra; each is documented in
# docs/installation.rst.
EXEMPT = {
    # pins protobuf<3.20, which would constrain every model; installed separately
    "dac": "descript-audio-codec",
    # optional text normalization, the zonos2-norm extra
    "nemo_text_processing": "zonos2-norm",
}


def _name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def _third_party_imports() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(ZONOS2.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                root = module.split(".")[0]
                if root in sys.stdlib_module_names or root in ("mstar", "__future__"):
                    continue
                found.setdefault(root, []).append(str(path.relative_to(REPO)))
    return found


def test_zonos2_extra_covers_every_import():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    provided = {_name(r) for r in project["dependencies"] + extras["zonos2"]}
    assert "zonos2-norm" in extras  # where nemo_text_processing comes from

    missing = {
        root: files for root, files in _third_party_imports().items()
        if root not in EXEMPT and DIST.get(root, root).replace("_", "-") not in provided
    }
    assert not missing, f"imported by zonos2 but not in its extra or core deps: {missing}"


def test_speaker_encoder_remote_code_needs_transformers():
    # The hub's remote code imports transformers too, which the AST scan cannot see.
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert "transformers" in {_name(r) for r in extras["zonos2"]}


def test_missing_transformers_names_the_extra_and_the_text_only_option(monkeypatch):
    from mstar.model.zonos2.speaker_encoder import Qwen3SpeakerEncoder

    monkeypatch.setitem(sys.modules, "transformers", None)  # import now raises
    with pytest.raises(ImportError, match=r"zonos2 extra.*speaker_encoder node group"):
        Qwen3SpeakerEncoder("unused", embedding_dim=4)


def test_missing_torchcodec_names_the_extra(monkeypatch):
    from mstar.model.zonos2.config import Zonos2Config
    from mstar.model.zonos2.zonos2_model import Zonos2Model

    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    model = Zonos2Model("unused", config=Zonos2Config(), skip_weight_loading=True)
    with pytest.raises(ImportError, match=r"torchcodec.*zonos2 extra"):
        model.load_audio("clip.wav", "cpu")
