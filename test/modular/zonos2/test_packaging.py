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


# -- supply chain: pinned hub revisions, tensor-only checkpoint loads ----------
class _Payload:
    """A pickled object that is not a tensor; weights_only must refuse it."""


def test_checkpoint_load_accepts_tensors_and_refuses_other_objects(tmp_path):
    import torch

    from mstar.model.zonos2.weight_loader import load_zonos2_state_dict

    weights = {"a.weight": torch.ones(2)}
    torch.save({"model": weights}, tmp_path / "model.pth")
    assert torch.equal(load_zonos2_state_dict(str(tmp_path))["a.weight"], weights["a.weight"])

    torch.save({"a.weight": torch.ones(2), "extra": _Payload()}, tmp_path / "model.pth")
    with pytest.raises(Exception, match="Weights only load failed"):
        load_zonos2_state_dict(str(tmp_path))


def test_checkpoint_download_is_pinned(monkeypatch):
    import huggingface_hub

    from mstar.model.zonos2.config import Zonos2Config
    from mstar.model.zonos2.weight_loader import resolve_zonos2_checkpoint

    seen = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda repo, **kw: seen.update(kw, repo=repo) or "/snap",
    )
    assert resolve_zonos2_checkpoint("Zyphra/ZONOS2", revision=Zonos2Config.checkpoint_revision) == "/snap"
    assert seen["revision"] == Zonos2Config.checkpoint_revision
    assert len(Zonos2Config.checkpoint_revision) == 40             # a full commit sha


@pytest.mark.parametrize("override", [None, "main"])
def test_config_load_downloads_the_pinned_or_overridden_checkpoint(monkeypatch, override):
    from mstar.model.zonos2 import weight_loader
    from mstar.model.zonos2.config import Zonos2Config
    from mstar.model.zonos2.zonos2_model import Zonos2Model

    seen = []
    monkeypatch.setattr(
        weight_loader, "resolve_zonos2_checkpoint",
        lambda path, cache_dir=None, revision=None: seen.append(revision) or "/ckpt",
    )
    monkeypatch.setattr(
        weight_loader, "load_zonos2_config_from_checkpoint",
        lambda ckpt, **overrides: Zonos2Config(**overrides),
    )
    kwargs = {} if override is None else {"checkpoint_revision": override}
    Zonos2Model("Zyphra/ZONOS2", **kwargs)
    assert seen == [override or Zonos2Config.checkpoint_revision]


def test_speaker_encoder_loads_the_pinned_revision(monkeypatch):
    import transformers
    from torch import nn

    from mstar.model.zonos2.config import Zonos2Config
    from mstar.model.zonos2.speaker_encoder import Qwen3SpeakerEncoder

    seen = {}

    def fake_from_pretrained(model_id, **kw):
        seen.update(kw)
        return nn.Identity()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_from_pretrained)
    Qwen3SpeakerEncoder(
        "m/id", embedding_dim=4, revision=Zonos2Config.speaker_encoder_revision,
    )
    assert seen["revision"] == Zonos2Config.speaker_encoder_revision
    assert seen["trust_remote_code"] is True
    assert len(Zonos2Config.speaker_encoder_revision) == 40
