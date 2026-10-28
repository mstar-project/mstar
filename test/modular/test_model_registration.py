"""Every registered model is reachable: CLI default config, and a docs row.

A model can be entirely correct and still unusable because one of the
registration sites was missed. These checks are pure filesystem and string work
-- no model module is imported -- so they run on CPU with no weights.
"""
import re
from pathlib import Path

from mstar.cli.main import DEFAULT_CONFIGS
from mstar.model.registry import HF_MODELS, MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[2]


def _default_configs() -> dict[str, str]:
    return DEFAULT_CONFIGS


def _expand_braces(name: str) -> list[str]:
    """``qwen3_5_{0.8,2}b`` -> ``qwen3_5_0.8b``, ``qwen3_5_2b``."""
    m = re.search(r"\{([^}]*)\}", name)
    if not m:
        return [name]
    return [
        expanded
        for alt in m.group(1).split(",")
        for expanded in _expand_braces(name[: m.start()] + alt + name[m.end():])
    ]


def _documented_models() -> set[str]:
    """Every name in the first cell of a docs/models.rst table row; a row may
    list several (``a`` / ``b``) or a brace pattern."""
    names = set()
    for cell in re.findall(r"^\s*\* - (.*)$", (ROOT / "docs/models.rst").read_text(), re.M):
        for name in re.findall(r"``([^`]+)``", cell):
            names.update(_expand_braces(name))
    return names


def test_registry_entries_resolve_on_disk():
    """Each (module, class) tuple names a file that defines that class."""
    for name, (module, cls) in MODEL_REGISTRY.items():
        path = ROOT / (module.replace(".", "/") + ".py")
        assert path.is_file(), f"{name}: no module at {module}"
        assert re.search(rf"^class {cls}\b", path.read_text(), re.M), (
            f"{name}: {module} does not define {cls}"
        )


def test_every_model_has_a_cli_default_config():
    """Without a DEFAULT_CONFIGS entry, `mstar serve <name>` exits 'unknown model'."""
    configs = _default_configs()
    missing = sorted(set(MODEL_REGISTRY) - set(configs))
    assert not missing, f"add to DEFAULT_CONFIGS in mstar/cli/main.py: {missing}"
    # The CLI namespace may hold deployment aliases beyond the registry, but
    # every config it names has to exist.
    for name, config in configs.items():
        assert (ROOT / "configs" / config).is_file(), f"{name}: missing configs/{config}"


def test_every_model_is_documented():
    missing = sorted(set(MODEL_REGISTRY) - _documented_models())
    assert not missing, f"add a row to docs/models.rst: {missing}"


def test_hf_models_are_registered():
    unknown = sorted(set(HF_MODELS) - set(MODEL_REGISTRY))
    assert not unknown, f"HF_MODELS names unregistered models: {unknown}"
