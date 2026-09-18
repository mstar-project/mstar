"""CPU import contract for the glm5_next package — the ``registry`` tier of
``env/ci_glm53.sh``.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The modules whose import must not touch a GPU-only dependency.
CPU_CONTRACT_MODULES = ("kda", "mhc", "config", "kda_state", "weight_loader")

_HARD_TIER_PROBE = """
import sys
for m in {modules!r}:
    __import__(f"mstar.model.glm5_next.{{m}}")
bad = [m for m in ("flashinfer", "triton") if m in sys.modules]
assert not bad, f"CPU-contract modules pulled {{bad}}"
print("ok")
"""


def test_cpu_contract_modules_import_without_gpu_deps():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("HF_HUB_OFFLINE", "1")
    proc = subprocess.run(
        [sys.executable, "-c", _HARD_TIER_PROBE.format(modules=CPU_CONTRACT_MODULES)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
        check=False,
    )
    assert proc.returncode == 0, (
        f"bare-interpreter import of {CPU_CONTRACT_MODULES} failed:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )
    assert proc.stdout.strip().endswith("ok")


def _import_or_skip(module_name: str):
    """Import; skip on a missing third-party module, fail on a missing mstar one."""
    try:
        return __import__(module_name, fromlist=["_"])
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "mstar":
            raise
        pytest.skip(f"{module_name} needs {e.name!r} (box venvs have it)")


def test_full_model_module_imports_under_stubs():
    glm = _import_or_skip("mstar.model.glm5_next.glm5_next_model")
    assert hasattr(glm, "Glm5NextModel")


def test_registry_entry_resolves_lazily():
    from mstar.model import registry

    # The lazy tuple form: the registry names the module and class without
    # importing them, so listing models never pulls a model's deps.
    assert registry.MODEL_REGISTRY["glm5_next"] == (
        "mstar.model.glm5_next.glm5_next_model", "Glm5NextModel",
    )
    assert registry.HF_MODELS["glm5_next"]["model_path_hf"] == "zai-org/GLM-5.3-Flash"

    glm = _import_or_skip("mstar.model.glm5_next.glm5_next_model")
    assert registry.get_model_class("glm5_next") is glm.Glm5NextModel
