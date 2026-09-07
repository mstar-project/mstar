"""Shared fixtures: locate the tiny random-weight Kimi-K3 checkpoint and import its HF
modeling code by path (the real checkpoint's modeling files are byte-for-byte the same
math, only formatting differs)."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

TINY_DIR = Path(os.environ.get(
    "KIMI_K3_TINY_DIR", "/scratch/m000137-pm06/atj10/kimi_k3_mstar/ref/tiny_kimi_k3"))
FULL_CFG_DIR = Path(os.environ.get(
    "KIMI_K3_FULL_CFG_DIR", "/scratch/m000137-pm06/atj10/kimi_k3_mstar/ref/hf_kimi_k3"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def tiny_dir() -> Path:
    if not (TINY_DIR / "config.json").exists():
        pytest.skip(f"tiny Kimi-K3 checkpoint not found at {TINY_DIR}")
    return TINY_DIR


@pytest.fixture(scope="session")
def hf_modeling(tiny_dir):
    """The HF modeling module of the tiny checkpoint, importable on CPU (fla only fails
    when a Triton kernel is launched)."""
    pkg = "kimi_k3_hf_ref"
    if tiny_dir not in [Path(p) for p in sys.path]:
        sys.path.insert(0, str(tiny_dir))
    # the modeling file does `from .configuration_kimi_k3 import ...`, so build a package
    import types
    package = types.ModuleType(pkg)
    package.__path__ = [str(FULL_CFG_DIR if (FULL_CFG_DIR / "modeling_kimi_linear.py").exists() else tiny_dir)]
    sys.modules[pkg] = package
    # The tiny checkpoint's copy imports a transformers path that moved in 4.57; the full
    # checkpoint's modeling file is the same math with current imports, so prefer it.
    code_dir = FULL_CFG_DIR if (FULL_CFG_DIR / "modeling_kimi_linear.py").exists() else tiny_dir
    # transformers 4.57 changed `check_model_inputs` to a decorator factory; the modeling
    # file (written for 4.56) applies it bare. Shim it to accept both forms.
    from transformers.utils import generic as _generic
    _orig = _generic.check_model_inputs
    if not getattr(_orig, "_k3_shim", False):
        def _shim(func=None, **kw):
            if callable(func):
                return _orig(**kw)(func)
            return _orig(func, **kw) if func is not None else _orig(**kw)
        _shim._k3_shim = True
        _generic.check_model_inputs = _shim
    _load_module(f"{pkg}.configuration_kimi_k3", code_dir / "configuration_kimi_k3.py")
    src = code_dir / ("modeling_kimi_linear.py" if code_dir == FULL_CFG_DIR else "modeling_kimi_k3_linear.py")
    mod = _load_module(f"{pkg}.modeling_kimi_linear", src)
    return mod


@pytest.fixture(scope="session")
def hf_text_config(tiny_dir, hf_modeling):
    import json
    cfg_mod = sys.modules["kimi_k3_hf_ref.configuration_kimi_k3"]
    raw = json.load(open(tiny_dir / "config.json"))
    tc = dict(raw["text_config"])
    tc.pop("auto_map", None)
    cfg = cfg_mod.KimiLinearConfig(**tc)
    cfg._attn_implementation = "eager"
    return cfg
