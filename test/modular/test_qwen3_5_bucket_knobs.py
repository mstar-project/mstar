"""The Qwen3.5 graph-bucket ladders come from the environment."""
import ast
import importlib
from pathlib import Path

import pytest
import torch

from mstar.utils import knobs


def _reload(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    import mstar.model.qwen3_5.submodules as m
    return importlib.reload(m)


def test_prefill_ladder_default_reaches_8k_and_the_knob_narrows_it(monkeypatch):
    m = _reload(monkeypatch, "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", None)
    assert m.LLMSubmodule.PREFILL_TOKEN_BUCKETS[-1] == 8704
    assert m.LLMSubmodule.PREFILL_TOKEN_BUCKETS[0] == 32
    m = _reload(monkeypatch, "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", "64,2048")
    assert m.LLMSubmodule.PREFILL_TOKEN_BUCKETS == [64, 2048]
    m = _reload(monkeypatch, "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", None)
    assert len(m.LLMSubmodule.PREFILL_TOKEN_BUCKETS) == 9


def test_decode_ladder_knob(monkeypatch):
    m = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", "1,2,4,8,16,32")
    assert m.LLMSubmodule.DECODE_CAPTURE_BATCH_SIZES == [1, 2, 4, 8, 16, 32]
    m = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", None)
    assert m.LLMSubmodule.DECODE_CAPTURE_BATCH_SIZES[-1] == 128


def test_prefill_buckets_hold_a_mixed_step_and_near_power_of_two_batches(monkeypatch):
    buckets = _reload(monkeypatch, "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", None).LLMSubmodule.PREFILL_TOKEN_BUCKETS
    # four 512-token prompts beside the largest decode bucket's rows, without the next bucket up
    assert min(b for b in buckets if b >= 4 * 512 + 124) == 2176
    # a 16-row batch with a few prompts a couple of tokens long stays captured
    assert 16 * 512 + 8 <= buckets[-1]


def _compile_modes(m):
    submodule = m.LLMSubmodule.__new__(m.LLMSubmodule)
    configs = m.LLMSubmodule.get_accelerator_graph_configs(submodule, torch.device("cpu"))
    return {c.capture_graph_walk: c.compile_mode for c in configs}


@pytest.fixture
def clean_knobs(monkeypatch):
    monkeypatch.delenv("MSTAR_COMPILE_DETERMINISTIC", raising=False)
    knobs.configure(None)
    yield
    knobs.configure(None)


def test_decode_compiles_tuned_and_prefill_deterministically_by_default(monkeypatch, clean_knobs):
    modes = _compile_modes(_reload(monkeypatch, "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", None))
    assert modes == {
        "decode": None, "prefill_text": "deterministic", "mixed": "deterministic", "prefill_vision": "deterministic",
    }


def test_the_switch_makes_every_text_walk_deterministic(monkeypatch, clean_knobs):
    m = _reload(monkeypatch, "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", None)
    knobs.configure({"compile_deterministic": True})
    assert set(_compile_modes(m).values()) == {"deterministic"}
    # the environment variable wins over the config, as for every knob
    monkeypatch.setenv("MSTAR_COMPILE_DETERMINISTIC", "0")
    assert _compile_modes(m)["decode"] is None


def test_only_qwen3_5_text_walks_read_the_switch():
    root = Path(__file__).resolve().parents[2]
    readers = sorted(
        str(p.relative_to(root)) for p in (root / "mstar").rglob("*.py")
        if "compile_deterministic" in p.read_text() and p.parts[-2:] != ("utils", "knobs.py")
    )
    assert readers == ["mstar/model/qwen3_5/submodules.py"]

    tree = ast.parse((root / readers[0]).read_text())
    methods = [
        f"{cls.name}.{fn.name}"
        for cls in tree.body if isinstance(cls, ast.ClassDef)
        for fn in cls.body if isinstance(fn, ast.FunctionDef)
        and any(isinstance(n, ast.Name) and n.id == "compile_deterministic" for n in ast.walk(fn))
    ]
    assert methods == ["LLMSubmodule.get_accelerator_graph_configs"]
