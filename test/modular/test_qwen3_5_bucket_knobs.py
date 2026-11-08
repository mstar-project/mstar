"""The Qwen3.5 graph-bucket ladders: the decode one from the environment, and
every prefill ladder a shipped config sets."""
import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.engine import recompile_limit
from mstar.engine.accelerator_graph_config import PackedAcceleratorGraphConfig
from mstar.utils import knobs


def _reload(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    import mstar.model.qwen3_5.submodules as m
    return importlib.reload(m)


def test_decode_ladder_knob(monkeypatch):
    m = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", "1,2,4,8,16,32")
    assert m.LLMSubmodule.DECODE_CAPTURE_BATCH_SIZES == [1, 2, 4, 8, 16, 32]
    m = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", None)
    assert m.LLMSubmodule.DECODE_CAPTURE_BATCH_SIZES[-1] == 128


def test_default_prefill_ladder_holds_every_power_of_two(monkeypatch):
    submodule = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", None).LLMSubmodule
    # a batch that fills a power of two doesn't pad past it
    assert {256, 512, 1024, 2048, 4096, 8192} <= set(submodule.PREFILL_TOKEN_BUCKETS)


def _shipped_prefill_ladders():
    """None (the default) and the ``prefill_token_buckets`` of every shipped
    config that sets them on a Qwen3.5 LLM."""
    import yaml

    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    from mstar.model.registry import get_model_class
    ladders = {str(None): None}
    for path in sorted((Path(__file__).resolve().parents[2] / "configs").rglob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        llm = (config.get("prefill_token_buckets") or {}).get("LLM")
        if llm is not None and issubclass(get_model_class(config["model"]), Qwen3_5DenseModel):
            ladders[path.name] = llm
    return ladders


def _captured_ladders(submodule, ladders) -> dict[str, PackedAcceleratorGraphConfig]:
    """Each packed capture walk's config once the engine has applied the
    serving config's ``ladders`` under the LLM's budget."""
    from mstar.engine.accelerator_graph_config import apply_token_ladders
    configs = submodule.get_accelerator_graph_configs(torch.device("cpu"))
    if ladders:
        # at the default min_chunk_tokens no step passes the budget
        apply_token_ladders(
            "LLM", configs, ladders, lambda walk, max_rows: submodule.MAX_BATCH_TOKENS,
            lambda config: config.capture_batch_sizes,
        )
    return {c.capture_graph_walk: c for c in configs if isinstance(c, PackedAcceleratorGraphConfig)}


@pytest.mark.parametrize("ladders", _shipped_prefill_ladders().values(), ids=_shipped_prefill_ladders().keys())
def test_shipped_ladders_stay_inside_dynamos_recompile_limit(monkeypatch, ladders):
    monkeypatch.delenv("MSTAR_QWEN35_DECODE_BUCKETS", raising=False)
    monkeypatch.delenv("MSTAR_DECODE_BUCKETS", raising=False)
    m = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", None)
    s = m.LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,)))
    captured = _captured_ladders(s, ladders)

    def pairs(rows, tokens):
        return sum(t >= r for r in rows for t in tokens)

    # one compile per decode bucket, per prefill bucket, and per (rows, tokens) pair of the
    # mixed and vision captures; past the limit the last buckets run uncompiled in their
    # graphs. The count is an estimate: a compile it doesn't model (another eager shape, a
    # guard failure) would leave the last bucket uncompiled, so one is kept spare.
    compiles = (
        len(s.DECODE_CAPTURE_BATCH_SIZES) + len(captured["prefill_text"].get_total_tokens(1))
        + sum(len(captured["mixed"].get_total_tokens(r)) for r in s.MIXED_CAPTURE_BATCH_SIZES)
        + pairs(s.PREFILL_MEDIA_CAPTURE_BATCH_SIZES, s.PREFILL_MEDIA_TOKEN_BUCKETS)
    )
    assert compiles < recompile_limit()


def _compile_modes(m):
    from mstar.model.multimodal import vision_encoder

    submodule = m.LLMSubmodule.__new__(m.LLMSubmodule)
    submodule._encoders = (vision_encoder((5,)),)
    configs = m.LLMSubmodule.get_accelerator_graph_configs(submodule, torch.device("cpu"))
    return {c.capture_graph_walk: c.compile_mode for c in configs}


@pytest.fixture
def clean_knobs(monkeypatch):
    monkeypatch.delenv("MSTAR_COMPILE_DETERMINISTIC", raising=False)
    knobs.configure(None)
    yield
    knobs.configure(None)


def test_decode_compiles_tuned_and_prefill_deterministically_by_default(monkeypatch, clean_knobs):
    modes = _compile_modes(_reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", None))
    assert modes == {
        "decode": None, "prefill_text": "deterministic", "mixed": "deterministic", "prefill_vision": "deterministic",
    }


def test_the_switch_makes_every_text_walk_deterministic(monkeypatch, clean_knobs):
    m = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", None)
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
    assert methods == ["LLMSubmodule.capture_compile_mode"]
