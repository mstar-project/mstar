"""The Qwen3.5 graph-bucket ladders: the decode one from the environment, the
prefill one from the deployment's model_kwargs; and every prefill ladder a
shipped config sets, for whichever model it serves."""
import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.engine import recompile_limit
from mstar.utils import knobs


def _reload(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    import mstar.model.qwen3_5.submodules as m
    return importlib.reload(m)


def test_prefill_ladder_from_model_kwargs():
    from mstar.model.qwen3_5.config import LLM_MIXED
    from mstar.model.qwen3_5.submodules import LLMSubmodule
    default = list(LLMSubmodule.PREFILL_TOKEN_BUCKETS)
    sub = LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,)), prefill_token_buckets=[8192, 3072, 2048])
    assert sub.PREFILL_TOKEN_BUCKETS == [2048, 3072, 8192]
    # an instance override, not the class's
    assert LLMSubmodule.PREFILL_TOKEN_BUCKETS == default
    assert LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,))).PREFILL_TOKEN_BUCKETS == default
    # the ladders the text-prefill and mixed captures are taken on; a mixed
    # step keeps only the powers of two past 2176
    configs = sub.get_accelerator_graph_configs(torch.device("cpu"))
    ladders = {c.capture_graph_walk: c.capture_token_lengths for c in configs if c.capture_graph_walk != "decode"}
    assert ladders["prefill_text"] == [2048, 3072, 8192]
    assert ladders[LLM_MIXED] == [2048, 8192]


@pytest.mark.parametrize("buckets", [[], [512, 512], [0, 64], 2048])
def test_a_bad_prefill_ladder_is_refused(buckets):
    """Refused when the submodule is built, not taken as the default (an empty
    list), captured twice (a repeat), or failing later with a TypeError (a scalar)."""
    from mstar.model.qwen3_5.submodules import LLMSubmodule
    with pytest.raises(ValueError, match="prefill_token_buckets"):
        LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,)), prefill_token_buckets=buckets)


def test_decode_ladder_knob(monkeypatch):
    m = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", "1,2,4,8,16,32")
    assert m.LLMSubmodule.DECODE_CAPTURE_BATCH_SIZES == [1, 2, 4, 8, 16, 32]
    m = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", None)
    assert m.LLMSubmodule.DECODE_CAPTURE_BATCH_SIZES[-1] == 128


def test_prefill_buckets_hold_near_power_of_two_batches(monkeypatch):
    submodule = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", None).LLMSubmodule
    buckets = submodule.PREFILL_TOKEN_BUCKETS
    # a 16-row batch with a few prompts a couple of tokens long stays captured
    assert 16 * 512 + 8 <= buckets[-1]
    # a batch that fills a power of two doesn't pad past it
    assert {256, 512, 1024, 2048, 4096, 8192} <= set(buckets)


def _configs_with_prefill_ladders():
    """``{config: (model class, ladder)}`` for every shipped config that sets
    ``model_kwargs.prefill_token_buckets``."""
    import yaml

    from mstar.model.registry import get_model_class
    ladders = {}
    for path in sorted((Path(__file__).resolve().parents[2] / "configs").rglob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        buckets = (config.get("model_kwargs") or {}).get("prefill_token_buckets")
        if buckets is not None:
            ladders[path.name] = (get_model_class(config["model"]), buckets)
    return ladders


def _shipped_prefill_ladders():
    """None (the default) and every ``prefill_token_buckets`` a Qwen3.5 config sets."""
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    ladders = {str(None): None}
    for name, (cls, buckets) in _configs_with_prefill_ladders().items():
        if cls is Qwen3_5DenseModel:
            ladders[name] = buckets
    return ladders


@pytest.mark.parametrize("config", _configs_with_prefill_ladders())
def test_every_shipped_prefill_ladder_is_a_model_argument(config):
    """A config's ladder is a named argument of its model, not lost in ``**kwargs``;
    Qwen3-Omni's Thinker captures its prefill on it."""
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
    cls, buckets = _configs_with_prefill_ladders()[config]
    assert cls in (Qwen3_5DenseModel, Qwen3OmniModel), (
        f"a {cls.__name__} config sets prefill_token_buckets, which no test checks")
    assert "prefill_token_buckets" in inspect.signature(cls.__init__).parameters, (
        f"prefill_token_buckets isn't a named argument of {cls.__name__}.__init__")
    if cls is Qwen3OmniModel:
        from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
        node = ThinkerSubmodule(None, SimpleNamespace(im_end_token_id=1), prefill_token_buckets=buckets)
        configs = {c.capture_graph_walk: c for c in node.get_accelerator_graph_configs(torch.device("cpu"))}
        assert configs["prefill_text"].capture_token_lengths == sorted(buckets)


@pytest.mark.parametrize("buckets", _shipped_prefill_ladders().values(), ids=_shipped_prefill_ladders().keys())
def test_shipped_ladders_hold_a_mixed_step_without_the_next_power_of_two(buckets):
    """Four 512-token prompts beside the rest of the largest mixed batch replay
    at 2176, not padded to 4096, on every ladder a config ships."""
    from mstar.model.qwen3_5.submodules import LLMSubmodule
    s = LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,)), prefill_token_buckets=buckets)
    tokens = 4 * 512 + s.MIXED_CAPTURE_BATCH_SIZES[-1] - 4
    assert min(b for b in s.MIXED_TOKEN_BUCKETS if b >= tokens) == 2176


@pytest.mark.parametrize("buckets", _shipped_prefill_ladders().values(), ids=_shipped_prefill_ladders().keys())
def test_shipped_ladders_stay_inside_dynamos_recompile_limit(monkeypatch, buckets):
    monkeypatch.delenv("MSTAR_QWEN35_DECODE_BUCKETS", raising=False)
    monkeypatch.delenv("MSTAR_DECODE_BUCKETS", raising=False)
    m = _reload(monkeypatch, "MSTAR_QWEN35_DECODE_BUCKETS", None)
    s = m.LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,)), prefill_token_buckets=buckets)

    def pairs(rows, tokens):
        return sum(t >= r for r in rows for t in tokens)

    # one compile per decode bucket, per prefill bucket, and per (rows, tokens) pair of the
    # mixed and vision captures; past the limit the last buckets run uncompiled in their
    # graphs. The count is an estimate: a compile it doesn't model (another eager shape, a
    # guard failure) would leave the last bucket uncompiled, so one is kept spare.
    compiles = (
        len(s.DECODE_CAPTURE_BATCH_SIZES) + len(s.PREFILL_TOKEN_BUCKETS)
        + pairs(s.MIXED_CAPTURE_BATCH_SIZES, s.MIXED_TOKEN_BUCKETS)
        + pairs(s.PREFILL_VISION_CAPTURE_BATCH_SIZES, s.PREFILL_VISION_TOKEN_BUCKETS)
    )
    assert compiles < recompile_limit()


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
    assert methods == ["LLMSubmodule.get_accelerator_graph_configs"]
