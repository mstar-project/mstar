"""The Qwen3.5 graph-bucket ladders: the decode one from the environment, the
prefill one from the deployment's model_kwargs."""
import importlib
from types import SimpleNamespace

import pytest
import torch


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
    sub = LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,)), prefill_token_buckets=[8192, 3072])
    assert sub.PREFILL_TOKEN_BUCKETS == [3072, 8192]
    # an instance override, not the class's
    assert LLMSubmodule.PREFILL_TOKEN_BUCKETS == default
    assert LLMSubmodule(None, SimpleNamespace(stop_token_ids=(1,))).PREFILL_TOKEN_BUCKETS == default
    # the ladder the text-prefill and mixed captures are taken on
    configs = sub.get_accelerator_graph_configs(torch.device("cpu"))
    ladders = {c.capture_graph_walk: c.capture_token_lengths for c in configs if c.capture_graph_walk != "decode"}
    assert ladders["prefill_text"] == ladders[LLM_MIXED] == [3072, 8192]


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
