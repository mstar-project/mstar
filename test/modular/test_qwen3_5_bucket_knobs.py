"""The Qwen3.5 graph-bucket ladders come from the environment."""
import importlib


def _reload(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    import mstar.model.qwen3_5.submodules as m
    return importlib.reload(m)


def test_prefill_ladder_default_reaches_8k_and_the_knob_narrows_it(monkeypatch):
    m = _reload(monkeypatch, "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", None)
    assert m.LLMSubmodule.PREFILL_TOKEN_BUCKETS[-1] == 8192
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
