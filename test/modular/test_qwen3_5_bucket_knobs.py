"""The Qwen3.5 graph-bucket ladders come from the environment."""
import importlib

import torch


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


def test_text_walks_compile_deterministically(monkeypatch):
    m = _reload(monkeypatch, "MSTAR_QWEN35_PREFILL_TOKEN_BUCKETS", None)
    submodule = m.LLMSubmodule.__new__(m.LLMSubmodule)
    configs = m.LLMSubmodule.get_accelerator_graph_configs(submodule, torch.device("cpu"))
    assert {c.capture_graph_walk for c in configs} >= {"decode", "prefill_text", "mixed"}
    assert {c.compile_mode for c in configs} == {"deterministic"}
