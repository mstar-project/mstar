"""Engine knobs resolve as env var, then the deployment config's ``knobs:``
block, then the default; a malformed config block fails at startup."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.utils import knobs


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in knobs.KNOBS.values():
        monkeypatch.delenv(var, raising=False)
    knobs.configure(None)
    yield
    knobs.configure(None)


def test_env_then_config_then_default(monkeypatch):
    knobs.configure({"kv_plan_cache": 1, "ingraph_decode_rope": True})
    assert knobs.kv_plan_cache() == 1 and knobs.ingraph_decode_rope() is True
    assert knobs.kv_pinned_indptrs() is False, "unset in both: the default"
    monkeypatch.setenv("MSTAR_INGRAPH_DECODE_ROPE", "0")
    assert knobs.ingraph_decode_rope() is False, "a set env var wins"
    # scripts export an empty variable to mean "not set"
    monkeypatch.setenv("MSTAR_KV_PLAN_CACHE", "")
    assert knobs.kv_plan_cache() == 1, "an empty env var defers to the config"


@pytest.mark.parametrize("bad", [{"kv_plan_cahce": 1}, {"kv_plan_cache": "1"}])
def test_a_malformed_block_fails_the_startup_check(bad):
    with pytest.raises(ValueError):
        knobs.check(bad)


def test_the_in_graph_sampler_scatter_is_on_by_default(monkeypatch):
    assert knobs.sampler_ingraph_scatter() is True and knobs.sampler_slots() == 4096
    knobs.configure({"sampler_ingraph_scatter": False})
    assert knobs.sampler_ingraph_scatter() is False
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "1")
    assert knobs.sampler_ingraph_scatter() is True
