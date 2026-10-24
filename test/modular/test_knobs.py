"""Engine knobs resolve as env var, then the deployment config's ``knobs:``
block, then the default; a misspelled config key fails at startup."""

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


def test_defaults_without_config_or_env():
    assert knobs.device_loopback_enabled() is True
    assert knobs.sampler_ingraph_scatter() is False
    assert knobs.sampler_slots() == 4096
    assert knobs.ingraph_decode_rope() is False
    assert knobs.kv_chain_lazy_steps() == 1
    assert knobs.kv_plan_cache() == 0
    assert knobs.kv_pinned_indptrs() is False
    assert knobs.gc_gen0_threshold() == 0


def test_config_sets_knobs():
    knobs.configure({
        "device_loopback": False, "ingraph_decode_rope": True,
        "kv_plan_cache": 2, "sampler_slots": 512, "kv_chain_lazy_steps": 16,
        "gc_gen0_threshold": 50000,
    })
    assert knobs.device_loopback_enabled() is False
    assert knobs.ingraph_decode_rope() is True
    assert knobs.kv_plan_cache() == 2
    assert knobs.sampler_slots() == 512
    assert knobs.kv_chain_lazy_steps() == 16
    assert knobs.gc_gen0_threshold() == 50000


def test_env_overrides_config(monkeypatch):
    knobs.configure({"kv_plan_cache": 1, "ingraph_decode_rope": True})
    monkeypatch.setenv("MSTAR_KV_PLAN_CACHE", "0")
    monkeypatch.setenv("MSTAR_INGRAPH_DECODE_ROPE", "0")
    assert knobs.kv_plan_cache() == 0
    assert knobs.ingraph_decode_rope() is False


def test_an_empty_env_var_defers_to_config(monkeypatch):
    knobs.configure({"kv_plan_cache": 1})
    monkeypatch.setenv("MSTAR_KV_PLAN_CACHE", "")
    assert knobs.kv_plan_cache() == 1


def test_reconfigure_replaces_rather_than_merges():
    knobs.configure({"kv_plan_cache": 1})
    knobs.configure({"ingraph_decode_rope": True})
    assert knobs.kv_plan_cache() == 0


@pytest.mark.parametrize("bad", [
    {"kv_plan_cahce": 1},
    {"kv_plan_cache": "1"},
    {"kv_plan_cache": 1.5},
    ["kv_plan_cache"],
])
def test_bad_blocks_fail(bad):
    with pytest.raises(ValueError):
        knobs.check(bad)
    with pytest.raises(ValueError):
        knobs.configure(bad)


def test_every_knob_has_its_env_var_documented():
    docs = open("docs/environment_variables.rst").read()
    for key, var in knobs.KNOBS.items():
        assert f"``{var}``" in docs, var
        assert key in open("docs/serving.rst").read(), key
