"""Engine knobs resolve as env var, then the deployment config's ``knobs:``
block, then the platform's default, then the knob's own; a malformed config
block fails at startup."""

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


def test_the_platform_default_sits_under_the_config(monkeypatch):
    from mstar import platforms
    from mstar.platforms import RocmPlatform

    rocm = RocmPlatform()
    monkeypatch.setattr(platforms, "resolve_platform", lambda: rocm)
    knobs.configure(None)
    assert knobs.ingraph_decode_rope() is True and knobs.sampler_ingraph_scatter() is True
    assert knobs.kv_plan_cache() == 0, "a knob the platform leaves alone keeps its default"
    knobs.configure({"ingraph_decode_rope": False})
    assert knobs.ingraph_decode_rope() is False, "the config wins over the platform"
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "0")
    assert knobs.sampler_ingraph_scatter() is False, "a set env var wins over the platform"


@pytest.mark.parametrize("bad", [{"kv_plan_cahce": 1}, {"kv_plan_cache": "1"}])
def test_a_malformed_block_fails_the_startup_check(bad):
    with pytest.raises(ValueError):
        knobs.check(bad)


def test_step_thread_core_from_the_config_and_the_environment(monkeypatch):
    assert knobs.step_thread_core() == "auto"
    # yaml reads a bare off as False
    for value, expected in ((False, "off"), (True, "auto"), ("auto", "auto"), ("OFF", "off"), (12, 12)):
        knobs.configure({"step_thread_core": value})
        assert knobs.step_thread_core() == expected
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "off")
    assert knobs.step_thread_core() == "off"
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "7")
    assert knobs.step_thread_core() == 7
    monkeypatch.setenv("MSTAR_STEP_THREAD_CORE", "first")
    with pytest.raises(ValueError, match="MSTAR_STEP_THREAD_CORE"):
        knobs.step_thread_core()


@pytest.mark.parametrize("value", [-1, "on", "3-4", 1.5])
def test_a_bad_step_thread_core_fails_the_startup_check(value):
    with pytest.raises(ValueError, match="step_thread_core"):
        knobs.check({"step_thread_core": value})
