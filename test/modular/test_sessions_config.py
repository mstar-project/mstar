"""The model's session declaration and the deployment's overrides of it.

A session's caps are a deployment property (how many, how long) layered over a
model property (which resources hold state), so the merge and its refusals are
what these pin.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.model.sessions import (
    SessionOverflowPolicy,
    SessionResourceConfig,
    SessionsConfig,
    SessionTTLMode,
    apply_sessions_yaml_overrides,
)


def _config(**kwargs) -> SessionsConfig:
    base = {"resources": {"kv_cache": SessionResourceConfig(max_state=64)}}
    base.update(kwargs)
    return SessionsConfig(**base)


# ── the declaration itself ──────────────────────────────────────────────────

def test_string_enums_are_coerced():
    cfg = SessionsConfig(
        resources={"kv_cache": {"max_state": 8, "overflow_policy": "error"}},
        ttl_mode="absolute",
    )

    assert cfg.ttl_mode is SessionTTLMode.ABSOLUTE
    resource = cfg.resources["kv_cache"]
    assert isinstance(resource, SessionResourceConfig)
    assert resource.overflow_policy is SessionOverflowPolicy.ERROR


def test_interruptible_is_declared_but_refused_for_now():
    with pytest.raises(ValueError, match="not implemented"):
        _config(interruptible=True)


def test_refuses_nonsense_caps():
    with pytest.raises(ValueError, match="max_concurrent_sessions"):
        _config(max_concurrent_sessions=0)
    with pytest.raises(ValueError, match="default_timeout_s"):
        _config(default_timeout_s=900.0, max_timeout_s=600.0)
    with pytest.raises(ValueError, match="ceiling"):
        _config(max_timeout_s=10 * 24 * 3600)
    with pytest.raises(ValueError, match="max_state"):
        SessionResourceConfig(max_state=0)


def test_timeout_resolution():
    cfg = _config(default_timeout_s=120.0, max_timeout_s=600.0)

    assert cfg.resolve_timeout_s(None) == 120.0
    assert cfg.resolve_timeout_s(300) == 300.0
    with pytest.raises(ValueError, match="exceeds"):
        cfg.resolve_timeout_s(601)
    with pytest.raises(ValueError, match="positive"):
        cfg.resolve_timeout_s(0)


# ── the deployment's overrides ──────────────────────────────────────────────

def test_no_block_leaves_the_declaration_alone():
    cfg = _config()
    assert apply_sessions_yaml_overrides(cfg, {}) is cfg


def test_block_can_disable_sessions_outright():
    assert apply_sessions_yaml_overrides(
        _config(), {"sessions": {"enabled": False}}
    ) is None


def test_block_tunes_caps_and_budgets():
    merged = apply_sessions_yaml_overrides(_config(), {"sessions": {
        "max_concurrent_sessions": 32,
        "default_timeout_s": 60.0,
        "resources": {"kv_cache": {"max_state": 256, "overflow_policy": "error"}},
    }})

    assert merged.max_concurrent_sessions == 32
    assert merged.default_timeout_s == 60.0
    assert merged.resources["kv_cache"].max_state == 256
    assert (
        merged.resources["kv_cache"].overflow_policy
        is SessionOverflowPolicy.ERROR
    )


def test_overrides_do_not_mutate_the_model_s_declaration():
    cfg = _config()
    apply_sessions_yaml_overrides(cfg, {"sessions": {
        "resources": {"kv_cache": {"max_state": 1}},
    }})

    assert cfg.resources["kv_cache"].max_state == 64


def test_block_cannot_invent_a_session_resource():
    with pytest.raises(ValueError, match="does not hold across a session"):
        apply_sessions_yaml_overrides(_config(), {"sessions": {
            "resources": {"recurrent": {"max_state": 4}},
        }})


def test_block_on_a_model_without_sessions_is_refused():
    with pytest.raises(ValueError, match="does not support sessions"):
        apply_sessions_yaml_overrides(None, {"sessions": {"max_timeout_s": 60}})


def test_unknown_key_is_refused():
    with pytest.raises(ValueError, match="unknown key"):
        apply_sessions_yaml_overrides(_config(), {"sessions": {"ttl": 30}})
