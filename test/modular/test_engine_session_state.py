"""Which resources the engine marks as session-state holders.

The model names the resources whose state lives for a session; the engine also
marks anything built against one of them, because that state addresses the
state being kept — position counters left at zero over retained KV pages would
have the next request write over the session's own context.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.engine.engine import Engine
from mstar.engine.resources import Resource, StepRunner
from mstar.model.sessions import SessionResourceConfig, SessionsConfig


class _Res(Resource):
    """A resource that holds session state: the hooks are what qualifies it."""

    def __init__(self, deps: tuple[str, ...] = ()):
        self._deps = set(deps)

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def depends_on(self):
        return set(self._deps)

    def retain_session_state(self, rid, session_id):
        return


def _engine(resources):
    engine = Engine.__new__(Engine)
    engine._resources = resources
    engine._runner = StepRunner(resources)
    return engine


def _specs(*keys):
    return {key: object() for key in keys}


def _config(*keys, **kwargs):
    return SessionsConfig(
        resources={key: SessionResourceConfig() for key in keys}, **kwargs
    )


def test_no_sessions_config_marks_nothing():
    resources = {"kv": _Res(), "pos": _Res(deps=("kv",))}
    engine = _engine(resources)

    engine._open_session_state(_specs("kv", "pos"), None)

    assert all(r.session_config is None for r in resources.values())


def test_the_runner_reports_what_was_marked():
    resources = {"kv": _Res(), "pos": _Res(deps=("kv",)), "sampler": _Res()}
    engine = _engine(resources)

    engine._open_session_state(
        _specs("kv", "pos", "sampler"), _config("kv"),
    )

    assert engine._runner.session_resource_keys() == ["kv", "pos"]


def test_the_named_resource_and_its_dependents_are_marked():
    resources = {
        "kv": _Res(),
        "pos": _Res(deps=("kv",)),
        "sampler": _Res(),
    }
    engine = _engine(resources)

    engine._open_session_state(
        _specs("kv", "pos", "sampler"), _config("kv"),
    )

    assert resources["kv"].session_config is not None
    assert resources["pos"].session_config is not None
    assert resources["sampler"].session_config is None


def test_the_named_resource_keeps_its_own_budget():
    resources = {"kv": _Res(), "pos": _Res(deps=("kv",))}
    config = SessionsConfig(
        resources={"kv": SessionResourceConfig(max_state=128)}
    )
    engine = _engine(resources)

    engine._open_session_state(_specs("kv", "pos"), config)

    assert resources["kv"].session_config.max_state == 128
    # the derived one is unbounded: the budget belongs to the state it addresses
    assert resources["pos"].session_config.max_state is None


def test_a_dependency_on_a_plain_resource_marks_nothing():
    resources = {"kv": _Res(), "pos": _Res(deps=("kv",))}
    engine = _engine(resources)

    engine._open_session_state(_specs("kv", "pos"), _config("pos"))

    assert resources["kv"].session_config is None
    assert resources["pos"].session_config is not None


def test_a_dependent_without_the_session_hooks_is_left_alone():
    # the attention resource plans per step and holds nothing across requests;
    # marking it would skip its ordinary per-request teardown
    class _Plain(Resource):
        def __init__(self, deps):
            self._deps = set(deps)

        @classmethod
        def build(cls, spec, info):
            raise NotImplementedError

        def depends_on(self):
            return set(self._deps)

    resources = {"kv": _Res(), "attn": _Plain(("kv",)), "rope": _Res(deps=("kv",))}
    engine = _engine(resources)

    engine._open_session_state(_specs("kv", "attn", "rope"), _config("kv"))

    assert resources["attn"].session_config is None
    assert resources["rope"].session_config is not None


def test_a_resource_without_the_session_hooks_is_refused():
    # it would inherit the base no-ops and hold nothing, silently
    class _Plain(Resource):
        @classmethod
        def build(cls, spec, info):
            raise NotImplementedError

    engine = _engine({"ring": _Plain()})

    with pytest.raises(ValueError, match="does not implement the session hooks"):
        engine._open_session_state(_specs("ring"), _config("ring"))


def test_a_session_resource_the_model_does_not_declare_is_refused():
    engine = _engine({"kv": _Res()})

    with pytest.raises(ValueError, match="does not declare"):
        engine._open_session_state(_specs("kv"), _config("recurrent"))



