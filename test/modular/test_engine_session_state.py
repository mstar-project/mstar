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





# ── a starting session never inherits what is parked under its id ────────────

class _IngestRunner:
    def __init__(self, held=()):
        self.held = set(held)
        self.calls: list[str] = []

    def session_holds_state(self, session_id):
        return session_id in self.held

    def ingest_request(self, rid, overrides=None, session=None):
        self.calls.append(f"ingest:{rid}")

    def remove_session(self, session_id):
        self.held.discard(session_id)
        self.calls.append(f"remove_session:{session_id}")


class _Submodule:
    def __init__(self, session_states=()):
        self.session_states = dict.fromkeys(session_states, object())

    def cleanup_session(self, session_id):
        self.session_states.pop(session_id, None)


def _ingesting_engine(runner, submodule):
    from types import SimpleNamespace

    engine = Engine.__new__(Engine)
    engine._runner = runner
    engine._request_sessions = {}
    engine._submodules = {"node": SimpleNamespace(submodule=submodule)}
    return engine


@pytest.mark.parametrize("held, states", [(("s",), ()), ((), ("s",))])
def test_a_starting_session_drops_state_left_under_its_id(held, states):
    from mstar.model.sessions import RequestSession

    runner, submodule = _IngestRunner(held), _Submodule(states)
    engine = _ingesting_engine(runner, submodule)

    engine.add_request("r0", session=RequestSession("s"))

    assert runner.calls == ["remove_session:s", "ingest:r0"]
    assert submodule.session_states == {}
    assert engine._request_sessions == {"r0": "s"}


def test_a_resumed_session_keeps_its_state():
    from mstar.model.sessions import RequestSession

    runner = _IngestRunner(held=("s",))
    engine = _ingesting_engine(runner, _Submodule(("s",)))

    engine.add_request("r1", session=RequestSession("s", resumed=True))

    assert runner.calls == ["ingest:r1"]


def test_a_second_partition_s_ingest_does_not_clear_the_first_s_state():
    from mstar.model.sessions import RequestSession

    runner, submodule = _IngestRunner(), _Submodule()
    engine = _ingesting_engine(runner, submodule)
    engine.add_request("r0", session=RequestSession("s"))
    # the first partition has started writing session state
    submodule.session_states["s"] = object()

    engine.add_request("r0", session=RequestSession("s"))

    assert "s" in submodule.session_states
    assert runner.calls == ["ingest:r0", "ingest:r0"]


# ── telling a continuing session its last token is in the KV ────────────────

class _RemovingRunner:
    def remove_request(self, rid, session_id=None):
        pass

    def remove_session(self, session_id):
        pass


class _StatefulSubmodule:
    def __init__(self):
        from mstar.model.submodule_base import PerRequestState

        self.session_states: dict[str, PerRequestState] = {}

    def session_state(self, session_id):
        from mstar.model.submodule_base import PerRequestState

        return self.session_states.setdefault(session_id, PerRequestState())

    def cleanup_request(self, rid):
        pass

    def cleanup_session(self, session_id):
        self.session_states.pop(session_id, None)


def _removing_engine(*nodes):
    from types import SimpleNamespace

    engine = Engine.__new__(Engine)
    engine._runner = _RemovingRunner()
    engine._request_sessions = {}
    engine._submodules = {
        node: SimpleNamespace(submodule=_StatefulSubmodule()) for node in nodes
    }
    return engine


def _flag(engine, node, session_id="s"):
    from mstar.model.submodule_base import OVERSHOT_LAST_ITER

    state = engine._submodules[node].submodule.session_states.get(session_id)
    return None if state is None else state.kwargs.get(OVERSHOT_LAST_ITER)


def test_an_overshot_node_tells_its_session_the_token_is_in_the_kv():
    engine = _removing_engine("LLM", "other")
    engine._request_sessions["r0"] = "s"

    engine.remove_request("r0", overshot_nodes={"LLM"})

    assert _flag(engine, "LLM") is True
    # a node that did not overshoot gets no state it never asked for
    assert engine._submodules["other"].submodule.session_states == {}


def test_a_turn_that_stopped_exactly_clears_an_earlier_turn_s_flag():
    engine = _removing_engine("LLM")
    engine._request_sessions["r0"] = "s"
    engine.remove_request("r0", overshot_nodes={"LLM"})

    engine._request_sessions["r1"] = "s"
    engine.remove_request("r1")

    assert _flag(engine, "LLM") is None


def test_a_sessionless_or_ending_request_flags_nothing():
    engine = _removing_engine("LLM")

    engine.remove_request("r0", overshot_nodes={"LLM"})
    engine._request_sessions["r1"] = "s"
    engine.remove_request("r1", end_session=True, overshot_nodes={"LLM"})

    assert engine._submodules["LLM"].submodule.session_states == {}
