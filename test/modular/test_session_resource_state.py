"""Session state in the resource layer: who keeps it, and the budget.

``StepRunner`` is the only thing that knows a request belongs to a session, so
it decides per resource whether a removal frees the state or hands it to the
session — and it is what applies the overflow policy once the state is the
session's. A resource that holds no session state must see exactly the calls it
saw before.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.engine.resources import Resource, StepRunner
from mstar.model.sessions import SessionOverflowPolicy, SessionResourceConfig


class _Stub(Resource):
    """Holds a per-rid integer as its "state", and a per-session one."""

    def __init__(self, name: str, deps: tuple[str, ...] = ()):
        self.name = name
        self._deps = set(deps)
        self.calls: list[str] = []
        self.requests: dict[str, int] = {}
        self.sessions: dict[str, int] = {}

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError("stub is constructed directly")

    def depends_on(self) -> set[str]:
        return set(self._deps)

    # request lifecycle
    def ingest_request(self, rid, overrides=None):
        self.calls.append(f"ingest:{rid}")
        self.requests[rid] = 0

    def remove_request(self, rid):
        self.calls.append(f"remove:{rid}")
        self.requests.pop(rid, None)

    # session lifecycle
    def adopt_session_state(self, rid, session_id):
        self.calls.append(f"adopt:{rid}:{session_id}")
        held = self.sessions.pop(session_id, None)
        if held is not None:
            self.requests[rid] = held

    def retain_session_state(self, rid, session_id):
        self.calls.append(f"retain:{rid}:{session_id}")
        self.sessions[session_id] = self.requests.pop(rid, 0)

    def remove_session(self, session_id):
        self.calls.append(f"remove_session:{session_id}")
        self.sessions.pop(session_id, None)

    def session_state_size(self, session_id):
        return self.sessions.get(session_id, 0)


def _runner(session_keys=(), max_state=None, policy=None, derived=()):
    """A runner over three stubs. ``session_keys`` hold session state under the
    given budget; ``derived`` hold it with no budget of their own, which is what
    the engine gives a resource built against a session resource."""
    resources = {
        "kv": _Stub("kv"),
        "pos": _Stub("pos", deps=("kv",)),
        "sampler": _Stub("sampler"),
    }
    for key in session_keys:
        resources[key].session_config = SessionResourceConfig(
            max_state=max_state,
            overflow_policy=policy or SessionOverflowPolicy.CLEAR,
        )
    for key in derived:
        resources[key].session_config = SessionResourceConfig()
    return StepRunner(resources), resources


# ── dispatch ────────────────────────────────────────────────────────────────

def test_a_sessionless_request_behaves_exactly_as_before():
    runner, res = _runner(session_keys=("kv",))

    runner.ingest_request("r0")
    runner.remove_request("r0")

    assert res["kv"].calls == ["ingest:r0", "remove:r0"]
    assert res["sampler"].calls == ["ingest:r0", "remove:r0"]


def test_only_session_resources_are_asked_to_hold_state():
    runner, res = _runner(session_keys=("kv",))

    runner.ingest_request("r0", session_id="s")
    runner.remove_request("r0", session_id="s")

    assert res["kv"].calls == ["ingest:r0", "adopt:r0:s", "retain:r0:s"]
    # not a session resource: freed with the request, and never adopted
    assert res["sampler"].calls == ["ingest:r0", "remove:r0"]


def test_the_session_hands_its_state_to_the_next_request():
    runner, res = _runner(session_keys=("kv",))
    kv = res["kv"]

    runner.ingest_request("r0", session_id="s")
    kv.requests["r0"] = 12  # the request built some state
    runner.remove_request("r0", session_id="s")

    assert kv.sessions == {"s": 12}

    runner.ingest_request("r1", session_id="s")

    assert kv.requests["r1"] == 12
    assert kv.sessions == {}  # held by the request while it runs


def test_remove_session_frees_every_session_resource_and_nothing_else():
    runner, res = _runner(session_keys=("kv", "pos"))
    runner.ingest_request("r0", session_id="s")
    runner.remove_request("r0", session_id="s")

    runner.remove_session("s")

    assert "remove_session:s" in res["kv"].calls
    assert "remove_session:s" in res["pos"].calls
    assert not any(c.startswith("remove_session") for c in res["sampler"].calls)


def test_session_resource_keys_names_what_holds_state():
    runner, _ = _runner(session_keys=("kv", "pos"))

    assert runner.session_resource_keys() == ["kv", "pos"]


# ── the budget ──────────────────────────────────────────────────────────────

def test_state_within_budget_is_left_alone():
    runner, res = _runner(session_keys=("kv",), max_state=10)
    runner.ingest_request("r0", session_id="s")
    res["kv"].requests["r0"] = 10

    runner.remove_request("r0", session_id="s")

    assert res["kv"].sessions == {"s": 10}
    assert runner.take_session_error("s") is None


def test_no_budget_means_no_check():
    runner, res = _runner(session_keys=("kv",))
    runner.ingest_request("r0", session_id="s")
    res["kv"].requests["r0"] = 10_000

    runner.remove_request("r0", session_id="s")

    assert res["kv"].sessions == {"s": 10_000}


def test_the_clear_policy_drops_the_whole_session_state():
    runner, res = _runner(
        session_keys=("kv",), max_state=10,
        policy=SessionOverflowPolicy.CLEAR,
    )
    runner.ingest_request("r0", session_id="s")
    res["kv"].requests["r0"] = 11

    runner.remove_request("r0", session_id="s")

    assert res["kv"].sessions == {}
    # the session survives; only its state went
    assert runner.take_session_error("s") is None


def test_the_error_policy_clears_and_owes_the_next_request_an_error():
    runner, res = _runner(
        session_keys=("kv",), max_state=10,
        policy=SessionOverflowPolicy.ERROR,
    )
    runner.ingest_request("r0", session_id="s")
    res["kv"].requests["r0"] = 40

    runner.remove_request("r0", session_id="s")

    assert res["kv"].sessions == {}
    error = runner.take_session_error("s")
    assert error is not None
    assert "40 > 10" in error
    # reported once
    assert runner.take_session_error("s") is None


def test_ending_the_session_drops_the_owed_error_too():
    runner, res = _runner(
        session_keys=("kv",), max_state=1,
        policy=SessionOverflowPolicy.ERROR,
    )
    runner.ingest_request("r0", session_id="s")
    res["kv"].requests["r0"] = 40
    runner.remove_request("r0", session_id="s")

    runner.remove_session("s")

    assert runner.take_session_error("s") is None


def test_a_breach_on_one_resource_clears_every_session_resource():
    # the others' state addresses what is being dropped, so a partial clear
    # would leave the session describing a context it no longer holds
    runner, res = _runner(session_keys=("kv",), derived=("pos",), max_state=10)
    runner.ingest_request("r0", session_id="s")
    res["kv"].requests["r0"] = 40
    res["pos"].requests["r0"] = 40

    runner.remove_request("r0", session_id="s")

    assert res["kv"].sessions == {}
    assert res["pos"].sessions == {}


@pytest.mark.parametrize("policy", list(SessionOverflowPolicy))
def test_every_policy_leaves_the_session_usable_or_told(policy):
    runner, res = _runner(session_keys=("kv",), max_state=4, policy=policy)
    runner.ingest_request("r0", session_id="s")
    res["kv"].requests["r0"] = 99
    runner.remove_request("r0", session_id="s")

    assert res["kv"].session_state_size("s") <= 4
    runner.ingest_request("r1", session_id="s")  # never raises
