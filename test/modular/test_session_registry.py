"""API-server session tracking: validation, TTL, and the teardown tombstone.

The registry is the only thing that decides what a session request may do, so
these pin the refusals a client sees and the point at which an id stops being
usable — from the flag validation through the tombstone that stands until the
conductor confirms the state is gone.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.api_server.sessions import SessionError, SessionRegistry
from mstar.model.sessions import SessionResourceConfig, SessionsConfig


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _registry(**kwargs):
    config = SessionsConfig(
        resources={"kv_cache": SessionResourceConfig(max_state=32)},
        **kwargs,
    )
    clock = _Clock()
    torn_down: list[str] = []
    reg = SessionRegistry(config, teardown=torn_down.append, clock=clock)
    return reg, clock, torn_down


def _start(reg, request_id="r0", session_id=None, **kwargs):
    return reg.resolve(
        start_session=True, resume_session=False, end_session=False,
        session_id=session_id, session_timeout_s=None,
        request_id=request_id, **kwargs,
    )


def _resume(reg, session_id, request_id, end_session=False):
    return reg.resolve(
        start_session=False, resume_session=True, end_session=end_session,
        session_id=session_id, session_timeout_s=None, request_id=request_id,
    )


# ── flag validation ─────────────────────────────────────────────────────────

def test_a_request_with_no_session_flags_is_not_a_session_request():
    reg, _, _ = _registry()

    resolved = reg.resolve(
        start_session=False, resume_session=False, end_session=False,
        session_id=None, session_timeout_s=None, request_id="r0",
    )

    assert resolved.session_id is None
    assert reg.snapshot() == []


def test_sessions_on_a_deployment_without_them_is_a_400():
    reg = SessionRegistry(None, teardown=lambda _sid: None)

    with pytest.raises(SessionError) as e:
        _start(reg)
    assert e.value.status == 400


def test_start_and_resume_together_is_refused():
    reg, _, _ = _registry()

    with pytest.raises(SessionError, match="mutually exclusive") as e:
        reg.resolve(
            start_session=True, resume_session=True, end_session=False,
            session_id="s", session_timeout_s=None, request_id="r0",
        )
    assert e.value.status == 400


def test_naming_a_session_without_start_or_resume_is_refused():
    reg, _, _ = _registry()

    with pytest.raises(SessionError, match="start_session or resume_session"):
        reg.resolve(
            start_session=False, resume_session=False, end_session=False,
            session_id="s", session_timeout_s=None, request_id="r0",
        )


def test_resume_needs_an_id():
    reg, _, _ = _registry()

    with pytest.raises(SessionError, match="resume_session requires"):
        reg.resolve(
            start_session=False, resume_session=True, end_session=False,
            session_id=None, session_timeout_s=None, request_id="r0",
        )


def test_end_session_on_its_own_points_at_the_delete_route():
    reg, _, _ = _registry()

    with pytest.raises(SessionError, match="DELETE /sessions"):
        reg.resolve(
            start_session=False, resume_session=False, end_session=True,
            session_id="s", session_timeout_s=None, request_id="r0",
        )


def test_a_timeout_over_the_deployment_maximum_is_refused():
    reg, _, _ = _registry(default_timeout_s=60.0, max_timeout_s=120.0)

    with pytest.raises(SessionError, match="exceeds") as e:
        reg.resolve(
            start_session=True, resume_session=False, end_session=False,
            session_id=None, session_timeout_s=600.0, request_id="r0",
        )
    assert e.value.status == 400


# ── start ───────────────────────────────────────────────────────────────────

def test_start_without_an_id_mints_one_and_reports_it():
    reg, _, _ = _registry()

    resolved = _start(reg)

    assert resolved.created is True
    assert resolved.session_id
    assert [s["session_id"] for s in reg.snapshot()] == [resolved.session_id]


def test_start_with_an_id_keeps_it_and_does_not_report_it_as_minted():
    reg, _, _ = _registry()

    resolved = _start(reg, session_id="mine")

    assert (resolved.session_id, resolved.created) == ("mine", False)


def test_start_on_an_existing_id_is_a_409():
    reg, _, _ = _registry()
    _start(reg, request_id="r0", session_id="mine")

    with pytest.raises(SessionError, match="already exists") as e:
        _start(reg, request_id="r1", session_id="mine")
    assert e.value.status == 409


def test_start_past_the_concurrency_cap_is_a_429():
    reg, _, _ = _registry(max_concurrent_sessions=2)
    _start(reg, request_id="r0", session_id="a")
    _start(reg, request_id="r1", session_id="b")

    with pytest.raises(SessionError) as e:
        _start(reg, request_id="r2", session_id="c")
    assert e.value.status == 429


def test_a_closing_session_does_not_count_against_the_cap():
    reg, _, _ = _registry(max_concurrent_sessions=1)
    _start(reg, request_id="r0", session_id="a")
    reg.finish_request("r0")
    reg.delete("a")

    _start(reg, request_id="r1", session_id="b")  # no raise


# ── a full deployment: keep the parked state, or evict it ───────────────────

def test_keep_is_the_default_and_refuses_rather_than_evicting():
    reg, _, torn_down = _registry(max_concurrent_sessions=1)
    _start(reg, request_id="r0", session_id="a")
    reg.finish_request("r0")  # idle, but KEEP holds it until close or TTL

    with pytest.raises(SessionError) as e:
        _start(reg, request_id="r1", session_id="b")

    assert e.value.status == 429
    assert torn_down == []


def test_evict_makes_room_by_tearing_down_the_idle_session():
    reg, _, torn_down = _registry(
        max_concurrent_sessions=1, parked_policy="evict",
    )
    _start(reg, request_id="r0", session_id="a")
    reg.finish_request("r0")

    assert _start(reg, request_id="r1", session_id="b").session_id == "b"

    assert torn_down == ["a"]
    # the evicted id is held until its teardown is confirmed
    with pytest.raises(SessionError, match="evicted to make room"):
        _resume(reg, "a", "r2")


def test_evict_takes_the_least_recently_used_idle_session():
    reg, clock, torn_down = _registry(
        max_concurrent_sessions=2, parked_policy="evict",
    )
    _start(reg, request_id="r0", session_id="old")
    reg.finish_request("r0")
    clock.advance(10.0)
    _start(reg, request_id="r1", session_id="new")
    reg.finish_request("r1")

    _start(reg, request_id="r2", session_id="third")

    assert torn_down == ["old"]


def test_a_session_with_a_request_in_flight_is_never_evicted():
    # it is writing its state right now
    reg, clock, torn_down = _registry(
        max_concurrent_sessions=1, parked_policy="evict",
    )
    _start(reg, request_id="r0", session_id="busy")
    clock.advance(1000.0)

    with pytest.raises(SessionError) as e:
        _start(reg, request_id="r1", session_id="b")

    assert e.value.status == 429
    assert torn_down == []


def test_evict_skips_a_session_that_is_already_closing():
    reg, _, torn_down = _registry(
        max_concurrent_sessions=2, parked_policy="evict",
    )
    _start(reg, request_id="r0", session_id="a")
    reg.finish_request("r0")
    reg.delete("a")  # closing: it no longer counts, and is not a victim
    _start(reg, request_id="r1", session_id="b")
    reg.finish_request("r1")

    # one live session against a cap of 2: room without evicting
    _start(reg, request_id="r2", session_id="c")

    assert torn_down == ["a"]


# ── resume ──────────────────────────────────────────────────────────────────

def test_resume_of_an_unknown_session_is_a_404():
    reg, _, _ = _registry()

    with pytest.raises(SessionError) as e:
        _resume(reg, "nope", "r0")
    assert e.value.status == 404


def test_resume_with_a_request_in_flight_is_a_409():
    reg, _, _ = _registry()
    _start(reg, request_id="r0", session_id="s")

    with pytest.raises(SessionError, match="already has a request in flight") as e:
        _resume(reg, "s", "r1")
    assert e.value.status == 409


def test_resume_is_allowed_once_the_previous_request_finished():
    reg, _, _ = _registry()
    _start(reg, request_id="r0", session_id="s")
    reg.finish_request("r0")

    resolved = _resume(reg, "s", "r1")

    assert (resolved.session_id, resolved.created) == ("s", False)


def test_an_interruptible_deployment_allows_a_concurrent_resume():
    # the config refuses the flag for now (nothing routes a second request's
    # inputs into an in-progress one), so set it past the guard to reach the
    # registry branch that bidirectional streaming will use
    reg, _, _ = _registry()
    reg.config.interruptible = True
    _start(reg, request_id="r0", session_id="s")

    assert _resume(reg, "s", "r1").session_id == "s"


def test_resume_of_a_closing_session_is_a_409():
    reg, _, _ = _registry()
    _start(reg, request_id="r0", session_id="s")
    reg.finish_request("r0")
    reg.delete("s")

    with pytest.raises(SessionError, match="being torn down") as e:
        _resume(reg, "s", "r1")
    assert e.value.status == 409


# ── ending and teardown ─────────────────────────────────────────────────────

def test_note_ending_holds_the_tombstone_before_the_request_finishes():
    reg, _, torn_down = _registry()
    _start(reg, request_id="r0", session_id="s")
    reg.note_ending("s")

    with pytest.raises(SessionError, match="being torn down"):
        _resume(reg, "s", "r1")
    # the conductor tears it down with the request; no standalone ask
    assert torn_down == []


def test_delete_asks_the_conductor_once_and_holds_the_id():
    reg, _, torn_down = _registry()
    _start(reg, request_id="r0", session_id="s")
    reg.finish_request("r0")

    reg.delete("s")
    reg.delete("s")

    assert torn_down == ["s"]
    assert reg.snapshot()[0]["closing"] is True


def test_delete_with_a_request_in_flight_is_a_409():
    reg, _, torn_down = _registry()
    _start(reg, request_id="r0", session_id="s")

    with pytest.raises(SessionError, match="request in flight") as e:
        reg.delete("s")
    assert (e.value.status, torn_down) == (409, [])


def test_delete_of_an_unknown_session_is_a_404():
    reg, _, _ = _registry()

    with pytest.raises(SessionError) as e:
        reg.delete("nope")
    assert e.value.status == 404


def test_the_id_is_usable_again_only_after_the_conductor_acks():
    reg, _, _ = _registry()
    _start(reg, request_id="r0", session_id="s")
    reg.finish_request("r0")
    reg.delete("s")

    with pytest.raises(SessionError):
        _start(reg, request_id="r1", session_id="s")

    reg.torn_down("s")

    assert _start(reg, request_id="r2", session_id="s").session_id == "s"
    assert len(reg.snapshot()) == 1


def test_a_failed_request_takes_its_session_down():
    reg, _, torn_down = _registry()
    _start(reg, request_id="r0", session_id="s")

    reg.finish_request("r0", failed=True, error="engine blew up")

    assert torn_down == ["s"]
    with pytest.raises(SessionError, match="engine blew up"):
        _resume(reg, "s", "r1")


def test_finish_request_for_a_sessionless_request_is_a_no_op():
    reg, _, torn_down = _registry()

    reg.finish_request("stranger", failed=True, error="boom")

    assert torn_down == []


# ── TTL ─────────────────────────────────────────────────────────────────────

def test_an_idle_session_expires_and_is_torn_down_once():
    reg, clock, torn_down = _registry(default_timeout_s=30.0)
    _start(reg, request_id="r0", session_id="s")
    reg.finish_request("r0")

    clock.advance(29.0)
    assert reg.sweep() == []

    clock.advance(2.0)
    assert reg.sweep() == ["s"]
    assert reg.sweep() == []  # already closing
    assert torn_down == ["s"]


def test_idle_ttl_is_refreshed_by_activity():
    reg, clock, _ = _registry(default_timeout_s=30.0)
    _start(reg, request_id="r0", session_id="s")
    clock.advance(25.0)
    reg.finish_request("r0")  # stamps last activity

    clock.advance(20.0)

    assert reg.sweep() == []


def test_absolute_ttl_ignores_activity():
    reg, clock, _ = _registry(default_timeout_s=30.0, ttl_mode="absolute")
    _start(reg, request_id="r0", session_id="s")
    clock.advance(25.0)
    reg.finish_request("r0")

    clock.advance(10.0)

    assert reg.sweep() == ["s"]


def test_a_session_with_a_request_in_flight_is_never_collected():
    reg, clock, torn_down = _registry(default_timeout_s=10.0)
    _start(reg, request_id="r0", session_id="s")

    clock.advance(1000.0)

    assert reg.sweep() == []
    assert torn_down == []


def test_a_tombstone_nobody_asked_about_is_chased_by_the_sweep():
    # end_session normally rides on the request's own teardown; if that request
    # never reached the conductor, nothing would ever ask
    reg, clock, torn_down = _registry()
    _start(reg, request_id="r0", session_id="s")
    reg.note_ending("s")
    reg.finish_request("r0")

    clock.advance(10.0)
    assert reg.sweep() == []
    assert torn_down == []

    clock.advance(reg._tombstone_grace_s)
    reg.sweep()

    assert torn_down == ["s"]


def test_a_tombstone_that_is_never_acked_releases_its_id():
    reg, clock, torn_down = _registry()
    _start(reg, request_id="r0", session_id="s")
    reg.finish_request("r0")
    reg.delete("s")

    clock.advance(reg._tombstone_grace_s + 1)
    reg.sweep()

    # asked once, never confirmed: the id goes rather than sticking forever
    assert torn_down == ["s"]
    assert reg.snapshot() == []


def test_a_failed_end_session_request_still_asks_once():
    reg, _, torn_down = _registry()
    _start(reg, request_id="r0", session_id="s")
    reg.note_ending("s")

    reg.finish_request("r0", failed=True, error="preprocess exploded")

    assert torn_down == ["s"]


def test_sweep_on_a_deployment_without_sessions_is_a_no_op():
    reg = SessionRegistry(None, teardown=lambda _sid: None)

    assert reg.sweep() == []
    assert reg.enabled is False


def test_snapshot_reports_the_deadline_and_the_in_flight_request():
    reg, clock, _ = _registry(default_timeout_s=60.0)
    _start(reg, request_id="r0", session_id="s")
    clock.advance(10.0)

    [entry] = reg.snapshot()

    assert entry["active_request_ids"] == ["r0"]
    assert entry["age_s"] == 10.0
    assert entry["expires_in_s"] == 50.0


def test_fail_all_forgets_everything():
    reg, _, _ = _registry()
    _start(reg, request_id="r0", session_id="s")

    reg.fail_all("conductor died")

    assert reg.snapshot() == []
    assert reg.session_of("r0") is None
