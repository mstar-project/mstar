"""The session store behind ``session_id`` / ``resume_session``: LRU with
promotion on read, in-flight sessions pinned against eviction and against a
second writer, a timeout on idle sessions, and explicit ending."""

from __future__ import annotations

import pytest

from mstar.model.cosmos3.sessions import SessionStore


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _store(capacity=2, ttl=100.0, max_ttl=1000.0):
    clock = _Clock()
    return SessionStore(capacity, ttl, max_ttl, clock=clock), clock


def test_lru_promotes_on_read_and_evicts_the_oldest_idle():
    store, _ = _store(capacity=2)
    store.put("a", 1)
    store.put("b", 2)
    assert store.get("a") == 1                # promoted: b is now the oldest
    store.put("c", 3)
    assert store.ids() == ["a", "c"] and "b" not in store
    assert store.get("zzz") is None


def test_live_sessions_are_never_evicted_and_refuse_a_second_writer():
    store, _ = _store(capacity=1)
    store.begin("w", "r1")                    # a rollout in flight, nothing stored yet
    with pytest.raises(ValueError, match="in use by request 'r1'"):
        store.begin("w", "r2")
    store.begin("w", "r1")                    # the same request may re-pin
    for sid in ("x", "y", "z"):               # idle sessions churn around it
        store.put(sid, 0)
    assert store.live("w") == "r1" and store.ids()[-1] == "z" and len(store) == 2
    store.put("w", 42)                        # the rollout stores and unpins
    assert store.live("w") is None and "w" in store
    store.begin("w", "r2")                    # a follow-up can take it
    assert store.get("w") == 42


def test_release_unpins_a_removed_request():
    store, _ = _store(capacity=2)
    store.begin("w", "r1")
    store.release("r1")                       # cancelled before storing: nothing to keep
    assert "w" not in store and len(store) == 0
    store.put("w", 1)
    store.begin("w", "r2")
    store.release("r2")                       # cancelled mid-resume: the old state stays, unpinned
    assert store.get("w") == 1 and store.live("w") is None
    store.release("ghost")                    # a request that held nothing


def test_idle_sessions_expire_live_ones_do_not():
    store, clock = _store(capacity=4, ttl=10.0, max_ttl=60.0)
    store.put("short", 1)
    store.put("long", 2, ttl_s=30.0)
    store.put("capped", 3, ttl_s=1e9)        # clamped to the maximum
    store.begin("busy", "r1")
    clock.t = 11.0
    assert store.get("short") is None and store.get("long") == 2
    clock.t = 61.0
    assert store.get("long") is None and store.get("capped") is None
    assert store.live("busy") == "r1"        # pinned, no expiry
    with pytest.raises(ValueError, match="session_timeout_s"):
        store.ttl(0)
    assert store.ttl(None) == 10.0 and store.ttl(120.0) == 60.0


def test_end_forgets_the_session():
    store, _ = _store()
    store.put("w", 1)
    store.begin("w", "r1")
    store.end("w")
    assert "w" not in store and store.live("w") is None and len(store) == 0
    store.end("w")                            # idempotent
