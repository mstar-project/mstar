#!/usr/bin/env python3
"""The session rules a live deployment has to enforce, one scenario per rule.

``session_request.py`` walks the happy path and ``budget_probe.py`` the state
budget; this covers the rest of the session config — the concurrency cap and
what a full deployment does with a new session, the two TTL modes, and the
refusals a client can provoke. Each scenario reads the knobs it needs from the
deployment's config and refuses to run against a config that cannot show the
behaviour, so a passing run means something.

    CUDA_VISIBLE_DEVICES=7 CONFIG=configs/text_session/ttl_idle.yaml \
        bash test/text_session/launch_server.sh
    python test/text_session/scenarios.py ttl_idle \
        --url http://localhost:8137 --config configs/text_session/ttl_idle.yaml

``--list`` names the scenarios. See configs/text_session/ for which config each
one wants, and test/text_session/session_matrix.sbatch for the whole matrix.
"""

import argparse
import contextlib
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import yaml

from mstar import MStarClient

REMEMBER = "Remember this number: {}. Reply with just the word OK."
RECALL = "What was the number I asked you to remember? Answer with digits only."
# distinct per session, so a recall says *which* session's state was kept
NUMBERS = ("8675309", "1234567", "2718281", "3141592", "1618033")
# something long enough that the request is still running when the next one is
# sent, for the checks about a session with a request in flight
LONG_TURN = "Count from 1 to 400, one number per line."
LONG_TOKENS = 512


class Wrong(SystemExit):
    """This scenario cannot prove anything against this deployment."""

    def __init__(self, detail: str):
        super().__init__(2)
        print(f"WRONG CONFIG: {detail}", flush=True)


@dataclass
class Deployment:
    """The session knobs the scenarios care about, as configured."""

    max_sessions: int
    capacity_policy: str
    ttl_mode: str
    timeout_s: float
    max_timeout_s: float

    @classmethod
    def read(cls, path: Path) -> "Deployment":
        sessions = yaml.safe_load(path.read_text()).get("sessions") or {}
        if not sessions:
            raise Wrong(f"{path} has no `sessions:` block")
        return cls(
            max_sessions=int(sessions["max_concurrent_sessions"]),
            capacity_policy=sessions.get("capacity_policy", "keep"),
            ttl_mode=sessions.get("ttl_mode", "idle"),
            timeout_s=float(sessions["default_timeout_s"]),
            max_timeout_s=float(sessions["max_timeout_s"]),
        )


class Checks:
    def __init__(self):
        self._results: list[tuple[bool, str]] = []

    def that(self, ok: bool, what: str, detail: str = "") -> bool:
        self._results.append((bool(ok), what))
        verdict = "PASS" if ok else "FAIL"
        print(f"  [{verdict}] {what}{f' -- {detail}' if detail else ''}",
              flush=True)
        return bool(ok)

    def skip(self, what: str, why: str) -> None:
        # not a failure: a check whose setup the server outran, e.g. a request
        # that finished before the race it was meant to lose could be run
        print(f"  [SKIP] {what} -- {why}", flush=True)

    def exit_code(self) -> int:
        failed = [what for ok, what in self._results if not ok]
        print(f"\n{len(self._results) - len(failed)}/{len(self._results)} checks passed")
        for what in failed:
            print(f"  failed: {what}")
        return 1 if failed else 0


# ── talking to the server ───────────────────────────────────────────────────

# Every session id this run has been handed. The server lists counts, never
# ids, so the scenarios keep track of their own.
_OPENED: set[str] = set()


def _turn(client, text, tokens=16, **session):
    # top_k=1: the recall checks must not hang on the model's sampling default
    result = client.generate(
        text=text, max_output_tokens=tokens, top_k=1, **session,
    )
    if result.session_id:
        _OPENED.add(result.session_id)
    return result


def _refused(call) -> int | None:
    """The status a refused call came back with, or None if it succeeded."""
    try:
        call()
    except requests.HTTPError as e:
        return e.response.status_code
    return None


def _start(client, number: str, **kwargs) -> str:
    return _turn(
        client, REMEMBER.format(number), start_session=True, **kwargs,
    ).session_id


def _recall(client, session_id: str) -> str:
    return _turn(
        client, RECALL, resume_session=True, session_id=session_id,
    ).text or ""


def _snapshot(client, session_id: str) -> dict | None:
    """``GET /sessions/{id}``, retried once if the connection was dropped.

    The server closes an idle keep-alive connection after 5s (uvicorn's
    default) and these scenarios poll on about that cadence, so a pooled socket
    can be reset between two reads. Only this read is retried: resending a
    generation POST could open a second session.
    """
    try:
        return client.session(session_id)
    except requests.ConnectionError:
        return client.session(session_id)


def _snapshot_all(client) -> list[dict]:
    """Every session this run opened that the server still holds."""
    entries = [_snapshot(client, s) for s in sorted(_OPENED)]
    return [entry for entry in entries if entry is not None]


def _held(client) -> list[str]:
    return [s["session_id"] for s in _snapshot_all(client)]


def _wait_released(client, session_id: str, timeout: float = 60.0) -> bool:
    """Wait for the id to be released, which only happens once the worker has
    confirmed the session's state is gone."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _snapshot(client, session_id) is None:
            return True
        time.sleep(0.25)
    return False


def purge(client, when: str) -> None:
    """End every session this run opened that the server still holds.

    The scenarios run one after another against one server and most of them
    count sessions, so each has to start from an empty deployment and leave one
    behind. A session with a request in flight refuses the delete, so give it a
    moment to finish first.
    """
    for _ in range(60):
        entries = _snapshot_all(client)
        if not entries:
            return
        for entry in entries:
            if entry["closing"] or entry["active_request_ids"]:
                continue
            try:
                client.end_session(entry["session_id"])
            except requests.HTTPError:
                pass  # raced the teardown, or went busy; the retry handles it
        time.sleep(0.5)
    print(f"WARNING: sessions still held {when} the scenario: {_held(client)}")


class _InFlight:
    """A long request in a session, running in a thread.

    The checks about a session that is busy need one, and a server quick enough
    to finish it first must not turn them into failures — ``busy`` says whether
    the request was still running when it was looked at.
    """

    def __init__(self, client, session_id: str):
        self._client = client
        self.session_id = session_id
        self.error: Exception | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            _turn(
                self._client, LONG_TURN, tokens=LONG_TOKENS,
                resume_session=True, session_id=self.session_id,
            )
        except Exception as e:  # noqa: BLE001 -- reported, not handled
            self.error = e

    def __enter__(self) -> "_InFlight":
        self._thread.start()
        # let it reach the server and be registered as the session's live one
        deadline = time.time() + 15
        while time.time() < deadline:
            entry = _snapshot(self._client, self.session_id)
            if entry and entry["active_request_ids"]:
                break
            time.sleep(0.1)
        return self

    def busy(self) -> bool:
        entry = _snapshot(self._client, self.session_id)
        return bool(entry and entry["active_request_ids"])

    def __exit__(self, *exc):
        self._thread.join(timeout=300)
        return False


# ── scenarios ───────────────────────────────────────────────────────────────

def capacity_refuse(client, dep, ck) -> None:
    """A deployment at its cap refuses a new session and keeps the ones it has."""
    if dep.capacity_policy != "keep":
        raise Wrong(f"capacity_policy is {dep.capacity_policy!r}, want 'keep'")
    if dep.max_sessions > len(NUMBERS):
        raise Wrong(f"max_concurrent_sessions={dep.max_sessions} is too many")

    ids = [_start(client, NUMBERS[i]) for i in range(dep.max_sessions)]
    ck.that(
        sorted(_held(client)) == sorted(ids),
        f"the server holds its {dep.max_sessions} sessions",
        f"{_held(client)}",
    )
    ck.that(
        _refused(lambda: _turn(client, "Say OK.", start_session=True)) == 429,
        "a start past the cap is refused with a 429",
    )
    ck.that(
        NUMBERS[0] in _recall(client, ids[0]),
        "the sessions already open were left alone",
    )

    client.end_session(ids[0])
    ck.that(
        _wait_released(client, ids[0]),
        "ending a session releases its id",
    )
    freed = _turn(client, "Say OK.", start_session=True).session_id
    ck.that(bool(freed), "a start succeeds once a session has been ended")


def capacity_evict(client, dep, ck) -> None:
    """A deployment at its cap evicts its LRU idle session instead of refusing,
    and never evicts one with a request in flight."""
    if dep.capacity_policy != "evict":
        raise Wrong(f"capacity_policy is {dep.capacity_policy!r}, want 'evict'")
    if dep.max_sessions < 2:
        raise Wrong("max_concurrent_sessions must be at least 2 to have an LRU")

    # oldest first, so the first one opened is the least recently used
    ids = []
    for i in range(dep.max_sessions):
        ids.append(_start(client, NUMBERS[i]))
        time.sleep(0.2)

    newcomer = _turn(client, "Say OK.", start_session=True).session_id
    ck.that(bool(newcomer), "a start past the cap is admitted, not refused")
    victim, survivor = ids[0], ids[-1]
    ck.that(
        _wait_released(client, victim),
        "the least recently used idle session was torn down",
        f"holds {_held(client)}",
    )
    ck.that(
        _refused(
            lambda: _turn(
                client, RECALL, resume_session=True, session_id=victim,
            )
        ) == 404,
        "resuming the evicted session is a 404",
    )
    ck.that(
        NUMBERS[dep.max_sessions - 1] in _recall(client, survivor),
        "the session that survived kept its context",
    )

    # a session being written to right now is not a candidate, so a deployment
    # whose sessions are all in flight refuses even under `evict`
    live = [e["session_id"] for e in _snapshot_all(client) if not e["closing"]]
    flights = [_InFlight(client, s) for s in live]
    what = "a start is refused while every session is in flight"
    with contextlib.ExitStack() as stack:
        for flight in flights:
            stack.enter_context(flight)
        if len(flights) >= dep.max_sessions and all(f.busy() for f in flights):
            ck.that(
                _refused(
                    lambda: _turn(client, "Say OK.", start_session=True)
                ) == 429,
                what,
            )
        else:
            ck.skip(what, "the requests finished before the cap was reached")
    for flight in flights:
        if flight.error is not None:
            ck.that(False, "the in-flight turns completed", f"{flight.error}")


def ttl_idle(client, dep, ck) -> None:
    """An idle TTL expires a session, and traffic pushes it out."""
    if dep.ttl_mode != "idle":
        raise Wrong(f"ttl_mode is {dep.ttl_mode!r}, want 'idle'")
    if dep.timeout_s > 120:
        raise Wrong(f"default_timeout_s={dep.timeout_s} is too long to wait out")

    session_id = _start(client, NUMBERS[0])
    first = _snapshot(client, session_id)
    ck.that(
        first is not None and first["expires_in_s"] <= dep.timeout_s,
        "a new session expires within its timeout",
        f"{first and first['expires_in_s']}s of {dep.timeout_s}s",
    )

    # a turn well inside the TTL must push the deadline back out
    time.sleep(dep.timeout_s * 0.5)
    halfway = _snapshot(client, session_id)
    ck.that(
        halfway is not None, "the session is still live halfway through its TTL",
    )
    ck.that(
        NUMBERS[0] in _recall(client, session_id),
        "a turn inside the TTL is served from the session's state",
    )
    refreshed = _snapshot(client, session_id)
    ck.that(
        refreshed is not None
        and refreshed["expires_in_s"] > dep.timeout_s * 0.6,
        "serving a turn refreshes the idle deadline",
        f"{refreshed and refreshed['expires_in_s']}s left",
    )

    # and now let it go idle for a whole TTL
    print(f"  idling for {dep.timeout_s + 5:.0f}s", flush=True)
    time.sleep(dep.timeout_s + 5)
    ck.that(
        session_id not in _held(client),
        "an idle session is collected once its TTL passes",
        f"holds {_held(client)}",
    )
    ck.that(
        _refused(
            lambda: _turn(
                client, RECALL, resume_session=True, session_id=session_id,
            )
        ) == 404,
        "resuming an expired session is a 404",
    )


def ttl_absolute(client, dep, ck) -> None:
    """An absolute TTL expires a session however busy it has been."""
    if dep.ttl_mode != "absolute":
        raise Wrong(f"ttl_mode is {dep.ttl_mode!r}, want 'absolute'")
    if dep.timeout_s > 120:
        raise Wrong(f"default_timeout_s={dep.timeout_s} is too long to wait out")

    session_id = _start(client, NUMBERS[0])
    started = time.time()
    left: list[float] = []
    served = 0
    expiry_status = None
    # keep using it until it is taken away: the point is that traffic does not
    # save it, so the loop runs past the deadline on purpose
    while time.time() - started < dep.timeout_s + 15:
        entry = _snapshot(client, session_id)
        if entry is not None:
            left.append(entry["expires_in_s"])
        status = _refused(
            lambda: _turn(
                client, "Say OK.", resume_session=True, session_id=session_id,
            )
        )
        if status is not None:
            expiry_status = status
            break
        served += 1
        time.sleep(max(0.0, dep.timeout_s / 6))

    ck.that(served >= 2, "the session served several turns", f"{served} turns")
    ck.that(
        len(left) >= 2 and all(b < a + 0.5 for a, b in zip(left, left[1:], strict=False)),
        "the deadline is not refreshed by traffic",
        f"expires_in_s {[round(v, 1) for v in left]}",
    )
    ck.that(
        expiry_status in (404, 409),
        "a turn after the absolute deadline is refused",
        f"status {expiry_status}",
    )
    ck.that(
        time.time() - started >= dep.timeout_s,
        "it was not taken away before its deadline",
    )
    ck.that(
        _wait_released(client, session_id),
        "the expired session's id is released",
    )


def limits(client, dep, ck) -> None:
    """The refusals a client can provoke: bad flags, a session that is busy."""
    ck.that(
        _refused(
            lambda: _turn(
                client, "hi", start_session=True,
                session_timeout_s=dep.max_timeout_s + 1,
            )
        ) == 400,
        "asking for a longer TTL than the deployment allows is a 400",
    )
    ck.that(
        _refused(
            lambda: _turn(
                client, "hi", start_session=True, session_timeout_s=-1,
            )
        ) == 400,
        "a negative session_timeout_s is a 400",
    )
    ck.that(
        _refused(
            lambda: _turn(client, "hi", start_session=True, resume_session=True)
        ) == 400,
        "start_session with resume_session is a 400",
    )
    ck.that(
        _refused(lambda: _turn(client, "hi", resume_session=True)) == 400,
        "resume_session without a session_id is a 400",
    )
    ck.that(
        _refused(lambda: _turn(client, "hi", session_id="orphan")) == 400,
        "naming a session with neither flag is a 400",
    )
    ck.that(
        _refused(lambda: client.end_session("no-such-session")) == 404,
        "deleting an unknown session is a 404",
    )

    session_id = _start(client, NUMBERS[0])
    with _InFlight(client, session_id) as flight:
        if flight.busy():
            ck.that(
                _refused(
                    lambda: _turn(
                        client, RECALL, resume_session=True,
                        session_id=session_id,
                    )
                ) == 409,
                "resuming a session with a request in flight is a 409",
            )
            ck.that(
                _refused(lambda: client.end_session(session_id)) == 409,
                "deleting a session with a request in flight is a 409",
            )
        else:
            ck.skip(
                "a session with a request in flight refuses a second one",
                "the request finished before a second could be sent",
            )
    if flight.error is not None:
        ck.that(False, "the in-flight turn completed", f"{flight.error}")

    # and it is resumable again now that nothing is in flight
    ck.that(
        NUMBERS[0] in _recall(client, session_id),
        "the session is resumable once its request has finished",
    )
    client.end_session(session_id)


SCENARIOS = {
    "capacity_refuse": capacity_refuse,
    "capacity_evict": capacity_evict,
    "ttl_idle": ttl_idle,
    "ttl_absolute": ttl_absolute,
    "limits": limits,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", nargs="?", choices=sorted(SCENARIOS))
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument(
        "--config", type=Path, required=False,
        help="the deployment's config, read for the knobs the scenario needs",
    )
    ap.add_argument("--list", action="store_true", help="name the scenarios")
    args = ap.parse_args(argv)

    if args.list or not args.scenario:
        for name, fn in sorted(SCENARIOS.items()):
            print(f"{name:18} {(fn.__doc__ or '').splitlines()[0]}")
        return 0 if args.list else 2
    if args.config is None:
        ap.error("--config is required to run a scenario")

    dep = Deployment.read(args.config)
    client = MStarClient(args.url)
    if not client.health():
        print(f"no healthy server at {args.url}")
        return 1
    counts = client.session_counts()
    if counts["live"] or counts["closing"]:
        # their ids are not ours to see, so they can only be waited out
        print(
            f"WARNING: the server already holds {counts['live']} live and "
            f"{counts['closing']} closing sessions this run did not open; "
            "scenarios that count sessions may misread them"
        )

    print(f"scenario {args.scenario} against {args.config} ({dep})", flush=True)
    ck = Checks()
    try:
        SCENARIOS[args.scenario](client, dep, ck)
    finally:
        purge(client, "after")
    return ck.exit_code()


if __name__ == "__main__":
    sys.exit(main())
