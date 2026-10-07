#!/usr/bin/env python3
"""Drive a persistent session end to end and check it actually carried state.

Launch the server first (test/text_session/launch_server.sh), then:

    python test/text_session/session_request.py

The proof is the second turn: it is sent on its own, with nothing about the
first turn in its prompt, so an answer that refers back to the first turn can
only have come from the KV the session kept. The control run sends the same
second turn with no session and should not know it.

The tail of the script walks the teardown barrier — delete the session, wait for
the id to disappear (which only happens once the worker has confirmed its state
is gone), and check that resuming it is then a 404.

    python test/text_session/session_request.py --url http://localhost:8000
    python test/text_session/session_request.py --no-control
"""

import argparse
import sys
import time

import requests

from mstar import MStarClient

FIRST = "Remember this number: 8675309. Reply with just the word OK."
SECOND = "What was the number I asked you to remember? Answer with digits only."
NUMBER = "8675309"


def _say(step: str, detail: str = "") -> None:
    print(f"[{step}] {detail}" if detail else f"[{step}]", flush=True)


def _status_of(call) -> int | None:
    """The HTTP status a refused call came back with, or None if it succeeded."""
    try:
        call()
    except requests.HTTPError as e:
        return e.response.status_code
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument(
        "--no-control", action="store_true",
        help="skip the sessionless run that shows the answer needs the session",
    )
    args = parser.parse_args(argv)

    client = MStarClient(args.url)
    if not client.health():
        _say("FAIL", f"no healthy server at {args.url}")
        return 1

    checks: dict[str, bool] = {}

    # --- turn one opens the session -------------------------------------
    first = client.generate(
        text=FIRST, start_session=True, session_timeout_s=300,
        max_output_tokens=args.max_output_tokens, top_k=1,
    )
    session_id = first.session_id
    if not session_id:
        _say("FAIL", "the server returned no session_id for start_session=True")
        return 1
    _say("turn 1", f"session {session_id}: {first.text!r}")
    _say("sessions", f"{client.session_counts()}, this one: {client.session(session_id)}")

    # --- turn two resumes it; its prompt says nothing about turn one -----
    second = client.generate(
        text=SECOND, resume_session=True, session_id=session_id,
        max_output_tokens=args.max_output_tokens, top_k=1,
    )
    _say("turn 2", f"{second.text!r}")
    checks["the session carried the number over"] = NUMBER in (second.text or "")

    if not args.no_control:
        control = client.generate(
            text=SECOND, max_output_tokens=args.max_output_tokens, top_k=1,
        )
        _say("control (no session)", f"{control.text!r}")
        checks["a sessionless request does not know it"] = (
            NUMBER not in (control.text or "")
        )

    # --- validation a client can see ------------------------------------
    checks["resuming an unknown session is a 404"] = _status_of(
        lambda: client.generate(
            text="hi", resume_session=True, session_id="no-such-session",
        )
    ) == 404
    checks["starting a session whose id is taken is a 409"] = _status_of(
        lambda: client.generate(
            text="hi", start_session=True, session_id=session_id,
        )
    ) == 409

    # --- teardown, all the way to the worker ----------------------------
    _say("delete", f"{client.end_session(session_id)}")
    deadline = time.time() + 30
    while time.time() < deadline:
        if client.session(session_id) is None:
            break
        time.sleep(0.2)
    # the id is only released once the worker confirms the state is gone
    checks["the id is released after the teardown ack"] = (
        client.session(session_id) is None
    )
    checks["resuming it afterwards is a 404"] = _status_of(
        lambda: client.generate(
            text="hi", resume_session=True, session_id=session_id,
        )
    ) == 404

    print()
    for what, ok in checks.items():
        _say("PASS" if ok else "FAIL", what)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
