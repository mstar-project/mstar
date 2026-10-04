#!/usr/bin/env python3
"""Structural proof that a session's KV really accumulates across requests.

Model-independent, unlike ``session_request.py``: it never reads the generated
text. The deployment gives each session ``max_state`` pages, so sending long
turns into one session must eventually push the held state over the budget. What
happens then is the deployment's ``overflow_policy``, and this script checks
whichever one the config names:

``error``
    the turn after the overflow is refused with the budget error. A session that
    quietly dropped its state could never trip it.
``clear``
    the turns keep succeeding and the session stays live, and the worker logged
    that it cleared the session. Pass ``--server-log`` for that last check.

    CUDA_VISIBLE_DEVICES=7 CONFIG=configs/text_session/keep_error.yaml \
        bash test/text_session/launch_server.sh
    python test/text_session/budget_probe.py --url http://localhost:8137 \
        --config configs/text_session/keep_error.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import requests
import yaml

from mstar import MStarClient

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs/test_text_session.yaml"
)
PAGE_SIZE = 128  # BAGEL's KV page, the unit max_state counts
# What the engine logs when it drops a session that outgrew its budget.
CLEAR_LOG = "over its budget"


def _kv_session_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    return config["sessions"]["resources"]["kv"]


def _cleared_in_log(path: Path, session_id: str) -> bool:
    """Did the worker log a clear for this session? Best effort: the log is
    written by another process, so give it a moment to arrive."""
    deadline = time.time() + 10
    while time.time() < deadline:
        text = path.read_text(errors="replace") if path.exists() else ""
        if CLEAR_LOG in text and session_id in text:
            return True
        time.sleep(0.5)
    return False


def _end_quietly(client, session_id: str | None) -> None:
    """Leave the deployment empty: the matrix job runs other scenarios against
    this server afterwards, and they count sessions."""
    if session_id is None:
        return
    try:
        client.end_session(session_id)
    except requests.HTTPError as e:
        print(f"could not end session {session_id}: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="the deployment's config, read for max_state and overflow_policy",
    )
    ap.add_argument(
        "--server-log", type=Path,
        help="the server's log, grepped for the clear under the clear policy",
    )
    ap.add_argument("--turn-words", type=int, default=3000)
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args(argv)

    kv = _kv_session_config(args.config)
    pages, policy = kv["max_state"], kv.get("overflow_policy", "clear")
    budget_tokens = pages * PAGE_SIZE
    client = MStarClient(args.url)
    prompt = " ".join(["context"] * args.turn_words) + "\nReply with just OK."

    if not client.health():
        print(f"no healthy server at {args.url}")
        return 1
    print(f"budget: {pages} pages (~{budget_tokens} tokens), policy {policy!r}")

    opened: list[str] = []
    try:
        return _probe(client, args, prompt, pages, policy, opened)
    finally:
        _end_quietly(client, opened[0] if opened else None)


def _probe(client, args, prompt, pages, policy, opened) -> int:
    """Send turns into one session until its state goes over the budget, and
    check what the configured policy does about it."""
    budget_tokens = pages * PAGE_SIZE
    session_id = None
    for turn in range(1, args.max_turns + 1):
        opening = (
            {"start_session": True} if session_id is None
            else {"resume_session": True, "session_id": session_id}
        )
        try:
            result = client.generate(text=prompt, max_output_tokens=4, **opening)
        except requests.HTTPError as e:
            body = e.response.text
            print(f"turn {turn}: HTTP {e.response.status_code}: {body[:300]}")
            if policy != "error":
                print(f"\nFAIL: the {policy!r} policy must not fail a turn.")
                return 1
            if "budget" not in body:
                return 1
            print(
                f"\nPASS: the session accumulated past its {pages} pages "
                f"({budget_tokens} tokens) and the next turn was refused, "
                f"on turn {turn}."
            )
            return 0
        session_id = session_id or result.session_id
        if session_id and not opened:
            opened.append(session_id)
        held = [s["session_id"] for s in client.sessions()]
        print(f"turn {turn}: ok, session {session_id}, server holds {held}")

    if policy == "error":
        print(
            f"\nINCONCLUSIVE: {args.max_turns} turns of ~{args.turn_words} "
            f"words did not reach {pages} pages. Raise --turn-words."
        )
        return 1

    # the clear policy: the session is still being served, and cleared quietly
    checks = {
        "every turn was served": True,
        "the session is still live": session_id in [
            s["session_id"] for s in client.sessions()
        ],
    }
    if args.server_log:
        checks["the worker logged the clear"] = _cleared_in_log(
            args.server_log, session_id,
        )
    else:
        print("no --server-log: not checking that a clear was logged")
    print()
    for what, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {what}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
