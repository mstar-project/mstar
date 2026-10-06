#!/usr/bin/env python3
"""Structural proof that a session's KV really accumulates across requests.

Model-independent, unlike ``session_request.py``: it never reads the generated
text. The deployment gives each session ``max_state`` pages, so sending long
turns into one session must eventually push the held state over the budget --
and the turn after that must be refused with the budget error. A session that
quietly dropped its state between turns could never trip it.

    CUDA_VISIBLE_DEVICES=7 CONFIG=configs/text_session/capacity_keep.yaml \
        bash test/text_session/launch_server.sh
    python test/text_session/budget_probe.py --url http://localhost:8137 \
        --config configs/text_session/capacity_keep.yaml
"""

import argparse
import sys
from pathlib import Path

import requests
import yaml

from mstar import MStarClient

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs/test_text_session.yaml"
)
PAGE_SIZE = 128  # BAGEL's KV page, the unit max_state counts


def _kv_session_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    return config["sessions"]["resources"]["kv"]


def _end_quietly(client, session_id: str | None) -> None:
    """Leave the deployment empty: the matrix job runs other scenarios against
    this server afterwards, and they count sessions."""
    if session_id is None:
        return
    try:
        client.end_session(session_id)
    except requests.HTTPError as e:
        # the budget error already took the session with it: a failed request
        # ends its session, so a 404 here is the expected outcome
        if e.response.status_code != 404:
            print(f"could not end session {session_id}: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="the deployment's config, read for the kv budget",
    )
    ap.add_argument("--turn-words", type=int, default=3000)
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args(argv)

    pages = _kv_session_config(args.config)["max_state"]
    budget_tokens = pages * PAGE_SIZE
    client = MStarClient(args.url)
    prompt = " ".join(["context"] * args.turn_words) + "\nReply with just OK."

    if not client.health():
        print(f"no healthy server at {args.url}")
        return 1
    print(f"budget: {pages} pages (~{budget_tokens} tokens)")

    opened: list[str] = []
    try:
        return _probe(client, args, prompt, pages, opened)
    finally:
        _end_quietly(client, opened[0] if opened else None)


def _probe(client, args, prompt, pages, opened) -> int:
    """Send turns into one session until its state goes over the budget, which
    must get the turn after it refused."""
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
            if "budget" not in body:
                print("\nFAIL: refused, but not for the state budget.")
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

    print(
        f"\nINCONCLUSIVE: {args.max_turns} turns of ~{args.turn_words} words "
        f"did not reach {pages} pages. Raise --turn-words."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
