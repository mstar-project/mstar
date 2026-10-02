#!/usr/bin/env python3
"""Structural proof that a session's KV really accumulates across requests.

Model-independent, unlike ``session_request.py``: it never reads the generated
text. The deployment gives each session ``max_state`` pages with the ``error``
overflow policy, so sending long turns into one session must eventually push the
held state over the budget — and the turn after that must be refused with the
budget error. A session that quietly dropped its state could never trip it.

    CUDA_VISIBLE_DEVICES=7 bash test/text_session/launch_server.sh
    python test/text_session/budget_probe.py --url http://localhost:8137
"""

import argparse
import sys
from pathlib import Path

import requests
import yaml

from mstar import MStarClient

CONFIG = Path(__file__).resolve().parents[2] / "configs/test_text_session.yaml"
PAGE_SIZE = 128  # BAGEL's KV page, the unit max_state counts


def _budget_pages() -> int:
    config = yaml.safe_load(CONFIG.read_text())
    return config["sessions"]["resources"]["kv"]["max_state"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--turn-words", type=int, default=3000)
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args(argv)

    pages = _budget_pages()
    budget_tokens = pages * PAGE_SIZE
    client = MStarClient(args.url)
    prompt = " ".join(["context"] * args.turn_words) + "\nReply with just OK."

    if not client.health():
        print(f"no healthy server at {args.url}")
        return 1

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
            if "budget" in body:
                print(
                    f"\nPASS: the session accumulated past its {pages} pages "
                    f"({budget_tokens} tokens) and the next turn was refused, "
                    f"on turn {turn}."
                )
                return 0
            return 1
        session_id = session_id or result.session_id
        held = [s["session_id"] for s in client.sessions()]
        print(f"turn {turn}: ok, session {session_id}, server holds {held}")

    print(
        f"\nINCONCLUSIVE: {args.max_turns} turns of ~{args.turn_words} words did "
        f"not reach {pages} pages. Raise --turn-words."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
