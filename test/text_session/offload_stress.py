#!/usr/bin/env python3
"""Drive many sessions through a KV pool far too small for them, and check the
deployment neither hangs nor loses a session's context.

Meant for ``configs/text_session/tp2_offload.yaml``: parked sessions outnumber
the device pool many times over, so nearly every resume reloads from host and
nearly every admission offloads someone else's idle session. Under TP that is
the path where the ranks have to agree on which sessions are idle at every step;
when they don't, one rank comes up short of pages and the instance stops.

Two phases:

``exhaustion``  ``--sessions`` conversations of ``--turns`` turns each, run
                ``--concurrency`` at a time. Turn 1 states a number, the middle
                turns pad the context, the last asks for the number back.
``abort``       resume a session and cancel the request moments later, the
                window where the session's teardown races its next request.
                Every outcome but a hang is fine here; a 410 on the next
                resume is the expected one.

Then a fresh request has to complete: a deployment that hung answers nothing.
Exit status is nonzero on any timeout, server error or a failed probe.

    CONFIG=configs/text_session/tp2_offload.yaml CUDA_VISIBLE_DEVICES=0,1 \\
        bash test/text_session/launch_server.sh
    python test/text_session/offload_stress.py --url http://localhost:8000
"""

import argparse
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

from mstar import MStarClient

STATE = "Remember this number: {number}. Reply with just the word OK."
ASK = "What was the number I asked you to remember? Answer with digits only."
FILLER = (
    "Here is some unrelated text to read and then ignore: the lighthouse keeper "
    "logged the tides, the ferry ran late, and the gulls argued over the pier. "
)


class Tally:
    def __init__(self):
        self._lock = threading.Lock()
        self.counts: Counter = Counter()
        self.errors: list[str] = []

    def add(self, key: str, error: str | None = None) -> None:
        with self._lock:
            self.counts[key] += 1
            if error is not None and len(self.errors) < 20:
                self.errors.append(error)


def _kwargs(args) -> dict:
    # top_k=1: a missed recall is then the context, not sampling
    return {"max_output_tokens": args.max_output_tokens, "top_k": 1}


def _conversation(args, index: int, tally: Tally) -> None:
    client = MStarClient(args.url, timeout=args.request_timeout)
    rng = random.Random(args.seed * 100003 + index)
    number = str(rng.randint(1000000, 9999999))
    session_id = None
    try:
        first = client.generate(
            text=STATE.format(number=number), start_session=True, **_kwargs(args),
        )
        session_id = first.session_id
        for _ in range(args.turns - 2):
            client.generate(
                text=FILLER * args.filler + "Reply with just the word OK.",
                resume_session=True, session_id=session_id, **_kwargs(args),
            )
        answer = client.generate(
            text=ASK, resume_session=True, session_id=session_id, **_kwargs(args),
        ).text or ""
        tally.add("recalled" if number in answer else "missed")
    except requests.Timeout:
        tally.add("timeout", f"conversation {index}: timed out")
    except requests.HTTPError as e:
        status = e.response.status_code
        tally.add(f"http {status}", f"conversation {index}: {status} {e.response.text[:160]}")
    finally:
        if session_id is not None:
            try:
                client.end_session(session_id)
            except requests.RequestException:
                pass


def _abort_race(args, index: int, tally: Tally) -> None:
    client = MStarClient(args.url, timeout=args.request_timeout)
    try:
        first = client.generate(
            text=STATE.format(number=index), start_session=True, **_kwargs(args),
        )
    except requests.RequestException as e:
        tally.add("setup failed", f"abort {index}: {e}")
        return
    session_id = first.session_id
    # Non-streaming, so the server notices the cancel when the socket closes.
    cancelled = MStarClient(args.url, timeout=args.abort_after_s)
    try:
        cancelled.generate(
            text=FILLER * args.filler + ASK, resume_session=True,
            session_id=session_id, **_kwargs(args),
        )
        tally.add("finished before the cancel")
    except requests.Timeout:
        tally.add("cancelled")
    except requests.HTTPError as e:
        tally.add(f"cancel http {e.response.status_code}")
    # What the session does next is free to vary (live, 410, gone); it must
    # answer, and never with a server error.
    try:
        client.generate(
            text=ASK, resume_session=True, session_id=session_id, **_kwargs(args),
        )
        tally.add("resumed after")
    except requests.Timeout:
        tally.add("timeout", f"abort {index}: the resume after the cancel timed out")
    except requests.HTTPError as e:
        status = e.response.status_code
        tally.add(
            f"resume after: http {status}",
            None if status in (404, 409, 410) else f"abort {index}: {status} {e.response.text[:160]}",
        )
    try:
        client.end_session(session_id)
    except requests.RequestException:
        pass


def _run(phase: str, fn, args, n: int) -> Tally:
    tally = Tally()
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for future in [pool.submit(fn, args, i, tally) for i in range(n)]:
            future.result()
    print(f"[{phase}] {dict(tally.counts)} in {time.monotonic() - start:.1f}s")
    for error in tally.errors:
        print(f"  {error}")
    return tally


def _bad(tally: Tally) -> int:
    return sum(
        count for key, count in tally.counts.items()
        if key == "timeout" or (key.startswith(("http 5", "resume after: http 5")))
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--sessions", type=int, default=48)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--filler", type=int, default=12,
                    help="filler repeats per padding turn (~35 tokens each)")
    ap.add_argument("--aborts", type=int, default=64)
    ap.add_argument("--abort-after-s", type=float, default=0.15)
    ap.add_argument("--max-output-tokens", type=int, default=16)
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=1,
                    help="repeat both phases, against one server")
    args = ap.parse_args(argv)

    if not MStarClient(args.url).health():
        print(f"no healthy server at {args.url}")
        return 1
    bad = missed = 0
    for round_ in range(args.rounds):
        args.seed += round_
        exhaustion = _run("exhaustion", _conversation, args, args.sessions)
        bad += _bad(exhaustion)
        missed += exhaustion.counts["missed"]
        if args.aborts:
            bad += _bad(_run("abort", _abort_race, args, args.aborts))
    try:
        MStarClient(args.url, timeout=60).generate(text="Say OK.", **_kwargs(args))
        probe = True
    except requests.RequestException as e:
        print(f"[probe] a fresh request failed: {e}")
        probe = False
    print(f"timeouts/5xx: {bad}, missed recalls: {missed}, probe ok: {probe}")
    return 0 if bad == 0 and probe else 1


if __name__ == "__main__":
    sys.exit(main())
