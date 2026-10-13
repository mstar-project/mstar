#!/usr/bin/env python3
"""How reliably does a resumed session carry its context? Measured, not sampled.

``session_request.py`` sends one two-turn conversation, which cannot tell a
session that lost its state from a small model that answered badly. This repeats
the recall many times and puts two controls beside it:

``session``    turn 1 states a fact, a resumed turn 2 asks for it back.
``stateless``  one request whose prompt holds both turns. The model's own ceiling
               on this question -- no session involved.
``cold``       the recall question alone. The floor: anything it "recalls" here
               it guessed.

A session rate near the stateless rate means the handover is sound and what is
left is the model. A session rate near the cold rate means the context is not
reaching the model.

    CUDA_VISIBLE_DEVICES=7 bash test/text_session/launch_server.sh
    python test/text_session/recall_stress.py --url http://localhost:8137 \
        --trials 10 --greedy
    python test/text_session/recall_stress.py --url http://localhost:8137 \
        --multi 5          # multi-turn: three facts, recalled after three turns
"""

import argparse
import random
import sys

import requests

from mstar import MStarClient

STATE = "Remember this number: {number}. Reply with just the word OK."
ASK = "What was the number I asked you to remember? Answer with digits only."
# Three facts stated over three turns, then asked for in the order they were
# given, so a failure says at what depth the context stopped being there.
FACTS = (
    ("number", "Remember this number: {number}. Reply with just the word OK.",
     "What was the number I asked you to remember? Answer with digits only."),
    ("word", "The code word is {word}. Reply with just the word OK.",
     "What is the code word? Answer with one word."),
    ("color", "My favorite color is {color}. Reply with just the word OK.",
     "What is my favorite color? Answer with one word."),
)
WORDS = ("banana", "trombone", "lighthouse", "pepper", "walrus", "cactus")
COLORS = ("teal", "magenta", "indigo", "amber", "crimson", "olive")


def _facts(rng) -> dict:
    return {
        "number": str(rng.randint(1000000, 9999999)),
        "word": rng.choice(WORDS),
        "color": rng.choice(COLORS),
    }


class Runner:
    def __init__(self, client, tokens: int, greedy: bool):
        self.client = client
        self.kwargs = {"max_output_tokens": tokens}
        if greedy:
            # top_k=1 rather than temperature=0: unambiguously greedy whatever
            # the sampler does with a zero temperature
            self.kwargs["top_k"] = 1
        self.errors: list[str] = []

    def turn(self, text, **session) -> str:
        try:
            return self.client.generate(text=text, **session, **self.kwargs).text or ""
        except requests.HTTPError as e:
            self.errors.append(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
            return ""

    def session_arm(self, facts: dict) -> bool:
        """Turn 1 states the fact; a resumed turn 2 asks for it."""
        first = self.client.generate(
            text=STATE.format(**facts), start_session=True, **self.kwargs,
        )
        session_id = first.session_id
        if not session_id:
            self.errors.append("no session_id for start_session=True")
            return False
        try:
            answer = self.turn(ASK, resume_session=True, session_id=session_id)
        finally:
            try:
                self.client.end_session(session_id)
            except requests.HTTPError:
                pass
        return facts["number"] in answer

    def stateless_arm(self, facts: dict) -> bool:
        """Both turns in one prompt: what the model can do without a session."""
        return facts["number"] in self.turn(
            f"{STATE.format(**facts)}\nOK\n{ASK}"
        )

    def cold_arm(self, facts: dict) -> bool:
        """The question alone: whatever this scores, it guessed."""
        return facts["number"] in self.turn(ASK)

    def multi_arm(self, facts: dict) -> list[bool]:
        """Three facts over three turns, then each asked for in turn."""
        first = self.client.generate(
            text=FACTS[0][1].format(**facts), start_session=True, **self.kwargs,
        )
        session_id = first.session_id
        if not session_id:
            self.errors.append("no session_id for start_session=True")
            return [False] * len(FACTS)
        got: list[bool] = []
        try:
            for _, state, _ in FACTS[1:]:
                self.turn(
                    state.format(**facts), resume_session=True,
                    session_id=session_id,
                )
            for key, _, ask in FACTS:
                answer = self.turn(
                    ask, resume_session=True, session_id=session_id,
                ).lower()
                got.append(facts[key].lower() in answer)
        finally:
            try:
                self.client.end_session(session_id)
            except requests.HTTPError:
                pass
        return got


    def loose_arm(self, rng) -> dict[str, bool]:
        """A conversational follow-up, not a crisp fact.

        Two things a resumed turn can get wrong that a number recall does not
        show: losing the reference entirely, and losing track of who said what.
        How a model renders a resuming turn decides both, so they are scored.
        """
        pet, other = rng.sample(list(WORDS), 2)
        name = rng.choice(("Naomi", "Priya", "Dolores", "Wen"))
        got = {}

        first = self.client.generate(
            text=f"I have two cats, {pet.title()} and {other.title()}.",
            start_session=True, **self.kwargs,
        )
        session_id = first.session_id
        answer = self.turn(
            "What did I just tell you about?", resume_session=True,
            session_id=session_id,
        ).lower()
        got["reference kept"] = pet in answer or "cat" in answer
        self.client.end_session(session_id)

        first = self.client.generate(
            text=f"My name is {name} and I am debugging an inference engine.",
            start_session=True, **self.kwargs,
        )
        session_id = first.session_id
        answer = self.turn(
            "What is my name?", resume_session=True, session_id=session_id,
        ).lower()
        got["name recalled"] = name.lower() in answer
        # the model answering as though the user's name were its own
        got["roles kept straight"] = not any(
            claim in answer
            for claim in (f"i'm {name.lower()}", f"i am {name.lower()}",
                          f"my name is {name.lower()}")
        )
        self.client.end_session(session_id)
        return got


def _rate(hits: int, n: int) -> str:
    return f"{hits}/{n} ({100.0 * hits / n:.0f}%)" if n else "n/a"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--trials", type=int, default=10,
                    help="two-turn recalls, each with its own session")
    ap.add_argument("--multi", type=int, default=0,
                    help="multi-turn conversations: 3 facts, then 3 recalls")
    ap.add_argument("--loose", type=int, default=0,
                    help="conversational follow-ups: reference and role scoring")
    ap.add_argument("--max-output-tokens", type=int, default=24)
    ap.add_argument("--greedy", action="store_true",
                    help="top_k=1, so a flaky result is not sampling noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kwarg", action="append", default=[], metavar="K=V",
                    help="an extra model_kwarg, e.g. --kwarg temperature=0.2")
    args = ap.parse_args(argv)

    client = MStarClient(args.url)
    if not client.health():
        print(f"no healthy server at {args.url}")
        return 1
    rng = random.Random(args.seed)
    run = Runner(client, args.max_output_tokens, args.greedy)
    for pair in args.kwarg:
        key, _, value = pair.partition("=")
        run.kwargs[key] = value
    print(f"sampling: {'greedy (top_k=1)' if args.greedy else 'the deployment default'}")

    scores = {"session": 0, "stateless": 0, "cold": 0}
    for trial in range(1, args.trials + 1):
        facts = _facts(rng)
        got = {
            "session": run.session_arm(facts),
            "stateless": run.stateless_arm(facts),
            "cold": run.cold_arm(facts),
        }
        for arm, ok in got.items():
            scores[arm] += ok
        print(
            f"trial {trial:3d} {facts['number']}: "
            + "  ".join(
                f"{arm}={'hit ' if ok else 'miss'}" for arm, ok in got.items()
            ),
            flush=True,
        )

    depth = [0] * len(FACTS)
    for trial in range(1, args.multi + 1):
        facts = _facts(rng)
        got = run.multi_arm(facts)
        for i, ok in enumerate(got):
            depth[i] += ok
        print(
            f"multi {trial:3d}: "
            + "  ".join(
                f"{FACTS[i][0]}({'hit ' if ok else 'miss'})"
                for i, ok in enumerate(got)
            ),
            flush=True,
        )

    loose: dict[str, int] = {}
    for trial in range(1, args.loose + 1):
        got = run.loose_arm(rng)
        for what, ok in got.items():
            loose[what] = loose.get(what, 0) + ok
        print(
            f"loose {trial:3d}: "
            + "  ".join(f"{w}({'hit ' if ok else 'miss'})" for w, ok in got.items()),
            flush=True,
        )

    print()
    if args.trials:
        print(f"session   {_rate(scores['session'], args.trials)}  "
              "resumed turn recalls turn 1")
        print(f"stateless {_rate(scores['stateless'], args.trials)}  "
              "both turns in one prompt (the model's ceiling)")
        print(f"cold      {_rate(scores['cold'], args.trials)}  "
              "the question alone (the floor)")
    if args.multi:
        print("\nmulti-turn, 3 facts stated then asked after 3 more turns:")
        for i, (key, _, _) in enumerate(FACTS):
            print(f"  {key:7} stated in turn {i + 1}: {_rate(depth[i], args.multi)}")
    if args.loose:
        print("\nconversational follow-ups:")
        for what, hits in loose.items():
            print(f"  {what:20} {_rate(hits, args.loose)}")
    if run.errors:
        print(f"\n{len(run.errors)} request error(s):")
        for error in run.errors[:10]:
            print(f"  {error}")
    # the session arm should not be worse than the floor; the ceiling is the
    # model's problem, not the session machinery's
    if args.trials and scores["session"] <= scores["cold"] < scores["stateless"]:
        print("\nthe session arm is no better than the floor: the context is "
              "not reaching the model")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
