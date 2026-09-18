"""Generate, replay and shrink cases, and keep the corpus.

No code in this module knows what it tests. It knows three things only:

* how to make a case from a seed
* how to tell if a case still fails in the same way
* how to make a failed case small enough to read
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from fuzzer.common.case import Case, Failure, Op, failure_from
from fuzzer.common.machine import StateMachine

CORPUS_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def run_case(machine_cls: type[StateMachine], case: Case) -> Failure | None:
    """Replay one case.

    Returns the failure. Returns None if the case passed.
    """
    try:
        machine = machine_cls(case.config)
    except Exception as exc:  # noqa: BLE001 - a bad config is a finding too
        return failure_from(exc, -1)

    for index, op in enumerate(case.ops):
        try:
            machine.execute(op)
            machine.check()
        except Exception as exc:  # noqa: BLE001
            return failure_from(exc, index)
    try:
        machine.final_check()
    except Exception as exc:  # noqa: BLE001
        return failure_from(exc, len(case.ops))
    return None


def generate(
    machine_cls: type[StateMachine], seed: int, num_ops: int,
) -> tuple[Case, Failure | None]:
    """Build and run a case in one pass.

    Generation interleaves with execution, because ``gen_op`` reads live
    state. Most generated ops then apply. The driver records each op that it
    runs, so the case replays on its own.
    """
    rng = random.Random(seed)
    config = machine_cls.gen_config(rng)
    case = Case(machine_cls.name, config, [], {"seed": seed})

    try:
        machine = machine_cls(config)
    except Exception as exc:  # noqa: BLE001
        return case, failure_from(exc, -1)

    for index in range(num_ops):
        op = machine.gen_op(rng)
        case.ops.append(op)
        try:
            machine.execute(op)
            machine.check()
        except Exception as exc:  # noqa: BLE001
            return case, failure_from(exc, index)
    try:
        machine.final_check()
    except Exception as exc:  # noqa: BLE001
        return case, failure_from(exc, len(case.ops))
    return case, None


# ---------------------------------------------------------------------------
# shrinking
# ---------------------------------------------------------------------------

def _still_fails(
    machine_cls: type[StateMachine], case: Case, signature: tuple,
) -> bool:
    failure = run_case(machine_cls, case)
    return failure is not None and failure.signature == signature


def shrink(
    machine_cls: type[StateMachine],
    case: Case,
    signature: tuple,
    max_rounds: int = 40,
) -> Case:
    """Delta-debug the op list, then the individual ops, then the config.

    The shrinker runs to a fixpoint, not to a step budget. Every candidate
    must fail with the *same signature*. Without that rule, a shrink step can
    find a different bug and move the search to it.
    """
    best = case

    for _ in range(max_rounds):
        start_len = len(best.ops)
        start_config = dict(best.config)

        # 1. drop contiguous runs, coarse to fine
        size = max(1, len(best.ops) // 2)
        while size >= 1:
            index = 0
            while index < len(best.ops):
                trimmed = best.ops[:index] + best.ops[index + size:]
                if trimmed and _still_fails(
                    machine_cls, best.replace_ops(trimmed), signature
                ):
                    best = best.replace_ops(trimmed)
                else:
                    index += size
            size //= 2

        # 2. simplify what is left, op by op
        index = 0
        while index < len(best.ops):
            for candidate in machine_cls.simplify_op(best.ops[index]):
                ops = list(best.ops)
                ops[index] = candidate
                if _still_fails(machine_cls, best.replace_ops(ops), signature):
                    best = best.replace_ops(ops)
                    break
            index += 1

        # 3. shrink the config last: most likely to change the bug
        for candidate_config in machine_cls.shrink_config(best.config):
            if _still_fails(
                machine_cls, best.replace_config(candidate_config), signature
            ):
                best = best.replace_config(candidate_config)
                break

        if len(best.ops) == start_len and best.config == start_config:
            break

    best.notes = dict(case.notes)
    best.notes["signature"] = list(signature)
    return best


# ---------------------------------------------------------------------------
# searching
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """What one search over a range of seeds found."""

    cases_run: int
    elapsed: float
    failures: list[tuple[Case, Failure]]

    @property
    def ok(self) -> bool:
        return not self.failures


def search(
    machine_cls: type[StateMachine],
    seeds: Iterable[int],
    num_ops: int = 60,
    time_budget: float | None = None,
    do_shrink: bool = True,
    stop_after: int | None = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> SearchResult:
    """Run seeds until the budget runs out, shrinking whatever fails.

    Distinct signatures only: a generator that trips one invariant on 900 of
    1000 seeds reports one bug, not 900.

    The search stops at the first failure, unless ``stop_after`` is None. A
    case stops at its first failed op. Thus a frequent failure hides the
    invariants that come later.
    """
    started = time.monotonic()
    seen: set[tuple] = set()
    failures: list[tuple[Case, Failure]] = []
    count = 0

    for seed in seeds:
        if time_budget is not None and time.monotonic() - started > time_budget:
            break
        count += 1
        case, failure = generate(machine_cls, seed, num_ops)
        if on_progress is not None:
            on_progress(count, len(failures))
        if failure is None or failure.signature in seen:
            continue
        seen.add(failure.signature)
        # Trim to the op that failed before handing it to the shrinker.
        trimmed = case.replace_ops(case.ops[: failure.op_index + 1])
        if not _still_fails(machine_cls, trimmed, failure.signature):
            trimmed = case
        if do_shrink:
            trimmed = shrink(machine_cls, trimmed, failure.signature)
        failures.append((trimmed, run_case(machine_cls, trimmed) or failure))
        if stop_after is not None and len(failures) >= stop_after:
            break

    return SearchResult(count, time.monotonic() - started, failures)


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

def corpus_dir(tier: str, machine_name: str) -> Path:
    """Give the directory that holds the saved cases of one machine."""
    return CORPUS_ROOT / tier / "corpus" / machine_name


def load_corpus(tier: str, machine_name: str) -> list[tuple[Path, Case]]:
    """Read every saved case of one machine, in the order of the file names."""
    directory = corpus_dir(tier, machine_name)
    if not directory.is_dir():
        return []
    return [(path, Case.load(path)) for path in sorted(directory.glob("*.json"))]


def save_to_corpus(tier: str, case: Case, label: str) -> Path:
    """Write one case to the corpus of its machine. ``label`` names the file."""
    directory = corpus_dir(tier, case.machine)
    path = directory / f"{label}.json"
    case.save(path)
    return path


__all__ = [
    "Case",
    "Failure",
    "Op",
    "SearchResult",
    "corpus_dir",
    "generate",
    "load_corpus",
    "run_case",
    "save_to_corpus",
    "search",
    "shrink",
]
