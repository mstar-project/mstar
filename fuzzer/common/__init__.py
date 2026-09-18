"""The shared harness: cases, state machines, generation and shrinking.

This package uses the standard library only, so tier 0 runs on a CI machine
with nothing installed but the repository itself. Tier 1 and tier 2 will use
the same driver once their machines exist.
"""

from fuzzer.common.case import Case, Failure, InvariantError, Op
from fuzzer.common.driver import (
    SearchResult,
    corpus_dir,
    generate,
    load_corpus,
    run_case,
    save_to_corpus,
    search,
    shrink,
)
from fuzzer.common.machine import StateMachine, require

__all__ = [
    "Case",
    "Failure",
    "InvariantError",
    "Op",
    "SearchResult",
    "StateMachine",
    "corpus_dir",
    "generate",
    "load_corpus",
    "require",
    "run_case",
    "save_to_corpus",
    "search",
    "shrink",
]
