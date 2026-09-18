"""The unit of work: one fuzz case that replays, and the record of a failure.

A case is a program, not a value. It holds a config and an ordered list of ops
that drive one state machine. The corpus files, the shrinker and the CI replay
all work on this pair. A failure report is therefore always a short list of ops
that a person can read.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _freeze(value: Any) -> Any:
    """Turn lists back into tuples, so a replayed op equals a generated one."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return {k: _freeze(v) for k, v in sorted(value.items())}
    return value


@dataclass(frozen=True)
class Op:
    """One transition. ``args`` is positional and JSON-able; ``execute``
    interprets it."""

    kind: str
    args: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "args", _freeze(self.args))

    def __repr__(self) -> str:
        inner = ", ".join(repr(a) for a in self.args)
        return f"{self.kind}({inner})"

    def to_json(self) -> list:
        return [self.kind, list(self.args)]

    @staticmethod
    def from_json(blob: list) -> "Op":
        return Op(blob[0], _freeze(blob[1]))


@dataclass
class Case:
    """One replayable program: a config and the ops that drive a machine.

    ``machine`` is the name the tier registry maps back to a class.
    """

    machine: str
    config: dict
    ops: list[Op] = field(default_factory=list)
    # Free-form data: the seed of the case, the signature that the shrinker
    # kept, and the reason to keep the case in the corpus.
    notes: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "machine": self.machine,
            "config": self.config,
            "ops": [op.to_json() for op in self.ops],
            "notes": self.notes,
        }

    @staticmethod
    def from_json(blob: dict) -> "Case":
        return Case(
            machine=blob["machine"],
            config=blob["config"],
            ops=[Op.from_json(o) for o in blob["ops"]],
            notes=blob.get("notes", {}),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")

    @staticmethod
    def load(path: Path) -> "Case":
        return Case.from_json(json.loads(Path(path).read_text()))

    def replace_ops(self, ops: list[Op]) -> "Case":
        return Case(self.machine, dict(self.config), list(ops), dict(self.notes))

    def replace_config(self, config: dict) -> "Case":
        return Case(self.machine, config, list(self.ops), dict(self.notes))

    def pretty(self) -> str:
        lines = [f"machine: {self.machine}", f"config:  {self.config}", "ops:"]
        lines += [f"  {i:3d}  {op!r}" for i, op in enumerate(self.ops)]
        return "\n".join(lines)


class InvariantError(AssertionError):
    """The system under test broke an invariant that it must hold.

    ``invariant`` is a stable id. A message carries generated request IDs and
    drifts across a shrink step, so only the id can be part of a signature.
    """

    def __init__(self, invariant: str, message: str):
        super().__init__(f"[{invariant}] {message}")
        self.invariant = invariant
        self.message = message


@dataclass
class Failure:
    """What a failed case reports back. ``op_index`` is -1 if the machine
    failed while it was built."""

    signature: tuple
    exc_type: str
    message: str
    op_index: int
    tb: str

    def __str__(self) -> str:
        return f"{self.exc_type} at op {self.op_index}: {self.message}"


def failure_from(exc: BaseException, op_index: int) -> Failure:
    """Build a signature that survives shrinking.

    For an ``InvariantError`` the signature is (type, invariant id). For an
    error that mstar raised, it is the deepest frame inside the repo. That
    frame stays put while the shrinker removes the data around it.
    """
    if isinstance(exc, InvariantError):
        sig = ("invariant", exc.invariant)
    else:
        frames = traceback.extract_tb(exc.__traceback__)
        site = ""
        for frame in reversed(frames):
            # Use the deepest frame that is not part of the harness.
            if "/fuzzer/" not in frame.filename:
                site = f"{Path(frame.filename).name}:{frame.lineno}"
                break
        sig = (type(exc).__name__, site)
    return Failure(
        signature=sig,
        exc_type=type(exc).__name__,
        message=str(exc),
        op_index=op_index,
        tb="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )
