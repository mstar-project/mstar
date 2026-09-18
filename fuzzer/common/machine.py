"""The contract that each fuzzable state machine implements."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar

from fuzzer.common.case import InvariantError, Op

__all__ = ["StateMachine", "InvariantError", "require"]


def require(invariant: str, condition: bool, message: str) -> None:
    """Fail the case if ``condition`` is false. ``invariant`` is the stable
    name the shrinker uses as the signature, so keep it unique."""
    if not condition:
        raise InvariantError(invariant, message)


class StateMachine(ABC):
    """One system under test, driven by a generated list of ops.

    A subclass must hold two properties, or the shrinker breaks:

    * **Deterministic.** The same config and ops give the same result.
      Replace anything that reads the clock, the address space or a global.
    * **Tolerant.** The shrinker deletes ops. Thus ``execute`` receives ops
      that the generator made for a state that no longer exists. Skip an op
      that no longer applies. Only the system under test may fail a case.
    """

    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def gen_config(cls, rng: random.Random) -> dict:
        """Sample the shape of the system: the sizes, the topology, the limits."""

    @abstractmethod
    def __init__(self, config: dict) -> None:
        """Build a fresh system under test from ``config``. No rng here:
        everything random is already in the config or in an op."""

    @abstractmethod
    def gen_op(self, rng: random.Random) -> Op:
        """Sample the next op. May read live state to prefer ops that mean
        something now; must still replay out of context."""

    @abstractmethod
    def execute(self, op: Op) -> None:
        """Apply ``op`` to the system under test and to the shadow model."""

    # These two methods are optional on purpose. A machine can do all of its
    # checks inside `execute`. For example, the graph machine checks the
    # result of each call at the call site. Thus these methods do nothing by
    # default, and they are not abstract.

    def check(self) -> None:
        """Check the invariants that hold after each op."""
        return

    def final_check(self) -> None:
        """Check the invariants that hold only at quiesce.

        The machine first gives back every resource. Conservation and the
        absence of leaks are the properties to check here.
        """
        return

    # -- shrinking hooks -----------------------------------------------------

    @classmethod
    def simplify_op(cls, op: Op) -> Iterator[Op]:
        """Simpler ops to try in place of ``op``. Default: pull each integer
        argument toward zero."""
        for i, arg in enumerate(op.args):
            if isinstance(arg, bool) or not isinstance(arg, int):
                continue
            for candidate in (0, 1, arg // 2):
                if candidate < arg:
                    args = list(op.args)
                    args[i] = candidate
                    yield Op(op.kind, tuple(args))

    @classmethod
    def shrink_config(cls, config: dict) -> Iterator[dict]:
        """Smaller configs to try. Default: none, since changing the topology
        usually changes which bug you are looking at."""
        return iter(())
