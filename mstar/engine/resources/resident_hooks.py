"""What a resource overrides to take part in resident planning (`resident.py`)."""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    # `resident.py` imports `Resource`, which inherits from here
    from mstar.engine.resources.resident import (
        CapturedResidentStep,
        ResidentPlan,
        ResidentPlanner,
        ResidentStream,
        ResidentWrites,
    )
    from mstar.engine.resources.step import ResourceStep, StepContext


# `max_repeats` for a resource that has no limit of its own
ANY_REPEATS = 1 << 30


def no_resident_effect(*args) -> None:
    """Explicit choice for a binding operation with no state effect."""


def unbounded_repeats(step, ctx) -> int:
    """Explicit choice for an owner with no per-run capacity limit."""
    return ANY_REPEATS


def unbound_horizon(step, ctx) -> None:
    """Ordinary resident support without immutable dynamic-row addressing."""
    raise ValueError("this owner has no execution-time binding for newly admitted rows")


@dataclass(frozen=True)
class ResidentBinding:
    """One owner's complete resident execution contract, resolved before capture.

    All operations are required. An owner explicitly supplies a no-op only for
    effects it does not have. check_streams is pure and runs before bind allocates
    tables; check_horizon validates reversible admission and captured addressing
    before any owner reserves, including auxiliary capture owners with no
    mutation step (None). Those checks also enforce any auxiliary horizon
    capacity. Admission/commit/rollback remain Resource's one
    lifecycle. Actual-count settlement shares its rewind implementation.

    The worker retains request allocations and excludes their movement while
    readers exist; table pins and descriptor credits do not replace that hold.
    """

    check_streams: Callable[[str, Mapping[str, ResidentStream]], str | None]
    bind: Callable[[ResidentPlanner, str], None]
    row: Callable[[int, int, list[int], ResidentWrites], None]
    stage: Callable[[ResidentPlan], None]
    gather: Callable[[CapturedResidentStep], None]
    run_gather: Callable[[CapturedResidentStep], None]
    commit_in_graph: Callable[[CapturedResidentStep], None]
    advance: Callable[[ResidentPlan], None]
    check_horizon: Callable[[ResourceStep | None, StepContext], None]
    max_repeats: Callable[[ResourceStep, StepContext], int]
    rewind_run: Callable[[ResourceStep, StepContext, int, Collection], None]

    @property
    def inert(self) -> bool:
        """No planner-local table or execution effect to share between members."""
        return all(effect is no_resident_effect for effect in (
            self.bind, self.row, self.stage, self.gather, self.run_gather,
            self.commit_in_graph, self.advance, self.rewind_run,
        ))


class ResidentHooks:
    """A captured decode step reads per-request state off tables on the device;
    each owner of that state keeps its own table. A resource overrides these
    only once it has one."""

    def ask_repeats(self, walk: str, *, join_rows: bool = False) -> None:
        """Before capture: the config asks steps of ``walk`` to repeat, so
        build what serves that if there is a choice."""
        return

    def ask_stop_odds(self, walk: str, stop_ids) -> str | None:
        """Before capture: score ``stop_ids`` on each sample of ``walk``; why it can't, or None."""
        return f"resource: {type(self).__name__} samples nothing"

    def repeat_note(self, walk: str) -> str | None:
        """After capture: what ``ask_repeats`` changed for ``walk``, for its line."""
        return None

    def repeat_refusal(self, walk: str, host_step=None) -> str | None:
        """Why no step of ``walk`` may replay with no host work between
        replays, from what is fixed by the end of capture, or None if steps
        may. ``host_step``: this resource's part of the walk's step, for a walk
        planned on the host, whose one plan every replay reads; None for a
        resident walk. A resource refuses until it knows better: whatever it
        stages per step would go stale. Returning None promises that finite
        capacity is bounded by max_repeats, committed per-iteration accounting
        is reversible by rewind_run, and changing replay inputs are supplied
        by replay_columns or advanced by the capture. The defaults remain
        valid for resources with no such limit, accounting or changing input;
        a stateless/read-only resource need not override no-op hooks."""
        del walk, host_step
        return f"{type(self).__name__} keeps no per-replay support"

    def max_repeats(self, step, ctx) -> int:
        """How many times this step may replay, for a resource that doesn't
        refuse (`repeat_refusal`): what only this step decides, which can
        only narrow."""
        return ANY_REPEATS

    def rewind_run(self, step, ctx, replays: int, rows: Collection) -> None:
        """A run landed ``replays`` iterations short of what its commit counted
        (``device_run``): take them back from what is counted per iteration,
        for ``rows`` only (the step's rows still the requests it counted for)."""
        return

    def replay_columns(self, step, ctx, repeats: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """What each replay of a step that repeats reads and its capture doesn't
        advance itself: ``(column, target)`` pairs, ``column[:, i]`` going into
        ``target`` before replay i. Queued on the step's stream after its first."""
        return []

    def resident_binding(self) -> ResidentBinding | str:
        """Resolve the owner's resident operations once, before table allocation.

        A resource without a complete binding stays host-planned. The binding
        separately checks whether a resident step can admit new rows beside
        already submitted captures; resident execution alone does not imply it.
        """
        return f"{type(self).__name__} keeps no resident table"
