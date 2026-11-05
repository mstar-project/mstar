"""What a resource overrides to take part in resident planning (`resident.py`)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # `resident.py` imports `Resource`, which inherits from here
    from mstar.engine.resources.resident import (
        CapturedResidentStep,
        ResidentPlan,
        ResidentPlanner,
        ResidentWrites,
    )


# `max_repeats` for a resource that has no limit of its own
ANY_REPEATS = 1 << 30


class ResidentHooks:
    """A captured decode step reads per-request state off tables on the device;
    each owner of that state keeps its own table. A resource overrides these
    only once it has one."""

    def repeat_refusal(self) -> str | None:
        """Why no step may replay with no host work between replays, from what
        is fixed by the end of capture, or None if steps may. A resource refuses
        until it knows better: whatever it stages per step would go stale."""
        return f"{type(self).__name__} keeps no per-replay support"

    def max_repeats(self, step, ctx) -> int:
        """How many times this step may replay, for a resource that doesn't
        refuse (`repeat_refusal`): what only this step decides, which can
        only narrow."""
        return ANY_REPEATS

    def next_replay(self, step, ctx) -> None:
        """Queue, on the step's stream, what the next replay of a step that
        repeats reads and its capture doesn't advance itself."""
        return

    def resident_refusal(self) -> str | None:
        """Why a node holding this resource can't run on resident tables, or None if it can."""
        return f"{type(self).__name__} keeps no resident table"

    def bind_resident(self, planner: ResidentPlanner, key: str) -> None:
        """Declare this resource's tables on the node's planner, once, at load.
        ``key`` is its resource key, under which each step and capture name its
        stream (`ResidentWrites.streams`, `CapturedResidentStep.streams`)."""
        return

    def resident_row(self, rid: int, slot: int, row: list[int], writes: ResidentWrites) -> None:
        """Set this resource's cells of ``rid``'s row: it is new or stale. ``row``
        has the planner's narrow columns; the owner of a wide table queues its
        used cells as a run of their own (`ResidentWrites.run`)."""
        return

    def resident_stage(self, plan: ResidentPlan) -> None:
        """Queue this step's per-row changes on ``plan.writes``. Host work only:
        this can run beside a live graph."""
        return

    def resident_gather(self, step: CapturedResidentStep) -> None:
        """Record, inside the capture, the reads that fill what the step's forward
        addresses. Runs at warmup and capture only; a replay repeats the kernels.
        An attention sets `CapturedResidentStep.reads_tables` or ``host_planned``,
        for whether its kernel reads its pages off the tables or a host plan."""
        return

    def resident_commit(self, step: CapturedResidentStep) -> None:
        """Record, inside the capture and after the forward, the step's table updates."""
        return

    def resident_advance(self, plan: ResidentPlan) -> None:
        """Advance the host's copy of the table by the rule ``resident_commit`` recorded."""
        return
