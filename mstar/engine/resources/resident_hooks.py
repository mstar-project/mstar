"""What a resource overrides to take part in resident planning (`resident.py`)."""
from __future__ import annotations

from collections.abc import Collection, Mapping
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


# `max_repeats` for a resource that has no limit of its own
ANY_REPEATS = 1 << 30


class ResidentHooks:
    """A captured decode step reads per-request state off tables on the device;
    each owner of that state keeps its own table. A resource overrides these
    only once it has one."""

    def ask_repeats(self, walk: str) -> None:
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
        stages per step would go stale."""
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

    def resident_refusal(self, key: str, streams: Mapping[str, ResidentStream]) -> str | None:
        """Why a walk of a node holding this resource can't run on resident
        tables, or None if it can, asked at load. ``streams``: each resource's
        stream on the walk's steps, by resource key; ``key`` is this one's."""
        del key, streams
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

    def resident_run_gather(self, step: CapturedResidentStep) -> None:
        """As ``resident_gather``, for a capture that is a WHILE run's body: no
        host stands between its iterations, or tells it how far the last run got."""
        return

    def resident_commit(self, step: CapturedResidentStep) -> None:
        """Record, inside the capture and after the forward, the step's table updates."""
        return

    def resident_advance(self, plan: ResidentPlan) -> None:
        """Advance the host's copy of the table by the rule ``resident_commit`` recorded."""
        return
