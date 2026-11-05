"""Which chain steps stand for several iterations of their Loop.

A step of a steady batch needs nothing from the host between iterations: its
capture advances every row itself, so the host plans once, the device replays,
and the stops are read when the step lands. A row that stopped inside the step
costs the iterations the device ran past it, and the next step's, already
launched: up to 2N-1, against the one a speculated step costs.

The serving config's ``max_step_repeats`` asks for this; 1 is off. What a walk
needs is decided once per (node, walk), here. What a step can have is decided
twice: here against its rows and its Loop, then by its resources
(``Resource.max_repeats``), which can only narrow it.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mstar.worker.speculation import Speculation

logger = logging.getLogger(__name__)


class StepRepeats:
    def __init__(self, limit: int = 1):
        if type(limit) is not int or limit < 1:
            raise ValueError(f"max_step_repeats is {limit!r}; it must be an integer of at least 1")
        self.limit = limit
        self._walks: dict[tuple[str, str], bool] = {}

    def ceiling(self, speculation: Speculation, in_chain: int, repeat_refusal, loop_room) -> int:
        """The most ``speculation``'s step can repeat. Only a chain's step
        whose rows all continue: a fresh row needs a plan of its own, and until
        the chain has run a step, other ready work hasn't had its look in."""
        batch = speculation.scheduled_batch
        if (
            self.limit == 1 or in_chain < 1 or not speculation.is_new_iter or not speculation.is_same_node
            or len(batch.request_to_worker_graph) != len(speculation.continuing_rids) or batch.chunk_ranges
        ):
            return 1
        key = (batch.node_name, batch.graph_walk)
        allowed = self._walks.get(key)
        if allowed is None:
            refusal = repeat_refusal(*key)
            if refusal is not None:
                logger.info("max_step_repeats=%d: steps of %s/%s run once each: %s", self.limit, *key, refusal)
            allowed = self._walks[key] = refusal is None
        return max(1, min(self.limit, loop_room())) if allowed else 1
