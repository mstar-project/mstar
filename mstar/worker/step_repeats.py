"""Which chain steps stand for several iterations of their Loop.

A step of a steady batch needs nothing from the host between iterations: its
capture advances every row itself, so the host plans once, the device replays,
and the stops are read when the step lands. A row that stopped inside the step
costs the iterations the device ran past it, and the next step's, already
launched: up to 2N-1, against the one a speculated step costs.

The serving config's ``step_repeats:`` block asks for this per node; a node it
doesn't name runs every step once. Which walks may repeat is settled once, at
startup, after capture; the log has one line per Loop walk of each node, saying
for each capability ``on``, ``off: not asked for``, or ``off: <owner>: <reason>``.
What a step can have is decided twice: here against its rows and its Loop, then
by its resources (``Resource.max_repeats``), which can only narrow it.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

from mstar.engine.resources.resident import NOT_ASKED
from mstar.graph.base import EdgeKind, LoopEdges, loop_edges
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION

if TYPE_CHECKING:
    from mstar.worker.speculation import Speculation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepeatConfig:
    """One node's ``step_repeats:`` settings. Every setting but ``max`` shapes
    a step that repeats, so setting one on a node whose ``max`` is 1 is an
    error."""
    # the most iterations one step may stand for
    max: int = 1

    @classmethod
    def parse(cls, node: str, block: object) -> RepeatConfig:
        if not isinstance(block, Mapping):
            raise ValueError(f"step_repeats for {node!r} must be a mapping, got {block!r}")
        known = {f.name for f in fields(cls)}
        if unknown := sorted(set(block) - known):
            raise ValueError(f"step_repeats for {node!r} sets {unknown}, which are not settings: {sorted(known)}")
        config = cls(**block)
        if type(config.max) is not int or config.max < 1:
            raise ValueError(f"step_repeats for {node!r}: max is {config.max!r}; it must be an integer of at least 1")
        default = cls()
        needs = [k for k in block if k != "max" and getattr(config, k) != getattr(default, k)]
        if config.max == 1 and needs:
            raise ValueError(f"step_repeats for {node!r} sets {needs}, which need max above 1")
        return config


def parse_step_repeats(block: object, nodes: Collection[str]) -> dict[str, RepeatConfig]:
    """The serving config's ``step_repeats:`` block, node -> its settings."""
    if block is None:
        return {}
    if not isinstance(block, Mapping):
        raise ValueError(f"step_repeats must map node names to settings, got {block!r}")
    if unknown := sorted(set(block) - set(nodes)):
        raise ValueError(f"step_repeats names {unknown}, which are not nodes of the model")
    return {node: RepeatConfig.parse(node, settings) for node, settings in block.items()}


def edge_refusal(edges: LoopEdges) -> str | None:
    """Why a run can't serve one of ``edges``, or None. A loop-back signal is
    carried between replays (``NodeSubmodule.repeat_carry``), and a
    per-iteration output is served by a column per replay (``run_columns``),
    as items of its own."""
    for name, kind in sorted(edges.inputs.items()):
        if kind is EdgeKind.PER_ITERATION:
            return f"stream buffer: its input {name!r} is streamed, an item each iteration"
    for (signal, dest), kind in sorted(edges.outputs.items()):
        if kind is EdgeKind.PER_ITERATION and dest != EMPTY_DESTINATION and (signal, dest) not in edges.items:
            return f"graph runtime: {signal!r} -> {dest!r} is read once an iteration by its node"
    return None


def loop_back(edges: LoopEdges) -> frozenset[str]:
    """The signals ``edges`` loop back."""
    return frozenset(signal for (signal, _), kind in edges.outputs.items() if kind is EdgeKind.LOOP_BACK)


def run_columns(edges: LoopEdges, inline: Collection[str]) -> frozenset[str]:
    """The per-iteration output signals a step that repeats keeps a column of
    per replay: all but a client copy that rides the step's token column
    (``inline``: the node's inline client signals)."""
    return frozenset(
        signal for (signal, dest), kind in edges.outputs.items()
        if kind is EdgeKind.PER_ITERATION and dest != EMPTY_DESTINATION
        and not (dest == EMIT_TO_CLIENT and signal in inline)
    )


class StepRepeats:
    """``configs``: the serving config's settings per node; ``edges``: this
    worker's (node, walk) pairs whose walk runs the node in a Loop, with the
    node's edges there."""

    def __init__(self, configs: Mapping[str, RepeatConfig], edges: Mapping[tuple[str, str], LoopEdges]):
        self._configs = configs
        self.edges = edges
        self._walks: set[tuple[str, str]] = set()

    def limit(self, node: str) -> int:
        config = self._configs.get(node)
        return 1 if config is None else config.max

    @property
    def asked(self) -> bool:
        """Whether any Loop walk here may stand for more than one iteration."""
        return any(self.limit(node) > 1 for node, _ in self.edges)

    def carried(self) -> dict[tuple[str, str], frozenset[str]]:
        """The loop-back signals of each walk asked to repeat, which a run
        carries between replays."""
        return {key: loop_back(edges) for key, edges in self.edges.items() if self.limit(key[0]) > 1}

    def settle(
        self, resident_refusal: Callable[[str, str], str | None], repeat_refusal: Callable[[str, str], str | None],
    ) -> None:
        """Which walks may repeat, and the line for each."""
        for key in sorted(self.edges):
            limit = self.limit(key[0])
            resident = resident_refusal(*key)
            if limit == 1:
                repeats = NOT_ASKED
            elif resident is not None:
                repeats = "engine: it plans on the host"
            else:
                repeats = repeat_refusal(*key)
            if repeats is None:
                self._walks.add(key)
            logger.info(
                "%s/%s: resident %s | repeats %s", *key,
                "on" if resident is None else f"off: {resident}",
                f"on, up to {limit}" if repeats is None else f"off: {repeats}",
            )

    def ceiling(self, speculation: Speculation, in_chain: int, loop_room) -> int:
        """The most ``speculation``'s step can repeat. Only a chain's step
        whose rows all continue: a fresh row needs a plan of its own, and until
        the chain has run a step, other ready work hasn't had its look in."""
        batch = speculation.scheduled_batch
        if (
            in_chain < 1 or not speculation.is_new_iter or not speculation.is_same_node
            or len(batch.request_to_worker_graph) != len(speculation.continuing_rids) or batch.chunk_ranges
        ):
            return 1
        if (batch.node_name, batch.graph_walk) not in self._walks:
            return 1
        return max(1, min(self.limit(batch.node_name), loop_room()))


def walk_edges(walks: Mapping[str, object], pairs: Iterable[tuple[str, str]]) -> dict[tuple[str, str], LoopEdges]:
    """``loop_edges`` for each (node, walk) of ``pairs`` that runs in a Loop."""
    out = {}
    for node, walk in pairs:
        section = walks.get(walk)
        edges = None if section is None else loop_edges(section, node)
        if edges is not None:
            out[(node, walk)] = edges
    return out
