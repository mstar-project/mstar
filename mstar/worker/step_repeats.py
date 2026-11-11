"""Which chain steps of a decode Loop stand for several iterations.

A step of a steady batch needs nothing from the host between iterations: its
capture advances every row itself. So the host plans once, launches the capture
k times, and reads the stops when the step lands. A row that stopped inside the
step costs the iterations the device ran past it and the next step's.

The serving config's ``step_repeats:`` block asks for this per node; a node it
doesn't name runs every step once. Which walks may repeat is settled once, after
capture (`StepRepeats.settle`). How far a step goes is decided twice: here against
its rows and its Loop, then by its resources (``Resource.max_repeats``), which can
only narrow it.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, NamedTuple

from mstar.engine.device_run import WalkAsk
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION

if TYPE_CHECKING:
    from mstar.graph.base import GraphSection
    from mstar.worker.speculation import Speculation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepeatConfig:
    """One node's ``step_repeats:`` settings."""
    # the most iterations one step may stand for
    max: int = 1
    # Ready rows wait for a full-length run: after a step that ran once, the next
    # takes no new rows. Trades admission latency for longer runs.
    ready_waits: bool = False

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
        for name in ("ready_waits",):
            if type(getattr(config, name)) is not bool:
                raise ValueError(
                    f"step_repeats for {node!r}: {name} is {getattr(config, name)!r}; it must be a boolean"
                )
            if config.max == 1 and getattr(config, name):
                raise ValueError(f"step_repeats for {node!r} sets {name}, which needs max above 1")
        return config


def parse_step_repeats(block: object, nodes: Collection[str]) -> dict[str, RepeatConfig]:
    """The serving config's ``step_repeats:`` block: node -> its settings."""
    if block is None:
        return {}
    if not isinstance(block, Mapping):
        raise ValueError(f"step_repeats must map node names to settings, got {block!r}")
    if unknown := sorted(set(block) - set(nodes)):
        raise ValueError(f"step_repeats names {unknown}, which are not nodes of this model")
    return {node: RepeatConfig.parse(node, settings) for node, settings in block.items()}


class LoopSignals(NamedTuple):
    """A node's edges in its one-node Loop, as a step that repeats sees them."""
    # the signals one iteration hands the next on the device
    loop_back: frozenset[str]
    # why a step can't stand for several iterations; None if it can
    refusal: str | None


def loop_signals(section: GraphSection, node: str, streamed: Collection[str]) -> LoopSignals | None:
    """``node``'s edges in ``section``'s Loop that runs it, or None if none does.

    Every output has to be a loop-back or a client token, the only per-iteration
    outputs the step's token column serves; every other input has to be the same
    each iteration. ``streamed``: the edge names arriving on streaming edges,
    which only the partition topology knows (a walk's graph is built without them).
    """
    loops = [loop for loop in section.get_loops().values() if node in loop.get_nodes()]
    inner = next((loop for loop in loops if not set(loop.section.get_loops()) & {lp.name for lp in loops}), None)
    if inner is None:
        return None
    nodes = inner.get_nodes()
    if len(nodes) != 1:
        return LoopSignals(frozenset(), f"graph: its Loop {inner.name!r} runs {sorted(nodes)}, not this node alone")
    this = nodes[node]
    back = {edge.name for edge in this.outputs if edge.next_node == node}
    if arriving := sorted(set(streamed) & set(this.input_names)):
        return LoopSignals(frozenset(), f"graph: its inputs {arriving} are streamed in, a value an iteration")
    leaving = {edge.name for edge in (*inner.outputs, *inner.accumulated_outputs)}
    for edge in this.outputs:
        if edge.name in leaving and edge.next_node != EMPTY_DESTINATION:
            return LoopSignals(frozenset(), f"graph: output {edge.name!r} leaves the Loop")
        if edge.next_node not in (node, EMIT_TO_CLIENT, EMPTY_DESTINATION):
            return LoopSignals(frozenset(), f"graph: {edge.name!r} -> {edge.next_node!r} is read once an iteration")
    return LoopSignals(frozenset(back), None)


class StepRepeats:
    """``configs``: the serving config's settings per node. ``loops``: this
    worker's (node, walk) pairs whose walk runs a node of ``configs`` in a Loop."""

    def __init__(self, configs: Mapping[str, RepeatConfig], loops: Mapping[tuple[str, str], LoopSignals]):
        self._configs = configs
        self.loops = loops
        self._walks: set[tuple[str, str]] = set()

    def asked_walks(self) -> dict[str, dict[str, WalkAsk]]:
        """Node -> the walks a step of which may repeat if capture allows it."""
        asked: dict[str, dict[str, WalkAsk]] = {}
        for (node, walk), signals in self.loops.items():
            if signals.refusal is None:
                config = self._configs[node]
                asked.setdefault(node, {})[walk] = WalkAsk(signals.loop_back)
        return asked

    def settle(self, refusal: Callable[[str, str], str | None]) -> None:
        """Which walks repeat, with one log line each. A node the config names
        with no walk that can raises, naming why."""
        whys: dict[str, list[str]] = {node: [] for node in self._configs}
        for key, signals in sorted(self.loops.items()):
            why = signals.refusal or refusal(*key)
            if why is None:
                self._walks.add(key)
                logger.info("%s/%s: repeats on, up to %d", *key, self._configs[key[0]].max)
                whys.pop(key[0], None)
            else:
                logger.info("%s/%s: repeats off: %s", *key, why)
                if key[0] in whys:
                    whys[key[0]].append(f"{key[1]}: {why}")
        for node, reasons in whys.items():
            raise ValueError(
                f"step_repeats asks {node!r} to repeat, but no walk of it can: {reasons or 'no Loop runs it'}"
            )

    def loop_back(self, node: str, walk: str) -> frozenset[str]:
        return self.loops[(node, walk)].loop_back

    def may_repeat(self, node: str, walk: str) -> bool:
        return (node, walk) in self._walks

    def owes_run(self, node: str, walk: str, repeats: int, owed: bool) -> bool:
        """Whether the next step is a run that ready rows wait for: once after a
        step of a walk that may repeat but ran one iteration, so one narrowed to
        1 can't keep them out. A prompt or chunk step of the node owes none."""
        config = self._configs.get(node)
        return config is not None and config.ready_waits and repeats == 1 and not owed and self.may_repeat(node, walk)

    def ceiling(self, speculation: Speculation, in_chain: int, loop_room: Callable[[], int]) -> int:
        """The most ``speculation``'s step can repeat. Only a chain's step whose
        rows all continue: a fresh row needs a plan of its own, and until the
        chain has run a step other ready work hasn't had its look in."""
        batch = speculation.scheduled_batch
        key = (batch.node_name, batch.graph_walk)
        if (
            key not in self._walks or in_chain < 1 or not speculation.is_new_iter or not speculation.is_same_node
            or len(batch.request_to_worker_graph) != len(speculation.continuing_rids) or batch.chunk_ranges
        ):
            return 1
        return max(1, min(self._configs[key[0]].max, loop_room()))
