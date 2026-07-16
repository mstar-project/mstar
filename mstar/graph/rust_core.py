"""RFC #130 Steps 4-5 seam: translate ``GraphSection`` trees into the Rust
walk core's spec.

``walks_to_json`` produces the JSON that ``mstar_rust.WalkSet.from_json``
compiles; a ``WalkSet.state(walk)`` is then the Rust counterpart of a
per-request ``WorkerGraphIO`` — same readiness, completion-routing, loop
iteration, and termination semantics (asserted by
``test/rust/test_walk_parity.py``, which drives both implementations with
identical event sequences).

Scope: the walk state machine only. Streaming buffer semantics and the
worker/conductor wiring stay in Python for now — this module is the
translation seam those later steps build on.
"""

from __future__ import annotations

import json

from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Parallel,
    Sequential,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION

# mstar's sentinel destinations -> the Rust core's.
_DEST = {EMIT_TO_CLIENT: "EMIT_TO_CLIENT", EMPTY_DESTINATION: "EMPTY_DESTINATION"}


def edge_to_spec(edge: GraphEdge) -> dict:
    return {
        "next_node": _DEST.get(edge.next_node, edge.next_node),
        "name": edge.name,
        "persist": bool(edge.persist),
        "output_modality": edge.output_modality or None,
    }


def section_to_spec(section: GraphSection) -> dict:
    if isinstance(section, GraphNode):
        return {
            "kind": "node",
            "name": section.name,
            "input_names": sorted(section.input_names),
            "outputs": [edge_to_spec(e) for e in section.outputs],
        }
    if isinstance(section, Loop):
        return {
            "kind": "loop",
            "name": section.name,
            "body": section_to_spec(section.section),
            "max_iters": int(section.max_iters),
            "outputs": [edge_to_spec(e) for e in section.outputs],
            "accumulated_outputs": [
                edge_to_spec(e) for e in section.accumulated_outputs
            ],
        }
    if isinstance(section, (Sequential, Parallel)):
        kind = "sequential" if isinstance(section, Sequential) else "parallel"
        return {
            "kind": kind,
            "sections": [section_to_spec(s) for s in section.sections],
        }
    raise TypeError(f"unknown GraphSection type: {type(section).__name__}")


def walks_to_json(walks: dict[str, GraphSection]) -> str:
    """``{walk_name: GraphSection}`` -> the WalkSet JSON spec."""
    return json.dumps({name: section_to_spec(s) for name, s in walks.items()})
