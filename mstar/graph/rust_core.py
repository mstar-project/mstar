"""Compile a ``GraphSection`` into the flat spec the Rust graph runtime takes.

The translation seam. Everything structural that Python derives at
construction time — input/output signatures, ``Loop._loop_back_inputs``,
``Loop._external_inputs`` — is handed over verbatim rather than re-derived in
Rust, so the two sides cannot disagree about the shape of the graph.

The spec is per worker graph and immutable; the runtime shares one compiled
copy across every request, which is what removes the per-request
``deepcopy(section)`` in ``WorkerGraphQueues.add_request``.
"""
import logging

from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential

logger = logging.getLogger(__name__)

# Rust caps a node's readiness mask at one u64.
MAX_INPUTS = 64


class UnsupportedGraphError(Exception):
    """The section uses something the Rust core does not model yet.

    Raised at compile time so a caller can fall back to Python for this worker
    graph instead of diverging silently at runtime.
    """


def _edge_spec(edge: GraphEdge) -> dict:
    return {
        "name": edge.name,
        "dest": edge.next_node,
        "persist": edge.persist,
        "new_token": edge.conductor_new_token,
        "streaming": edge.is_streaming,
        "modality": edge.output_modality,
    }


def _direct_entities(section: GraphSection) -> list[GraphSection]:
    """Entities a registry manages directly.

    Mirrors ``GraphStateRegistry._set_managed_entities``: recurse through
    Sequential/Parallel, stop at a GraphNode or a Loop (a Loop's contents
    belong to its own registry).
    """
    if isinstance(section, (GraphNode, Loop)):
        return [section]
    if not isinstance(section, (Sequential, Parallel)):
        raise UnsupportedGraphError(f"unknown GraphSection type {type(section).__name__}")
    return [e for s in section.sections for e in _direct_entities(s)]


def compile_section(
    section: GraphSection,
    leader_nodes: set[str] | None = None,
) -> dict:
    """Returns the ``GraphRuntime`` kwargs for ``section``.

    ``leader_nodes`` marks which nodes the scheduler may initiate work for
    (rank 0 of a TP group); defaults to every node, matching a non-TP worker.
    """
    nodes = section.get_nodes()
    loops = section.get_loops()

    for name, node in nodes.items():
        if len(node.input_names) > MAX_INPUTS:
            raise UnsupportedGraphError(
                f"node {name!r} has {len(node.input_names)} inputs; "
                f"the Rust readiness mask holds {MAX_INPUTS}"
            )

    node_specs = [
        {
            # Sorted so a node's input slot indices are stable across
            # processes; Python's input_names is a set.
            "name": name,
            "inputs": sorted(node.input_names),
            "streaming_inputs": sorted(node._streaming_inputs),
            "outputs": [_edge_spec(e) for e in node.outputs],
        }
        for name, node in nodes.items()
    ]

    loop_specs: list[dict] = []
    seen: set[str] = set()

    def walk(sec: GraphSection, parent: str | None) -> None:
        if isinstance(sec, GraphNode):
            return
        if isinstance(sec, Loop):
            if sec.name in seen:
                raise UnsupportedGraphError(f"duplicate loop name {sec.name!r}")
            seen.add(sec.name)
            members = _direct_entities(sec.section)
            loop_specs.append({
                "name": sec.name,
                "max_iters": sec.max_iters,
                "parent": parent,
                "member_nodes": [e.name for e in members if isinstance(e, GraphNode)],
                "outputs": [_edge_spec(e) for e in sec.outputs],
                "accumulated": [_edge_spec(e) for e in sec.accumulated_outputs],
                # Verbatim from Python's construction-time derivation.
                "loop_back": sorted(sec._loop_back_inputs),
                "external_inputs": sorted(sec._external_inputs),
            })
            walk(sec.section, sec.name)
            return
        if not isinstance(sec, (Sequential, Parallel)):
            raise UnsupportedGraphError(f"unknown GraphSection type {type(sec).__name__}")
        for s in sec.sections:
            walk(s, parent)

    walk(section, None)
    if len(loop_specs) != len(loops):
        raise UnsupportedGraphError(
            f"walked {len(loop_specs)} loops but get_loops() reports {len(loops)}; "
            "a Loop is reachable by a path the walk does not follow"
        )

    return {
        "nodes": node_specs,
        "loops": loop_specs,
        "leader_nodes": sorted(leader_nodes) if leader_nodes is not None else sorted(nodes),
    }


def try_compile_section(
    section: GraphSection,
    leader_nodes: set[str] | None = None,
    wg_id: str = "?",
) -> dict | None:
    """``compile_section`` that returns None (with a logged reason) instead of
    raising — for the adoption path, where an unsupported worker graph should
    fall back to Python rather than fail the worker."""
    try:
        return compile_section(section, leader_nodes)
    except UnsupportedGraphError as exc:
        logger.warning(
            "Rust graph runtime disabled for worker graph %s: %s", wg_id, exc
        )
        return None
