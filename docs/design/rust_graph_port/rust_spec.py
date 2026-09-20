"""GraphSection -> Rust spec dicts. Stand-in for mstar/graph/rust_core.py.

Reads the real Python objects so both sides describe the same graph by
construction — including `Loop._loop_back_inputs` / `_external_inputs`, which
Python already derives from `get_inputs_outputs()` and we hand over verbatim
rather than re-deriving in Rust.
"""
from mstar.graph.base import GraphNode, Loop, Parallel, Sequential


def _edge(e):
    return {
        "name": e.name, "dest": e.next_node, "persist": e.persist,
        "new_token": e.conductor_new_token, "streaming": e.is_streaming,
        "modality": e.output_modality,
    }


def _direct_entities(sec):
    """Mirrors GraphStateRegistry._set_managed_entities: stop at a Loop."""
    if isinstance(sec, (GraphNode, Loop)):
        return [sec]
    assert isinstance(sec, (Sequential, Parallel))
    return [e for s in sec.sections for e in _direct_entities(s)]


def to_rust_spec(section, leader_nodes=None):
    """Returns (nodes, loops) kwargs for GraphRuntime."""
    all_nodes = section.get_nodes()
    nodes = [
        {
            "name": n.name,
            "inputs": sorted(n.input_names),
            "streaming_inputs": sorted(n._streaming_inputs),
            "outputs": [_edge(e) for e in n.outputs],
        }
        for n in all_nodes.values()
    ]

    loops = []

    def walk(sec, parent):
        if isinstance(sec, GraphNode):
            return
        if isinstance(sec, Loop):
            members = _direct_entities(sec.section)
            loops.append({
                "name": sec.name,
                "max_iters": sec.max_iters,
                "parent": parent,
                "member_nodes": [e.name for e in members if isinstance(e, GraphNode)],
                "outputs": [_edge(e) for e in sec.outputs],
                "accumulated": [_edge(e) for e in sec.accumulated_outputs],
                "loop_back": sorted(sec._loop_back_inputs),
                "external_inputs": sorted(sec._external_inputs),
            })
            walk(sec.section, sec.name)
            return
        for s in sec.sections:
            walk(s, parent)

    walk(section, None)
    return {
        "nodes": nodes,
        "loops": loops,
        "leader_nodes": list(leader_nodes) if leader_nodes is not None else list(all_nodes),
    }
