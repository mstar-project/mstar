"""``validate_deployment``'s checks for nodes that draft and verify tokens."""
from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from typing import Any

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.graph.base import GraphNode, GraphSection
from mstar.utils import knobs


def _graph_nodes(section: GraphSection) -> Iterator[GraphNode]:
    if isinstance(section, GraphNode):
        yield section
        return
    for child in getattr(section, "sections", None) or ():
        yield from _graph_nodes(child)
    child = getattr(section, "section", None)
    if child is not None:
        yield from _graph_nodes(child)


def async_scheduling_of(model, config: Mapping[str, Any]) -> dict[str, tuple[bool, str]]:
    """Each verifying node whose steps the worker won't plan ahead, and why:
    a graph node that turns async scheduling off, or a parallel node outside
    TP async scheduling. Read by the conductor and by every worker alike."""
    from mstar.worker.lockstep import tp_async_sched_setting

    names = {decl.node for decl in model.verify_nodes()}
    out: dict[str, tuple[bool, str]] = {}
    if not names:
        return out
    for walk, graph in model.get_graph_walk_graphs().items():
        for node in _graph_nodes(graph):
            if node.name in names and not node.enable_async_scheduling:
                out[node.name] = (False, f"walk {walk!r} turns async scheduling off for the node")
    tp_on, tp_nodes = tp_async_sched_setting(os.environ.get("MSTAR_TP_ASYNC_SCHED"), config.get("tp_async_sched"))
    for group in config.get("node_groups") or []:
        # lockstep only within a TP x SP block; data-parallel replicas each schedule alone
        if int(group.get("tp_size") or 1) * int(group.get("sp_size") or 1) > 1:
            for node in group.get("node_names") or []:
                if node in names and not (tp_on and (tp_nodes is None or node in tp_nodes)):
                    out.setdefault(node, (False, "TP async scheduling is off (MSTAR_TP_ASYNC_SCHED)"))
    return out


def _methods_accepting(block):
    """The engine's drafters, and a method the block names, so its other keys are still checked."""
    from mstar.engine.resources.draft.config import DRAFT_METHODS, DraftMethod

    method = block.get("method") if isinstance(block, Mapping) else None
    if isinstance(method, str) and method not in DRAFT_METHODS:
        return {**DRAFT_METHODS, method: DraftMethod(method, needs_model="model" in block)}
    return DRAFT_METHODS


def check_draft(model, config: Mapping[str, Any], specs: list[NodeResourceSpec] | None) -> list[str]:
    """Every problem with the deployment's verifying nodes, from config alone."""
    from mstar.engine.resources.draft.config import DraftConfig
    from mstar.engine.resources.draft.startup import check_deployment, node_chunks_prompts

    block = (config.get("model_kwargs") or {}).get("draft")
    decls = list(model.verify_nodes())
    if block is not None and not decls:
        # the engine owns the schema: a typo or an unsupported block is never silently ignored
        DraftConfig.parse(block, _methods_accepting(block))
        return [f"{type(model).__name__} declares no node that verifies drafts; drop model_kwargs.draft"]
    if not decls:
        return []
    # the model drafts as the deployment says, and only when it says so: a model that
    # builds its own DraftConfig could drift from the block every other check reads
    if block is None:
        return [f"{type(model).__name__} declares verifying nodes {sorted(d.node for d in decls)} "
                "without a model_kwargs.draft block, which alone turns speculative decoding on"]
    parsed = DraftConfig.parse(block, _methods_accepting(block))
    drifted = [decl.node for decl in decls if decl.config != parsed]
    if drifted:
        return [f"nodes {drifted} verify with a draft config other than model_kwargs.draft's ({parsed})"]
    if specs is None:
        # declaring the node resources failed, which validation reports; every check below reads them
        return []
    names = {decl.node for decl in decls}
    streaming_out = {
        node.name: True
        for graph in model.get_graph_walk_graphs().values()
        for node in _graph_nodes(graph)
        if node.name in names and any(edge.is_streaming for edge in node.outputs)
    }
    side_lane = [
        node for group in config.get("node_groups") or [] if group.get("side_stream")
        for node in group.get("node_names") or []
    ]
    from mstar.engine.resources.draft.startup import model_combined_walks

    problems = check_deployment(
        decls, {spec.resource_key: spec for spec in specs or ()}, streaming_out, side_lane,
        async_scheduling_of(model, config), knobs.resident_planning() != "off",
        chunked=[decl.node for decl in decls if node_chunks_prompts(model, decl.node)],
        combined_walk_of=model_combined_walks(model),
        max_seq_len=max_seq_len if isinstance(max_seq_len := config.get("max_seq_len"), int) else None,
    )
    return problems
