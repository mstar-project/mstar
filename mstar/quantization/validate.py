"""The boot check: a deployment's ``quantization:`` block against its model, before any worker starts.

Each placed node with a declared weights root is built on the meta device at its TP size (one rank's
geometry, no memory), its checkpoint's headers are read when the files are local, and the resolver runs as
the worker will; every problem joins ``validate_deployment``'s list.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.distributed.communication import CommGroup
from mstar.quantization.checkpoint import scan_checkpoint
from mstar.quantization.formats import AUTO, BF16
from mstar.quantization.resolve import resolve_plan
from mstar.quantization.spec import QuantizationSpecError, parse_quantization


def _placement(config: Mapping[str, Any]) -> dict[str, int]:
    """node -> its TP size (tp_size; SP replicates weights). Malformed groups are skipped here; the
    placement check reports them."""
    out: dict[str, int] = {}
    groups = config.get("node_groups")
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, Mapping) or not isinstance(group.get("node_names"), list):
            continue
        tp = group.get("tp_size", 1)
        tp = tp if isinstance(tp, int) and not isinstance(tp, bool) and tp > 0 else 1
        for node in group["node_names"]:
            if isinstance(node, str):
                out[node] = tp
    return out


def _local_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


def quantization_problems(model: Any, config: Mapping[str, Any]) -> list[str]:
    raw = config.get("quantization")
    try:
        spec = parse_quantization(raw)
    except QuantizationSpecError as e:
        return list(e.problems)
    placed = _placement(config)
    problems: list[str] = []
    unknown = sorted(set(spec.nodes) - set(placed))
    if unknown:
        problems.append(f"quantization.nodes names {unknown}, which no node group places")
    getter = getattr(model, "get_quantization_roots", None)
    roots = [r for r in (getter() if getter is not None else []) if r.node in placed]
    declared = {r.node for r in roots}
    for node in sorted(set(spec.nodes) & set(placed) - declared):
        if not spec.nodes[node].is_default:
            problems.append(f"quantization.nodes.{node}: {type(model).__name__} declares no quantizable layers there")
    top_asks = spec.top.format not in (AUTO, BF16) or spec.top.rules
    if spec.given and top_asks and not roots:
        problems.append(f"quantization: {type(model).__name__} declares no quantizable layers on any placed node")
    capability = _local_capability()
    for decl in roots:
        node_spec = spec.for_node(decl.node)
        directory = decl.checkpoint_dir()
        try:
            ckpt = scan_checkpoint(directory) if directory else None
        except (OSError, ValueError) as e:
            problems.append(f"node {decl.node!r}: reading the checkpoint's headers failed: {type(e).__name__}: {e}")
            continue
        if node_spec.is_default and (ckpt is None or not ckpt.quantized):
            continue
        if decl.build is None:
            continue
        tp = placed[decl.node]
        with torch.device("meta"):
            root = decl.build(CommGroup(my_global_rank=0, my_group_rank=0, group_members=list(range(tp))))
        problems += resolve_plan(root, decl, node_spec, ckpt, capability).problems
    return problems
