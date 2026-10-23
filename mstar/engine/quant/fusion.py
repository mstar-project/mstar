"""Step 5b: fuse each activation quantize into the op that produces it, in compiled graphs.

A post-grad inductor pass over ``mstar::fp8_quant_per_token``:

- after ``mstar::fused_add_rms_norm``, the norm becomes ``mstar::fused_add_rms_norm_fp8_quant``,
  which mutates as it did and also returns the quantized norm;
- after an ATen op (SiLU-and-mul, an ATen norm), the quantize becomes ATen ops that
  inductor fuses into the producer's kernel;
- after anything else (a custom op, a graph input) it stays.

Eager forwards never see the pass, so they stay unfused; the two differ by rounding order.
``MSTAR_QUANT_FUSE=0`` turns it off.
"""
from __future__ import annotations

import hashlib
import logging
import operator
import os
from pathlib import Path

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx import Graph, Node

from mstar.engine.quant.ops import FP8, FP8_MAX, FUSED_NORM_MAX_K, MIN_SCALE

logger = logging.getLogger(__name__)

aten, prims = torch.ops.aten, torch.ops.prims
_QUANT = torch.ops.mstar.fp8_quant_per_token.default
_VIEWS = {aten.view.default, aten.reshape.default, aten._unsafe_view.default}


def _fake_mode(nodes):
    return next(n.meta["val"].fake_mode for n in nodes if isinstance(n.meta.get("val"), torch.Tensor))


def _call(graph: Graph, mode, target, *args) -> Node:
    """A ``call_function`` node with its fake ``val``, so inductor can lower it."""
    node = graph.call_function(target, args)
    vals = [a.meta["val"] if isinstance(a, Node) else a for a in args]
    with unset_fake_temporarily(), mode:
        node.meta["val"] = target(*vals)
    return node


def _getitem(graph: Graph, mode, node: Node, index: int) -> Node:
    item = graph.call_function(operator.getitem, (node, index))
    item.meta["val"] = node.meta["val"][index]
    return item


def _outputs(quant: Node) -> tuple[Node | None, Node | None]:
    """The quantize's (values, scales) getitem users."""
    items = {u.args[1]: u for u in quant.users if u.target is operator.getitem}
    return items.get(0), items.get(1)


def _through_views(node: Node) -> Node:
    while node.op == "call_function" and node.target in _VIEWS:
        node = node.args[0]
    return node


def _quantize_users(node: Node, found: list[Node]) -> list[Node]:
    """The quantizes that read ``node``, directly or through views."""
    for user in node.users:
        if user.target is _QUANT:
            found.append(user)
        elif user.op == "call_function" and user.target in _VIEWS:
            _quantize_users(user, found)
    return found


def _norm_of(node: Node):
    """``(auto_functionalized, kwargs)`` if ``node`` is the norm output of a
    ``mstar::fused_add_rms_norm`` on whole (unviewed) bases, else None."""
    if node.target is not operator.getitem:
        return None
    af = node.args[0]
    if af.target is not torch.ops.higher_order.auto_functionalized_v2:
        return None
    if af.args[0] is not torch.ops.mstar.fused_add_rms_norm.default:
        return None
    kw = af.kwargs
    if not (kw.get("_input_alias") and kw.get("_residual_alias")):
        return None
    # getitem 0 is the op's own (None) output, then the bases after the call
    if node.args[1] != kw["_input_base_index"] + 1:
        return None
    return af, kw


def _fuse_norm(graph: Graph, af: Node, kw: dict, quants: list[Node]) -> None:
    mode = _fake_mode(kw["_all_bases"])
    with graph.inserting_before(af):
        fused = graph.call_function(
            torch.ops.higher_order.auto_functionalized_v2,
            (torch.ops.mstar.fused_add_rms_norm_fp8_quant.default,), dict(kw),
        )
        vals = {k: ([b.meta["val"] for b in v] if k == "_all_bases" else v.meta["val"] if isinstance(v, Node) else v)
                for k, v in kw.items()}
        with unset_fake_temporarily(), mode:
            fused.meta["val"] = torch.ops.higher_order.auto_functionalized_v2(
                torch.ops.mstar.fused_add_rms_norm_fp8_quant.default, **vals,
            )
        values, scales = _getitem(graph, mode, fused, 0), _getitem(graph, mode, fused, 1)
    # the old op returned nothing, so its base i sits at getitem i + 1, now i + 2
    for user in list(af.users):
        with graph.inserting_after(scales):
            user.replace_all_uses_with(_getitem(graph, mode, fused, user.args[1] + 1))
    for quant in quants:
        shape = list(quant.args[0].meta["val"].shape)
        q_out, s_out = _outputs(quant)
        with graph.inserting_after(scales):
            if q_out is not None:
                q_out.replace_all_uses_with(_call(graph, mode, aten.view.default, values, shape))
            if s_out is not None:
                s_out.replace_all_uses_with(_call(graph, mode, aten.view.default, scales, [*shape[:-1], 1]))


def _decompose(graph: Graph, quant: Node) -> None:
    """``fp8_quant_per_token`` as ATen ops; the portable path, but the scale divides."""
    x = quant.args[0]
    mode = _fake_mode([x])
    with graph.inserting_before(quant):
        xf = _call(graph, mode, prims.convert_element_type.default, x, torch.float32)
        amax = _call(graph, mode, aten.amax.default, _call(graph, mode, aten.abs.default, xf), [-1], True)
        scale = _call(graph, mode, aten.clamp_min.default,
                      _call(graph, mode, aten.mul.Tensor, amax, 1.0 / FP8_MAX), MIN_SCALE)
        # post-grad, so clamp as its decomposed pair
        q = _call(graph, mode, aten.div.Tensor, xf, scale)
        q = _call(graph, mode, aten.clamp_max.default, _call(graph, mode, aten.clamp_min.default, q, -FP8_MAX), FP8_MAX)
        q = _call(graph, mode, prims.convert_element_type.default, q, FP8)
    q_out, s_out = _outputs(quant)
    if q_out is not None:
        q_out.replace_all_uses_with(q)
    if s_out is not None:
        s_out.replace_all_uses_with(scale)


def _describe(node: Node) -> str:
    if node.target is operator.getitem:
        parent = node.args[0]
        kw = {k: v for k, v in parent.kwargs.items() if k.startswith("_") and k != "_all_bases"}
        return f"getitem {node.args[1]} of {parent.target}({parent.args[:1]}, {kw})"
    return f"{node.op} {node.target}"


def _is_aten(node: Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace in ("aten", "prims")
    )


def fuse_quantizes(graph: Graph) -> dict[str, int]:
    """Rewrite ``graph`` in place; returns how many quantizes took each path."""
    counts = {"norm": 0, "aten": 0, "kept": 0}
    for quant in [n for n in graph.nodes if n.target is _QUANT]:
        if not quant.users:  # rewired by an earlier norm fusion
            continue
        src = _through_views(quant.args[0])
        match = _norm_of(src)
        if match is not None and src.meta["val"].dim() == 2 and src.meta["val"].shape[-1] <= FUSED_NORM_MAX_K:
            quants = _quantize_users(src, [])
            _fuse_norm(graph, *match, quants)
            counts["norm"] += len(quants)
        elif _is_aten(src):
            _decompose(graph, quant)
            counts["aten"] += 1
        else:
            counts["kept"] += 1
            logger.debug("quant fusion: kept a quantize of %s", _describe(src))
    graph.eliminate_dead_code()
    return counts


class QuantFusionPass(CustomGraphPass):
    def __call__(self, graph: Graph) -> None:
        counts = fuse_quantizes(graph)
        if any(counts.values()):
            logger.info("quant fusion: %s", counts)

    def uuid(self) -> bytes:
        # recompile, rather than reuse cached graphs, when this file changes
        return hashlib.sha256(Path(__file__).read_bytes()).digest()


def install() -> None:
    """Register the pass with inductor, once per process, unless MSTAR_QUANT_FUSE=0."""
    import torch._inductor.config as inductor_config

    if os.environ.get("MSTAR_QUANT_FUSE", "1") == "0":
        logger.info("quant fusion off (MSTAR_QUANT_FUSE=0)")
        return
    current = inductor_config.post_grad_custom_post_pass
    if isinstance(current, QuantFusionPass):
        return
    if current is not None:
        raise RuntimeError(f"inductor post_grad_custom_post_pass is already {current!r}; cannot add quant fusion")
    inductor_config.post_grad_custom_post_pass = QuantFusionPass()
