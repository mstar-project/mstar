"""Tuned hipBLASLt solutions for a flow transformer's large GEMMs on AMD GPUs.

hipBLASLt picks a solution per shape from its own table. At a DiT's GEMMs (thousands of rows over a weight that
misses every cache) that pick can sit well off the best of its solutions. A product's table
(mstar/kernels/configs/<arch>/<product>/dit_linear.json, from benchmark/kernels/dit_linear.py) names, per
"rows{r}-n{n}-k{k}", the solution to run instead; shapes it does not name stay on inductor's GEMM. The indices are
hipBLASLt's, so a table tuned on another stack is ignored. A transformer opts in with
``compile_transformer_forward(tuned_linears=True)``.

This is a sibling of the decode-linear table in ``rocm_gemm`` and not more entries in it: that table and that module
are hashed into the decode buckets' compile keys, so DiT shapes there would recompile every text model.
"""

from __future__ import annotations

import functools
import importlib.util
import logging

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

from mstar.kernels import KernelOp, ShapePolicy, Tolerance, configs
from mstar.platforms.rocm_gemm import (
    _decode_key,
    _gemm_inputs,
    _gemm_shape,
    _skinny_gemm_fake,
    _skinny_gemm_reference,
    _stack,
    _val,
    _weight_of,
)

logger = logging.getLogger(__name__)


@functools.cache
def dit_linear_table(device_index: int = 0) -> dict[str, dict]:
    """This device's tuned entries by shape key; empty without a table for this stack."""
    from mstar.platforms import current_platform

    facts = current_platform.device_facts(device_index)
    config = configs.load_config("dit_linear", facts.arch, facts.product)
    if config is None:
        return {}
    stack = (config.get("provenance") or {}).get("stack")
    if stack != _stack():
        logger.warning("dit_linear table for %s/%s was tuned on %s, not %s: not used",
                       facts.arch, facts.product, stack, _stack())
        return {}
    return {k: v for k, v in config["classes"].items() if k != "default" and v.get("impl") == "hipblaslt"}


def _hipblaslt_dit_linear():
    import aiter

    aiter.hipb_create_extension()

    def dit_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        entry = dit_linear_table(x.device.index or 0).get(_decode_key(x.shape[0], weight.shape[0], x.shape[1]))
        if entry is None:
            return torch.nn.functional.linear(x, weight, bias)
        return aiter.hipb_mm(x, weight.t(), int(entry["solution"]), bias=bias)

    return dit_linear


DIT_LINEAR = KernelOp(
    "dit_linear",
    "(Tensor x, Tensor weight, Tensor? bias) -> Tensor",
    fake=_skinny_gemm_fake,
    reference=_skinny_gemm_reference,
    impls={f"rocm:{arch}": _hipblaslt_dit_linear for arch in ("gfx942", "gfx950")},
    shapes=ShapePolicy(
        dims=("rows", "n", "k"),
        key=_gemm_shape,
        inputs=_gemm_inputs,
        warm=({"rows": 32, "n": 3840, "k": 3840},),
        dtypes=(torch.bfloat16,),
    ),
    tolerance=Tolerance.accumulated(),
)


class DitLinearPass(CustomGraphPass):
    """The lowered graph: ``mm(x, w.T)`` whose shape the device's table names. A table change changes the hash, so
    inductor recompiles what it touches."""

    def __call__(self, graph: torch.fx.Graph) -> None:
        table = dit_linear_table(torch.cuda.current_device())
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target is not torch.ops.aten.mm.default:
                continue
            x, wt = node.args
            weight = _weight_of(wt)
            xv, wv = _val(x), _val(weight)
            if not isinstance(xv, torch.Tensor) or not isinstance(wv, torch.Tensor) or xv.dim() != 2:
                continue
            if xv.dtype != torch.bfloat16 or wv.dtype != xv.dtype or not wv.is_contiguous():
                continue
            if not all(isinstance(d, int) for d in (*xv.shape, *wv.shape)):
                continue
            if _decode_key(xv.shape[0], wv.shape[0], xv.shape[1]) not in table:
                continue
            with graph.inserting_before(node):
                new = graph.call_function(torch.ops.mstar.dit_linear.default, (x, weight, None))
            new.meta.update(node.meta)
            node.replace_all_uses_with(new)
            graph.erase_node(node)
            if not wt.users:
                graph.erase_node(wt)

    def uuid(self) -> bytes:
        return get_hash_for_files((__file__,)) + repr(sorted(dit_linear_table(0).items())).encode()


@functools.cache
def dit_linear_pass() -> DitLinearPass | None:
    """The pass, or None where nothing would change (no AITER, another arch, no table for this device)."""
    if importlib.util.find_spec("aiter") is None or not torch.cuda.is_available():
        return None
    from mstar.platforms import current_platform

    if not current_platform.is_rocm() or current_platform.arch() not in ("gfx942", "gfx950"):
        return None
    return DitLinearPass() if dit_linear_table(torch.cuda.current_device()) else None
