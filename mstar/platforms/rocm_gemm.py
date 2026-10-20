"""Skinny GEMMs for decode-sized captured buckets on AMD GPUs.

A linear with one or two rows is a weight stream. AITER's ``wvSpltK`` (the
kernel vLLM calls ``wvSplitK``) reads the weight once across every CU and beats
both hipBLASLt (10-30% on MI300X) and inductor's own one-row GEMV reduction;
from four rows on it loses, badly once K * rows outgrows its LDS tile. Inductor
can't pick it, so two graph passes swap such linears for ``mstar::skinny_gemm``:
one before inductor's decompositions (a one-row ``mm`` becomes a reduction
there), one after (``mm``/``addmm`` that only appear in the lowered graph).
``RocmPlatform.inductor_overrides`` installs them for buckets of up to
``SKINNY_MAX_ROWS`` tokens only.

AITER is imported on first call, never at module import.
"""

from __future__ import annotations

import functools
import importlib.util
import logging

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

logger = logging.getLogger(__name__)

SKINNY_MAX_ROWS = 2
# what AITER's prebuilt module_custom serves
_ARCHS = ("gfx90a", "gfx942", "gfx950")
_DTYPES = (torch.bfloat16, torch.float16)
# shapes already logged, so a server log lists what the pass covers once each
_logged: set[tuple] = set()


@functools.cache
def _cu_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _skinny_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    from aiter.ops.custom import wvSpltK

    x = x.contiguous()
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    wvSpltK(weight, x, out, x.shape[0], _cu_count(x.device.index or 0))
    if bias is not None:
        out += bias
    return out


_LIB = torch.library.Library("mstar", "FRAGMENT")
_LIB.define("skinny_gemm(Tensor x, Tensor weight, Tensor? bias) -> Tensor")
# The same op with any input layout allowed (it makes x contiguous itself), so
# inductor can finish the norm feeding a one-row GEMM in one kernel. Not for
# two rows: there inductor may pick a strided layout and pay for the copy.
_LIB.define(
    "skinny_gemm_any(Tensor x, Tensor weight, Tensor? bias) -> Tensor",
    tags=(torch.Tag.flexible_layout,),
)
for _name in ("skinny_gemm", "skinny_gemm_any"):
    _LIB.impl(_name, _skinny_gemm, "CUDA")


def _skinny_gemm_fake(x, weight, bias):
    return x.new_empty((x.shape[0], weight.shape[0]))


for _name in ("skinny_gemm", "skinny_gemm_any"):
    torch.library.register_fake(f"mstar::{_name}", _skinny_gemm_fake, lib=_LIB)


def _val(node):
    """A node's fake value: ``val`` after AOT, ``example_value`` in dynamo's graph."""
    if not isinstance(node, torch.fx.Node):
        return None
    return node.meta.get("val", node.meta.get("example_value"))


def _weight_of(node) -> torch.fx.Node | None:
    """``w`` when ``node`` is ``w`` transposed (how a linear reaches ``mm``)."""
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None
    aten = torch.ops.aten
    if node.target is aten.permute.default and list(node.args[1]) == [1, 0]:
        return node.args[0]
    if node.target is aten.t.default:
        return node.args[0]
    return None


def fits(x, weight, bias) -> bool:
    """Whether ``x @ weight.T (+ bias)`` is a GEMM the skinny kernel serves:
    up to ``SKINNY_MAX_ROWS`` rows, fp16/bf16, K a multiple of 8, a contiguous [N, K] weight with
    N > 8 (the kernel's tiles), bias absent, [N] or [rows, N]."""
    xv, wv = _val(x), _val(weight)
    if not isinstance(xv, torch.Tensor) or not isinstance(wv, torch.Tensor):
        return False
    if xv.dim() != 2 or wv.dim() != 2 or xv.dtype not in _DTYPES or wv.dtype != xv.dtype:
        return False
    rows, k = xv.shape
    n, wk = wv.shape
    if not all(isinstance(d, int) for d in (rows, k, n, wk)):
        return False  # symbolic: a forward compiled for any batch
    if not 1 <= rows <= SKINNY_MAX_ROWS or k != wk or k % 8 or n <= 8:
        return False
    if not wv.is_contiguous():
        return False
    if bias is not None:
        # a bias, or a residual inductor folded into the addmm
        bv = _val(bias)
        if not isinstance(bv, torch.Tensor) or bv.dtype != xv.dtype:
            return False
        if tuple(bv.shape) not in ((n,), (rows, n)):
            return False
    return True


def _rewrite(graph: torch.fx.Graph, node: torch.fx.Node, x, weight, bias) -> None:
    op = torch.ops.mstar.skinny_gemm_any if _val(x).shape[0] == 1 else torch.ops.mstar.skinny_gemm
    with graph.inserting_before(node):
        new = graph.call_function(op.default, (x, weight, bias))
    new.meta.update(node.meta)
    shape = (*_val(x).shape, _val(weight).shape[0], bias is not None)
    if shape not in _logged:
        _logged.add(shape)
        logger.info("skinny GEMM: rows=%d k=%d n=%d bias=%s", *shape)
    node.replace_all_uses_with(new)
    graph.erase_node(node)


class SkinnyLinearPrePass(CustomGraphPass):
    """Dynamo's graph, before decomposition: ``linear(x, w, b)`` that `fits`."""

    _TARGETS = (torch._C._nn.linear, torch.nn.functional.linear)

    def __call__(self, graph: torch.fx.Graph) -> None:
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self._TARGETS:
                continue
            args = [*node.args, None, None][:3]
            x = args[0]
            weight = node.kwargs.get("weight", args[1])
            bias = node.kwargs.get("bias", args[2])
            if fits(x, weight, bias):
                _rewrite(graph, node, x, weight, bias)

    def uuid(self) -> bytes:
        return get_hash_for_files((__file__,))


class SkinnyGemmPass(CustomGraphPass):
    """The lowered graph: ``mm(x, w.T)`` and ``addmm(b, x, w.T)`` that `fits`."""

    def __call__(self, graph: torch.fx.Graph) -> None:
        aten = torch.ops.aten
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target is aten.mm.default:
                bias, (x, wt) = None, node.args
            elif node.target is aten.addmm.default and not node.kwargs:
                bias, x, wt = node.args[:3]
            else:
                continue
            weight = _weight_of(wt)
            if weight is None or not fits(x, weight, bias):
                continue
            _rewrite(graph, node, x, weight, bias)
            if not wt.users:
                graph.erase_node(wt)

    def uuid(self) -> bytes:
        return get_hash_for_files((__file__,))


@functools.cache
def skinny_gemm_passes() -> tuple[SkinnyLinearPrePass, SkinnyGemmPass] | None:
    """The (pre-grad, post-grad) passes, or None where the kernel can't run
    (no AITER, another arch)."""
    if importlib.util.find_spec("aiter") is None or not torch.cuda.is_available():
        return None
    from mstar.platforms import current_platform

    if not current_platform.is_rocm() or current_platform.arch() not in _ARCHS:
        return None
    return SkinnyLinearPrePass(), SkinnyGemmPass()
