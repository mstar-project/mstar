"""Skinny GEMMs for decode-sized captured buckets on AMD GPUs.

A linear with one or two rows is a weight stream. AITER's ``wvSpltK`` reads the
weight once across every CU and beats both hipBLASLt (10-30% on MI300X) and
inductor's own one-row GEMV reduction;
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

from mstar.kernels import KernelOp, ShapePolicy, Tolerance

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


def _aiter_skinny_gemm():
    from aiter.ops.custom import wvSpltK

    def skinny_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
        wvSpltK(weight, x, out, x.shape[0], _cu_count(x.device.index or 0))
        if bias is not None:
            out += bias
        return out

    return skinny_gemm


def _skinny_gemm_reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    out = x @ weight.T
    return out if bias is None else out + bias


def _skinny_gemm_fake(x, weight, bias):
    return x.new_empty((x.shape[0], weight.shape[0]))


def _gemm_shape(x, weight, bias) -> dict[str, int]:
    return {"rows": x.shape[0], "n": weight.shape[0], "k": x.shape[1]}


def _gemm_inputs(shape, dtype, device, generator=None):
    x = torch.randn(shape["rows"], shape["k"], dtype=dtype, device=device, generator=generator)
    weight = torch.randn(shape["n"], shape["k"], dtype=dtype, device=device, generator=generator)
    return x, weight * shape["k"] ** -0.5, None


def _skinny_op(name: str, tags=()) -> KernelOp:
    return KernelOp(
        name,
        "(Tensor x, Tensor weight, Tensor? bias) -> Tensor",
        fake=_skinny_gemm_fake,
        reference=_skinny_gemm_reference,
        impls={f"rocm:{arch}": _aiter_skinny_gemm for arch in _ARCHS},
        shapes=ShapePolicy(
            dims=("rows", "n", "k"),
            key=_gemm_shape,
            inputs=_gemm_inputs,
            warm=(
                {"rows": 1, "n": 4096, "k": 4096}, {"rows": 2, "n": 24576, "k": 4096},
                {"rows": 2, "n": 4096, "k": 12288},
            ),
            dtypes=(torch.bfloat16, torch.float16),
        ),
        # one rounding of a long fp32 sum, plus summation order
        tolerance=Tolerance.accumulated(),
        tags=tags,
    )


SKINNY_GEMM = _skinny_op("skinny_gemm")
# The same op with any input layout allowed (it makes x contiguous itself), so
# inductor can finish the norm feeding a one-row GEMM in one kernel. Not for
# two rows: there inductor may pick a strided layout and pay for the copy.
SKINNY_GEMM_ANY = _skinny_op("skinny_gemm_any", tags=(torch.Tag.flexible_layout,))


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


# --- decode linears past the skinny rows ---------------------------------------------------
#
# From three rows on, a decode linear runs hipBLASLt's own pick, which at a few rows can sit well off the best of
# hipBLASLt's solutions for the shape. A product's table (mstar/kernels/configs/<arch>/<product>/decode_linear.json,
# from benchmark/kernels' linear search) names, per "rows{r}-n{n}-k{k}", the solution to run instead; shapes it
# does not name keep F.linear. The indices are hipBLASLt's, so a table tuned on another stack is ignored.
DECODE_LINEAR_MAX_ROWS = 128


def _stack() -> str:
    import triton

    return f"torch-{torch.__version__}-hip-{torch.version.hip}-triton-{triton.__version__}".replace("+", "_")


@functools.cache
def decode_linear_table(device_index: int = 0) -> dict[str, dict]:
    """This device's tuned decode-linear entries by shape key; empty without a table for this stack."""
    from mstar.kernels import configs
    from mstar.platforms import current_platform

    facts = current_platform.device_facts(device_index)
    config = configs.load_config("decode_linear", facts.arch, facts.product)
    if config is None:
        return {}
    stack = (config.get("provenance") or {}).get("stack")
    if stack != _stack():
        logger.warning("decode_linear table for %s/%s was tuned on %s, not %s: not used",
                       facts.arch, facts.product, stack, _stack())
        return {}
    return {k: v for k, v in config["classes"].items() if k != "default" and v.get("impl") == "hipblaslt"}


def _decode_key(rows: int, n: int, k: int) -> str:
    return f"rows{rows}-n{n}-k{k}"


def _hipblaslt_decode_linear():
    import aiter

    aiter.hipb_create_extension()

    def decode_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        entry = decode_linear_table(x.device.index or 0).get(_decode_key(x.shape[0], weight.shape[0], x.shape[1]))
        if entry is None:
            return torch.nn.functional.linear(x, weight, bias)
        return aiter.hipb_mm(x, weight.t(), int(entry["solution"]), bias=bias)

    return decode_linear


DECODE_LINEAR = KernelOp(
    "decode_linear",
    "(Tensor x, Tensor weight, Tensor? bias) -> Tensor",
    fake=_skinny_gemm_fake,
    reference=_skinny_gemm_reference,
    impls={f"rocm:{arch}": _hipblaslt_decode_linear for arch in ("gfx942", "gfx950")},
    shapes=ShapePolicy(
        dims=("rows", "n", "k"),
        key=_gemm_shape,
        inputs=_gemm_inputs,
        warm=({"rows": 4, "n": 4096, "k": 4096}, {"rows": 64, "n": 24576, "k": 4096}),
        dtypes=(torch.bfloat16,),
    ),
    tolerance=Tolerance.accumulated(),
)


class DecodeLinearPass(CustomGraphPass):
    """The lowered graph: ``mm(x, w.T)`` whose shape the device's table names, for decode buckets past the
    skinny rows. A table change changes the pass's hash, so inductor recompiles what it touches."""

    def __call__(self, graph: torch.fx.Graph) -> None:
        table = decode_linear_table(torch.cuda.current_device())
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
                new = graph.call_function(torch.ops.mstar.decode_linear.default, (x, weight, None))
            new.meta.update(node.meta)
            node.replace_all_uses_with(new)
            graph.erase_node(node)
            if not wt.users:
                graph.erase_node(wt)

    def uuid(self) -> bytes:
        return get_hash_for_files((__file__,)) + repr(sorted(decode_linear_table(0).items())).encode()


@functools.cache
def decode_linear_pass() -> DecodeLinearPass | None:
    """The pass, or None where nothing would change (no AITER, another arch, no table for this device)."""
    if importlib.util.find_spec("aiter") is None or not torch.cuda.is_available():
        return None
    from mstar.platforms import current_platform

    if not current_platform.is_rocm() or current_platform.arch() not in ("gfx942", "gfx950"):
        return None
    return DecodeLinearPass() if decode_linear_table(torch.cuda.current_device()) else None


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
