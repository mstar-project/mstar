"""Decode-sized GEMMs for captured buckets on AMD GPUs.

A linear with up to 16 rows is a weight stream. AITER's ``wvSpltK`` reads the
weight once across every CU and beats both hipBLASLt (10-30% on MI300X) and
inductor's own one-row GEMV reduction; from four rows on it loses, badly once
K * rows outgrows its LDS tile. ``mstar::decode_gemv`` serves 1-16 rows on a
weight stream in Triton (``mstar.kernels.triton.gemv``) or HIP
(``mstar.kernels.hip.gemv``), and ``mstar::decode_gemv_silu_mul`` also folds a
gated MLP's ``silu(gate) * up`` into the gate/up projection's store.

Inductor can't pick these, so two graph passes swap such linears for them:
one before inductor's decompositions (a one-row ``mm`` becomes a reduction
there, and the gated activation is still one ``mul`` of a ``silu``), one after
(``mm``/``addmm`` that only appear in the lowered graph). Which kernel serves
a shape is the device's tuned ``decode_gemv`` config (``mstar.kernels.
configs``): ``"aiter"``, ``"triton"``, ``"hip"`` or ``"hipblaslt"`` (left to
inductor).
Without one, 1-2 row linears go to AITER and the rest stay on hipBLASLt, as
before the Triton kernel. ``RocmPlatform.inductor_overrides`` installs the
passes for buckets of up to ``DECODE_GEMV_MAX_ROWS`` tokens only.

AITER, Triton and the HIP build load on first call, never at module import.
"""

from __future__ import annotations

import functools
import importlib.util
import logging
import operator

import torch
import torch.nn.functional as F
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files

from mstar.kernels import KernelOp, ShapePolicy, Tolerance, configs

logger = logging.getLogger(__name__)

# what AITER serves when no tuned config says otherwise
SKINNY_MAX_ROWS = 2
# the largest bucket the passes are installed for
DECODE_GEMV_MAX_ROWS = 16
# what AITER's prebuilt module_custom serves
_ARCHS = ("gfx90a", "gfx942", "gfx950")
IMPLS = ("aiter", "triton", "hip", "hipblaslt")
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

# the shape classes a decode_gemv config is keyed by
_ROW_CLASSES = (
    ("rows1", lambda s: s["rows"] == 1), ("rows2", lambda s: s["rows"] == 2),
    ("rows4", lambda s: s["rows"] <= 4), ("rows8", lambda s: s["rows"] <= 8),
    ("rows16", lambda s: s["rows"] <= DECODE_GEMV_MAX_ROWS),
)


def _decode_kernel(op: KernelOp, silu_mul: bool, rows: int, n: int, k: int, dtype: torch.dtype):
    """The kernel (HIP or Triton, at its knobs) the device's config names for
    this shape, as a callable with the op's signature. HIP serves bf16 only."""
    from mstar.kernels.triton import gemv as triton_gemv

    entry = op.config({"rows": rows, "n": n, "k": k}) or {}
    knobs = {name: value for name, value in entry.items() if name != "impl"}
    if entry.get("impl") == "hip" and dtype == torch.bfloat16:
        from mstar.kernels.hip import gemv as hip_gemv

        kernel, knobs = (hip_gemv.decode_gemv_silu_mul if silu_mul else hip_gemv.decode_gemv), {
            **hip_gemv.DEFAULT_KNOBS, **knobs}
    else:
        kernel = triton_gemv.decode_gemv_silu_mul if silu_mul else triton_gemv.decode_gemv
        knobs = {**triton_gemv.DEFAULT_KNOBS, **{k: v for k, v in knobs.items() if k in triton_gemv.KNOBS}}
    return functools.partial(kernel, knobs=knobs)


def _tuned(name: str, silu_mul: bool):
    """A factory for ``name``'s implementation: each shape's kernel, picked
    from the config on the shape's first call."""

    def factory():
        op = DECODE_GEMV_SILU_MUL if silu_mul else DECODE_GEMV
        chosen: dict[tuple, object] = {}

        def run(x: torch.Tensor, weight: torch.Tensor, *bias) -> torch.Tensor:
            key = (x.shape[0], weight.shape[0], x.shape[1], x.dtype)
            kernel = chosen.get(key)
            if kernel is None:
                kernel = chosen[key] = _decode_kernel(op, silu_mul, *key)
            return kernel(x, weight, *bias)

        return run

    factory.__name__ = f"_tuned_{name}"
    return factory


def _silu_mul_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate, up = (x @ weight.T).chunk(2, dim=-1)
    return F.silu(gate) * up


def _silu_mul_fake(x, weight):
    return x.new_empty((x.shape[0], weight.shape[0] // 2))


def _silu_mul_inputs(shape, dtype, device, generator=None):
    x, weight, _ = _gemm_inputs(shape, dtype, device, generator)
    return x, weight


DECODE_GEMV = KernelOp(
    "decode_gemv",
    "(Tensor x, Tensor weight, Tensor? bias) -> Tensor",
    fake=_skinny_gemm_fake,
    reference=_skinny_gemm_reference,
    impls={f"rocm:{arch}": _tuned("decode_gemv", False) for arch in ("gfx942", "gfx950")},
    shapes=ShapePolicy(
        dims=("rows", "n", "k"),
        key=_gemm_shape,
        inputs=_gemm_inputs,
        classes=_ROW_CLASSES,
        warm=(
            {"rows": 1, "n": 4096, "k": 4096}, {"rows": 4, "n": 4096, "k": 12288},
            {"rows": 16, "n": 24576, "k": 4096},
        ),
        dtypes=(torch.bfloat16, torch.float16),
    ),
    tolerance=Tolerance.accumulated(),
)

# ``silu(x @ gate.T) * (x @ up.T)`` for ``weight = [gate; up]``; shapes name
# the stacked weight's N
DECODE_GEMV_SILU_MUL = KernelOp(
    "decode_gemv_silu_mul",
    "(Tensor x, Tensor weight) -> Tensor",
    fake=_silu_mul_fake,
    reference=_silu_mul_reference,
    impls={f"rocm:{arch}": _tuned("decode_gemv_silu_mul", True) for arch in ("gfx942", "gfx950")},
    shapes=ShapePolicy(
        dims=("rows", "n", "k"),
        key=lambda x, weight: {"rows": x.shape[0], "n": weight.shape[0], "k": x.shape[1]},
        inputs=_silu_mul_inputs,
        classes=_ROW_CLASSES,
        warm=({"rows": 1, "n": 24576, "k": 4096}, {"rows": 16, "n": 12288, "k": 2048}),
        dtypes=(torch.bfloat16, torch.float16),
    ),
    tolerance=Tolerance.accumulated(),
)


@functools.cache
def _have_aiter() -> bool:
    return importlib.util.find_spec("aiter") is not None


def gemv_impl(rows: int, n: int, k: int) -> str:
    """Which kernel serves ``[rows, k] @ [n, k].T`` here: ``"aiter"``,
    ``"triton"``, ``"hip"`` or ``"hipblaslt"``, from the device's
    ``decode_gemv`` config."""
    entry = DECODE_GEMV.config({"rows": rows, "n": n, "k": k})
    if entry is None:
        impl = "aiter" if rows <= SKINNY_MAX_ROWS else "hipblaslt"
    else:
        impl = entry.get("impl", "hipblaslt")
        if impl not in IMPLS:
            raise ValueError(f"decode_gemv config names an unknown implementation {impl!r}")
    return "hipblaslt" if impl == "aiter" and not _have_aiter() else impl


def fuses_silu_mul(rows: int, n: int, k: int) -> bool:
    """Whether a gated MLP's gate/up projection of stacked width ``n`` and its
    ``silu(gate) * up`` run as one ``decode_gemv_silu_mul`` here."""
    entry = DECODE_GEMV_SILU_MUL.config({"rows": rows, "n": n, "k": k})
    return entry is not None and entry.get("impl") in ("triton", "hip")


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


def _shape(x, weight, bias) -> tuple[int, int, int] | None:
    """``(rows, n, k)`` when ``x @ weight.T (+ bias)`` is a GEMM the decode
    kernels serve: up to ``DECODE_GEMV_MAX_ROWS`` rows, fp16/bf16, K a multiple
    of 8, a contiguous [N, K] weight with N > 8 (AITER's tiles), bias absent,
    [N] or [rows, N]; else None."""
    xv, wv = _val(x), _val(weight)
    if not isinstance(xv, torch.Tensor) or not isinstance(wv, torch.Tensor):
        return None
    if xv.dim() != 2 or wv.dim() != 2 or xv.dtype not in _DTYPES or wv.dtype != xv.dtype:
        return None
    rows, k = xv.shape
    n, wk = wv.shape
    if not all(isinstance(d, int) for d in (rows, k, n, wk)):
        return None  # symbolic: a forward compiled for any batch
    if not 1 <= rows <= DECODE_GEMV_MAX_ROWS or k != wk or k % 8 or n <= 8:
        return None
    if not wv.is_contiguous():
        return None
    if bias is not None:
        # a bias, or a residual inductor folded into the addmm
        bv = _val(bias)
        if not isinstance(bv, torch.Tensor) or bv.dtype != xv.dtype:
            return None
        if tuple(bv.shape) not in ((n,), (rows, n)):
            return None
    return rows, n, k


def _op_for(x, weight, bias):
    """The op to run ``x @ weight.T (+ bias)`` as, or None to leave it to inductor."""
    shape = _shape(x, weight, bias)
    if shape is None:
        return None
    impl = gemv_impl(*shape)
    if impl == "aiter":
        return torch.ops.mstar.skinny_gemm_any if shape[0] == 1 else torch.ops.mstar.skinny_gemm
    if impl in ("triton", "hip"):
        return torch.ops.mstar.decode_gemv
    return None


def fits(x, weight, bias) -> bool:
    """Whether a pass rewrites ``x @ weight.T (+ bias)``."""
    return _op_for(x, weight, bias) is not None


def _log_once(kind: str, shape: tuple) -> None:
    if (kind, *shape) not in _logged:
        _logged.add((kind, *shape))
        logger.info("decode GEMM on %s: rows=%d n=%d k=%d", kind, *shape)


def _rewrite(graph: torch.fx.Graph, node: torch.fx.Node, x, weight, bias) -> bool:
    op = _op_for(x, weight, bias)
    if op is None:
        return False
    with graph.inserting_before(node):
        new = graph.call_function(op.default, (x, weight, bias))
    new.meta.update(node.meta)
    _log_once(op.default.name().removeprefix("mstar::") + (" + bias" if bias is not None else ""),
              _shape(x, weight, bias))
    node.replace_all_uses_with(new)
    graph.erase_node(node)
    return True


_SPLITS = ("split", torch.split, torch.Tensor.split)
_SILUS = (F.silu, torch.nn.functional.silu)
_MULS = (operator.mul, torch.mul, torch.Tensor.mul)


def _is_call(node, targets) -> bool:
    return isinstance(node, torch.fx.Node) and node.op in ("call_function", "call_method") and node.target in targets


def _gated_activation(node: torch.fx.Node):
    """``(linear, x, weight)`` when ``node`` is ``silu(gate) * up`` for ``gate,
    up = linear(x, weight).split(n // 2, dim=-1)`` with no other use of the
    linear or its halves; else None."""
    if not _is_call(node, _MULS) or len(node.args) != 2 or node.kwargs:
        return None
    act, up = node.args
    if not _is_call(act, _SILUS) or act.kwargs.get("inplace") or len(act.args) != 1:
        return None
    gate = act.args[0]
    halves = []
    for half in (gate, up):
        if not _is_call(half, (operator.getitem,)) or len(half.users) != 1:
            return None
        halves.append(half)
    split = gate.args[0]
    if up.args[0] is not split or (gate.args[1], up.args[1]) != (0, 1) or not _is_call(split, _SPLITS):
        return None
    if len(split.users) != 2 or len(act.users) != 1:
        return None
    linear = split.args[0]
    if not _is_call(linear, SkinnyLinearPrePass._TARGETS) or len(linear.users) != 1:
        return None
    args = [*linear.args, None, None][:3]
    x, weight = args[0], linear.kwargs.get("weight", args[1])
    if linear.kwargs.get("bias", args[2]) is not None:
        return None
    shape = _shape(x, weight, None)
    if shape is None or shape[1] % 2:
        return None
    size = split.args[1] if len(split.args) > 1 else split.kwargs.get("split_size")
    dim = split.args[2] if len(split.args) > 2 else split.kwargs.get("dim", 0)
    if size != shape[1] // 2 or dim not in (-1, 1):
        return None
    return linear, x, weight, shape


def _fuse_gated_activation(graph: torch.fx.Graph, node: torch.fx.Node) -> bool:
    match = _gated_activation(node)
    if match is None:
        return False
    linear, x, weight, shape = match
    # the config compares the fused op with the best unfused pair
    if not fuses_silu_mul(*shape):
        return False
    with graph.inserting_before(node):
        new = graph.call_function(torch.ops.mstar.decode_gemv_silu_mul.default, (x, weight))
    new.meta.update(node.meta)
    _log_once("decode_gemv_silu_mul", shape)
    node.replace_all_uses_with(new)
    act, up = node.args
    gate = act.args[0]
    split = gate.args[0]
    for dead in (node, act, gate, up, split, linear):
        graph.erase_node(dead)
    return True


class SkinnyLinearPrePass(CustomGraphPass):
    """Dynamo's graph, before decomposition: ``linear(x, w, b)`` that `fits`."""

    _TARGETS = (torch._C._nn.linear, torch.nn.functional.linear)

    def __call__(self, graph: torch.fx.Graph) -> None:
        # gated activations first: they consume the gate/up linear
        for node in list(graph.nodes):
            _fuse_gated_activation(graph, node)
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target not in self._TARGETS:
                continue
            args = [*node.args, None, None][:3]
            x = args[0]
            weight = node.kwargs.get("weight", args[1])
            bias = node.kwargs.get("bias", args[2])
            _rewrite(graph, node, x, weight, bias)

    def uuid(self) -> bytes:
        return _pass_hash()


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
            if weight is None or not _rewrite(graph, node, x, weight, bias):
                continue
            if not wt.users:
                graph.erase_node(wt)

    def uuid(self) -> bytes:
        return _pass_hash()


def _pass_hash() -> bytes:
    """Inductor's cache key for the passes: this file and the configs that pick
    the kernel per shape, so a changed config recompiles."""
    from mstar.platforms import current_platform

    files = [__file__]
    if torch.cuda.is_available():
        facts = current_platform.device_facts()
        for op in (DECODE_GEMV, DECODE_GEMV_SILU_MUL):
            for product in (facts.product, None):
                path = configs.config_path(op.name, facts.arch or "unknown", product)
                if path.is_file():
                    files.append(str(path))
    return get_hash_for_files(tuple(files))


@functools.cache
def skinny_gemm_passes() -> tuple[SkinnyLinearPrePass, SkinnyGemmPass] | None:
    """The (pre-grad, post-grad) passes, or None where no decode kernel can run
    (another arch; neither AITER nor a tuned config for the Triton kernel)."""
    if not torch.cuda.is_available():
        return None
    from mstar.platforms import current_platform

    if not current_platform.is_rocm() or current_platform.arch() not in _ARCHS:
        return None
    facts = current_platform.device_facts()
    if not _have_aiter() and configs.load_config(DECODE_GEMV.name, facts.arch, facts.product) is None:
        return None
    return SkinnyLinearPrePass(), SkinnyGemmPass()
