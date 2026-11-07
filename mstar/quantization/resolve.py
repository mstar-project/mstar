"""From a node's spec, its model's declaration and its checkpoint to one plan per layer.

The model declares, per node, its weights root (``QuantizationRoot``): which linears may go low precision and
which stay bf16 (``QuantizableLinears``), and how checkpoint names land on its parameters
(``CheckpointRules``). The resolver walks the root's modules (built on the meta device at the deployment's TP
size, so geometry is per rank), and gives each quantizable layer a ``LayerPlan``: the format the checkpoint
stores it in, the format it will run in, and what the loader does between the two. Everything it can't do is
a problem, reported at boot with the rest; a default that doesn't fit a layer is a note and the layer stays
bf16, so a rule the user wrote is never quietly ignored but ``format: fp8_block`` on a model with a few odd
widths still boots.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.quant_linear import QuantGeometry as Geometry
from mstar.model.components.quant_linear import QuantizableLinears, quant_capability
from mstar.model.loader.base import CheckpointRules, _StackedMatcher
from mstar.quantization.checkpoint import UNKNOWN, CheckpointQuant, StoredBase
from mstar.quantization.formats import AUTO, BF16, FORMATS, served
from mstar.quantization.spec import NodeQuantSpec, pattern_matches

LINEAR, EXPERTS, OTHER = "linear", "experts", "other"
MIXED = "mixed"
# stored formats the loader reads directly or dequantizes as it loads, and the checkpoint layouts it reads them in
LOADABLE = ("fp8_block", "fp8_channel", "fp8_tensor")
LOADABLE_LAYOUTS = {"w4a16": ("compressed-tensors", "awq", "gptq"), "mxfp4": ("compressed-tensors",),
                    "nvfp4": ("modelopt",)}
# stored formats ``auto`` refuses on a GPU without their kernels rather than dequantizing at 4x the memory
_NO_SILENT_DEQUANT = ("nvfp4",)


def loadable(stored: StoredBase) -> bool:
    """Whether this build loads (or dequantizes) a base stored this way."""
    if stored.format not in LOADABLE:
        return False
    layouts = LOADABLE_LAYOUTS.get(stored.format)
    return layouts is None or stored.layout in layouts


@dataclass(frozen=True)
class QuantizationRoot:
    """A node's weights root as quantization sees it. ``root_type`` is the module class the node's weight
    loader fills (``load_weights_into`` recognises it); ``build`` makes one on the current device for a
    communication group (the boot check builds it on the meta device at the deployment's TP size);
    ``checkpoint_dir`` returns the checkpoint directory when its files are local, else None."""

    node: str
    root_type: type[nn.Module]
    linears: QuantizableLinears
    rules: CheckpointRules
    checkpoint_dir: Callable[[], str | None]
    build: Callable[[CommGroup], nn.Module] | None = None


class Action(str, Enum):
    KEEP = "keep"            # stored bf16, served bf16
    DIRECT = "direct"        # served in the stored format, loaded as stored
    DEQUANT = "dequant"      # stored quantized (or mixed), dequantized to bf16 as it loads, served bf16
    QUANTIZE = "quantize"    # stored bf16, quantized once loaded
    REQUANT = "requant"      # stored quantized or mixed, dequantized as it loads, then quantized once loaded


@dataclass(frozen=True)
class LayerPlan:
    path: str
    kind: str
    stored: str
    runtime: str
    action: Action
    # what chose `runtime`: a rule's origin, `format`, `auto`, or `model` (the model keeps it bf16)
    origin: str
    block: tuple[int, int] | None = None
    # trailing sub-blocks of a fused linear that stay bf16 beside the format (``QuantizableLinears.bf16_tails``)
    tail: int = 0
    # how the checkpoint stores it, when every shard is stored alike (its layout and params pick the loader)
    source: StoredBase | None = None


@dataclass
class QuantPlan:
    node: str
    layers: dict[str, LayerPlan] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def by_action(self, *actions: Action) -> list[LayerPlan]:
        return [layer for layer in self.layers.values() if layer.action in actions]

    @property
    def active(self) -> bool:
        return any(layer.action is not Action.KEEP for layer in self.layers.values())

    def summary(self) -> str:
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for layer in self.layers.values():
            counts[(layer.runtime, layer.action.value)] += 1
        return ", ".join(f"{n} {fmt} ({action})" for (fmt, action), n in sorted(counts.items()))


def head_geometry(geometry: Geometry, tail: int) -> Geometry:
    """``geometry`` less its last ``tail`` sub-blocks: what a format holds when they stay bf16."""
    if not tail:
        return geometry
    rows = geometry.rows[:-tail]
    return Geometry(sum(rows), geometry.k, rows, geometry.cols, geometry.sharded)


def layer_geometry(module: nn.Module, kind: str) -> Geometry | None:
    if kind == EXPERTS:
        return module.experts_geometry()
    declared = getattr(module, "quant_geometry", None)
    if declared is not None:
        return declared()
    weight = getattr(module, "weight", None)
    if weight is None or weight.dim() != 2:
        return None
    n, k = weight.shape
    return Geometry(n, k, (n,), (k,))


def geometry_problem(
    fmt: str, geometry: Geometry, direct: bool, kind: str, block: tuple[int, int] | None = None,
    stored_layout: bool | None = None, source: StoredBase | None = None,
) -> str | None:
    """Why ``fmt``'s kernels (or, ``direct``, the checkpoint's scale blocks of ``block``) don't fit ``geometry``;
    ``stored_layout`` (default ``direct``): the weight keeps the checkpoint's layout, partial last block included;
    ``source``: how the checkpoint stores it (a 4-bit checkpoint's group and symmetry)."""
    stored_layout = direct if stored_layout is None else stored_layout
    n, k = geometry.n, geometry.k
    if fmt in ("fp8_tensor", "fp8_channel"):
        if n % 16 or k % 16:
            return f"{fmt} needs N and K multiples of 16 per rank; this layer is {n} x {k}"
        if fmt == "fp8_tensor" and direct and len(geometry.rows) > 1:
            return f"one per-tensor scale can't hold the scales of its {len(geometry.rows)} checkpoint shards"
        return None
    if fmt == "fp8_block":
        bn, bk = block or FORMATS[fmt].block
        # K is quantized in whole groups; quantizing here takes whole row blocks too, while a checkpoint's partial
        # last row block (GLM's kv_a, 576 rows) serves as stored where no shard boundary cuts it
        if k % bk or (not stored_layout and n % bn):
            return f"fp8_block runs N and K in whole {bn}x{bk} blocks per rank; this layer is {n} x {k}"
        if kind == EXPERTS and any(r % bk for r in geometry.rows):
            return f"fp8_block quantizes the expert width in groups of {bk}; it is {geometry.rows[0]} per rank"
        rows = geometry.rows if geometry.sharded else geometry.rows[:-1]
        if direct and (any(r % bn for r in rows) or any(c % bk for c in geometry.cols)):
            return (f"a checkpoint scale block would straddle two of this rank's sub-blocks "
                    f"(rows {list(geometry.rows)}, cols {list(geometry.cols)}, block {bn}x{bk})")
        return None
    if fmt in ("mxfp8", "nvfp4"):
        if k % 128 or n % 16:
            return f"{fmt} needs K a multiple of 128 and N of 16 per rank; this layer is {n} x {k}"
    return None


def discover_layers(root: nn.Module) -> dict[str, tuple[str, nn.Module]]:
    """``path -> (kind, module)`` for every layer a format could apply to: linears (anything with a
    ``quant_capability``) and MoE routed experts (``<block>.experts`` of a block that declares
    ``experts_quant_capability``). Other linears' children (``QuantLinear`` internals) are not walked."""
    found: dict[str, tuple[str, nn.Module]] = {}
    for path, module in root.named_modules():
        if not path:
            continue
        if hasattr(module, "experts_quant_capability"):
            found[f"{path}.experts"] = (EXPERTS, module)
        elif quant_capability(module) is not None:
            found[path] = (LINEAR, module)
    return found


def stored_formats(
    ckpt: CheckpointQuant | None, rules: CheckpointRules, layer_paths: set[str],
    param_names: set[str] | None = None, by_shard: dict[str, dict] | None = None,
) -> tuple[dict[str, set[StoredBase]], dict[str, StoredBase]]:
    """``(per layer path: the stored formats of the checkpoint bases that land on it, per base outside any
    layer that is stored quantized and lands on one of ``param_names``)``. A base is mapped by ``<base>.weight``
    through the rules, so a ``weight_packed`` base maps the way its ``weight`` would; one that lands on no
    parameter (a layer this deployment doesn't build) never loads, so it doesn't count. ``by_shard``, when given,
    also gets each layer's stored formats per stacked shard id."""
    per_layer: dict[str, set[StoredBase]] = defaultdict(set)
    stray: dict[str, StoredBase] = {}
    if ckpt is None:
        return per_layer, stray
    match = _StackedMatcher(list(rules.stacked_params))  # indexed: thousands of per-expert rules
    for base, stored in ckpt.bases.items():
        native = rules.native(base + ".weight")
        if native is None:
            continue
        target, shard = match(native)
        path = target.rpartition(".")[0]
        if path in layer_paths:
            per_layer[path].add(stored)
            if by_shard is not None:
                by_shard.setdefault(path, {}).setdefault(shard, set()).add(stored)
        elif stored.format != BF16 and (param_names is None or target in param_names):
            stray[path] = stored
    return per_layer, stray


def _head_stored(module: nn.Module, tail: int, stored: str, block, by_shard: dict) -> tuple[str | None, object]:
    """The stored format of a fused layer's rows before its ``tail`` sub-blocks, when its tail shards are all
    bf16 (``_stored_of`` of the rest); None when the layer has no tail or the tail is stored quantized."""
    blocks_of = getattr(module, "quant_shard_blocks", None)
    if not tail or blocks_of is None or not by_shard:
        return None, None
    first_tail = len(module.quant_geometry().rows) - tail
    head: set[StoredBase] = set()
    for shard, formats in by_shard.items():
        if min(blocks_of(shard)) >= first_tail:
            if any(s.format != BF16 for s in formats):
                return None, None
        else:
            head |= formats
    return _stored_of(head)[:2]


def _stored_of(stored: set[StoredBase]) -> tuple[str, tuple[int, int] | None, StoredBase | None]:
    """A layer's stored format, block and source from its shards'; shards of one 4-bit format stored with
    different params (groups, symmetry) can't share one layer's storage, so they count as mixed."""
    if not stored:
        return BF16, None, None
    formats = {s.format for s in stored}
    if len(formats) > 1:
        return MIXED, None, None
    only = next(iter(stored))
    if len(stored) == 1:
        return only.format, only.block, only
    if only.format in LOADABLE_LAYOUTS:
        return MIXED, None, None
    blocks = {s.block for s in stored}
    return only.format, (only.block if len(blocks) == 1 else None), None


def resolve_plan(
    root: nn.Module, decl: QuantizationRoot, spec: NodeQuantSpec, ckpt: CheckpointQuant | None,
    device_capability: tuple[int, int] | None = None,
) -> QuantPlan:
    """The plan for ``root`` (one rank's weights root, any device), with its notes and problems."""
    plan = QuantPlan(decl.node)
    where = f"node {decl.node!r}"
    layers = discover_layers(root)
    params = {name for name, _ in root.named_parameters()}
    by_shard: dict[str, dict] = {}
    stored_by_layer, stray = stored_formats(ckpt, decl.rules, set(layers), params, by_shard)

    for rule in spec.rules:
        if not any(pattern_matches(path, rule.pattern) for path in layers):
            plan.problems.append(
                f"{where}: quantization {rule.origin} ({rule.pattern!r}) matches none of its "
                f"{len(layers)} quantizable layers (e.g. {sorted(layers)[:3]})"
            )

    unloadable: dict[str, list[str]] = defaultdict(list)
    # (origin, runtime, why) -> the layers a rule can't apply to, reported once per rule and reason
    refused: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for path, (kind, module) in layers.items():
        shards = stored_by_layer.get(path, set())
        stored, block, source = _stored_of(shards)
        head_stored, head_block = _head_stored(module, decl.linears.tail_of(path), stored, block,
                                               by_shard.get(path, {}))
        bad = sorted({s.format if s.format not in LOADABLE else f"{s.format} ({s.layout})"
                      for s in shards if s.format != BF16 and not loadable(s)})
        if bad:
            for name in bad:
                unloadable[name].append(path)
            continue
        if any(s.param("act_order") for s in shards):
            plan.problems.append(
                f"{where}: {path!r} comes from a GPTQ act-order checkpoint (desc_act, a non-trivial g_idx); this "
                "build serves int4 groups in K order only. Use a checkpoint quantized without desc_act (or a "
                "compressed-tensors one)")
            continue
        try:
            candidate = decl.linears.selected(path)
        except ValueError as e:
            plan.problems.append(f"{where}: {e}")
            continue
        rule = spec.rule_for(path)
        if not candidate:
            if rule is not None and rule.format != BF16:
                plan.problems.append(
                    f"{where}: {rule.origin} asks {rule.format} for {path!r}, which the model keeps bf16")
            runtime, origin = BF16, "model"
        elif rule is not None:
            runtime, origin = rule.format, rule.origin
        else:
            runtime, origin = spec.format, "format"
        explicit = origin not in ("format", "model")
        if runtime == AUTO:
            origin = AUTO
            runtime = stored if stored in LOADABLE and served(stored, kind) else BF16
            # (a 4-bit head would be rounded a second time by the re-quantization below, so it serves bf16)
            if (candidate and head_stored in LOADABLE and served(head_stored, kind) and head_stored != BF16
                    and head_stored not in LOADABLE_LAYOUTS):
                # a fused layer whose bf16 tail is all that differs: its head serves as stored
                runtime, block = head_stored, head_block
            elif stored == MIXED:
                plan.notes.append(f"{path}: the checkpoint stores its shards in different formats; served bf16")

        tail = decl.linears.tail_of(path) if candidate and runtime not in (BF16, AUTO) else 0
        if tail and stored != BF16:
            # the stored rows dequantize and quantize again at load in the format's blocks (block fp8 at 128 x 128
            # gives back the checkpoint's bytes; hypothesis for other blocks), the tail stays bf16
            plan.notes.append(f"{path}: re-quantized at load, its last {tail} sub-blocks kept bf16 (stored "
                              f"{head_stored or stored} with a bf16 tail)")
            stored = MIXED
        if runtime != BF16:
            problem = _runtime_problem(module, kind, runtime, stored, block, device_capability, tail, source)
            if problem is not None:
                if explicit or (origin == "format" and _always_wrong(problem)):
                    refused[(origin, runtime, problem)].append(path)
                    continue
                if origin == AUTO and stored in _NO_SILENT_DEQUANT and _always_wrong(problem):
                    # an FP4 checkpoint dequantized to bf16 is ~4x its size: only when asked for
                    plan.problems.append(f"{where}: {path!r} is stored {stored}, which this GPU can't run ({problem}); "
                                         "ask for `format: bf16` to serve it dequantized")
                    continue
                plan.notes.append(f"{path}: stays bf16 ({origin} {runtime}: {problem})")
                runtime, tail = BF16, 0

        action = _action(stored, runtime)
        if action is Action.DIRECT and kind == LINEAR and not hasattr(module, "quant_prepare"):
            plan.notes.append(f"{path}: {type(module).__name__} holds no checkpoint layout; re-quantized at load")
            action = Action.REQUANT
        if action is Action.DIRECT:
            geometry = layer_geometry(module, kind)
            misfit = geometry_problem(runtime, geometry, direct=True, kind=kind, block=block, source=source)
            if misfit is not None:
                # the checkpoint's scales don't land on this rank's blocks: dequantize, then quantize the shard
                # (or serve it bf16, when the shard has no whole blocks to quantize)
                unquantizable = geometry_problem(runtime, geometry, direct=False, kind=kind)
                if unquantizable is None:
                    plan.notes.append(f"{path}: re-quantized per rank ({misfit})")
                    action = Action.REQUANT
                else:
                    plan.notes.append(f"{path}: stays bf16 ({misfit}; {unquantizable})")
                    runtime, action = BF16, Action.DEQUANT
        if action is None:
            refused[(origin, runtime, f"it is stored {stored}, and serving one quantized format as another "
                                      f"compounds two roundings; ask for {stored} or bf16 there")].append(path)
            continue
        tail = tail if runtime != BF16 else 0
        plan.layers[path] = LayerPlan(path, kind, stored, runtime, action, origin, block, tail, source)

    for (origin, runtime, problem), paths in refused.items():
        what = repr(paths[0]) if len(paths) == 1 else f"{len(paths)} layers (e.g. {paths[:2]})"
        plan.problems.append(f"{where}: {what} can't run {runtime} ({origin}): {problem}")
    for fmt, paths in sorted(unloadable.items()):
        plan.problems.append(
            f"{where}: the checkpoint stores {len(paths)} layer(s) as {fmt} (e.g. {paths[:2]}); this build has "
            f"no loader for {fmt} checkpoints yet"
        )
    for path, stored in sorted(stray.items()):
        if not loadable(stored):
            plan.problems.append(f"{where}: checkpoint linear {path!r} is stored {stored.format}, outside any "
                                 "layer the model declares, and this build can't dequantize it")
        else:
            plan.layers[path] = LayerPlan(path, OTHER, stored.format, BF16, Action.DEQUANT, "model", stored.block,
                                          source=stored)
    return plan


def _always_wrong(problem: str) -> bool:
    # a format the build doesn't serve, or a GPU without its tensor cores: never a per-layer fallback
    return problem.startswith(("this build", "needs compute capability"))


def _runtime_problem(
    module: nn.Module, kind: str, runtime: str, stored: str, block, device_capability, tail: int = 0,
    source=None,
) -> str | None:
    if not served(runtime, kind):
        return f"this build serves {runtime} on no {kind} layer yet"
    if kind == EXPERTS and runtime == "nvfp4" and stored != "nvfp4":
        return "this build serves nvfp4 experts as a checkpoint stores them, not quantized at load"
    need = FORMATS[runtime].min_capability
    if device_capability is not None and tuple(device_capability) < need:
        have = device_capability
        return f"needs compute capability {need[0]}.{need[1]}, this GPU is {have[0]}.{have[1]}"
    capability = module.experts_quant_capability() if kind == EXPERTS else quant_capability(module)
    if not capability.capable:
        return capability.reason
    geometry = layer_geometry(module, kind)
    if geometry is None:
        return "its weight geometry is unknown"
    if tail:
        if kind != LINEAR or len(geometry.rows) <= tail or not hasattr(module, "quant_quantize"):
            return f"a bf16 tail of {tail} sub-blocks needs a merged linear with more; it has {len(geometry.rows)}"
        geometry = head_geometry(geometry, tail)
    # as stored, the layer keeps the checkpoint's block and layout; quantized here, the format's
    as_stored = stored == runtime
    return geometry_problem(runtime, geometry, direct=False, kind=kind, block=block if as_stored else None,
                            stored_layout=as_stored, source=source if as_stored else None)


def _action(stored: str, runtime: str) -> Action | None:
    if stored == BF16:
        return Action.KEEP if runtime == BF16 else Action.QUANTIZE
    if stored == MIXED:
        return Action.DEQUANT if runtime == BF16 else Action.REQUANT
    if runtime == BF16:
        return Action.DEQUANT
    if runtime == stored:
        return Action.DIRECT
    return None


__all__ = [
    "LOADABLE", "LOADABLE_LAYOUTS", "loadable", "Action", "Geometry", "LayerPlan", "QuantPlan", "QuantizationRoot",
    "UNKNOWN",
    "discover_layers", "geometry_problem", "layer_geometry", "resolve_plan", "stored_formats",
]
