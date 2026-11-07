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
from mstar.model.loader.base import CheckpointRules, apply_stacked
from mstar.quantization.checkpoint import UNKNOWN, CheckpointQuant, StoredBase
from mstar.quantization.formats import AUTO, BF16, FORMATS, served
from mstar.quantization.spec import NodeQuantSpec, pattern_matches

LINEAR, EXPERTS, OTHER = "linear", "experts", "other"
MIXED = "mixed"
# stored formats the loader reads directly or dequantizes as it loads
LOADABLE = ("fp8_block", "fp8_channel", "fp8_tensor")


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


def geometry_problem(fmt: str, geometry: Geometry, direct: bool, kind: str) -> str | None:
    """Why ``fmt``'s kernels (or, ``direct``, the checkpoint's scale blocks) don't fit ``geometry``."""
    n, k = geometry.n, geometry.k
    if fmt in ("fp8_tensor", "fp8_channel"):
        if n % 16 or k % 16:
            return f"{fmt} needs N and K multiples of 16 per rank; this layer is {n} x {k}"
        if fmt == "fp8_tensor" and direct and len(geometry.rows) > 1:
            return f"one per-tensor scale can't hold the scales of its {len(geometry.rows)} checkpoint shards"
        return None
    if fmt == "fp8_block":
        bn, bk = FORMATS[fmt].block
        if n % bn or k % bk:
            return f"fp8_block runs N and K in whole {bn}x{bk} blocks per rank; this layer is {n} x {k}"
        if kind == EXPERTS and any(r % bk for r in geometry.rows):
            return f"fp8_block quantizes the expert width in groups of {bk}; it is {geometry.rows[0]} per rank"
        if direct and (any(r % bn for r in geometry.rows) or any(c % bk for c in geometry.cols)):
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
) -> tuple[dict[str, set[StoredBase]], dict[str, StoredBase]]:
    """``(per layer path: the stored formats of the checkpoint bases that land on it, per base outside any
    layer that is stored quantized)``. A base is mapped by ``<base>.weight`` through the rules, so a
    ``weight_packed`` base maps the way its ``weight`` would."""
    per_layer: dict[str, set[StoredBase]] = defaultdict(set)
    stray: dict[str, StoredBase] = {}
    if ckpt is None:
        return per_layer, stray
    stacked = list(rules.stacked_params)
    for base, stored in ckpt.bases.items():
        native = rules.native(base + ".weight")
        if native is None:
            continue
        target, _ = apply_stacked(native, stacked)
        path = target.rpartition(".")[0]
        if path in layer_paths:
            per_layer[path].add(stored)
        elif stored.format != BF16:
            stray[path] = stored
    return per_layer, stray


def _stored_of(stored: set[StoredBase]) -> tuple[str, tuple[int, int] | None]:
    if not stored:
        return BF16, None
    formats = {s.format for s in stored}
    if len(formats) > 1:
        return MIXED, None
    only = next(iter(stored))
    blocks = {s.block for s in stored}
    return only.format, (only.block if len(blocks) == 1 else None)


def resolve_plan(
    root: nn.Module, decl: QuantizationRoot, spec: NodeQuantSpec, ckpt: CheckpointQuant | None,
    device_capability: tuple[int, int] | None = None,
) -> QuantPlan:
    """The plan for ``root`` (one rank's weights root, any device), with its notes and problems."""
    plan = QuantPlan(decl.node)
    where = f"node {decl.node!r}"
    layers = discover_layers(root)
    stored_by_layer, stray = stored_formats(ckpt, decl.rules, set(layers))

    for rule in spec.rules:
        if not any(pattern_matches(path, rule.pattern) for path in layers):
            plan.problems.append(
                f"{where}: quantization {rule.origin} ({rule.pattern!r}) matches none of its "
                f"{len(layers)} quantizable layers (e.g. {sorted(layers)[:3]})"
            )

    unloadable: dict[str, list[str]] = defaultdict(list)
    for path, (kind, module) in layers.items():
        stored, block = _stored_of(stored_by_layer.get(path, set()))
        if stored not in (BF16, MIXED) and stored not in LOADABLE:
            unloadable[stored].append(path)
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
            if stored == MIXED:
                plan.notes.append(f"{path}: the checkpoint stores its shards in different formats; served bf16")

        if runtime != BF16:
            problem = _runtime_problem(module, kind, runtime, stored, block, device_capability)
            if problem is not None:
                if explicit or (origin == "format" and _always_wrong(problem)):
                    plan.problems.append(f"{where}: {path!r} can't run {runtime} ({origin}): {problem}")
                    continue
                plan.notes.append(f"{path}: stays bf16 ({origin} {runtime}: {problem})")
                runtime = BF16

        action = _action(stored, runtime)
        if action is Action.DIRECT and kind == LINEAR and not hasattr(module, "quant_prepare"):
            plan.notes.append(f"{path}: {type(module).__name__} holds no checkpoint layout; re-quantized at load")
            action = Action.REQUANT
        if action is Action.DIRECT:
            misfit = geometry_problem(runtime, layer_geometry(module, kind), direct=True, kind=kind)
            if misfit is not None:
                # the checkpoint's scales don't land on this rank's blocks: dequantize, then quantize the shard
                plan.notes.append(f"{path}: re-quantized per rank ({misfit})")
                action = Action.REQUANT
        if action is None:
            plan.problems.append(
                f"{where}: {path!r} is stored {stored} and {origin} asks {runtime}; serving one quantized format "
                f"as another compounds two roundings. Ask for {stored} or bf16 there"
            )
            continue
        plan.layers[path] = LayerPlan(path, kind, stored, runtime, action, origin, block)

    for fmt, paths in sorted(unloadable.items()):
        plan.problems.append(
            f"{where}: the checkpoint stores {len(paths)} layer(s) as {fmt} (e.g. {paths[:2]}); this build has "
            f"no loader for {fmt} checkpoints yet"
        )
    for path, stored in sorted(stray.items()):
        if stored.format not in LOADABLE:
            plan.problems.append(f"{where}: checkpoint linear {path!r} is stored {stored.format}, outside any "
                                 "layer the model declares, and this build can't dequantize it")
        else:
            plan.layers[path] = LayerPlan(path, OTHER, stored.format, BF16, Action.DEQUANT, "model", stored.block)
    return plan


def _always_wrong(problem: str) -> bool:
    # a format the build doesn't serve, or a GPU without its tensor cores: never a per-layer fallback
    return problem.startswith(("this build", "needs compute capability"))


def _runtime_problem(
    module: nn.Module, kind: str, runtime: str, stored: str, block, device_capability,
) -> str | None:
    if not served(runtime, kind):
        return f"this build serves {runtime} on no {kind} layer yet"
    need = FORMATS[runtime].min_capability
    if device_capability is not None and tuple(device_capability) < need:
        have = device_capability
        return f"needs compute capability {need[0]}.{need[1]}, this GPU is {have[0]}.{have[1]}"
    capability = module.experts_quant_capability() if kind == EXPERTS else quant_capability(module)
    if not capability.capable:
        return capability.reason
    if runtime == "fp8_block" and stored == "fp8_block" and block not in (None, FORMATS["fp8_block"].block):
        return f"the checkpoint's blocks are {block}; the kernels take {FORMATS['fp8_block'].block}"
    geometry = layer_geometry(module, kind)
    if geometry is None:
        return "its weight geometry is unknown"
    return geometry_problem(runtime, geometry, direct=False, kind=kind)


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
    "LOADABLE", "Action", "Geometry", "LayerPlan", "QuantPlan", "QuantizationRoot", "UNKNOWN",
    "discover_layers", "geometry_problem", "layer_geometry", "resolve_plan", "stored_formats",
]
