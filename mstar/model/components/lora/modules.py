"""Per-request LoRA declared from a model's own modules.

``module_lora_spec`` walks a node's weights root (the module its weight loader fills;
built on the meta device at one rank, so any process can declare it, and only once the
spec is read, so a deployment without LoRA builds nothing), finds every
LoRA-capable module (``capable.py``), groups them into targets by their role in a layer
stack (``layers.N.<role>`` / ``blocks.N.<role>``; a module outside the stacks is a target
of its own), and places checkpoint modules with the root's own ``CheckpointRules`` (the
ones its weight loader uses). The model states only what its modules cannot: how a
target's rows relate to the step's tokens (``rows``), modules it calls outside the step
(``exclude``, ``deferred``), names kept for compatibility (``names``), the prefixes
adapters write the root under (``adapter_roots``) and adapter-only renames (``aliases``).
A module itself names the children it runs through their weights rather than their
forward (``lora_unservable``), which the walk refuses.

``join_lora_specs`` makes one spec of several nodes' (an LLM's and its vision tower's),
each node's pool holding its own targets (``LoraSpec.for_node``).

``bind_module_lora`` binds each placed module to its delta when the node's resources
include its pool (``NodeSubmodule.bind_node_resources`` calls it), checking the real,
possibly sharded, module against the declaration first. A deployment without a ``lora:``
block builds no pool, so nothing is bound and every module runs as before.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NamedTuple

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.lora.config import LORA_RESOURCE_KEY, LoraModuleRef, LoraSpec, LoraTarget
from mstar.model.components.lora.capable import (
    LoraCapable,
    lora_capability,
    make_lora_capable,
)
from mstar.model.loader.base import CheckpointRules, StackedParamRule, apply_stacked

logger = logging.getLogger(__name__)

# a target whose linear sees one row per segment (an LM head over each row's last token)
PER_SEGMENT = "segment"
RowSpec = int | Literal["segment"]

# `<stack>.<layer>.<role>`, the first stack in the path
_STACK = re.compile(r"^(?P<stack>(?:.*?\.)?(?:layers|blocks))\.(?P<layer>\d+)\.(?P<role>.+)$")


class AdapterRoot(NamedTuple):
    """A prefix adapters write this node's modules under, and the checkpoint prefix the
    weight loader's remap reads in its place (the two are equal for the checkpoint's own
    layout, which is listed too: a spec claims the modules under its roots)."""

    adapter_prefix: str
    checkpoint_prefix: str


class LoraMap(ABC):
    """A ``LoraSpec.module_map`` declared from a model's modules, which binding reads:
    places a checkpoint module on the spec's targets, says why one lands on none, and
    gives the map that binds each node's modules."""

    @abstractmethod
    def __call__(self, module: str) -> LoraModuleRef | None:
        ...

    @abstractmethod
    def explain(self, module: str) -> str | None:
        """Why ``module`` lands on no target, when the walk recorded a reason."""

    @abstractmethod
    def ignores(self, module: str) -> bool:
        """``LoraSpec.ignored_modules``: a module the model never runs."""

    @abstractmethod
    def part(self, node: str) -> ModuleLoraMap:
        """The map that binds ``node``'s modules."""


class ModuleLoraMap(LoraMap):
    """Places a checkpoint module on the spec's targets, through the module the weight
    loader fills: the module path (or its rewrite under each ``AdapterRoot``, then
    ``aliases``) goes through the loader's ``CheckpointRules`` (prefix, remap, stacked
    rules, then any adapter-only stacked rules) to a native module, which ``bindings``
    names. Also what binding reads: ``bindings`` below ``root_path`` (the weights root's
    path in the node submodule), the ``deferred`` ones the node adds itself, and why a
    module the walk found is not a target (``unservable``)."""

    def __init__(
        self,
        bindings: Mapping[str, LoraModuleRef],
        rules: CheckpointRules = CheckpointRules(),
        adapter_stacked_params: Sequence[StackedParamRule] = (),
        adapter_roots: Sequence[AdapterRoot] = (),
        aliases: Mapping[str, str] | None = None,
        unservable: Mapping[str, str] | None = None,
        ignore: Sequence[str] = (),
        root_path: str = "",
        deferred: frozenset[str] = frozenset(),
    ):
        self.bindings = dict(bindings)
        self.unservable = dict(unservable or {})
        self.root_path = root_path
        self.deferred = frozenset(deferred)
        self.rules = rules
        self._stacked = [*rules.stacked_params, *adapter_stacked_params]
        roots = [AdapterRoot(*r) for r in adapter_roots]
        if rules.prefix:
            # the checkpoint's own layout, and the root's alone
            roots += [AdapterRoot(rules.prefix, rules.prefix), AdapterRoot("", rules.prefix)]
        self._roots = tuple(roots)
        self._aliases = dict(aliases or {})
        self._ignore = tuple(ignore)
        # modules with a target per parameter (MoE experts), which a checkpoint adapts per
        # expert, never as the container
        self._containers = frozenset(_module_path(b, ref) for b, ref in self.bindings.items() if ref.param)
        # placements by module path, which an adapter header repeats per factor and file
        self._placed: dict[str, LoraModuleRef | None] = {}

    def _candidates(self, module: str):
        yield self._aliases.get(module, module)
        for root in self._roots:
            if module.startswith(root.adapter_prefix):
                rewritten = root.checkpoint_prefix + module[len(root.adapter_prefix):]
                yield self._aliases.get(rewritten, rewritten)

    def _native(self, candidate: str) -> tuple[str, str | int | None] | None:
        name = self.rules.native(candidate + ".weight")
        if not name:
            return None
        native, shard_id = apply_stacked(name, self._stacked)
        return native.removesuffix(".weight"), shard_id

    def __call__(self, module: str) -> LoraModuleRef | None:
        if module not in self._placed:
            self._placed[module] = self._place(module)
        return self._placed[module]

    def _place(self, module: str) -> LoraModuleRef | None:
        for candidate in self._candidates(module):
            found = self._native(candidate)
            if found is None:
                continue
            ref = self.bindings.get(found[0])
            if ref is not None:
                return ref._replace(shard_id=found[1])
        return None

    def explain(self, module: str) -> str | None:
        """Why ``module`` lands on no target, when the walk recorded a reason for it or
        for a module holding it (an expert's projection, inside the experts)."""
        for candidate in self._candidates(module):
            found = self._native(candidate)
            if found is None:
                continue
            if found[0] in self._containers:
                return (
                    f"{module}: experts take LoRA per expert, experts.<N>.<proj>; fused 3-D "
                    "expert adapters are not read yet"
                )
            parts = found[0].split(".")
            for end in range(len(parts), 0, -1):
                reason = self.unservable.get(".".join(parts[:end]))
                if reason is not None:
                    return f"{module}: {reason}"
        return None

    def claim(self, module: str) -> int:
        """How specifically this node claims ``module``: its longest adapter root prefix
        that ``module`` is under, or -1."""
        return max(
            (len(r.adapter_prefix) for r in self._roots if module.startswith(r.adapter_prefix)),
            default=-1,
        )

    def ignores(self, module: str) -> bool:
        return any(fnmatch.fnmatchcase(module, p) for p in self._ignore)

    def part(self, node: str) -> ModuleLoraMap:
        return self


class _JoinedLoraMap(LoraMap):
    """Several nodes' ``ModuleLoraMap``s as one spec's (``join_lora_specs``): a
    checkpoint module goes to the node whose adapter roots claim it most specifically,
    which places or refuses it."""

    def __init__(self, parts: Mapping[str, ModuleLoraMap]):
        self._parts = dict(parts)

    def _claimant(self, module: str) -> ModuleLoraMap:
        return max(self._parts.values(), key=lambda m: m.claim(module))

    def __call__(self, module: str) -> LoraModuleRef | None:
        return self._claimant(module)(module)

    def explain(self, module: str) -> str | None:
        return self._claimant(module).explain(module)

    def ignores(self, module: str) -> bool:
        return any(m.ignores(module) for m in self._parts.values())

    def part(self, node: str) -> ModuleLoraMap:
        return self._parts[node]


def _match(path: str, patterns) -> str | None:
    """The first pattern (fnmatch, children included) ``path`` falls under."""
    for p in patterns:
        if fnmatch.fnmatchcase(path, p) or fnmatch.fnmatchcase(path, p + ".*"):
            return p
    return None


def _declared(path: str, declared: Mapping[str, str]) -> str | None:
    """The reason ``declared`` gives for ``path`` or a module holding it."""
    while True:
        reason = declared.get(path)
        if reason is not None or "." not in path:
            return reason
        path = path.rsplit(".", 1)[0]


class _Found(NamedTuple):
    path: str
    layer: int
    target: LoraTarget
    param: str | None = None


_EXCLUDED = "excluded by the model's LoRA declaration"


def module_lora_spec(
    build: Callable[[CommGroup], nn.Module],
    *,
    nodes: set[str],
    root_path: str,
    rules: CheckpointRules | None = None,
    adapter_stacked_params: Sequence[StackedParamRule] = (),
    adapter_roots: Sequence[AdapterRoot | tuple[str, str]] = (),
    aliases: Mapping[str, str] | None = None,
    rows: Mapping[str, RowSpec] | None = None,
    names: Mapping[str, str] | None = None,
    exclude: Sequence[str] | Mapping[str, str] = (),
    deferred: Sequence[str] = (),
    ignore: Sequence[str] = (),
    embeddings: bool = False,
    convert_keys: Callable | None = None,
) -> LoraSpec:
    """A node's LoRA spec, a target per role of the LoRA-capable modules under its weights
    root, keyed ``LORA_RESOURCE_KEY``: the key the serving config's ``resources: lora:``
    block and the API server's adapter selection name.

    ``build`` makes the root, the module the node's weight loader fills, as one rank sees
    it: it gets the trivial group (the pool shards the targets for its own rank) and runs
    on the meta device, when the spec is first read (``LoraSpec.declare``). ``root_path``
    is the root's attribute path in the node's submodule (``""`` for the submodule
    itself), where binding finds it. ``rules`` are the loader's
    (``CheckpointRules``); by default the root's own ``checkpoint_rules()``, so placement
    cannot drift from loading. ``adapter_stacked_params`` adds rules only adapters need
    (Tinker's split projections). Module paths, ``rows`` / ``names`` keys and ``exclude``
    / ``deferred`` patterns are relative to ``root``; a role is the path below a stack
    layer (``self_attn.qkv_proj``), or the whole path for a module outside the stacks.

    * ``rows``: role (fnmatch) -> ``PER_SEGMENT`` or a fold ``k``, for a target that sees
      other rows than the step's tokens (an LM head after the last-token select, a merger
      folding ``k`` patches). Every other target reads one row per token.
    * ``names``: role -> target name, overriding the default (the role's last component,
      or the role with dots as underscores where two would collide).
    * ``exclude``: module paths (fnmatch, children included) not served, optionally
      mapped to the reason (a module called outside the planned step); an adapter
      touching one is refused with it.
    * ``deferred``: modules whose delta the node adds itself, where the step's rows are
      planned (``preprocess``, the forward), because it also calls them before the step
      (an embedding run in ``prepare_inputs``): bound as ``lora_deferred``, which the
      module's own forward never adds.
    * ``ignore``: checkpoint module paths (fnmatch) an adapter may hold that this
      deployment never runs, neither loaded nor refused.
    * ``embeddings``: serve token-embedding targets. Off by default: a node that embeds
      in ``prepare_inputs`` calls the embedding outside its planned step; turn it on when
      every call is in the step, or list the embedding in ``deferred``.

    A module with a target per parameter (``LoraCapable.lora_targets``: MoE experts'
    ``gate_up_proj`` and ``down_proj``) is bound once per parameter, under its role plus
    the parameter, named ``<module>_<parameter>`` (``experts_gate_up_proj``) by default.
    Its targets (``experts > 1``) are declared in ``dropped_targets``, served only when the
    YAML's ``targets:`` names them.
    Linear-like modules that cannot take a delta (a depthwise conv, a router) are
    recorded with the reason, which an adapter touching them is refused with, and so are
    the children a module declares in ``lora_unservable`` (path below it -> reason, each
    one of its submodules): ones it runs through their weights itself, never through
    their forward (folded at load, an ``F.linear`` over the weight).
    """
    def declare() -> dict[str, Any]:
        with torch.device("meta"):
            root = build(CommGroup.trivial())
        own_rules = getattr(root, "checkpoint_rules", None)
        placing = rules
        if placing is None:
            placing = own_rules() if own_rules is not None else CheckpointRules()
        elif own_rules is not None:
            raise ValueError(
                f"{type(root).__name__} states its checkpoint rules; LoRA placement takes them "
                "from it rather than from a copy"
            )
        excluded = dict(exclude) if isinstance(exclude, Mapping) else dict.fromkeys(exclude, _EXCLUDED)
        row_specs, renames = dict(rows or {}), dict(names or {})
        groups: dict[tuple[str | None, str], list[_Found]] = defaultdict(list)
        unservable: dict[str, str] = {}
        declared: dict[str, str] = {}  # modules' own `lora_unservable`, by exact path
        for path, module in root.named_modules():
            # children the module runs through their weights itself (pre-order: before them)
            prefix = f"{path}." if path else ""
            for child, reason in getattr(module, "lora_unservable", {}).items():
                try:
                    module.get_submodule(child)
                except AttributeError:
                    raise ValueError(
                        f"{type(module).__name__}.lora_unservable names {child!r}, which is not "
                        "one of its submodules"
                    ) from None
                declared[prefix + child] = reason
            if not path:
                continue
            capability = lora_capability(module)
            if capability is None:
                continue
            reason = _declared(path, declared)
            if reason is None and (pattern := _match(path, excluded)) is not None:
                reason = excluded[pattern]
            if reason is not None:
                unservable[path] = reason
                continue
            if capability.capable is None:
                unservable[path] = capability.reason
                continue
            m = _STACK.match(path)
            for param, target in _lora_targets(capability.capable, module).items():
                if target.kind == "embedding" and not embeddings:
                    unservable[path] = "a token embedding, which this node does not serve LoRA on"
                    continue
                stack, role, layer = (None, path, 0) if m is None else (m["stack"], m["role"], int(m["layer"]))
                if param is not None:
                    role = f"{role}.{param}"
                    renames.setdefault(role, f"{role.split('.')[-2]}_{param}")
                groups[(stack, role)].append(_Found(path, layer, target, param))
        if not groups:
            raise ValueError(f"{type(root).__name__} has no LoRA-capable modules")

        stacked = [f.layer for (stack, _), found in groups.items() if stack is not None for f in found]
        num_layers = max(stacked, default=0) + 1
        target_names = _target_names(list(groups), renames)
        unused = set(renames) - {role for _, role in groups}
        if unused:
            raise ValueError(f"`names` names roles {sorted(unused)} that {type(root).__name__} does not have")

        targets: dict[str, LoraTarget] = {}
        dropped: dict[str, LoraTarget] = {}
        bindings: dict[str, LoraModuleRef] = {}
        for key, found in groups.items():
            stack, role = key
            first = found[0].target
            differ = [f.path for f in found if f.target != first]
            if differ:
                raise ValueError(
                    f"LoRA target {role!r} has other shapes at {differ[:3]} than at "
                    f"{found[0].path}; exclude them or give them a stack of their own"
                )
            layers = tuple(sorted(f.layer for f in found))
            if stack is not None and layers == tuple(range(num_layers)):
                layers = None
            name = target_names[key]
            # experts targets are served only where the YAML's `targets:` names them: each
            # slot holds every expert's A and B (GBs for a large MoE)
            (targets if first.experts == 1 else dropped)[name] = dataclasses.replace(
                first, layers=layers, **_row_fields(row_specs, role),
            )
            for f in found:
                path = f.path if f.param is None else f"{f.path}.{f.param}"
                bindings[path] = LoraModuleRef(name, f.layer, param=f.param)
        unmatched = [p for p in row_specs if not any(fnmatch.fnmatchcase(r, p) for _, r in groups)]
        if unmatched:
            raise ValueError(f"`rows` patterns {unmatched} match no LoRA target role")
        deferred_paths = frozenset(p for p in bindings if _match(p, deferred) is not None)
        unmatched = [p for p in deferred if not any(_match(b, [p]) for b in bindings)]
        if unmatched:
            raise ValueError(f"`deferred` patterns {unmatched} match no LoRA target module")
        per_param = sorted(p for p in deferred_paths if bindings[p].param is not None)
        if per_param:
            raise ValueError(
                f"`deferred` matches {per_param[:3]}, a target per parameter (MoE experts), whose "
                "delta its dispatch adds; a node cannot add it itself"
            )
        if unservable:
            logger.info(
                "LoRA on %s: %d module(s) take no delta, e.g. %s",
                sorted(nodes), len(unservable), [f"{p} ({r})" for p, r in list(unservable.items())[:3]],
            )

        module_map = ModuleLoraMap(
            bindings, rules=placing, adapter_stacked_params=adapter_stacked_params,
            adapter_roots=adapter_roots, aliases=aliases, unservable=unservable, ignore=ignore,
            root_path=root_path, deferred=deferred_paths,
        )
        return {"num_layers": num_layers, "targets": targets, "dropped_targets": dropped,
                "module_map": module_map, "ignored_modules": module_map.ignores}

    return LoraSpec(
        resource_key=LORA_RESOURCE_KEY, nodes=set(nodes), convert_keys=convert_keys, declare=declare,
    )



def _lora_targets(capable: type, module: nn.Module) -> dict[str | None, LoraTarget]:
    """``capable.lora_targets`` of ``module``, which the walk asks before any swap: a
    plain linear is not its capable twin yet, so its one target is the twin's."""
    if capable.lora_targets is LoraCapable.lora_targets:
        return {None: capable.lora_target(module)}
    return capable.lora_targets(module)


def _row_fields(rows: Mapping[str, RowSpec], role: str) -> dict[str, Any]:
    matched = [v for p, v in rows.items() if fnmatch.fnmatchcase(role, p)]
    if not matched:
        return {}
    if len(set(matched)) > 1:
        raise ValueError(f"`rows` gives LoRA target role {role!r} several row layouts: {matched}")
    value = matched[0]
    if value == PER_SEGMENT:
        return {"per_segment": True}
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"`rows` for {role!r} must be {PER_SEGMENT!r} or a fold >= 1; got {value!r}")
    return {"token_fold": value}


def _target_names(keys: list[tuple[str | None, str]], names: Mapping[str, str]) -> dict:
    """Each group's target name: an override, else the role's last component, else the
    role with dots as underscores, else that behind its stack's."""
    out = {key: names.get(key[1], key[1].rsplit(".", 1)[-1]) for key in keys}
    for widen in (
        lambda key: key[1].replace(".", "_"),
        lambda key: f"{(key[0] or 'root').replace('.', '_')}_{key[1].replace('.', '_')}",
    ):
        counts = Counter(out.values())
        out = {
            key: name if counts[name] == 1 or key[1] in names else widen(key)
            for key, name in out.items()
        }
    counts = Counter(out.values())
    clash = sorted(name for name, n in counts.items() if n > 1)
    if clash:
        raise ValueError(f"LoRA target names {clash} name several roles; rename them with `names`")
    return out


def join_lora_specs(*specs: LoraSpec) -> LoraSpec:
    """One spec over the nodes of ``specs``, each a node's from ``module_lora_spec`` (an
    LLM's and its vision tower's): their targets, each naming its node, under the first's
    key. A checkpoint module is placed by the node whose ``AdapterRoot``s claim it most
    specifically, so one adapter file serves them all. Declared, like its parts, on first
    read."""
    convert_keys = specs[0].convert_keys
    if any(spec.convert_keys is not convert_keys for spec in specs):
        raise ValueError("joined LoRA specs must read checkpoints alike (convert_keys)")

    def declare() -> dict[str, Any]:
        joined: dict[str, dict[str, LoraTarget]] = {"targets": {}, "dropped_targets": {}}
        parts: dict[str, ModuleLoraMap] = {}
        for spec in specs:
            (node,) = spec.nodes
            for field, into in joined.items():
                for name, target in getattr(spec, field).items():
                    if name in joined["targets"] or name in joined["dropped_targets"]:
                        raise ValueError(f"LoRA target {name!r} is declared on two nodes; rename one with `names`")
                    layers = target.layers if target.layers is not None else tuple(range(spec.num_layers))
                    into[name] = dataclasses.replace(target, node=node, layers=layers)
            parts[node] = spec.module_map
        module_map = _JoinedLoraMap(parts)
        return {"num_layers": max(s.num_layers for s in specs), **joined, "module_map": module_map,
                "ignored_modules": module_map.ignores}

    return LoraSpec(
        resource_key=specs[0].resource_key, nodes=set().union(*(spec.nodes for spec in specs)),
        convert_keys=convert_keys, declare=declare,
    )


# -- binding ------------------------------------------------------------------------------


def _module_path(path: str, ref: LoraModuleRef) -> str:
    """The module a binding adapts: the binding's path, less its parameter if it has one."""
    return path if ref.param is None else path.removesuffix(f".{ref.param}")


def _locate(submodule: nn.Module, module_map: ModuleLoraMap, key: str) -> dict[str, nn.Module]:
    """The submodule's module at each binding path, under the weights root's
    ``root_path`` (aliased paths included: a submodule may register a root module under
    a short name of its own before the root itself)."""
    modules = dict(submodule.named_modules(remove_duplicate=False))
    prefix = f"{module_map.root_path}." if module_map.root_path else ""
    paths = {b: prefix + _module_path(b, ref) for b, ref in module_map.bindings.items()}
    missing = [b for b, path in paths.items() if path not in modules]
    if missing:
        raise ValueError(
            f"{type(submodule).__name__} has no {missing[:5]} under {module_map.root_path!r}, "
            f"which LoRA resource {key!r} declares ({len(missing)} of "
            f"{len(module_map.bindings)} missing); was its spec built from this node's "
            "weights root, and is root_path its attribute path?"
        )
    return {b: modules[path] for b, path in paths.items()}


def _check(
    module: LoraCapable, param: str | None, declared: LoraTarget, shard: LoraTarget, path: str, key: str,
) -> None:
    """The bound module is the one declared, and the pool's shard of the target is the
    module's own rank-local shape (catches a target split across ranks while the module
    is whole, or the reverse)."""
    own = module.lora_targets()[param]
    if own.expert_split is not None:
        first, count = own.expert_split
        raise ValueError(
            f"LoRA resource {key!r}: {path} holds experts {first}..{first + count - 1} of "
            f"{own.experts} on this rank (expert parallelism, `ep_config: parallel: ep`), and "
            "LoRA on experts is served under tensor parallelism only; serve this node with "
            f"`parallel: tp`, or leave its experts targets out of `resources: {LORA_RESOURCE_KEY}: targets:`"
        )
    plain = dataclasses.replace(declared, layers=None, node=None, per_segment=False, token_fold=1)
    if own != plain:
        raise ValueError(f"LoRA resource {key!r}: {path} is {own}, but the spec declares {plain}")
    if param is None:
        local = module.lora_local_shape()
    else:
        # a stacked parameter [experts, out, in] is its target's rank-local shape
        experts, out, in_features = getattr(module, param).shape
        if experts != shard.experts:
            raise ValueError(
                f"LoRA resource {key!r}: {path} holds {experts} experts on this rank, but the "
                f"pool's target has {shard.experts}"
            )
        local = (in_features, out)
    if local != (shard.in_features, shard.out_features):
        raise ValueError(
            f"LoRA resource {key!r}: {path} computes {local[0]} -> {local[1]} on this rank, but "
            f"the pool shards its target to {shard.in_features} -> {shard.out_features}; a "
            "module not split across the node's tensor-parallel group needs a target that is "
            "whole on every rank"
        )


def bind_module_lora(submodule: nn.Module, resources: Mapping[str, Any]) -> None:
    """Bind every module a ``module_lora_spec`` placed on this node to its delta, for the
    node's LoRA pool among ``resources``; nothing is touched when the node serves no
    adapters. The pool's step is the node's own segments (``Resource.steps_from_segments``),
    which address each row to its adapter."""
    from mstar.engine.resources.lora.pool import LoraAdapterPool

    for key, pool in resources.items():
        if not isinstance(pool, LoraAdapterPool):
            continue
        spec = pool.spec
        if not isinstance(spec.module_map, LoraMap):
            raise ValueError(
                f"LoRA resource {key!r} was not declared with `module_lora_spec`, so nothing "
                "binds the node's modules to it; build the node's spec with `module_lora_spec`"
            )
        # a node's pool holds its own targets (`LoraSpec.for_node`)
        (node,) = spec.nodes
        module_map = spec.module_map.part(node)
        modules = _locate(submodule, module_map, key)
        for path, ref in module_map.bindings.items():
            if ref.target not in spec.targets:
                continue  # a target not served (`LoraSpec.dropped_targets`)
            module = modules[path]
            held = (
                getattr(module, "lora_params", {}).get(ref.param) if ref.param is not None
                else getattr(module, "lora_delta", None) or getattr(module, "lora_deferred", None)
            )
            if held is not None:
                raise ValueError(f"{path} already holds a LoRA delta from another resource of this node")
            original = type(module)
            module = make_lora_capable(module)
            try:
                _check(module, ref.param, spec.targets[ref.target], pool.targets[ref.target], path, key)
            except Exception:
                module.__class__ = original
                raise
            delta = pool.layer_delta(ref.target, ref.layer)
            if ref.param is not None:
                module.lora_params = {**module.lora_params, ref.param: delta}
            elif path in module_map.deferred:
                module.lora_deferred = delta
            else:
                module.lora_delta = delta
