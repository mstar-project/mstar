"""Per-request LoRA declared from a model's own modules.

``module_lora_spec`` walks a node's weights root (the module its weight loader fills;
built on the meta device at one rank, so any process can declare it), finds every
LoRA-capable module (``capable.py``), groups them into targets by their role in a layer
stack (``layers.N.<role>`` / ``blocks.N.<role>``; a module outside the stacks is a target
of its own), and places checkpoint modules with the root's own ``CheckpointRules`` (the
ones its weight loader uses). The model states only what its modules cannot: how a
target's rows relate to the step's tokens (``rows``), modules it calls outside the step
(``exclude``, ``deferred``), names kept for compatibility (``names``), the prefixes
adapters write the root under (``adapter_roots``) and adapter-only renames (``aliases``).

``bind_module_lora`` binds each placed module to its delta when the node's resources
include the spec's pool (``NodeSubmodule.bind_node_resources`` calls it), checking the
real, possibly sharded, module against the declaration first. A deployment without a
``lora:`` block declares no spec, so nothing is bound and every module runs as before.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.lora.config import LoraModuleRef, LoraSpec, LoraTarget
from mstar.model.components.lora.capable import (
    LoraCapable,
    lora_capability,
    make_lora_capable,
)
from mstar.model.loader.base import CheckpointRules, StackedParamRule, apply_stacked

if TYPE_CHECKING:
    from mstar.engine.resources.lora.pool import LoraAdapterPool

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


class ModuleLoraMap:
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
        # the other nodes' maps of the same model (``link``)
        self._siblings: tuple[ModuleLoraMap, ...] = ()

    def link(self, siblings: Sequence[ModuleLoraMap]) -> None:
        """Leave a module another node's map of the same model claims more specifically
        to that map (``link_lora_specs``)."""
        self._siblings = tuple(m for m in siblings if m is not self)

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
        """``LoraSpec.ignored_modules``: a module the model never runs (``ignore``), or one
        another node's spec claims more specifically, which that spec loads or refuses."""
        if any(fnmatch.fnmatchcase(module, p) for p in self._ignore):
            return True
        if not self._siblings:
            return False
        own = self.claim(module)
        return any(s.claim(module) > own for s in self._siblings)


def _match(path: str, patterns) -> str | None:
    """The first pattern (fnmatch, children included) ``path`` falls under."""
    for p in patterns:
        if fnmatch.fnmatchcase(path, p) or fnmatch.fnmatchcase(path, p + ".*"):
            return p
    return None


class _Found(NamedTuple):
    path: str
    layer: int
    target: LoraTarget


_EXCLUDED = "excluded by the model's LoRA declaration"


def meta_root(build: Callable[[CommGroup], nn.Module]) -> nn.Module:
    """A node's weights root as ``module_lora_spec`` reads it: built on the meta device
    (no memory), as one rank sees it (``build`` gets the trivial group; the pool shards
    the targets for its own rank)."""
    with torch.device("meta"):
        return build(CommGroup.trivial())


def module_lora_spec(
    root: nn.Module,
    *,
    resource_key: str,
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
    """A node's LoRA spec, a target per role of the LoRA-capable modules under ``root``.

    ``root`` is the module the node's weight loader fills, at one rank (built on the meta
    device is enough), and ``root_path`` its attribute path in the node's submodule
    (``""`` for the submodule itself), where binding finds it. ``rules`` are the loader's
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

    Linear-like modules that cannot take a delta (a depthwise conv, MoE experts) are
    recorded with the reason, which an adapter touching them is refused with, and so are
    the children a module declares in ``lora_unservable`` (child name -> reason): ones it
    runs through their weights itself, never through their forward (folded at load, an
    ``F.linear`` over the weight).
    """
    own_rules = getattr(root, "checkpoint_rules", None)
    if rules is None:
        rules = own_rules() if own_rules is not None else CheckpointRules()
    elif own_rules is not None:
        raise ValueError(
            f"{type(root).__name__} states its checkpoint rules; LoRA placement takes them "
            "from it rather than from a copy"
        )
    excluded = dict(exclude) if isinstance(exclude, Mapping) else dict.fromkeys(exclude, _EXCLUDED)
    rows, names = dict(rows or {}), dict(names or {})
    groups: dict[tuple[str | None, str], list[_Found]] = defaultdict(list)
    unservable: dict[str, str] = {}
    for path, module in root.named_modules():
        # children the module runs through their weights itself (pre-order: before them)
        prefix = f"{path}." if path else ""
        excluded.update({prefix + c: r for c, r in getattr(module, "lora_unservable", {}).items()})
        if not path:
            continue
        capability = lora_capability(module)
        if capability is None:
            continue
        pattern = _match(path, excluded)
        if pattern is not None:
            unservable[path] = excluded[pattern]
            continue
        if capability.capable is None:
            unservable[path] = capability.reason
            continue
        target = capability.capable.lora_target(module)
        if target.kind == "embedding" and not embeddings:
            unservable[path] = "a token embedding, which this node does not serve LoRA on"
            continue
        m = _STACK.match(path)
        if m is None:
            groups[(None, path)].append(_Found(path, 0, target))
        else:
            groups[(m["stack"], m["role"])].append(_Found(path, int(m["layer"]), target))
    if not groups:
        raise ValueError(f"{type(root).__name__} has no LoRA-capable modules")

    stacked = [f.layer for (stack, _), found in groups.items() if stack is not None for f in found]
    num_layers = max(stacked, default=0) + 1
    target_names = _target_names(list(groups), names)
    unused = set(names) - {role for _, role in groups}
    if unused:
        raise ValueError(f"`names` names roles {sorted(unused)} that {type(root).__name__} does not have")

    targets: dict[str, LoraTarget] = {}
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
        targets[name] = dataclasses.replace(first, layers=layers, **_row_fields(rows, role))
        for f in found:
            bindings[f.path] = LoraModuleRef(name, f.layer)
    unmatched = [p for p in rows if not any(fnmatch.fnmatchcase(r, p) for _, r in groups)]
    if unmatched:
        raise ValueError(f"`rows` patterns {unmatched} match no LoRA target role")
    deferred_paths = frozenset(p for p in bindings if _match(p, deferred) is not None)
    unmatched = [p for p in deferred if not any(_match(b, [p]) for b in bindings)]
    if unmatched:
        raise ValueError(f"`deferred` patterns {unmatched} match no LoRA target module")
    if unservable:
        logger.info(
            "LoRA %s: %d module(s) take no delta, e.g. %s",
            resource_key, len(unservable), [f"{p} ({r})" for p, r in list(unservable.items())[:3]],
        )

    module_map = ModuleLoraMap(
        bindings, rules=rules, adapter_stacked_params=adapter_stacked_params,
        adapter_roots=adapter_roots, aliases=aliases, unservable=unservable, ignore=ignore,
        root_path=root_path, deferred=deferred_paths,
    )
    return LoraSpec(
        resource_key=resource_key, nodes=set(nodes), num_layers=num_layers, targets=targets,
        module_map=module_map, ignored_modules=module_map.ignores, convert_keys=convert_keys,
    )


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


def link_lora_specs(specs: Sequence[LoraSpec]) -> list[LoraSpec]:
    """A model's specs for several nodes (an LLM and its vision tower), told about each
    other: each leaves the modules another claims more specifically (``AdapterRoot``) to
    it, so one adapter file serves both nodes and a module either places is not refused
    by the other."""
    maps = [s.module_map for s in specs if isinstance(s.module_map, ModuleLoraMap)]
    for m in maps:
        m.link(maps)
    return list(specs)


# -- binding ------------------------------------------------------------------------------


def _locate(submodule: nn.Module, module_map: ModuleLoraMap, key: str) -> dict[str, nn.Module]:
    """The submodule's module at each binding path, under the weights root's
    ``root_path`` (aliased paths included: a submodule may register a root module under
    a short name of its own before the root itself)."""
    modules = dict(submodule.named_modules(remove_duplicate=False))
    prefix = f"{module_map.root_path}." if module_map.root_path else ""
    missing = [b for b in module_map.bindings if prefix + b not in modules]
    if missing:
        raise ValueError(
            f"{type(submodule).__name__} has no {missing[:5]} under {module_map.root_path!r}, "
            f"which LoRA resource {key!r} declares ({len(missing)} of "
            f"{len(module_map.bindings)} missing); was its spec built from this node's "
            "weights root, and is root_path its attribute path?"
        )
    return {b: modules[prefix + b] for b in module_map.bindings}


def _check(module: LoraCapable, declared: LoraTarget, shard: LoraTarget, path: str, key: str) -> None:
    """The bound module is the one declared, and the pool's shard of the target is the
    module's own rank-local shape (catches a target split across ranks while the module
    is whole, or the reverse)."""
    own = module.lora_target()
    plain = dataclasses.replace(declared, layers=None, per_segment=False, token_fold=1)
    if own != plain:
        raise ValueError(f"LoRA resource {key!r}: {path} is {own}, but the spec declares {plain}")
    local = module.lora_local_shape()
    if local != (shard.in_features, shard.out_features):
        raise ValueError(
            f"LoRA resource {key!r}: {path} computes {local[0]} -> {local[1]} on this rank, but "
            f"the pool shards its target to {shard.in_features} -> {shard.out_features}; a "
            "module not split across the node's tensor-parallel group needs a target that is "
            "whole on every rank"
        )


def bind_module_lora(submodule: nn.Module, resources: Mapping[str, Any]) -> tuple[str, ...]:
    """Bind every module a ``module_lora_spec`` placed to its delta, for each LoRA pool
    among ``resources``; returns those pools' keys (empty, and nothing touched, when the
    node serves no adapters).

    The node must have said how its rows meet the pool's segments (``lora_segment_rows``), since
    a step's ``LoraStep`` is what addresses each row to its adapter."""
    from mstar.engine.resources.lora.pool import LoraAdapterPool

    pools: list[tuple[str, LoraAdapterPool]] = [
        (key, r) for key, r in resources.items() if isinstance(r, LoraAdapterPool)
    ]
    for key, pool in pools:
        if not isinstance(pool.spec.module_map, ModuleLoraMap):
            raise ValueError(
                f"LoRA resource {key!r} was not declared with `module_lora_spec`, so nothing "
                "binds the node's modules to it; build the node's spec with `module_lora_spec`"
            )
    if not pools:
        return ()
    if not getattr(submodule, "lora_segment_rows", False):
        raise ValueError(
            f"{type(submodule).__name__} serves LoRA adapters but does not say how its rows "
            "meet them: set `lora_segment_rows = True` when `declare_step`'s segments are, in "
            "order, the rows its linears see (each a request's span of tokens), and the "
            "engine adds the LoraStep"
        )
    for key, pool in pools:
        spec = pool.spec
        module_map: ModuleLoraMap = spec.module_map
        modules = _locate(submodule, module_map, key)
        for path, ref in module_map.bindings.items():
            if ref.target not in spec.targets:
                continue  # the YAML's `targets` dropped it
            module = modules[path]
            if getattr(module, "lora_delta", None) is not None or getattr(module, "lora_deferred", None) is not None:
                raise ValueError(f"{path} already holds a LoRA delta from another resource of this node")
            original = type(module)
            module = make_lora_capable(module)
            try:
                _check(module, spec.targets[ref.target], pool.targets[ref.target], path, key)
            except Exception:
                module.__class__ = original
                raise
            delta = pool.layer_delta(ref.target, ref.layer)
            if path in module_map.deferred:
                module.lora_deferred = delta
            else:
                module.lora_delta = delta
    return tuple(key for key, _ in pools)
