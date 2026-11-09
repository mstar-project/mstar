"""The engine's side: quantization around a node's ``get_submodule``.

The engine manager opens a ``node_scope`` while the model builds and loads a node; ``load_weights_into``
(every model's loader goes through it) asks ``active_scope()`` whether the module it fills is a declared
weights root, and if so the scope resolves that root's plan, prepares it and intercepts the stream. When
``get_submodule`` returns, ``finish`` quantizes what loaded bf16. A model only declares
(``Model.get_quantization_roots``); a node with no declared root and no ``quantization:`` block costs nothing.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextvars import ContextVar
from typing import Any

import torch
from torch import nn

from mstar.model.loader.base import StackedParamRule
from mstar.quantization import apply
from mstar.quantization.checkpoint import CheckpointQuant, scan_checkpoint
from mstar.quantization.formats import AUTO, BF16
from mstar.quantization.resolve import QuantizationRoot, QuantPlan, resolve_plan
from mstar.quantization.spec import NodeQuantSpec, parse_quantization

logger = logging.getLogger(__name__)

_ACTIVE: ContextVar[NodeQuantization | None] = ContextVar("mstar_quantization_scope", default=None)


def device_capability(device: torch.device | str | None) -> tuple[int, int] | None:
    from mstar.platforms import current_platform

    if device is None or torch.device(device).type != "cuda":
        return None
    return current_platform.cuda_capability(torch.device(device))


class NodeQuantization:
    """One node's quantization while it loads: its spec, its declared roots, and each root's plan."""

    def __init__(self, node: str, spec: NodeQuantSpec, roots: list[QuantizationRoot],
                 device: torch.device | str | None = None, checkpoint: CheckpointQuant | None = None):
        """``checkpoint``: what the stream stores, when it comes from no directory (an in-memory stream)."""
        self.node = node
        self.spec = spec
        self.roots = roots
        self.device = device
        self._given = checkpoint
        self._checkpoints: dict[str, CheckpointQuant | None] = {}
        # id(root module) -> (module, plan), in load order
        self.plans: dict[int, tuple[nn.Module, QuantPlan]] = {}
        self._finalized: set[int] = set()

    @contextlib.contextmanager
    def activated(self) -> Iterator[NodeQuantization]:
        """This scope as the active one for the body (the engine manager's ``node_scope`` does it per node)."""
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    def _checkpoint(self, decl: QuantizationRoot) -> CheckpointQuant | None:
        if self._given is not None:
            return self._given
        directory = decl.checkpoint_dir()
        if directory is None:
            return None
        if directory not in self._checkpoints:
            self._checkpoints[directory] = scan_checkpoint(directory)
        return self._checkpoints[directory]

    def plan_for(self, module: nn.Module) -> QuantPlan | None:
        """The plan for ``module`` if it is one of this node's declared roots (resolved, checked and prepared
        on first sight), else None."""
        key = id(module)
        if key in self.plans:
            return self.plans[key][1]
        decl = next((r for r in self.roots if type(module) is r.root_type), None)
        if decl is None:
            return None
        plan = resolve_plan(module, decl, self.spec, self._checkpoint(decl), device_capability(self.device))
        if plan.problems:
            raise ValueError(f"quantization of node {self.node!r} refused:\n  - " + "\n  - ".join(plan.problems))
        for note in plan.notes:
            logger.info("%s: %s", self.node, note)
        apply.prepare(module, plan)
        self.plans[key] = (module, plan)
        return plan

    def prepare_meta(self, module: nn.Module) -> list[nn.Module]:
        """Resolve and prepare the declared roots at or under ``module`` while it is on the meta device; returns
        them, for ``mark_unloaded`` once ``to_empty`` has given them storage (``materialize`` does both)."""
        roots = [m for m in module.modules() if any(type(m) is r.root_type for r in self.roots)]
        for root in roots:
            self.plan_for(root)
        return roots

    def mark_unloaded(self, roots: Iterable[nn.Module]) -> None:
        for root in roots:
            apply.mark_unloaded(root, self.plans[id(root)][1])

    def wrap_load(
        self, module: nn.Module, weights: Iterable[tuple[str, torch.Tensor]],
        stacked_params: list[StackedParamRule] | None, name_remapper: Callable[[str], str | None] | None,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        plan = self.plan_for(module)
        if plan is None or not plan.active:
            return weights
        decl = next(r for r in self.roots if type(module) is r.root_type)
        return apply.intercept(weights, plan, stacked_params, name_remapper, ckpt=self._checkpoint(decl),
                               prefix=decl.rules.prefix)

    def finalize_within(self, module: nn.Module) -> None:
        """Finalize the loaded roots at or under ``module`` (once each): quantize what loaded bf16 before the
        model's own post-load processing (``process_weights_after_loading``) reads the weights."""
        inside = {id(m) for m in module.modules()}
        for key, (root, plan) in self.plans.items():
            if key in inside and key not in self._finalized:
                apply.finalize(root, plan, self.device)
                self._finalized.add(key)

    def finish(self, submodule: nn.Module | None) -> None:
        """Finalize every root this node loaded; refuse a spec that asked for a format no root took."""
        for key, (module, plan) in self.plans.items():
            if key not in self._finalized:
                apply.finalize(module, plan, self.device)
                self._finalized.add(key)
        asked = self.spec.format not in (AUTO, BF16) or any(rule.format != BF16 for rule in self.spec.rules)
        if submodule is not None and asked and not self.plans:
            raise ValueError(
                f"quantization asks {self.spec.format} for node {self.node!r}, but its weights never went through "
                f"a declared root ({[r.root_type.__name__ for r in self.roots]})"
            )


def active_scope() -> NodeQuantization | None:
    return _ACTIVE.get()


@contextlib.contextmanager
def node_scope(
    model: Any, node: str, model_config: Mapping[str, Any], device: torch.device | str | None = None,
) -> Iterator[NodeQuantization | None]:
    """Quantization for ``node`` while the body builds and loads it; yields None when there is nothing to do."""
    spec = parse_quantization(model_config.get("quantization")).for_node(node)
    getter = getattr(model, "get_quantization_roots", None)
    roots = [r for r in (getter() if getter is not None else []) if r.node == node]
    if not roots:
        # validate_deployment refused a block naming this node or asking a format of a model with no roots
        yield None
        return
    scope = NodeQuantization(node, spec, roots, device)
    token = _ACTIVE.set(scope)
    try:
        yield scope
    finally:
        _ACTIVE.reset(token)


def materialize(module: nn.Module, device: torch.device | str) -> nn.Module:
    """``module.to_empty(device=device)`` for a module built on the meta device, with the active scope's declared
    roots under it prepared first: each layer that loads as stored allocates its stored format (MXFP4 experts
    their packed bytes), never the bf16 weight it replaces, and its scales are marked not loaded after."""
    scope = active_scope()
    roots = scope.prepare_meta(module) if scope is not None else []
    module.to_empty(device=device)
    if scope is not None:
        scope.mark_unloaded(roots)
    return module


def load_submodule(model: Any, node: str, model_config: Mapping[str, Any] | None = None,
                   device: torch.device | str = "cpu", **kwargs: Any):
    """``model.get_submodule(node, device, **kwargs)`` under the node's quantization scope: how the engine
    manager loads a node, for anything else that loads one (tests, tools)."""
    with node_scope(model, node, model_config or {}, device) as scope:
        submodule = model.get_submodule(node, device, **kwargs)
        if scope is not None:
            scope.finish(submodule)
    return submodule
