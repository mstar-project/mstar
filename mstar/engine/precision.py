"""Each node's precision policy, declared by the model and fixed at load.

``NodePrecision`` says how a node's numbers are stored and computed: the dtype
its params are cast to, which of its modules stay fp32, and whether its forward
runs under autocast. The engine applies it at load and in every forward,
warm-up and capture; there is no model-wide dtype.
"""
from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from mstar.engine.resources.spec import parse_yaml_dtype, require_unscaled_dtype


@dataclass(frozen=True)
class NodePrecision:
    # None keeps the dtype the node was built in
    param_dtype: torch.dtype | None
    # fnmatch globs on the node's module, param or buffer names; a matched
    # module stays fp32 with everything under it
    fp32_modules: tuple[str, ...] = ()
    # wrap the forward in torch.amp.autocast(param_dtype); off, every op runs in
    # its inputs' dtypes and any fp32 the model needs is explicit
    autocast: bool = False

    def __post_init__(self):
        if isinstance(self.fp32_modules, str):
            raise TypeError(f"fp32_modules is a tuple of globs, not the string {self.fp32_modules!r}")
        object.__setattr__(self, "fp32_modules", tuple(self.fp32_modules))
        if self.autocast and self.param_dtype is None:
            raise ValueError("autocast needs a param_dtype to autocast in")
        if self.param_dtype is not None:
            require_unscaled_dtype(self.param_dtype, "param_dtype")

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        """What the forward's autocast scope runs in; None runs it without one."""
        return self.param_dtype if self.autocast else None


def fp32_names(module: torch.nn.Module, globs: tuple[str, ...]) -> set[str]:
    """Names of the modules, params and buffers ``globs`` match. A glob that
    matches nothing is an error: it would otherwise be a silently ineffective
    setting."""
    names = [name for name, _ in module.named_modules() if name]
    names += [name for name, _ in module.named_parameters()]
    names += [name for name, _ in module.named_buffers()]
    matched: set[str] = set()
    for glob in globs:
        hits = fnmatch.filter(names, glob)
        if not hits:
            raise ValueError(
                f"fp32_modules glob {glob!r} matches no module, param or buffer of "
                f"{type(module).__name__}"
            )
        matched.update(hits)
    return matched


def _target_dtypes(module: torch.nn.Module, precision: NodePrecision):
    """Each floating param and buffer with its owner, its name in the owner,
    its full name, and the dtype ``precision`` gives it (None: keep it).

    A tensor already in a quantized dtype (fp8, fp4: below 16 bits) is left out:
    a quant pass or a pre-quantized checkpoint put it there with its scales, and
    a cast would undo it."""
    islands = fp32_names(module, precision.fp32_modules)

    def pinned(name: str) -> bool:
        return any(name == top or name.startswith(top + ".") for top in islands)

    for prefix, child in module.named_modules():
        for kind in (child._parameters, child._buffers):
            for local, tensor in kind.items():
                if tensor is None or not tensor.is_floating_point() or tensor.dtype.itemsize < 2:
                    continue
                full = f"{prefix}.{local}" if prefix else local
                yield child, kind, local, full, torch.float32 if pinned(full) else precision.param_dtype


def apply_precision(module: torch.nn.Module, precision: NodePrecision) -> torch.nn.Module:
    """Cast ``module``'s floating params and buffers to ``precision``, in place.

    Call it on the meta module before ``to_empty`` so storage is allocated in its
    final dtype; on a loaded module it casts, and is a no-op once applied. A
    param keeps its identity (``.data`` is swapped), so attributes a loader hung
    on it survive, and a module's own ``_apply`` is not consulted.
    """
    for child, kind, local, _, dtype in list(_target_dtypes(module, precision)):
        tensor = kind[local]
        if dtype is None or tensor.dtype == dtype:
            continue
        if kind is child._parameters:
            tensor.data = tensor.data.to(dtype)
        else:
            kind[local] = tensor.to(dtype)
    return module


def precision_mismatches(module: torch.nn.Module, precision: NodePrecision) -> list[str]:
    """Floating params of ``module`` not in the dtype ``precision`` gives them.
    After every node is cast, a mismatch is a module two nodes share under
    policies that disagree: the later cast won."""
    return [
        f"{full} ({kind[local].dtype}, policy {dtype})"
        for child, kind, local, full, dtype in _target_dtypes(module, precision)
        if kind is child._parameters and dtype is not None and kind[local].dtype != dtype
    ]


def refuse_legacy_autocast_yaml(model_config: Mapping[str, Any]) -> None:
    """The model-wide ``autocast_dtype`` key is gone; say what replaced it."""
    if "autocast_dtype" in model_config:
        raise ValueError(
            "top-level `autocast_dtype:` in the serving config is no longer read; "
            "set a node's precision instead, e.g.\n"
            "nodes:\n  LLM:\n    precision:\n      param_dtype: float32"
        )


_YAML_KEYS = {"param_dtype", "fp32_modules", "autocast"}


def apply_yaml_precision(
    precision: NodePrecision, node_name: str, block: Mapping[str, Any],
) -> NodePrecision:
    """A deployment's ``nodes.<name>.precision`` block over the model's policy.

    YAML may tighten but not loosen: it may turn autocast off but never on, add
    fp32 modules but not drop the model's, and change the param dtype only of a
    node that has one (None pins the dtype the node was built in).
    """
    unknown = sorted(block.keys() - _YAML_KEYS)
    if unknown:
        raise ValueError(
            f"nodes.{node_name}.precision: unknown key(s) {unknown}; accepted: {sorted(_YAML_KEYS)}"
        )
    updates: dict[str, Any] = {}
    if block.get("autocast") is not None:
        if block["autocast"] and not precision.autocast:
            raise ValueError(
                f"node {node_name!r} runs without autocast, so nodes.{node_name}.precision "
                "cannot turn it on"
            )
        updates["autocast"] = bool(block["autocast"])
    if block.get("param_dtype") is not None:
        if precision.param_dtype is None:
            raise ValueError(
                f"node {node_name!r} keeps the dtype it was built in (param_dtype None), "
                f"so nodes.{node_name}.precision cannot set param_dtype"
            )
        updates["param_dtype"] = parse_yaml_dtype(block["param_dtype"])
    if block.get("fp32_modules"):
        extra = tuple(g for g in block["fp32_modules"] if g not in precision.fp32_modules)
        updates["fp32_modules"] = precision.fp32_modules + extra
    return replace(precision, **updates)


def yaml_precision_blocks(
    model_config: Mapping[str, Any], node_names: set[str],
) -> dict[str, Mapping[str, Any]]:
    """``nodes.<name>.precision`` per node, checking names and keys against
    ``node_names``, every node the deployment places."""
    nodes = model_config.get("nodes") or {}
    unknown = sorted(nodes.keys() - node_names)
    if unknown:
        raise ValueError(f"nodes: unknown node(s) {unknown}; the deployment places {sorted(node_names)}")
    blocks: dict[str, Mapping[str, Any]] = {}
    for name, node_block in nodes.items():
        extra = sorted((node_block or {}).keys() - {"precision"})
        if extra:
            raise ValueError(f"nodes.{name}: unknown key(s) {extra}; accepted: ['precision']")
        if (node_block or {}).get("precision"):
            blocks[name] = node_block["precision"]
    return blocks
