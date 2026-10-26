"""The serving config's LoRA adapters, resolved once before the conductor starts.

The API server reads every adapter its ``resources: lora:`` block names, refusing the
deployment if any cannot be served, and writes what it found back into the block: each
adapter's digest, which every later read must match, and the targets the adapters touch.
The conductor is handed that config, and spawns the workers with it, so every process and
rank builds the same pools and serves the weights the API server keyed the prefix cache
under.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from mstar.engine.resources.lora.config import LORA_RESOURCE_KEY, LoraSpec

logger = logging.getLogger(__name__)


def lora_spec(specs: Iterable[Any]) -> LoraSpec | None:
    """The deployment's LoRA spec among its declared resource specs
    (``declared_resources``: whole, not yet split into a pool per node)."""
    return next((s for s in specs if isinstance(s, LoraSpec)), None)


def served_lora_targets(spec: LoraSpec) -> list[str] | None:
    """The targets a deployment keeps when its adapters are fixed at startup (runtime
    updates off, no YAML ``targets``): those one of its adapters touches, as the
    ``targets`` list every process then applies.
    Every other target would hold zero weights in every slot, yet cost its kernels on
    every step and its memory (the LM head's and the embedding's are vocabulary-sized),
    and a node left with none gets no pool. None when nothing narrows."""
    from mstar.engine.resources.lora.store import adapter_targets

    if spec.runtime_updates or spec.dropped_targets or not spec.adapters:
        return None
    used = set().union(*(adapter_targets(spec, path) for path in spec.adapters.values()))
    return None if not used or used == set(spec.targets) else sorted(used)


def resolve_lora_adapters(spec: LoraSpec | None, model_config: dict) -> dict[str, bytes]:
    """Check every adapter ``spec`` serves (``check_adapters``: a bad one fails the
    deployment, every bad one named), write their digests and the targets they touch
    into ``model_config``'s ``resources: lora:`` block and onto ``spec``, which then is
    what every worker builds, and return the digests."""
    if spec is None or not spec.adapters:
        return {}
    from mstar.engine.resources.lora.store import check_adapters

    digests = check_adapters(spec)
    resolved: dict[str, Any] = {"adapter_digests": {name: d.hex() for name, d in digests.items()}}
    targets = served_lora_targets(spec)
    if targets is not None:
        logger.info("LoRA: serving only the targets the adapters touch: %s", targets)
        resolved["targets"] = targets
    model_config["resources"][LORA_RESOURCE_KEY].update(resolved)
    spec.apply_yaml_overrides(**resolved)
    return digests
