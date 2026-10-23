"""Which LoRA specs a deployment builds from its serving config's ``resources: lora:``
block: the model's declared specs, sized by the block, narrowed to the targets its
adapters touch, and those left with no targets dropped. ``mstar.model.base.serving_resources``
adds them to the model's other specs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from mstar.engine.resources.lora.config import LORA_RESOURCE_KEY, LoraSpec, lora_enabled

if TYPE_CHECKING:
    from mstar.engine.resources.spec import NodeResourceSpec
    from mstar.model.base import Model


def declared_lora_specs(model: Model, model_config: Mapping[str, Any]) -> list[LoraSpec]:
    """The model's LoRA specs, the one keyed ``lora`` first, when the YAML has a ``lora:``
    block; none otherwise. The block sizes every spec (one per node that serves adapters);
    a block under another LoRA spec's own key overrides it there (applied with the other
    ``resources:`` overrides)."""
    block = (model_config.get("resources") or {}).get(LORA_RESOURCE_KEY)
    if block is None:
        return []
    lora = list(model.get_lora_specs())
    if not lora:
        raise ValueError(
            f"the serving config has a `resources: {LORA_RESOURCE_KEY}:` block, but "
            f"{type(model).__name__} serves no LoRA adapters"
        )
    if lora[0].resource_key != LORA_RESOURCE_KEY:
        raise ValueError(
            f"{type(model).__name__}'s first LoRA spec declares {lora[0].resource_key!r}; "
            f"it is keyed {LORA_RESOURCE_KEY!r}"
        )
    declared = {name for spec in lora for name in spec.targets}
    unknown = sorted(set(block.get("targets") or ()) - declared)
    if unknown:
        raise ValueError(
            f"`resources: {LORA_RESOURCE_KEY}: targets:` names {unknown}, which "
            f"{type(model).__name__} does not declare; it declares {sorted(declared)}"
        )
    for spec in lora[1:]:
        spec.apply_yaml_overrides(**block)
    return lora


def drop_lora_specs_without_targets(specs: list[NodeResourceSpec]) -> list[NodeResourceSpec]:
    """Leave out a node's LoRA spec whose targets the YAML's ``targets`` list dropped
    entirely: its pool would hold slots for adapters it can add nothing to. The modules
    it owned are then no one's to load, so the remaining specs refuse an adapter touching
    them, as a dropped target is refused."""
    lora = [s for s in specs if isinstance(s, LoraSpec)]
    dropped = [s for s in lora if not s.targets]
    if not dropped:
        return specs
    if dropped[0] is lora[0]:
        raise ValueError(
            f"`resources: {lora[0].resource_key}: targets:` leaves the {sorted(lora[0].nodes)} "
            f"spec no targets; keep at least one of {sorted(lora[0].dropped_targets)}"
        )

    def owned_by_dropped(module: str) -> bool:
        return any(d.ignored_modules is None or not d.ignored_modules(module) for d in dropped)

    for spec in lora:
        if spec.targets:
            spec.stop_ignoring(owned_by_dropped)
    return [s for s in specs if not (isinstance(s, LoraSpec) and not s.targets)]


def served_lora_targets(lora: list[LoraSpec]) -> list[str] | None:
    """The targets a deployment keeps when its YAML lists no ``targets``: those one of
    its configured adapters touches, as the
    ``targets`` list every process then applies. Every other target would hold zero
    weights in every slot, yet cost its kernels on every step and its memory (the LM
    head's and the embedding's are vocabulary-sized). The first spec (the LLM's) keeps
    all of its targets if no adapter touches it; a later spec left with none is not
    built. None when nothing narrows. The conductor computes it once, before any worker
    starts, so every process builds the same specs; an adapter that cannot be read
    fails the deployment here rather than leaving one process with every target.

    One list for every spec, as the YAML's: specs whose target names coincide keep a
    name either one uses."""
    from mstar.engine.resources.lora.store import adapter_targets

    if not lora or lora[0].dropped_targets or not lora[0].adapters:
        return None
    kept: set[str] = set()
    for i, spec in enumerate(lora):
        used = set().union(*(adapter_targets(spec, path) for path in spec.adapters.values()))
        kept |= used if used or i > 0 else set(spec.targets)
    declared = {name for spec in lora for name in spec.targets}
    return None if kept == declared else sorted(kept)


# (id(model), resources block) -> (model, its LoRA specs); holding the model keeps its id
# from being reused while the entry lives (a deployment's model lives as long as its process)
_LORA_SPECS: dict[tuple[int, str], tuple[Model, list[LoraSpec]]] = {}


def lora_specs(model: Model | None, model_config: Mapping[str, Any] | None) -> list[LoraSpec]:
    """The deployment's LoRA specs, YAML applied, the one keyed ``lora`` first; none when
    its config turns LoRA on nowhere. Built once per model and ``resources:`` block in a
    process (building one reads the model's module tree), so its callers there share the
    specs, and with them a runtime load or unload recorded on one."""
    if model is None or not model_config:
        return []
    if not lora_enabled(model_config):
        return []
    from mstar.model.base import serving_resources

    key = (id(model), json.dumps(model_config["resources"], sort_keys=True, default=str))
    if key not in _LORA_SPECS:
        specs = [s for s in serving_resources(model, model_config) if isinstance(s, LoraSpec)]
        _LORA_SPECS[key] = (model, specs)
    return _LORA_SPECS[key][1]
