"""Refusing quantized checkpoints that a model can't load.

A quantized checkpoint names its format in its config and stores each weight as parts: fp8 or
packed integer values, plus scales and sometimes zero points. A loader that knows only unquantized
weights reads such a checkpoint without an error. fp8 values copy into bf16 parameters unscaled,
and the scales and packed weights match no parameter, so the model serves garbage. The loader
refuses instead, unless the caller names the checkpoint's ``quant_method`` as one it loads.
"""
from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple


class UnservedQuantizationError(ValueError):
    """A quantized checkpoint, or part of a quantized weight, that the model has no way to load."""


class CheckpointQuantization(NamedTuple):
    method: str  # the config's quant_method
    source: str  # the file that declares it


# A quantized weight's parts besides its values (fp8 block and per-tensor scales, ModelOpt and
# compressed-tensors NVFP4, compressed-tensors packed int4, AWQ/GPTQ). An unquantized model has no
# parameter for any of them, so one that loads into nothing means the checkpoint is quantized, even
# when its config doesn't say so.
QUANTIZED_WEIGHT_PARTS = frozenset({
    "weight_scale_inv", "weight_scale", "input_scale", "weight_scale_2", "weight_global_scale",
    "input_global_scale", "weight_packed", "weight_zero_point", "qweight", "qzeros",
})


def is_quantized_part(name: str) -> bool:
    """Whether checkpoint key ``name`` is a part of a quantized weight."""
    return name.rpartition(".")[2] in QUANTIZED_WEIGHT_PARTS


def checkpoint_quantization(path: str | Path) -> CheckpointQuantization | None:
    """The quantization that the checkpoint at ``path`` (a directory, or a file in one) declares.

    It is looked for where the producers write it: ``quantization_config`` in ``config.json``,
    top level or one section down (HF's own methods, compressed-tensors, newer ModelOpt
    exports); ``hf_quant_config.json`` (older ModelOpt exports); ``quantize_config.json``
    (AutoGPTQ) and ``quant_config.json`` (AutoAWQ). Returns None when none of these says so.
    """
    root = Path(path)
    if not root.is_dir():
        root = root.parent
    config = _read_json(root / "config.json")
    if isinstance(config, dict):
        for section in (config, *(v for v in config.values() if isinstance(v, dict))):
            quant = section.get("quantization_config")
            if isinstance(quant, dict) and quant:
                return CheckpointQuantization(str(quant.get("quant_method", "unknown")), str(root / "config.json"))
    modelopt = _read_json(root / "hf_quant_config.json")
    if isinstance(modelopt, dict) and isinstance(modelopt.get("quantization"), dict):
        return CheckpointQuantization("modelopt", str(root / "hf_quant_config.json"))
    for name, method, key in (("quantize_config.json", "gptq", "bits"), ("quant_config.json", "awq", "w_bit")):
        legacy = _read_json(root / name)
        if isinstance(legacy, dict) and key in legacy:
            return CheckpointQuantization(method, str(root / name))
    return None


def refuse_unserved_quantization(path: str | Path, quant_methods: Collection[str] = ()) -> None:
    """Raise if the checkpoint at ``path`` is quantized with a method that is not in ``quant_methods``,
    unless the node loading it reads it under a quantization scope (``mstar.quantization``), whose
    plans load each quantized layer and refuse at boot what they can't."""
    quant = checkpoint_quantization(path)
    if quant is None or quant.method in quant_methods or _scope_reads(path):
        return
    serves = "only unquantized checkpoints"
    if quant_methods:
        serves = f"unquantized checkpoints and quant_method {', '.join(sorted(quant_methods))}"
    raise UnservedQuantizationError(
        f"{quant.source} declares a quantized checkpoint (quant_method {quant.method}). This model "
        f"loads {serves}, so these weights would load as garbage. Serve an unquantized checkpoint of "
        "the model."
    )


def _read_json(path: Path) -> object:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        # absent or not JSON: the file declares nothing
        return None


def _scope_reads(path: str | Path) -> bool:
    """Whether the active quantization scope declares a weights root read from ``path``'s directory."""
    # here, not at the top: mstar.quantization imports this package's loader
    from mstar.quantization.scope import active_scope

    scope = active_scope()
    root = Path(path) if Path(path).is_dir() else Path(path).parent
    return scope is not None and any(Path(d).resolve() == root.resolve() for d in scope.checkpoint_dirs())
