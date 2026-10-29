"""Tuned kernel configs, keyed by architecture and product.

A kernel's launch parameters (block sizes, warps, stages) live in
``configs/<arch>/<kernel>.json``, with ``configs/<arch>/<product>/<kernel>.json``
for a product whose timings differ (MI300X and MI325X share gfx942 binaries
but not their best tiles). Inside, configs are keyed by the op's shape class::

    {"kernel": "rms_norm", "arch": "gfx942", "product": null,
     "classes": {"decode": {"num_warps": 4}, "default": {"num_warps": 8}},
     "provenance": {"job": "...", "sha": "...", "stack": "..."}}

The kernel harness (``benchmark/kernels``) writes these from its results
store; a config is promoted only after a model-level A/B.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


def config_path(kernel: str, arch: str, product: str | None = None, root: Path | None = None) -> Path:
    """Where ``kernel``'s config for ``arch`` (and ``product``) lives."""
    base = Path(CONFIG_ROOT if root is None else root) / arch
    return base / product / f"{kernel}.json" if product else base / f"{kernel}.json"


@functools.cache
def _read(path: Path) -> Mapping[str, Any] | None:
    try:
        with open(path) as f:
            config = json.load(f)
    except FileNotFoundError:
        return None
    if not isinstance(config.get("classes"), dict):
        raise ValueError(f"{path}: a kernel config needs a 'classes' table")
    return config


def load_config(
    kernel: str, arch: str | None, product: str | None = None, root: Path | None = None,
) -> Mapping[str, Any] | None:
    """``kernel``'s config for this device: the product's if it has one, else
    the arch's; None when neither exists."""
    if arch is None:
        return None
    paths = [config_path(kernel, arch, product, root)] if product else []
    paths.append(config_path(kernel, arch, None, root))
    for path in paths:
        config = _read(path)
        if config is not None:
            return config
    return None


def select(config: Mapping[str, Any] | None, shape_class: str | None) -> Mapping[str, Any] | None:
    """The entry for ``shape_class``, else the config's ``default``."""
    if config is None:
        return None
    classes = config["classes"]
    if shape_class is not None and shape_class in classes:
        return classes[shape_class]
    return classes.get("default")
