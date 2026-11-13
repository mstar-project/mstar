"""Strict, selective loading of Praxis-1 component weights.

The public checkpoint uses ``backbone.encoder.`` and ``action_expert.``
prefixes. A worker can inspect the index before fetching weights and fetch
only the shard names returned by :func:`required_shards` for its node.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"


class Praxis1WeightError(ValueError):
    """A checkpoint cannot be loaded into the requested component."""


def _weight_map(root: Path) -> dict[str, str]:
    index_path = root / INDEX
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text())["weight_map"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise Praxis1WeightError(f"invalid Praxis-1 index {index_path}") from exc
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(
                not isinstance(key, str)
                or not isinstance(file, str)
                or not file.endswith(".safetensors")
                or Path(file).name != file
                for key, file in weight_map.items()
            )
        ):
            raise Praxis1WeightError(f"invalid tensor names or shard paths in {index_path}")
        return weight_map
    single = root / SINGLE
    if single.is_file():
        with safe_open(str(single), framework="pt") as shard:
            return dict.fromkeys(shard.keys(), SINGLE)
    raise Praxis1WeightError(f"no {INDEX} or {SINGLE} in {root}")


def required_shards(root: str | Path, prefix: str) -> tuple[str, ...]:
    """Shard filenames for a component; no weight tensor is read for an index."""
    if not prefix:
        raise ValueError("a component prefix is required")
    weight_map = _weight_map(Path(root))
    files = {file for name, file in weight_map.items() if name.startswith(prefix)}
    if not files:
        raise Praxis1WeightError(f"no checkpoint tensors start with {prefix!r}")
    return tuple(sorted(files))


def _local_key(name: str) -> str:
    # The reference's dense FFN is an nn.Sequential; M* names its two
    # children explicitly. Every other ported parameter retains its name.
    name = name.replace(".ffn.0.linear.", ".ffn.linear.")
    name = name.replace(".ffn.1.weight", ".ffn.down.weight")
    if name.startswith("0.linear."):
        return "linear." + name[len("0.linear.") :]
    if name == "1.weight":
        return "down.weight"
    return name


def load_component(
    module: nn.Module,
    root: str | Path,
    prefix: str,
    *,
    assign: bool = False,
) -> None:
    """Load one ported component with exact keys, shapes and stored dtypes.

    The index selects shards, and ``safe_open`` reads only selected tensors.
    ``assign=True`` is for a module on the meta device. A normal load refuses
    implicit dtype casts, including a float32 MoE routing bias cast to bf16.
    """
    root = Path(root)
    weight_map = _weight_map(root)
    shards = required_shards(root, prefix)
    loaded: dict[str, torch.Tensor] = {}
    for file in shards:
        shard_path = root / file
        if not shard_path.is_file():
            raise Praxis1WeightError(f"missing Praxis-1 shard {shard_path}")
        indexed = {name for name, mapped in weight_map.items() if mapped == file}
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            stored = set(shard.keys())
            if stored != indexed:
                raise Praxis1WeightError(
                    f"{file} differs from the index: missing {sorted(indexed - stored)}, "
                    f"unexpected {sorted(stored - indexed)}"
                )
            for name in sorted(indexed):
                if name.startswith(prefix):
                    local = _local_key(name[len(prefix) :])
                    if local in loaded:
                        raise Praxis1WeightError(f"two checkpoint tensors map to {local!r}")
                    loaded[local] = shard.get_tensor(name)

    expected = module.state_dict(keep_vars=True)
    missing = sorted(set(expected) - set(loaded))
    unexpected = sorted(set(loaded) - set(expected))
    shapes = [name for name in sorted(set(expected) & set(loaded)) if expected[name].shape != loaded[name].shape]
    dtypes = [
        name
        for name in sorted(set(expected) & set(loaded))
        if not assign and expected[name].dtype != loaded[name].dtype
    ]
    meta = [name for name, value in expected.items() if value.is_meta] if not assign else []
    if missing or unexpected or shapes or dtypes or meta:
        raise Praxis1WeightError(
            f"cannot load {prefix!r}: missing {missing}, unexpected {unexpected}, "
            f"shape mismatch {shapes}, dtype mismatch {dtypes}, meta tensors {meta}"
        )
    module.load_state_dict(loaded, strict=True, assign=assign)
