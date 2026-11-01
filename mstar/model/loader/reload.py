"""Checks a checkpoint against the loaded one before an in-place reload.

A reload copies into parameters that were sized, fused and sharded for the
checkpoint the deployment started with, and compiled code and captured graphs
hold pointers to that storage. So the new checkpoint must carry the same tensor
names and shapes, and the same model and tokenizer configs where it ships them;
only the dtype may differ between floating types, since the copy converts into
the existing storage as the startup load does.

The loaded checkpoint is described by a ``CheckpointSnapshot`` taken when it
was loaded, not by re-reading its directory: the directory may since have been
deleted, or saved over in place by a trainer, which would make the new
checkpoint compare equal to itself.

Everything here reads safetensors headers and small config files only, so a
refusal leaves the loaded weights untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

# A dtype change between these converts on copy; any other change is refused.
_FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16"})
# Compared when both checkpoints ship them, as parsed JSON (tokenizer.model as
# bytes), each minus the keys below: at any depth for config.json, at the top
# level for the tokenizer files. Those keys name the writer, the storage dtype or
# a runtime setting, none of which changes the weights' shapes or what a token
# id means.
_CONFIG_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "tokenizer.model")
_IGNORED_KEYS = {
    "config.json": frozenset({
        "_name_or_path", "transformers_version", "torch_dtype", "dtype", "_commit_hash",
        "use_cache", "_attn_implementation", "attn_implementation",
        "output_attentions", "output_hidden_states", "return_dict", "torchscript",
    }),
    "tokenizer.json": frozenset({"truncation", "padding"}),
    "tokenizer_config.json": frozenset({
        "model_max_length", "padding_side", "truncation_side", "clean_up_tokenization_spaces",
    }),
}


class WeightReloadRefusedError(ValueError):
    """The checkpoint cannot be reloaded in place; nothing was copied."""


def safetensors_files(path: str | Path) -> list[Path]:
    """The files the loader reads for ``path``: the file itself, the shards an
    index names, or a lone ``model.safetensors``. Refuses a shard outside the
    checkpoint's directory."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise WeightReloadRefusedError(f"no checkpoint at {path}")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text())["weight_map"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise WeightReloadRefusedError(f"unreadable {index.name}: {exc}") from exc
        files = []
        for name in sorted(set(weight_map.values())):
            # Checked on the name, not the resolved path: in the Hugging Face
            # cache every file of a snapshot is a symlink into ``blobs/``.
            if Path(name).is_absolute() or os.path.normpath(name).split(os.sep)[0] == os.pardir:
                raise WeightReloadRefusedError(f"{index.name} names {name!r}, outside {path}")
            files.append(path / name)
        return files
    single = path / "model.safetensors"
    if single.is_file():
        return [single]
    raise WeightReloadRefusedError(f"no model.safetensors or model.safetensors.index.json in {path}")


def tensor_specs(
    path: str | Path, skip_fragments: Iterable[str] = (),
) -> dict[str, tuple[tuple[int, ...], str]]:
    """``name -> (shape, safetensors dtype)`` from the headers under ``path``,
    leaving out names containing any of ``skip_fragments``."""
    skip = tuple(skip_fragments)
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for file in safetensors_files(path):
        try:
            with safe_open(str(file), framework="pt") as f:
                for name in f.keys():
                    if any(fragment in name for fragment in skip):
                        continue
                    if name in specs:
                        raise WeightReloadRefusedError(f"{name} appears in more than one shard")
                    tensor = f.get_slice(name)
                    specs[name] = (tuple(tensor.get_shape()), tensor.get_dtype())
        except WeightReloadRefusedError:
            raise
        except Exception as exc:  # noqa: BLE001 - safetensors raises its own error type
            raise WeightReloadRefusedError(f"unreadable {file.name}: {exc}") from exc
    return specs


def _config_dir(path: Path) -> Path:
    return path.parent if path.is_file() else path


def _without(value: Any, ignored: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {k: _without(v, ignored) for k, v in value.items() if k not in ignored}
    if isinstance(value, list):
        return [_without(v, ignored) for v in value]
    return value


def _read_config(file: Path) -> Any:
    """What of ``file`` is compared: config.json as a dict, the other JSON files
    as a digest of their parsed content (formatting and key order don't count,
    and a large tokenizer.json is not held), tokenizer.model as a digest of its
    bytes."""
    name = file.name
    try:
        if not name.endswith(".json"):
            return hashlib.sha256(file.read_bytes()).hexdigest()
        content = json.loads(file.read_text())
    except (OSError, ValueError) as exc:
        raise WeightReloadRefusedError(f"unreadable {name}: {exc}") from exc
    if not isinstance(content, dict):
        raise WeightReloadRefusedError(f"{name} is not a JSON object")
    ignored = _IGNORED_KEYS.get(name, frozenset())
    if name == "config.json":
        return _without(content, ignored)
    kept = {k: v for k, v in content.items() if k not in ignored}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_configs(path: Path) -> dict[str, Any]:
    return {
        name: _read_config(file)
        for name in _CONFIG_FILES
        if (file := _config_dir(path) / name).is_file()
    }


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    """What a reload is checked against, read from the checkpoint when it was
    loaded: tensor names, shapes and dtypes, and the configs it shipped."""

    path: str
    specs: dict[str, tuple[tuple[int, ...], str]]
    configs: dict[str, Any]


def snapshot_checkpoint(path: str | Path, skip_fragments: Iterable[str] = ()) -> CheckpointSnapshot:
    """Describe the checkpoint at ``path`` for later ``check_reload_compatible``
    calls. Raises ``WeightReloadRefusedError`` if it cannot be read."""
    return CheckpointSnapshot(
        path=str(path), specs=tensor_specs(path, skip_fragments), configs=_read_configs(Path(path)),
    )


def _config_mismatches(new: Path, reference: CheckpointSnapshot) -> list[str]:
    mismatched = []
    new_configs = _read_configs(new)
    for name in _CONFIG_FILES:
        if name not in new_configs or name not in reference.configs:
            continue
        new_cfg, ref_cfg = new_configs[name], reference.configs[name]
        if name == "config.json":
            differ = sorted(k for k in new_cfg.keys() | ref_cfg.keys() if new_cfg.get(k) != ref_cfg.get(k))
            if differ:
                mismatched.append(f"{name} differs in {differ[:5]}")
        elif new_cfg != ref_cfg:
            mismatched.append(f"{name} differs")
    return mismatched


def check_reload_compatible(
    new: str | Path, reference: CheckpointSnapshot | str | Path, skip_fragments: Iterable[str] = (),
) -> None:
    """Raise ``WeightReloadRefusedError`` unless ``new`` can be copied in place over
    the weights ``reference`` describes (the snapshot taken when they were
    loaded, or a path, read now): the same tensor names and shapes (names
    containing ``skip_fragments`` aside), float dtypes or unchanged ones, and the
    same configs where both ship them."""
    new = Path(new)
    if not isinstance(reference, CheckpointSnapshot):
        reference = snapshot_checkpoint(reference, skip_fragments)
    new_specs = tensor_specs(new, skip_fragments)
    ref_specs = reference.specs
    problems = []
    missing = sorted(ref_specs.keys() - new_specs.keys())
    extra = sorted(new_specs.keys() - ref_specs.keys())
    if missing:
        problems.append(f"{len(missing)} tensor(s) missing, e.g. {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} tensor(s) the loaded checkpoint does not have, e.g. {extra[:3]}")
    for name in sorted(new_specs.keys() & ref_specs.keys()):
        (new_shape, new_dtype), (ref_shape, ref_dtype) = new_specs[name], ref_specs[name]
        if new_shape != ref_shape:
            problems.append(f"{name}: shape {list(new_shape)}, loaded {list(ref_shape)}")
        elif new_dtype != ref_dtype and not {new_dtype, ref_dtype} <= _FLOAT_DTYPES:
            problems.append(f"{name}: dtype {new_dtype}, loaded {ref_dtype}")
        if len(problems) >= 8:
            break
    problems += _config_mismatches(new, reference)
    if problems:
        raise WeightReloadRefusedError(f"{new} does not match the loaded checkpoint: " + "; ".join(problems))
