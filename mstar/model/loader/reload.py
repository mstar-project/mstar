"""Checks a checkpoint against the loaded one before an in-place reload.

A reload copies into parameters that were sized, fused and sharded for the
checkpoint the deployment started with, and compiled code and captured graphs
hold pointers to that storage. So the new checkpoint must carry the same tensor
names and shapes, and the same model and tokenizer configs where it ships them;
only the dtype may differ between floating types, since the copy converts into
the existing storage as the startup load does.

Everything here reads safetensors headers and small config files only, so a
refusal leaves the loaded weights untouched.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from safetensors import safe_open

# A dtype change between these converts on copy; any other change is refused.
_FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16"})
# Compared when both checkpoints ship them. config.json minus these keys,
# which name the writer or the storage dtype rather than the architecture.
_CONFIG_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "tokenizer.model")
_CONFIG_IGNORED_KEYS = frozenset({"_name_or_path", "transformers_version", "torch_dtype", "dtype"})


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


def _config_mismatches(new: Path, reference: Path) -> list[str]:
    mismatched = []
    for name in _CONFIG_FILES:
        new_file, ref_file = _config_dir(new) / name, _config_dir(reference) / name
        if not (new_file.is_file() and ref_file.is_file()):
            continue
        if name == "config.json":
            try:
                new_cfg, ref_cfg = json.loads(new_file.read_text()), json.loads(ref_file.read_text())
            except (OSError, ValueError) as exc:
                raise WeightReloadRefusedError(f"unreadable {name}: {exc}") from exc
            if not (isinstance(new_cfg, dict) and isinstance(ref_cfg, dict)):
                raise WeightReloadRefusedError(f"{name} is not a JSON object")
            keys = (new_cfg.keys() | ref_cfg.keys()) - _CONFIG_IGNORED_KEYS
            differ = sorted(k for k in keys if new_cfg.get(k) != ref_cfg.get(k))
            if differ:
                mismatched.append(f"{name} differs in {differ[:5]}")
        elif new_file.read_bytes() != ref_file.read_bytes():
            mismatched.append(f"{name} differs")
    return mismatched


def check_reload_compatible(
    new: str | Path, reference: str | Path, skip_fragments: Iterable[str] = (),
) -> None:
    """Raise ``WeightReloadRefusedError`` unless ``new`` can be copied in place over
    weights loaded from ``reference``: the same tensor names and shapes (names
    containing ``skip_fragments`` aside), float dtypes or unchanged ones, and the
    same configs where both ship them."""
    new, reference = Path(new), Path(reference)
    new_specs = tensor_specs(new, skip_fragments)
    ref_specs = tensor_specs(reference, skip_fragments)
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
