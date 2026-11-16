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
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from safetensors import safe_open

from mstar.model.reload_errors import WeightReloadRefusedError

# A dtype change between these converts on copy; any other change is refused.
_FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16"})
# Compared when both checkpoints ship them. config.json minus these keys,
# which name the writer or the storage dtype rather than the architecture.
_CONFIG_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "tokenizer.model")
_CONFIG_IGNORED_KEYS = frozenset({"_name_or_path", "transformers_version", "torch_dtype", "dtype"})


@dataclass(frozen=True)
class ReloadManifest:
    """The compatibility facts from the checkpoint actually loaded at startup."""
    tensors: dict[str, tuple[tuple[int, ...], str]]
    configs: dict[str, dict | bytes]


def safetensors_files(path: str | Path) -> list[Path]:
    """The files the loader reads for ``path``: the file itself, the shards an
    index names, or a lone ``model.safetensors``. Index names must stay within
    the checkpoint directory lexically; Hugging Face snapshot shard symlinks
    may resolve into its shared blobs directory."""
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
            if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
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


def reload_manifest(path: str | Path, skip_fragments: Iterable[str] = ()) -> ReloadManifest:
    """Read a small, durable compatibility snapshot before the source path changes."""
    path = Path(path)
    configs: dict[str, dict | bytes] = {}
    for name in _CONFIG_FILES:
        file = _config_dir(path) / name
        if not file.is_file():
            continue
        try:
            if name == "config.json":
                value = json.loads(file.read_text())
                if not isinstance(value, dict):
                    raise WeightReloadRefusedError(f"{name} is not a JSON object")
                configs[name] = value
            else:
                configs[name] = sha256(file.read_bytes()).digest()
        except (OSError, ValueError) as exc:
            raise WeightReloadRefusedError(f"unreadable {name}: {exc}") from exc
    return ReloadManifest(tensors=tensor_specs(path, skip_fragments), configs=configs)


def _config_mismatches(new: ReloadManifest, reference: ReloadManifest) -> list[str]:
    mismatched = []
    for name in _CONFIG_FILES:
        if name not in new.configs or name not in reference.configs:
            continue
        if name == "config.json":
            new_cfg, ref_cfg = new.configs[name], reference.configs[name]
            keys = (new_cfg.keys() | ref_cfg.keys()) - _CONFIG_IGNORED_KEYS
            differ = sorted(k for k in keys if new_cfg.get(k) != ref_cfg.get(k))
            if differ:
                mismatched.append(f"{name} differs in {differ[:5]}")
        elif new.configs[name] != reference.configs[name]:
            mismatched.append(f"{name} differs")
    return mismatched


def check_reload_compatible(
    new: str | Path, reference: str | Path | ReloadManifest, skip_fragments: Iterable[str] = (),
) -> None:
    """Raise ``WeightReloadRefusedError`` unless ``new`` can be copied in place over
    weights loaded from ``reference``: the same tensor names and shapes (names
    containing ``skip_fragments`` aside), float dtypes or unchanged ones, and the
    same configs where both ship them."""
    new = Path(new)
    new_manifest = reload_manifest(new, skip_fragments)
    ref_manifest = (
        reference if isinstance(reference, ReloadManifest)
        else reload_manifest(reference, skip_fragments)
    )
    new_specs, ref_specs = new_manifest.tensors, ref_manifest.tensors
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
    problems += _config_mismatches(new_manifest, ref_manifest)
    if problems:
        raise WeightReloadRefusedError(f"{new} does not match the loaded checkpoint: " + "; ".join(problems))
