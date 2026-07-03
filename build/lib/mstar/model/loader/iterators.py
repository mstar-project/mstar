"""Streaming iterators over safetensors checkpoints.

Yields ``(key, tensor)`` one at a time so the full state_dict never has
to fit in memory.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open


def _resolve_safetensors_device(device: torch.device | str) -> str:
    """safetensors accepts ``"cuda"`` (no index) and ``"cpu"`` only —
    not ``"cuda:0"``. Map our device strings to its conventions.
    """
    s = str(device)
    return "cuda" if s.startswith("cuda") else s


def iter_safetensors_file(
    path: str | Path,
    device: torch.device | str = "cpu",
    prefix: str | None = None,
    keys: set[str] | None=None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, tensor)`` from a single safetensors file."""
    st_device = _resolve_safetensors_device(device)

    with safe_open(str(path), framework="pt", device=st_device) as f:
        for key in f.keys():
            if (prefix is not None and not key.startswith(prefix)) or \
                    (keys is not None and key not in keys):
                continue

            tensor = f.get_tensor(key)
            if str(device) != st_device:
                tensor = tensor.to(device, non_blocking=True)
            yield key, tensor


def iter_safetensors_shards(
    repo_dir: str | Path, device: torch.device | str = "cpu",
    prefix: str | None = None,
    keys: set[str] | None=None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, tensor)`` from a sharded HF safetensors checkpoint.

    Looks for ``model.safetensors.index.json`` in ``repo_dir``; if absent,
    falls back to a single ``model.safetensors`` file.
    """
    repo_dir = Path(repo_dir)
    index_path = repo_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)

        if prefix is not None or keys is not None:
            relevant_keys = [
                key for key in index["weight_map"]
                if (prefix is None or key.startswith(prefix)) and \
                   (keys is None or key in keys)
            ]
            shard_files = sorted(set([
                index["weight_map"][key] for key in relevant_keys
            ]))
        else:
            shard_files = sorted(set(index["weight_map"].values()))

        for shard_file in shard_files:
            yield from iter_safetensors_file(
                repo_dir / shard_file, device=device,
                prefix=prefix, keys=keys
            )
        return
    single = repo_dir / "model.safetensors"
    if single.exists():
        yield from iter_safetensors_file(
            single, device=device,
            prefix=prefix, keys=keys
        )
        return
    raise FileNotFoundError(
        f"No safetensors checkpoint found in {repo_dir} "
        f"(looked for model.safetensors.index.json and model.safetensors)"
    )
