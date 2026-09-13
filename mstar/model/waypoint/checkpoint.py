"""Resolve and validate the two checkpoints used by Waypoint.

Resolution happens before model construction so a bad repository, partial local
download, or incompatible ``config.yaml`` cannot allocate several GiB of GPU
storage before failing. Local paths always win and never touch Hugging Face.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

from mstar.model.waypoint.config import (
    WaypointConfig,
)

WAYPOINT_CONFIG_FILE = "config.yaml"
WAYPOINT_WEIGHT_ALLOW_PATTERNS = (
    WAYPOINT_CONFIG_FILE,
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
)
TAEHV_CHECKPOINT_FILE = "taehv1_5.pth"
TAEHV_UPSTREAM_REVISION = "7dc60ec6601af2e668e31bc70acc4cb3665e4c22"
TAEHV_UPSTREAM_ARCHIVE = (
    "https://github.com/madebyollin/taehv/archive/"
    f"{TAEHV_UPSTREAM_REVISION}.zip"
)

_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


def require_taehv_runtime() -> None:
    """Fail before checkpoint download or module allocation if TAEHV is absent."""
    try:
        taehv = import_module("taehv")
    except ImportError as exc:
        raise RuntimeError(
            "Waypoint requires the pinned TAEHV runtime. Install the index-safe "
            "dependencies with `uv pip install -e '.[waypoint]'`, then install "
            "TAEHV separately with `uv pip install --no-deps "
            f"'taehv @ {TAEHV_UPSTREAM_ARCHIVE}'`. The TAEHV source package "
            "requires uv>=0.4.0 or pip>=24.3."
        ) from exc
    if not hasattr(taehv, "TAEHV"):
        raise RuntimeError(
            "The installed `taehv` module does not export TAEHV. Ensure the "
            "index-safe dependencies are installed with `uv pip install -e "
            "'.[waypoint]'`, then remove the invalid module and install the pinned "
            "source separately with `uv pip install --no-deps "
            f"'taehv @ {TAEHV_UPSTREAM_ARCHIVE}'`; use uv>=0.4.0 or pip>=24.3 "
            "because older pip can install an empty UNKNOWN wheel."
        )


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    if isinstance(value, dict):
        return {key: _as_tuple(item) for key, item in value.items()}
    return value


def _local_source(source: str | Path, kind: str) -> Path | None:
    raw = str(source)
    path = Path(source).expanduser()
    if path.exists():
        return path.resolve()
    if raw.startswith(("/", ".", "~")) or not _HF_REPO_ID.fullmatch(raw):
        raise FileNotFoundError(f"{kind} local path does not exist: {path}")
    return None


def _checkpoint_weight_files(checkpoint_dir: Path) -> tuple[Path, ...]:
    single = checkpoint_dir / "model.safetensors"
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if single.is_file():
        return (single,)
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Waypoint checkpoint {checkpoint_dir} has no model.safetensors or "
            "model.safetensors.index.json."
        )
    try:
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid Waypoint safetensors index {index_path}: {exc}") from exc
    if not shard_names or any(not isinstance(name, str) for name in shard_names):
        raise ValueError(f"Waypoint safetensors index {index_path} has no valid weight shards.")
    shards = tuple(checkpoint_dir / name for name in shard_names)
    escaped = [path for path in shards if path.parent.resolve() != checkpoint_dir.resolve()]
    if escaped:
        raise ValueError(f"Waypoint safetensors index contains a path outside {checkpoint_dir}.")
    missing = [path.name for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Waypoint checkpoint {checkpoint_dir} is incomplete; missing shards: {missing}."
        )
    return shards


def _expected_manifest(config: WaypointConfig) -> dict[str, Any]:
    return {
        "model_type": "waypoint-1.5",
        "inference_fps": config.inference_fps,
        "temporal_compression": config.temporal_compression,
        "taehv_ae": config.taehv_ae,
        "ae_uri": config.ae_uri,
        "prompt_conditioning": config.prompt_conditioning,
        "channels": config.channels,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "d_model": config.d_model,
        "mlp_ratio": config.mlp_ratio,
        "causal": True,
        "moe": config.moe,
        "n_buttons": config.n_buttons,
        "patch": config.patch,
        "base_fps": config.base_fps,
        "local_window": config.local_window,
        "global_window": config.global_window,
        "global_pinned_dilation": config.global_pinned_dilation,
        "global_attn_period": config.global_attn_period,
        "global_attn_offset": config.global_attn_offset,
        "n_frames": config.max_frames,
        "rope_impl": config.rope_impl,
        "value_residual": config.value_residual,
        "gated_attn": config.gated_attn,
        "noise_conditioning": config.noise_conditioning,
        "ctrl_conditioning": config.ctrl_conditioning,
        "ctrl_cond_dropout": config.ctrl_cond_dropout,
        "ctrl_conditioning_period": config.ctrl_conditioning_period,
        "scheduler_sigmas": config.scheduler_sigmas,
    }


def validate_waypoint_checkpoint(checkpoint_dir: str | Path, config: WaypointConfig) -> Path:
    """Validate files and checkpoint facts, returning an absolute directory."""
    config.validate_supported_deployment()
    directory = Path(checkpoint_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Waypoint checkpoint directory not found: {directory}")
    config_path = directory / WAYPOINT_CONFIG_FILE
    if not config_path.is_file():
        raise FileNotFoundError(f"Waypoint checkpoint is missing {config_path}.")
    _checkpoint_weight_files(directory)

    try:
        manifest = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid Waypoint checkpoint config {config_path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Waypoint checkpoint config {config_path} must contain a mapping.")

    expected = _expected_manifest(config)
    mismatches = []
    for key, expected_value in expected.items():
        if key not in manifest:
            mismatches.append(f"{key}=<missing> (expected {expected_value!r})")
            continue
        actual = _as_tuple(manifest[key])
        if actual != expected_value:
            mismatches.append(f"{key}={actual!r} (expected {expected_value!r})")

    manifest_geometry = (
        manifest.get("tokens_per_frame"),
        manifest.get("height"),
        manifest.get("width"),
    )
    expected_geometry = (config.tokens_per_frame, config.height, config.width)
    if manifest_geometry != expected_geometry:
        mismatches.append(
            "checkpoint geometry "
            f"{manifest_geometry!r} (expected {expected_geometry!r} for "
            f"{config.variant!r})"
        )
    conv_kw = _as_tuple(manifest.get("conv_kw"))
    expected_conv = {"kernel_size": config.patch, "stride": config.patch}
    if conv_kw != expected_conv:
        mismatches.append(f"conv_kw={conv_kw!r} (expected {expected_conv!r})")
    if mismatches:
        raise ValueError(
            f"Waypoint checkpoint {config_path} is incompatible with {config.variant!r}: "
            + "; ".join(mismatches)
            + "."
        )
    return directory


def resolve_waypoint_checkpoint(
    source: str | Path,
    config: WaypointConfig,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> Path:
    """Resolve a local directory or HF repo ID and validate it before allocation."""
    local = _local_source(source, "Waypoint checkpoint")
    if local is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Resolving a Waypoint Hugging Face ID requires the `waypoint` extra: "
                "install with `uv pip install -e '.[waypoint]'`."
            ) from exc
        try:
            local = Path(
                snapshot_download(
                    repo_id=str(source),
                    cache_dir=None if cache_dir is None else str(cache_dir),
                    revision=revision,
                    allow_patterns=list(WAYPOINT_WEIGHT_ALLOW_PATTERNS),
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download required Waypoint files from {source!r}: {exc}"
            ) from exc
    return validate_waypoint_checkpoint(local, config)


def resolve_taehv_checkpoint(
    source: str | Path,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> Path:
    """Resolve only ``taehv1_5.pth`` from a local path or HF repository."""
    local = _local_source(source, "TAEHV checkpoint")
    if local is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Resolving TAEHV weights requires the `waypoint` extra: "
                "install with `uv pip install -e '.[waypoint]'`."
            ) from exc
        try:
            local = Path(
                hf_hub_download(
                    repo_id=str(source),
                    filename=TAEHV_CHECKPOINT_FILE,
                    cache_dir=None if cache_dir is None else str(cache_dir),
                    revision=revision,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download required TAEHV file from {source!r}: {exc}"
            ) from exc
    checkpoint = local if local.is_file() else local / TAEHV_CHECKPOINT_FILE
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No TAEHV checkpoint for source={str(source)!r}; looked for {checkpoint}."
        )
    return checkpoint.resolve()
