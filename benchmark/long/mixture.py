"""Config + request-pool construction for the soak harness.

A soak config is a YAML file: top-level knobs (model, rate, in-flight cap,
duration…) plus a ``requests`` list of weighted request configs, each naming a
``req_type`` (modality), a ``dataset`` to draw prompts/inputs from, per-request
``model_kwargs`` (e.g. generation resolution), and a ``weight`` (the sampling
probability; the weights must sum to 1). At startup each entry's dataset is
materialised once into a pool of ``RequestInput``s; the driver samples the pool
with replacement and overlays the entry's ``model_kwargs``.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

import yaml

from benchmark.base import RequestType
from benchmark.dataset import (
    Food101Dataset,
    LibriSpeechDataset,
    SeedTTSDataset,
    TxtFileDataset,
    UCF101Dataset,
    VBenchDataset,
    VideoMMEDataset,
)
from benchmark.request import RequestInput

# Datasets that need a local/download cache but take it as a keyword.
_DEFAULT_TEXT_FILE = "benchmark/assets/simple_text_queries.txt"


def _build_dataset_requests(
    dataset: str,
    req_type: RequestType,
    pool_size: int,
    cache_dir: str,
    dataset_kwargs: dict[str, Any],
) -> list[RequestInput]:
    """Materialise ``pool_size`` ``RequestInput``s for one (dataset, req_type).

    Mirrors ``benchmark.runner.Benchmark._get_dataset`` but as a standalone
    factory keyed by the dataset name used in the YAML. ``dataset_kwargs`` are
    per-entry overrides (locale, data dirs, prompt file, …).
    """
    dk = dataset_kwargs
    if dataset == "text":
        ds = TxtFileDataset(
            filename=dk.get("filename", _DEFAULT_TEXT_FILE),
            num_requests=pool_size,
            req_type=req_type,
        )
    elif dataset == "vbench":
        ds = VBenchDataset(
            cache_dir=dk.get("vbench_cache_dir", cache_dir),
            task=req_type,
            num_requests=pool_size,
        )
    elif dataset == "libri":
        ds = LibriSpeechDataset(
            local_file_dir=cache_dir,
            num_requests=pool_size,
            req_type=req_type,
            split=dk.get("split", "validation"),
            cache_dir=dk.get("cache_dir"),
        )
    elif dataset == "food101":
        ds = Food101Dataset(
            num_requests=pool_size,
            req_type=req_type,
            prompts=dk.get("prompts"),
            split=dk.get("split", "validation"),
            cache_dir=dk.get("cache_dir"),
        )
    elif dataset == "ucf101":
        ds = UCF101Dataset(
            local_file_dir=cache_dir,
            num_requests=pool_size,
            req_type=req_type,
            split=dk.get("split", "train"),
            cache_dir=dk.get("cache_dir"),
        )
    elif dataset == "video_mme":
        ds = VideoMMEDataset(
            num_requests=pool_size,
            req_type=req_type,
            data_dir=dk.get("video_mme_dir"),
            cache_dir=dk.get("cache_dir", cache_dir),
        )
    elif dataset == "seed_tts":
        ds = SeedTTSDataset(
            num_requests=pool_size,
            req_type=req_type,
            locale=dk.get("locale", "en"),
            data_dir=dk.get("seed_tts_dir"),
            cache_dir=dk.get("cache_dir", cache_dir),
        )
    else:
        raise ValueError(
            f"unknown dataset {dataset!r} (known: text, vbench, libri, food101, "
            f"ucf101, video_mme, seed_tts)"
        )
    reqs = ds.get_requests()
    if not reqs:
        raise ValueError(
            f"dataset {dataset!r} produced no requests for {req_type.value}"
        )
    return reqs


@dataclass
class ReqConfig:
    """One weighted entry of the request mixture."""

    req_type: RequestType
    dataset: str
    weight: float
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    label: str = ""  # human-readable tag for per-entry reporting
    pool: list[RequestInput] = field(default_factory=list, repr=False)

    def sample(self, rng: random.Random) -> RequestInput:
        """A pooled input with this entry's ``model_kwargs`` overlaid.

        ``copy.copy`` (not ``dataclasses.replace``) so the pooled media bytes
        are shared read-only and ``RequestInput.__post_init__`` doesn't re-read
        / re-encode files per request.
        """
        base = rng.choice(self.pool)
        req = copy.copy(base)
        req.model_kwargs = {**base.model_kwargs, **self.model_kwargs}
        return req


@dataclass
class SoakConfig:
    model: str
    system: str = "ours"  # "ours" (/generate) | "ours_openai" (/v1/*)
    rate: float = 8.0  # Poisson arrival rate, requests/sec
    max_in_flight: int = 32  # admission cap (server-facing concurrency)
    duration_s: float = 3600.0
    request_timeout_s: float = 300.0
    pool_size: int = 64  # prompts materialised per mixture entry
    cache_dir: str = "./.soak_cache"
    report_interval_s: float = 15.0
    window_s: float = 60.0  # moving-average window
    warmup_s: float = 0.0  # exclude the first N seconds from the summary
    seed: int = 0
    requests: list[ReqConfig] = field(default_factory=list)

    @property
    def weights(self) -> list[float]:
        return [r.weight for r in self.requests]


def load_config(path: str, overrides: dict[str, Any] | None = None) -> SoakConfig:
    """Parse a soak YAML, validate the weights sum to 1, and build the pools.

    ``overrides`` (from CLI flags) win over YAML values for the scalar top-level
    knobs, so a single YAML can be swept over rate / concurrency / duration.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")

    entries_raw = raw.pop("requests", None)
    if not entries_raw:
        raise ValueError(f"{path}: missing non-empty 'requests' list")

    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    known = {
        "model", "system", "rate", "max_in_flight", "duration_s",
        "request_timeout_s", "pool_size", "cache_dir", "report_interval_s",
        "window_s", "warmup_s", "seed",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys: {sorted(unknown)}")
    merged = {**raw, **overrides}
    if "model" not in merged:
        raise ValueError(f"{path}: 'model' is required")

    cfg = SoakConfig(**merged)

    total_w = 0.0
    for e in entries_raw:
        req_type = RequestType(e["req_type"])
        rc = ReqConfig(
            req_type=req_type,
            dataset=e["dataset"],
            weight=float(e["weight"]),
            model_kwargs=e.get("model_kwargs", {}) or {},
            dataset_kwargs=e.get("dataset_kwargs", {}) or {},
            label=e.get("label") or f"{req_type.value}:{e['dataset']}",
        )
        total_w += rc.weight
        cfg.requests.append(rc)

    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(
            f"{path}: request weights must sum to 1.0, got {total_w:.6f}"
        )

    # Build each entry's prompt/input pool once, up front (network / disk here,
    # never on the hot path).
    for rc in cfg.requests:
        rc.pool = _build_dataset_requests(
            dataset=rc.dataset,
            req_type=rc.req_type,
            pool_size=cfg.pool_size,
            cache_dir=cfg.cache_dir,
            dataset_kwargs=rc.dataset_kwargs,
        )
    return cfg
