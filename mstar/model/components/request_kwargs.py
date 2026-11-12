"""Sampling defaults and request-kwarg checks shared by the engine-sampler models.

The API normalizes the generation knobs (``normalize_generation_kwargs``:
numeric strings, ``16.0`` -> 16) on every route before a model sees them, so
nothing here coerces a client value: one of the wrong type or range is a
``ValueError``, the client's 400. ``process_prompt`` runs it on the data worker
so a bad value fails before any work, and the conductor-side readers run it
again so none can reach a step.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType

from mstar.engine.resources.sampler.config import (
    BOOL_KNOBS,
    INT_KNOB_RULES,
    REAL_KNOB_RULES,
    SamplingReqConfig,
    check_bool_knob,
    check_int_knob,
    check_real_knob,
)
from mstar.engine.state import SampleSettings
from mstar.model.utils import load_generation_defaults
from mstar.utils.generation_kwargs import check_seed

logger = logging.getLogger(__name__)

# model_kwargs forwarded to the engine sampler when the request sets them
SAMPLING_KEYS = (
    "temperature", "top_p", "top_k", "min_p",
    "repetition_penalty", "penalize_prompt", "ignore_eos",
)
# the ASR references decode greedily; a checkpoint's generation_config.json overrides
GREEDY_SAMPLING: Mapping[str, object] = MappingProxyType({"temperature": 0.0, "top_p": 1.0})

_INT_RULES = {**INT_KNOB_RULES, "max_output_tokens": (">= 1", lambda v: v >= 1)}
_STRS = ("language", "task", "initial_prompt", "assistant_prefix")
_TIMESTAMPS = (False, True, "segment", "word")


def sampling_defaults(checkpoint_dir, base: Mapping[str, object] = GREEDY_SAMPLING) -> dict:
    """``base`` (the model's built-in recipe) under the knobs the checkpoint's
    generation_config.json sets."""
    checkpoint = load_generation_defaults(checkpoint_dir)
    return {**base, **{k: v for k, v in checkpoint.items() if k in SAMPLING_KEYS}}


def generation_config_dir(model_path_hf: str | None, cache_dir: str | None) -> str | None:
    """The directory holding the checkpoint's generation_config.json: a local
    checkpoint, or a Hub id's cached snapshot. Never fetched, so a model that
    builds in the conductor does no Hub IO there; None when there is none."""
    if not model_path_hf:
        return None
    if Path(model_path_hf).is_dir():
        return model_path_hf
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            model_path_hf, "generation_config.json", cache_dir=cache_dir, local_files_only=True,
        )
    except (FileNotFoundError, ValueError):  # not cached, or not a Hub id
        logger.info("No cached generation_config.json for %s; the built-in sampling defaults apply",
                    model_path_hf)
        return None
    return str(Path(path).parent)


def sampler_capabilities(defaults: Mapping[str, object]) -> dict:
    """``SamplerSpec`` capability flags: on only when the defaults use them, so a
    request ``min_p`` or penalty on a greedy model is refused at admission."""
    return {
        "enable_repetition_penalty": defaults.get("repetition_penalty", 1) != 1,
        "enable_min_p": bool(defaults.get("min_p")),
    }


def sampling_config(defaults: Mapping[str, object], kwargs: dict | None) -> SamplingReqConfig:
    """The request's sampler config: ``defaults`` under the knobs it sets."""
    kw = checked_request_kwargs(kwargs)
    return SamplingReqConfig(**{**defaults, **{k: kw[k] for k in SAMPLING_KEYS if k in kw}})


def sample_settings(config: SamplingReqConfig) -> SampleSettings:
    """``config`` as the settings a declared sampler takes, every field `SampleSettings` declares."""
    return SampleSettings(**{field.name: getattr(config, field.name) for field in fields(SampleSettings)})


def checked_request_kwargs(kwargs: dict | None) -> dict:
    """``kwargs`` without its nulls (a null is an unset key), or ``ValueError``
    naming a field of the wrong type or range."""
    out = {k: v for k, v in (kwargs or {}).items() if v is not None}
    for key, value in out.items():
        if key == "seed":
            check_seed(value)
        elif key in REAL_KNOB_RULES:
            check_real_knob(key, value, *REAL_KNOB_RULES[key])
        elif key in _INT_RULES:
            check_int_knob(key, value, *_INT_RULES[key])
        elif key in BOOL_KNOBS:
            check_bool_knob(key, value)
        elif key in _STRS and not isinstance(value, str):
            raise ValueError(f"{key} must be a string, got {value!r}")
    if out.get("timestamps") is not None and out["timestamps"] not in _TIMESTAMPS:
        raise ValueError(f"timestamps must be one of {_TIMESTAMPS}, got {out['timestamps']!r}")
    return out
