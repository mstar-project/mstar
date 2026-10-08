"""Request kwargs the ASR models read, checked before they reach a worker.

The OpenAI routes forward unknown form fields as model kwargs, so a value can
be a string or nonsense. ``process_prompt`` runs in the API process and raises
here, which fails the request with an error. The conductor-side readers call
the same check so a bad value can never reach a step.
"""
from __future__ import annotations

import math

_INTS = ("max_output_tokens", "seed")
_FLOATS = ("temperature", "top_p")
_BOOLS = ("ignore_eos",)
_STRS = ("language", "task", "initial_prompt", "assistant_prefix")
_TIMESTAMPS = (False, True, "segment", "word")


def _as_bool(key: str, value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
        return value.lower() in ("true", "1")
    raise ValueError(f"{key} must be true or false, got {value!r}")


def checked_request_kwargs(kwargs: dict | None) -> dict:
    """``kwargs`` with the known fields coerced, or ``ValueError`` naming the field."""
    out = dict(kwargs or {})
    for key in _INTS:
        if out.get(key) is None:
            continue
        value = out[key]
        try:
            if isinstance(value, bool) or int(value) != float(value):
                raise ValueError
            out[key] = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer, got {value!r}") from None
        if key == "max_output_tokens" and out[key] < 1:
            raise ValueError(f"max_output_tokens must be at least 1, got {out[key]}")
        # the conductor's seed is an int64
        if key == "seed" and not -2**63 <= out[key] < 2**63:
            raise ValueError(f"seed must fit in an int64, got {out[key]}")
    for key in _FLOATS:
        if out.get(key) is None:
            continue
        value = out[key]
        try:
            if isinstance(value, bool):
                raise ValueError
            out[key] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number, got {value!r}") from None
        if not math.isfinite(out[key]) or out[key] < 0:
            raise ValueError(f"{key} must be a finite non-negative number, got {value!r}")
    if out.get("top_p") is not None and not 0 < out["top_p"] <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {out['top_p']}")
    for key in _BOOLS:
        if out.get(key) is not None:
            out[key] = _as_bool(key, out[key])
    for key in _STRS:
        if out.get(key) is not None and not isinstance(out[key], str):
            raise ValueError(f"{key} must be a string, got {out[key]!r}")
    if out.get("timestamps") is not None and out["timestamps"] not in _TIMESTAMPS:
        raise ValueError(f"timestamps must be one of {_TIMESTAMPS}, got {out['timestamps']!r}")
    return out
