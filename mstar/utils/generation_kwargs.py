"""The client-facing generation knobs, sampling and output geometry, checked once where every endpoint submits.

One meaning per knob on every model: ``null`` is the
same as leaving the key out, a value is coerced to the knob's type only when it
denotes exactly that value (``"0.7"``, ``"5"``, ``5.0`` for an integer; never a
bool), and a value out of range is a ``ValueError`` (a 400) before any worker
sees the request. A multi-stage model's ``<stage>_<knob>`` (``talker_temperature``)
follows the rule for ``<knob>``.
"""

import re

from mstar.engine.resources.sampler.config import (
    BOOL_KNOBS,
    INT_KNOB_RULES,
    REAL_KNOB_RULES,
    check_bool_knob,
    check_int_knob,
    check_real_knob,
)


def check_seed(seed) -> None:
    """Raise ``ValueError`` unless ``seed`` is None or an integer that fits in int64.

    int64 is the widest seed a sampler can store or pass to ``manual_seed``,
    and past it the request cannot even be packed for the conductor.
    """
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}.")
    if not -(2**63) <= seed < 2**63:
        raise ValueError(f"seed must fit in a signed 64-bit integer, got {seed}.")


def offset_seed(seed: int, i: int) -> int:
    """``seed + i`` wrapped to signed int64, as the Rust frontend computes it."""
    return (seed + i + 2**63) % 2**64 - 2**63


# knob -> (rule text, predicate); the sampling ones are the sampler's own
_REAL_RULES = {
    **REAL_KNOB_RULES,
    "fps": ("> 0", lambda v: v > 0),
    "guidance_scale": ("of any sign", lambda v: True),
    "speed": ("> 0", lambda v: v > 0),
}
_INT_RULES = {
    **INT_KNOB_RULES,
    "max_output_tokens": (">= 1", lambda v: v >= 1),
    "num_inference_steps": (">= 1", lambda v: v >= 1),
    "num_step": (">= 1", lambda v: v >= 1),
    "rollout_horizon": (">= 0", lambda v: v >= 0),
    "num_frames": (">= 1", lambda v: v >= 1),
    "height": (">= 1", lambda v: v >= 1),
    "width": (">= 1", lambda v: v >= 1),
}
_BOOL_KNOBS = (*BOOL_KNOBS, "do_sample")

# other names for max_output_tokens, in precedence order after it
MAX_OUTPUT_TOKENS_ALIASES = ("max_new_tokens", "max_completion_tokens", "max_tokens")


# an integer literal, optionally with a zero fraction: "5", "-3", "5.0"; no exponent
_INTEGRAL = re.compile(r"[+-]?[0-9]+(?:\.0*)?")


def _coerce_real(value):
    """A numeric string as its float; anything else unchanged, for the checks to judge."""
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    return value


def _coerce_int(value):
    """An integer-valued number or string as its int; anything else unchanged."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and _INTEGRAL.fullmatch(text := value.strip()):
        return int(text.split(".")[0])
    return value


def _knob(key: str, names) -> str | None:
    """The knob ``key`` names: itself, or a ``<stage>_<knob>`` prefix of one."""
    for name in names:
        if key == name or key.endswith("_" + name):
            return name
    return None


def normalize_generation_kwargs(model_kwargs: dict | None) -> dict:
    """``model_kwargs`` with nulls dropped, length aliases folded into
    ``max_output_tokens``, and every known knob coerced, then type- and range-checked.

    Raises ``ValueError`` naming the knob and its allowed range.
    """
    kwargs = {k: v for k, v in (model_kwargs or {}).items() if v is not None}
    for alias in MAX_OUTPUT_TOKENS_ALIASES:
        value = kwargs.pop(alias, None)
        if value is not None:
            kwargs.setdefault("max_output_tokens", value)
    for key, value in kwargs.items():
        if key == "seed":
            kwargs[key] = value = _coerce_int(value)
            check_seed(value)
        elif (name := _knob(key, _REAL_RULES)) is not None:
            kwargs[key] = value = _coerce_real(value)
            check_real_knob(key, value, *_REAL_RULES[name])
        elif (name := _knob(key, _INT_RULES)) is not None:
            kwargs[key] = value = _coerce_int(value)
            check_int_knob(key, value, *_INT_RULES[name])
        elif _knob(key, _BOOL_KNOBS) is not None:
            check_bool_knob(key, value)
    return kwargs
