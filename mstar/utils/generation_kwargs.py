"""The client-facing generation knobs, checked once where every endpoint submits.

One meaning per knob on every model: ``null`` is the
same as leaving the key out, types are strict (no ``"0.7"``), and a value out of
range is a ``ValueError`` (a 400) before any worker sees the request. A
multi-stage model's ``<stage>_<knob>`` (``talker_temperature``) follows the
rule for ``<knob>``.
"""

import math


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


# knob -> (rule text, predicate) for real-valued knobs
_REAL_RULES = {
    "temperature": (">= 0 (0 is greedy)", lambda v: v >= 0),
    "top_p": ("in (0, 1] (1 is off; greedy is temperature 0)", lambda v: 0 < v <= 1),
    "min_p": ("in [0, 1] (0 is off)", lambda v: 0 <= v <= 1),
    "repetition_penalty": ("> 0 (1 is off)", lambda v: v > 0),
}
_INT_RULES = {
    "top_k": (">= 0 (0 is off)", lambda v: v >= 0),
    "max_output_tokens": (">= 1", lambda v: v >= 1),
}
_BOOL_KNOBS = ("ignore_eos", "penalize_prompt")

# other names for max_output_tokens, in precedence order after it
MAX_OUTPUT_TOKENS_ALIASES = ("max_new_tokens", "max_completion_tokens", "max_tokens")


def _knob(key: str, names) -> str | None:
    """The knob ``key`` names: itself, or a ``<stage>_<knob>`` prefix of one."""
    for name in names:
        if key == name or key.endswith("_" + name):
            return name
    return None


def normalize_generation_kwargs(model_kwargs: dict | None) -> dict:
    """``model_kwargs`` with nulls dropped, length aliases folded into
    ``max_output_tokens``, and every known knob type- and range-checked.

    Raises ``ValueError`` naming the knob and its allowed range.
    """
    kwargs = {k: v for k, v in (model_kwargs or {}).items() if v is not None}
    for alias in MAX_OUTPUT_TOKENS_ALIASES:
        value = kwargs.pop(alias, None)
        if value is not None:
            kwargs.setdefault("max_output_tokens", value)
    for key, value in kwargs.items():
        if key == "seed":
            check_seed(value)
        elif (name := _knob(key, _REAL_RULES)) is not None:
            rule, ok = _REAL_RULES[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number {rule}; got {value!r}")
            if not math.isfinite(value) or not ok(value):
                raise ValueError(f"{key} must be a finite number {rule}; got {value!r}")
        elif (name := _knob(key, _INT_RULES)) is not None:
            rule, ok = _INT_RULES[name]
            if isinstance(value, bool) or not isinstance(value, int) or not ok(value):
                raise ValueError(f"{key} must be an integer {rule}; got {value!r}")
        elif _knob(key, _BOOL_KNOBS) is not None and not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false; got {value!r}")
    return kwargs
