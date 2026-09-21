"""Speaking-rate and quality bucket resolution for Zonos2 TTS.

This is a port of ``../ZONOS2/python/zonos2/tts/conditioning.py``. The reference
reads its bucket definitions through a ``ServerArgs`` duck-type with server-side
overrides. Here they come from :class:`Zonos2Config`, which parses them from
``params.json``.

The bucket-spec grammar is unchanged, so the same checkpoint gives the same
bucket index as the reference server:

* speaking rate — contiguous ranges that start at 0 and end open. For example,
  ``["0-3", "3-6", ..., "60+"]``.
* quality — exact values (``"0"``), closed ranges (``"-30--25"``), or open
  ranges (``"22050+"``).

The caller passes the resolved indices to
:meth:`~mstar.model.zonos2.prompt.TTSPromptBuilder.build`.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from mstar.model.zonos2.config import Zonos2Config

# The reference uses this frame rate to convert bytes/second into bytes/frame.
# It applies when a checkpoint declares a bucket count but no explicit ranges.
_SPEAKING_RATE_FPS = 86.0 * (44070.0 / 44000.0)
_DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND = 15.0

_SPEAKING_RATE_CLOSED_BUCKET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$")
_SPEAKING_RATE_OPEN_BUCKET_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*$")
_QUALITY_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_QUALITY_EXACT_BUCKET_RE = re.compile(rf"^\s*({_QUALITY_NUMBER_RE})\s*$")
_QUALITY_CLOSED_BUCKET_RE = re.compile(
    rf"^\s*({_QUALITY_NUMBER_RE})\s*-\s*({_QUALITY_NUMBER_RE})\s*$"
)
_QUALITY_OPEN_BUCKET_RE = re.compile(rf"^\s*({_QUALITY_NUMBER_RE})\s*\+\s*$")


# ---------------------------------------------------------------------------
# Speaking rate
# ---------------------------------------------------------------------------


def _parse_speaking_rate_bucket(spec: str) -> tuple[float, float | None]:
    closed = _SPEAKING_RATE_CLOSED_BUCKET_RE.match(str(spec))
    if closed is not None:
        return float(closed.group(1)), float(closed.group(2))

    open_ended = _SPEAKING_RATE_OPEN_BUCKET_RE.match(str(spec))
    if open_ended is not None:
        return float(open_ended.group(1)), None

    raise ValueError(
        f"Invalid speaking-rate bucket {spec!r}; expected ranges like '0-3' or '60+'."
    )


def _speaking_rate_bucket_ranges(config: Zonos2Config) -> list[tuple[float, float | None]]:
    """Parse and validate the speaking-rate ranges of the checkpoint.

    The ranges must tile ``[0, inf)``. They must be contiguous and ordered, they
    must start at 0, and the last one must be open. If not, a rate can fall in a
    gap.
    """
    ranges = [_parse_speaking_rate_bucket(spec) for spec in config.speaking_rate_buckets]
    if not ranges:
        return ranges

    first_low, _ = ranges[0]
    if not math.isclose(first_low, 0.0, abs_tol=1e-9):
        raise ValueError("speaking-rate buckets must start at 0.")

    previous_high: float | None = None
    for idx, (low, high) in enumerate(ranges):
        if low < 0.0:
            raise ValueError("speaking-rate buckets must use non-negative ranges.")
        if high is not None and high <= low:
            raise ValueError(f"speaking-rate bucket {idx} has an empty or inverted range.")
        if previous_high is None and idx > 0:
            raise ValueError(
                "speaking-rate buckets cannot define ranges after an open-ended bucket."
            )
        if previous_high is not None and not math.isclose(low, previous_high, abs_tol=1e-9):
            raise ValueError("speaking-rate buckets must be contiguous and ordered.")
        previous_high = high

    if ranges[-1][1] is not None:
        raise ValueError("speaking-rate buckets must end with an open-ended range like '60+'.")
    return ranges


def _speaking_rate_bucket_for_rate(
    rate_bytes_per_second: float,
    *,
    num_buckets: int,
    ranges: list[tuple[float, float | None]],
) -> int:
    if rate_bytes_per_second <= 0:
        raise ValueError("speaking_rate must be positive.")

    if ranges:
        for idx, (_, high) in enumerate(ranges):
            if high is None or (
                rate_bytes_per_second < high
                and not math.isclose(rate_bytes_per_second, high, rel_tol=1e-12, abs_tol=1e-9)
            ):
                return idx
        return len(ranges) - 1

    # There are no explicit ranges. The buckets tile [0, 1) bytes-per-frame
    # uniformly.
    rate_bytes_per_frame = rate_bytes_per_second / _SPEAKING_RATE_FPS
    bucket = int(rate_bytes_per_frame * num_buckets)
    return min(max(bucket, 0), num_buckets - 1)


def _neutral_speaking_rate_bytes_per_second(
    ranges: list[tuple[float, float | None]],
) -> float:
    """Return the rate that a ``speed`` multiplier of 1.0 gives.

    This is the middle bucket.
    """
    if not ranges:
        return _DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND

    low, high = ranges[len(ranges) // 2]
    if high is None:
        return max(low, _DEFAULT_SPEAKING_RATE_BYTES_PER_SECOND)
    return (low + high) / 2.0


def resolve_speaking_rate_bucket(
    config: Zonos2Config,
    *,
    speaking_rate_bucket: int | None = None,
    speaking_rate: float | None = None,
    speed: float | None = None,
) -> int | None:
    """Resolve a speaking-rate bucket index, or return ``None`` for no token.

    Give exactly one of these: an explicit bucket index, a rate in
    bytes/second, or a ``speed`` multiplier of the model's neutral rate.
    """
    supplied = [
        speaking_rate_bucket is not None,
        speaking_rate is not None,
        speed is not None,
    ]
    if sum(supplied) == 0:
        return None
    if sum(supplied) > 1:
        raise ValueError("Provide only one of speaking_rate_bucket, speaking_rate, or speed.")

    num_buckets = config.speaking_rate_num_buckets
    if num_buckets <= 0:
        # A bare ``speed`` on a checkpoint without rate buckets does nothing,
        # and it is not an error. The reference behaves the same way, because
        # speed is advisory but a bucket is not.
        if speed is not None:
            return None
        raise ValueError("Current model does not support speaking-rate conditioning.")

    if speaking_rate_bucket is not None:
        bucket = int(speaking_rate_bucket)
        if bucket < 0 or bucket >= num_buckets:
            raise ValueError(
                f"speaking_rate_bucket must be in [0, {num_buckets - 1}], got {bucket}."
            )
        return bucket

    ranges = _speaking_rate_bucket_ranges(config)
    if ranges and len(ranges) != num_buckets:
        raise ValueError(
            f"Model has {num_buckets} speaking-rate buckets, but config defines "
            f"{len(ranges)} ranges."
        )

    if speaking_rate is not None:
        return _speaking_rate_bucket_for_rate(
            float(speaking_rate), num_buckets=num_buckets, ranges=ranges,
        )

    assert speed is not None
    speed_value = float(speed)
    if speed_value <= 0:
        raise ValueError("speed must be positive.")
    return _speaking_rate_bucket_for_rate(
        _neutral_speaking_rate_bytes_per_second(ranges) * speed_value,
        num_buckets=num_buckets,
        ranges=ranges,
    )


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


def _parse_quality_bucket(spec: str) -> tuple[str, float, float | None]:
    value = str(spec)
    exact = _QUALITY_EXACT_BUCKET_RE.match(value)
    if exact is not None:
        return "exact", float(exact.group(1)), None

    closed = _QUALITY_CLOSED_BUCKET_RE.match(value)
    if closed is not None:
        return "range", float(closed.group(1)), float(closed.group(2))

    open_ended = _QUALITY_OPEN_BUCKET_RE.match(value)
    if open_ended is not None:
        return "range", float(open_ended.group(1)), None

    raise ValueError(
        f"Invalid quality bucket {spec!r}; expected exact values like '0', "
        "ranges like '-30--25', or open-ended ranges like '22050+'."
    )


def _quality_bucket_specs(
    config: Zonos2Config, feature: str,
) -> list[tuple[str, float, float | None]]:
    raw = (config.quality_buckets or {}).get(feature, ())
    specs = [_parse_quality_bucket(spec) for spec in raw]
    for idx, (kind, low, high) in enumerate(specs):
        if not math.isfinite(low):
            raise ValueError(f"quality_buckets.{feature} must use finite bucket values.")
        if kind == "range":
            if high is not None and not math.isfinite(high):
                raise ValueError(f"quality_buckets.{feature} must use finite bucket values.")
            if high is not None and high <= low:
                raise ValueError(
                    f"quality_buckets.{feature} has an empty or inverted range at index {idx}."
                )
    return specs


def _quality_bucket_for_value(value: Any, config: Zonos2Config, feature: str) -> int | None:
    """Find the bucket of a raw metric value.

    Quality buckets need not tile the line, unlike the speaking-rate buckets. A
    value outside every range clamps to the first or the last range.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None

    specs = _quality_bucket_specs(config, feature)
    if not specs:
        return None

    for idx, (kind, low, _) in enumerate(specs):
        if kind == "exact" and math.isclose(value, low, rel_tol=1e-12, abs_tol=1e-9):
            return idx

    range_indexes = [idx for idx, (kind, _, _) in enumerate(specs) if kind == "range"]
    if not range_indexes:
        return None

    for idx in range_indexes:
        _, low, high = specs[idx]
        if high is None:
            if value >= low:
                return idx
        elif idx == range_indexes[-1]:
            if low <= value <= high:
                return idx
        elif low <= value < high:
            return idx

    _, first_low, _ = specs[range_indexes[0]]
    if value < first_low:
        return range_indexes[0]
    return range_indexes[-1]


def _quality_control_to_feature_list(value: Any, features: tuple[str, ...]) -> list[Any]:
    if value is None:
        return [None] * len(features)
    if isinstance(value, Mapping):
        return [value.get(feature) for feature in features]
    if isinstance(value, (list, tuple)):
        return [value[idx] if idx < len(value) else None for idx in range(len(features))]
    raise ValueError("quality_buckets and quality_values must be a list or feature-name object.")


def resolve_quality_buckets(
    config: Zonos2Config,
    *,
    quality_buckets: Any = None,
    quality_values: Any = None,
) -> list[int | None] | None:
    """Resolve the quality bucket index of each feature, or ``None`` for no tokens.

    Give either explicit bucket indices or raw metric values (``lufs``,
    ``trailing_silence_s``, and others). Key them by feature name, or give them
    in ``config.quality_features`` order. A ``None`` entry emits no token for
    that feature.
    """
    if quality_buckets is None and quality_values is None:
        return None
    if quality_buckets is not None and quality_values is not None:
        raise ValueError("Provide only one of quality_buckets or quality_values.")

    features = config.quality_features
    counts = config.quality_bucket_counts
    if not features or sum(counts) <= 0:
        raise ValueError("Current model does not support quality conditioning.")
    if any(count <= 0 for count in counts):
        raise ValueError("Every configured quality feature must define at least one bucket.")

    if quality_buckets is not None:
        raw_buckets = _quality_control_to_feature_list(quality_buckets, features)
        resolved: list[int | None] = []
        for feature, count, raw_bucket in zip(features, counts, raw_buckets, strict=True):
            if raw_bucket is None:
                resolved.append(None)
                continue
            bucket = int(raw_bucket)
            if bucket < 0 or bucket >= count:
                raise ValueError(
                    f"quality_buckets.{feature} must be in [0, {count - 1}], got {bucket}."
                )
            resolved.append(bucket)
        return resolved

    raw_values = _quality_control_to_feature_list(quality_values, features)
    return [
        _quality_bucket_for_value(raw_value, config, feature) if raw_value is not None else None
        for feature, raw_value in zip(features, raw_values, strict=True)
    ]
