"""Piecewise-capture telemetry, shared by encoders.

A captured replay and a silent eager fallback produce identical results, so the
path taken is counted rather than inferred; ``encoder_path_counts()`` lets a
benchmark assert it measured the path it meant to.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ENCODER_PATH_COUNTS: dict[str, int] = {}
_SEEN_LAYOUTS: set[tuple] = set()
_SEEN_LAYOUTS_CAP = 512          # bounded: a long run must not leak keys
_WARNED_NO_BUCKET: set[str] = set()



def note_encoder_path(path: str) -> None:
    _ENCODER_PATH_COUNTS[path] = _ENCODER_PATH_COUNTS.get(path, 0) + 1


def encoder_path_counts() -> dict[str, int]:
    """Snapshot of {path: count}, e.g. {"vision.piecewise": 96, "vision.eager": 0}."""
    return dict(_ENCODER_PATH_COUNTS)


def note_encoder_layout(kind: str, n_seg: int, total_tokens: int, fitted: bool) -> None:
    """Log once per distinct layout; WARN the first time one fits no bucket —
    the only visible signal that buckets are mis-sized, since the fallback is
    otherwise silent."""
    if not fitted and kind not in _WARNED_NO_BUCKET:
        _WARNED_NO_BUCKET.add(kind)
        logger.warning(
            "%s encoder: NO capture bucket fits segments=%d total_tokens=%d — "
            "falling back to eager. Widen CAPTURE_BATCH_SIZES_%s / "
            "CAPTURE_TOKENS_%s; piecewise numbers from this run are "
            "NOT measuring the captured path.",
            kind, n_seg, total_tokens, kind.upper(), kind.upper(),
        )
    key = (kind, n_seg, total_tokens)
    if key in _SEEN_LAYOUTS:
        return
    if len(_SEEN_LAYOUTS) < _SEEN_LAYOUTS_CAP:
        _SEEN_LAYOUTS.add(key)
    logger.info("%s encoder layout: segments=%d total_tokens=%d bucket=%s",
                kind, n_seg, total_tokens, "HIT" if fitted else "MISS")


