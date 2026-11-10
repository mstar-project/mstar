"""/v1/images/generations (text-to-image) and /v1/images/edits (image-to-image) handlers."""

from __future__ import annotations

import os
import re

DEFAULT_MAX_IMAGES_PER_REQUEST = 10


def max_images_per_request() -> int:
    """Largest ``n`` one image request may ask for: ``MSTAR_MAX_IMAGES_PER_REQUEST``,
    default 10 (OpenAI's range for the endpoint), at least 1. The Rust frontend
    parses it the same way: an ASCII integer, surrounding whitespace allowed,
    anything else the default."""
    raw = os.environ.get("MSTAR_MAX_IMAGES_PER_REQUEST", "").strip()
    if not re.fullmatch(r"[+-]?[0-9]+", raw):
        return DEFAULT_MAX_IMAGES_PER_REQUEST
    return max(1, int(raw))


def image_count(req) -> int:
    """The request's ``n``, 1 when omitted. Outside ``1..max_images_per_request()``
    it raises ``ValueError`` (a 400), before anything is submitted."""
    n = getattr(req, "n", None)
    if n is None:
        return 1
    limit = max_images_per_request()
    if not 1 <= n <= limit:
        raise ValueError(f"n must be between 1 and {limit}, got {n}")
    return n
