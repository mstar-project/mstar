"""Small shared helpers for the OpenAI-compatible layer."""

from __future__ import annotations


def check_inline_result(response_format, n=1) -> None:
    """Refuse what an image or video route would otherwise drop: results come
    back inline as ``b64_json`` (there is no store to serve a ``url`` from),
    and a route that runs one request returns one result."""
    if response_format not in (None, "", "b64_json"):
        raise ValueError(f"response_format {response_format!r} is not served; results are returned as b64_json")
    if n not in (None, 1):
        raise ValueError(f"n={n}: this route returns one result per request")
