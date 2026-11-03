"""What each worker checks in its environment at startup: the Rust extension
and the knob that picks its wire codec. Workers inherit the conductor's
environment, so the conductor checks these once, before any rank starts."""

from collections.abc import Iterator

from mstar.communication.codec import default_codec
from mstar.graph.runtime.utils import require_rust_extension


def check_knobs() -> Iterator[str]:
    try:
        require_rust_extension()
    except ImportError as e:
        yield str(e)
    try:
        default_codec()
    except ValueError as e:
        yield str(e)
