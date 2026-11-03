"""What each worker checks in its environment at startup: the Rust extension,
and the knobs that pick its transport and wire codec. Workers inherit the
conductor's environment, so the conductor checks these once, before any rank
starts."""

from collections.abc import Iterator

from mstar.communication.codec import default_codec
from mstar.communication.communicator import requested_transport
from mstar.graph.runtime.utils import require_rust_extension


def check_knobs() -> Iterator[str]:
    try:
        require_rust_extension()
    except ImportError as e:
        yield str(e)
    transport = None
    try:
        transport = requested_transport()
    except ValueError as e:
        yield str(e)
    try:
        default_codec()
    except ValueError as e:
        yield str(e)
    if transport == "0":
        yield (
            "MSTAR_RUST_ZMQ=0 pins the pyzmq communicator, which the graph runtime "
            "cannot run on. Set MSTAR_RUST_ZMQ to 1 or AUTO."
        )
