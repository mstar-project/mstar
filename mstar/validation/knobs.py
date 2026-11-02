"""The environment knobs each worker reads at startup to pick its transport and
wire codec. Workers inherit the conductor's environment."""

from collections.abc import Iterator

from mstar.communication.codec import default_codec
from mstar.communication.communicator import requested_transport


def check_knobs() -> Iterator[str]:
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
