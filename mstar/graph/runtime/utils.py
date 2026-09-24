
import logging
import os
from enum import IntEnum

logger = logging.getLogger(__name__)


class GraphRuntimeType(IntEnum):
    PYTHON = 0
    RUST = 1


def resolve_graph_runtime_type(log: bool=True) -> GraphRuntimeType:
    requested = os.getenv("MSTAR_RUST_GRAPH", "AUTO")
    if requested not in ("0", "1", "AUTO"):
        raise ValueError(
            f"MSTAR_RUST_GRAPH must be 0, 1, or AUTO; got {requested!r}"
        )
    choice = requested
    if choice == "AUTO":
        try:
            import mstar.graph.runtime.rust  # noqa: F401
        except ImportError:
            choice = "0"
        else:
            # The Rust runtime holds an Arc to the Rust transport and refuses
            # a pyzmq one, so AUTO must not select it where MSTAR_RUST_ZMQ
            # pins pyzmq: that pairing is a worker that raises at startup, and
            # the transport flag is documented as safe to set per process.
            # Explicit MSTAR_RUST_GRAPH=1 still raises there -- asking for
            # both is a contradiction, and a silent downgrade would hide it.
            #
            # Read here rather than asked of make_communicator, which would
            # build one. AUTO there means the same Rust transport as far as
            # this decision goes: what makes it fall back is the extension
            # failing to import, which the branch above has just ruled out.
            if os.getenv("MSTAR_RUST_ZMQ", "AUTO").upper() == "0":
                logger.warning(
                    "MSTAR_RUST_GRAPH=AUTO selected the Python graph runtime: "
                    "MSTAR_RUST_ZMQ=0 pins the pyzmq transport, which the Rust "
                    "runtime cannot share. Set MSTAR_RUST_ZMQ=1 for the Rust "
                    "one."
                )
                choice = "0"
            else:
                choice = "1"
    resolved = GraphRuntimeType(int(choice))
    # Both halves: a support bundle has to show what the worker was asked for
    # as well as what it ended up on, which AUTO makes different questions.
    logger.info(
        "graph runtime: %s (MSTAR_RUST_GRAPH=%s)",
        resolved.name.lower(), requested,
    )
    return resolved
