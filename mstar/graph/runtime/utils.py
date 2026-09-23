
from enum import IntEnum
import logging
import os


logger = logging.getLogger(__name__)


class GraphRuntimeType(IntEnum):
    PYTHON = 0
    RUST = 1


def resolve_graph_runtime_type(log: bool=True) -> GraphRuntimeType:
    choice = os.getenv("MSTAR_RUST_GRAPH", "AUTO")
    if choice not in ("0", "1", "AUTO"):
        raise ValueError(f"MSTAR_RUST_GRAPH must be 0, 1, or AUTO; got {choice!r}")
    if choice == "AUTO":
        try:
            import mstar.graph.runtime.rust  # noqa: F401
        except ImportError:
            choice = "0"
        else:
            choice = "1"
    logger.info("Running with graph runtime %s (MSTAR_RUST_GRAPH=%s)", choice, choice)

    return GraphRuntimeType(int(choice))
