"""The environment knobs each worker reads at startup to pick its graph runtime,
transport and wire codec, and the YAML's ``mps:`` switch, which depends on the
platform. Workers inherit the conductor's environment."""

from collections.abc import Iterator, Mapping
from typing import Any

from mstar.communication.codec import default_codec
from mstar.communication.communicator import requested_transport
from mstar.graph.runtime.utils import requested_graph_runtime
from mstar.platforms import current_platform


def check_knobs() -> Iterator[str]:
    runtime = transport = None
    try:
        runtime = requested_graph_runtime()
    except ValueError as e:
        yield str(e)
    try:
        transport = requested_transport()
    except ValueError as e:
        yield str(e)
    try:
        default_codec()
    except ValueError as e:
        yield str(e)
    if runtime == "1" and transport == "0":
        # AUTO falls back to the Python runtime here; an explicit 1 is a contradiction
        yield (
            "MSTAR_RUST_GRAPH=1 needs the Rust communicator, but MSTAR_RUST_ZMQ=0 "
            "pins pyzmq. Set MSTAR_RUST_ZMQ to 1 or AUTO, or MSTAR_RUST_GRAPH to 0 or AUTO."
        )


def check_mps(config: Mapping[str, Any]) -> Iterator[str]:
    """``mps: true`` runs the workers as clients of a CUDA MPS daemon."""
    mps = config.get("mps", False)
    if not isinstance(mps, bool):
        yield f"mps must be true or false, got {mps!r}"
    elif mps and not current_platform.is_cuda():
        yield f"mps: true needs an NVIDIA GPU, not {current_platform!r}"
