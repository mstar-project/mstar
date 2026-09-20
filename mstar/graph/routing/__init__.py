"""Batch-router harness — see ``mstar.graph.routing.base``.

Selected by ``MSTAR_RUST_GRAPH`` alongside the graph runtime, since a Rust
router needs the Rust runtime's state to route against:

===========  ==========================================================
``0``        Python routing. The default.
``shadow``   Python routes; a Rust router is asked for the same batch
             and the two are compared. Behavior cannot change.
``1``        Rust routes.
===========  ==========================================================

``shadow`` and ``1`` fall back to Python, with a logged reason, whenever the
Rust router cannot serve the worker's configuration.
"""
import logging

from mstar.graph.routing.base import BatchRouterBase, RoutingWorld
from mstar.graph.routing.python import PythonBatchRouter
from mstar.graph.routing.shadow import ShadowBatchRouter
from mstar.graph.runtime import MODE_PYTHON, MODE_RUST, graph_runtime_mode
from mstar.graph.runtime.shadow import DivergenceReport

logger = logging.getLogger(__name__)

__all__ = [
    "BatchRouterBase",
    "PythonBatchRouter",
    "RoutingWorld",
    "ShadowBatchRouter",
    "make_batch_router",
]


def make_batch_router(
    world: RoutingWorld,
    mode: str | None = None,
    report: DivergenceReport | None = None,
) -> BatchRouterBase:
    """Build the router for one worker, per ``MSTAR_RUST_GRAPH``."""
    mode = graph_runtime_mode() if mode is None else mode
    python = PythonBatchRouter(world)
    if mode == MODE_PYTHON:
        return python

    from mstar.graph.routing.rust import RustBatchRouter

    rust = RustBatchRouter.build(world)
    if rust is None:
        logger.warning(
            "Rust batch router unavailable for this worker; routing in Python"
        )
        return python
    if mode == MODE_RUST:
        return rust
    return ShadowBatchRouter(python, rust, report=report)
