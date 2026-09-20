"""Graph-runtime harness: one interface, swappable backend.

``MSTAR_RUST_GRAPH`` selects it:

===========  ==========================================================
``0``        Python only. The default; nothing Rust is constructed.
``shadow``   Python answers, Rust mirrors, divergences are logged.
``1``        Rust answers. Python is not constructed.
===========  ==========================================================

``MSTAR_RUST_GRAPH_STRICT=1`` turns a shadow divergence into a raise.

Any worker graph the Rust core cannot compile falls back to Python for that
graph alone, with a logged reason — a model using an unmodelled construct
degrades rather than failing.
"""
import logging
import os

from mstar.graph.base import GraphSection
from mstar.graph.runtime.base import BatchRouting, GraphRuntimeBase, RequestSnapshot
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.runtime.rust import RustGraphRuntime, RustWorkerRuntimes, rust_available
from mstar.graph.runtime.shadow import DivergenceReport, ShadowGraphRuntime

logger = logging.getLogger(__name__)

__all__ = [
    "BatchRouting",
    "DivergenceReport",
    "GraphRuntimeBase",
    "PythonGraphRuntime",
    "RequestSnapshot",
    "RustGraphRuntime",
    "RustWorkerRuntimes",
    "install_graph_runtimes",
    "ShadowGraphRuntime",
    "graph_runtime_mode",
    "make_graph_runtime",
    "rust_available",
]

MODE_PYTHON = "0"
MODE_SHADOW = "shadow"
MODE_RUST = "1"


def graph_runtime_mode() -> str:
    mode = os.getenv("MSTAR_RUST_GRAPH", MODE_PYTHON).lower()
    if mode in ("", "0", "off", "false", "python"):
        return MODE_PYTHON
    if mode in ("shadow",):
        return MODE_SHADOW
    if mode in ("1", "on", "true", "rust"):
        return MODE_RUST
    logger.warning("Unknown MSTAR_RUST_GRAPH=%r; using python", mode)
    return MODE_PYTHON


def make_graph_runtime(
    section: GraphSection,
    wg_id: str,
    worker_id: str = "",
    workers: list[str] | None = None,
    leader_nodes: set[str] | None = None,
    tensor_manager=None,
    sharding_config=None,
    mode: str | None = None,
    report: DivergenceReport | None = None,
    io_store: dict | None = None,
) -> GraphRuntimeBase:
    """Build the runtime for one worker graph, per ``MSTAR_RUST_GRAPH``."""
    mode = graph_runtime_mode() if mode is None else mode

    def _python() -> PythonGraphRuntime:
        return PythonGraphRuntime(
            section, wg_id, tensor_manager=tensor_manager,
            sharding_config=sharding_config, worker_id=worker_id,
            io_store=io_store,
        )

    if mode == MODE_PYTHON:
        return _python()

    shared = RustWorkerRuntimes.build(
        {wg_id: section}, worker_id=worker_id,
        graph_walks={wg_id: set()}, workers=workers, leader_nodes=leader_nodes,
    )
    rust = None if shared is None else shared.view(wg_id, section)
    if rust is None:
        logger.warning(
            "Falling back to the Python graph runtime for worker graph %s", wg_id
        )
        return _python()

    if mode == MODE_RUST:
        return rust
    return ShadowGraphRuntime(_python(), rust, report=report)


def install_graph_runtimes(
    queues: dict,
    worker_id: str,
    workers: list[str] | None = None,
    leader_nodes: set[str] | None = None,
    mode: str | None = None,
    report: DivergenceReport | None = None,
) -> None:
    """Give every worker graph on this worker its runtime.

    Under a Rust-backed mode the graphs share ONE extension object, because
    routing needs to see which local graph owns a destination node — so they
    cannot be built independently. Each queue gets a per-graph view, and the
    harness interface each of them exposes is unchanged.

    Under the Python mode this is a no-op: the queues already built their own.
    """
    mode = graph_runtime_mode() if mode is None else mode
    if mode == MODE_PYTHON:
        return

    sections = {wg_id: q.worker_graph.section for wg_id, q in queues.items()}
    walks = {wg_id: set(q.graph_walks) for wg_id, q in queues.items()}
    shared = RustWorkerRuntimes.build(
        sections, worker_id=worker_id, graph_walks=walks,
        workers=workers, leader_nodes=leader_nodes,
    )
    if shared is None:
        logger.warning(
            "Rust graph runtime unavailable for this worker's graphs; staying on Python"
        )
        return

    for wg_id, queue in queues.items():
        rust = shared.view(wg_id, sections[wg_id])
        if mode == MODE_RUST:
            queue.runtime = rust
        else:
            queue.runtime = ShadowGraphRuntime(
                PythonGraphRuntime(
                    sections[wg_id], wg_id, tensor_manager=queue.tensor_manager,
                    worker_id=worker_id, io_store=queue.per_request_queues,
                ),
                rust, report=report,
            )
