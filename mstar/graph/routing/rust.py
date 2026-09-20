"""Rust-backed router — not yet able to serve; ``build`` returns None.

This file is the plug point, and the reason it is empty is measured rather
than assumed. Two things have to land before a Rust router can beat the
Python one, and neither is about the Rust:

**1. The return shape has to stop being per-request Python objects.**
``NodeOutputRouting`` holds ``GraphEdge`` objects with real
``TensorPointerInfo`` lists. Constructing those is **27% of the batched
routing path** (measure: stub ``GraphEdge.clone`` and re-run
``docs/design/rust_graph_port/bench_postprocess.py`` — 11695 -> 8574
ns/rid/completion). A Rust backend that hands this shape back pays that 27%
plus a boundary crossing plus rebuilding the tensor infos, so it starts
behind. ``_send_outputs`` / ``_register_outputs`` need to consume flat,
batch-shaped data first — per-worker edge groups and flat ``(rid, uuid)``
arrays, which is what ``mstar_rust.BatchRouting`` already returns.

**2. The Rust runtime has to be per-worker, not per-worker-graph.**
Routing needs what one worker graph cannot see: which *local* worker graph
owns a destination node (so the edge is ingested rather than sent), the
request's ``ShardingConfig``, and the cross-worker fanout. Today
``mstar.graph.runtime.RustGraphRuntime`` compiles one section and owns only
its own states, and ``shard.rs``'s ``ShardMap`` is built from a single
trivial group rather than from the real config.

Until then the factory falls back to Python, and the worker is unaffected.
See docs/design/rust_graph_port §11.
"""
import logging

from mstar.graph.routing.base import BatchRouterBase, RoutingWorld

logger = logging.getLogger(__name__)


class RustBatchRouter(BatchRouterBase):
    backend = "rust"

    @classmethod
    def build(cls, world: RoutingWorld) -> "RustBatchRouter | None":
        """None until the prerequisites in this module's docstring land."""
        logger.info(
            "Rust batch router not available: routing still returns per-request "
            "NodeOutputRouting objects, and the Rust runtime is scoped to a "
            "single worker graph. See docs/design/rust_graph_port §11."
        )
        return None

    def route_batch(self, node_name, request_to_worker_graph, graph_walk):
        raise NotImplementedError(
            "RustBatchRouter.build() returns None; nothing constructs this yet"
        )
