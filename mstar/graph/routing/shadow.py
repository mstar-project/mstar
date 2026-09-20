"""Run two routers side by side and report where they disagree.

``primary`` answers — enabling the shadow cannot change behavior. The other
backend is asked for the same batch and its routing compared field by field.

A router MUTATES state (the completion advances every request's walk, and
locally-destined edges are ingested), so unlike the graph-runtime shadow the
two cannot both run against the same world. The shadow router therefore only
engages when the backend keeps its own state — which is exactly the Rust
case, where the walk state lives in the extension.
"""
import logging
import os
from collections import Counter

from mstar.graph.routing.base import BatchRouterBase
from mstar.graph.runtime.shadow import DivergenceReport

logger = logging.getLogger(__name__)


def _edges(lst) -> Counter:
    """Edges as a comparable multiset. Includes the sharding fields, which is
    where a fanout planned from the wrong config would show up."""
    return Counter(
        (e.name, e.next_node, e.persist, e.conductor_new_token, e.is_streaming,
         e._shard_dim, e._total_fanin, tuple(i.uuid for i in e.tensor_info))
        for e in lst
    )


def _routing_key(routing) -> dict:
    return {
        "routed_local": _edges(routing.routed_to_this_worker_graph),
        "persist": _edges(routing.persist),
        "emit": _edges(routing.emit_to_client),
        "new_token": _edges(routing.new_token_outputs),
        "streaming_local": _edges(routing.streaming_local),
        "completed": tuple(sorted(routing.completed_worker_graph_ids)),
        "is_first_tp_rank": routing.is_first_tp_rank,
        "to_workers": {w: _edges(es) for w, es in routing.to_workers.items()},
        "streaming_to_workers": {
            w: _edges(es) for w, es in routing.streaming_to_workers.items()
        },
    }


class ShadowBatchRouter(BatchRouterBase):
    backend = "shadow"

    def __init__(
        self,
        primary: BatchRouterBase,
        shadow: BatchRouterBase,
        strict: bool | None = None,
        report: DivergenceReport | None = None,
    ):
        super().__init__(primary.world)
        self.primary = primary
        self.shadow = shadow
        self.strict = (
            os.getenv("MSTAR_RUST_GRAPH_STRICT", "0") == "1" if strict is None else strict
        )
        self.report = report if report is not None else DivergenceReport()
        self.enabled = True

    def route_batch(self, node_name, request_to_worker_graph, graph_walk):
        out = self.primary.route_batch(node_name, request_to_worker_graph, graph_walk)
        if not self.enabled:
            return out
        try:
            other = self.shadow.route_batch(
                node_name, request_to_worker_graph, graph_walk
            )
        except Exception as exc:  # noqa: BLE001 - the shadow must never break the worker
            self._diverge(node_name, f"shadow raised: {exc!r}")
            return out

        if sorted(out) != sorted(other):
            self._diverge(
                node_name,
                f"routed different requests: {sorted(out)} != {sorted(other)}",
            )
            return out
        for rid in out:
            a, b = _routing_key(out[rid]), _routing_key(other[rid])
            if a != b:
                diffs = [f"{k}: {a[k]!r} != {b[k]!r}" for k in a if a[k] != b[k]]
                self._diverge(node_name, f"rid={rid}: " + "; ".join(diffs))
                break
        return out

    def _diverge(self, node_name: str, detail: str) -> None:
        msg = self.report.record("route", node_name, detail)
        full = (
            f"batch router divergence ({self.primary.backend} vs "
            f"{self.shadow.backend}) {msg}"
        )
        if self.strict:
            raise AssertionError(full)
        logger.error("%s; disabling the shadow router", full)
        self.enabled = False
