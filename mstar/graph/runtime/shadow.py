"""Run two graph runtimes side by side and report where they disagree.

Python stays authoritative: every return value the worker sees comes from it,
so enabling the shadow cannot change behavior. The other backend is driven
through the identical event sequence and its state compared after each one.

``MSTAR_RUST_GRAPH_STRICT=1`` raises on the first divergence instead of
logging — what a parity test wants. Without it, a diverged request is
detached (its shadow is dropped, once, with a logged reason) so one bad
request does not spam every subsequent event.
"""
import logging
import os
from collections import Counter
from dataclasses import dataclass, field

from mstar.graph.base import (
    GraphEdge,
    NameAndDest,
    NodeCompletionOutput,
    SpeculativeNodeInfo,
)
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.graph.runtime.base import COMPARED_FIELDS, BatchRouting, GraphRuntimeBase

logger = logging.getLogger(__name__)


@dataclass
class DivergenceReport:
    count: int = 0
    detached: set[str] = field(default_factory=set)
    first: str | None = None
    by_kind: Counter = field(default_factory=Counter)
    #: Divergences attributable to a known Python bug rather than to the
    #: shadow being wrong. Counted and sampled, never fatal.
    expected: Counter = field(default_factory=Counter)
    expected_samples: dict[str, str] = field(default_factory=dict)

    def record(self, kind: str, rid: str, detail: str) -> str:
        self.count += 1
        self.by_kind[kind] += 1
        msg = f"[{kind}] rid={rid}: {detail}"
        if self.first is None:
            self.first = msg
        return msg

    def record_expected(self, kind: str, detail: str) -> None:
        self.expected[kind] += 1
        self.expected_samples.setdefault(kind, detail)

    def clean(self) -> bool:
        return self.count == 0

    def summary(self) -> str:
        parts = [f"{self.count} divergence(s)"]
        if self.by_kind:
            parts.append("by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(self.by_kind.items())))
        if self.expected:
            parts.append("expected: " + ", ".join(f"{k}={v}" for k, v in sorted(self.expected.items())))
        if self.detached:
            parts.append(f"{len(self.detached)} request(s) detached")
        return "; ".join(parts)


def _edge_keys(edges: list[GraphEdge]) -> Counter:
    """Edges as a comparable multiset. Tensor identity is excluded — the Rust
    core carries handles and Python owns the uuids."""
    return Counter(
        (e.name, e.next_node, e.persist, e.conductor_new_token, e.is_streaming)
        for e in edges
    )


class ShadowGraphRuntime(GraphRuntimeBase):
    """``primary`` answers; ``shadow`` is mirrored and compared."""

    backend = "shadow"

    def __init__(
        self,
        primary: GraphRuntimeBase,
        shadow: GraphRuntimeBase,
        strict: bool | None = None,
        report: DivergenceReport | None = None,
        fields: tuple[str, ...] = COMPARED_FIELDS,
        allow_stale_ready: bool = True,
    ):
        self.primary = primary
        self.shadow = shadow
        self.fields = fields
        self.allow_stale_ready = allow_stale_ready
        self.strict = (
            os.getenv("MSTAR_RUST_GRAPH_STRICT", "0") == "1" if strict is None else strict
        )
        self.report = report if report is not None else DivergenceReport()
        self._live: set[str] = set()

    # -- comparison ---------------------------------------------------------

    def _diverge(self, kind: str, rid: str, detail: str) -> None:
        msg = self.report.record(kind, rid, detail)
        full = (
            f"graph runtime divergence ({self.primary.backend} vs "
            f"{self.shadow.backend}) {msg}"
        )
        if self.strict:
            raise AssertionError(full)
        logger.error("%s; detaching shadow for this request", full)
        self.report.detached.add(rid)
        self._live.discard(rid)

    def _compare(self, rid: str, where: str) -> None:
        if rid not in self._live:
            return
        try:
            a, b = self.primary.snapshot(rid), self.shadow.snapshot(rid)
        except Exception as exc:  # shadow raised where primary did not
            self._diverge("snapshot", rid, f"after {where}: {exc!r}")
            return
        diffs = []
        for field_name in self.fields:
            pa, pb = getattr(a, field_name), getattr(b, field_name)
            if pa == pb:
                continue
            if field_name == "ready" and self.allow_stale_ready and pb < pa:
                # Python over-reports `ready`: a loop member that completed
                # before its loop's last entity is re-added by
                # `register_ingested_input` reading its still-full
                # ready_signals (graph/base.py:742). The extras hold
                # already-consumed inputs. `rust <= python` is the real
                # invariant; a strict superset the other way is a true bug.
                self.report.record_expected(
                    "stale_ready", f"rid={rid} after {where}: python also ready {sorted(pa - pb)}"
                )
                continue
            diffs.append(f"{field_name}: {pa!r} != {pb!r}")
        if diffs:
            self._diverge("state", rid, f"after {where}: " + "; ".join(diffs))

    # -- lifecycle ----------------------------------------------------------

    def add_requests(self, rids: list[str]) -> None:
        self.primary.add_requests(rids)
        self.shadow.add_requests(rids)
        self._live.update(rids)
        for rid in rids:
            self._compare(rid, "add_requests")

    def remove_requests(self, rids: list[str]) -> None:
        self.primary.remove_requests(rids)
        self.shadow.remove_requests(rids)
        self._live.difference_update(rids)
        self.report.detached.difference_update(rids)

    def has_request(self, rid: str) -> bool:
        return self.primary.has_request(rid)

    def known_requests(self) -> list[str]:
        return self.primary.known_requests()

    # -- ingest -------------------------------------------------------------

    def ingest(self, rid: str, edges: list[GraphEdge], can_buffer: bool = True) -> list[GraphEdge]:
        left = self.primary.ingest(rid, edges, can_buffer)
        if rid in self._live:
            shadow_left = self.shadow.ingest(rid, edges, can_buffer)
            if _edge_keys(left) != _edge_keys(shadow_left):
                self._diverge(
                    "ingest", rid,
                    f"unclaimed edges differ: {sorted(_edge_keys(left))} != "
                    f"{sorted(_edge_keys(shadow_left))}",
                )
            self._compare(rid, "ingest")
        return left

    # -- completion ---------------------------------------------------------

    def complete(self, rid: str, node_name: str) -> NodeCompletionOutput:
        out = self.primary.complete(rid, node_name)
        if rid in self._live:
            shadow_out = self.shadow.complete(rid, node_name)
            if _edge_keys(out.output_edges) != _edge_keys(shadow_out.output_edges):
                self._diverge(
                    "complete", rid,
                    f"{node_name} routed {sorted(_edge_keys(out.output_edges))} != "
                    f"{sorted(_edge_keys(shadow_out.output_edges))}",
                )
            elif out.filtered_signals != shadow_out.filtered_signals:
                self._diverge(
                    "complete", rid,
                    f"{node_name} filtered {sorted(out.filtered_signals)} != "
                    f"{sorted(shadow_out.filtered_signals)}",
                )
            self._compare(rid, f"complete({node_name})")
        return out

    def complete_and_route_batch(
        self, node_name: str, rids: list[str], graph_walk: str,
    ) -> BatchRouting:
        out = self.primary.complete_and_route_batch(node_name, rids, graph_walk)
        live = [r for r in rids if r in self._live]
        if live:
            shadow_out = self.shadow.complete_and_route_batch(node_name, live, graph_walk)
            if set(out.completed) & set(live) != set(shadow_out.completed):
                self._diverge(
                    "batch", live[0],
                    f"{node_name} completed {sorted(set(out.completed) & set(live))} != "
                    f"{sorted(shadow_out.completed)}",
                )
            for rid in live:
                self._compare(rid, f"complete_and_route_batch({node_name})")
        return out

    # -- scheduling ---------------------------------------------------------

    def ready_nodes(self, rid: str) -> set[str]:
        return self.primary.ready_nodes(rid)

    def ready_for_streaming(self, rid: str) -> set[str]:
        return self.primary.ready_for_streaming(rid)

    def input_slots(self, rid, node_name, next_iter=False):
        slots = self.primary.input_slots(rid, node_name, next_iter)
        if rid in self._live:
            other = self.shadow.input_slots(rid, node_name, next_iter)
            # uuid identity is Python's; the shadow must agree on which
            # signals arrived and how many tensors each carries.
            shape = {n: (len(s.uuids), s.final_stream_chunk) for n, s in slots.items()}
            other_shape = {n: (len(s.uuids), s.final_stream_chunk) for n, s in other.items()}
            if shape != other_shape:
                self._diverge(
                    "input_slots", rid,
                    f"{node_name}(next_iter={next_iter}): {shape} != {other_shape}",
                )
        return slots

    def pop_ready(self, node_name: str, rids: list[str]) -> list[str]:
        popped = self.primary.pop_ready(node_name, rids)
        live = [r for r in rids if r in self._live]
        if live:
            shadow_popped = self.shadow.pop_ready(node_name, live)
            expected = [r for r in popped if r in self._live]
            if expected != shadow_popped:
                # Same stale-ready asymmetry as in _compare: Python can pop a
                # rid the shadow never considered ready. The other direction
                # is a real divergence.
                if self.allow_stale_ready and set(shadow_popped) < set(expected):
                    self.report.record_expected(
                        "stale_ready",
                        f"{node_name}: python also popped "
                        f"{sorted(set(expected) - set(shadow_popped))}",
                    )
                else:
                    self._diverge(
                        "pop_ready", live[0],
                        f"{node_name} popped {expected} != {shadow_popped}",
                    )
        return popped

    def push_back(self, node_name: str, rids: list[str]) -> None:
        self.primary.push_back(node_name, rids)
        live = [r for r in rids if r in self._live]
        if live:
            self.shadow.push_back(node_name, live)
            for rid in live:
                self._compare(rid, f"push_back({node_name})")

    # -- loops --------------------------------------------------------------

    def stop_loops(self, rid: str, loop_names: set[str]) -> set[NameAndDest]:
        signals = self.primary.stop_loops(rid, loop_names)
        if rid in self._live:
            shadow_signals = self.shadow.stop_loops(rid, loop_names)
            if signals != shadow_signals:
                self._diverge(
                    "stop_loops", rid,
                    f"{sorted(loop_names)} -> {sorted(signals)} != {sorted(shadow_signals)}",
                )
            self._compare(rid, "stop_loops")
        return signals

    def loop_indices(self, rid: str) -> dict[str, int]:
        return self.primary.loop_indices(rid)

    def nested_loop_idxs_for_node(self, rid: str, node_name: str) -> NestedLoopIndices:
        idx = self.primary.nested_loop_idxs_for_node(rid, node_name)
        if rid in self._live:
            other = self.shadow.nested_loop_idxs_for_node(rid, node_name)
            if (idx.loop_name_order, idx.wg_fwd_pass_idx) != (
                other.loop_name_order, other.wg_fwd_pass_idx
            ):
                self._diverge(
                    "nested_loop_idxs", rid,
                    f"{node_name}: {idx} != {other}",
                )
        return idx

    def nested_loop_idxs(self, rid: str, loop_name: str) -> NestedLoopIndices:
        idx = self.primary.nested_loop_idxs(rid, loop_name)
        if rid in self._live:
            other = self.shadow.nested_loop_idxs(rid, loop_name)
            if (idx.loop_name_order, idx.wg_fwd_pass_idx) != (
                other.loop_name_order, other.wg_fwd_pass_idx
            ):
                self._diverge("nested_loop_idxs", rid, f"{loop_name}: {idx} != {other}")
        return idx

    def loop_names(self) -> set[str]:
        return self.primary.loop_names()

    def is_done(self, rid: str) -> bool:
        return self.primary.is_done(rid)

    def reset(self, rid: str) -> None:
        self.primary.reset(rid)
        if rid in self._live:
            self.shadow.reset(rid)
            self._compare(rid, "reset")

    def num_times_run(self, rid: str) -> int:
        return self.primary.num_times_run(rid)

    # -- speculation --------------------------------------------------------

    def ingest_for_speculation(
        self, rid: str, edges: list[GraphEdge], source_node: str,
    ) -> list[SpeculativeNodeInfo]:
        found = self.primary.ingest_for_speculation(rid, edges, source_node)
        if rid in self._live:
            shadow_found = self.shadow.ingest_for_speculation(rid, edges, source_node)
            key = lambda xs: sorted(  # noqa: E731
                (x.node_name, x.is_new_loop_iter, x.loop_name) for x in xs
            )
            if key(found) != key(shadow_found):
                self._diverge(
                    "speculation", rid,
                    f"from {source_node}: {key(found)} != {key(shadow_found)}",
                )
        return found

    def clear_speculative_inputs(self, rid: str) -> None:
        self.primary.clear_speculative_inputs(rid)
        if rid in self._live:
            self.shadow.clear_speculative_inputs(rid)
