"""Head-to-head: the same workload under each graph-runtime backend.

Drives ``make_graph_runtime(mode=...)`` so both backends run the identical
event sequence through the identical interface — the numbers differ only by
implementation.

    python docs/design/rust_graph_port/bench_harness.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.runtime import make_graph_runtime
from mstar.graph.special_destinations import SPECIAL_DESTINATIONS

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "test" / "rust"))
from test_graph_runtime_parity import decode_loop, nested_loops  # noqa: E402


def _tinfo(uuid: str) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[8, 16], dtype="bfloat16", nbytes=256, address=0, stride=[16, 1],
        uuid=uuid, source_session_id="h:1", source_entity="w0",
    )


def run(section, seeds, order, mode: str, n_rids: int, rounds: int) -> tuple[float, float]:
    """Returns (us per add_request, ns per rid per node completion)."""
    nodes = section.get_nodes()
    rt = make_graph_runtime(section, wg_id="wg", worker_id="w0", mode=mode)
    rids = [f"r{i}" for i in range(n_rids)]

    t0 = time.perf_counter()
    rt.add_requests(rids)
    add_us = (time.perf_counter() - t0) / n_rids * 1e6

    for rid in rids:
        for dest, name in seeds:
            rt.ingest(rid, [GraphEdge(next_node=dest, name=name,
                                      tensor_info=[_tinfo(f"{rid}-{name}")])])

    completions = 0
    t0 = time.perf_counter()
    for r in range(rounds):
        for node_name in order:
            for i, e in enumerate(nodes[node_name].outputs):
                e.tensor_info = [_tinfo(f"{node_name}{r}-{i}")]
            for rid in rids:
                rt.pop_ready(node_name, [rid])
                for e in rt.complete(rid, node_name).output_edges:
                    if e.next_node not in SPECIAL_DESTINATIONS and e.next_node in nodes:
                        rt.ingest(rid, [e])
                if rt.is_done(rid):
                    rt.reset(rid)
                completions += 1
    step_ns = (time.perf_counter() - t0) / completions * 1e9 if completions else 0.0
    return add_us, step_ns


CASES = [
    ("decode w=0", lambda: decode_loop(0, 10_000), [("llm", "input_ids"), ("llm", "position")],
     ["llm", "sampler"]),
    ("decode w=4", lambda: decode_loop(4, 10_000), [("llm", "input_ids"), ("llm", "position")],
     ["llm", "sampler"] + [f"side{i}" for i in range(4)]),
    ("nested", lambda: nested_loops(1000, 2), [("pre", "x")],
     ["pre", "step", "step", "post"]),
]


def main():
    print("Per-request admission (add_request), us/rid:")
    print(f"{'graph':>12} {'python':>10} {'rust':>10} {'speedup':>9}")
    for label, build, _seeds, _order in CASES:
        py, _ = run(build(), [], [], "0", 256, 0)
        rs, _ = run(build(), [], [], "1", 256, 0)
        print(f"{label:>12} {py:>10.1f} {rs:>10.2f} {py / rs:>8.0f}x")

    print("\nPer-rid per-node-completion (complete + route), ns:")
    print(f"{'graph':>12} {'B':>5} {'python':>10} {'rust':>10} {'speedup':>9}")
    for label, build, seeds, order in CASES:
        for b in (8, 64):
            _, py = run(build(), seeds, order, "0", b, 40)
            _, rs = run(build(), seeds, order, "1", b, 40)
            print(f"{label:>12} {b:>5} {py:>10.0f} {rs:>10.0f} {py / rs:>8.1f}x")

    print("\nNote: this is the per-event API, one boundary crossing per rid.")
    print("complete_and_route_batch collapses those to one per batch; see README §1.")


if __name__ == "__main__":
    main()
