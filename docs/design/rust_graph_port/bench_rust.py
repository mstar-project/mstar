"""Rust prototype vs the Python baseline, same graph and same batch.

Build first:
    cd docs/design/rust_graph_port/proto && cargo build --release \
      && cp target/release/libmstar_graph_proto.so mstar_graph_proto.so
Run:
    PYTHONPATH=docs/design/rust_graph_port/proto \
      python docs/design/rust_graph_port/bench_rust.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "proto"))

import bench_python as bp
from graph_fixture import make_section
from mstar_graph_proto import GraphRuntime
from rust_spec import to_rust_spec


def build_runtime(width: int, tp_size: int = 1, section=None) -> GraphRuntime:
    """Compiled from the same Python GraphSection the baseline drives."""
    section = section if section is not None else make_section(width)
    return GraphRuntime(
        **to_rust_spec(section),
        workers=[f"w{i}" for i in range(tp_size)], me="w0",
    )


def bench(width: int, B: int, iters: int, tp_size: int = 1):
    rt = build_runtime(width, tp_size)
    handles = rt.add_requests([f"r{i}" for i in range(B)])
    uid = 0
    for name in ("input_ids", "position"):
        rt.ingest_batch(handles, "llm", name, list(range(uid, uid + B)))
        uid += B
    rt.complete_and_route_batch("llm", handles, list(range(uid, uid + B)), [1] * B)
    uid += B

    n_out = 3 + width
    tlens = [1] * (B * n_out)
    payload = list(range(uid, uid + B * n_out))

    t0 = time.perf_counter()
    for _ in range(iters):
        rt.complete_and_route_batch("sampler", handles, payload, tlens)
        rt.complete_and_route_batch("llm", handles, payload[:B], [1] * B)
        for i in range(width):
            rt.complete_and_route_batch(f"side{i}", handles, payload[:B], [1] * B)
    # one full loop iteration per pass; charge only the sampler step to match
    # the Python baseline, which also measures one node's postprocess
    dt = (time.perf_counter() - t0) / iters
    per_node = dt / (2 + width)
    return per_node / B * 1e9


def bench_sched(width: int, B: int, iters: int):
    rt = build_runtime(width)
    handles = rt.add_requests([f"r{i}" for i in range(B)])
    for name in ("input_ids", "position"):
        rt.ingest_batch(handles, "llm", name, list(range(B)))
    last = [0] * (2 + width)
    t0 = time.perf_counter()
    for _ in range(iters):
        rt.ready_scan(last)
    return (time.perf_counter() - t0) / iters / B * 1e9


def main():
    print(f"{'B':>5} {'width':>6} {'py postproc':>12} {'rs postproc':>12} {'speedup':>8}   "
          f"{'py sched':>9} {'rs sched':>9} {'speedup':>8}   (ns/rid)")
    for width in (0, 4):
        for B in (1, 8, 32, 128):
            mgr = bp.build_manager(width)
            rids = [f"r{i}" for i in range(B)]
            bp.add_requests(mgr, rids, width, 1)
            bp.seed(mgr, rids, width)
            it = max(4, 4000 // B)
            py_post = bp.bench_postprocess(mgr, rids, width, it)
            py_sched = bp.bench_sched_scan(mgr, rids, width, it)
            rs_post = bench(width, B, it)
            rs_sched = bench_sched(width, B, it)
            print(f"{B:>5} {width:>6} {py_post:>12.0f} {rs_post:>12.0f} "
                  f"{py_post / rs_post:>7.1f}x   {py_sched:>9.0f} {rs_sched:>9.0f} "
                  f"{py_sched / rs_sched:>7.1f}x")


if __name__ == "__main__":
    main()
