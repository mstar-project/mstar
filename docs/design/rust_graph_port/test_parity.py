"""Smoke parity: same event sequence through Python WorkerGraphsManager and the
Rust prototype; compare loop iters, routed destinations, emit/persist, doneness.

PYTHONPATH=docs/design/rust_graph_port/proto python docs/design/rust_graph_port/test_parity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "proto"))

import bench_python as bp
from bench_rust import build_runtime
from graph_fixture import WALK, sampler_output_edges

B, WIDTH, ITERS = 4, 2, 6


def main():
    mgr = bp.build_manager(WIDTH)
    rids = [f"r{i}" for i in range(B)]
    bp.add_requests(mgr, rids, WIDTH, 1)

    rt = build_runtime(WIDTH)
    handles = rt.add_requests(rids)

    from graph_fixture import _tinfo

    from mstar.graph.base import GraphEdge
    q = mgr.queues[bp.WG_ID]
    uid = 0
    for name in ("input_ids", "position"):
        for rid in rids:
            q.process_new_inputs(rid, [GraphEdge(next_node="llm", name=name,
                                                 tensor_info=[_tinfo(f"{rid}-{name}")])])
        rt.ingest_batch(handles, "llm", name, list(range(uid, uid + B)))
        uid += B

    n_out = 3 + WIDTH
    for it in range(ITERS):
        # llm
        for rid in rids:
            mgr.mark_node_complete(rid, bp.WG_ID, "llm")
            mgr.process_node_outputs(rid, node_name="llm", graph_walk=WALK, outputs=[
                GraphEdge(next_node="sampler", name="logits", tensor_info=[_tinfo(f"{rid}-lg{it}")])])
        rt.complete_and_route_batch("llm", handles, list(range(uid, uid + B)), [1] * B)
        uid += B

        # sampler
        py_emit = 0
        for rid in rids:
            mgr.mark_node_complete(rid, bp.WG_ID, "sampler")
            r = mgr.process_node_outputs(
                rid, node_name="sampler", graph_walk=WALK,
                outputs=[e.clone() for e in sampler_output_edges(WIDTH, rid)])
            py_emit += len(r.emit_to_client)
        rs = rt.complete_and_route_batch(
            "sampler", handles, list(range(uid, uid + B * n_out)), [1] * (B * n_out))
        uid += B * n_out
        assert len(rs.emit) == py_emit, (it, len(rs.emit), py_emit)
        assert len(rs.persist) == py_emit, (it, len(rs.persist), py_emit)

        # side nodes
        for i in range(WIDTH):
            for rid in rids:
                mgr.mark_node_complete(rid, bp.WG_ID, f"side{i}")
                mgr.process_node_outputs(rid, node_name=f"side{i}", graph_walk=WALK, outputs=[
                    GraphEdge(next_node="__emit_to_client__", name=f"aux{i}", persist=True,
                              tensor_info=[_tinfo(f"{rid}-a{i}{it}")])])
            rt.complete_and_route_batch(f"side{i}", handles, list(range(uid, uid + B)), [1] * B)
            uid += B

        py_iters = {rid: mgr.get_dynamic_loop_iters(rid, bp.PART)["decode_loop"] for rid in rids}
        rs_iters = rt.loop_iters(handles, "decode_loop")
        assert list(py_iters.values()) == rs_iters, (it, py_iters, rs_iters)

        py_ready = {rid: set(mgr.queues[bp.WG_ID].per_request_queues[rid].ready_node_names)
                    for rid in rids}
        scan = rt.ready_scan([0] * (2 + WIDTH))
        assert scan is not None and all(scan[0] in s for s in py_ready.values()), (it, py_ready, scan)

    print(f"parity OK: {ITERS} loop iterations, B={B}, width={WIDTH}, "
          f"loop_iters={rt.loop_iters(handles, 'decode_loop')}")


if __name__ == "__main__":
    main()
