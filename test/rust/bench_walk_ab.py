"""Walk-layer A/B: Python WorkerGraphIO vs PureRustWorkerGraphIO vs the
minimal uuid protocol (the target boundary). Interleaved per request so
background load hits all columns equally. Informational — run by CI after
the test suite; prints a table, never fails."""
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_walk_parity import _loop_graph  # noqa: E402

from mstar.graph.base import GraphEdge  # noqa: E402
from mstar.graph.graph_io import WorkerGraphIO  # noqa: E402
from mstar.graph.rust_core import (  # noqa: E402
    PureRustWorkerGraphIO,
    walks_to_json,
)
from mstar_rust import WalkSet  # noqa: E402

ITERS, REQS = 64, 60
SECTION = _loop_graph(ITERS)
STEPS = ITERS + 1
WS = WalkSet.from_json(walks_to_json({"walk": SECTION}))


def drive_io(io):
    io.ingest_input(GraphEdge(next_node="L", name="seed"))
    io.ingest_input(GraphEdge(next_node="L", name="fb"))
    while not io.wg_state_registry.is_done:
        name = sorted(io.ready_node_names)[0]
        io.ready_node_names.discard(name)
        comp = io.mark_node_complete(name)
        for e in comp.output_edges:
            if e.next_node in io.nodes:
                io.ingest_input(e)


def drive_minimal():
    st = WS.state("walk")
    st.seed_with([("L", "seed", 1), ("L", "fb", 2)])
    u = 3
    while not st.is_done():
        n = st.ready_nodes()[0]
        st.schedule(n)
        names = ("fb", "tok", "final") if n == "L" else ("out",)
        st.complete_full(n, [(e, [u + i]) for i, e in enumerate(names)])
        u += 3


RUNS = {
    "python registries": lambda: drive_io(WorkerGraphIO(deepcopy(SECTION), wg_id="w")),
    "pure rust adapter": lambda: drive_io(PureRustWorkerGraphIO(SECTION, "w")),
    "minimal uuid protocol": drive_minimal,
}
for fn in RUNS.values():
    fn()  # warm

totals = dict.fromkeys(RUNS, 0.0)
for _ in range(REQS):  # interleaved
    for name, fn in RUNS.items():
        t0 = time.perf_counter()
        fn()
        totals[name] += time.perf_counter() - t0

print(f"\nwalk A/B ({REQS} reqs x {STEPS} steps, interleaved)")
base = totals["python registries"]
for name, tot in totals.items():
    per = tot / REQS / STEPS * 1e6
    print(f"  {name:24} {per:8.2f} us/step   ({base / tot:4.2f}x vs python)")
