"""Per-rid message cost in `_send_outputs` — the other batch-size-scaling loop.

`repr` was the eager `str(msg)` both communicators passed to a disabled
`logger.debug`; dropping it is the "before -> after" column below. `codec` is
the irreducible part, and only goes away when Rust owns the send.
"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graph_fixture import sampler_output_edges

from mstar.api_server.request_types import APIServerMessage, ResultTensors
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.utils.ipc_format import InputSignals, WorkerMessage, WorkerMessageType

N = 20000


def timeit(fn, msg):
    t = time.perf_counter()
    for _ in range(N):
        fn(msg)
    return (time.perf_counter() - t) / N * 1e6


def main():
    fi = CurrentForwardPassInfo(request_id="r0", fwd_index=3, random_seed=0,
                                max_tokens=128, graph_walk="decode", partition_name="p0")
    peer = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(request_id="r0", inputs=sampler_output_edges(0, "r0"),
                          request_info=fi, partition_name="p0"))
    li = NestedLoopIndices(loop_name_order=["decode_loop"],
                           loop_indices={"decode_loop": 7}, wg_fwd_pass_idx=0)
    emit = APIServerMessage(
        message_type="result_tensors",
        body=ResultTensors(request_id="r0", modality="text",
                           graph_edge=sampler_output_edges(0, "r0")[2],
                           loop_indices=li, metadata={}))

    print(f"{'message':>24} {'repr':>8} {'codec':>8} {'before':>8} {'after':>8}  (us/send)")
    for name, m in (("INPUT_SIGNALS (peer)", peer), ("result_tensors (client)", emit)):
        a, b = timeit(str, m), timeit(pickle.dumps, m)
        print(f"{name:>24} {a:>8.2f} {b:>8.2f} {a + b:>8.2f} {b:>8.2f}")


if __name__ == "__main__":
    main()
