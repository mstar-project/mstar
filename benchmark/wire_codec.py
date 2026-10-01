"""A/B the typed-msgpack wire codec against pickle, per message shape.

Why per shape: the two classes of message land on opposite sides of the
trade. A small control message carries a handful of scalars, so dropping
pickle's framing makes it smaller, and the codec's per-field closure costs
little. A tensor-carrying message repeats ``TensorPointerInfo`` once per
tensor, and every one of its ~15 field NAMES goes on the wire again -- msgpack
has no string memo where pickle has one -- so both size and time go the other
way, and they diverge with the tensor count.

Whoever encodes in Python pays this: the whole Python runtime path, and the
api server's data worker on every RESULT_TENSORS. The point of the Rust stack
is that the sender stops being Python, so this measures what is still on the
Python side rather than a cost the design intends to keep.

    python benchmark/wire_codec.py
    python benchmark/wire_codec.py --edges 1,8,64,256 --iters 1000
    python benchmark/wire_codec.py --json
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import time

import torch

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.api_server.request_types import APIServerMessage, ResultTensors
from mstar.communication.wire import decode, encode
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    DrainRequest,
    FailRequests,
    InputSignals,
    WorkerGraphsDone,
    WorkerMessage,
    WorkerMessageType,
)


def _tensor_info(i: int) -> TensorPointerInfo:
    """A descriptor as the SHM transport stamps one: the arena fields set, so
    the map form carries every key it will carry in production."""
    return TensorPointerInfo(
        dims=[4, 8], dtype=torch.float32, nbytes=128, address=0x7F000000 + i,
        stride=[8, 1], uuid=(1 << 48) | i,
        source_session_id="host.example:5555", source_entity="worker_0",
        shm_segment=f"mstar_arena_worker_0_{i % 4}", shm_offset=4096 * i,
        _source_node_name="prefill", _source_graph_walk="decode",
    )


def _fwd_info() -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="req-0", graph_walk="decode", fwd_index=7,
        random_seed=1234, max_tokens=256,
    )


def _input_signals(n_edges: int) -> WorkerMessage:
    """Worker -> worker, one tensor per edge. The hot one."""
    return WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(
            request_id="req-0",
            inputs=[
                GraphEdge(name=f"sig{i}", next_node="decoder",
                          tensor_info=[_tensor_info(i)])
                for i in range(n_edges)
            ],
            request_info=_fwd_info(),
        ),
    )


def _graphs_done(n_edges: int) -> ConductorMessage:
    """Worker -> conductor: persist signals carry the descriptors."""
    return ConductorMessage(
        message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
        body=WorkerGraphsDone(
            request_id="req-0", worker_graph_ids=[0], is_first_tp_rank=True,
            persist_signals={
                f"sig{i}": [_tensor_info(i)] for i in range(n_edges)
            },
            new_token_counts={"tok": 1},
            output_signal_names=[f"sig{i}" for i in range(n_edges)],
        ),
    )


def _result_tensors(n_edges: int) -> APIServerMessage:
    """Worker -> api server, once per output chunk. Named in review as a path
    the data worker pays on every message."""
    return APIServerMessage(
        message_type="result_tensors",
        body=ResultTensors(
            request_id="req-0", modality="text",
            graph_edge=GraphEdge(
                name="tok", next_node="EMIT_TO_CLIENT",
                tensor_info=[_tensor_info(i) for i in range(n_edges)],
            ),
            loop_indices=NestedLoopIndices(
                loop_name_order=["decode_loop"],
                loop_indices={"decode_loop": 3}, wg_fwd_pass_idx=7,
            ),
        ),
    )


def _control_tiny() -> WorkerMessage:
    """Two scalars: the smallest thing that crosses an edge."""
    return WorkerMessage(
        message_type=WorkerMessageType.DRAIN_REQUEST,
        body=DrainRequest(request_id="req-0"),
    )


def _control_busy() -> ConductorMessage:
    """Still tensor-free, but with a dict of strings -- a control message
    with enough in it to be worth encoding."""
    return ConductorMessage(
        message_type=ConductorMessageType.FAIL_REQUESTS,
        body=FailRequests(errors={
            f"req-{i}": f"RuntimeError: node prefill failed on rank {i}"
            for i in range(16)
        }),
    )


#: Tensor-free shapes, measured once each.
CONTROL_SHAPES = {
    "control/DrainRequest": _control_tiny,
    "control/FailRequests": _control_busy,
}


#: label -> builder. Anything taking an edge count is swept; ``_control``
#: has no tensors so it is measured once.
TENSOR_SHAPES = {
    "InputSignals": _input_signals,
    "WorkerGraphsDone": _graphs_done,
    "ResultTensors": _result_tensors,
}


def _median_us(fn, arg, iters: int) -> float:
    # Warm first: the codec compiles a field plan per class on first use, and
    # charging that to iteration one would swamp a microsecond measurement.
    fn(arg)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(arg)
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def measure(msg, iters: int) -> dict:
    pickled, wired = pickle.dumps(msg), encode(msg)
    # Both directions must actually reconstruct, or we would be timing a
    # codec that silently drops fields.
    assert decode(wired) is not None
    assert pickle.loads(pickled) is not None
    return {
        "encode_pickle_us": _median_us(pickle.dumps, msg, iters),
        "encode_wire_us": _median_us(encode, msg, iters),
        "decode_pickle_us": _median_us(pickle.loads, pickled, iters),
        "decode_wire_us": _median_us(decode, wired, iters),
        "bytes_pickle": len(pickled),
        "bytes_wire": len(wired),
    }


def _row(label: str, r: dict) -> str:
    return (
        f"{label:<26} "
        f"{r['encode_pickle_us']:>7.1f} {r['encode_wire_us']:>7.1f} "
        f"{r['encode_wire_us'] / r['encode_pickle_us']:>6.2f}x  "
        f"{r['decode_pickle_us']:>7.1f} {r['decode_wire_us']:>7.1f} "
        f"{r['decode_wire_us'] / r['decode_pickle_us']:>6.2f}x  "
        f"{r['bytes_pickle']:>8,} {r['bytes_wire']:>8,} "
        f"{r['bytes_wire'] / r['bytes_pickle']:>6.2f}x"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--edges", default="2,8,64",
                    help="comma-separated tensor counts to sweep")
    ap.add_argument("--iters", type=int, default=300,
                    help="timed iterations per measurement (median reported)")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable results instead of a table")
    args = ap.parse_args(argv)

    edge_counts = [int(x) for x in args.edges.split(",") if x.strip()]
    results: dict[str, dict] = {
        label: measure(build(), args.iters)
        for label, build in CONTROL_SHAPES.items()
    }
    for label, build in TENSOR_SHAPES.items():
        for n in edge_counts:
            results[f"{label}[{n}]"] = measure(build(n), args.iters)

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(f"median of {args.iters}; wire = typed msgpack, ratios are wire/pickle "
          "(<1 is better)\n")
    print(f"{'message':<26} {'encP':>7} {'encW':>7} {'':>7}  "
          f"{'decP':>7} {'decW':>7} {'':>7}  {'szP':>8} {'szW':>8}")
    for label, r in results.items():
        print(_row(label, r))
    print("\nencode/decode in microseconds, sizes in bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
