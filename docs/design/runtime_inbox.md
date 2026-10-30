---
orphan: true
---

# A runtime-owned inbox (design note, not implemented)

Status: proposed, parked. Nothing measured today is limited by it; this note records the design and the numbers behind it so the work can start from here when a workload is.

## The input path today

A worker's main thread drains its control socket and handles each message in Python (`Worker._process_message_list`):

1. **Decode** the message (`wire.decode`).
2. **Dispatch** on its type: `NEW_REQUEST`, `INPUT_SIGNALS`, `EDGE_ITEMS`, `TENSOR_RECEIVED`, `REMOVE_REQUEST`, `STOP_LOOPS` and the TP control messages.
3. **Graph-level work:** resolve the request's handle, park a message for a request not here yet or drop and ACK one for a removed request, register streamed items with their stream buffers, ingest signal-only edges into the graph runtime.
4. **Start the edges' tensor reads**, then later poll them (`get_ready_tensors`), ACK the producers and hand ready edges to the graph runtime or the stream buffers.

Only steps 2 and 4's device copies need Python objects. Steps 1 and 3 are bookkeeping the graph runtime mostly already owns in Rust, done in Python per message.

## What it costs (H200)

- **Streamed items, before EDGE_ITEMS** (Orpheus, LLM and SNAC on two GPUs, c=64): the SNAC worker spent 77% of its wall time on ~5,100 single-token INPUT_SIGNALS a second: starting reads 39% (76 µs a message), decode 23% (~29 µs), an empty non-streamed read call 7.6%. EDGE_ITEMS (#908) brought that to 16% at ~180 frames a second, ~0.67 ms per 52-row frame, of which rebuilding a descriptor per item is 0.15 ms.
- **Per request** (Qwen3.5 0.8B, one GPU, c=128, `worker.msg.*` spans from #857): about 0.68 ms of main-thread CPU per request. New request 0.37 ms (reading the prompt tensor 0.18, engine 0.11, runtime 0.05), removal 0.24 ms (engine 0.13, the "Request cleanup complete" INFO line 0.07), input signals 0.06 ms.
- **TP followers** (Qwen3.5 TP2): the input path is 4–6% of wall time.

The per-request cost scales with the request rate, so it matters for short requests at high concurrency, which is where Qwen3.5's prompt and prefill-token traffic lands.

## Proposal

The graph runtime drains the worker's inbox itself, in the same begin / resume shape as postprocess (#857), and returns to Python only what needs Python, as columns:

- **In Rust:** draining the socket (the Rust ZMQ transport already receives the bytes), decoding, handle resolution, parking and dropping, ingesting signal-only edges, registering streamed uuids with stream buffers once those are in Rust (buffers as lists of uuids, chunk policies with Rust backends), the arena's read bookkeeping (placement, the per-call batching #908 does in Python), and batched ACKs.
- **Returned to Python, one call per drain:** new requests (engine admission, per-request resource setup), removals (engine and tensor-manager teardown), and the reads that need a device copy, as one column per frame: uuid, segment, offset, size, dtype, shape. Python does the copies in one gathered H2D per frame (as `_read_many_from_arena` does) and puts the tensors into the store in one call.

Constraints it has to keep:

- **Per-peer order.** Messages from one peer are handled in arrival order, and a parked message replays before later ones for the same request.
- **Lockstep** (invariant 11). A follower handles control messages in rank 0's order; anything the inbox decides must be derived from replicated state or replayed, never from per-rank timing.
- **Both runtimes** (invariant 5). The Python runtime gets the same begin / resume API over today's handlers, and parity tests drive both with the same message streams, as `test_stream_completion_order.py` does for EDGE_ITEMS.
- **Handles stay local** (invariant 14). Translation happens inside the inbox; nothing it returns carries another worker's handle.

## Staging

Each step is useful alone and measurable with `benchmark/worker_phases`:

1. **Read EDGE_ITEMS from its columns:** no `TensorPointerInfo` or `GraphEdge` per item on the receiver; the read path, ACKs and stream buffers take uuids and columns.
2. **Decode and graph-level ingest in Rust** for INPUT_SIGNALS and EDGE_ITEMS.
3. **NEW_REQUEST and REMOVE_REQUEST**: decode and runtime bookkeeping in Rust, with the engine's part returned as columns.

## When to pick it up

When a served workload shows the input path, not the GPU, bounding a worker: `worker.main.process_messages` and `worker.main.check_ready_tensors` a large share of wall time with `worker.postprocess.event_sync` near zero. As of #908 none of the measured configs does: Orpheus and Qwen3-Omni on two GPUs are waiting on their GPUs.
