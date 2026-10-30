"""Speculation over a real graph runtime, Python and Rust, on the paths the
parity tests in ``test_spec_rows`` do not reach.

* A decode loop that consumes a stream, run under random speculation outcomes
  (kept, refused, a row dropped). Every item a producer sends must reach the
  executed steps in order, exactly once, with the chunk info the model is told
  matching the tensors it gets, and the final flag landing once.
* A columnar speculation joined by fresh rows of another walk: the same step as
  the per-row path, and the GPU thread's setup of it never calls the runtime,
  which may be inside a call that released the GIL (``route_and_send_inline``).
* A columnar prep whose rows carry a loop-external input: it falls back to
  per-row inputs and builds the same steps as the per-row path.
"""
import logging
import random
from types import SimpleNamespace

import pytest
import test_spec_rows as T
import torch

from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, TensorPointerInfo
from mstar.graph.runtime.base import ColumnarEdgeSpecs
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.model.base import WorkerGraph
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import Connection
from mstar.worker.spec_rows import DeferredRowInputs
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import PendingBatch

CODES, CTX = "codes", "ctx"


class _TM(T._StubTM):
    """``_StubTM`` that can also store a request's tensors, as a stream's
    chunks and a persisted input are stored."""

    _next_uuid = 1 << 40

    def store_and_return_tensor_info(self, request_id, tensors, skip_cuda_sync=False, **_):
        out = {}
        for name, ts in tensors.items():
            out[name] = []
            for t in ts:
                _TM._next_uuid += 1
                info = TensorPointerInfo(
                    dims=list(t.shape), dtype=t.dtype, nbytes=t.numel() * t.element_size(),
                    address=0, stride=list(t.stride()), uuid=_TM._next_uuid,
                    source_session_id="h:1", source_entity=T.ME,
                )
                self.tensor_store.put_tensor(request_id, info.uuid, t, info)
                out[name].append(info)
        return out


def _loop(extra_input: str, streaming: bool) -> Loop:
    """The decode loop of ``test_spec_rows`` with one more input on its node."""
    node = GraphNode(
        name=T.NODE, input_names=[T.SIG, extra_input],
        outputs=[GraphEdge(next_node=T.NODE, name=T.SIG)],
    )
    if streaming:
        node._register_streaming({extra_input})
    return Loop(name=T.LOOP, section=node, max_iters=1 << 20, outputs=[])


def _worker(kind: str, section: Loop):
    def runtime(kind, *_a, **_k):
        common = dict(
            my_worker_id=T.ME,
            my_worker_graphs=[WorkerGraph(
                section=section, graph_walks={T.WALK}, ranks=[0], worker_graph_id=T.WG,
            )],
            all_wg_ids_to_graph_walks={T.WG: {T.WALK}}, all_wg_ids_to_dyn_loops={T.WG: {T.LOOP}},
            all_wg_ids_to_nodes={T.WG: {T.NODE}}, node_to_partition={T.NODE: T.PART},
            sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        )
        if kind == "python":
            tm = _TM(T.PythonTensorBookkeeping())
            return PythonGraphRuntime(**common, tensor_manager=tm, communicator=None), tm
        from mstar.communication.tensor_store import RustTensorBookkeeping
        from mstar.graph.runtime import rust as rust_runtime
        book = RustTensorBookkeeping()
        return rust_runtime.RustGraphRuntime(**common, bookkeeping=book), _TM(book)

    shared, T._runtime = T._runtime, runtime
    try:
        w = T._worker(kind)
    finally:
        T._runtime = shared
    w.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(
        release_cached_prefix=lambda batch: None,
    ))
    return w


# --- a stream through speculation ----------------------------------------------

def _values(w, uuids) -> list[int]:
    return [int(v) for u in uuids for v in w.tensor_manager.get_tensor(u).flatten().tolist()]


def _run_stream(kind: str, seed: int, n_rids=3, n_items=9, chunk=2):
    """Drive the loop until every row's stream ends. Returns, per row, the
    items sent, the items executed steps consumed, the items still held by
    its buffer, and how often a step reported its stream final."""
    rng = random.Random(seed)
    section = _loop(CODES, streaming=True)
    w = _worker(kind, section)
    w.streams = StreamBufferManager(
        w.request_state, w.tensor_manager,
        [Connection(
            from_partition="LLM", to_partition=T.PART, edge_name=CODES,
            chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=chunk),
        )],
        {T.WALK: section},
    )
    rids = T._admit(w, n_rids)
    for r in rids:
        w.streams.add_request(r, SimpleNamespace(partition_name=T.PART, stream_lead_items={}))
    sbuf = {r: w.request_state.per_request_info[r].stream_buffers[CODES] for r in rids}
    sent = dict.fromkeys(rids, 0)
    seen = {r: [] for r in rids}
    left = {}
    finals = dict.fromkeys(rids, 0)
    alive = set(rids)

    def produce():
        for r in sorted(alive):
            for _ in range(min(rng.randint(0, 2), n_items - sent[r])):
                # stored with the one reference its stream destination
                # holds, which the buffer takes over, as the worker does
                (info,) = w.tensor_manager.store_and_return_tensor_info(
                    r, {CODES: [torch.tensor([float(sent[r])])]},
                )[CODES]
                w.tensor_manager.tensor_store.increment_ref(info.uuid, 1)
                sbuf[r].pre_read_register(info.uuid)
                sbuf[r].put(info.uuid, info.uuid)
                sent[r] += 1
            if sent[r] == n_items and not sbuf[r].producer_done:
                sbuf[r].signal_done(n_items)

    def record(nb):
        """What an executed step hands the model."""
        for r in nb.request_ids:
            inputs = nb.per_request_input_tensors[r]
            if CODES not in inputs:
                continue
            vals = [int(v) for t in inputs[CODES] for v in t.flatten().tolist()]
            meta = nb.per_request_input_metadata.get(r)
            info = meta.stream_chunks.get(CODES) if meta else None
            assert info is not None, f"seed {seed}: rid {r} got a chunk with no chunk info"
            assert (info.start_offset, info.num_items) == (len(seen[r]), len(vals)), (
                f"seed {seed}: rid {r} told {info}, got {vals} after {seen[r]}"
            )
            seen[r].extend(vals)
            finals[r] += r in nb.final_stream_rids

    def drain(r):
        """The rest of the row's stream, as its buffer still holds it."""
        vals = []
        while (se := w.streams.pop_edge(sbuf[r], CODES, r)) is not None:
            vals += _values(w, se.uuids)
        sbuf[r]._update_buffer()  # items short of a chunk
        left[r] = vals + _values(w, sbuf[r]._buffer)

    def retire(r):
        # No loop-back output means its loop ended: stop polling it.
        drain(r)
        info = w.request_state.per_request_info[r]
        info.stream_buffers.clear()
        info.stream_buffers_by_consumer.clear()

    def plain_step():
        """Poll every buffer, then run whichever rows are ready."""
        w._poll_stream_buffers()
        popped = w._graph_runtime.pop_rids(T.NODE, T.WALK, sorted(alive))
        if not len(popped.wg_ids.keys):
            return None
        batch = T.ScheduledBatch(
            node_name=T.NODE, graph_walk=T.WALK,
            request_to_worker_graph=dict(zip(popped.wg_ids.keys, popped.wg_ids.values, strict=True)),
            input_edges=popped.input_edges, output_signals=list(popped.output_signals),
        )
        w._set_in_flight_flag(batch, True)
        return PendingBatch(
            batch=batch, node_batch=w._build_executing_batch(batch), node_name=T.NODE,
            partition=T.PART, graph_walk=T.WALK, future=None,
        )

    pending = None
    for _ in range(400):
        if not alive or all(finals[r] for r in alive):
            break
        produce()
        if pending is None:
            pending = plain_step()
            if pending is not None:
                record(pending.node_batch)
            continue
        target = w._choose_spec_target(pending.batch)
        spec = w._try_speculate_next(pending, target) if target else None
        outcome = rng.choice(["keep", "keep", "refuse", "drop"]) if spec else "none"
        if outcome in ("keep", "drop"):
            missing = (rng.choice(sorted(spec.continuing_rids)),) if outcome == "drop" else ()
            w._thread_outputs_to_speculative(spec, T._rows_only(pending, missing))
            pending = T._advance(w, pending, spec)
            for r in missing:
                retire(r)
            alive -= set(missing)
            if pending.node_batch.request_ids:
                record(pending.node_batch)
            else:
                pending = None
        else:
            if spec is not None:
                w._settle_speculation(spec, success=False)
            w._clear_in_flight_flag(pending.batch)
            T._route(w, pending)
            pending = None
    for r in alive:
        drain(r)
    return rids, sent, seen, left, finals, alive


@pytest.mark.parametrize("kind", T.KINDS)
def test_a_stream_reaches_executed_steps_once_and_in_order(kind, caplog):
    caplog.set_level(logging.ERROR)
    for seed in range(40):
        rids, sent, seen, left, finals, alive = _run_stream(kind, seed)
        for r in rids:
            assert seen[r] + left[r] == list(range(sent[r])), (
                f"seed {seed}: rid {r} consumed {seen[r]}, its buffer holds {left[r]}"
            )
            if r in alive:
                assert (left[r], finals[r]) == ([], 1), f"seed {seed}: rid {r}"


# --- fresh rows of another walk --------------------------------------------------

class _NoCalls:
    """The graph runtime as the GPU thread may use it while the main thread is
    in a call that released the GIL: class attributes only."""

    def __init__(self, rt):
        self._rt = rt

    def __getattr__(self, name):
        value = getattr(type(self._rt), name, None)
        if value is not None and not callable(value) and not isinstance(value, property):
            return value
        raise AssertionError(f"the GPU thread called the graph runtime: {name}")


@pytest.mark.parametrize("kind", T.KINDS)
def test_fresh_rows_of_another_walk_and_the_gpu_thread_setup(kind, monkeypatch):
    w = T._worker(kind, mixed=True)
    w.scheduler.label_for = lambda node, request_walks: (
        (T.MIXED, request_walks) if len(set(request_walks.values())) > 1
        else (next(iter(request_walks.values())), {})
    )
    decode = T._admit(w, 3)
    prefill = T._admit(w, 2, start=3, walk=T.PRE)
    pending = T._first_step(w, decode)
    fresh = T._pop(w, prefill, T.PRE)
    fresh.request_walks = dict.fromkeys(prefill, T.PRE)
    fresh.walk_output_signals = {T.PRE: list(fresh.output_signals)}
    fresh.graph_walk = T.MIXED
    w.scheduler.fresh = fresh

    general = T._speculate(w, pending, rows=False, monkeypatch=monkeypatch)
    rows = T._speculate(w, pending, rows=True, monkeypatch=monkeypatch)
    assert rows.scheduled_batch.request_walks == {
        **dict.fromkeys(decode, T.WALK), **dict.fromkeys(prefill, T.PRE),
    }
    assert T._view(rows) == T._view(general)
    outputs = T._rows_only(pending)
    w._thread_outputs_to_speculative(general, outputs)
    w._thread_outputs_to_speculative(rows, outputs)
    assert T._view(rows) == T._view(general)

    batch, rt = rows.scheduled_batch, w._graph_runtime
    assert {walk: list(s) for walk, s in batch.walk_output_signals.items()} == {
        walk: list(rt.get_output_signals(T.NODE, walk)) for walk in (T.WALK, T.PRE)
    }
    # only the decode walk keeps its output on the device
    engine = SimpleNamespace(
        device_loopback_signals=lambda node, walk: [T.SIG] if walk == T.WALK else [],
    )
    w._device_loopback = {}
    expected = w._loopback_rids(engine, batch)
    w._device_loopback = {}
    w._graph_runtime = _NoCalls(rt)
    assert w._loopback_rids(engine, batch) == expected


# --- a loop-external input -------------------------------------------------------

@pytest.mark.parametrize("kind", T.KINDS)
def test_a_loop_external_input_falls_back_to_rows_and_persists(kind, monkeypatch):
    w = _worker(kind, _loop(CTX, streaming=False))
    rids = T._admit(w, 3)
    block = ColumnarEdgeSpecs.empty()
    for r in rids:
        stored = w.tensor_manager.store_and_return_tensor_info(r, {CTX: [torch.tensor([100.0 + r])]})
        block.add(r, CTX, [stored[CTX][0].uuid], False, next_node=T.NODE)
    w._graph_runtime.ingest_inputs_batch(block)
    pending = T._first_step(w, rids)
    for _ in range(4):
        general = T._speculate(w, pending, rows=False, monkeypatch=monkeypatch)
        rows = T._speculate(w, pending, rows=True, monkeypatch=monkeypatch)
        assert not isinstance(rows.node_batch.per_request_input_tensors, DeferredRowInputs)
        assert T._view(rows) == T._view(general)
        outputs = T._rows_only(pending)
        w._thread_outputs_to_speculative(general, outputs)
        w._thread_outputs_to_speculative(rows, outputs)
        assert T._view(rows) == T._view(general)
        inputs = rows.node_batch.per_request_input_tensors
        assert {r: [float(t) for t in inputs[r][CTX]] for r in rids} == {
            r: [100.0 + r] for r in rids
        }
        pending = T._advance(w, pending, rows)
