"""MSTAR_FAST_SEND: flag-on vs flag-off equivalence for the emit path.

The flag trims per-rid Python around the steady decode emit path
(``Worker._register_outputs`` / ``Worker._send_outputs``):

- ``_inline_emit_uuids`` is computed once per rid per step (stashed on the
  routing object by ``_register_outputs``, reused by ``_send_outputs``);
- the empty-set ``register_for_send`` call is skipped;
- the manager bookkeeping (``buffer_new_tokens`` / ``buffer_output_signals``
  / ``register_output_loop_indices``) is written inline against one hoisted
  ``per_request_info`` reference;
- the ``{"inline_values": ...}`` metadata dict is not built on the slim
  steady path where its only consumer (the full ``ResultTensors``) is
  already skipped by MSTAR_SLIM_EMIT2.

These tests drive the two real methods (unbound, against a stub worker) with
hand-built routing and assert that everything observable — the pickled
outgoing payloads, the manager's flushed-later state, the ref-count releases
— is byte-identical between the two flag settings. Pure CPU, no GPU, no
server.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import torch

from mstar.api_server.request_types import ResultTensorsBatch
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.worker.node_manager_utils import (
    NodeOutputRouting,
    PerRequestInfo,
    WorkerGraphsManager,
)
from mstar.worker.worker import Worker

# ---------------------------------------------------------------------------
# Stub collaborators — record every observable side effect.
# ---------------------------------------------------------------------------

class _RecordingTensorManager:
    def __init__(self):
        self.deref_calls: list[tuple] = []
        self.register_calls: list[tuple] = []
        self.persist_calls: list[tuple] = []

    def dereference(self, request_id, uuid, n=1):
        self.deref_calls.append((request_id, uuid, n))

    def register_for_send(self, request_id, tensor_infos, skip_cuda_sync=False):
        self.register_calls.append(
            (request_id, frozenset(info.uuid for info in tensor_infos), skip_cuda_sync)
        )

    def set_persist(self, request_id, uuid, persist):
        self.persist_calls.append((request_id, uuid, persist))

    def get_tensor(self, request_id, uuid):
        # Since #149, _send_outputs derives per-signal new-token COUNTS from
        # tensor.numel(); a deterministic single-element tensor gives count 1
        # per new-token signal, identically under both flag settings.
        return torch.tensor([sum(uuid.encode()) % 1000], dtype=torch.int64)


class _RecordingCommunicator:
    def __init__(self):
        self.sent: list[tuple] = []

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))


class _StubWorker:
    """Bare object carrying exactly the state the two real methods read,
    with the real (unbound) Worker methods attached."""

    _send_outputs = Worker._send_outputs
    _register_outputs = Worker._register_outputs
    _inline_emit_uuids = Worker._inline_emit_uuids

    def __init__(self, fast_send: bool, rids: list[str]):
        self._inline_emit = True
        self._batch_emit = True
        self._slim_emit = True
        self._slim_emit2 = True
        self._fast_send = fast_send
        # MSTAR_SCHED_PACK landed after this stub was written and
        # _inline_emit_uuids reads it unconditionally; default off matches the
        # flag's own default (byte-identical-off path).
        self._sched_pack = False
        self._slim_emit_sent: set[tuple[str, str]] = set()
        self._slim_emit_loop_layout: dict[tuple[str, str], tuple] = {}
        self.tensor_manager = _RecordingTensorManager()
        self.communicator = _RecordingCommunicator()
        self.worker_graphs_manager = WorkerGraphsManager(
            queues={},
            per_request_info={
                rid: PerRequestInfo(
                    node_to_workers={},
                    dyn_loop_to_workers={},
                    worker_graph_ids=[],
                    sharding_config=None,
                )
                for rid in rids
            },
            base_sharding_config=None,
            worker_id="worker_0",
            all_worker_graph_ids_to_graph_walks={},
            all_worker_graph_ids_to_nodes={},
            all_worker_graph_ids_to_dyn_loops={},
        )


# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------

EDGE_NAME = "new_text_token"


def _tpi(uuid: str) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[1], dtype="torch.int64", nbytes=8, address=0, stride=[1],
        uuid=uuid, source_session_id="host:1", source_entity="worker_0",
    )


def _routing(rid: str, share_loopback_uuid: bool = False) -> NodeOutputRouting:
    """A steady thinker-decode step's routing for one rid: one inline-
    qualifying emit edge + one loop-back edge. ``share_loopback_uuid=True``
    makes the loop-back edge reference the emit tensor's uuid, which must
    disqualify the emit edge from the inline path (condition (c))."""
    emit_uuid = f"uuid-emit-{rid}"
    loop_uuid = emit_uuid if share_loopback_uuid else f"uuid-loop-{rid}"
    emit_edge = GraphEdge(
        next_node="EMIT_TO_CLIENT", name=EDGE_NAME,
        tensor_info=[_tpi(emit_uuid)], output_modality="text",
    )
    token_edge = GraphEdge(
        next_node="EMIT_TO_CLIENT", name=EDGE_NAME,
        tensor_info=[_tpi(emit_uuid)], conductor_new_token=True,
    )
    loop_edge = GraphEdge(
        next_node="LLM", name="text_inputs", tensor_info=[_tpi(loop_uuid)],
    )
    return NodeOutputRouting(
        routed_to_this_worker_graph=[loop_edge],
        is_first_tp_rank=True,
        persist=[],
        to_workers={},
        emit_to_client=[emit_edge],
        new_token_outputs=[token_edge],
    )


def _nli(step: int) -> NestedLoopIndices:
    return NestedLoopIndices(
        loop_name_order=["thinker_decode_loop"],
        loop_indices={"thinker_decode_loop": step},
        wg_fwd_pass_idx=step,
    )


def _run_step(
    worker: _StubWorker,
    rids: list[str],
    step: int,
    share_loopback_uuid: bool = False,
):
    """One _register_outputs + per-rid _send_outputs pass, mirroring the
    single _postprocess_batch call sequence. Returns the batch collector."""
    routing_per_request = {
        rid: _routing(rid, share_loopback_uuid) for rid in rids
    }
    prem_per_request = {rid: {EDGE_NAME: [1000 + step]} for rid in rids}
    batch = SimpleNamespace(node_objects={rid: None for rid in rids})
    worker._register_outputs(
        batch, routing_per_request,
        prematerialized_per_request=prem_per_request,
    )
    collector: list = []
    for rid in rids:
        worker._send_outputs(
            rid, routing_per_request[rid],
            nested_loop_indices=_nli(step),
            graph_walk="thinker_decode",
            partition_name="default",
            prematerialized_new_tokens=prem_per_request[rid],
            batch_collector=collector,
        )
    return collector, routing_per_request


def _observable_state(worker: _StubWorker, collector: list) -> bytes:
    """Everything the rest of the system can see, as one picklable blob:
    the outgoing batch payload, the immediately-sent messages, the manager
    state flushed later on WORKER_GRAPHS_DONE, the slim-emit template
    state, and the ref-count releases."""
    wgm = worker.worker_graphs_manager
    return pickle.dumps({
        "batch_payload": ResultTensorsBatch(items=collector),
        "sent": worker.communicator.sent,
        "per_request": {
            rid: (
                info.pending_new_token_counts,
                info.current_output_chunks,
                info.output_loop_indices,
                info.pending_persist_signals,
            )
            for rid, info in wgm.per_request_info.items()
        },
        "slim_sent": sorted(worker._slim_emit_sent),
        "slim_layout": worker._slim_emit_loop_layout,
        "deref": worker.tensor_manager.deref_calls,
        "persist": worker.tensor_manager.persist_calls,
    })


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

RIDS = ["rid-a", "rid-b", "rid-c"]


def test_template_then_steady_steps_byte_identical():
    """Step 0 (template ResultTensors) + steps 1-2 (slim steady items):
    every observable is byte-identical flag-on vs flag-off."""
    blobs = {}
    for fast in (False, True):
        worker = _StubWorker(fast_send=fast, rids=RIDS)
        states = []
        for step in range(3):
            collector, _ = _run_step(worker, RIDS, step)
            states.append(_observable_state(worker, collector))
        blobs[fast] = states
    assert blobs[False] == blobs[True]


def test_steady_step_uses_slim_items_and_no_immediate_sends():
    """Sanity that the scenario exercises the intended path: after the
    template step, the collector holds SlimResultTokens with loop_key set,
    and nothing was sent directly (no completions, no non-inline edges)."""
    worker = _StubWorker(fast_send=True, rids=RIDS)
    _run_step(worker, RIDS, 0)
    collector, _ = _run_step(worker, RIDS, 1)
    assert len(collector) == len(RIDS)
    for item in collector:
        assert type(item).__name__ == "SlimResultTokens"
        assert item.loop_key is not None and item.loop_indices is None
        assert item.values == [1001]
    assert worker.communicator.sent == []


def test_non_inline_edge_byte_identical():
    """When the emit uuid is also referenced by a loop-back edge, the edge
    must stay on the SHM path (full ResultTensors sent immediately) — and
    the uuid must be registered for send — identically under both flags."""
    blobs = {}
    registers = {}
    for fast in (False, True):
        worker = _StubWorker(fast_send=fast, rids=RIDS)
        collector, _ = _run_step(worker, RIDS, 0, share_loopback_uuid=True)
        blobs[fast] = _observable_state(worker, collector)
        registers[fast] = worker.tensor_manager.register_calls
    assert blobs[False] == blobs[True]
    # The register calls themselves must match too: non-empty sets are
    # never skipped by the flag.
    assert registers[False] == registers[True]
    assert all(uuids for _, uuids, _ in registers[True])


def test_empty_register_skipped_only_with_flag():
    """The one intended call-shape divergence: with every emit uuid inline,
    register_for_send gets an empty set — a no-op the flag skips."""
    for fast, expect_calls in ((False, len(RIDS)), (True, 0)):
        worker = _StubWorker(fast_send=fast, rids=RIDS)
        _run_step(worker, RIDS, 0)
        calls = worker.tensor_manager.register_calls
        assert len(calls) == expect_calls
        assert all(uuids == frozenset() for _, uuids, _ in calls)


def test_stash_matches_flag_off_computation():
    """The stashed set must equal exactly what _send_outputs would have
    recomputed (same routing, same prem) — and stay unset when off."""
    for fast in (False, True):
        worker = _StubWorker(fast_send=fast, rids=RIDS)
        _, routing_per_request = _run_step(worker, RIDS, 0)
        for _rid, routing in routing_per_request.items():
            expected = worker._inline_emit_uuids(
                routing, {EDGE_NAME: [1000]}
            )
            if fast:
                assert routing.inline_emit_uuids == expected
            else:
                assert routing.inline_emit_uuids is None
