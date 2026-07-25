"""MSTAR_EMIT_SIDECAR: byte-identity of the sidecar path vs the legacy path.

The mandated byte-identity harness: drive
recorded fake step streams through BOTH paths —

- legacy: the real ``Worker._register_outputs`` + ``Worker._send_outputs``
  with the winning emit stack (BATCH+SLIM+SLIM2), one batch message per step;
- sidecar: the real ``Worker._send_outputs_sidecar`` producing entries, the
  real ``SidecarRecordBuilder`` assembling StepRecords, each record
  pickle-round-tripped (the wire) into the real ``SidecarState``

— and assert the SEQUENCE OF MESSAGES to the api_server and the conductor is
byte-identical (pickle bytes), per destination. Scenarios: steady decode,
request completion incl. WGD, loop-stop terminal step, non-inline edges
(shared-uuid disqualification and prem-less prefill rows), mixed-walk rows,
and the mixed-POPULATION step (scoped + legacy rid in one batch — the one
documented relaxation: the step's inline items split across two batch
messages, one per FIFO, each byte-identical to a single-population legacy
run).

Also asserts the wholesale-ownership invariant: the worker's PerRequestInfo
accumulators are NEVER written for a scoped rid; and the record-cheapness
invariant: steady-state records carry no GraphEdge and no NestedLoopIndices
objects.

Pure CPU, no GPU, no server (the lifecycle test spawns the real sidecar
process, still CPU-only with CUDA_VISIBLE_DEVICES="").
"""

from __future__ import annotations

import pickle
import time
from types import SimpleNamespace

import torch
import zmq

from mstar.api_server.request_types import (
    APIServerMessage,
    ResultTensorsBatch,
)
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.worker.emit_sidecar import (
    SIDECAR_WALKS,
    SIDECAR_WALKS_I2T_EXTRA,
    SidecarClient,
    SidecarRecordBuilder,
    SidecarState,
    StepRecord,
)
from mstar.worker.node_manager_utils import (
    NodeOutputRouting,
    PerPartitionInfo,
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
        # Deterministic CPU tensor for the prem-less D2H fallback path.
        return torch.tensor([sum(uuid.encode()) % 1000], dtype=torch.int64)

    def get_rx_info(self, request_id):
        return []

    def get_tx_info(self, request_id):
        return []


class _RecordingCommunicator:
    def __init__(self):
        self.sent: list[tuple] = []

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))


def _fwd_info(rid: str) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=rid,
        graph_walk="thinker_decode",
        requires_cfg=False,
        fwd_index=0,
        random_seed=0,
        max_tokens=177,
        sampling_config={},
    )


class _StubWorker:
    """Bare object carrying exactly the state the real methods read, with
    the real (unbound) Worker methods attached. Same pattern as
    test_fast_send_emit.py, extended with the sidecar twin."""

    _send_outputs = Worker._send_outputs
    _send_outputs_sidecar = Worker._send_outputs_sidecar
    _register_outputs = Worker._register_outputs
    _inline_emit_uuids = Worker._inline_emit_uuids

    def __init__(self, rids: list[str]):
        # Winning emit stack — the baseline MSTAR_EMIT_SIDECAR requires.
        self._inline_emit = True
        self._batch_emit = True
        self._slim_emit = True
        self._slim_emit2 = True
        self._fast_send = False
        # Pre-existing drift fix: MSTAR_SCHED_PACK landed after this stub was
        # written and _inline_emit_uuids now reads it unconditionally. Default
        # off, matching the flag's own default, so this stub exercises the
        # byte-identical-off path like every other flag here.
        self._sched_pack = False
        self._slim_emit_sent: set[tuple[str, str]] = set()
        self._slim_emit_loop_layout: dict[tuple[str, str], tuple] = {}
        self._sidecar_client: SidecarRecordBuilder | None = None
        self.tensor_manager = _RecordingTensorManager()
        self.communicator = _RecordingCommunicator()
        self.profile_info = SimpleNamespace(per_rid_graph_timings={})
        self.worker_graphs_manager = WorkerGraphsManager(
            queues={},
            per_request_info={
                rid: PerRequestInfo(
                    node_to_workers={},
                    dyn_loop_to_workers={},
                    worker_graph_ids=[],
                    sharding_config=None,
                    per_partition_info={
                        "default": PerPartitionInfo(
                            current_fwd_info=_fwd_info(rid),
                        )
                    },
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

TOKEN_EDGE = "new_text_token"
FINAL_EDGE = "thinker_final_text"


def _tpi(uuid: str) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[1], dtype="torch.int64", nbytes=8, address=0, stride=[1],
        uuid=uuid, source_session_id="host:1", source_entity="worker_0",
    )


def _nli(step: int) -> NestedLoopIndices:
    return NestedLoopIndices(
        loop_name_order=["thinker_decode_loop"],
        loop_indices={"thinker_decode_loop": step},
        wg_fwd_pass_idx=step,
    )


def _prefill_nli() -> NestedLoopIndices:
    return NestedLoopIndices(
        loop_name_order=[],
        loop_indices={},
        wg_fwd_pass_idx=0,
    )


def _steady_routing(
    rid: str, step: int, share_loopback_uuid: bool = False
) -> NodeOutputRouting:
    """A steady thinker-decode step: one inline-qualifying emit edge, its
    new-token signal, and a loop-back edge. ``share_loopback_uuid=True``
    makes the loop-back edge reference the emit uuid, which disqualifies the
    emit edge from the inline path (condition (c)) — the non-inline case."""
    emit_uuid = f"uuid-emit-{rid}-{step}"
    loop_uuid = emit_uuid if share_loopback_uuid else f"uuid-loop-{rid}-{step}"
    emit_edge = GraphEdge(
        next_node="EMIT_TO_CLIENT", name=TOKEN_EDGE,
        tensor_info=[_tpi(emit_uuid)], output_modality="text",
    )
    token_edge = GraphEdge(
        next_node="EMIT_TO_CLIENT", name=TOKEN_EDGE,
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


def _terminal_routing(rid: str, step: int) -> NodeOutputRouting:
    """The loop-stop terminal step: the steady inline emit edge PLUS the
    loop's declared terminal output (a non-inline full-ResultTensors edge —
    its name is not prematerialized), a persist edge, a peer-worker signal,
    and the worker-graph completion that triggers WGD."""
    routing = _steady_routing(rid, step)
    routing.emit_to_client.append(GraphEdge(
        next_node="EMIT_TO_CLIENT", name=FINAL_EDGE,
        tensor_info=[_tpi(f"uuid-final-{rid}")], output_modality="text",
    ))
    routing.persist = [GraphEdge(
        next_node="CONDUCTOR", name="kv_handle",
        tensor_info=[_tpi(f"uuid-persist-{rid}")], persist=True,
    )]
    routing.to_workers = {"worker_1": [GraphEdge(
        next_node="Talker", name="thinker_hidden",
        tensor_info=[_tpi(f"uuid-peer-{rid}")],
    )]}
    routing.completed_worker_graph_ids = [f"wg-{rid}"]
    return routing


def _prefill_routing(rid: str) -> NodeOutputRouting:
    """A prem-less row (prefill_text / thinker_mixed chunk): its emit edge is
    non-inline (nothing prematerialized) and its new token materializes via
    the worker-side get_tensor().cpu() fallback."""
    emit_uuid = f"uuid-prefill-emit-{rid}"
    return NodeOutputRouting(
        routed_to_this_worker_graph=[],
        is_first_tp_rank=True,
        persist=[],
        to_workers={},
        emit_to_client=[GraphEdge(
            next_node="EMIT_TO_CLIENT", name=TOKEN_EDGE,
            tensor_info=[_tpi(emit_uuid)], output_modality="text",
        )],
        new_token_outputs=[GraphEdge(
            next_node="EMIT_TO_CLIENT", name=TOKEN_EDGE,
            tensor_info=[_tpi(emit_uuid)], conductor_new_token=True,
        )],
    )


# ---------------------------------------------------------------------------
# Drivers — mirror _postprocess_batch's send-loop tail for each path.
# ---------------------------------------------------------------------------

def _wire(rec):
    """The worker→sidecar hop: records cross a pickled ZMQ boundary."""
    return pickle.loads(pickle.dumps(rec))


def _run_step_legacy(worker: _StubWorker, step_ops: list[tuple]) -> None:
    """step_ops: [(rid, routing, nli, prem), ...] in batch order."""
    routing_per_request = {rid: routing for rid, routing, _, _ in step_ops}
    prem_per_request = {rid: prem for rid, _, _, prem in step_ops}
    batch = SimpleNamespace(
        node_objects={rid: None for rid, _, _, _ in step_ops}
    )
    worker._register_outputs(
        batch, routing_per_request,
        prematerialized_per_request=prem_per_request,
    )
    collector: list = []
    for rid, routing, nli, prem in step_ops:
        worker._send_outputs(
            rid, routing,
            nested_loop_indices=nli,
            graph_walk="thinker_decode",
            partition_name="default",
            prematerialized_new_tokens=prem,
            batch_collector=collector,
        )
    if collector:
        worker.communicator.send("api_server", APIServerMessage(
            message_type="result_tensors_batch",
            body=ResultTensorsBatch(items=collector),
        ))


def _run_step_sidecar(
    worker: _StubWorker, state: SidecarState, step_ops: list[tuple]
) -> StepRecord | None:
    routing_per_request = {rid: routing for rid, routing, _, _ in step_ops}
    prem_per_request = {rid: prem for rid, _, _, prem in step_ops}
    batch = SimpleNamespace(
        node_objects={rid: None for rid, _, _, _ in step_ops}
    )
    worker._register_outputs(
        batch, routing_per_request,
        prematerialized_per_request=prem_per_request,
    )
    entries: list = []
    for rid, routing, nli, prem in step_ops:
        entry = worker._send_outputs_sidecar(
            rid, routing,
            nested_loop_indices=nli,
            partition_name="default",
            prematerialized_new_tokens=prem,
        )
        if entry is not None:
            entries.append(entry)
    if not entries:
        return None
    rec = worker._sidecar_client.build_step(entries)
    state.handle(_wire(rec))
    return rec


def _sidecar_setup(rids: list[str]) -> tuple[_StubWorker, SidecarState]:
    worker = _StubWorker(rids)
    worker._sidecar_client = SidecarRecordBuilder()
    state = SidecarState(_RecordingCommunicator(), worker_id="worker_0")
    for rid in rids:
        state.handle(_wire(worker._sidecar_client.register_rid(rid)))
    return worker, state


def _stream(sent: list[tuple], dest: str) -> list[bytes]:
    return [pickle.dumps(msg) for d, msg in sent if d == dest]


def _run_scenario(rids: list[str], steps: list[list[tuple]]):
    """Run one scenario through both paths. Returns (legacy_worker,
    sidecar_worker, sidecar_state, records)."""
    legacy = _StubWorker(rids)
    for step_ops in steps:
        _run_step_legacy(legacy, step_ops)
    sidecar_worker, state = _sidecar_setup(rids)
    records = []
    for step_ops in steps:
        records.append(_run_step_sidecar(sidecar_worker, state, step_ops))
    return legacy, sidecar_worker, state, records


def _assert_byte_identical(
    legacy: _StubWorker, sidecar_worker: _StubWorker, state: SidecarState
) -> None:
    # api_server stream: all-scoped scenarios must emit NOTHING client-bound
    # from the worker; the sidecar's stream must match legacy's byte for
    # byte, in order.
    assert _stream(sidecar_worker.communicator.sent, "api_server") == []
    assert _stream(sidecar_worker.communicator.sent, "conductor") == []
    assert (
        _stream(state.communicator.sent, "api_server")
        == _stream(legacy.communicator.sent, "api_server")
    )
    # conductor stream (WGD bodies included).
    assert (
        _stream(state.communicator.sent, "conductor")
        == _stream(legacy.communicator.sent, "conductor")
    )
    # peer-worker traffic stays on the worker, byte-identical.
    for dest in {d for d, _ in legacy.communicator.sent}:
        if dest in ("api_server", "conductor"):
            continue
        assert (
            _stream(sidecar_worker.communicator.sent, dest)
            == _stream(legacy.communicator.sent, dest)
        )
    # tensor-lifecycle side effects stay on the worker, identical.
    assert (
        sidecar_worker.tensor_manager.deref_calls
        == legacy.tensor_manager.deref_calls
    )
    assert (
        sidecar_worker.tensor_manager.register_calls
        == legacy.tensor_manager.register_calls
    )
    assert (
        sidecar_worker.tensor_manager.persist_calls
        == legacy.tensor_manager.persist_calls
    )


def _assert_worker_never_wrote_accumulators(worker: _StubWorker) -> None:
    """The hardest invariant — wholesale ownership: for scoped
    rids the worker's PerRequestInfo accumulators are never written."""
    for info in worker.worker_graphs_manager.per_request_info.values():
        assert info.pending_new_token_counts == {}
        assert info.current_output_chunks == []
        assert info.output_loop_indices == {}


# ---------------------------------------------------------------------------
# Byte-identity scenarios.
# ---------------------------------------------------------------------------

RIDS = ["rid-a", "rid-b", "rid-c"]


def test_steady_decode_byte_identical():
    """Template step + slim steady steps, 3 rids × 4 steps."""
    steps = [
        [
            (rid, _steady_routing(rid, s), _nli(s), {TOKEN_EDGE: [1000 + s]})
            for rid in RIDS
        ]
        for s in range(4)
    ]
    legacy, sidecar_worker, state, records = _run_scenario(RIDS, steps)
    _assert_byte_identical(legacy, sidecar_worker, state)
    _assert_worker_never_wrote_accumulators(sidecar_worker)
    # Record cheapness: post-template steady records carry no
    # GraphEdge and no NestedLoopIndices — ints, indices, token lists only.
    for rec in records[1:]:
        for _, _new_tokens, items, boundary in rec.entries:
            assert boundary is None
            for _, _, values, layout_idx, wg_fwd, loop_vals, edge in items:
                assert edge is None
                assert isinstance(layout_idx, int)
                assert isinstance(wg_fwd, int)
                assert all(isinstance(v, int) for v in loop_vals)
                assert all(isinstance(v, int) for v in values)


def test_completion_wgd_byte_identical():
    """Steady steps then a loop-stop terminal step: terminal non-inline
    emit + persist flush + peer-worker signal + WGD, all byte-identical
    (the WGD body carries the sidecar-accumulated new_tokens /
    output_signal_names / output_loop_indices)."""
    steps = [
        [
            (rid, _steady_routing(rid, s), _nli(s), {TOKEN_EDGE: [1000 + s]})
            for rid in RIDS
        ]
        for s in range(3)
    ]
    steps.append([
        (rid, _terminal_routing(rid, 3), _nli(3), {TOKEN_EDGE: [1003]})
        for rid in RIDS
    ])
    legacy, sidecar_worker, state, _ = _run_scenario(RIDS, steps)
    _assert_byte_identical(legacy, sidecar_worker, state)
    _assert_worker_never_wrote_accumulators(sidecar_worker)
    # Mechanism-alive: one WGD per rid, from the sidecar.
    assert state.wgd_sent == len(RIDS)
    wgd = [m for d, m in state.communicator.sent if d == "conductor"]
    body = wgd[0].body
    # Since #149 the WGD carries per-signal token COUNTS (numel), not values:
    # 4 tokens accumulated for TOKEN_EDGE across the 4 steps.
    assert body.new_token_counts == {TOKEN_EDGE: 4}
    assert body.output_signal_names == [TOKEN_EDGE] * 4 + [FINAL_EDGE]


def test_multi_partition_wgd_flush_byte_identical():
    """Two completions for one rid (partition-style): the first WGD flushes
    the accumulators, the second starts from empty — flush semantics must
    match the manager's exactly on both paths."""
    rid = "rid-a"
    steps = [
        [(rid, _steady_routing(rid, 0), _nli(0), {TOKEN_EDGE: [7]})],
        [(rid, _terminal_routing(rid, 1), _nli(1), {TOKEN_EDGE: [8]})],
        [(rid, _steady_routing(rid, 2), _nli(2), {TOKEN_EDGE: [9]})],
        [(rid, _terminal_routing(rid, 3), _nli(3), {TOKEN_EDGE: [10]})],
    ]
    legacy, sidecar_worker, state, _ = _run_scenario([rid], steps)
    _assert_byte_identical(legacy, sidecar_worker, state)
    wgd = [m for d, m in state.communicator.sent if d == "conductor"]
    # Per-signal token counts (numel): 2 tokens per completion window.
    assert wgd[0].body.new_token_counts == {TOKEN_EDGE: 2}
    assert wgd[1].body.new_token_counts == {TOKEN_EDGE: 2}


def test_non_inline_shared_uuid_byte_identical():
    """Condition (c): the emit uuid is also on the loop-back edge, so the
    edge must stay on the SHM path — an immediate full result_tensors
    message per rid, in rid order, before any batch message; and the uuid
    must be registered for send. Interleaves with steady steps so the
    template protocol sees the disqualified step in the middle."""
    steps = [
        [
            (rid, _steady_routing(rid, 0), _nli(0), {TOKEN_EDGE: [1000]})
            for rid in RIDS
        ],
        [
            (
                rid,
                _steady_routing(rid, 1, share_loopback_uuid=True),
                _nli(1),
                {TOKEN_EDGE: [1001]},
            )
            for rid in RIDS
        ],
        [
            (rid, _steady_routing(rid, 2), _nli(2), {TOKEN_EDGE: [1002]})
            for rid in RIDS
        ],
    ]
    legacy, sidecar_worker, state, _ = _run_scenario(RIDS, steps)
    _assert_byte_identical(legacy, sidecar_worker, state)
    # Sanity: the disqualified step really produced immediate full sends.
    kinds = [
        m.message_type for d, m in state.communicator.sent
        if d == "api_server"
    ]
    assert kinds.count("result_tensors") == len(RIDS)


def test_mixed_walk_rows_byte_identical():
    """A thinker_mixed-shaped step: decode rows (inline, prematerialized)
    plus a prem-less chunk row whose token rides the worker-side D2H
    fallback and whose emit edge is non-inline — all sidecar-scoped."""
    rids = ["rid-a", "rid-b", "rid-p"]
    steps = [
        [
            ("rid-a", _steady_routing("rid-a", 0), _nli(0), {TOKEN_EDGE: [1]}),
            ("rid-b", _steady_routing("rid-b", 0), _nli(0), {TOKEN_EDGE: [2]}),
        ],
        [
            ("rid-a", _steady_routing("rid-a", 1), _nli(1), {TOKEN_EDGE: [3]}),
            ("rid-b", _steady_routing("rid-b", 1), _nli(1), {TOKEN_EDGE: [4]}),
            ("rid-p", _prefill_routing("rid-p"), _prefill_nli(), None),
        ],
    ]
    legacy, sidecar_worker, state, _ = _run_scenario(rids, steps)
    _assert_byte_identical(legacy, sidecar_worker, state)
    _assert_worker_never_wrote_accumulators(sidecar_worker)


def test_mixed_population_step_per_fifo_identical():
    """One batch with a scoped rid AND a legacy rid (colocated topologies).
    Full-stream byte identity cannot hold here — the step's inline items
    split across two batch messages, one per producer FIFO — the documented
    relaxation. Each population's stream must be byte-identical to a
    single-population legacy run, so the api_server (which routes items
    independently) sees identical per-(rid, name) streams."""
    scoped, legacy_rid = "rid-s", "rid-l"

    def _ops(rid, s):
        return (rid, _steady_routing(rid, s), _nli(s), {TOKEN_EDGE: [s]})

    # Baselines: each rid alone through the legacy path.
    base_scoped = _StubWorker([scoped])
    base_legacy = _StubWorker([legacy_rid])
    for s in range(3):
        _run_step_legacy(base_scoped, [_ops(scoped, s)])
        _run_step_legacy(base_legacy, [_ops(legacy_rid, s)])

    # Mixed-population run: scoped rid via the record path, legacy rid via
    # _send_outputs, in the same step (mirrors _postprocess_batch's per-rid
    # split).
    worker, state = _sidecar_setup([scoped])
    worker.worker_graphs_manager.per_request_info[legacy_rid] = (
        PerRequestInfo(
            node_to_workers={}, dyn_loop_to_workers={},
            worker_graph_ids=[], sharding_config=None,
            per_partition_info={
                "default": PerPartitionInfo(current_fwd_info=_fwd_info(legacy_rid)),
            },
        )
    )
    for s in range(3):
        rid_s, routing_s, nli_s, prem_s = _ops(scoped, s)
        rid_l, routing_l, nli_l, prem_l = _ops(legacy_rid, s)
        batch = SimpleNamespace(node_objects={rid_s: None, rid_l: None})
        worker._register_outputs(
            batch, {rid_s: routing_s, rid_l: routing_l},
            prematerialized_per_request={rid_s: prem_s, rid_l: prem_l},
        )
        collector: list = []
        entry = worker._send_outputs_sidecar(
            rid_s, routing_s, nested_loop_indices=nli_s,
            partition_name="default", prematerialized_new_tokens=prem_s,
        )
        worker._send_outputs(
            rid_l, routing_l, nested_loop_indices=nli_l,
            graph_walk="thinker_decode", partition_name="default",
            prematerialized_new_tokens=prem_l, batch_collector=collector,
        )
        if collector:
            worker.communicator.send("api_server", APIServerMessage(
                message_type="result_tensors_batch",
                body=ResultTensorsBatch(items=collector),
            ))
        state.handle(_wire(worker._sidecar_client.build_step([entry])))

    assert (
        _stream(state.communicator.sent, "api_server")
        == _stream(base_scoped.communicator.sent, "api_server")
    )
    assert (
        _stream(worker.communicator.sent, "api_server")
        == _stream(base_legacy.communicator.sent, "api_server")
    )
    # The legacy rid's accumulators ARE written on the worker; the scoped
    # rid's never are.
    infos = worker.worker_graphs_manager.per_request_info
    assert infos[scoped].pending_new_token_counts == {}
    assert infos[legacy_rid].pending_new_token_counts == {TOKEN_EDGE: 3}


def test_fast_send_flag_invariant_on_sidecar_path():
    """The rider re-landing FAST_SEND's
    empty-register skip by launching with MSTAR_FAST_SEND=1 alongside the
    sidecar. The sidecar path must be byte-invariant to that flag: with it
    on, _send_outputs_sidecar reuses the inline set stashed by
    _register_outputs; with it off, it recomputes — same records, same
    messages, same derefs. The only permitted divergence is the skipped
    empty register_for_send call (the rider itself)."""
    streams = {}
    for fast in (False, True):
        worker, state = _sidecar_setup(RIDS)
        worker._fast_send = fast
        for s in range(3):
            _run_step_sidecar(worker, state, [
                (
                    rid, _steady_routing(rid, s), _nli(s),
                    {TOKEN_EDGE: [1000 + s]},
                )
                for rid in RIDS
            ])
        streams[fast] = (
            _stream(state.communicator.sent, "api_server"),
            _stream(state.communicator.sent, "conductor"),
            worker.tensor_manager.deref_calls,
            [c for c in worker.tensor_manager.register_calls if c[1]],
            len(worker.tensor_manager.register_calls),
        )
    assert streams[False][:4] == streams[True][:4]
    # Steady inline decode: every register set is empty, so flag-on skips
    # them all (the rider) and flag-off keeps the no-op calls.
    assert streams[False][4] == 3 * len(RIDS)
    assert streams[True][4] == 0


def test_rid_remove_drops_sidecar_state():
    rids = ["rid-a"]
    steps = [[
        ("rid-a", _steady_routing("rid-a", 0), _nli(0), {TOKEN_EDGE: [1]}),
    ]]
    _, worker, state, _ = _run_scenario(rids, steps)
    assert state.pending_new_tokens and state.slim_sent
    state.handle(_wire(worker._sidecar_client.remove_rid("rid-a")))
    assert state.rids == {}
    assert state.pending_new_tokens == {}
    assert state.current_output_chunks == {}
    assert state.output_loop_indices == {}
    assert state.slim_sent == set()
    assert state.slim_layout == {}
    assert state.edge_templates == {}


# ---------------------------------------------------------------------------
# MSTAR_SIDECAR_I2T: admission walk-gate widening (worker.py _add_new_request
# builds ``my_walks`` per rid, per worker, then checks ``my_walks <=
# self._sidecar_walks``; these tests exercise that exact subset semantics
# with representative ``my_walks`` sets rather than standing up a full
# Worker/conductor, since the admission plumbing around it — engine_manager,
# tensor_manager, RDMA reads — is unrelated to the walk-gate decision itself).
# ---------------------------------------------------------------------------

def test_sidecar_walks_i2t_extra_composition():
    """The widened set adds exactly the vision walks, on top of the
    untouched base set (flag-off admission decisions are unaffected — see
    worker.py's ``self._sidecar_walks = SIDECAR_WALKS`` when
    MSTAR_SIDECAR_I2T=0)."""
    assert SIDECAR_WALKS == {"thinker_decode", "prefill_text", "thinker_mixed"}
    assert SIDECAR_WALKS_I2T_EXTRA == {
        "prefill_vision", "prefill_multimodal", "encode_vision",
    }
    assert SIDECAR_WALKS.isdisjoint(SIDECAR_WALKS_I2T_EXTRA)


def test_sidecar_admission_gate_i2t_walks():
    """Mirrors worker.py:_add_new_request's admission check
    (``my_walks <= self._sidecar_walks``) for representative ``my_walks``
    sets, without standing up a full Worker."""
    base = SIDECAR_WALKS
    widened = SIDECAR_WALKS | SIDECAR_WALKS_I2T_EXTRA

    # A pure-decode worker (e.g. the decode rank of qwen3omni_2gpu_pd.yaml,
    # or any topology where prefill and decode are on separate node_groups):
    # every request type, i2t included, already clears the BASE set — the
    # new flag changes nothing here (decode was never the problem).
    decode_only = {"thinker_decode"}
    assert decode_only <= base
    assert decode_only <= widened

    # A vision-only prefill worker (prefill_vision isolated on its own
    # node_group, no other modality sharing the rank): excluded by the base
    # set, admitted once MSTAR_SIDECAR_I2T widens it.
    vision_prefill_only = {"prefill_vision"}
    assert not (vision_prefill_only <= base)
    assert vision_prefill_only <= widened

    # Merged prefill (MSTAR_MERGED_PREFILL): same story.
    merged_prefill_only = {"prefill_multimodal"}
    assert not (merged_prefill_only <= base)
    assert merged_prefill_only <= widened

    # Chunked vision (MSTAR_CHUNKED_PREFILL_V2_VISION): the rid's worker
    # graphs span BOTH "encode_vision" and "prefill_vision" on this worker;
    # both must be allowed or the subset check correctly stays excluded.
    chunked_vision = {"encode_vision", "prefill_vision"}
    assert not (chunked_vision <= base)
    assert chunked_vision <= widened

    # Safety invariant: i2s (image input, SPEECH output) still touches
    # Talker/Code2Wav walks on any worker that also hosts them (e.g. the
    # colocated qwen3omni_colocated.yaml topology, where Thinker and Talker
    # share one node_group/worker) — those walks are NOT in either set, so
    # i2s stays excluded from the sidecar regardless of the flag. This is
    # what makes the widening safe without a separate "is this i2t vs i2s"
    # check: the existing whole-rid subset semantics already enforce it.
    i2s_shared_worker = {
        "prefill_vision", "thinker_decode",
        "talker_prefill", "talker_decode",
    }
    assert not (i2s_shared_worker <= base)
    assert not (i2s_shared_worker <= widened)

    # Same invariant for s2t sharing a worker with vision walks (e.g. the
    # PD-disagg prefill rank, which bundles prefill_text/prefill_audio/
    # prefill_vision on one node_group per qwen3omni_2gpu_pd.yaml): audio
    # stays out of scope for THIS flag (a separate idea's territory) even
    # though vision is now admitted in isolation.
    prefill_rank_with_audio = {"prefill_text", "prefill_audio", "prefill_vision"}
    assert not (prefill_rank_with_audio <= base)
    assert not (prefill_rank_with_audio <= widened)


def _prefill_vision_routing(rid: str) -> NodeOutputRouting:
    """Shape of prefill_vision's (and prefill_multimodal's) Thinker node
    output (qwen3_omni_model.py's ``prefill_vision`` Sequential): one
    prem-less EMIT_TO_CLIENT "new_token" text edge (the request's FIRST
    token — prefill is single-shot, nothing prematerialized ahead of it,
    same as ``_prefill_routing`` above) PLUS the two StreamingGraphEdges to
    the Talker partition (thinker_states / thinker_mask), which land in
    ``to_workers`` when Talker is on a different worker (mirrors
    ``_terminal_routing``'s peer-worker signal). This is the row shape
    MSTAR_SIDECAR_I2T newly admits into the sidecar."""
    emit_uuid = f"uuid-vision-emit-{rid}"
    return NodeOutputRouting(
        routed_to_this_worker_graph=[],
        is_first_tp_rank=True,
        persist=[],
        to_workers={"worker_1": [
            GraphEdge(
                next_node="Talker", name="thinker_states",
                tensor_info=[_tpi(f"uuid-states-{rid}")],
            ),
            GraphEdge(
                next_node="Talker", name="thinker_mask",
                tensor_info=[_tpi(f"uuid-mask-{rid}")],
            ),
        ]},
        emit_to_client=[GraphEdge(
            next_node="EMIT_TO_CLIENT", name=TOKEN_EDGE,
            tensor_info=[_tpi(emit_uuid)], output_modality="text",
        )],
        new_token_outputs=[GraphEdge(
            next_node="EMIT_TO_CLIENT", name=TOKEN_EDGE,
            tensor_info=[_tpi(emit_uuid)], conductor_new_token=True,
        )],
    )


def test_prefill_vision_shaped_row_byte_identical():
    """The row MSTAR_SIDECAR_I2T newly admits (first-token emit off a
    prefill_vision walk, including its peer-worker Talker signal) goes
    through the SAME item-processing code as any other sidecar-scoped row —
    no new branch was added, only the admission gate moved. Runs it as the
    rid's first (single-shot) step, then two ordinary decode steps, so the
    scenario matches a real i2t rid's life: one prefill_vision row followed
    by thinker_decode rows, all on one sidecar-scoped rid.

    NOTE on the one documented byte-level exception: the message that ships
    the FIRST inline template immediately after an earlier NON-inline row
    for the same (rid, name) is value-identical but not byte-identical
    across the two paths — a PRE-EXISTING Stage-1 gap, unrelated to
    MSTAR_SIDECAR_I2T (repros with plain prefill_text/thinker_decode, zero
    vision involvement: the interned name string ships in the earlier
    record's ``new_names``, so the later record's template ``GraphEdge``
    references a name reconstructed from a SEPARATE pickle round-trip and
    loses the cross-field identity legacy gets for free in one process —
    see fix20/ideas/s2-sidecar-i2t.md "risks"). No prior test caught it
    because every existing scenario either keeps a rid in one population
    from its first step, or never continues a non-inline row's rid into a
    later inline step for the same name. It affects every modality at the
    prefill-row -> first-decode-token transition, not just i2t; not fixed
    here (out of this change's scope) but pinned as value-equal so a real
    regression (wrong VALUES, not just wrong bytes) still fails loudly."""
    rid = "rid-i2t"
    steps = [
        [(rid, _prefill_vision_routing(rid), _prefill_nli(), None)],
        [(rid, _steady_routing(rid, 0), _nli(0), {TOKEN_EDGE: [1000]})],
        [(rid, _steady_routing(rid, 1), _nli(1), {TOKEN_EDGE: [1001]})],
    ]
    legacy, sidecar_worker, state, _ = _run_scenario([rid], steps)
    assert _stream(sidecar_worker.communicator.sent, "api_server") == []
    assert _stream(sidecar_worker.communicator.sent, "conductor") == []
    legacy_msgs = [m for d, m in legacy.communicator.sent if d == "api_server"]
    sidecar_msgs = [m for d, m in state.communicator.sent if d == "api_server"]
    assert len(legacy_msgs) == len(sidecar_msgs) == 3
    # Step 0 (the prefill_vision row itself) and step 2 (an ordinary slim
    # steady step) are fully byte-identical; step 1 (the post-non-inline
    # template) is the documented value-only exception above.
    assert pickle.dumps(legacy_msgs[0]) == pickle.dumps(sidecar_msgs[0])
    assert legacy_msgs[1] == sidecar_msgs[1]
    assert pickle.dumps(legacy_msgs[2]) == pickle.dumps(sidecar_msgs[2])
    _assert_worker_never_wrote_accumulators(sidecar_worker)
    # The peer-worker (Talker) signal from the prefill_vision row must have
    # gone out, byte-identical to legacy, on both paths.
    peer_msgs = [
        m for d, m in sidecar_worker.communicator.sent if d == "worker_1"
    ]
    assert len(peer_msgs) == 1
    assert peer_msgs[0].body.request_id == rid
    assert (
        _stream(sidecar_worker.communicator.sent, "worker_1")
        == _stream(legacy.communicator.sent, "worker_1")
    )


def test_prefill_shaped_then_inline_template_pickle_gap_is_preexisting():
    """Pins the pre-existing gap documented above using ONLY
    already-shipped walk semantics (prefill_text/thinker_decode shape, no
    vision, no MSTAR_SIDECAR_I2T-specific code) — proof it predates and is
    independent of this change: a prefill-shaped (non-inline) row followed
    by an inline decode step for the same rid+name was simply never
    exercised by the original Stage-1 suite (every non-inline scenario
    there stops after the disqualified row or belongs to a rid that never
    continues). Documents current behavior (value-equal, one byte-level
    exception) rather than asserting it is desirable; a future fix to
    ``SidecarState``/``SidecarRecordBuilder`` that preserves cross-record
    name identity would tighten this to full byte-identity and should
    update this test."""
    rid = "rid-p"
    steps = [
        [(rid, _prefill_routing(rid), _prefill_nli(), None)],
        [(rid, _steady_routing(rid, 0), _nli(0), {TOKEN_EDGE: [1000]})],
        [(rid, _steady_routing(rid, 1), _nli(1), {TOKEN_EDGE: [1001]})],
    ]
    legacy, sidecar_worker, state, _ = _run_scenario([rid], steps)
    legacy_msgs = [m for d, m in legacy.communicator.sent if d == "api_server"]
    sidecar_msgs = [m for d, m in state.communicator.sent if d == "api_server"]
    byte_identical = [
        pickle.dumps(a) == pickle.dumps(b)
        for a, b in zip(legacy_msgs, sidecar_msgs, strict=False)
    ]
    assert byte_identical == [True, False, True]
    assert all(a == b for a, b in zip(legacy_msgs, sidecar_msgs, strict=False))


# ---------------------------------------------------------------------------
# Lifecycle.
# ---------------------------------------------------------------------------

def test_client_hwm_trip_marks_failed_never_blocks():
    """An HWM/NOBLOCK failure marks the client failed permanently (the
    worker then runs _disable_sidecar) — send never blocks or raises."""
    client = SidecarClient.__new__(SidecarClient)
    SidecarRecordBuilder.__init__(client)
    client.worker_id = "worker_t"
    client.failed = False
    client.hwm_trips = 0
    client.records_sent = 0

    class _FullSocket:
        def send_pyobj(self, rec, flags=0):
            raise zmq.Again()

    client._socket = _FullSocket()
    reg = client.register_rid("rid-x")
    assert client.send(reg) is False
    assert client.failed and client.hwm_trips == 1
    # Permanently failed: no further sends are attempted.
    assert client.send(reg) is False
    assert client.hwm_trips == 1


def test_sidecar_process_lifecycle():
    """Spawn the REAL sidecar process (CPU-only), feed it register + steps
    over the real PUSH/PULL pair, receive the batch messages on a fake
    api_server socket, and shut it down within the 5 s drain deadline."""
    import shutil
    import tempfile

    from mstar.communication.communicator import ZMQCommunicator

    prefix = tempfile.mkdtemp(prefix="mstar_sidecar_t_")
    try:
        fake_api_server = ZMQCommunicator(
            my_id="api_server", push_ids=[], ipc_socket_path_prefix=prefix,
        )
        client = SidecarClient(
            worker_id="worker_t", socket_path_prefix=prefix, log_level="INFO",
        )
        try:
            assert client.send(client.register_rid("rid-x"))
            rid_idx = client.rid_idx("rid-x")
            name_idx = client.name_idx(TOKEN_EDGE)
            layout_idx = client.layout_idx(_nli(0))
            for s in range(2):
                edge = None
                if client.ship_edge(rid_idx, name_idx, True):
                    edge = _steady_routing("rid-x", s).emit_to_client[0]
                entry = (
                    rid_idx,
                    [(name_idx, [100 + s])],
                    [(name_idx, 1, [100 + s], layout_idx, s, (s,), edge)],
                    None,
                )
                assert client.send(client.build_step([entry]))

            got: list = []
            deadline = time.monotonic() + 60.0
            while len(got) < 2 and time.monotonic() < deadline:
                got.extend(fake_api_server.get_all_new_messages())
                time.sleep(0.05)
            assert len(got) == 2, f"expected 2 batch messages, got {len(got)}"
            assert all(m.message_type == "result_tensors_batch" for m in got)
            assert type(got[0].body.items[0]).__name__ == "ResultTensors"
            assert type(got[1].body.items[0]).__name__ == "SlimResultTokens"
            assert got[1].body.items[0].values == [101]
            assert client.healthy()
        finally:
            client.shutdown()
        assert not client.proc.is_alive()
    finally:
        shutil.rmtree(prefix, ignore_errors=True)
