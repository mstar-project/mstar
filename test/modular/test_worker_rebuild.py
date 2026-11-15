"""A dropped request comes back through rebuild rows, which the graph runtime never holds.

The runtime keeps the request in its decode walk with its loop-back input pending while rows of the policy's walk
prefill its kept ids. Those rows are rows of the step like any other, so the batch builder, admission, refusals
and failures treat them as rows, and only where the worker hands a step's rows to the graph runtime does it leave
them out. Postprocess takes them off the step first and lands their rebuild, or the runtime would route a
discarded sample or count a row it never scheduled.
"""

from __future__ import annotations

from array import array
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_kv_preemption import PAGE_SIZE, _pool, _running
from test_kv_preemption import _step as _kv_step
from test_kv_prefix_lease import _StubTransfer

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.engine import ExecutingBatch, PagedKVCache
from mstar.engine.resources import AllocationFailed, Residency, StepAllowance, StepContext
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import BatchedModelOutput
from mstar.testing_utils.engine import make_engine
from mstar.testing_utils.model import make_model
from mstar.testing_utils.worker import make_worker, preempting
from mstar.worker import worker as worker_mod
from mstar.worker.engine_manager import InlineStreams
from mstar.worker.executing_batch import build_executing_batch
from mstar.worker.host_pool import HostPool
from mstar.worker.micro_scheduler import CHAIN_LANE, SIDE_LANE, ScheduledBatch
from mstar.worker.rebuild import (
    KeptIds,
    LayoutRow,
    NodeWrites,
    PlannedRow,
    Prefilling,
    RebuildCursor,
    Rebuilds,
    RowRecord,
)
from mstar.worker.speculation import Speculation
from mstar.worker.step_executor import PendingBatch
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import _make_rebuilds

NODE, PREFILL, DECODE = "LLM", "prefill", "decode"
R, A = 7, 8
POLICY = RebuildPolicy(walks=frozenset({PREFILL}), ids_input="text_inputs", sample_output="new_token", loop="loop")


def _graphs() -> dict:
    """``NODE`` reads ``text_inputs`` and outputs ``new_token``, and loop ``loop`` runs it in ``DECODE``."""
    node = SimpleNamespace(input_names={"text_inputs"}, outputs=[SimpleNamespace(name="new_token")])
    loop = SimpleNamespace(get_nodes=lambda: {NODE: node})
    return {
        PREFILL: SimpleNamespace(get_nodes=lambda: {NODE: node}, get_loops=dict),
        DECODE: SimpleNamespace(get_nodes=lambda: {NODE: node}, get_loops=lambda: {"loop": loop}),
    }


def _record(prompt=(1, 2, 3)) -> RowRecord:
    """A prefill step that read ``prompt`` as int64 ids."""
    return RowRecord({"text_inputs": [torch.tensor(prompt)]}, {"is_prefill": True})


def _kept(*sampled: int, prompt=(1, 2, 3)) -> KeptIds:
    """A log of one prefill step that read ``prompt``, and ``sampled``."""
    return KeptIds((LayoutRow(PREFILL, len(prompt)),), [_record(prompt)], array("i", sampled))


def _cleanup(node, rids, wgs):
    assert not rids, "the runtime was asked about a row it never scheduled"
    return [], []


def _worker(**overrides):
    eng = SimpleNamespace(dropped=[], landed=[], released=[])
    eng.residency = lambda _, rid: Residency.DROPPED if rid in eng.dropped else Residency.RESIDENT
    eng.drop_request = lambda _, rid, tokens: eng.dropped.append(rid)
    eng.land_rebuild = lambda node, rid: eng.landed.append(rid)
    eng.release_chunk_inputs = lambda rid, node: eng.released.append(rid)
    eng.release_cached_prefix = lambda batch: None
    tm = SimpleNamespace(get_tensor=None, get_tensors=None, cleanup_collectable=lambda *a: None)
    seen = []
    # keeps the rows the worker hands it in ``seen``, and holds no row, so any routing, stop or loop-count
    # call fails the test
    runtime = SimpleNamespace(
        cleanup_consumed_inputs=_cleanup, clear_pending_loop_stops=lambda: None, get_rid_string=str,
        push_back_node=lambda node, rids, wgs: seen.extend(rids),
        set_in_flight=lambda node, rids, wgs, value: seen.extend(rids),
        commit_speculation=lambda spec_id, ok, dropped, node, wg_ids, scheduled_rids: seen.extend(scheduled_rids),
        speculate_node=lambda node, walks, rows, counts, strides: seen.extend(rows) or [],
    )
    infos = {rid: CurrentForwardPassInfo(str(rid), DECODE, 0, 0, 8) for rid in (R, A)}
    w = make_worker(**{
        "worker_id": "w",
        "tensor_manager": tm,
        "streams": StreamBufferManager(SimpleNamespace(per_request_info={}), tm, []),  # no streams
        "rebuilds": Rebuilds({NODE: POLICY}, _graphs(), HostPool(1 << 10, torch.device("cpu"))),
        "graph_runtime": runtime,
        "engine_manager": SimpleNamespace(
            get_engine=lambda node: eng, engine=SimpleNamespace(paged_kv_caches=list, publisher_nodes={}),
        ),
        "scheduler": SimpleNamespace(
            held_until={}, backlog={}, failed_rids=set(), prefilling=lambda node, rid, walk: None,
        ),
        "request_state": SimpleNamespace(
            get_partition_for_node=lambda node: "p", get_fwd_info=lambda r, p: infos[r],
        ),
        **overrides,
    })
    w._kv_writes = {"kv": {NODE: NodeWrites(frozenset({PREFILL, DECODE}), frozenset({"loop"}))}}
    preempting(w)
    w.seen, w.failed, w.infos = seen, {}, infos
    w.lifecycle.fail = lambda errors, status=500: w.failed.update(errors)
    return w, eng


def _rebuilding(fresh: dict[int, int] | None = None):
    """A worker rebuilding R in one step beside ``fresh`` prompt rows of its walk, and the step it builds."""
    w, eng = _worker()
    # A's prompt row carries no inputs here, so its ids count as kept already
    w._rebuilds._kept[NODE].update({R: _kept(4), A: _kept()})
    eng.dropped.append(R)
    w._rebuilds.start(R, NODE, None)
    batch = ScheduledBatch(NODE, PREFILL, {R: 0, **(fresh or {})}, rebuild_rids=frozenset({R}))
    return w, eng, batch, build_executing_batch(w.pipeline._services, batch)


def _config(growth=(("loop", 1),), labels=("main",), walk=PREFILL) -> dict[str, KVReqConfig]:
    layout = StreamLayout(spans=(PrefixSpan(2, 2, walk),), growth=growth)
    return {"kv": KVReqConfig(kv_layout=dict.fromkeys(labels, layout), loop_caps={loop: 1 for loop, _ in growth})}


@pytest.mark.parametrize("talker_walk", [None, "talk", PREFILL], ids=[
    "own pool", "pool shared with a node that writes nothing for it",
    "pool shared with a node with no rebuild policy that writes for it",
])
def test_a_request_no_rebuild_can_write_again_isnt_rebuildable_in_its_pools(monkeypatch, talker_walk):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    # the streams don't bear on the class, and no resource publishes
    w, _ = _worker(
        streams=SimpleNamespace(add_request=lambda rid, info: None, consumer_node={}),
        wakeup_event=SimpleNamespace(register_futures=lambda futures: None),
    )
    if talker_walk is not None:
        w._kv_writes["kv"]["talker"] = NodeWrites(frozenset({talker_walk}), frozenset())
    kv = _pool()
    w._graph_runtime.add_request, w._graph_runtime.get_sharding_config = lambda request_id, **kw: int(request_id), id
    w._graph_runtime.register_stop_limits = lambda rid, info: None
    w.engine_manager.add_request = lambda rid, configs, info=None: kv.ingest_request(rid, configs["kv"])
    w.request_state.add_request, w.request_state.per_request_info = lambda rid, info: None, defaultdict(dict)
    w.tensor_manager.register_request, w.tensor_manager.start_read_tensors = lambda *a: None, lambda *a, **k: []
    # 12's KV grows in a loop the node's policy doesn't name, so no rebuild can write it again.
    # 13's prompt was written in a walk the node's policy doesn't prefill, so it keeps no ids.
    # 10 lays out nothing, so nothing refuses it.
    configs = {R: _config(), 12: _config(growth=(("embeds_loop", 1),)), 13: _config(walk="audio"),
               10: {"kv": KVReqConfig(kv_layout={})}}
    for rid, config in configs.items():
        info = SimpleNamespace(partition_name="p", graph_walk=PREFILL, resource_configs=config, stream_lead_items={})
        w.ingest._add_new_request(SimpleNamespace(request_id=str(rid), partition_worker_graph_ids=None, loop_caps={},
                                                  worker_graph_to_workers={}, initial_inputs=[], request_info=info))
    refused = {rid for rid in configs if not kv._overrides[rid].rebuildable}
    assert refused == ({R, 12, 13} if talker_walk == PREFILL else {12, 13}), (
        "a request nothing can drop would count as one a drop frees"
    )


def test_postprocess_appends_kept_samples_and_lands_a_rebuild_row_beside_them():
    from test_dropped_row_stream_stop import _engine, _ReachedRoutingError, _Runtime, _Sub, _worker

    eng, landed = _engine(_Sub()), []
    eng.land_rebuild = lambda node, rid: landed.append(rid)
    w, node, walk = _worker(eng, _Runtime(), (R, 1, 2)), "vocoder", "w"
    w._rebuilds._policies, w._rebuilds._kept = {node: POLICY}, {node: {1: _kept()}}
    w._rebuilds._prefilling = {node: set()}
    w._rebuilds._cursors[(node, R)] = RebuildCursor((PlannedRow(PREFILL, 2, 2, _record()),), array("i"))
    batch = _step((R, 1, 2), walk, node)
    eng.prepare_inputs(batch)
    pending = PendingBatch(ScheduledBatch(node, walk, {R: 0, 1: 0, 2: 0}, rebuild_rids=frozenset({R})), batch, node,
                           "p", walk, Future())
    outputs = BatchedModelOutput(per_rid_outputs={rid: {"new_token": [torch.tensor([rid + 4])]} for rid in (1, 2)})
    with pytest.raises(_ReachedRoutingError):  # the harness stops at output routing, after the append and the landing
        w.executor.postprocess(pending, outputs)
    assert list(w._rebuilds._kept[node][1].sampled) == [5], "a drop would find no samples and refuse"
    assert landed == [R], "the rebuilt request would never be ready to decode again"


def test_a_request_with_images_that_samples_a_pad_id_never_leaves_an_older_one_waiting_for_good(monkeypatch):
    """A request with images that samples an image pad keeps no ids from then on, since its rebuild row would give
    the pad an encoder row. Its pool had admitted it as a request a drop frees and was never told otherwise, so
    with no host pool nothing could move it. An older request that ran short beside it waited for it, and it waited
    for the older one."""
    from test_dropped_row_stream_stop import _engine, _ReachedRoutingError, _Runtime, _Sub, _worker

    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv = _pool()
    # R and A hold 3 of the 10 pages each, and the 4 left are what A grows into
    _running(kv, R, 3, arrival=1)
    _running(kv, A, 3, arrival=2)
    eng = _engine(_Sub())
    w, node, walk = _worker(eng, _Runtime(), (R, A)), "vocoder", "w"
    w.engine_manager.engine = SimpleNamespace(paged_kv_caches=lambda: [PagedKVCache("kv", frozenset({node}), kv)])
    preempting(w)
    w._rebuilds._policies = {node: replace(POLICY, pad_ids=frozenset({90}))}
    # A's prompt step read an encoder's output besides its ids
    media = RowRecord({"text_inputs": [torch.tensor([1, 2, 3])], "embeds": [torch.zeros(1, 2)]}, {"is_prefill": True})
    w._rebuilds._kept = {node: {A: replace(_kept(), records=[media])}}
    w._rebuilds._keeps_ids, w._rebuilds._prefilling = {node: {}}, {node: set()}
    batch = _step((R, A), walk, node, rebuild=None)
    eng.prepare_inputs(batch)
    pending = PendingBatch(ScheduledBatch(node, walk, {R: 0, A: 0}), batch, node, "p", walk, Future())
    # A samples the image pad
    outputs = BatchedModelOutput(per_rid_outputs={R: {"new_token": [torch.tensor([5])]},
                                                  A: {"new_token": [torch.tensor([90])]}})
    with pytest.raises(_ReachedRoutingError):
        w.executor.postprocess(pending, outputs)

    for _ in range(2):
        _kv_step(kv, R, PAGE_SIZE)
    for _ in range(4):
        outcome = _kv_step(kv, A, PAGE_SIZE)
        if not outcome.ok:
            # R borrowed what A lends, and A takes it back
            assert kv.preempt(A, outcome.reason.pages_short, [], [R], lambda rid: True).to_drop == [R]
            kv.drop(R, kv._laid_out_tokens(R))
            outcome = _kv_step(kv, A, PAGE_SIZE)
        assert outcome.ok, "A ran short beside an older request that can't move it, and each waits for the other"


@pytest.mark.parametrize("build", ["executing", "speculative"])
def test_postprocess_keeps_a_host_copy_of_what_a_prefill_row_read_besides_its_ids(build):
    from test_dropped_row_stream_stop import _engine, _ReachedRoutingError, _Runtime, _Sub, _worker

    eng = _engine(_Sub())
    eng.land_rebuild = lambda node, rid: None
    w, node, walk = _worker(eng, _Runtime(), (R, 1, 2)), "vocoder", "w"
    w._rebuilds._policies = {node: replace(POLICY, walks=frozenset({walk}))}
    w._rebuilds._copied, w._rebuilds._ids_walks = {node: {walk: frozenset({"embeds"})}}, {node: frozenset({walk})}
    w._rebuilds._kept = {node: {1: KeptIds((LayoutRow(walk, 4),))}}
    w._rebuilds._prefilling = {node: {1}}
    w._rebuilds._pool, w._rebuilds._allowance = (pool := HostPool(1 << 10, torch.device("cpu"))), StepAllowance()
    w._rebuilds._keeps_ids = {node: {1: pool.take(64)}}
    embeds = {rid: torch.full((2,), rid) for rid in (R, 1, 2)}
    batch = _step((R, 1, 2), walk, node, per_request_input_tensors={
        rid: {"embeds": [] if build == "speculative" else [t], "text_inputs": [t]} for rid, t in embeds.items()
    })
    if build == "speculative":
        # a speculated step gets its embeds only once the encoder has run
        spec = Speculation(ScheduledBatch(node, walk, {1: 0, 2: 0}), batch, {("embeds", node)}, {R, 1, 2}, "p",
                           is_new_iter=False, is_same_node=False)
        w.speculator.thread_outputs(spec, BatchedModelOutput({rid: {"embeds": [t]} for rid, t in embeds.items()}))
    for rid, info in batch.per_request_info.items():
        info.graph_walk, info.step_metadata = walk, {"prefill_order": (rid,)}
    eng.prepare_inputs(batch)
    pending = PendingBatch(ScheduledBatch(node, walk, {1: 0, 2: 0}), batch, node, "p", walk, Future())
    outputs = BatchedModelOutput(per_rid_outputs={rid: {"new_token": [torch.tensor([rid + 4])]} for rid in (1, 2)})
    with pytest.raises(_ReachedRoutingError):
        w.executor.postprocess(pending, outputs)
    # what the worker does once the step's outputs are sent
    w._rebuilds.flush()

    assert w._rebuilds._kept[node][1].records[0].inputs["embeds"][0].tolist() == embeds[1].tolist(), (
        "a rebuild would need the encoder to run again"
    )


def test_a_rebuild_row_runs_under_its_recorded_metadata_even_when_empty():
    w, _ = _worker()
    w.infos[R].step_metadata = {"prefill_order": ()}  # decode's metadata
    record = replace(_record(), step_metadata={})
    w._rebuilds._cursors[(NODE, R)] = RebuildCursor((PlannedRow(PREFILL, 2, 2, record),), array("i"))

    node_batch = build_executing_batch(
        w.pipeline._services, ScheduledBatch(NODE, PREFILL, {R: 0}, rebuild_rids=frozenset({R})),
    )

    assert node_batch.per_request_info[R].step_metadata == {}, "the rebuild row would run under decode's metadata"


@pytest.mark.parametrize("why", ["no ids", "backlog", "prefill walk", "samples short"])
def test_a_drop_is_refused_when_its_rebuild_could_not_follow(why):
    w, eng = _worker()
    w._rebuilds._kept[NODE].update({} if why == "no ids" else {R: _kept(5)})
    w.scheduler.backlog.update({(NODE, DECODE): ScheduledBatch(NODE, DECODE, {R: 0})} if why == "backlog" else {})
    w.infos[R].graph_walk = PREFILL if why == "prefill walk" else DECODE
    w.infos[R].dynamic_loop_iter_counts["loop"] = 1 if why == "samples short" else 0

    with pytest.raises(RuntimeError):
        w.preemption._drop(R, NODE)
    assert not eng.dropped, "the request would lose pages a step still reads, or come back as nothing or wrongly keyed"


def test_a_request_whose_kept_ids_dont_rebuild_what_its_kv_holds_is_never_dropped():
    from test_kv_drop import _engine, _manager, _prefill

    w, _ = _worker()
    kv = _manager()
    _prefill(kv, R, 4)
    w.preemption._engine_manager = SimpleNamespace(get_engine=lambda node: _engine({"kv": kv}))
    w._rebuilds._kept[NODE][R] = _kept(5, prompt=(1, 2, 3))  # one prompt id short of what the KV holds

    with pytest.raises(RuntimeError, match="holds 4"):
        w.preemption._drop(R, NODE)
    assert kv.residency(R) is Residency.RESIDENT and kv._streams[R]["main"].page_indices, (
        "the request would lose its KV, and its rebuild would write one token less than it held"
    )


def test_a_lockstep_node_keeps_ids_and_sets_the_pool_aside():
    """Rank 0 journals a lockstep node's drops, and every rank rebuilds the request from the ids it keeps."""
    model = make_model(rebuild_policies=lambda: {NODE: POLICY}, get_graph_walk_graphs=_graphs)

    my_walks = {(NODE, PREFILL), (NODE, DECODE)}
    engine = SimpleNamespace(host_allowance=None, node_dtype=lambda node: torch.float32)
    rebuilds = _make_rebuilds(model, {"rebuild_host_mb": 1}, my_walks, {NODE}, torch.device("cpu"), engine)

    assert rebuilds.pool is not None and NODE in rebuilds.nodes, "a follower would have no ids to replay a drop from"


def _make_media_rebuilds(config: dict) -> Rebuilds:
    """`_make_rebuilds` on a worker whose node copies media."""
    from test_rebuild_policy import MEDIA, VISION
    from test_rebuild_policy import _graphs as media_graphs

    model = SimpleNamespace(rebuild_policies=lambda: {NODE: MEDIA}, get_graph_walk_graphs=media_graphs)
    my_walks = {(NODE, PREFILL), (NODE, VISION), (NODE, DECODE)}
    engine = SimpleNamespace(host_allowance=StepAllowance(), node_dtype=lambda node: torch.float32)
    return _make_rebuilds(model, config, my_walks, set(), torch.device("cpu"), engine)


@pytest.mark.parametrize("config, mb", [({}, 1024), ({"rebuild_host_mb": 2}, 2)])
def test_rebuild_host_mb_alone_sizes_the_pool_of_a_node_that_copies_media(monkeypatch, config, mb):
    """``rebuild_media_host_mb`` was added to ``rebuild_host_mb`` where a node copies media, so two keys sized one
    pool and only ever added up."""
    monkeypatch.setattr(worker_mod, "HostPool", lambda nbytes, device: SimpleNamespace(nbytes=nbytes))

    assert _make_media_rebuilds(config).pool.nbytes == mb << 20, "the pool would be another size than the key says"


def test_a_follower_of_a_lockstep_node_sets_aside_twice_the_pool():
    """A follower holds the logs of requests rank 0 has torn down until it tears them down too."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(worker_mod, "HostPool", lambda nbytes, device: SimpleNamespace(nbytes=nbytes))
    model = SimpleNamespace(rebuild_policies=lambda: {NODE: POLICY}, get_graph_walk_graphs=_graphs)
    engine = SimpleNamespace(host_allowance=StepAllowance(), node_dtype=lambda node: torch.float32)
    my_walks = {(NODE, PREFILL), (NODE, DECODE)}
    try:
        leader = _make_rebuilds(model, {"rebuild_host_mb": 1}, my_walks, set(), torch.device("cpu"), engine)
        follower = _make_rebuilds(
            model, {"rebuild_host_mb": 1}, my_walks, set(), torch.device("cpu"), engine, frozenset({NODE}),
        )
    finally:
        monkeypatch.undo()

    assert leader.pool.nbytes == 1 << 20 and follower.pool.nbytes == 2 << 20, "the follower's pool is rank 0's size"


def test_a_follower_of_a_lockstep_node_takes_room_past_its_pool_for_every_log():
    """Rank 0's pool decides which requests keep a log, so a follower whose pool is full still takes room for one."""
    model = SimpleNamespace(rebuild_policies=lambda: {NODE: POLICY}, get_graph_walk_graphs=_graphs)
    engine = SimpleNamespace(host_allowance=StepAllowance(), node_dtype=lambda node: torch.float32)
    my_walks = {(NODE, PREFILL), (NODE, DECODE)}
    rebuilds = _make_rebuilds(
        model, {"rebuild_host_mb": 1}, my_walks, {NODE}, torch.device("cpu"), engine, frozenset({NODE}),
    )
    writes = {"kv": {NODE: NodeWrites(frozenset({PREFILL, DECODE}), frozenset({"loop"}))}}
    # 6 MiB of ids, past the follower's 2 MiB
    layout = StreamLayout(spans=(PrefixSpan(3 << 18, 3 << 18, PREFILL),), growth=(("loop", 1),))

    ingested = rebuilds.ingest_request(R, writes, {"kv": [layout]}, unpausable=False)

    assert ingested == ({"kv": True}, set()) and rebuilds.kept(R, NODE) is not None, (
        "rank 0 can drop a request the follower keeps no log of"
    )


@pytest.mark.parametrize("engine_add", ["holds a", "refuses a"])
def test_intake_journals_each_log_rank_0s_pool_couldnt_hold_once_the_engine_holds_the_request(engine_add):
    """A follower takes room for every log, so rank 0 journals each one its pool refused, and the follower takes the
    request in as one no drop frees. When the engine refused a, a journaled refusal reached a follower whose KV had
    rolled a back, and the follower stopped."""
    w, eng = _worker(
        rebuilds=Rebuilds({NODE: POLICY}, _graphs(), HostPool(16, torch.device("cpu"))),
        streams=SimpleNamespace(add_request=lambda rid, info: None, consumer_node={}),
        wakeup_event=SimpleNamespace(register_futures=lambda futures: None),
    )
    journaled, held = [], []
    eng.keeps_no_ids = lambda node, rid: journaled.append((node, rid))

    def add_request(rid, configs, info=None):
        if rid == A and engine_add == "refuses a":
            raise RuntimeError("over capacity")
        held.append(rid)
    w.engine_manager.add_request = add_request
    w._graph_runtime.add_request, w._graph_runtime.get_sharding_config = lambda request_id, **kw: int(request_id), id
    w._graph_runtime.register_stop_limits = lambda rid, info: None
    w.request_state.add_request, w.request_state.per_request_info = lambda rid, info: None, defaultdict(dict)
    w.tensor_manager.register_request, w.tensor_manager.start_read_tensors = lambda *a: None, lambda *a, **k: []

    for rid in (R, A):
        info = SimpleNamespace(partition_name="p", graph_walk=PREFILL, resource_configs=_config(), stream_lead_items={})
        w.ingest._add_new_request(SimpleNamespace(request_id=str(rid), partition_worker_graph_ids=None, loop_caps={},
                                                  worker_graph_to_workers={}, initial_inputs=[], request_info=info))

    assert journaled == ([(NODE, A)] if A in held else []), (
        "a follower keeps a log rank 0's pool couldn't hold, or replays a refusal for a request no KV holds"
    )


def test_rebuild_host_mb_0_rebuilds_no_request_there():
    """``rebuild_host_mb: 0`` set no pool aside but left the node among the ones whose requests rebuild here, so
    preemption still ran its pass for the node, as for one with a rebuild policy."""
    rebuilds = _make_media_rebuilds({"rebuild_host_mb": 0})

    assert rebuilds.pool is None and not rebuilds.nodes, "a node with no pool would still count as one that drops"


def test_a_step_of_rebuild_rows_alone_routes_nothing_and_makes_them_resident():
    w, eng, batch, node_batch = _rebuilding()
    assert node_batch.step_context.rebuild_rids == {R} and node_batch.per_request_info[R].graph_walk == PREFILL
    assert node_batch.per_request_input_tensors[R]["text_inputs"][0].tolist() == [1, 2, 3]

    w.executor.postprocess(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())

    assert eng.landed == [R] and not w._rebuilds.rebuilding(R, NODE), "the request would never be ready to decode again"
    assert w.infos[R].graph_walk == DECODE, "the runtime's view of the request left its decode walk"


def test_a_chunk_short_of_the_prompts_end_advances_the_rebuild_and_lands_nothing():
    w, eng = _worker()
    w._rebuilds._kept[NODE][R] = _kept(4, 5)
    eng.dropped.append(R)
    w._rebuilds.start(R, NODE, None)
    batch = ScheduledBatch(NODE, PREFILL, {R: 0}, rebuild_rids=frozenset({R}))
    batch.chunk_ranges[R] = ChunkRange(0, 2, 4, 2)
    node_batch = build_executing_batch(w.pipeline._services, batch)

    w.executor.postprocess(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())

    assert not (eng.landed or eng.released), "the request would decode over KV the rebuild hasn't written"
    assert w._rebuilds._cursors[(NODE, R)].done == 2, "the next chunk would start over"


def test_a_prompts_rebuild_lands_and_leaves_the_prompts_input_to_its_last_chunk():
    w, eng = _worker()
    w.infos[R] = CurrentForwardPassInfo(str(R), PREFILL, 0, 0, 8)
    # its one prefill step was recorded on its first chunk
    w._rebuilds._kept[NODE][R] = _kept()
    eng.dropped.append(R)
    w._rebuilds.start(R, NODE, Prefilling(PREFILL, 4, 10))
    batch = ScheduledBatch(NODE, PREFILL, {R: 0}, rebuild_rids=frozenset({R}))
    batch.chunk_ranges[R] = ChunkRange(0, 4, 4, 4)
    node_batch = build_executing_batch(w.pipeline._services, batch)

    w.executor.postprocess(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())

    assert eng.landed == [R] and not w._rebuilds.rebuilding(R, NODE), "the prompt's next chunk would wait for good"
    assert not eng.released, "the prompt's next chunk would find no input to be cut from"
    assert w.preemption.counts[NODE].rebuild_tokens == 4


@pytest.mark.parametrize(
    "chunk", [None, ChunkRange(0, 2, 4, 2), ChunkRange(2, 4, 4, 2)], ids=["whole", "first", "later"],
)
def test_a_rebuild_rows_ids_go_to_the_device_on_its_first_chunk_only(chunk):
    w, eng = _worker()
    w._rebuilds._kept[NODE][R] = _kept(4, 5)
    eng.dropped.append(R)
    w._rebuilds.start(R, NODE, None)
    batch = ScheduledBatch(NODE, PREFILL, {R: 0}, rebuild_rids=frozenset({R}))
    if chunk is not None:
        batch.chunk_ranges[R] = chunk

    inputs = build_executing_batch(w.pipeline._services, batch).per_request_input_tensors[R]

    if chunk is None or chunk.start == 0:
        assert inputs["text_inputs"][0].tolist() == [1, 2, 3, 4]
    else:
        assert not inputs, "a later chunk is cut from the input the engine prepared on the first"


def test_a_request_dropped_in_its_second_prompt_step_reruns_both_and_its_prompt_goes_on_from_its_own_input():
    """The first row prepares its own input from the log, where the engine keeps the prompt's, so the prompt's input
    waits aside until the row that resumes it, and the rebuild lands only after that row."""
    w, eng = _worker()
    eng.restored = []
    eng.restore_chunk_inputs = lambda rid, node: eng.restored.append(rid)
    w.infos[R] = CurrentForwardPassInfo(str(R), PREFILL, 0, 0, 8)
    rows = (LayoutRow(PREFILL, 3), LayoutRow(PREFILL, 5))
    w._rebuilds._kept[NODE][R] = KeptIds(rows, [_record(), _record((4, 5, 6, 7, 8))])
    eng.dropped.append(R)
    w._rebuilds.start(R, NODE, Prefilling(PREFILL, 2, 5))

    def run(chunk=None):
        batch = ScheduledBatch(NODE, PREFILL, {R: 0}, rebuild_rids=frozenset({R}))
        if chunk is not None:
            batch.chunk_ranges[R] = chunk
        node_batch = build_executing_batch(w.pipeline._services, batch)
        w.executor.postprocess(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())
        return node_batch.per_request_input_tensors.get(R, {})

    first = run()
    assert first["text_inputs"][0].tolist() == [1, 2, 3], "the first row would read the prompt step's input"
    assert eng.released == eng.restored == [R] and not eng.landed, (
        "the first row's input would stay on the device, or the prompt's next chunk would find none"
    )
    assert not run(ChunkRange(0, 2, 2, 2)), "the prompt's row would prepare an input the engine already keeps"

    assert eng.landed == [R] and eng.released == [R], "the prompt's next chunk would find no input to be cut from"


def test_a_skipped_rebuild_row_fails_its_request():
    w, eng, batch, node_batch = _rebuilding()
    # as the engine leaves a row whose prepare_inputs returned None
    node_batch.request_ids = []
    node_batch.skipped_rids.add(R)

    w.executor.postprocess(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())

    assert R in w.failed and not eng.landed, "the request would wait on a rebuild that never runs"


def test_a_rebuild_row_the_prefix_cache_served_whole_lands():
    w, eng, batch, node_batch = _rebuilding()
    # as the engine leaves a row whose walk the prefix cache holds whole: it runs no forward
    node_batch.request_ids = []
    node_batch.skipped_rids.add(R)
    node_batch.cached_rids.add(R)

    w.executor.postprocess(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())

    assert R not in w.failed and eng.landed == [R], "a rebuild the cache served would fail its request"


@pytest.mark.parametrize("site", ["push back", "in flight", "settle", "spec target", "unsubmitted speculation"])
def test_the_graph_runtime_never_sees_a_rebuild_row(site):
    w, _ = _worker()
    batch = ScheduledBatch(NODE, DECODE, {R: 0, A: 1}, rebuild_rids=frozenset({R}))
    speculation = Speculation(scheduled_batch=batch, node_batch=None, consumed_edges=set(), continuing_rids=set(),
                              partition="p", is_new_iter=False, is_same_node=False)
    if site == "push back":
        refused = SimpleNamespace(request_ids=[R, A], cached_rids=set(), no_forward_rids=set(),
                                  admit_error=AllocationFailed(
            message="out of pages", pages_short=1, label="main", request_id=A,
        ))
        w.admission.requeue(batch, refused)
    elif site == "in flight":
        w.executor._set_in_flight(batch, True)
    elif site == "settle":
        w.speculator.settle(speculation, success=True)
    elif site == "spec target":
        w.speculator._choose_spec_target(batch)
    else:
        w.pipeline._handle_error(RuntimeError("boom"), (), None, speculation)

    assert A in w.seen and R not in w.seen, "the runtime would hold a node for a row it never scheduled"
    assert not w.failed, "the rebuild of a step that never ran stays in the log for the next step"


def test_a_terminal_refusal_of_a_rebuild_row_fails_only_its_request():
    w, _ = _worker()
    batch = ScheduledBatch(NODE, PREFILL, {R: 0, A: 1}, rebuild_rids=frozenset({R}))

    w.admission.requeue(batch, SimpleNamespace(
        request_ids=[R, A], cached_rids=set(), no_forward_rids=set(),
        admit_error=SimpleNamespace(terminal=True, request_ids=(R,), message="too long"),
    ))

    assert w.failed.keys() == {R} and w.seen == [A], (
        "the other row would fail, or go back as a row it never was"
    )


def test_a_refused_rebuild_row_is_held_like_any_row():
    from test_admit_failure_handling import _alloc_failed, _refusal_worker, _refused

    worker = _refusal_worker()
    batch = ScheduledBatch(NODE, PREFILL, {R: 0, A: 1}, rebuild_rids=frozenset({R}))

    worker.pipeline._handle_admit_failure(batch, _refused(batch, _alloc_failed(R)))

    assert list(worker.admission.held_until) == [R] and worker._graph_runtime.pushed_back == [A], (
        "the retry would be refused the same way"
    )


def test_a_rebuild_row_stays_out_of_the_runtime_and_publishes_nothing():
    w, eng, batch, node_batch = _rebuilding(fresh={A: 0})

    assert list(node_batch.request_ids) == [R, A] and batch.graph_rows() == {A: 0}, (
        "the runtime would route the rebuild row"
    )
    assert w.lifecycle.publishable(node_batch) == [A], "the rebuild row would publish with MSTAR_LAZY_PUBLISH=0"


def test_a_yield_away_to_rebuild_rows_alone_runs_them(monkeypatch):
    from test_worker_metrics import _looping_worker, _StopLoopError

    import mstar.worker.step_pipeline as pipeline_mod

    w, _, _, postprocessed, _ = _looping_worker(monkeypatch, schedule=[("decode", 1), None, None], speculate={},
                                             refuse=set())
    built = []

    def yield_to_rebuild(pending):
        """A chain that yields to a rebuild, once."""
        if built or pending is None:
            return None, None
        batch = ScheduledBatch(NODE, "rebuild", {R: 0}, rebuild_rids=frozenset({R}))
        built.append(batch)
        # through the module, which the harness stubs for its own steps
        node_batch = pipeline_mod.build_executing_batch(w.pipeline._services, batch)
        return Speculation(scheduled_batch=batch, node_batch=node_batch, consumed_edges=set(),
                           continuing_rids=set(), partition="p", is_new_iter=False, is_same_node=False,
                           is_yield_away=True), None

    w.speculator.build = yield_to_rebuild
    with pytest.raises(_StopLoopError):
        w.run()
    assert postprocessed == ["decode", "rebuild"], "the yield-away step was built and thrown away"


@pytest.mark.parametrize("where", ["being built", "in flight"])
def test_a_step_that_raises_fails_its_rebuild_rows_with_its_other_rows(where):
    w, _ = _worker()
    w.executor.clear_in_flight = lambda batch: None
    batch = ScheduledBatch(NODE, PREFILL, {A: 0, R: 0}, rebuild_rids=frozenset({R}))
    stale = PendingBatch(batch, _step(), NODE, "p", PREFILL, None) if where == "in flight" else None

    w.pipeline._handle_error(RuntimeError("boom"), (stale,), None if stale else batch)

    assert w.failed.keys() == {R, A}, "the dropped request would never be rebuilt or failed"


def _step(rids=(R, A), walk=PREFILL, node=NODE, rebuild=R, **fields) -> ExecutingBatch:
    """A step of ``rids``, where the row ``rebuild`` only recomputes KV."""
    ctx = StepContext(request_ids=rids, graph_walk=walk, slot=0, capture=False, rebuild_rids=frozenset({rebuild}))
    return ExecutingBatch(node_name=node, per_request_info={rid: SimpleNamespace(rid=rid) for rid in rids},
                          step_context=ctx, **fields)


@pytest.mark.parametrize("branch", ["rows_only", "mixed", "tensor"])
def test_a_rebuild_row_leaves_its_requests_loopback_entry(branch):
    engine = make_engine(_loopback_steps={R: (NODE, DECODE), A: (NODE, DECODE)})
    batch = _step(rows_only=branch == "rows_only", loopback_rids=frozenset({A} if branch == "mixed" else ()))
    if branch == "mixed":
        batch.step_context.request_walks = {R: PREFILL, A: DECODE}

    engine._record_loopback(batch)

    assert engine._loopback_steps[R] == (NODE, DECODE), "the resumed decode would not read the last kept id"


def test_a_pass_that_runs_no_step_starts_the_queued_host_copies(monkeypatch):
    """Queued host copies started only once a step's outputs were sent, and a drop waits for them. With every request
    waiting on the pool, no step ran, so no copy started and nothing could be dropped to let one run. Each such pass
    refills the allowance, so a copy larger than one allowance lands over two passes, with no tick between them."""
    from test_loop_turns import TURNS, _looping_worker, _Runtime, _scheduler, _StopLoopError
    from test_rebuild_policy import _copying, _vision_step

    rebuilds = _copying(R)
    # half of the row's 16-byte copy
    rebuilds._allowance = StepAllowance(8)
    rebuilds.record(_vision_step((R,), {}, completion_event=None))
    w = _looping_worker(monkeypatch, *_scheduler(_Runtime(), TURNS))
    w._rebuilds = w.pipeline._rebuilds = rebuilds
    waits = []

    def wait_for_work(timeout_ms):
        waits.append(timeout_ms)
        if len(waits) == 2:
            raise _StopLoopError

    w.communicator = w.pipeline.communicator = SimpleNamespace(send=lambda *a, **k: None, wait_for_work=wait_for_work)
    with pytest.raises(_StopLoopError):
        w.run()

    [record] = rebuilds._kept[NODE][R].records
    assert record.ready(), "the request would wait for its copies with no step to start them"
    assert waits == [0, 10], "the loop would wait its tick while copies are queued"


def test_the_scheduler_sees_the_rows_every_lane_holds():
    w = make_worker()
    shared = w.pipeline.in_flight_rids

    w.pipeline._hold_rows(CHAIN_LANE, [R])
    w.pipeline._hold_rows(SIDE_LANE, [A])
    held = set(shared)
    w.pipeline._hold_rows(CHAIN_LANE, ())

    assert w.scheduler.in_flight_rids is shared and held == {R, A} and shared == {A}, (
        "the scheduler shares the set, and would start a rebuild for a row a step in flight holds"
    )


def test_a_follower_worker_takes_room_past_twice_the_rebuild_pool(monkeypatch):
    """`Worker.__init__` makes its rebuilds for the nodes the worker follows. Without them a follower set aside what
    rank 0 does and refused a log rank 0 kept once its pool was full."""
    monkeypatch.delenv("MSTAR_TP_ASYNC_SCHED", raising=False)
    monkeypatch.setattr(worker_mod, "make_communicator", lambda **kw: SimpleNamespace(
        register_event_for_poll=lambda event: None))
    monkeypatch.setattr(worker_mod, "EventWakeup", object)
    monkeypatch.setattr(worker_mod, "create_tensor_communication_manager", lambda **kw: SimpleNamespace(
        transfer_engine=None, my_session_id="s", get_tensor=None))
    monkeypatch.setattr(worker_mod, "_make_graph_runtime", lambda **kw: SimpleNamespace(
        set_node_metadata=lambda *a, **kw: None, set_inline_streams=lambda keys: None))
    engine = SimpleNamespace(
        get_stop_policies=lambda node, walk: None, paged_kv_caches=list, host_allowance=StepAllowance(),
        node_dtype=lambda node: torch.float32,
    )
    monkeypatch.setattr(worker_mod, "EngineManager", SimpleNamespace(
        build=lambda *a, **kw: SimpleNamespace(
            engine=engine, get_engine=lambda node: engine, side_keys=(),
            inline_streams=InlineStreams(frozenset(), {}, {}),
        )))
    # the worker runs with no model, so its rebuilds come from one that declares ``NODE``'s policy
    model, make = make_model(rebuild_policies=lambda: {NODE: POLICY}, get_graph_walk_graphs=_graphs), _make_rebuilds
    monkeypatch.setattr(
        worker_mod, "_make_rebuilds", lambda m, config, *args: make(model, {"rebuild_host_mb": 1}, *args),
    )
    # rank 1 of two on ``NODE``
    groups = SimpleNamespace(
        init_dist=lambda **kw: None, get_instance_rank_for_node=lambda node: 1,
        get_instance_world_size_for_node=lambda node: 2,
    )
    section = SimpleNamespace(get_nodes=lambda: {NODE}, get_loops=dict)
    worker = worker_mod.Worker(
        worker_id="w1", worker_ids=["w0", "w1"], model=None,
        my_worker_graphs=[SimpleNamespace(section=section, graph_walks={PREFILL, DECODE})],
        model_config={}, all_worker_graph_ids_to_graph_walks={}, all_worker_graph_ids_to_nodes={},
        all_worker_graph_ids_to_dyn_loops={}, sharding_config=None, parallel_groups=groups, device=torch.device("cpu"),
    )
    writes = {"kv": {NODE: NodeWrites(frozenset({PREFILL, DECODE}), frozenset({"loop"}))}}
    # 6 MiB of ids, past the follower's 2 MiB
    layout = StreamLayout(spans=(PrefixSpan(3 << 18, 3 << 18, PREFILL),), growth=(("loop", 1),))

    assert worker.layout.followed_nodes == {NODE}
    assert worker._rebuilds.pool.nbytes == 2 << 20, "the follower's pool is no larger than rank 0's"
    assert worker._rebuilds.ingest_request(R, writes, {"kv": [layout]}, unpausable=False).unheld == set(), (
        "rank 0 can drop a request the follower keeps no log of"
    )
