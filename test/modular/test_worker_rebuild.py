"""A dropped request comes back through a rebuild row, which the graph runtime never sees.

The runtime keeps the request in its decode walk with its loop-back input pending while the row prefills its kept
ids. The scheduler hands the row out with a step of its walk, and every worker path that reads a step's rows has to
know the row is there, or it routes a discarded sample, fails the wrong requests on a refused admit, or loses the
row when its step never goes out.
"""

from __future__ import annotations

from array import array
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from test_kv_preemption import _pool
from test_kv_prefix_lease import _StubTransfer

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import AllocationFailed, KVStep, PositionStep, StepContext
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout, mark_guaranteed
from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import AdmitRuntimeError, Segment
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import ARNodeInputs, BatchedModelOutput
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.rebuild import HostCopies, KeptIds, PrefillStep, RebuildRow, Rebuilds
from mstar.worker.worker import PendingBatch, Speculation, Worker
from mstar.worker.worker_metrics import PreemptionCounts

NODE, PREFILL, VIT, DECODE = "LLM", "prefill", "vit", "decode"
R, A = 7, 8
POLICY = RebuildPolicy(walk=PREFILL, ids_input="text_inputs", sample_output="new_token", loop="loop")


def _graphs() -> dict:
    """``NODE`` reads ``text_inputs`` and outputs ``new_token``, and loop ``loop`` runs it in ``DECODE``. In ``VIT``
    it reads an image's ``img_emb`` alone, as Bagel's LLM does."""
    node = SimpleNamespace(input_names={"text_inputs"}, outputs=[SimpleNamespace(name="new_token")])
    vit = SimpleNamespace(input_names={"img_emb"}, outputs=[])
    loop = SimpleNamespace(get_nodes=lambda: {NODE: node})
    return {
        PREFILL: SimpleNamespace(get_nodes=lambda: {NODE: node}, get_loops=dict),
        VIT: SimpleNamespace(get_nodes=lambda: {NODE: vit}, get_loops=dict),
        DECODE: SimpleNamespace(get_nodes=lambda: {NODE: node}, get_loops=lambda: {"loop": loop}),
    }


def _keep(w: Worker, rid: int, *sampled: int, prompt=(1, 2, 3)) -> None:
    """``rid`` kept a prompt of one step on ``NODE``, and ``sampled``."""
    w._rebuilds._accepted[NODE].add(rid)
    w._rebuilds._kept[NODE][rid] = KeptIds([PrefillStep(PREFILL, (torch.tensor(prompt),))], array("i", sampled))


def _cleanup(node, rids, wgs):
    assert not rids, "the runtime was asked about a row it never scheduled"
    return [], []


class _Scheduler:
    """The micro-scheduler's rebuild queue and holds, as the worker uses them."""

    def __init__(self, held: list):
        self.rows, self.taken, self.held_until, self.backlog, self.failed_rids = {}, [], {}, {}, set()
        self.hold_requests = held.extend

    def queue_rebuild(self, rid, row):
        self.rows[rid] = row

    def rebuild_queued(self, rid):
        return rid in self.rows

    def take_rebuilds(self, rids):
        self.taken.extend(rids)
        for rid in rids:
            self.rows.pop(rid, None)


def _worker():
    eng = SimpleNamespace(dropped=[], landed=[], prepared=[], is_offloaded=lambda node, rid: False)
    eng.prepare_rebuild = lambda node, rid, rebuild: eng.prepared.append((rid, rebuild))
    eng.is_resident, eng.drop_request = lambda _, rid: rid not in eng.dropped, lambda _, rid: eng.dropped.append(rid)
    eng.land_rebuild = lambda node, rid: eng.landed.append(rid)
    w = Worker.__new__(Worker)
    w.worker_id, w._phase_period, w.enable_nvtx, w.enable_prof, w.device = "w", 0, False, False, torch.device("cpu")
    w._in_flight_rids, w._last_active, w._stream_buffer_rids = set(), {}, set()
    w._rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    w._kv_keys, w._rebuilding = {NODE: ["kv"]}, frozenset()
    w._dropped, w._victims, w._preempted, w._preemptions = {}, {}, {}, defaultdict(PreemptionCounts)
    w._pending_removes, w._pending_drains, w._draining_rids = {}, set(), set()
    # holds no row, so any routing, stop or loop-count call fails the test
    w._graph_runtime = SimpleNamespace(cleanup_consumed_inputs=_cleanup, clear_pending_loop_stops=lambda: None,
                                       get_rid_string=str, push_back_node=lambda node, rids, wgs: w.pushed.extend(rids))
    w.engine_manager = SimpleNamespace(get_engine=lambda node: eng)
    w.held, w.pushed, w.failed = [], [], {}
    w.scheduler = _Scheduler(w.held)
    w.infos = {rid: CurrentForwardPassInfo(str(rid), DECODE, 0, 0, 8) for rid in (R, A)}
    w.request_state = SimpleNamespace(get_partition_for_node=lambda node: "p", get_fwd_info=lambda r, p: w.infos[r])
    w.tensor_manager = SimpleNamespace(get_tensor=None, cleanup_collectable=lambda *a: None)
    w._fail_requests = w.failed.update
    return w, eng


def _rebuilding(fresh: dict[int, int] | None = None):
    """A worker with R dropped and its rebuild row queued, and the step the worker builds when the scheduler
    hands that row out with ``fresh`` prompt rows of its walk."""
    w, eng = _worker()
    # A's prompt row carries no inputs here, so its ids count as kept already
    _keep(w, R, 4)
    _keep(w, A)
    eng.dropped.append(R)
    w._rebuild(R, NODE)
    batch = ScheduledBatch(NODE, PREFILL, dict(fresh or {}), rebuild_rows=dict(w.scheduler.rows))
    return w, eng, batch, w._build_executing_batch(batch)


def _config(growth=(("loop", 1),), labels=("main",), walk=PREFILL) -> dict[str, KVReqConfig]:
    layout = StreamLayout(spans=(PrefixSpan(2, 2, walk),), growth=growth)
    return {"kv": KVReqConfig(kv_layout=dict.fromkeys(labels, layout), loop_caps={loop: 1 for loop, _ in growth})}


def test_a_rebuild_tells_the_node_what_its_row_writes_and_queues_the_row():
    w, eng, _, _ = _rebuilding()

    [(rid, rebuild)] = eng.prepared
    assert (rid, rebuild.walk, list(rebuild.ids)) == (R, PREFILL, [1, 2, 3, 4]), "the KV would lay out other ids"
    assert w.scheduler.rebuild_queued(R), "the row would never reach a step"


@pytest.mark.parametrize("shared", [False, True], ids=["own pool", "pool shared with a node with no rebuild policy"])
def test_a_request_no_rebuild_can_write_again_is_guaranteed_in_its_pools(monkeypatch, shared):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    w, _ = _worker()
    if shared:
        w._kv_keys["talker"] = ["kv"]
    kv, w._my_consumer_connections, w._unprocessed_messages = _pool(), [], {}
    w._graph_runtime.add_request, w._graph_runtime.get_sharding_config = lambda request_id, **kw: int(request_id), id
    w.engine_manager.evictable_nodes = list
    w.engine_manager.add_request = lambda rid, configs, info=None: kv.ingest_request(rid, configs["kv"])
    w.request_state.add_request, w.request_state.per_request_info = lambda rid, info: None, defaultdict(dict)
    w.tensor_manager.register_request, w.tensor_manager.start_read_tensors = lambda *a: None, lambda *a, **k: []
    w.wakeup_event = SimpleNamespace(register_futures=lambda futures: None)
    # 12's KV grows in a loop the node's policy doesn't name, so no rebuild can write it again.
    # 13's prompt was written in a walk the node's policy doesn't prefill, so it keeps no ids.
    # 10 lays out nothing, so it isn't guaranteed.
    configs = {R: _config(), 12: _config(growth=(("embeds_loop", 1),)), 13: _config(walk="audio"),
               10: {"kv": KVReqConfig(kv_layout={})}}
    for rid, config in configs.items():
        info = SimpleNamespace(partition_name="p", graph_walk=PREFILL, resource_configs=config, stream_lead_items={})
        w._add_new_request(SimpleNamespace(request_id=str(rid), partition_worker_graph_ids=None, loop_caps={},
                                           worker_graph_to_workers={}, initial_inputs=[], request_info=info))
    guaranteed = {rid for rid in configs if kv._overrides[rid].guaranteed}
    assert guaranteed == ({R, 12, 13} if shared else {12, 13}), (
        "a request nothing can drop would reserve only its prompt"
    )


def test_postprocess_appends_kept_samples_and_lands_a_rebuild_row_beside_them():
    from test_dropped_row_stream_stop import _engine, _ReachedRoutingError, _Runtime, _Sub, _worker

    eng, landed = _engine(_Sub()), []
    eng.land_rebuild = lambda node, rid: landed.append(rid)
    w, node, walk = _worker(eng, _Runtime()), "vocoder", "w"
    w._rebuilds._policies = {node: POLICY}
    w._rebuilds._kept = {node: {1: KeptIds([PrefillStep(walk, (torch.tensor([1]),))]), R: KeptIds()}}
    batch = _step((R, 1, 2), walk, node)
    eng.prepare_inputs(batch)
    pending = PendingBatch(ScheduledBatch(node, walk, {1: 0, 2: 0}), batch, node, "p", walk, Future())
    outputs = BatchedModelOutput(per_rid_outputs={rid: {"new_token": [torch.tensor([rid + 4])]} for rid in (1, 2)})
    with pytest.raises(_ReachedRoutingError):  # the harness stops at output routing, after the append and the landing
        w._postprocess_batch(pending, outputs)
    assert list(w._rebuilds._kept[node][1].sampled) == [5], "a drop would find no samples and refuse"
    assert landed == [R], "the rebuilt request would never be ready to decode again"


@pytest.mark.parametrize("build", ["executing", "speculative"])
def test_postprocess_keeps_a_host_copy_of_what_a_prefill_row_read_besides_its_ids(build):
    from test_dropped_row_stream_stop import _engine, _ReachedRoutingError, _Runtime, _Sub, _worker

    eng = _engine(_Sub())
    eng.land_rebuild = lambda node, rid: None
    w, node, walk = _worker(eng, _Runtime()), "vocoder", "w"
    w._rebuilds._policies = {node: replace(POLICY, copied_inputs={walk: ("embeds",)})}
    w._rebuilds._accepted, w._rebuilds._kept = {node: {1}}, {node: {R: KeptIds()}}
    embeds = {rid: torch.full((2,), rid) for rid in (R, 1, 2)}
    batch = _step((R, 1, 2), walk, node, per_request_input_tensors={
        rid: {"embeds": [] if build == "speculative" else [t], "ids": [t]} for rid, t in embeds.items()
    })
    if build == "speculative":
        # a speculated step gets its embeds only once the encoder has run
        spec = Speculation(ScheduledBatch(node, walk, {1: 0, 2: 0}), batch, {("embeds", node)}, {R, 1, 2}, "p",
                           is_new_iter=False, is_same_node=False)
        w._thread_outputs_to_speculative(spec, BatchedModelOutput({rid: {"embeds": [t]} for rid, t in embeds.items()}))
    for rid, info in batch.per_request_info.items():
        info.step_metadata = {"prefill_order": (rid,)}
    w.request_state.get_partition_for_node, w.request_state.get_fwd_info = lambda node: "p", batch.per_request_info.get
    eng.prepare_inputs(batch)
    pending = PendingBatch(ScheduledBatch(node, walk, {1: 0, 2: 0}), batch, node, "p", walk, Future())
    outputs = BatchedModelOutput(per_rid_outputs={rid: {"new_token": [torch.tensor([rid + 4])]} for rid in (1, 2)})
    with pytest.raises(_ReachedRoutingError):
        w._postprocess_batch(pending, outputs)

    (step,) = w._rebuilds._kept[node][1].steps
    assert step.copies.inputs == {"embeds": [embeds[1]]}, "a rebuild would need the encoder to run again"
    assert step.copies.step_metadata == {"prefill_order": (1,)}, "the rebuild row would run under decode's metadata"
    assert w._rebuilds._kept[node][R].steps == [], "a rebuild row would be kept as a step of its prefill"


def _rebuild_rows(w: Worker, run) -> list[RebuildRow]:
    """Run each rebuild row the scheduler holds, through ``run(batch, node_batch)``, until none is left."""
    rows = []
    while w.scheduler.rows:
        ((rid, row),) = w.scheduler.rows.items()
        w.scheduler.take_rebuilds((rid,))  # as the step's launch does
        batch = ScheduledBatch(row.node, row.walk, rebuild_rows={rid: row})
        rows.append(row)
        run(batch, w._build_executing_batch(batch))
    return rows


def test_a_prompt_on_several_walks_rebuilds_a_row_per_step_in_order_and_lands_after_the_last():
    w, eng = _worker()
    w._rebuilds = Rebuilds({NODE: replace(POLICY, copied_inputs={VIT: ("img_emb",)})}, _graphs())
    images = [torch.ones(4, 2), torch.zeros(4, 2)]
    w._rebuilds._accepted[NODE].add(R)
    w._rebuilds._kept[NODE][R] = KeptIds([
        PrefillStep(PREFILL, (torch.tensor([1, 2]),)),
        *[PrefillStep(VIT, (), HostCopies({"img_emb": [image]}, {"image": i})) for i, image in enumerate(images)],
        PrefillStep(PREFILL, (torch.tensor([3, 4, 5]),)),
    ], array("i", [6, 7]))
    eng.dropped.append(R)
    w._rebuild(R, NODE)

    served = []

    def run(batch, node_batch):
        assert not eng.landed, "the request would decode before its KV is whole"
        if batch.graph_walk == VIT and not served:
            # the cache serves the first image whole, as `Engine._skip_cached_prefix` takes the row out of the step
            node_batch.drop_rids({R})
            node_batch.cached_rids.add(R)
            served.append(R)
        pending = PendingBatch(batch, node_batch, NODE, "p", batch.graph_walk, Future())
        w._postprocess_batch(pending, BatchedModelOutput())

    rows = _rebuild_rows(w, run)

    assert [row.walk for row in rows] == [PREFILL, VIT, VIT, PREFILL], "a row would write another walk's span"
    assert [row.inputs.get("img_emb") for row in rows[1:3]] == [[images[0]], [images[1]]], (
        "an image row would read another image"
    )
    assert [row.step_metadata for row in rows] == [None, {"image": 0}, {"image": 1}, None], (
        "an image row would run under decode's metadata, or another step's"
    )
    assert rows[3].inputs["text_inputs"][0].tolist() == [3, 4, 5, 6], "the samples would follow another step"
    assert eng.landed == [R], "a row the cache served would leave its request dropped"


def _models() -> dict:
    """Bagel, Qwen3-Omni and Higgs Audio, with the node that decodes text and its paged KV cache."""
    from test_model_kv_layouts import _bagel, _higgs

    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    bagel, omni = _bagel(), object.__new__(Qwen3OmniModel)
    bagel.config.max_position_embeddings, bagel.config.num_timesteps, bagel._image_gen_remote_handoff = 4096, 3, False
    omni.config = Qwen3OmniModelConfig()
    return {"Bagel": (bagel, "LLM", "kv"), "Qwen3-Omni": (omni, "Thinker", "thinker_kv"),
            "Higgs": (_higgs(), "LLM", "kv_cache")}


def _from_model(model, node: str, cache: str) -> tuple[Worker, SimpleNamespace]:
    """A worker that rebuilds ``node``'s requests as ``model`` declares, checked against its own walks."""
    w, eng = _worker()
    w._kv_keys = {node: [cache]}
    w._rebuilds = Rebuilds(model.rebuild_policies(), model.get_graph_walk_graphs())
    return w, eng


def _media(walk: str) -> PrefixSpan:
    return PrefixSpan(20, 1, walk, kind="digest")


# Each request as its model lays it out: the spans, how many spans the model keys, whether the request carries prefix
# keys (it does while the prefix cache is on), the labels, whether the request speaks its answer, and whether it is
# preemptible. A Bagel edit in think mode keys only its leading text, and so does a Qwen3-Omni prompt with a video.
_TEXT, _CLOSE = PrefixSpan(5, 5, "prefill_text"), PrefixSpan(4, 4, "prefill_text")
_EDIT = (_TEXT, _media("prefill_vae"), _media("prefill_vit"), _CLOSE)
_VIDEO = (_TEXT, _media("prefill_vision"), _CLOSE)
_MEDIA_REQUESTS = {
    "Bagel image chat": ("Bagel", (_TEXT, _media("prefill_vit"), _CLOSE), 3, True, 1, False, True),
    "Bagel chat with two images": ("Bagel", (_TEXT, *[_media("prefill_vit")] * 2, _CLOSE), 4, True, 1, False, True),
    "Bagel edit with think mode": ("Bagel", _EDIT, 1, True, 1, False, False),
    "Bagel edit with think mode, cache off": ("Bagel", _EDIT, 1, False, 1, False, True),
    "Bagel edit or guided generation": ("Bagel", (_TEXT, _media("prefill_vit"), _CLOSE), 3, True, 3, False, False),
    "Qwen3-Omni audio in, text out": ("Qwen3-Omni", (_TEXT, _media("prefill_audio"), _CLOSE), 3, True, 1, False, True),
    "Qwen3-Omni image in, text out": ("Qwen3-Omni", (_TEXT, _media("prefill_vision"), _CLOSE), 3, True, 1, False, True),
    "Qwen3-Omni video in, text out": ("Qwen3-Omni", _VIDEO, 1, True, 1, False, False),
    "Qwen3-Omni video in, cache off": ("Qwen3-Omni", _VIDEO, 1, False, 1, False, True),
    "Qwen3-Omni audio out": ("Qwen3-Omni", (_TEXT, _media("prefill_audio"), _CLOSE), 3, True, 1, True, False),
    "Higgs transcription": ("Higgs", (_TEXT, _media("prefill_audio"), _CLOSE), 3, True, 1, False, True),
}


@pytest.mark.parametrize("request_", _MEDIA_REQUESTS)
def test_a_media_request_is_preemptible_only_where_its_models_rebuild_rows_can_write_it_again(request_):
    name, spans, keyed, cached, labels, speaks, preemptible = _MEDIA_REQUESTS[request_]
    model, node, cache = _models()[name]
    w, _ = _from_model(model, node, cache)
    ((_, loop),) = model.kv_labels()[cache]["main"].growth
    layout = StreamLayout(spans=spans, growth=((loop, 1),), keyed_spans=keyed)
    # the CFG nodes write their own copy of the prompt under each guidance label
    guidance = dict.fromkeys(("cfg_text", "cfg_img")[:labels - 1], StreamLayout(spans=(_TEXT,)))
    config = KVReqConfig(
        loop_caps={loop: 8}, prefix_keys={"main": [b"k"]} if cached else None, kv_layout={"main": layout, **guidance},
    )
    # an answer spoken aloud has the Talker write rows of the Thinker's states
    talker = KVReqConfig(kv_layout={"main": StreamLayout(spans=(PrefixSpan(9, 9, "talker_prefill", kind="rows"),))})
    configs = {cache: config, **({"talker_kv": talker} if speaks else {})}
    mark_guaranteed(configs, streaming=False, output_modalities=["text", "audio"] if speaks else ["text"])

    w._guarantee(R, configs)

    assert config.guaranteed == (not preemptible), "a request no rows can rebuild would be preempted, or the reverse"
    assert (R in w._rebuilds._accepted.get(node, set())) == preemptible, (
        "a guaranteed request would record prefill steps no row reads, or a preemptible one would record none"
    )


@pytest.mark.parametrize("name, digest_walk, input_name", [
    ("Bagel", "prefill_vit", "img_emb"), ("Bagel", "prefill_vae", "img_emb"),
    ("Qwen3-Omni", "prefill_audio", "audio_embeds"), ("Higgs", "prefill_audio", "audio_embeds"),
])
def test_a_rebuilt_prompt_on_several_walks_holds_the_pages_and_positions_of_an_uninterrupted_run(
    monkeypatch, name, digest_walk, input_name,
):
    from test_bagel_guidance_sink import _llm
    from test_kv_drop import _manager

    from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule
    from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

    # every commit checks the walk that wrote each span
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    model, node, cache = _models()[name]
    w, eng = _from_model(model, node, cache)
    ((decode, loop),) = model.kv_labels()[cache]["main"].growth
    declare = {
        "Bagel": _llm().declare_step, "Qwen3-Omni": partial(ThinkerSubmodule.declare_step, None),
        "Higgs": partial(HiggsAudioLLMSubmodule.declare_step, None),
    }[name]
    kv, rope, starts = _manager(), RopeManager(PositionConfig(kv_cache=cache), torch.device("cpu")), []

    def write(walk, length):
        starts.append(rope.position(R, "main"))
        steps = declare(walk, [R], [ARNodeInputs(input_seq_len=length)], per_request_info={}).steps.values()
        (kv_step,), (position,) = [[s for s in steps if isinstance(s, kind)] for kind in (KVStep, PositionStep)]
        segments = (Segment(R, "main", length),)
        kv_step, ctx = replace(kv_step, segments=segments), StepContext((R,), walk, 0, False)
        assert kv.admit(kv_step, ctx).ok, f"the pool refused {walk}"
        kv.plan(kv_step, ctx)
        kv.commit(kv_step, ctx)
        rope.commit(replace(position, segments=segments), ctx)

    # An attachment holds its rows and two sentinels. A Bagel image moves the position by one, and audio by its length.
    rows, attachment = torch.zeros(18, 2), _media(digest_walk)
    spans = (_TEXT, attachment if name == "Bagel" else replace(attachment, advance=attachment.length), _CLOSE)
    layout = StreamLayout(spans=spans, growth=((loop, 1),))
    kv.ingest_request(R, KVReqConfig(kv_layout={"main": layout}, loop_caps={loop: 8}))
    rope.ingest_request(R)
    for span in spans:
        write(span.walk, span.length)
    for _ in range(2):
        write(decode, 1)
    stream = kv._streams[R]["main"]
    held, uninterrupted = (stream.stored_len, len(stream.page_indices), rope.position(R, "main")), starts[:3]
    kv.drop(R)
    rope.drop(R)
    eng.prepare_rebuild = lambda node, rid, rebuild: kv.prepare_rebuild(rid, rebuild)
    eng.land_rebuild = lambda node, rid: kv.land_rebuild(rid)
    assert w._rebuilds.keep(R, node, [KVReqConfig(kv_layout={"main": layout})])
    copied = model.rebuild_policies()[node].copied_inputs[digest_walk]
    copies = HostCopies({copied_name: [rows] if copied_name == input_name else [] for copied_name in copied}, {})
    # the samples before 9 are in the KV, and 9 is the next decode step's input
    w._rebuilds._kept[node][R] = KeptIds([
        PrefillStep("prefill_text", (torch.arange(5),)), PrefillStep(digest_walk, (), copies),
        PrefillStep("prefill_text", (torch.arange(5, 9),)),
    ], array("i", [7, 8, 9]))
    eng.dropped.append(R)
    w._rebuild(R, node)
    starts.clear()

    def run(batch, node_batch):
        inputs = node_batch.per_request_input_tensors[R]
        write(batch.graph_walk, len(inputs["text_inputs"][0]) if "text_inputs" in inputs else len(rows) + 2)
        w._land_rebuilds(node_batch)

    _rebuild_rows(w, run)

    assert starts == uninterrupted, "a row would place its span where the uninterrupted prefill didn't"
    assert (stream.stored_len, len(stream.page_indices), rope.position(R, "main")) == held, (
        "the rebuilt stream would hold other slots, pages or positions than before its drop"
    )


def test_a_rebuild_row_runs_under_its_copied_metadata_even_when_empty():
    w, _ = _worker()
    w.infos[R].step_metadata = {"prefill_order": ()}  # decode's metadata
    row = RebuildRow(NODE, PREFILL, {"text_inputs": [torch.tensor([1, 2])]}, {})

    node_batch = w._build_executing_batch(ScheduledBatch(NODE, PREFILL, rebuild_rows={R: row}))

    assert node_batch.per_request_info[R].step_metadata == {}, "the rebuild row would run under decode's metadata"


@pytest.mark.parametrize("why", ["no ids", "in flight", "backlog", "rebuild queued", "prefill walk", "samples short"])
def test_a_drop_is_refused_when_its_rebuild_could_not_follow(why):
    w, eng = _worker()
    if why != "no ids":
        _keep(w, R, 5)
    w._in_flight_rids.update({R} if why == "in flight" else ())
    w.scheduler.backlog.update({(NODE, DECODE): ScheduledBatch(NODE, DECODE, {R: 0})} if why == "backlog" else {})
    w.scheduler.rows.update({R: RebuildRow(NODE, PREFILL, {})} if why == "rebuild queued" else {})
    w.infos[R].graph_walk = PREFILL if why == "prefill walk" else DECODE
    w.infos[R].dynamic_loop_iter_counts["loop"] = 1 if why == "samples short" else 0

    with pytest.raises(RuntimeError):
        w._drop(R, NODE)
    assert not eng.dropped, "the request would lose pages a step still reads, or come back as nothing or wrongly keyed"


@pytest.mark.parametrize("state", ["resident", "queued", "in its step"])
def test_a_request_is_rebuilt_only_once_per_drop(state):
    w, eng = _worker()
    _keep(w, R, 5)
    eng.dropped += [] if state == "resident" else [R]
    w.scheduler.rows.update(queued := {R: RebuildRow(NODE, PREFILL, {})} if state == "queued" else {})
    if state == "in its step":
        w._in_flight_rids, w._rebuilding = {R}, frozenset({R})
    with pytest.raises(RuntimeError):
        w._rebuild(R, NODE)
    assert w.scheduler.rows == queued and not eng.prepared, (
        "a rebuild would lay out a stream that is live or already being rebuilt"
    )


def test_a_dropped_request_is_rebuilt_while_a_step_of_another_node_holds_it():
    """Orpheus's SNAC decoder can run on the LLM's worker, so a SNAC step can be in flight when the request's LLM
    KV is rebuilt."""
    w, eng = _worker()
    _keep(w, R, 5)
    eng.dropped.append(R)
    w._in_flight_rids.add(R)

    w._rebuild(R, NODE)

    assert w.scheduler.rebuild_queued(R), "the request never comes back, or its worker's main loop raises"


def test_a_step_of_rebuild_rows_alone_routes_nothing_and_makes_them_resident():
    w, eng, batch, node_batch = _rebuilding()
    assert node_batch.rebuild_rids == {R} and node_batch.per_request_info[R].graph_walk == PREFILL
    assert node_batch.per_request_input_tensors[R]["text_inputs"][0].tolist() == [1, 2, 3]

    w._postprocess_batch(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())

    assert eng.landed == [R], "the request would never be ready to decode again"
    assert w.infos[R].graph_walk == DECODE, "the runtime's view of the request left its decode walk"
    assert (R, NODE) in w._last_active, "the rebuilt request would be the coldest eviction victim"


def test_a_rebuild_row_leads_its_step_and_stays_out_of_the_runtime():
    w, eng, batch, node_batch = _rebuilding(fresh={A: 0})

    assert list(node_batch.request_ids) == [R, A], "the step's rows would not lead with the rebuild row"
    assert list(batch.request_to_worker_graph) == [A], "the runtime would route the rebuild row"
    assert w._publishable_request_ids(node_batch) == [A], "the rebuild row would publish with MSTAR_LAZY_PUBLISH=0"


def test_the_run_loop_takes_a_rebuild_row_once_its_step_goes_out_and_a_remove_waits_for_it(monkeypatch):
    from test_worker_metrics import _looping_worker, _StopLoopError

    w, _, postprocessed, _ = _looping_worker(monkeypatch, schedule=[None, None], speculate={}, refuse=set())
    w._pending_removes, w._rid = {}, lambda wire_rid: wire_rid
    w._graph_runtime = SimpleNamespace(get_sharding_config=lambda rid: pytest.fail("removed under its rebuild step"))
    queue, taken, calls, build = {R: RebuildRow(NODE, "rebuild", {})}, [], [], w._build_executing_batch
    w.scheduler.take_rebuilds = lambda rids: taken.extend(rids) or [queue.pop(rid) for rid in rids]
    after = w.scheduler.get_next_batch

    def next_batch(state, **kw):
        """The queued row while it is queued, twice at most, then the harness's steps, which end the loop."""
        if queue and len(calls) < 2:
            calls.append(None)
            return ScheduledBatch(NODE, "rebuild", rebuild_rows=dict(queue))
        return after(state, **kw)

    w.scheduler.get_next_batch = next_batch
    w._handle_main_loop_error = lambda *args: None  # the loop keeps running, as with the real handler

    def build_after_one_raise(batch):
        if len(calls) == 1:
            raise RuntimeError("the step's inputs failed")  # before `_set_pending`, so the row must stay queued
        node_batch = build(batch)
        node_batch.rebuild_rids = frozenset(batch.rebuild_rows)
        return node_batch

    w._build_executing_batch = build_after_one_raise
    # Remove R once, on the pass after its step went out, while that step is still pending.
    w._process_messages = lambda: taken and not w._pending_removes and w._remove_request(SimpleNamespace(request_id=R))
    with pytest.raises(_StopLoopError):
        w.run()
    assert taken == [R] and postprocessed == ["rebuild"], "a raise before `_set_pending` would lose the rebuild row"
    assert w._pending_removes == {R: None}, "the REMOVE tore its request down mid-step"


def test_a_yield_away_to_rebuild_rows_alone_runs_them(monkeypatch):
    from test_worker_metrics import _looping_worker, _StopLoopError

    w, _, postprocessed, _ = _looping_worker(monkeypatch, schedule=[("decode", 1), None, None], speculate={},
                                             refuse=set())
    taken, built = [], []
    w.scheduler.take_rebuilds = taken.extend

    def yield_to_rebuild(pending, consecutive, fair):
        """A chain that yields to a queued rebuild row, once."""
        if built or pending is None:
            return None, None
        batch = ScheduledBatch(NODE, "rebuild", rebuild_rows={R: RebuildRow(NODE, "rebuild", {})})
        node_batch = w._build_executing_batch(batch)
        node_batch.rebuild_rids = frozenset(batch.rebuild_rows)
        built.append(batch)
        return Speculation(scheduled_batch=batch, node_batch=node_batch, consumed_edges=set(), continuing_rids=set(),
                           partition="p", is_new_iter=False, is_same_node=False, is_yield_away=True), None

    w._build_speculation = yield_to_rebuild
    with pytest.raises(_StopLoopError):
        w.run()
    assert postprocessed == ["decode", "rebuild"] and taken == [R], "the yield-away step was built and thrown away"


@pytest.mark.parametrize("where", ["being built", "in flight"])
def test_a_step_that_raises_fails_its_rebuild_rows_with_its_other_rows(where):
    w, _ = _worker()
    w._clear_in_flight_flag = lambda batch: None
    batch = ScheduledBatch(NODE, PREFILL, {A: 0}, rebuild_rows={R: RebuildRow(NODE, PREFILL, {})})
    stale = PendingBatch(batch, _step(), NODE, "p", PREFILL, None) if where == "in flight" else None

    w._handle_main_loop_error(RuntimeError("boom"), (stale,), None if stale else batch)

    assert w.failed.keys() == {R, A}, "the dropped request would never be rebuilt or failed"


@pytest.mark.parametrize("reason, outcome", [
    (AllocationFailed("short", 1, "main", R), "held"),
    (AdmitRuntimeError("unservable", request_ids=frozenset({R})), "failed"),
    (AllocationFailed("short", 1, "main", A), "requeued"),
])
def test_a_refused_admit_puts_the_rebuild_row_back_or_fails_only_its_request(reason, outcome):
    w, _, batch, node_batch = _rebuilding(fresh={A: 0})
    node_batch.admit_error = reason
    node_batch.rebuild_rids |= {11}  # a second rebuild row, which left the step when its prepare_inputs raised
    w._handle_allocation_failure = lambda *a: w.held.append("base")
    w.scheduler.take_rebuilds(node_batch.rebuild_rids)  # as the step's launch does, before its admit

    w._handle_admit_failure(batch, node_batch)

    assert w.scheduler.rebuild_queued(R) == (outcome != "failed"), "the rebuild row left its queue for good"
    assert not w.scheduler.rebuild_queued(11), "a row that left the step would run again"
    assert w.failed.keys() == ({R} if outcome == "failed" else set()), "the refusal failed the wrong requests"
    assert w.held == {"held": [R], "failed": [], "requeued": ["base"]}[outcome], "the other rows lost the base path"
    assert w.pushed == ([] if outcome == "requeued" else [A]), "the step's prompt row would never be scheduled again"


def _step(rids=(R, A), walk=PREFILL, node=NODE, rebuild=R, **fields) -> ExecutingBatch:
    """A step of ``rids``, where the row ``rebuild`` only recomputes KV."""
    return ExecutingBatch(node_name=node, per_request_info={rid: SimpleNamespace(rid=rid) for rid in rids},
                          rebuild_rids=frozenset({rebuild}), **fields,
                          step_context=StepContext(request_ids=rids, graph_walk=walk, slot=0, capture=False))


@pytest.mark.parametrize("batched", [True, False])
def test_a_rebuild_rows_discarded_sample_never_stops_its_loop(batched):
    ends = {"decode_loop"}
    sub = SimpleNamespace(check_stop=lambda rid, info, outputs: ends,  # as Orpheus's, which doesn't test the walk
                          check_stop_batched=lambda rids, *_, step_walk: dict.fromkeys(rids, ends) if batched else None)
    engine = SimpleNamespace(_submodules={NODE: SimpleNamespace(submodule=sub)})

    stops = Engine.check_stop_for_batch(engine, _step(), dict.fromkeys((R, A), {"new_token": [0]}), host_rows=object())

    assert stops == {A: ends}, "the rebuild's sample would end the decode loop it rebuilds"


@pytest.mark.parametrize("branch", ["rows_only", "mixed", "tensor"])
def test_a_rebuild_row_leaves_its_requests_loopback_entry(branch):
    engine = SimpleNamespace(_loopback_steps={R: (NODE, DECODE), A: (NODE, DECODE)})
    batch = _step(rows_only=branch == "rows_only", loopback_rids=frozenset({A} if branch == "mixed" else ()))
    if branch == "mixed":
        batch.step_context.request_walks = {R: PREFILL, A: DECODE}

    Engine._record_loopback(engine, batch)

    assert engine._loopback_steps[R] == (NODE, DECODE), "the resumed decode would not read the last kept id"

