"""A dropped request gives its pages back without a host pool.

Preemption has to free pages when ``cpu_offload_pages`` is 0, the default, and a dropped request comes back by
recomputing its KV from its token ids. So a drop copies nothing: pages the index also holds stay cached for that
recompute to hit, and the rest go back to the free list. A drop is refused, before anything changes, when
recomputing could not bring the request back or something still uses its pages.
"""

from __future__ import annotations

import sys
from array import array
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.engine import Engine
from mstar.engine.resources.base import EngineResourceInfo, Rebuild, Residency, Resource
from mstar.engine.resources.kv.bounded.manager import BoundedKVManager
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.ring.manager import RingKVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.lora.pool import LoraAdapterPool
from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import PublishedPositionInfo, RopeManager
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig, RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import FULL_ADMIT_NOT_READY, FULL_ADMIT_REBUILD, Segment, StepContext
from mstar.model.base import LabelLayout, Span
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.testing_utils.model import StubModel, make_model

PAGE_SIZE = 16
WALK = "prefill"
NODE = "LLM"
CPU = torch.device("cpu")


def _manager() -> KVManager:
    cfg = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, max_num_pages=16,
                        page_size=PAGE_SIZE)
    kv = KVManager(cfg=cfg, name="kv", joint_comm_group=None, device=CPU, dtype=torch.float32,
                   transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")))
    kv.enable_prefix_cache(b"a root")
    return kv


def _step(rid: str, span: int) -> tuple[KVStep, StepContext]:
    step = KVStep(segments=(Segment(rid, "main", span),))
    return step, StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False)


def _config(tokens: list[int], cap: int = 50, kind: str = "ids") -> KVReqConfig:
    """A prompt keyed as the data worker does it, on a label whose samples are keyed too."""
    pages = [tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)]
    return KVReqConfig(
        prefix_keys={"main": chain(pages)}, prefix_tail={"main": pages[-1] if len(tokens) % PAGE_SIZE else []},
        prefix_decode={"main": "text_inputs"}, loop_caps={"decode_loop": cap},
        kv_layout={"main": StreamLayout(
            spans=(PrefixSpan(len(tokens), len(tokens), WALK, kind=kind),), growth=(("decode_loop", 1),), keyed_spans=1,
        )},
    )


def _prefill(kv: KVManager, rid: str, length: int, kind: str = "ids") -> None:
    """Ingest a keyed prompt of ``length`` ids and write it."""
    kv.ingest_request(rid, _config(list(range(length)), kind=kind))
    _write(kv, rid, length)


def _write(kv: KVManager, rid: str, *spans: int) -> None:
    """Write ``rid``'s next tokens, ``spans[i]`` of them in step ``i``."""
    for span in spans:
        step, ctx = _step(rid, span)
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
        kv.commit(step, ctx)


def _engine(resources: dict) -> Engine:
    return make_engine(_runner=StepRunner(resources), _submodules={NODE: make_mgmt(resources=resources)})


def test_each_span_reaches_the_kv_manager_with_its_kind():
    layout = LabelLayout(spans=(Span("ids", WALK, 4, 4, "text_inputs"), Span("digest", WALK, 6, 3, ("image", 0)),
                                Span("rows", WALK, 2, 2, "thinker_states")))
    config = KVReqConfig()
    config.apply_conductor_config(kv_layout={"main": layout.row()}, loop_caps={})

    assert [span.kind for span in config.kv_layout["main"].spans] == ["ids", "digest", "rows"], "media passes as ids"


def test_a_drop_frees_only_the_pages_the_request_held_alone():
    kv = _manager()
    _prefill(kv, "r", 2 * PAGE_SIZE + 5)
    free = kv._arena.num_free

    assert kv.drop("r", 2 * PAGE_SIZE + 5) == 1 == kv._arena.num_free - free, (
        "only the partial page is the request's alone"
    )
    assert len(kv._index.evictable()) == 2, "the two keyed pages should stay cached, evictable"
    kv.assert_pages_conserved()


@pytest.mark.parametrize(
    "refusal", ["step", "read", "ranks", "host", "guaranteed", "not rebuildable", "fewer ids", "more ids"],
)
def test_a_refused_drop_changes_nothing(refusal):
    kv = _manager()
    _prefill(kv, "r", PAGE_SIZE)
    kv._guaranteed.update({"r"} if refusal == "guaranteed" else ())
    kv._overrides["r"].rebuildable = refusal != "not rebuildable"
    assert refusal != "step" or kv.admit(*_step("r", 1)).ok
    kv._streams["r"]["main"].read_pending = refusal == "read"
    kv._world_size, kv._counts = (2, False) if refusal == "ranks" else (kv._world_size, kv._counts)
    kv._cpu_pool = SimpleNamespace(is_offloaded=lambda rid: refusal == "host")
    pages = list(kv._streams["r"]["main"].page_indices)

    with pytest.raises(RuntimeError):
        kv.drop("r", PAGE_SIZE + {"fewer ids": -1, "more ids": 1}.get(refusal, 0))
    assert kv._streams["r"]["main"].page_indices == pages and "r" not in kv._dropped, "the refused drop changed it"


def test_a_second_drop_before_the_rebuild_lands_is_refused():
    kv = _manager()
    _prefill(kv, "r", PAGE_SIZE, kind="ids")
    kv.drop("r", PAGE_SIZE)
    held = kv._dropped["r"]

    with pytest.raises(RuntimeError):
        kv.drop("r", PAGE_SIZE)
    assert kv._dropped["r"] == held, "a second drop would overwrite what the rebuild has to write again"


class _Recording:
    """A resource that records the rebuild hooks it is called with."""

    supports_preplan = False
    force_double_buffer = False
    lean_capture = False

    def __init__(self):
        self.calls = []

    def prepare_rebuild(self, rid, rebuild):
        self.calls.append(("prepare", rid, rebuild))

    def land_rebuild(self, rid):
        self.calls.append(("land", rid))


def test_the_engine_tells_every_resource_of_the_node_about_a_rebuild():
    first, second = _Recording(), _Recording()
    rebuild = Rebuild(WALK, array("i", [1, 2, 3]))
    engine = make_engine(_submodules={NODE: make_mgmt(resources={"first": first, "second": second})})

    engine.prepare_rebuild(NODE, "r", rebuild)
    engine.land_rebuild(NODE, "r")

    assert first.calls == second.calls == [("prepare", "r", rebuild), ("land", "r")], (
        "a resource the engine skipped would hold the layout from before the drop"
    )


def test_a_drop_keeps_the_sampler_and_starts_positions_and_recurrent_state_over():
    kv, rope = _manager(), RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)
    pool = RecurrentStatePool.build(RecurrentStateSpec("gdn_state", {NODE}, RecurrentStateConfig(
        num_layers=1, blocks={"state": RecurrentBlockConfig(shape=(2, 4), dtype=torch.float32)}, max_slots=4,
    )), EngineResourceInfo(device=CPU))
    sampler = SamplerResource(vocab_size=64, enable_repetion_penalty=True, device=CPU)
    engine = _engine({"kv": kv, "rope": rope, "gdn_state": pool, "sampler": sampler})
    engine._runner.ingest_request("r", {"sampler": SamplingReqConfig(temperature=1.0, repetition_penalty=1.3)})
    _prefill(kv, "r", PAGE_SIZE)
    rope._counters["r"]["main"] = PAGE_SIZE
    slot = pool._alloc("r", "main")
    pool._blocks["state"][:, slot.index] = 1.0
    sampler._sampler._step_offset["r"] = 7
    sampler._sampler.get_token_mask("r").add_tokens(3)

    engine.drop_request(NODE, "r", PAGE_SIZE)

    assert sampler._sampler._step_offset["r"] == 7, "the RNG offset can't be recomputed"
    assert sampler._sampler.get_token_mask("r")._seen_token_mask[3], "the repetition mask can't be recomputed"
    assert rope._counters["r"] == {}, "a cache hit only moves the counter forward"
    assert pool._slots["r"]["main"] is slot, "the recompute would wait for a slot"
    assert not pool._blocks["state"][:, slot.index].any(), "the recompute would start from the old state"


def test_a_refusal_by_any_resource_leaves_every_resource_alone():
    # as Whisper's decoder lays out its self-attention KV from ids and its cross-attention KV from rows
    kv, cross, rope = _manager(), _manager(), RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)
    engine = _engine({"rope": rope, "kv": kv, "cross": cross})
    rope.ingest_request("r")
    _prefill(kv, "r", PAGE_SIZE)
    _prefill(cross, "r", PAGE_SIZE, kind="rows")
    cross._guaranteed.add("r")
    rope._counters["r"]["main"] = PAGE_SIZE

    with pytest.raises(RuntimeError):
        engine.drop_request(NODE, "r", PAGE_SIZE)
    assert kv.residency("r") is Residency.RESIDENT and kv._streams["r"]["main"].page_indices, (
        "one KV cache dropped and the other kept"
    )
    assert rope._counters["r"] == {"main": PAGE_SIZE}, "the positions were reset for a request that keeps its KV"


def test_a_rebuilt_stream_keeps_its_footprint_and_is_keyed_as_a_fresh_prompt():
    kv, fresh = _manager(), _manager()
    prompt, sampled = list(range(20)), list(range(100, 130))
    kv.ingest_request("r", _config(prompt, cap=50))
    footprint = kv._streams["r"]["main"].footprint
    _write(kv, "r", len(prompt), *[1] * (len(sampled) - 1))
    kv.drop("r", len(prompt) + len(sampled) - 1)

    kv.prepare_rebuild("r", Rebuild(WALK, array("i", prompt + sampled)))
    fresh.ingest_request("f", _config(prompt + sampled[:-1], cap=50 - 29))
    fresh._streams["f"]["main"].chain.extend(sampled[-1:], PAGE_SIZE)
    rebuilt, want = kv._streams["r"]["main"], fresh._streams["f"]["main"]
    assert rebuilt.footprint == footprint, "the rebuild would hold more pages than the request declared"
    assert (rebuilt.chain.keys, rebuilt.chain.unkeyed) == (want.chain.keys, want.chain.unkeyed), (
        "the rebuilt pages would be filed under keys another request with these ids can't match"
    )


def test_a_rebuilt_stream_whose_growth_is_bounded_keeps_its_footprint():
    """The rebuild's span covers what the loop wrote, so the growth bound drops by that as the loop's cap does.
    Otherwise the stream would grow by those slots."""
    kv = _manager()
    prompt, sampled = list(range(20)), list(range(100, 130))
    config = _config(prompt, cap=500)
    config.kv_layout["main"] = replace(config.kv_layout["main"], max_growth=50)
    kv.ingest_request("r", config)
    footprint = kv._streams["r"]["main"].footprint
    _write(kv, "r", len(prompt), *[1] * (len(sampled) - 1))
    kv.drop("r", len(prompt) + len(sampled) - 1)

    kv.prepare_rebuild("r", Rebuild(WALK, array("i", prompt + sampled)))

    assert kv._streams["r"]["main"].footprint == footprint, (
        "the rebuild would hold more pages than the request declared"
    )


def test_a_publish_that_arrives_while_dropped_restores_neither_positions_nor_pages():
    kv, rope = _manager(), RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)
    engine = _engine({"kv": kv, "rope": rope})
    rope.ingest_request("r")
    _prefill(kv, "r", 2 * PAGE_SIZE)
    # what the conductor sends back with every INPUT_SIGNALS
    published = {"kv": kv.publish("r", NODE, WALK), "rope": PublishedPositionInfo(counters={"main": 2 * PAGE_SIZE})}
    engine.drop_request(NODE, "r", 2 * PAGE_SIZE)

    outcome = engine.check_ready(NODE, "r", SimpleNamespace(graph_walk=WALK, resource_publish_info=published))

    assert not outcome.ready, "its step would read pages it gave back"
    assert rope._counters["r"] == {} and not kv._streams["r"]["main"].page_indices, "the rebuild would start past 0"


def test_removing_a_dropped_or_relaid_request_leaks_no_page():
    kv = _manager()
    for rid in ("dropped", "relaid"):
        _prefill(kv, rid, 20)
        kv.drop(rid, 20)
    kv.prepare_rebuild("relaid", Rebuild(WALK, array("i", range(21))))
    kv.land_rebuild("relaid")
    assert kv.residency("relaid") is Residency.RESIDENT, "a rebuilt request would never be ready again"

    for rid in ("dropped", "relaid"):
        kv.remove_request(rid)
    kv.assert_pages_conserved()
    assert kv.residency("dropped") is Residency.RESIDENT, "the next request given this handle would never be ready"
    assert (counts := kv.page_counts()).used == counts.cached, "a page outlived its request outside the index"


def test_a_dropped_request_reads_as_dropped_until_its_rebuild_lands():
    kv = _manager()
    engine = _engine({"kv": kv, "rope": RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)})
    _prefill(kv, "r", 20)
    kv.drop("r", 20)
    kv.prepare_rebuild("r", Rebuild(WALK, array("i", range(21))))

    assert engine.residency(NODE, "r") is Residency.DROPPED, "the request would read as resident mid-rebuild"
    kv.land_rebuild("r")
    assert engine.residency(NODE, "r") is Residency.RESIDENT, "the rebuilt request would never be ready again"


@pytest.mark.parametrize("allow_reload, outcome", [(True, FULL_ADMIT_REBUILD), (False, FULL_ADMIT_NOT_READY)],
                         ids=["scheduler", "follower"])
def test_readiness_answers_a_dropped_request_with_a_rebuild_and_a_follower_with_not_ready(allow_reload, outcome):
    kv = _manager()
    engine = _engine({"kv": kv, "rope": RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)})
    _prefill(kv, "r", 20)
    kv.drop("r", 20)

    info = SimpleNamespace(graph_walk=WALK, resource_publish_info={})
    assert engine.check_ready(NODE, "r", info, allow_reload) == outcome, (
        "the decode row would wait on a rebuild nothing starts, or a follower would start one rank 0 never ran"
    )


def _boot_engine(resources: dict, split: frozenset = frozenset()) -> Engine:
    submodule = SimpleNamespace(split_batches_by_capture_key=lambda walk: walk in split)
    return make_engine(_submodules={NODE: make_mgmt(resources=resources, submodule=submodule)})


def _model(combined: dict | None = None) -> StubModel:
    """Declares a rebuild policy on ``NODE`` and, given ``combined``, batches its walks under a combined walk."""
    policy = SimpleNamespace(walk=WALK, ids_input="text_inputs", sample_output="new_token", loop="decode_loop")
    return make_model(rebuild_policies=lambda: {NODE: policy}, combined_walk_of=lambda: combined or {})


def test_a_resource_supports_no_drop_unless_it_opts_in():
    supported = (KVManager, RecurrentStatePool, SamplerResource, RopeManager, LoraAdapterPool)

    assert not Resource.supports_drop and not RingKVManager.supports_drop and not BoundedKVManager.supports_drop
    assert all(cls.supports_drop for cls in supported), "a resource that discards what a rebuild writes refused"


def test_a_rebuild_policy_on_a_node_with_a_resource_that_cant_drop_is_refused_at_load():
    _boot_engine({"kv": _manager()})._check_rebuild_policies(_model())

    with pytest.raises(ValueError, match=r"\['bounded'\] can't drop"):
        bounded = SimpleNamespace(supports_drop=False, supports_preplan=False, force_double_buffer=False,
                                  lean_capture=False)
        engine = _boot_engine({"kv": _manager(), "bounded": bounded})
        engine._check_rebuild_policies(_model())


@pytest.mark.parametrize("split, combined", [({WALK}, None), ({"mixed"}, {(NODE, WALK): "mixed"})])
def test_a_rebuild_policy_whose_steps_split_by_capture_key_is_refused_at_load(split, combined):
    with pytest.raises(ValueError, match="capture key"):
        _boot_engine({"kv": _manager()}, frozenset(split))._check_rebuild_policies(_model(combined))
