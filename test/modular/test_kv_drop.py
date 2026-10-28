"""A dropped request gives its pages back without a host pool.

Preemption has to free pages when ``cpu_offload_pages`` is 0, the default, and a dropped request comes back by
recomputing its KV from its token ids. So a drop copies nothing: pages the index also holds stay cached for that
recompute to hit, and the rest go back to the free list. A drop is refused, before anything changes, when
recomputing could not bring the request back or something still uses its pages.
"""

from __future__ import annotations

import sys
from array import array
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import PublishedPositionInfo, RopeManager
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig, RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.base import LabelLayout, PrefixKeys, Span
from mstar.model.submodule_base import HostRows

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


def _step(rid: str, span: int, walk: str = WALK) -> tuple[KVStep, StepContext]:
    step = KVStep(segments=(Segment(rid, "main", span),))
    return step, StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)


def _config(tokens: list[int], cap: int = 50, kind: str = "ids") -> KVReqConfig:
    """A prompt keyed as the data worker does it, on a label whose samples are keyed too."""
    pages = [tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)]
    return KVReqConfig(
        prefix_keys={"main": chain(pages)}, prefix_tail={"main": pages[-1] if len(tokens) % PAGE_SIZE else []},
        prefix_decode={"main": "text_inputs"}, loop_caps={"decode_loop": cap},
        kv_layout={"main": StreamLayout(
            spans=(PrefixSpan(len(tokens), len(tokens), WALK, kind=kind),), growth=(("decode_loop", 1),), keyed_spans=1,
        )},
        prompt_ids={"main": array("i", tokens).tobytes()},
    )


def _write(kv: KVManager, rid: str, span: int, walk: str = WALK) -> None:
    step, ctx = _step(rid, span, walk)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


def _prefill(kv: KVManager, rid: str, length: int, kind: str = "ids") -> None:
    """Ingest a keyed prompt of ``length`` ids and write it."""
    kv.ingest_request(rid, _config(list(range(length)), kind=kind))
    _write(kv, rid, length)


def _engine(resources: dict) -> Engine:
    engine = Engine.__new__(Engine)
    engine._runner, engine._submodules = StepRunner(resources), {NODE: SimpleNamespace(resources=resources)}
    engine._tp_follower_nodes = set()
    return engine


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

    assert kv.drop("r") == 1 == kv._arena.num_free - free, "only the partial page is the request's alone"
    assert len(kv._index.evictable()) == 2, "the two keyed pages should stay cached, evictable"
    kv.assert_pages_conserved()


@pytest.mark.parametrize("refusal", ["step", "read", "ranks", "host", "guaranteed"])
def test_a_refused_drop_changes_nothing(refusal):
    kv = _manager()
    _prefill(kv, "r", PAGE_SIZE)
    kv._overrides["r"].guaranteed = refusal == "guaranteed"
    assert refusal != "step" or kv.admit(*_step("r", 1)).ok
    kv._streams["r"]["main"].read_pending = refusal == "read"
    kv._world_size = 2 if refusal == "ranks" else kv._world_size
    kv._cpu_pool = SimpleNamespace(is_offloaded=lambda rid: refusal == "host")
    pages = list(kv._streams["r"]["main"].page_indices)

    with pytest.raises(RuntimeError):
        kv.drop("r")
    assert kv._streams["r"]["main"].page_indices == pages and "r" not in kv._dropped, "the refused drop changed it"


def test_a_drop_keeps_the_sampler_and_starts_positions_and_recurrent_state_over():
    kv, rope = _manager(), RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)
    pool = RecurrentStatePool.build(RecurrentStateSpec("gdn_state", {NODE}, RecurrentStateConfig(
        num_layers=1, blocks={"state": RecurrentBlockConfig(shape=(2, 4), dtype=torch.float32)}, max_slots=4,
    )), EngineResourceInfo(device=CPU))
    sampler = SamplerResource(vocab_size=64, enable_repetion_penalty=True, device=CPU)
    engine = _engine({"kv": kv, "rope": rope, "gdn_state": pool, "sampler": sampler})
    engine._runner.ingest_request("r", {
        "sampler": SamplingReqConfig(temperature=1.0, repetition_penalty=1.3), "kv": _config(list(range(PAGE_SIZE))),
    })
    _write(kv, "r", PAGE_SIZE)
    rope._counters["r"]["main"] = PAGE_SIZE
    slot = pool._alloc("r", "main")
    pool._blocks["state"][:, slot.index] = 1.0
    sampler._sampler._step_offset["r"] = 7
    sampler._sampler.get_token_mask("r").add_tokens(3)

    engine.drop_request(NODE, "r")

    assert sampler._sampler._step_offset["r"] == 7, "the RNG offset can't be recomputed"
    assert sampler._sampler.get_token_mask("r")._seen_token_mask[3], "the repetition mask can't be recomputed"
    assert rope._counters["r"] == {}, "a cache hit only moves the counter forward"
    assert pool._slots["r"]["main"] is slot, "the recompute would wait for a slot"
    assert not pool._blocks["state"][:, slot.index].any(), "the recompute would start from the old state"


def test_a_sampler_that_scatters_in_its_graph_refuses_a_drop():
    sampler = SamplerResource(vocab_size=64, enable_repetion_penalty=False, device=CPU)
    sampler._cg_buffers = SimpleNamespace(ingraph_scatter=True)
    with pytest.raises(RuntimeError):
        _engine({"sampler": sampler}).drop_request(NODE, "r")


def test_a_refusal_by_any_resource_leaves_every_resource_alone():
    # as Whisper's decoder lays out its self-attention KV from ids and its cross-attention KV from rows
    kv, cross, rope = _manager(), _manager(), RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)
    engine = _engine({"rope": rope, "kv": kv, "cross": cross})
    rope.ingest_request("r")
    _prefill(kv, "r", PAGE_SIZE)
    _prefill(cross, "r", PAGE_SIZE, kind="rows")
    cross._overrides["r"].guaranteed = True
    rope._counters["r"]["main"] = PAGE_SIZE

    with pytest.raises(RuntimeError):
        engine.drop_request(NODE, "r")
    assert kv.is_resident("r") and kv._streams["r"]["main"].page_indices, "one KV cache dropped and the other kept"
    assert rope._counters["r"] == {"main": PAGE_SIZE}, "the positions were reset for a request that keeps its KV"


def test_a_rebuilt_stream_keeps_its_footprint_and_is_keyed_as_a_fresh_prompt():
    kv, fresh = _manager(), _manager()
    prompt, sampled = list(range(20)), list(range(100, 130))
    kv.ingest_request("r", _config(prompt, cap=50))
    footprint = kv._streams["r"]["main"].footprint
    kv.drop("r")

    assert kv.lay_out_again("r", array("i", prompt + sampled)) == (PrefixSpan(49, 49, WALK),)
    fresh.ingest_request("f", _config(prompt + sampled[:-1], cap=50 - 29))
    fresh._streams["f"]["main"].chain.extend(sampled[-1:], PAGE_SIZE)
    rebuilt, want = kv._streams["r"]["main"], fresh._streams["f"]["main"]
    assert rebuilt.footprint == footprint, "the rebuild would hold more pages than the request declared"
    assert (rebuilt.chain.keys, rebuilt.chain.unkeyed) == (want.chain.keys, want.chain.unkeyed), (
        "the rebuilt pages would be filed under keys another request with these ids can't match"
    )


def _keyed_chat(tokens: list[int], cap: int) -> KVReqConfig:
    """A chat of three ids, an image whose walk is its own and the rest of ``tokens``, keyed as the data worker does
    it. The image ends on a page boundary, so it has no end key."""
    spans = (Span("ids", WALK, 3, 3, "text_inputs"), Span("digest", "vit", PAGE_SIZE - 3, 1, ("image", 0)),
             Span("ids", WALK, len(tokens) - 3, len(tokens) - 3, "text_inputs"))
    digests, texts = [None, b"image", None], {"text_inputs": [torch.tensor(tokens[:3]), torch.tensor(tokens[3:])]}
    pages, items = PreprocessWorkerThread.__new__(PreprocessWorkerThread)._page_layout(
        PrefixKeys("text_inputs", (WALK, "vit")), spans, digests, texts, PAGE_SIZE,
    )
    return KVReqConfig(
        prefix_keys={"main": chain(pages, items)},
        prefix_tail={"main": pages[-1] if sum(span.length for span in spans) % PAGE_SIZE else []},
        prefix_decode={"main": "text_inputs"}, loop_caps={"decode_loop": cap},
        kv_layout={"main": StreamLayout(spans=tuple(
            PrefixSpan(span.length, span.advance, span.walk, kind=span.kind, digest=digest)
            for span, digest in zip(spans, digests, strict=True)
        ), growth=(("decode_loop", 1),), keyed_spans=3)},
        prompt_ids={"main": array("i", tokens).tobytes()},
    )


def test_a_rebuilt_label_with_an_image_is_keyed_as_a_fresh_prompt_of_its_ids_and_image():
    kv = _manager()
    prompt, sampled = list(range(10)), list(range(100, 160))
    kv.ingest_request("r", _keyed_chat(prompt, cap=80))

    # the second rebuild starts from the keys and tail the first left
    for upto in (30, 60):
        kv.drop("r")
        kv.lay_out_again("r", array("i", prompt + sampled[:upto]))
        fresh = _manager()
        fresh.ingest_request("f", _keyed_chat(prompt + sampled[:upto - 1], cap=80 - (upto - 1)))
        fresh._streams["f"]["main"].chain.extend(sampled[upto - 1:upto], PAGE_SIZE)

        rebuilt, want = kv._streams["r"]["main"].chain, fresh._streams["f"]["main"].chain
        assert (rebuilt.keys, rebuilt.unkeyed, rebuilt.spans) == (want.keys, want.unkeyed, want.spans), (
            f"after {upto} samples the rebuilt pages would be filed under keys no request with this image and these "
            "ids can match"
        )


PROMPT = (PrefixSpan(3, 3, WALK), PrefixSpan(6, 4, WALK, kind="digest"), PrefixSpan(2, 2, WALK))


def _image_prompt(kv: KVManager, spans: tuple[PrefixSpan, ...] = PROMPT) -> None:
    """Ingest ``r`` with a prompt of text, an image and text on one walk, as Qwen3.5 lays it out, then drop it."""
    kv.ingest_request("r", KVReqConfig(
        kv_layout={"main": StreamLayout(spans=spans, growth=(("decode_loop", 1),))}, loop_caps={"decode_loop": 50},
        prompt_ids={"main": array("i", range(5)).tobytes()},
    ))
    kv.drop("r")


def test_a_rebuild_keeps_an_image_prompts_spans_and_lays_its_samples_after_them():
    kv = _manager()
    _image_prompt(kv)
    footprint = kv._streams["r"]["main"].footprint

    spans = kv.lay_out_again("r", array("i", [*range(5), 100, 101, 102]))

    assert spans == (*PROMPT[:2], PrefixSpan(4, 4, WALK)), "the image would lose its advance, or the samples a slot"
    assert kv._overrides["r"].loop_caps["decode_loop"] == 48 and kv._streams["r"]["main"].footprint == footprint, (
        "the rebuild would hold more pages than the request declared"
    )


@pytest.mark.parametrize("first", [2, 0])
def test_a_second_rebuild_of_an_image_prompt_counts_its_first_samples_once(first):
    kv = _manager()
    _image_prompt(kv)
    # ``first`` samples in the KV, then the next decode step's input
    kv.lay_out_again("r", array("i", [*range(5), *range(100, 101 + first)]))
    _write(kv, "r", 11 + first)  # the rebuild row
    kv.land_rebuild("r")
    _write(kv, "r", 1)
    _write(kv, "r", 1)
    kv.drop("r")

    spans = kv.lay_out_again("r", array("i", [*range(5), *range(100, 103 + first)]))

    last = PROMPT[-1].length + first + 2
    assert spans == (*PROMPT[:2], PrefixSpan(last, last, WALK)), "the first rebuild's samples would count twice"
    assert kv._overrides["r"].loop_caps["decode_loop"] == 48 - first, "the samples would count twice in the footprint"


def test_a_request_dropped_between_rebuild_rows_frees_their_pages_and_is_laid_out_the_same():
    kv = _manager()
    # the same prompt with its image on a walk of its own, as Bagel lays out an image chat
    _image_prompt(kv, (PrefixSpan(3, 3, WALK), PrefixSpan(20, 1, "vit", kind="digest"), PrefixSpan(2, 2, WALK)))
    free, ids = kv._arena.num_free, array("i", [*range(5), 100, 101])
    spans = kv.lay_out_again("r", ids)
    _write(kv, "r", 3)
    _write(kv, "r", 20, "vit")

    kv.drop("r")

    assert kv._arena.num_free == free, "the rows that ran would keep their pages"
    assert kv.lay_out_again("r", ids) == spans and kv._streams["r"]["main"].span_cursor.done == 0, (
        "the rebuild would resume past rows whose pages the drop gave up"
    )


def test_a_rebuild_that_writes_fewer_slots_than_its_layout_raises_under_debug_asserts(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager()
    _image_prompt(kv)
    kv.lay_out_again("r", array("i", [*range(5), 100, 101, 102]))
    _write(kv, "r", 12)  # one slot fewer than the layout's 13, as when the image copy is one row short

    with pytest.raises(AssertionError, match="wrong size"):
        kv.land_rebuild("r")


def test_a_rebuild_row_beside_another_keyed_row_keys_none_of_its_discarded_sample():
    kv = _manager()
    kv.ingest_request("r", _config(list(range(20)), cap=50))
    kv.ingest_request("a", _config(list(range(50, 70)), cap=50))
    kv.drop("r")
    kv.lay_out_again("r", array("i", [*range(20), 100, 101]))
    batch = ExecutingBatch(node_name=NODE, per_request_info={}, rebuild_rids=frozenset({"r"}),
                           step_context=StepContext(request_ids=("r", "a"), graph_walk=WALK, slot=0, capture=False))

    rows = HostRows(("r", "a"), {"text_inputs": torch.tensor([[7], [8]])})
    Engine.extend_prefix_chains(SimpleNamespace(_runner=StepRunner({"kv": kv})), batch, {}, host_rows=rows)
    kv.flush_chain_backlog()

    assert kv._streams["r"]["main"].chain.unkeyed == [16, 17, 18, 19, 100, 101], "it would key a discarded sample"
    assert kv._streams["a"]["main"].chain.covered_len == 21, "the other row's sample went unkeyed"


def test_a_publish_that_arrives_while_dropped_restores_neither_positions_nor_pages():
    kv, rope = _manager(), RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU)
    engine = _engine({"kv": kv, "rope": rope})
    rope.ingest_request("r")
    _prefill(kv, "r", 2 * PAGE_SIZE)
    # what the conductor sends back with every INPUT_SIGNALS
    published = {"kv": kv.publish("r", NODE, WALK), "rope": PublishedPositionInfo(counters={"main": 2 * PAGE_SIZE})}
    engine.drop_request(NODE, "r")

    outcome = engine.check_ready(NODE, "r", SimpleNamespace(graph_walk=WALK, resource_publish_info=published))

    assert not outcome.ready, "its step would read pages it gave back"
    assert rope._counters["r"] == {} and not kv._streams["r"]["main"].page_indices, "the rebuild would start past 0"


def test_removing_a_dropped_or_relaid_request_leaks_no_page():
    kv = _manager()
    for rid in ("dropped", "relaid"):
        _prefill(kv, rid, 20)
        kv.drop(rid)
    kv.lay_out_again("relaid", array("i", range(21)))
    _write(kv, "relaid", 20)
    kv.land_rebuild("relaid")
    assert kv.is_resident("relaid"), "a rebuilt request would never be ready again"

    for rid in ("dropped", "relaid"):
        kv.remove_request(rid)
    kv.assert_pages_conserved()
    assert kv.is_resident("dropped"), "the next request given this handle would never be ready"
    assert (counts := kv.page_counts()).used == counts.cached, "a page outlived its request outside the index"

