"""A request is admitted only once what it can still take fits, in arrival order.

The scheduler asks each resource whether a request is ready before it runs a
step, and "not ready" is the answer it already retries. So the pool answers it
from its reservations: a request waits until its own fits next to everything
already admitted, and nothing behind the first request waiting passes it, or a
stream of short requests would starve a long one forever. A request that could
never fit even alone is failed rather than queued, since it would block the
queue for good. Once admitted, no step of the request can run out of pages.
"""

from __future__ import annotations

import random
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.cpu_page_pool import OffloadedStream
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import (
    AdmissionDeferred,
    KVManager,
    KVSequenceInfo,
    PublishedKVInfo,
)
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.step import AdmitRuntimeError, AllocationFailed, Segment, StepContext

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the host pool pins its memory"
)

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
PREFILL = "prefill"
DECODE = "decode"
SEED = 20260930


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(
    max_num_pages: int = 16, rank: int = 0, world_size: int = 1,
    cpu_offload_pages: int = 0,
) -> KVManager:
    group = None
    if world_size > 1:
        group = SimpleNamespace(rank=rank, world_size=world_size)
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
            cpu_offload_pages=cpu_offload_pages,
        ),
        name="kv", joint_comm_group=group, transfer_engine_info=None,
        device=torch.device("cuda" if cpu_offload_pages else "cpu"),
        dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT, {"main": (PREFILL, DECODE)})
    return kv


def _request(tokens: list[int], max_tokens: int) -> KVReqConfig:
    """A keyed text request, stamped as Bagel stamps one."""
    whole = len(tokens) // PAGE_SIZE
    return KVReqConfig(
        max_tokens=max_tokens,
        prefix_keys={"main": chain([
            tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)
        ])},
        prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
        prompt_slots={"main": len(tokens)}, decode_labels=["main"],
    )


def _ready(kv: KVManager, rid: str):
    return kv.admit_retrieve(rid, NODE, PREFILL, None)


def _step(kv: KVManager, rid: str, span: int, walk: str = PREFILL):
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)
    outcome = kv.admit(step, ctx)
    if outcome.ok:
        kv.plan(step, ctx)
        kv.commit(step, ctx)
    return outcome


def _prefill(kv: KVManager, rid: str, prompt: int):
    """What prepare does before the step: probe the cache and cut the inputs."""
    matched = kv.resolve_cached_prefix(rid, NODE, PREFILL) or 0
    kv.apply_cached_prefix(rid, NODE, PREFILL, None, matched)
    return _step(kv, rid, prompt - matched)


# ── waiting ─────────────────────────────────────────────────────────────


def test_a_request_that_does_not_fit_waits_until_room_frees():
    kv = _manager(max_num_pages=16)
    kv.ingest_request("a", _request(list(range(100)), max_tokens=60))
    kv.ingest_request("b", _request(list(range(500, 600)), max_tokens=60))
    assert _ready(kv, "a").ready

    assert not _ready(kv, "b").ready

    kv.remove_request("a")
    assert _ready(kv, "b").ready, "room a gave back never reached the request waiting for it"


def test_nothing_passes_the_head_of_the_queue():
    kv = _manager(max_num_pages=16)
    kv.ingest_request("a", _request(list(range(100)), max_tokens=60))
    kv.ingest_request("long", _request(list(range(500, 600)), max_tokens=60))
    kv.ingest_request("short", _request(list(range(900, 910)), max_tokens=4))
    assert _ready(kv, "a").ready
    assert not _ready(kv, "long").ready

    # "short" would fit on its own, but it asked after "long"
    assert not _ready(kv, "short").ready, "a later request took the room the head waits for"

    kv.remove_request("a")
    assert _ready(kv, "long").ready and _ready(kv, "short").ready


def test_a_request_that_can_never_fit_is_failed_rather_than_queued():
    kv = _manager(max_num_pages=8)
    kv.ingest_request("huge", _request(list(range(200)), max_tokens=200))
    kv.ingest_request("small", _request(list(range(900, 910)), max_tokens=4))

    outcome = _ready(kv, "huge")

    assert isinstance(outcome.reason, AdmitRuntimeError)
    assert _ready(kv, "small").ready, "a request that can never run still blocked the queue"


def test_a_guessed_count_too_big_for_the_pool_waits_for_it_to_empty():
    kv = _manager(max_num_pages=8)
    kv.ingest_request("a", _request(list(range(20)), max_tokens=4))
    assert _ready(kv, "a").ready
    # keys but no count from the model: prompt and max_tokens are a guess
    guess = _request(list(range(200)), max_tokens=200)
    guess.prompt_slots = guess.decode_labels = None
    kv.ingest_request("guess", guess)

    waiting = _ready(kv, "guess")
    kv.remove_request("a")

    assert waiting.ok and not waiting.ready, "a guess was failed, or passed a request running"
    assert _ready(kv, "guess").ready, "a guess too big for the pool was kept from an empty one"


def test_a_request_nothing_counts_runs_where_a_counted_one_waits():
    kv = _manager(max_num_pages=16)
    kv.ingest_request("a", _request(list(range(100)), max_tokens=60))
    assert _ready(kv, "a").ready
    kv.ingest_request("counted", _request(list(range(500, 600)), max_tokens=60))
    # no count and no keys: max_seq_len is all that would bound it
    kv.ingest_request("uncounted", KVReqConfig(max_tokens=60))

    assert not _ready(kv, "counted").ready
    assert _ready(kv, "uncounted").ready, "a request with nothing to size it by was held back"


# ── the hit ─────────────────────────────────────────────────────────────


def test_the_head_holds_its_hit_as_it_is_admitted():
    kv = _manager(max_num_pages=32)
    prompt = list(range(4 * PAGE_SIZE))
    kv.ingest_request("a", _request(prompt, max_tokens=16))
    assert _ready(kv, "a").ready
    assert _prefill(kv, "a", len(prompt)).ok
    kv.ingest_request("b", _request(prompt, max_tokens=16))

    assert _ready(kv, "b").ready

    stream = kv._streams["b"]["main"]
    assert stream.gate_lease and len(stream.lease) == 3
    assert kv.resolve_cached_prefix("b", NODE, PREFILL) == 3 * PAGE_SIZE, (
        "prepare's probe took a second lease instead of answering from the one held"
    )
    assert kv._reserved["b"].pages == 5 - 3, "the leased pages were reserved as well"


def test_a_lease_prepare_never_cut_to_is_given_back_at_admit():
    kv = _manager(max_num_pages=32)
    prompt = list(range(4 * PAGE_SIZE))
    kv.ingest_request("a", _request(prompt, max_tokens=16))
    assert _ready(kv, "a").ready
    assert _prefill(kv, "a", len(prompt)).ok
    kv.ingest_request("b", _request(prompt, max_tokens=16))
    assert _ready(kv, "b").ready and kv._streams["b"]["main"].gate_lease

    # a guided walk, say: prepare leaves the inputs whole and writes them all
    assert _step(kv, "b", len(prompt)).ok

    stream = kv._streams["b"]["main"]
    assert stream.stored_len == len(prompt), (
        "the lease became the front of a stream whose step wrote the prompt from 0"
    )
    assert kv._reserved["b"].pages == 5, "the pages the lease gave back were not reserved again"


# ── where readiness is skipped ──────────────────────────────────────────


def test_a_step_that_skipped_readiness_waits_its_turn_at_admit():
    kv = _manager(max_num_pages=16)
    kv.ingest_request("a", _request(list(range(100)), max_tokens=60))
    assert _ready(kv, "a").ready
    kv.ingest_request("b", _request(list(range(500, 600)), max_tokens=60))
    free = kv._arena.num_free

    # a step that never asked readiness
    outcome = _step(kv, "b", 100)

    assert isinstance(outcome.reason, AdmissionDeferred)
    assert kv._arena.num_free == free, "a step refused its turn still took pages"


def _two_admitted(kv: KVManager) -> KVManager:
    """``a`` running, and ``b`` needing more room than the pool has left."""
    kv.ingest_request("a", _request(list(range(100)), max_tokens=60))
    assert _step(kv, "a", 100).ok
    kv.ingest_request("b", _request(list(range(500, 600)), max_tokens=60))
    return kv


def test_a_follower_takes_what_rank_zero_would_hold_back():
    leader = _two_admitted(_manager(max_num_pages=16, rank=0, world_size=2))
    follower = _two_admitted(_manager(max_num_pages=16, rank=1, world_size=2))

    # rank 0 admitted b and sent the step on; refusing it here would hang rank 0
    assert not _ready(leader, "b").ready
    assert _ready(follower, "b").ready and _step(follower, "b", 100).ok, (
        "a follower refused a step rank 0 already runs"
    )


def _cfg_parallel_cache(node: str, rank: int = 0, world_size: int = 1) -> KVManager:
    """One worker's cache under CFG parallel, which runs ``node`` of the three."""
    spec = KVSpec(
        resource_key="kv", nodes={"LLM", "LLM_cfg_text", "LLM_cfg_img"},
        config=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=16, page_size=PAGE_SIZE,
        ),
        leader="LLM",
    )
    group = SimpleNamespace(rank=rank, world_size=world_size) if world_size > 1 else None
    return KVManager.build(spec, EngineResourceInfo(
        device=torch.device("cpu"), joint_comm_group=group, kv_dtype=torch.float32,
        nodes=frozenset({node}),
    ))


def test_under_cfg_parallel_the_leaders_cache_decides_for_the_guidance_caches():
    leader = _two_admitted(_cfg_parallel_cache("LLM"))
    guidance = _two_admitted(_cfg_parallel_cache("LLM_cfg_text"))

    # each worker's own queue would order two requests differently
    assert not _ready(leader, "b").ready, "the leader's cache admitted past its room"
    assert _ready(guidance, "b").ready and _step(guidance, "b", 100).ok, (
        "a guidance cache refused a request on its own count"
    )


def test_under_cfg_parallel_and_tp_only_the_leaders_rank_zero_holds_a_request_back():
    held_back = {
        (node, rank)
        for node in ("LLM", "LLM_cfg_text") for rank in (0, 1)
        if not _ready(_two_admitted(_cfg_parallel_cache(node, rank, world_size=2)), "b").ready
    }

    assert held_back == {("LLM", 0)}, (
        "a pool other than the leader's rank 0 admitted a request by its own count"
    )


def test_a_guidance_cache_checks_no_room_of_its_own(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    guidance = _two_admitted(_cfg_parallel_cache("LLM_cfg_text"))

    # admitted by the leader, so past what this cache alone would have let in
    assert _step(guidance, "b", 100).ok, "a guidance cache failed the leader's decision"


def test_a_guidance_cache_short_of_pages_reports_it_rather_than_asserting(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    guidance = _cfg_parallel_cache("LLM_cfg_text")
    guidance.ingest_request("a", _request(list(range(100)), max_tokens=60))
    assert _step(guidance, "a", 100).ok
    guidance.ingest_request("b", _request(list(range(500, 700)), max_tokens=10))

    # b's 13 pages are more than the 8 a left: room only the leader's cache promises
    outcome = _step(guidance, "b", 200)

    assert isinstance(outcome.reason, AllocationFailed), (
        "a guidance cache treated room the leader promised as its own broken promise"
    )


# ── a batch that runs guidance for every row ────────────────────────────


def _guided_prefill(kv: KVManager, rids: tuple[str, ...], span: int):
    """Bagel's prefill_text once one row of the batch needs guidance: every
    row writes main and cfg_img, and main forks onto cfg_text first."""
    step = KVStep(
        segments=tuple(
            Segment(rid, label, span) for label in ("main", "cfg_img") for rid in rids
        ),
        pre_forks=(("main", "cfg_text"),),
    )
    ctx = StepContext(request_ids=rids, graph_walk=PREFILL, slot=0, capture=False)
    assert kv.admit(step, ctx).ok
    views = kv.plan(step, ctx)
    kv.commit(step, ctx)
    return views


def test_a_row_the_batch_guides_takes_no_page_its_request_never_counted(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(max_num_pages=32)
    labels = ["main", "cfg_text", "cfg_img"]
    guided = _request(list(range(500, 500 + 4 * PAGE_SIZE)), max_tokens=PAGE_SIZE)
    guided.needed_labels = labels
    guided.prompt_slots = dict.fromkeys(labels, 4 * PAGE_SIZE)
    kv.ingest_request("text", _request(list(range(4 * PAGE_SIZE)), max_tokens=PAGE_SIZE))
    kv.ingest_request("image", guided)
    assert _ready(kv, "text").ready and _ready(kv, "image").ready

    views = _guided_prefill(kv, ("text", "image"), 4 * PAGE_SIZE)

    assert kv._held_fresh("text") == 4, "the unguided row took pages for the guidance branches"
    view = next(v for v in views["cfg_img"].views if v.request_id == "text")
    assert set(view.page_idxs) == {SINK_PAGE}, "the unguided row's cfg_img was planned off the sink"


def test_a_row_counted_on_labels_its_walks_never_open_takes_no_page_there(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(max_num_pages=32)
    labels = ["main", "cfg_text", "cfg_img"]
    guided = _request(list(range(500, 500 + 4 * PAGE_SIZE)), max_tokens=PAGE_SIZE)
    guided.needed_labels = labels
    guided.prompt_slots = dict.fromkeys(labels, 4 * PAGE_SIZE)
    # Bagel counts a generating request on all three labels, but with guidance
    # off its walks open main alone, and its reservation counts only that
    unguided = _request(list(range(4 * PAGE_SIZE)), max_tokens=PAGE_SIZE)
    unguided.prompt_slots = dict.fromkeys(labels, 4 * PAGE_SIZE)
    kv.ingest_request("unguided", unguided)
    kv.ingest_request("image", guided)
    assert _ready(kv, "unguided").ready and _ready(kv, "image").ready

    views = _guided_prefill(kv, ("unguided", "image"), 4 * PAGE_SIZE)

    assert kv._held_fresh("unguided") == 4, "the unguided row took pages for guidance it never runs"
    view = next(v for v in views["cfg_img"].views if v.request_id == "unguided")
    assert set(view.page_idxs) == {SINK_PAGE}, "the unguided row's cfg_img was planned off the sink"


def _beside_a_guided_request(kv: KVManager) -> KVManager:
    """``text`` two pages into its prompt, and a guided ``image`` beside it."""
    labels = ["main", "cfg_text", "cfg_img"]
    guided = _request(list(range(500, 500 + 4 * PAGE_SIZE)), max_tokens=PAGE_SIZE)
    guided.needed_labels = labels
    guided.prompt_slots = dict.fromkeys(labels, 4 * PAGE_SIZE)
    kv.ingest_request("text", _request(list(range(4 * PAGE_SIZE)), max_tokens=PAGE_SIZE))
    kv.ingest_request("image", guided)
    assert _ready(kv, "text").ready and _ready(kv, "image").ready
    assert _step(kv, "text", 2 * PAGE_SIZE).ok
    return kv


def test_a_guided_fork_copies_nothing_onto_a_label_the_row_holds_nothing_in(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _beside_a_guided_request(_manager(max_num_pages=32))

    # main already holds two pages, so a fork onto cfg_text would copy them
    _guided_prefill(kv, ("text", "image"), 2 * PAGE_SIZE)

    assert "cfg_text" not in kv._streams["text"], "the guided fork copied main onto the unguided row's cfg_text"


def test_a_label_the_row_holds_nothing_in_keeps_no_length(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _beside_a_guided_request(_manager(max_num_pages=32))

    _guided_prefill(kv, ("text", "image"), 2 * PAGE_SIZE)

    assert kv._streams["text"]["cfg_img"].stored_len == 0, (
        "the unguided row's cfg_img kept a length its later views would stretch over the sink"
    )


# ── a stream read from another worker ───────────────────────────────────


def _published(generation: int) -> PublishedKVInfo:
    """Another worker's 100-token ``main``, at its reset ``generation``."""
    return PublishedKVInfo.build_for_rank(rank=0, world_size=1, seq_info={
        "main": KVSequenceInfo(
            seq_len=100, latest_kv_transfer_info="remote",
            page_indices=list(range(7)), reset_generation=generation,
        ),
    })


def test_a_producer_rewind_gives_back_what_the_index_lent():
    kv = _manager(max_num_pages=32)
    tokens = list(range(100))
    kv.ingest_request("a", _request(tokens, max_tokens=8))
    assert _prefill(kv, "a", 100).ok
    kv.remove_request("a")
    kv.ingest_request("b", _request(tokens, max_tokens=8))
    assert kv.admit_retrieve("b", NODE, DECODE, _published(1)).ok
    assert kv._streams["b"]["main"].hits, "the read took nothing from the index"

    # the producer replaced the stream, so the pages the index lent are dropped
    kv.admit_retrieve("b", NODE, DECODE, _published(2))

    assert kv._held_fresh("b") <= kv._reserved["b"].pages, (
        "a rewind dropped the index's pages and left the request to take them past its reservation"
    )


# ── a request moved to the host ─────────────────────────────────────────


def _offloaded_with_a_hit(kv: KVManager) -> str:
    """``b`` running on three pages lent by the index, then moved out."""
    prompt = list(range(4 * PAGE_SIZE))
    kv.ingest_request("a", _request(prompt, max_tokens=2 * PAGE_SIZE))
    assert _ready(kv, "a").ready and _prefill(kv, "a", len(prompt)).ok
    kv.ingest_request("b", _request(prompt, max_tokens=2 * PAGE_SIZE))
    assert _ready(kv, "b").ready and _prefill(kv, "b", len(prompt)).ok
    kv.remove_request("a")
    assert kv.offload("b") > 0, "the request did not move to the host"
    return "b"



class _StubHostPool:
    """The host pool `offload` and `reload` use, minus the copies."""

    def __init__(self):
        self.states: dict = {}

    def offload_stream(self, rid, label, gpu_kv_cache, gpu_page_indices, stored_len, position, **kwargs):
        self.states.setdefault(rid, {})[label] = OffloadedStream(
            cpu_page_indices=list(gpu_page_indices), stored_len=stored_len, position=position, **kwargs,
        )
        return True

    def sync(self):
        pass

    def is_offloaded(self, rid):
        return bool(self.states.get(rid))

    def labels(self, rid):
        return list(self.states.get(rid, {}))

    def remove_request(self, rid):
        self.states.pop(rid, None)


def _on_the_host(kv: KVManager):
    """``b`` running on pages lent by the index, then offloaded: its stream,
    and its hits, reservation and reset generation from before."""
    kv._cpu_pool = _StubHostPool()
    prompt = list(range(4 * PAGE_SIZE))
    kv.ingest_request("a", _request(prompt, max_tokens=2 * PAGE_SIZE))
    assert _ready(kv, "a").ready and _prefill(kv, "a", len(prompt)).ok
    kv.ingest_request("b", _request(prompt, max_tokens=2 * PAGE_SIZE))
    assert _ready(kv, "b").ready and _prefill(kv, "b", len(prompt)).ok
    kv.remove_request("a")
    stream = kv._streams["b"]["main"]
    before = (stream.hits, kv._reserved["b"].pages, stream.reset_generation)
    assert before[0], "b took nothing from the index"
    assert kv.offload("b") > 0, "the request did not move to the host"
    return stream, before


def test_an_offload_gives_back_what_the_index_lent():
    kv = _manager(max_num_pages=32)
    stream, (hits, owed, _) = _on_the_host(kv)

    assert (stream.hits, kv._reserved["b"].pages) == (0, owed + hits), (
        "the offloaded stream still counts the index's pages as lent to it"
    )


def test_an_offload_does_not_tell_readers_the_stream_was_rewritten():
    kv = _manager(max_num_pages=32)
    stream, (_, _, epoch) = _on_the_host(kv)

    assert stream.reset_generation == epoch, "an offload moved the stream's contents epoch"

@requires_cuda
def test_a_reload_takes_back_only_what_the_reservation_kept():
    kv = _manager(max_num_pages=32, cpu_offload_pages=32)
    prompt = list(range(4 * PAGE_SIZE))
    kv.ingest_request("a", _request(prompt, max_tokens=2 * PAGE_SIZE))
    assert _ready(kv, "a").ready and _prefill(kv, "a", len(prompt)).ok
    kv.ingest_request("b", _request(prompt, max_tokens=2 * PAGE_SIZE))
    assert _ready(kv, "b").ready and _prefill(kv, "b", len(prompt)).ok
    owed = kv._reserved["b"].pages - kv._held_fresh("b")

    assert kv.offload("b") > 0 and kv.reload("b")

    # the three lent pages come back as private copies, taken off the free list
    assert kv._reserved["b"].pages - kv._held_fresh("b") == owed, (
        "the reload left the request owed other than the growth it had before"
    )


@requires_cuda
def test_nothing_is_admitted_into_the_room_a_reload_will_take():
    kv = _manager(max_num_pages=16, cpu_offload_pages=32)
    _offloaded_with_a_hit(kv)
    # b comes back on four pages of its own and still owes two of growth: of
    # the fifteen, a request needing ten fits only if b's room goes to it
    kv.ingest_request("c", KVReqConfig(
        max_tokens=1, prompt_slots={"main": 10 * PAGE_SIZE - 1}, decode_labels=["main"],
    ))

    assert not _ready(kv, "c").ready, "a request was admitted into the room b left"
    assert kv.reload("b"), "the room b left was not there when it came back"


# ── the invariant ───────────────────────────────────────────────────────


def _drive(kv: KVManager, rng: random.Random, ops: int) -> list[str]:
    """Requests sharing a few prompts, admitted by readiness, prefilled after
    a probe as prepare does, decoded to their ``max_tokens`` and removed, with
    aborts along the way. Every step an admitted request runs must fit."""
    prompts = [list(range(base, base + rng.randrange(20, 90))) for base in (0, 1000, 2000)]
    waiting: dict[str, tuple[int, int]] = {}
    running: dict[str, list[int]] = {}
    log: list[str] = []
    for i in range(ops):
        roll = rng.random()
        if roll < 0.25 and len(waiting) + len(running) < 8:
            rid = f"r{i}"
            prompt = rng.choice(prompts) + [7] * rng.randrange(0, 20)
            max_tokens = rng.randrange(1, 3 * PAGE_SIZE)
            kv.ingest_request(rid, _request(prompt, max_tokens))
            waiting[rid] = (len(prompt), max_tokens)
            log.append(f"ingest {rid} prompt={len(prompt)} max_tokens={max_tokens}")
        elif roll < 0.5 and waiting:
            rid = rng.choice(list(waiting))
            if _ready(kv, rid).ready:
                prompt, max_tokens = waiting.pop(rid)
                assert _prefill(kv, rid, prompt).ok, f"{rid}'s prefill found no room: {log}"
                running[rid] = [max_tokens]
                log.append(f"admit {rid}")
        elif roll < 0.9 and running:
            rid = rng.choice(list(running))
            span = min(running[rid][0], rng.randrange(1, PAGE_SIZE))
            assert _step(kv, rid, span, DECODE).ok, f"{rid}'s decode found no room: {log}"
            running[rid][0] -= span
            log.append(f"decode {rid} {span}")
            if not running[rid][0]:
                del running[rid]
                kv.remove_request(rid)
                log.append(f"finish {rid}")
        elif waiting or running:
            rid = rng.choice(list(waiting) + list(running))
            waiting.pop(rid, None)
            running.pop(rid, None)
            kv.remove_request(rid)
            log.append(f"abort {rid}")
        kv.assert_pages_conserved()
    return log


def test_no_admitted_request_runs_out_of_pages(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(max_num_pages=24)

    log = _drive(kv, random.Random(SEED), ops=2000)

    assert sum(line.startswith("finish") for line in log) > 50, (
        f"seed {SEED}: too few requests ran to the end to say anything"
    )
    for rid in list(kv._streams):
        kv.remove_request(rid)
    assert kv._arena.num_free + len(kv._index.evictable()) == kv.config.max_num_pages - 1, (
        f"seed {SEED}: pages outlived every request that owned them"
    )
