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
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVSpec, KVStep
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import AdmissionDeferred, KVManager
from mstar.engine.resources.step import AdmitRuntimeError, Segment, StepContext

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
PREFILL = "prefill"
DECODE = "decode"
SEED = 20260930


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages: int = 16, rank: int = 0, world_size: int = 1) -> KVManager:
    group = None
    if world_size > 1:
        group = SimpleNamespace(rank=rank, world_size=world_size)
    kv = KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=group, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
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

    # a step continuing a speculation onto this node never asked readiness
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


def test_a_pool_other_workers_share_never_refuses():
    # CFG parallel: this worker runs one of the three nodes the cache serves
    spec = KVSpec(
        resource_key="kv", nodes={"LLM", "LLM_cfg_text", "LLM_cfg_img"},
        config=KVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=16, page_size=PAGE_SIZE,
        ),
    )
    alone, shared = (
        _two_admitted(KVManager.build(spec, EngineResourceInfo(
            device=torch.device("cpu"), kv_dtype=torch.float32, nodes=nodes,
        )))
        for nodes in (frozenset(spec.nodes), frozenset({"LLM"}))
    )

    # each worker's own queue would order two requests differently
    assert not _ready(alone, "b").ready
    assert _ready(shared, "b").ready and _step(shared, "b", 100).ok, (
        "a worker refused a request on its own count"
    )


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
