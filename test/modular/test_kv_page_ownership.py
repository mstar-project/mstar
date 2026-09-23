"""A page goes back to the free list when its last owner lets go.

Today that is the same moment its first owner lets go: a page belongs to one
request, the one whose ``page_indices`` names it, and teardown frees it.
Cross-request prefix reuse adds a second owner, the index, so the rule has to
become "when the count reaches zero" before a page can be shared at all. These
tests cover that rule while every count is still one, which is why none of it
changes what the engine does.
"""

from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.step import Segment, StepContext

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the host pool pins its memory"
)

PAGE_SIZE = 16
SEED = 20260920


class _StubTransfer:
    """No engine, no bytes moved: retrieves complete immediately."""

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


def _manager(max_num_pages: int = 16, cpu_offload_pages: int = 0) -> KVManager:
    """A manager on the CPU. Leave ``cpu_offload_pages`` at 0 unless the test
    is marked `requires_cuda`: the host pool pins its memory as it is built,
    and pinning needs a GPU even though this manager does not."""
    return KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
            cpu_offload_pages=cpu_offload_pages,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _ctx(*rids: str) -> StepContext:
    return StepContext(
        request_ids=tuple(rids), graph_walk="w", slot=0, capture=False,
    )


def _grow(kv: KVManager, rid: str, span: int, label: str = "main") -> None:
    """Admit and commit one step extending ``label`` by ``span`` tokens."""
    step = KVStep(segments=(Segment(rid, label, span),))
    ctx = _ctx(rid)
    assert kv.admit(step, ctx).ok
    kv.commit(step, ctx)


# ── the count ───────────────────────────────────────────────────────────


def test_acquire_takes_pages_at_one_owner():
    kv = _manager()

    pages = kv._arena.acquire(3)

    assert [kv._arena.num_owners[page] for page in pages] == [1, 1, 1]


def test_retain_adds_an_owner():
    kv = _manager()
    pages = kv._arena.acquire(2)

    kv._arena.retain(pages)

    assert [kv._arena.num_owners[page] for page in pages] == [2, 2]


def test_a_page_with_an_owner_left_stays_off_the_free_list():
    kv = _manager()
    free = kv._arena.num_free
    pages = kv._arena.acquire(1)
    kv._arena.retain(pages)

    kv._arena.release(pages)

    assert kv._arena.num_owners[pages[0]] == 1
    assert kv._arena.num_free == free - 1, "a page a second owner still reads was freed"


def test_a_page_released_to_zero_is_back_on_the_free_list():
    kv = _manager()
    free = kv._arena.num_free
    pages = kv._arena.acquire(1)
    kv._arena.retain(pages)
    kv._arena.release(pages)

    kv._arena.release(pages)

    assert kv._arena.num_owners[pages[0]] == 0
    assert kv._arena.num_free == free, "the last owner let go and the page stayed out"


def test_the_sink_page_keeps_its_one_owner():
    """It is acquired once at construction and never released."""
    kv = _manager()

    assert kv._arena.num_owners[SINK_PAGE] == 1


# ── through the manager ─────────────────────────────────────────────────


def test_an_admitted_request_owns_each_of_its_pages_once():
    kv = _manager()
    kv.ingest_request("r0")

    _grow(kv, "r0", 100)

    pages = kv._streams["r0"]["main"].page_indices
    assert [kv._arena.num_owners[page] for page in pages] == [1] * len(pages), (
        "nothing calls retain yet, so a request is the only owner of its pages"
    )


# ── sealed pages ────────────────────────────────────────────────────────


def test_a_reset_across_a_sealed_page_drops_the_pages():
    kv = _manager()
    kv.ingest_request("r0")
    _grow(kv, "r0", 100)
    stream = kv._streams["r0"]["main"]
    pages = list(stream.page_indices)
    free = kv._arena.num_free
    kv._arena.seal(pages[:1])

    kv.reset_request("r0")

    assert stream.page_indices == []
    assert kv._arena.num_free == free + len(pages), (
        "the stream kept pages it would have written over while others read them"
    )


def test_a_reset_on_an_unsealed_stream_keeps_its_pages():
    kv = _manager()
    kv.ingest_request("r0")
    _grow(kv, "r0", 100)
    stream = kv._streams["r0"]["main"]
    pages = list(stream.page_indices)
    free = kv._arena.num_free

    kv.reset_request("r0")

    assert stream.stored_len == 0
    assert stream.page_indices == pages
    assert kv._arena.num_free == free, "a rewind with nothing sealed still reuses its pages"


def test_a_sealed_page_comes_back_writable_when_it_is_freed():
    kv = _manager()
    pages = kv._arena.acquire(1)
    kv._arena.seal(pages)

    kv._arena.release(pages)

    assert not kv._arena.any_sealed(pages), "the next owner would inherit the seal"


# ── the invariant ───────────────────────────────────────────────────────


def _drive(kv: KVManager, rng: random.Random, ops: int, offload: bool = False):
    """Random operations against ``kv``, with the invariant checked after each.

    The pool is deliberately too small, so admission is refused often and the
    half-finished paths — a fork reserved, then a segment that does not fit —
    run here too. Returns the log of what it did, so a failure says what led up
    to it.
    """
    live: list[str] = []
    log: list[str] = []
    for i in range(ops):
        roll = rng.random()
        if not live or roll < 0.2:
            rid = f"r{i}"
            kv.ingest_request(rid)
            live.append(rid)
            log.append(f"ingest {rid}")
        elif roll < 0.6:
            rid = rng.choice(live)
            span = rng.randrange(1, 3 * PAGE_SIZE)
            pre = (("main", "alt"),) if rng.random() < 0.25 else ()
            post = (("main", "cfg"),) if rng.random() < 0.25 else ()
            step = KVStep(
                segments=(Segment(rid, "main", span),),
                pre_forks=pre, post_forks=post,
            )
            ctx = _ctx(rid)
            admitted = kv.admit(step, ctx).ok
            if admitted:
                kv.plan(step, ctx)
                kv.commit(step, ctx)
            log.append(
                f"step {rid} span={span} pre={len(pre)} post={len(post)} "
                f"admitted={admitted}"
            )
        elif roll < 0.75:
            rid = rng.choice(live)
            free = rng.random() < 0.5
            kv.reset_request(rid, free=free)
            log.append(f"reset {rid} free={free}")
        elif offload and roll < 0.85:
            rid = rng.choice(live)
            kv.offload(rid)
            kv.assert_pages_conserved()
            reloaded = kv.reload(rid)
            log.append(f"offload {rid} reloaded={reloaded}")
        else:
            rid = rng.choice(live)
            live.remove(rid)
            kv.remove_request(rid)
            log.append(f"remove {rid}")
        kv.assert_pages_conserved()
    return log


def _drain(kv: KVManager) -> None:
    for rid in list(kv._streams):
        kv.remove_request(rid)


def test_the_invariant_survives_a_random_lifecycle():
    """Ingest, steps with pre- and post-forks, resets and removes in a random
    order, on a pool too small to hold them all."""
    kv = _manager(max_num_pages=12)

    log = _drive(kv, random.Random(SEED), ops=400)

    assert sum(1 for line in log if "admitted=True" in line) > 50, (
        f"seed {SEED}: the pool was too small for the fuzz to store anything"
    )
    _drain(kv)
    assert kv._arena.num_free == kv.config.max_num_pages - 1, (
        f"seed {SEED}: pages outlived every request that owned them"
    )


@requires_cuda
def test_the_invariant_survives_offload_and_reload():
    """The same fuzz with the host pool in play. Offload hands the device pages
    back while the request lives on, which is the one place today where a page
    leaves an owner that is still around."""
    kv = _manager(max_num_pages=12, cpu_offload_pages=16)

    log = _drive(kv, random.Random(SEED), ops=400, offload=True)

    assert any("offload" in line for line in log), f"seed {SEED}: nothing offloaded"
    _drain(kv)
    assert kv._arena.num_free == kv.config.max_num_pages - 1, (
        f"seed {SEED}: pages outlived every request that owned them"
    )


def test_the_flag_puts_the_check_on_the_lifecycle(monkeypatch):
    """This is about the four call sites rather than the invariant itself: with
    the flag on, a count that disagrees with the streams is caught by the very
    operation that broke it."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager()
    kv.ingest_request("r0")
    _grow(kv, "r0", 100)

    kv._arena.retain(kv._streams["r0"]["main"].page_indices[:1])

    with pytest.raises(AssertionError, match="owner counts disagree"):
        _grow(kv, "r0", PAGE_SIZE)
