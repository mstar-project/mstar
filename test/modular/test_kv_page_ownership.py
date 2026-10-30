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
from collections import Counter, deque
from importlib.util import find_spec

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import KVCache, PageAllocator
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager, PageArena
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
        cfg=PagedKVConfig(
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


@pytest.mark.parametrize(
    ("seed", "capacity"),
    [(0, 4), (1, 7), (20260920, 11), (20261017, 17)],
    ids=["seed-0-pool-4", "seed-1-pool-7", "seed-20260920-pool-11", "seed-20261017-pool-17"],
)
def test_random_ownership_matches_logical_owners(seed: int, capacity: int):
    """Random acquire, seal, retain and release on a bare arena, checked after
    each against a model of who owns which page, never the arena's own counts."""
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=2, max_seq_len=64,
        max_num_pages=capacity, page_size=4, cpu_offload_pages=0,
    )
    arena = PageArena(
        kv_cache=KVCache(cfg, torch.device("cpu"), torch.float32),
        allocator=PageAllocator(capacity),
    )
    rng = random.Random(seed)
    all_pages = set(range(capacity))
    owners: dict[int, set[int]] = {}
    sealed: set[int] = set()
    history: deque[str] = deque(maxlen=32)
    operation_index = 0

    def references() -> Counter:
        # Owner memberships are the oracle; never read the arena to update it.
        return Counter(page for pages in owners.values() for page in pages)

    def check():
        expected = references()
        free = list(arena.allocator.free_pages.queue)
        assert len(free) == len(set(free)), f"duplicate free pages: {free}"
        assert set(free) == all_pages - expected.keys(), "free page membership differs"
        assert arena.num_free == capacity - len(expected), "free page count differs"
        assert arena.num_owners == [expected[page] for page in range(capacity)], (
            "reference counts differ from logical owners"
        )
        assert arena.sealed == [page in sealed for page in range(capacity)], (
            "sealed pages differ from the model"
        )
        assert sealed <= expected.keys(), "the model left a free page sealed"
        assert all(count == 1 or page in sealed for page, count in expected.items()), (
            "the model shared an unsealed page"
        )
        assert arena.any_sealed([]) is False
        assert arena.any_sealed(sorted(all_pages)) == bool(sealed)
        for page in all_pages:
            assert arena.any_sealed([page]) == (page in sealed)

    def apply(operation: str, owner: int = -1, pages=(), count: int = 0):
        nonlocal operation_index
        operation_index += 1
        pages = list(pages)
        history.append(
            f"{operation_index}: {operation} owner={owner} pages={pages} count={count}"
        )
        try:
            if operation == "acquire":
                assert owner not in owners
                free_before = all_pages - references().keys()
                queue_before = list(arena.allocator.free_pages.queue)
                got = arena.acquire(count)
                history[-1] += f" -> {got}"
                if count > len(free_before):
                    assert got is None, "an over-capacity allocation succeeded"
                    assert list(arena.allocator.free_pages.queue) == queue_before, (
                        "a failed allocation changed the free queue"
                    )
                else:
                    assert got is not None and len(got) == count
                    assert len(set(got)) == count, "an allocation repeated a page"
                    assert set(got) <= free_before, "an allocation took an owned page"
                    if got:
                        owners[owner] = set(got)
            elif operation == "seal":
                assert set(pages) <= references().keys()
                arena.seal(pages)
                sealed.update(pages)
            elif operation == "retain":
                assert owner not in owners and set(pages) <= sealed
                arena.retain(pages)
                if pages:
                    owners[owner] = set(pages)
            elif operation == "release":
                assert set(pages) <= owners[owner]
                arena.release(pages)
                owners[owner].difference_update(pages)
                if not owners[owner]:
                    del owners[owner]
                sealed.intersection_update(references())
            else:
                raise AssertionError(f"unknown operation: {operation}")
            check()
        except Exception as exc:
            exc.add_note(
                f"PageArena seed={seed}, capacity={capacity}, operation={operation_index}\n"
                "Last operations (recorded before execution):\n" + "\n".join(history)
            )
            raise

    # Force sharing, exhaustion, mixed final/non-final releases and reuse;
    # these boundaries must run even when a random sequence avoids them.
    apply("acquire", owner=0, count=3)
    apply("release", owner=0)
    shared, second, exclusive = sorted(owners[0])
    apply("seal", pages=[shared, second, exclusive])
    apply("retain", owner=1, pages=[shared, second])
    apply("retain", owner=2, pages=[shared])
    apply("acquire", owner=3, count=capacity - 3)
    apply("acquire", owner=4, count=1)
    apply("release", owner=0, pages=[shared, exclusive])
    apply("acquire", owner=4, count=2)
    # The sole free page was sealed: reacquiring it must give a writable page.
    apply("acquire", owner=5, count=1)
    apply("release", owner=1, pages=[shared, second])
    apply("release", owner=2, pages=[shared])
    apply("release", owner=0, pages=[second])
    apply("release", owner=3, pages=sorted(owners[3]))
    apply("release", owner=5, pages=sorted(owners[5]))
    apply("acquire", owner=6, count=0)
    apply("seal")
    apply("retain", owner=7)

    for i in range(400):
        roll = rng.randrange(10)
        if not owners or roll < 3:
            apply("acquire", owner=10 + i, count=rng.randint(1, capacity + 1))
        elif roll < 6:
            owner = rng.choice(sorted(owners))
            pages = rng.sample(sorted(owners[owner]), rng.randint(1, len(owners[owner])))
            apply("release", owner=owner, pages=pages)
        elif roll < 8 or not sealed:
            owned = sorted(references())
            pages = rng.sample(owned, rng.randint(1, len(owned)))
            apply("seal", pages=pages)
        else:
            pages = rng.sample(sorted(sealed), rng.randint(1, len(sealed)))
            apply("retain", owner=10 + i, pages=pages)

    for owner in sorted(owners):
        apply("release", owner=owner, pages=sorted(owners[owner]))
    # Exact full-pool reuse also checks that final releases neither lose nor
    # duplicate a page and that no previous lifetime's seal survives.
    apply("acquire", owner=1000, count=capacity)
    apply("release", owner=1000, pages=sorted(owners[1000]))


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


def _drive(kv: KVManager, seed: int, ops: int, offload: bool = False):
    """Random operations against ``kv``, with the invariant checked after each.

    The pool is deliberately too small, so admission is refused often and the
    half-finished paths — a fork reserved, then a segment that does not fit —
    run here too. Failures include the seed and the operations through the
    failing one; successful runs return the log for coverage checks.
    """
    rng = random.Random(seed)
    live: list[str] = []
    log: list[str] = []
    try:
        for i in range(ops):
            roll = rng.random()
            if not live or roll < 0.2:
                rid = f"r{i}"
                log.append(f"ingest {rid}")
                kv.ingest_request(rid)
                live.append(rid)
            elif roll < 0.6:
                rid = rng.choice(live)
                span = rng.randrange(1, 3 * PAGE_SIZE)
                pre = (("main", "alt"),) if rng.random() < 0.25 else ()
                post = (("main", "cfg"),) if rng.random() < 0.25 else ()
                log.append(f"step {rid} span={span} pre={len(pre)} post={len(post)}")
                step = KVStep(
                    segments=(Segment(rid, "main", span),),
                    pre_forks=pre, post_forks=post,
                )
                ctx = _ctx(rid)
                admitted = kv.admit(step, ctx).ok
                log[-1] += f" admitted={admitted}"
                if admitted:
                    kv.plan(step, ctx)
                    kv.commit(step, ctx)
            elif roll < 0.75:
                rid = rng.choice(live)
                free = rng.random() < 0.5
                log.append(f"reset {rid} free={free}")
                kv.reset_request(rid, free=free)
            elif offload and roll < 0.85:
                rid = rng.choice(live)
                log.append(f"offload {rid}")
                kv.offload(rid)
                kv.assert_pages_conserved()
                reloaded = kv.reload(rid)
                log[-1] += f" reloaded={reloaded}"
            else:
                rid = rng.choice(live)
                log.append(f"remove {rid}")
                live.remove(rid)
                kv.remove_request(rid)
            kv.assert_pages_conserved()
    except Exception as error:
        trace = "\n".join(f"{i}: {operation}" for i, operation in enumerate(log))
        error.add_note(f"KV lifecycle seed={seed}, offload={offload}\n{trace}")
        raise
    return log


def _drain(kv: KVManager) -> None:
    for rid in list(kv._streams):
        kv.remove_request(rid)


def test_the_invariant_survives_a_random_lifecycle():
    """Ingest, steps with pre- and post-forks, resets and removes in a random
    order, on a pool too small to hold them all."""
    kv = _manager(max_num_pages=12)

    log = _drive(kv, SEED, ops=400)

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

    log = _drive(kv, SEED, ops=400, offload=True)

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


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.skipif(find_spec("flashinfer") is None, reason="requires the optional FlashInfer backend")
def test_kv_lens_match_flashinfer_seq_lens(page_size):
    """The host lengths handed to FlashInfer's decode plan are exactly what it
    would have rebuilt from the indptrs itself."""
    from flashinfer.page import get_seq_lens

    from mstar.engine.resources.kv.plan import SequenceView, build_paged_indptrs

    views = [
        SequenceView("a", "main", page_idxs=[3], length=1, to_compute=1),
        SequenceView("b", "main", page_idxs=[4, 5], length=page_size + 1, to_compute=1),
        SequenceView("c", "main", page_idxs=[6, 7], length=2 * page_size, to_compute=1),
        SequenceView("d", "main", page_idxs=[], length=0, to_compute=0),
    ]
    ind = build_paged_indptrs(views, page_size)
    expected = get_seq_lens(ind.paged_kv_indptr, ind.paged_kv_last_page_len, page_size)
    assert ind.kv_lens.dtype == expected.dtype
    assert torch.equal(ind.kv_lens, expected)


def test_a_step_forks_only_the_rows_it_names():
    """A chunked prompt forks on its first chunk only: a step holding it beside
    a prompt already past its first chunk forks just the first."""
    kv = _manager()
    for rid in ("r0", "r1"):
        kv.ingest_request(rid)
        _grow(kv, rid, PAGE_SIZE)
    step = KVStep(
        segments=(Segment("r0", "main", 4), Segment("r1", "main", 4)),
        pre_forks=(("main", "alt"),), fork_rids=frozenset({"r0"}),
    )
    ctx = _ctx("r0", "r1")

    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)

    assert "alt" in kv._streams["r0"] and "alt" not in kv._streams["r1"]
    kv.assert_pages_conserved()
