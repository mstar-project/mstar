"""A matched prefix is held from the probe until admit takes it over.

The probe runs on the CPU when the request is ingested, before anything about
its step exists, so between matching and owning there is a window — now every
scheduling decision in between — in which nothing but the lease stops the pages
being evicted or handed out. Nothing probes again: the step is answered the
length the ingest settled, which is what lets two ranks that ingest at
different moments still skip the same tokens.
"""

from __future__ import annotations

import sys
import threading

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import Resource, StepRunner
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVReqConfig, KVStep
from mstar.engine.resources.kv.keys import chain, fingerprint
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the host pool pins its memory"
)

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
WALK = "prefill"


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


def _manager(max_num_pages: int = 32, cpu_offload_pages: int = 0) -> KVManager:
    kv = KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
            cpu_offload_pages=cpu_offload_pages,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cuda" if cpu_offload_pages else "cpu"),
        dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT)
    return kv


def _ctx(*rids: str) -> StepContext:
    return StepContext(
        request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False,
    )


def _keys(tokens: list[int]) -> list[bytes]:
    """The chain the preprocess worker would send for ``tokens``."""
    return chain([
        tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)
    ])


def _grow(kv: KVManager, rid: str, span: int, label: str = "main", **step) -> None:
    """One step's whole lifecycle: admit reserves, plan copies pre-forks,
    commit advances the lengths."""
    kvstep = KVStep(segments=(Segment(rid, label, span),), **step)
    ctx = _ctx(rid)
    assert kv.admit(kvstep, ctx).ok
    kv.plan(kvstep, ctx)
    kv.commit(kvstep, ctx)


def _seed(kv: KVManager, tokens: list[int]) -> list[bytes]:
    """Run one request to completion and index every full page it wrote.

    Stands in for the indexing that happens at commit; here the point is what
    a later request can then match.
    """
    keys = _keys(tokens)
    kv.ingest_request("seed", KVReqConfig(prefix_keys={"main": keys}))
    _grow(kv, "seed", len(tokens))
    pages = kv._streams["seed"]["main"].page_indices
    parent = None
    for index, key in enumerate(keys[:len(tokens) // PAGE_SIZE]):
        kv._index.insert(fingerprint(ROOT, key), pages[index], parent)
        parent = pages[index]
    return keys


# ── matching ────────────────────────────────────────────────────────────


def test_a_second_request_with_the_same_prompt_matches_what_the_first_wrote():
    kv = _manager()
    keys = _seed(kv, list(range(100)))
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))

    matched = kv.resolve_cached_prefix("r1", NODE, WALK)

    assert matched == 96, f"6 full pages were indexed but {matched} tokens matched"
    kv.assert_pages_conserved()


def test_the_ingest_is_what_walks_the_index():
    kv = _manager()
    keys = _seed(kv, list(range(100)))

    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))

    assert kv._streams["r1"]["main"].lease, (
        "nothing was held until the step asked, so the index the request "
        "matched would be whichever one its rank had reached by then"
    )
    kv.assert_pages_conserved()


def test_a_prefix_committed_after_the_ingest_is_not_picked_up():
    kv = _manager()
    tokens = list(range(100))
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": _keys(tokens)}))

    _seed(kv, tokens)

    assert kv.resolve_cached_prefix("r1", NODE, WALK) is None, (
        "the step matched pages indexed after this request was ingested, so "
        "two ranks keying at their own pace could skip different lengths"
    )
    kv.assert_pages_conserved()


def test_a_prompt_that_diverges_matches_only_what_they_share():
    kv = _manager()
    _seed(kv, list(range(100)))
    diverged = list(range(48)) + [999] * 52

    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": _keys(diverged)}))

    assert kv.resolve_cached_prefix("r1", NODE, WALK) == 48, (
        "the match ran past the page where the two prompts stop agreeing"
    )
    kv.assert_pages_conserved()


def test_a_page_aligned_prompt_loses_its_last_page():
    kv = _manager()
    keys = _seed(kv, list(range(96)))
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))

    assert kv.resolve_cached_prefix("r1", NODE, WALK) == 80, (
        "a full match would leave the request with no token to sample"
    )
    kv.assert_pages_conserved()


def test_a_request_that_matches_nothing_gets_no_lease():
    kv = _manager()
    _seed(kv, list(range(100)))
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": _keys([7] * 100)}))

    assert kv.resolve_cached_prefix("r1", NODE, WALK) is None, (
        "a prompt sharing no page with anything indexed was answered a length"
    )
    assert kv._streams["r1"]["main"].lease is None, (
        "a request that matched nothing is holding pages"
    )
    kv.assert_pages_conserved()


# ── the window ──────────────────────────────────────────────────────────


def test_a_refused_admit_answers_the_same_and_holds_one_lease():
    kv = _manager(max_num_pages=16)
    keys = _seed(kv, list(range(100)))
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))
    first = kv.resolve_cached_prefix("r1", NODE, WALK)
    held = list(kv._streams["r1"]["main"].lease)
    owners = [kv._arena.num_owners[page] for page in held]

    # nothing left for the unmatched tail
    kv.ingest_request("hog", KVReqConfig())
    while kv._arena.num_free:
        kv._arena.acquire(1)
    step = KVStep(segments=(Segment("r1", "main", 100),))
    refused = kv.admit(step, _ctx("r1"))

    assert not refused.ok, "the pool was empty but admit succeeded"
    assert kv.resolve_cached_prefix("r1", NODE, WALK) == first, (
        "the second probe answered a different length"
    )
    assert kv._streams["r1"]["main"].page_indices[:len(held)] == held, (
        "the refused admit gave up the pages it had converted"
    )
    assert [kv._arena.num_owners[page] for page in held] == owners, (
        "the second probe took a second reference"
    )


def test_remove_gives_back_a_lease_admit_never_took():
    kv = _manager()
    keys = _seed(kv, list(range(100)))
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))
    kv.resolve_cached_prefix("r1", NODE, WALK)
    leased = list(kv._streams["r1"]["main"].lease)
    owners = [kv._arena.num_owners[page] for page in leased]

    kv.remove_request("r1")

    assert [kv._arena.num_owners[page] for page in leased] == [
        count - 1 for count in owners
    ], "an unconsumed lease was not given back"
    kv.assert_pages_conserved()


def test_a_reset_gives_back_a_lease_admit_never_took():
    kv = _manager()
    keys = _seed(kv, list(range(100)))
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))
    kv.resolve_cached_prefix("r1", NODE, WALK)

    kv.reset_request("r1")

    assert kv._streams["r1"]["main"].lease is None, (
        "a reset request is still holding a lease nobody will convert"
    )
    kv.assert_pages_conserved()


# ── conversion ──────────────────────────────────────────────────────────


def test_admit_takes_the_lease_over_and_commit_clears_it():
    kv = _manager()
    keys = _seed(kv, list(range(100)))
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))
    matched = kv.resolve_cached_prefix("r1", NODE, WALK)
    leased = list(kv._streams["r1"]["main"].lease)

    _grow(kv, "r1", 100 - matched)

    stream = kv._streams["r1"]["main"]
    assert stream.page_indices[:len(leased)] == leased, (
        "admit allocated fresh pages instead of taking the matched ones"
    )
    assert stream.stored_len == 100, (
        "the matched pages were not counted as tokens this stream holds"
    )
    assert stream.lease is None, "commit left the lease on the stream"
    kv.assert_pages_conserved()


def _converted_by_a_refused_admit(kv: KVManager) -> list[int]:
    """Probe ``r1``, fill the pool, and let an admit convert the lease and refuse.

    The refusal leaves no step in flight, so the stream can be picked as an
    offload victim while it still holds the converted pages.
    """
    keys = _seed(kv, list(range(100)))
    kv.remove_request("seed")
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))
    matched = kv.resolve_cached_prefix("r1", NODE, WALK)
    leased = list(kv._streams["r1"]["main"].lease)
    kv.ingest_request("hog", KVReqConfig())
    _grow(kv, "hog", kv._arena.num_free * PAGE_SIZE)

    step = KVStep(segments=(Segment("r1", "main", 100 - matched),))
    assert not kv.admit(step, _ctx("r1")).ok, "the pool was full but admit succeeded"
    assert kv._streams["r1"]["main"].page_indices == leased, "admit did not convert"
    return leased


@requires_cuda
def test_a_converted_stream_offloaded_before_its_commit_gives_its_pages_back_once():
    kv = _manager(max_num_pages=16, cpu_offload_pages=32)
    leased = _converted_by_a_refused_admit(kv)

    assert kv.offload("r1") > 0, "the refused request did not move to the host"
    kv.remove_request("r1")

    assert [kv._arena.num_owners[page] for page in leased] == [1] * len(leased), (
        "the converted pages were given back once with the stream and again as a "
        "lease, out from under the index"
    )
    kv.assert_pages_conserved()


@requires_cuda
def test_a_converted_stream_answers_the_same_after_an_offload_and_a_reload():
    kv = _manager(max_num_pages=16, cpu_offload_pages=32)
    leased = _converted_by_a_refused_admit(kv)

    assert kv.offload("r1") > 0, "the refused request did not move to the host"
    assert kv.reload("r1"), "the request could not come back"

    assert kv.resolve_cached_prefix("r1", NODE, WALK) == len(leased) * PAGE_SIZE, (
        "after the reload the probe forgot what admit converted, so the retried "
        "step would declare the whole prompt over the tokens already held"
    )
    kv.assert_pages_conserved()


def test_a_refused_admit_retried_trims_the_same_and_commits_once():
    kv = _manager(max_num_pages=16)
    leased = _converted_by_a_refused_admit(kv)
    kv.remove_request("hog")

    retried = kv.resolve_cached_prefix("r1", NODE, WALK)
    _grow(kv, "r1", 100 - retried)

    stream = kv._streams["r1"]["main"]
    assert retried == len(leased) * PAGE_SIZE, "the retry cut a different length"
    assert stream.stored_len == 100 and stream.page_indices[:len(leased)] == leased, (
        "the retried step did not carry on from the converted pages"
    )
    kv.assert_pages_conserved()


def test_a_pre_fork_off_a_leased_stream_covers_the_whole_prefix():
    kv = _manager()
    keys = _seed(kv, list(range(100)))
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))
    matched = kv.resolve_cached_prefix("r1", NODE, WALK)

    _grow(kv, "r1", 100 - matched, pre_forks=(("main", "cfg_text"),))

    forked = kv._streams["r1"]["cfg_text"]
    assert forked.stored_len == matched, (
        "the fork target was sized off a stream whose lease had not converted"
    )
    assert len(forked.page_indices) >= matched // PAGE_SIZE, (
        "the fork target has fewer pages than the prefix it must cover"
    )
    assert not set(forked.page_indices) & set(kv._streams["r1"]["main"].page_indices), (
        "the fork target aliased the pages it was supposed to be a copy of"
    )
    kv.assert_pages_conserved()


# ── the length every resource agreed to ─────────────────────────────────


class _Holding(Resource):
    """A resource that holds a prefix of its own, shorter than the cache's."""

    def __init__(self, matched: int):
        self._matched = matched

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def resolve_cached_prefix(self, rid, node_name, graph_walk):
        return self._matched


def test_a_lease_is_cut_to_the_smallest_answer_and_the_rest_given_back():
    kv = _manager()
    keys = _seed(kv, list(range(100)))
    kv.remove_request("seed")
    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))
    runner = StepRunner(
        {"kv": kv, "other": _Holding(PAGE_SIZE)},
        node_resources={NODE: ["kv", "other"]},
    )

    matched = runner.resolve_cached_prefix("r1", NODE, WALK)
    leased = list(kv._streams["r1"]["main"].lease)
    runner.apply_cached_prefix("r1", NODE, WALK, None, matched)

    assert matched == PAGE_SIZE, "the runner did not take the smallest answer"
    assert kv._streams["r1"]["main"].lease == leased[:1], (
        "the lease still covers pages the step will now recompute"
    )
    assert all(kv._arena.num_owners[page] == 1 for page in leased[1:]), (
        "the pages past the agreed length are held by a lease nobody will convert"
    )
    kv.assert_pages_conserved()


# ── what is done with the lock down ─────────────────────────────────────


def _assert_hashed_with_the_lock_down(kv: KVManager, monkeypatch) -> list[bool]:
    """Fail any key hashed while this manager's lock is held. Returns the
    record of keys hashed, so a caller can tell a pass from a no-op."""
    real = manager_mod.fingerprint
    hashed: list[bool] = []

    def _fingerprint(*fields):
        free = []
        hashed.append(True)

        def _try():
            if kv._lock.acquire(blocking=False):
                kv._lock.release()
                free.append(True)

        attempt = threading.Thread(target=_try)
        attempt.start()
        attempt.join()
        assert free, (
            "a key was hashed with the manager's lock held, so every admit, "
            "commit and remove waited on the chain"
        )
        return real(*fields)

    monkeypatch.setattr(manager_mod, "fingerprint", _fingerprint)
    return hashed


def test_the_ingest_hashes_the_prompt_with_the_lock_down(monkeypatch):
    kv = _manager()
    keys = _seed(kv, list(range(100)))
    kv.remove_request("seed")
    hashed = _assert_hashed_with_the_lock_down(kv, monkeypatch)

    kv.ingest_request("r1", KVReqConfig(prefix_keys={"main": keys}))

    assert hashed, "the ingest hashed nothing, so the check above never ran"
    assert kv.resolve_cached_prefix("r1", NODE, WALK), "the probe matched nothing"
    kv.assert_pages_conserved()


# ── a remove on another thread ──────────────────────────────────────────


def _removed_in_the_window(kv: KVManager, monkeypatch, rid: str) -> None:
    """Remove ``rid`` from another thread once its label has been looked up,
    whenever that thread can take the manager's lock at that moment."""
    real = kv._keyed_label

    def _keyed_label(*args):
        label = real(*args)

        def _remove():
            if kv._lock.acquire(blocking=False):
                try:
                    kv.remove_request(rid)
                finally:
                    kv._lock.release()

        remover = threading.Thread(target=_remove)
        remover.start()
        remover.join()
        return label

    monkeypatch.setattr(kv, "_keyed_label", _keyed_label)


def _removed_while_hashing(kv: KVManager, monkeypatch, rid: str) -> None:
    """Remove ``rid`` from another thread while its chain is being hashed,
    whenever that thread can take the manager's lock at that moment."""
    real = manager_mod.fingerprint

    def _fingerprint(*fields):
        def _remove():
            if kv._lock.acquire(blocking=False):
                try:
                    kv.remove_request(rid)
                finally:
                    kv._lock.release()

        remover = threading.Thread(target=_remove)
        remover.start()
        remover.join()
        return real(*fields)

    monkeypatch.setattr(manager_mod, "fingerprint", _fingerprint)


def test_an_ingest_raced_by_a_remove_raises_nothing(monkeypatch):
    kv = _manager()
    keys = _seed(kv, list(range(64)))
    kv.remove_request("seed")
    _removed_while_hashing(kv, monkeypatch, "r0")

    try:
        kv.ingest_request("r0", KVReqConfig(prefix_keys={"main": keys}))
    except KeyError as error:
        pytest.fail(
            f"a remove while the chain was hashing failed the request: {error!r}"
        )

    assert kv.resolve_cached_prefix("r0", NODE, WALK) is None, (
        "a removed request came back out of the probe holding a lease"
    )
    kv.assert_pages_conserved()


def test_a_chain_extension_raced_by_a_remove_raises_nothing(monkeypatch):
    kv = _manager()
    tokens = list(range(40))
    kv.ingest_request("r0", KVReqConfig(
        prefix_keys={"main": _keys(tokens)},
        prefix_tail={"main": tokens[32:]},
        prefix_decode={"main": "text_inputs"},
    ))
    _grow(kv, "r0", len(tokens))
    _removed_in_the_window(kv, monkeypatch, "r0")

    try:
        kv.extend_prefix_chain(
            "r0", NODE, WALK, {"text_inputs": [torch.tensor([9000])]},
        )
    except KeyError as error:
        pytest.fail(
            f"a remove between the label lookup and the lock failed the "
            f"postprocess path: {error!r}"
        )
    kv.assert_pages_conserved()
