"""Two ranks that matched different lengths skip the same one.

Each rank keeps its own index and probes it, and the two are not symmetric by
construction: a generated page is keyed on each rank's own postprocess path,
whose timing against the next commit is that rank's. So at any instant one rank
holds a page another has not keyed yet, and the length they skip has to be the
smallest any of them matched — one span runs on every rank, and a rank cannot
skip pages it does not hold.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402
import torch  # noqa: E402

from mstar.engine.resources import StepRunner  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import (  # noqa: E402
    KVConfig,
    KVReqConfig,
    KVStep,
)
from mstar.engine.resources.kv.keys import chain  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
WALK = "prefill"
TOKENS = list(range(100))


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


def _keys(tokens: list[int]) -> list[bytes]:
    return chain([
        tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)
    ])


def _rank(indexed: int) -> KVManager:
    """A rank whose index holds the first ``indexed`` pages of ``TOKENS``."""
    kv = KVManager(
        cfg=KVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=64, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv._world_size = 2
    kv.enable_prefix_cache(ROOT)
    seen = TOKENS[:indexed * PAGE_SIZE]
    kv.ingest_request("seed", KVReqConfig(prefix_keys={"main": _keys(seen)}))
    step = KVStep(segments=(Segment("seed", "main", len(seen)),))
    ctx = StepContext(
        request_ids=("seed",), graph_walk=WALK, slot=0, capture=False,
    )
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)
    kv.remove_request("seed")
    return kv


def _engine(kv: KVManager):
    """The two hooks the leader's loop reaches, over a real runner."""
    runner = StepRunner({"kv": kv}, node_resources={NODE: ["kv"]})
    return SimpleNamespace(
        matched_prefixes=lambda node_name, request_id: runner.matched_prefixes(
            request_id, node_name,
        ),
        agree_prefix=lambda node_name, request_id, label, matched: (
            runner.agree_prefix(request_id, node_name, label, matched)
        ),
    )


def _rank_worker(kv: KVManager) -> Worker:
    w = Worker.__new__(Worker)
    w._tp_prefix_replies = {}
    engine = _engine(kv)
    w.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    return w


def _ingest(kv: KVManager, rid: str = "r1") -> None:
    kv.ingest_request(rid, KVReqConfig(prefix_keys={"main": _keys(TOKENS)}))


def _settle(leader: KVManager, follower: KVManager, rid: str = "r1"):
    """Play rank 0's schedule: take the minimum and hand it to both ranks."""
    rank0 = _rank_worker(leader)
    rank1 = _rank_worker(follower)
    rank0._tp_prefix_replies[rid] = {1: follower.matched_prefix(rid)}
    matched = Worker._agree_prefix(rank0, NODE, [rid])
    Worker._apply_agreed_prefix(rank0, NODE, matched)
    Worker._apply_agreed_prefix(rank1, NODE, matched)
    return matched


# ── the minimum ─────────────────────────────────────────────────────────


def test_two_ranks_that_matched_differently_skip_the_same_length():
    long_rank, short_rank = _rank(5), _rank(3)
    _ingest(long_rank)
    _ingest(short_rank)

    _settle(long_rank, short_rank)

    assert (
        long_rank.resolve_cached_prefix("r1", NODE, WALK)
        == short_rank.resolve_cached_prefix("r1", NODE, WALK)
        == 3 * PAGE_SIZE
    ), "the ranks would run the same span from different starting points"


def test_the_longer_rank_gives_back_what_it_agreed_not_to_skip():
    long_rank, short_rank = _rank(5), _rank(3)
    _ingest(long_rank)
    _ingest(short_rank)
    held = list(long_rank._streams["r1"]["main"].lease)

    _settle(long_rank, short_rank)

    assert long_rank._streams["r1"]["main"].lease == held[:3], (
        "the rank holds pages for a prefix the step will now recompute"
    )
    long_rank.assert_pages_conserved()
    short_rank.assert_pages_conserved()


def test_the_shorter_rank_keeps_everything_it_had():
    long_rank, short_rank = _rank(5), _rank(3)
    _ingest(long_rank)
    _ingest(short_rank)
    held = list(short_rank._streams["r1"]["main"].lease)

    _settle(long_rank, short_rank)

    assert short_rank._streams["r1"]["main"].lease == held, (
        "the minimum took pages off the rank that set it"
    )
    short_rank.assert_pages_conserved()


def test_a_rank_that_matched_nothing_settles_the_group_on_nothing():
    long_rank, cold_rank = _rank(5), _rank(0)
    _ingest(long_rank)
    _ingest(cold_rank)

    _settle(long_rank, cold_rank)

    assert long_rank.resolve_cached_prefix("r1", NODE, WALK) == 0, (
        "one rank with an empty index and the others still skipped tokens"
    )
    assert long_rank._streams["r1"]["main"].lease is None, (
        "a lease nothing will convert is holding pages out of the index"
    )
    long_rank.assert_pages_conserved()


# ── what the length survives ────────────────────────────────────────────


def test_the_agreed_length_survives_a_refused_admit():
    long_rank, short_rank = _rank(5), _rank(3)
    _ingest(long_rank)
    _ingest(short_rank)
    _settle(long_rank, short_rank)

    long_rank.ingest_request("hog", KVReqConfig())
    while long_rank._arena.num_free:
        long_rank._arena.acquire(1)
    step = KVStep(segments=(Segment("r1", "main", len(TOKENS) - 48),))
    ctx = StepContext(
        request_ids=("r1",), graph_walk=WALK, slot=0, capture=False,
    )
    assert not long_rank.admit(step, ctx).ok, "the pool was empty but admit succeeded"

    assert long_rank.resolve_cached_prefix("r1", NODE, WALK) == 3 * PAGE_SIZE, (
        "the retried step would skip a different length from the other ranks"
    )


def test_a_later_schedule_carries_no_agreement_and_changes_nothing():
    long_rank, short_rank = _rank(5), _rank(3)
    _ingest(long_rank)
    _ingest(short_rank)
    _settle(long_rank, short_rank)

    second = Worker._agree_prefix(_rank_worker(long_rank), NODE, ["r1"])

    assert second == {}, (
        "the length would be worked out again from replies that were already "
        "spent, after the ranks had moved on from them"
    )


def test_a_follower_keeps_its_lease_for_a_rid_the_message_leaves_out():
    short_rank = _rank(3)
    _ingest(short_rank)
    held = list(short_rank._streams["r1"]["main"].lease)

    Worker._apply_agreed_prefix(_rank_worker(short_rank), NODE, {})

    assert short_rank._streams["r1"]["main"].lease == held, (
        "a schedule that named no length for this request took its pages "
        "away, so an undeclared node would lose what it holds"
    )
    short_rank.assert_pages_conserved()


# ── before the group has answered ───────────────────────────────────────


def test_a_rank_above_one_offers_a_length_without_settling_it():
    rank = _rank(5)

    _ingest(rank)

    assert rank.matched_prefix("r1") == {"main": 5 * PAGE_SIZE}, (
        "the rank brought nothing to the agreement it is part of"
    )
    assert rank.resolve_cached_prefix("r1", NODE, WALK) is None, (
        "a step run now would skip what this rank alone matched, which is the "
        "one length the group has not agreed to"
    )
