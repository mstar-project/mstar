"""Prefix reuse under TP: rank 0 decides each hit, and the other ranks take the same pages.

Every rank indexes its own copy of the cache, and their page tables have to stay
identical (a sharded DSA prefill gathers cache slots across ranks). So a follower never
looks a prompt up itself: it adopts rank 0's hit, and waits while its own commit of the
step that wrote those pages trails rank 0's. A hit is stamped when it commits, which
every rank does in step order, so the indexes age and evict alike however each rank's
probes interleave with its commits.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import mstar.communication.wire_types  # noqa: E402,F401  (registers the tags)
from mstar.communication.wire import decode, encode  # noqa: E402
from mstar.engine.engine import Engine  # noqa: E402
from mstar.engine.resources import Resource, StepRunner  # noqa: E402
from mstar.engine.resources.base import CachedPrefix  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import (  # noqa: E402
    KVReqConfig,
    KVStep,
    PagedKVConfig,
    PrefixSpan,
)
from mstar.engine.resources.kv.keys import chain  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    PrefixHit,
    ScheduleTPNode,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker import worker as worker_mod  # noqa: E402
from mstar.worker.micro_scheduler import MicroScheduler, TPDivergence  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
WALK = "prefill"


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def remove_request(self, request_id):
        del request_id

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(
    rank: int, max_num_pages: int = 32, opt_in: bool = True, world: int = 2,
) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, num_qo_heads=2,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE, prefix_cache_tp=opt_in,
        ),
        name="kv", joint_comm_group=types.SimpleNamespace(rank=rank, world_size=world),
        transfer_engine_info=None, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert kv.enable_prefix_cache(ROOT) == opt_in
    return kv


def test_above_one_rank_the_cache_stays_off_unless_the_deployment_opts_in():
    assert _manager(0, opt_in=False)._index is None
    assert _manager(0)._index is not None


def _tokens(n: int, seed: int) -> list[int]:
    return [(seed * 7919 + i * 31) % 1000 for i in range(n)]


def _ingest(kv: KVManager, rid: str, tokens: list[int]) -> None:
    """The keys and layout the preprocess worker sends for ``tokens``."""
    whole = len(tokens) // PAGE_SIZE
    pages = [tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)]
    kv.ingest_request(rid, KVReqConfig(
        prefix_keys={"main": chain(pages)}, prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
        prefix_layout={"main": [PrefixSpan(len(tokens), len(tokens), WALK)]},
    ))


def _inputs(tokens: list[int]) -> ARNodeInputs:
    return ARNodeInputs(input_ids=torch.tensor(tokens), input_seq_len=len(tokens))


def _decide(kv: KVManager, rid: str) -> int:
    """Rank 0's probe as it sends the step, before the step is prepared."""
    return kv.resolve_cached_prefix(rid, NODE, WALK, None).tokens


def test_a_cap_keeps_only_whole_pages():
    # a snapshot pool may cap at a length this cache holds only part of a page of
    kv = _manager(0)
    shared = _tokens(3 * PAGE_SIZE + 1, seed=12)
    _ingest(kv, "a", shared)
    _run(kv, "a", shared)
    _ingest(kv, "b", shared[:3 * PAGE_SIZE] + _tokens(4, seed=13))
    assert _decide(kv, "b") == 3 * PAGE_SIZE
    capped = kv.cap_cached_prefix("b", NODE, WALK, CachedPrefix("main", PAGE_SIZE + 3, PAGE_SIZE + 3))
    assert capped == CachedPrefix("main", PAGE_SIZE, PAGE_SIZE)
    assert len(kv._streams["b"]["main"].lease) == 1


def _ctx(rid: str) -> StepContext:
    return StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False)


def _admit(kv: KVManager, rid: str, span: int) -> KVStep:
    step = KVStep(segments=(Segment(rid, "main", span),))
    assert kv.admit(step, _ctx(rid)).ok
    kv.plan(step, _ctx(rid))
    return step


def _run(kv: KVManager, rid: str, tokens: list[int], matched: int = 0) -> None:
    """One prefill step as each rank prepares it past a hit: probe, apply, admit, plan,
    commit."""
    inputs = _inputs(tokens)
    prefix = kv.resolve_cached_prefix(rid, NODE, WALK, inputs)
    assert prefix.tokens == matched
    kv.apply_cached_prefix(rid, NODE, WALK, inputs, prefix)
    kv.commit(_admit(kv, rid, len(tokens) - matched), _ctx(rid))


def _pages(kv: KVManager, rid: str) -> list[int]:
    return kv._streams[rid]["main"].page_indices


class _Submodule:
    def split_inputs(self, graph_walk, fwd_info, inputs, start, end):
        return ARNodeSubmodule.split_inputs(self, graph_walk, fwd_info, inputs, start, end)


def _rank_0_engine(kv: KVManager) -> Engine:
    """Rank 0's engine over ``kv``: decides a step's hits as it sends it, then prepares it."""
    engine = Engine.__new__(Engine)
    engine._resources = {"kv": kv}
    engine._runner = StepRunner({"kv": kv}, node_resources={NODE: ["kv"]})
    engine._submodules = {NODE: SimpleNamespace(submodule=_Submodule())}
    engine._keyed_walks = {NODE: {WALK}}
    engine._prefix_model = "_Model"
    engine._graph_runtime = SimpleNamespace(get_rid_string=lambda rid: f"wire-{rid}")
    return engine


def _batch(rid: str):
    return SimpleNamespace(
        node_name=NODE, graph_walk=WALK, request_ids=(rid,), step_context=_ctx(rid),
        per_request_info_wrapped={rid: None},
    )


class _Capping(Resource):
    """Serves some lengths only, as a recurrent pool's snapshots do: caps a hit at ``at``."""

    def __init__(self, at: int):
        self._at = at

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def cap_cached_prefix(self, rid, node_name, graph_walk, prefix):
        tokens = min(prefix.tokens, self._at)
        return prefix._replace(tokens=tokens, position=prefix.position - prefix.tokens + tokens)


def test_a_follower_takes_rank_0s_hit_and_never_probes_itself():
    ranks = leader, follower = _manager(0), _manager(1)
    shared = _tokens(5 * PAGE_SIZE + 3, seed=0)
    prompt = shared[:5 * PAGE_SIZE] + _tokens(20, seed=1)
    for kv in ranks:
        _ingest(kv, "a", shared)
        _run(kv, "a", shared)
        _ingest(kv, "b", prompt)

    matched = _decide(leader, "b")
    assert matched == 5 * PAGE_SIZE
    # its own index holds the pages, but only rank 0's word counts
    assert follower.resolve_cached_prefix("b", NODE, WALK, _inputs(prompt)).tokens == 0
    pages = leader.cached_prefix_pages("b", NODE, WALK, matched)
    assert follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    # adopted once: asking again takes no second reference
    assert follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    assert follower.resolve_cached_prefix("b", NODE, WALK, _inputs(prompt)).tokens == matched

    for kv in ranks:
        _run(kv, "b", prompt, matched)
    assert _pages(leader, "b") == _pages(follower, "b")
    assert _pages(leader, "b")[:5] == _pages(leader, "a")[:5]
    for kv in ranks:
        kv.assert_pages_conserved()


def test_a_follower_waits_until_its_own_commit_indexes_the_pages():
    leader, follower = _manager(0), _manager(1)
    shared = _tokens(4 * PAGE_SIZE + 5, seed=2)
    prompt = shared[:4 * PAGE_SIZE] + _tokens(9, seed=3)
    steps = {}
    for kv in (leader, follower):
        _ingest(kv, "a", shared)
        steps[kv] = _admit(kv, "a", len(shared))
    leader.commit(steps[leader], _ctx("a"))
    for kv in (leader, follower):
        _ingest(kv, "b", prompt)

    matched = _decide(leader, "b")
    assert matched == 4 * PAGE_SIZE
    pages = leader.cached_prefix_pages("b", NODE, WALK, matched)
    # rank 0 committed "a" first; this rank has admitted it but not committed it
    assert not follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    follower.commit(steps[follower], _ctx("a"))
    assert follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    assert follower.resolve_cached_prefix("b", NODE, WALK, _inputs(prompt)).tokens == matched


def test_a_waiting_follower_hashes_the_prompt_once(monkeypatch):
    """The follower polls its adopt until its own commit catches up; a long prompt is
    thousands of keys, rooted on the first poll only."""
    leader, follower = _manager(0), _manager(1)
    shared = _tokens(4 * PAGE_SIZE + 5, seed=2)
    steps = {}
    for kv in (leader, follower):
        _ingest(kv, "a", shared)
        steps[kv] = _admit(kv, "a", len(shared))
    leader.commit(steps[leader], _ctx("a"))
    for kv in (leader, follower):
        _ingest(kv, "b", shared[:4 * PAGE_SIZE] + _tokens(9, seed=3))
    matched = _decide(leader, "b")
    pages = leader.cached_prefix_pages("b", NODE, WALK, matched)

    hashed = []
    real = manager_mod.fingerprint
    monkeypatch.setattr(manager_mod, "fingerprint", lambda *f: hashed.append(f) or real(*f))
    for _ in range(3):
        assert not follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    # the whole chain, as a probe roots it, on the first poll only
    assert len(hashed) == len(follower._streams["b"]["main"].chain.keys) == 5
    follower.commit(steps[follower], _ctx("a"))
    hashed.clear()
    assert follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)
    assert hashed == []


def test_pages_that_differ_from_rank_0s_are_a_loud_error():
    leader, follower = _manager(0), _manager(1)
    shared = _tokens(3 * PAGE_SIZE + 1, seed=4)
    prompt = shared[:3 * PAGE_SIZE] + _tokens(5, seed=5)
    for kv in (leader, follower):
        _ingest(kv, "a", shared)
        _run(kv, "a", shared)
        _ingest(kv, "b", prompt)
    matched = _decide(leader, "b")
    pages = [page + 1 for page in leader.cached_prefix_pages("b", NODE, WALK, matched)]
    with pytest.raises(RuntimeError, match="diverged"):
        follower.adopt_cached_prefix("b", NODE, WALK, matched, pages)


def test_ranks_evict_alike_however_their_probes_and_commits_interleave():
    """Rank 0 probes a hit before another request commits, rank 1 after: under a stamp
    taken at lookup their indexes would age differently and evict different pages."""
    ranks = leader, follower = _manager(0, max_num_pages=12), _manager(1, max_num_pages=12)
    x, y, z = (_tokens(2 * PAGE_SIZE + 1, seed=s) for s in (6, 7, 8))
    hit = x[:2 * PAGE_SIZE] + _tokens(3, seed=9)
    x_pages = {}
    for kv in ranks:
        for rid, tokens in (("x", x), ("y", y)):
            _ingest(kv, rid, tokens)
            _run(kv, rid, tokens)
            x_pages.setdefault(kv, _pages(kv, rid)[:2])
            kv.remove_request(rid)
        _ingest(kv, "z", z)
        _ingest(kv, "hit", hit)

    z_steps = {kv: _admit(kv, "z", len(z)) for kv in ranks}
    matched = _decide(leader, "hit")  # before z commits
    leader.commit(z_steps[leader], _ctx("z"))
    follower.commit(z_steps[follower], _ctx("z"))  # after z commits
    pages = leader.cached_prefix_pages("hit", NODE, WALK, matched)
    assert follower.adopt_cached_prefix("hit", NODE, WALK, matched, pages)
    for kv in ranks:
        _run(kv, "hit", hit, matched)
        kv.remove_request("hit")
        kv.remove_request("z")

    # x was hit last, so a request that needs most of the pool evicts y and z first
    for kv in ranks:
        _ingest(kv, "big", _tokens(9 * PAGE_SIZE, seed=10))
        _run(kv, "big", _tokens(9 * PAGE_SIZE, seed=10))
    assert sorted(leader._index.pages()) == sorted(follower._index.pages())
    assert _pages(leader, "big") == _pages(follower, "big")
    for kv in ranks:
        assert set(x_pages[kv]) <= set(kv._index.pages())


def test_rank_0_prepares_the_length_it_decided_though_a_commit_since_holds_more():
    """Rank 0 decides step N+1's hits while step N still runs, and prepares N+1 after N
    commits. The followers were told b missed, so b runs whole on rank 0 too, though a's
    commit has since indexed the prefix they share."""
    leader = _manager(0)
    engine = _rank_0_engine(leader)
    a = _tokens(4 * PAGE_SIZE + 5, seed=2)
    b = a[:4 * PAGE_SIZE] + _tokens(9, seed=3)
    _ingest(leader, "a", a)
    a_step = _admit(leader, "a", len(a))
    _ingest(leader, "b", b)
    batch = _batch("b")
    assert engine.decide_cached_prefix(batch) == {}
    leader.commit(a_step, _ctx("a"))
    inputs = ARNodeInputs(input_ids=torch.tensor(b), input_seq_len=len(b))
    assert engine._skip_cached_prefix(batch, "b", inputs).input_seq_len == len(b)
    assert leader._streams["b"]["main"].lease is None


def test_rank_0_holds_only_the_agreed_pages_once_it_decides():
    """The KV cache matches 5 pages and another resource on the node serves 2: rank 0 lets
    the other 3 go as it decides, not when the step is prepared. A request that then fails
    to prepare holds on rank 0 what the followers adopted, and the ranks evict alike."""
    ranks = leader, follower = _manager(0, max_num_pages=12), _manager(1, max_num_pages=12)
    run = StepRunner({"kv": leader, "pool": _Capping(2 * PAGE_SIZE)},
                     node_resources={NODE: ["kv", "pool"]})
    shared, other = _tokens(5 * PAGE_SIZE + 1, seed=20), _tokens(5 * PAGE_SIZE + 1, seed=21)
    hit = shared[:5 * PAGE_SIZE] + _tokens(3, seed=22)
    for kv in ranks:
        for rid, tokens in (("a", shared), ("o", other)):
            _ingest(kv, rid, tokens)
            _run(kv, rid, tokens)
            kv.remove_request(rid)
        _ingest(kv, "b", hit)
    agreed = run.resolve_cached_prefix("b", NODE, WALK, None).tokens
    assert agreed == 2 * PAGE_SIZE
    pages = leader.cached_prefix_pages("b", NODE, WALK, agreed)
    assert follower.adopt_cached_prefix("b", NODE, WALK, agreed, pages)
    assert leader._streams["b"]["main"].lease == follower._streams["b"]["main"].lease == pages
    # b's prepare raises; the step's other request needs most of the pool
    big = _tokens(4 * PAGE_SIZE, seed=23)
    for kv in ranks:
        _ingest(kv, "big", big)
        _run(kv, "big", big)
    assert sorted(leader._index.pages()) == sorted(follower._index.pages())
    assert _pages(leader, "big") == _pages(follower, "big")


def test_under_tp_a_stream_keys_its_prompt_only():
    kv = _manager(1)
    kv.enable_prefix_cache(ROOT, {"main": (WALK, "decode")})
    assert kv._keyed_walks == {"main": frozenset({WALK})}


def test_a_keyed_miss_answers_zero_so_another_resource_cannot_skip_it():
    kv = _manager(0)
    _ingest(kv, "a", _tokens(3 * PAGE_SIZE + 2, seed=11))
    assert kv.resolve_cached_prefix("a", NODE, WALK, None) == CachedPrefix("main", 0, 0)


def test_a_dropped_layout_answers_zero_so_another_resource_cannot_skip_it():
    """A walk that disagrees with its layout ends the stream's chain; while the stream holds
    nothing it still answers 0, or a snapshot pool's hit would skip tokens it lacks."""
    kv = _manager(0, world=1)
    _ingest(kv, "a", _tokens(3 * PAGE_SIZE + 2, seed=14))
    short = _inputs(_tokens(PAGE_SIZE, seed=14))
    assert kv.resolve_cached_prefix("a", NODE, WALK, short) == CachedPrefix("main", 0, 0)
    assert kv._streams["a"]["main"].chain is None
    assert kv.resolve_cached_prefix("a", NODE, WALK, short) == CachedPrefix("main", 0, 0)


def test_a_combined_walk_decides_each_row_under_its_own_walk():
    """Rank 0 probes only the rows whose own walk is keyed, as their prepare does."""
    leader = _manager(0)
    engine = _rank_0_engine(leader)
    shared = _tokens(3 * PAGE_SIZE + 1, seed=15)
    _ingest(leader, "a", shared)
    _run(leader, "a", shared)
    for rid in ("b", "c"):
        _ingest(leader, rid, shared[:3 * PAGE_SIZE] + _tokens(2, seed=16))
    batch = SimpleNamespace(node_name=NODE, request_ids=("b", "c"), step_context=StepContext(
        request_ids=("b", "c"), graph_walk="mixed", slot=0, capture=False,
        request_walks={"b": WALK, "c": "decode"},
    ))
    hits = engine.decide_cached_prefix(batch)
    assert list(hits) == ["wire-b"] and hits["wire-b"].tokens == 3 * PAGE_SIZE
    assert leader._streams["c"]["main"].lease is None


def test_a_follower_adopts_a_combined_walks_rows_under_their_own_walks():
    seen = []
    engine = SimpleNamespace(
        adopt_cached_prefix=lambda node, walk, hits, walks=None: seen.append(walks) or True)
    sched = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        parallel_leader_nodes=set(),
    )
    sched.register_tp_follow(ScheduleTPNode(
        NODE, "mixed", ["r1", "r2"], spec_seq=3, walks=["decode", WALK], walk_idx=[1, 0],
        prefix_hits={"r1": PrefixHit(tokens=64, pages={"kv": [5]})},
    ))
    assert sched.settle_tp_follow_delta()
    assert seen == [{"r1": WALK, "r2": "decode"}]


def test_rank_0s_hits_cross_the_wire():
    body = ScheduleTPNode(
        "LLM", "prefill", ["r1", "r2"],
        prefix_hits={"r2": PrefixHit(tokens=256, pages={"kv": [3, 4], "kv_index": [7, 8]})},
    )
    out = decode(encode(WorkerMessage(WorkerMessageType.SCHEDULE_TP, body))).body
    assert out.prefix_hits == body.prefix_hits
    assert isinstance(out.prefix_hits["r2"], PrefixHit)
    assert ScheduleTPNode("LLM", "decode", ["r1"]).prefix_hits == {}


def _diverged_scheduler():
    """A follower's scheduler whose front head carries a hit this rank cannot take."""
    def adopt(node, walk, hits, walks=None):
        raise RuntimeError("KV kv: prefix index diverged from rank 0")

    engine = SimpleNamespace(adopt_cached_prefix=adopt)
    sched = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        parallel_leader_nodes=set(),
    )
    sched.register_tp_follow(ScheduleTPNode(
        NODE, WALK, ["r1", "r2"], spec_seq=3,
        prefix_hits={"r1": PrefixHit(tokens=64, pages={"kv": [5]})},
    ))
    return sched


def test_a_diverged_hit_is_not_a_step_error():
    """The other ranks already run the step this rank cannot build, so failing its
    requests leaves them in its collective: no step-error handler may catch it."""
    with pytest.raises(TPDivergence, match="diverged"):
        _diverged_scheduler().settle_tp_follow_delta()
    assert not issubclass(TPDivergence, Exception)


class _Stop(BaseException):
    """Ends ``run`` when nothing else does."""


def test_a_follower_whose_hit_diverged_exits(monkeypatch):
    """``Worker.run`` on a follower leaves on the divergence rather than serving on."""
    monkeypatch.setattr(worker_mod, "gc", SimpleNamespace(
        collect=lambda: None, freeze=lambda: None, get_freeze_count=lambda: 0,
    ))
    w = Worker.__new__(Worker)
    w.worker_id = "w1"
    w.scheduler = _diverged_scheduler()
    w.request_state = SimpleNamespace()
    w.enable_nvtx, w._phase_period, w._phase_buf = False, 0, {}
    # the speculation knobs #463 reads when the worker is built
    w._launch_wait_s, w._max_consecutive_spec = 0.005, 1024
    w._in_flight_rids = set()
    w.parallel_groups = SimpleNamespace(barrier_all=lambda: None, restore_default_timeout=lambda: None)
    w.engine_manager = SimpleNamespace(warmup_all=lambda: None)
    w.communicator = SimpleNamespace(send=lambda *a, **k: None, wait_for_work=lambda *a: None)
    passes = iter(range(3))

    def _messages():
        if next(passes) == 2:
            raise _Stop

    w._process_messages = _messages
    for name in (
        "_verify_tp_async_sched_agrees", "_init_engine_thread", "_await_admit_settled",
        "_apply_pending_removes_safe_to_drop", "_fail_requests", "_apply_pending_drains",
        "_apply_removes_whose_step_landed", "_check_ready_tensors", "_poll_stream_buffers",
        "_handle_main_loop_error",
    ):
        setattr(w, name, lambda *a, **k: None)
    with pytest.raises(TPDivergence):
        w.run()


def test_rank_0s_hit_is_plain_ints_whatever_a_resource_answers():
    """The wire takes Python ints only, and rank 0 encodes the hit after its leases are
    taken: a numpy or torch scalar from a resource must not reach it."""
    engine = Engine.__new__(Engine)
    engine._keyed_walks = {NODE: {WALK}}
    engine._graph_runtime = SimpleNamespace(get_rid_string=lambda rid: f"wire-{rid}")
    engine._runner = SimpleNamespace(
        resolve_cached_prefix=lambda rid, node, walk, inputs: CachedPrefix(
            "main", np.int64(2 * PAGE_SIZE), np.int64(2 * PAGE_SIZE)),
        cached_prefix_pages=lambda rid, node, walk, n: {"kv": [torch.tensor(3), np.int64(4)]},
    )
    batch = SimpleNamespace(node_name=NODE, request_ids=("a",), step_context=_ctx("a"))

    body = ScheduleTPNode(NODE, WALK, ["wire-a"], prefix_hits=engine.decide_cached_prefix(batch))
    out = decode(encode(WorkerMessage(WorkerMessageType.SCHEDULE_TP, body))).body
    assert out.prefix_hits == {"wire-a": PrefixHit(tokens=2 * PAGE_SIZE, pages={"kv": [3, 4]})}
