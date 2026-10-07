"""Admitting by the peak a set can reach, and letting a request behind the head past it.

The pool's default is the summed test in strict arrival order (see
``test_kv_admission``). A pool can instead fit a request against the peak the
admitted set will reach, round by round (``admission_fit: peak``), and let a
request behind the one that has waited longest in if it does not delay it
(``admission_order: backfill``, protected by EASY).

The peak test assumes the requests advance together, and mstar does not make
them. So in ``peak`` mode each page grant is checked: it is refused, for that
request alone, if the admitted requests could not all finish afterwards. A
refusal is a ``GrantDeferred``, which the worker answers by holding that one
request, not the batch it was in.
"""

from __future__ import annotations

import json
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import AllocationFailed, GrantDeferred, Segment, StepContext
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import Worker

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
PREFILL = "prefill"
DECODE = "decode"
SEED = 20261006


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
def _isolated(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    # the config says what a test means; the environment must not overrule it
    for name in ("MSTAR_KV_ADMISSION_FIT", "MSTAR_KV_ADMISSION_ORDER", "MSTAR_STEP_TELEMETRY_DIR"):
        monkeypatch.delenv(name, raising=False)


FIT_AND_ORDER = [("sum", "backfill"), ("peak", "backfill")]


def _manager(
    max_num_pages: int = 16, fit: str = "peak", order: str = "backfill",
    rank: int = 0, world_size: int = 1,
) -> KVManager:
    group = SimpleNamespace(rank=rank, world_size=world_size) if world_size > 1 else None
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
            admission_fit=fit, admission_order=order,
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


_next_base = iter(range(0, 10**9, 1000))


def _ingest(kv: KVManager, rid: str, prompt: int, max_tokens: int) -> None:
    """A request whose prompt shares nothing with any other."""
    base = next(_next_base)
    kv.ingest_request(rid, _request(list(range(base, base + prompt)), max_tokens))


def _ready(kv: KVManager, rid: str):
    return kv.admit_retrieve(rid, NODE, PREFILL, None)


def _run(kv: KVManager, spans: dict[str, int], walk: str = DECODE):
    """One batched step over ``spans``: admit, and if it was let in, plan and commit."""
    step = KVStep(segments=tuple(Segment(rid, "main", span) for rid, span in spans.items()))
    ctx = StepContext(request_ids=tuple(spans), graph_walk=walk, slot=0, capture=False)
    outcome = kv.admit(step, ctx)
    if outcome.ok:
        kv.plan(step, ctx)
        kv.commit(step, ctx)
    return outcome


def _prefill(kv: KVManager, rid: str, prompt: int):
    """What prepare does before the step: probe the cache and cut the inputs."""
    matched = kv.resolve_cached_prefix(rid, NODE, PREFILL) or 0
    kv.apply_cached_prefix(rid, NODE, PREFILL, None, matched)
    return _run(kv, {rid: prompt - matched}, PREFILL)


def _finish(kv: KVManager):
    for rid in list(kv._streams):
        kv.remove_request(rid)
    assert kv._arena.num_free + len(kv._index.evictable()) == kv.config.max_num_pages - 1


# ── backfill ────────────────────────────────────────────────────────────


def _head_waits_behind_a(kv: KVManager) -> None:
    """``a`` running on 7 of the 15 pages it reserves 8 of; ``head`` needs 10."""
    _ingest(kv, "a", prompt=108, max_tokens=20)
    assert _ready(kv, "a").ready and _prefill(kv, "a", 108).ok
    _ingest(kv, "head", prompt=140, max_tokens=20)
    assert not _ready(kv, "head").ready


@pytest.mark.parametrize("fit", ["sum", "peak"])
def test_a_short_request_passes_a_head_that_is_waiting(fit):
    """The mirror of ``test_nothing_passes_the_head_of_the_queue``."""
    kv = _manager(max_num_pages=16, fit=fit, order="backfill")
    kv.ingest_request("a", _request(list(range(100)), max_tokens=60))
    kv.ingest_request("long", _request(list(range(500, 600)), max_tokens=60))
    kv.ingest_request("short", _request(list(range(900, 910)), max_tokens=4))
    assert _ready(kv, "a").ready
    assert not _ready(kv, "long").ready

    assert _ready(kv, "short").ready, "a request that fits and ends first waited behind the head"
    assert "long" in kv._waiting and "long" not in kv._reserved

    kv.remove_request("a")
    assert _ready(kv, "long").ready, "the head was not admitted once there was room"


def test_nothing_passes_the_head_in_arrival_order():
    kv = _manager(max_num_pages=16, fit="peak", order="fifo")
    kv.ingest_request("a", _request(list(range(100)), max_tokens=60))
    kv.ingest_request("long", _request(list(range(500, 600)), max_tokens=60))
    kv.ingest_request("short", _request(list(range(900, 910)), max_tokens=4))
    assert _ready(kv, "a").ready
    assert not _ready(kv, "long").ready

    assert not _ready(kv, "short").ready, "peak alone let a request pass the head"


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_request_that_would_delay_the_head_is_refused(fit, order):
    kv = _manager(max_num_pages=16, fit=fit, order=order)
    _head_waits_behind_a(kv)
    # 6 pages for 30 rounds: still there when a goes at round 20, and the head
    # needs 10 of the 15 pages, leaving 5
    _ingest(kv, "delays", prompt=60, max_tokens=30)

    assert not _ready(kv, "delays").ready, "a request still running when the head starts passed it"


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_request_gone_before_the_head_starts_is_admitted(fit, order):
    kv = _manager(max_num_pages=16, fit=fit, order=order)
    _head_waits_behind_a(kv)
    # the same 6 pages, but gone at round 20, as a is
    _ingest(kv, "quick", prompt=76, max_tokens=20)

    assert _ready(kv, "quick").ready, "a request that ends before the head could start waited"


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_request_that_fits_beside_the_head_is_admitted(fit, order):
    kv = _manager(max_num_pages=16, fit=fit, order=order)
    _head_waits_behind_a(kv)
    # 5 pages for 40 rounds: it outlasts a, but the head leaves exactly that spare
    _ingest(kv, "beside", prompt=40, max_tokens=40)

    assert _ready(kv, "beside").ready, "a request that fits beside the head waited behind it"


@pytest.mark.parametrize(("fit", "order"), FIT_AND_ORDER)
def test_a_backfilled_request_is_leased_its_hit_like_a_head(fit, order):
    kv = _manager(max_num_pages=32, fit=fit, order=order)
    prompt = list(range(4 * PAGE_SIZE))
    kv.ingest_request("a", _request(prompt, max_tokens=16))
    assert _ready(kv, "a").ready and _prefill(kv, "a", len(prompt)).ok
    kv.ingest_request("big", _request(list(range(5000, 5000 + 26 * PAGE_SIZE)), max_tokens=PAGE_SIZE))
    assert not _ready(kv, "big").ready
    kv.ingest_request("b", _request(prompt, max_tokens=16))

    assert _ready(kv, "b").ready, "a request behind the head was not admitted"

    stream = kv._streams["b"]["main"]
    assert stream.gate_lease and len(stream.lease) == 3, "the hit was not leased as it was admitted"
    assert kv.resolve_cached_prefix("b", NODE, PREFILL) == 3 * PAGE_SIZE
    assert kv._reserved["b"].pages == 5 - 3


# ── the peak ────────────────────────────────────────────────────────────


def test_peak_admits_a_set_that_the_summed_test_refuses(monkeypatch):
    """Pool of 10 pages. A holds 2 and grows to 8; B holds 2, grows to 4, and ends
    early: the reservations sum to 12, and the pool is never asked for more than 8."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    outcomes = {}
    for fit in ("sum", "peak"):
        kv = _manager(max_num_pages=11, fit=fit, order="fifo")
        _ingest(kv, "a", prompt=2 * PAGE_SIZE, max_tokens=6 * PAGE_SIZE)
        assert _ready(kv, "a").ready and _prefill(kv, "a", 2 * PAGE_SIZE).ok
        _ingest(kv, "b", prompt=2 * PAGE_SIZE, max_tokens=2 * PAGE_SIZE)
        outcomes[fit] = _ready(kv, "b").ready
        if fit == "peak":
            # and they do run to the end, advancing together
            assert _prefill(kv, "b", 2 * PAGE_SIZE).ok
            for _ in range(2):
                assert _run(kv, {"a": PAGE_SIZE, "b": PAGE_SIZE}).ok
            kv.remove_request("b")
            for _ in range(4):
                assert _run(kv, {"a": PAGE_SIZE}).ok
            kv.remove_request("a")
            _finish(kv)

    assert outcomes == {"sum": False, "peak": True}


# ── the guard ───────────────────────────────────────────────────────────


def _counterexample(kv: KVManager) -> None:
    """Pool of 20. A and B claim 10 each and hold nothing, C holds 4 and claims 6.

    The peak test admits all three, as C is gone at round 32 and the other two
    only reach 20 together at round 144. The summed test refuses the set.
    """
    _ingest(kv, "c", prompt=4 * PAGE_SIZE, max_tokens=2 * PAGE_SIZE)
    assert _ready(kv, "c").ready and _prefill(kv, "c", 4 * PAGE_SIZE).ok
    for rid in ("a", "b"):
        _ingest(kv, rid, prompt=PAGE_SIZE, max_tokens=9 * PAGE_SIZE)
        assert _ready(kv, rid).ready, f"the peak test refused {rid}"
    assert kv._reserved["a"].pages == kv._reserved["b"].pages == 10
    assert kv._reserved["c"].pages == 6


def test_the_summed_test_refuses_the_counterexample():
    kv = _manager(max_num_pages=21, fit="sum", order="fifo")
    _ingest(kv, "c", prompt=4 * PAGE_SIZE, max_tokens=2 * PAGE_SIZE)
    assert _ready(kv, "c").ready and _prefill(kv, "c", 4 * PAGE_SIZE).ok
    _ingest(kv, "a", prompt=PAGE_SIZE, max_tokens=9 * PAGE_SIZE)
    assert _ready(kv, "a").ready
    _ingest(kv, "b", prompt=PAGE_SIZE, max_tokens=9 * PAGE_SIZE)

    assert not _ready(kv, "b").ready


def _grow_a_and_b_until_refused(kv: KVManager):
    """Advance A and B alone, a page at a time, as a scheduler that never picks C would."""
    assert _prefill(kv, "a", PAGE_SIZE).ok and _prefill(kv, "b", PAGE_SIZE).ok
    for _ in range(40):
        for rid in ("a", "b"):
            outcome = _run(kv, {rid: PAGE_SIZE})
            if not outcome.ok:
                return rid, outcome
    raise AssertionError("A and B grew to the end of the pool without being refused")


def test_the_guard_refuses_the_grant_that_leaves_no_way_to_finish(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(max_num_pages=21, fit="peak", order="fifo")
    _counterexample(kv)

    rid, outcome = _grow_a_and_b_until_refused(kv)

    assert isinstance(outcome.reason, GrantDeferred), (
        f"the refusal was {type(outcome.reason).__name__}: a refused grant must not look "
        "like a shortage of pages, which the worker answers by holding the whole batch"
    )
    assert outcome.reason.request_id == rid and outcome.reason.label == "main"
    held = {r: kv._held_fresh(r) for r in ("a", "b", "c")}
    assert held["c"] == 4 and kv._arena.num_free == 20 - sum(held.values())
    assert kv._arena.num_free >= 1, "the guard let the pool run out before refusing"
    assert kv._admitted_are_safe(), "a refused grant left an unsafe state"


def test_advancing_c_frees_the_room_and_every_request_finishes(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(max_num_pages=21, fit="peak", order="fifo")
    _counterexample(kv)
    refused, _ = _grow_a_and_b_until_refused(kv)

    # C runs to its end, which the guard never stands in the way of
    assert _run(kv, {"c": PAGE_SIZE}).ok and _run(kv, {"c": PAGE_SIZE}).ok
    kv.remove_request("c")

    left = {"a": 9 * PAGE_SIZE, "b": 9 * PAGE_SIZE}
    for rid in left:
        left[rid] -= kv._streams[rid]["main"].stored_len - PAGE_SIZE
    for _ in range(200):
        if not left:
            break
        for rid in list(left):
            outcome = _run(kv, {rid: min(PAGE_SIZE, left[rid])})
            if outcome.ok:
                left[rid] -= min(PAGE_SIZE, left[rid])
            else:
                assert isinstance(outcome.reason, GrantDeferred)
            if outcome.ok and left[rid] == 0:
                del left[rid]
                kv.remove_request(rid)
    assert not left, f"{refused} and the other never finished: {left}"
    _finish(kv)


class _Runtime:
    def __init__(self):
        self.pushed_back: list[str] = []

    def push_back_node(self, node_name, rids, wg_ids):
        del node_name, wg_ids
        self.pushed_back.extend(rids)


class _FakeWorker:
    """Binds the admit-failure handler onto stubs for its collaborators."""

    _handle_admit_failure = Worker._handle_admit_failure
    _push_back_batch = Worker._push_back_batch

    def __init__(self):
        self._graph_runtime = _Runtime()
        self.held: list[str] = []
        self.scheduler = SimpleNamespace(hold_requests=self.held.extend)
        self.offload_calls: list[str] = []

    def _handle_allocation_failure(self, batch, node_batch):
        # the real one holds the whole batch, or offloads a victim
        self.offload_calls.append(node_batch.node_name)
        self._push_back_batch(batch)
        self.scheduler.hold_requests(list(batch.request_to_worker_graph))


def _batch_of(rids):
    batch = ScheduledBatch(
        node_name=NODE, graph_walk=DECODE, request_to_worker_graph=dict.fromkeys(rids, "wg"),
    )
    return batch, SimpleNamespace(node_name=NODE, admit_error=None, failed_resource=None)


def _two_in_a_batch_one_unsafe(first: str, second: str):
    """A and C in one decode batch, A's grant unsafe and C's not."""
    kv = _manager(max_num_pages=21, fit="peak", order="fifo")
    _counterexample(kv)
    # A and B to 7 pages each: 2 free, which C needs, and one more to A would take one
    assert _prefill(kv, "a", PAGE_SIZE).ok and _prefill(kv, "b", PAGE_SIZE).ok
    for _ in range(6):
        assert _run(kv, {"a": PAGE_SIZE}).ok and _run(kv, {"b": PAGE_SIZE}).ok
    spans = {first: PAGE_SIZE, second: PAGE_SIZE}
    return kv, spans


@pytest.mark.parametrize(("first", "second"), [("a", "c"), ("c", "a")])
def test_a_refused_grant_holds_only_its_request_not_the_batch(first, second):
    kv, spans = _two_in_a_batch_one_unsafe(first, second)
    batch, node_batch = _batch_of(["a", "c"])

    outcome = _run(kv, spans)
    assert isinstance(outcome.reason, GrantDeferred) and outcome.reason.request_id == "a"
    node_batch.admit_error = outcome.reason
    worker = _FakeWorker()
    worker._handle_admit_failure(batch, node_batch)

    assert sorted(worker._graph_runtime.pushed_back) == ["a", "c"], "the batch was not re-queued"
    assert worker.held == ["a"], f"held {worker.held}: C is what has to progress"
    assert worker.offload_calls == [], "a deferred grant went down the offload path"
    # what an earlier segment of the refused admit was granted stays with its stream
    # and is counted, so the state is still one every request can finish from
    assert kv._admitted_are_safe()
    # and with A held, C goes alone and gets its page
    assert _run(kv, {"c": PAGE_SIZE}).ok


def test_a_refusal_that_is_not_a_deferral_still_holds_the_batch():
    worker = _FakeWorker()
    batch, node_batch = _batch_of(["a", "c"])
    node_batch.admit_error = AllocationFailed(
        message="out of pages", pages_short=1, label="main", request_id="a",
    )

    worker._handle_admit_failure(batch, node_batch)

    assert sorted(worker.held) == ["a", "c"]
    assert worker.offload_calls == [NODE]


# ── a guidance cache does not decide ────────────────────────────────────


def _guidance_cache() -> KVManager:
    """The cache of ``LLM_cfg_text`` under CFG parallel: the leader admits for it."""
    from mstar.engine.resources.kv.config import KVSpec

    spec = KVSpec(
        resource_key="kv", nodes={"LLM", "LLM_cfg_text", "LLM_cfg_img"}, leader="LLM",
        config=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=16, page_size=PAGE_SIZE,
            admission_fit="peak", admission_order="backfill",
        ),
    )
    info = SimpleNamespace(
        device=torch.device("cpu"), joint_comm_group=None, transfer_engine_info=None,
        kv_dtype=torch.float32, needs_remote_transfer=False, nodes=frozenset({"LLM_cfg_text"}),
    )
    kv = KVManager.build(spec, info)
    assert not kv._leads and kv._peak
    return kv


def test_a_guidance_cache_does_not_check_a_grant_the_leader_already_decided(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    guidance = _guidance_cache()
    _ingest(guidance, "a", prompt=100, max_tokens=60)
    assert _run(guidance, {"a": 100}, PREFILL).ok
    _ingest(guidance, "b", prompt=100, max_tokens=60)

    assert _run(guidance, {"b": 100}, PREFILL).ok, "a guidance cache refused the leader's grant"


def test_a_guidance_cache_short_of_pages_reports_a_failed_allocation_not_a_deferral(monkeypatch):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    guidance = _guidance_cache()
    _ingest(guidance, "a", prompt=100, max_tokens=60)
    assert _run(guidance, {"a": 100}, PREFILL).ok
    _ingest(guidance, "b", prompt=200, max_tokens=10)

    outcome = _run(guidance, {"b": 200}, PREFILL)

    assert isinstance(outcome.reason, AllocationFailed) and not isinstance(
        outcome.reason, GrantDeferred
    ), "the guard ran on a pool that decides nothing"


# ── configuration ───────────────────────────────────────────────────────


def test_a_pool_admits_by_the_summed_test_in_arrival_order_unless_told_otherwise():
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=16, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )

    assert (kv._fit, kv._order, kv._planned) == ("sum", "fifo", False)


def test_the_environment_overrules_the_yaml(monkeypatch):
    cfg = PagedKVConfig(num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096)
    cfg.apply_yaml_overrides(admission_fit="sum", admission_order="fifo")
    assert cfg.resolved_admission() == ("sum", "fifo")

    monkeypatch.setenv("MSTAR_KV_ADMISSION_FIT", "peak")
    monkeypatch.setenv("MSTAR_KV_ADMISSION_ORDER", "backfill")
    assert cfg.resolved_admission() == ("peak", "backfill")

    kv = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert (kv._peak, kv._backfill) == (True, True), "the environment did not reach the pool"


def _bagel_spec():
    """The spec Bagel hands the engine: a PagedKVConfig with no admission keys set."""
    from mstar.engine.resources.kv.config import KVSpec

    return KVSpec(
        resource_key="kv", nodes={"LLM"}, leader="LLM",
        config=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=16, page_size=PAGE_SIZE, num_qo_heads=4,
        ),
    )


def _build(spec):
    info = SimpleNamespace(
        device=torch.device("cpu"), joint_comm_group=None, transfer_engine_info=None,
        kv_dtype=torch.float32, needs_remote_transfer=False, nodes=None,
    )
    return KVManager.build(spec, info)


def test_the_environment_reaches_the_pool_the_engine_builds_from_a_models_spec(monkeypatch):
    plain = _build(_bagel_spec())
    assert (plain._fit, plain._order) == ("sum", "fifo")

    monkeypatch.setenv("MSTAR_KV_ADMISSION_FIT", "peak")
    monkeypatch.setenv("MSTAR_KV_ADMISSION_ORDER", "backfill")
    kv = _build(_bagel_spec())

    assert (kv._fit, kv._order, kv._planned) == ("peak", "backfill", True)


def test_the_yaml_reaches_the_pool_the_engine_builds_from_a_models_spec():
    spec = _bagel_spec()
    spec.apply_yaml_overrides(admission_fit="peak", admission_order="backfill")

    kv = _build(spec)

    assert (kv._fit, kv._order) == ("peak", "backfill")


@pytest.mark.parametrize("key", ["admission_fit", "admission_order", "backfill_protect"])
def test_a_mode_that_is_not_one_is_an_error_not_a_default(key):
    cfg = PagedKVConfig(num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096)

    with pytest.raises(ValueError, match=key):
        cfg.apply_yaml_overrides(**{key: "sideways"})


def test_a_mode_in_the_environment_that_is_not_one_is_an_error(monkeypatch):
    monkeypatch.setenv("MSTAR_KV_ADMISSION_FIT", "sideways")
    cfg = PagedKVConfig(num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096)

    with pytest.raises(ValueError, match="MSTAR_KV_ADMISSION_FIT"):
        cfg.resolved_admission()


# ── the event log ───────────────────────────────────────────────────────


def _rows(directory) -> list[dict]:
    from mstar.engine.resources.kv import admission_log

    admission_log.flush_all()
    return [
        json.loads(line)
        for path in sorted(directory.glob("admission_kv_pid*.jsonl"))
        for line in path.read_text().splitlines()
    ]


def test_the_pool_logs_what_it_decides_when_asked_to(monkeypatch, tmp_path):
    monkeypatch.setenv("MSTAR_STEP_TELEMETRY_DIR", str(tmp_path))
    kv = _manager(max_num_pages=21, fit="peak", order="backfill")
    _counterexample(kv)
    _ingest(kv, "d", prompt=PAGE_SIZE, max_tokens=9 * PAGE_SIZE)
    for _ in range(3):
        _ready(kv, "d")
    rid, outcome = _grow_a_and_b_until_refused(kv)
    for name in ("a", "b", "c", "d"):
        kv.remove_request(name)

    rows = _rows(tmp_path)

    by_event = {}
    for row in rows:
        by_event.setdefault(row["event"], []).append(row)
    assert {r["rid"] for r in by_event["reserve"]} == {"a", "b", "c"}
    assert [r["rid"] for r in by_event["wait"]] == ["d"], "a wait was logged for every ask"
    assert by_event["grant_deferred"][0]["rid"] == rid
    assert {r["rid"] for r in by_event["release"]} == {"a", "b", "c"}
    assert all(r["reserved_s"] >= 0 for r in by_event["release"])
    row = by_event["reserve"][0]
    assert {
        "ts_wall", "ts_mono", "pool", "rid", "event", "is_head", "fit", "order",
        "planned_peak", "plan_capacity", "supply", "n_reserved", "n_waiting", "planner_us",
    } <= row.keys()
    assert (row["fit"], row["order"], row["pool"]) == ("peak", "backfill", "kv")


def test_the_pool_logs_nothing_when_not_asked_to(tmp_path):
    kv = _manager(max_num_pages=21, fit="peak", order="backfill")
    _counterexample(kv)

    assert kv._alog is None
    assert list(tmp_path.iterdir()) == []


def test_a_default_pool_decides_the_same_with_the_log_on(monkeypatch, tmp_path):
    """The summed test in arrival order, run through the logged path, answers as it does unlogged."""
    answers = {}
    for logged in (False, True):
        if logged:
            monkeypatch.setenv("MSTAR_STEP_TELEMETRY_DIR", str(tmp_path))
        kv = _manager(max_num_pages=16, fit="sum", order="fifo")
        kv.ingest_request("a", _request(list(range(100)), max_tokens=60))
        kv.ingest_request("long", _request(list(range(500, 600)), max_tokens=60))
        kv.ingest_request("short", _request(list(range(900, 910)), max_tokens=4))
        got = [_ready(kv, rid).ready for rid in ("a", "long", "short")]
        kv.remove_request("a")
        got += [_ready(kv, rid).ready for rid in ("long", "short")]
        answers[logged] = got

    assert answers[False] == answers[True] == [True, False, False, True, True]


# ── the invariant, with requests advancing at their own pace ────────────


def _drive(kv: KVManager, rng: random.Random, ops: int):
    """Requests arriving at random, admitted by readiness (some behind a waiting
    head), prefilled, then decoded at random rates, so some are far ahead of
    others, until their ``max_tokens``; and aborted now and then. Every step
    that is refused is refused as a deferral, never as a shortage of pages. Then
    arrivals stop and what was admitted must all finish: no deadlock.

    Returns what happened, as counts."""
    prompts = [list(range(base, base + rng.randrange(8, 50))) for base in (0, 1000, 2000)]
    waiting: dict[str, tuple[int, int]] = {}
    admitted: dict[str, list[int]] = {}
    running: dict[str, int] = {}
    seen = dict(finished=0, aborted=0, backfilled=0, deferred=0)
    log: list[str] = []

    def step(rid: str, span: int, walk: str):
        outcome = _prefill(kv, rid, span) if walk == PREFILL else _run(kv, {rid: span}, walk)
        if not outcome.ok:
            assert isinstance(outcome.reason, GrantDeferred), (
                f"{rid}: {type(outcome.reason).__name__}, not a deferral: {log}"
            )
            seen["deferred"] += 1
        return outcome

    def advance(rid: str, limit: int | None = None):
        if rid in admitted:
            if step(rid, *admitted[rid][:1], PREFILL).ok:
                running[rid] = admitted.pop(rid)[1]
                log.append(f"prefill {rid}")
            return
        span = min(running[rid], rng.randrange(1, limit or PAGE_SIZE))
        if step(rid, span, DECODE).ok:
            running[rid] -= span
            log.append(f"decode {rid} {span}")
            if not running[rid]:
                del running[rid]
                kv.remove_request(rid)
                seen["finished"] += 1
                log.append(f"finish {rid}")

    def admit(rid: str):
        head = next(iter(kv._waiting), None)
        if _ready(kv, rid).ready:
            prompt, max_tokens = waiting.pop(rid)
            admitted[rid] = [prompt, max_tokens]
            seen["backfilled"] += head not in (None, rid)
            log.append(f"admit {rid}")

    def invariants():
        kv.assert_pages_conserved()
        assert kv._admitted_are_safe(), f"admitted requests cannot all finish: {log}"
        assert kv._arena.num_free >= 0

    for i in range(ops):
        roll = rng.random()
        if roll < 0.22 and len(waiting) + len(admitted) + len(running) < 10:
            rid = f"r{i}"
            prompt = rng.choice(prompts) + [7] * rng.randrange(0, 20)
            max_tokens = rng.randrange(1, 8 * PAGE_SIZE)
            kv.ingest_request(rid, _request(prompt, max_tokens))
            waiting[rid] = (len(prompt), max_tokens)
            log.append(f"ingest {rid} prompt={len(prompt)} max_tokens={max_tokens}")
        elif roll < 0.42 and waiting:
            admit(rng.choice(list(waiting)))
        elif roll < 0.985 and (admitted or running):
            advance(rng.choice(list(admitted) + list(running)))
        elif roll >= 0.985 and (waiting or admitted or running):
            rid = rng.choice(list(waiting) + list(admitted) + list(running))
            for group in (waiting, admitted, running):
                group.pop(rid, None)
            kv.remove_request(rid)
            seen["aborted"] += 1
            log.append(f"abort {rid}")
        invariants()

    # nothing arrives now: every request admitted, or waiting for room, must finish
    for round_ in range(5000):
        if not (waiting or admitted or running):
            break
        progressed = len(log)
        for rid in list(waiting):
            admit(rid)
        for rid in list(admitted) + list(running):
            advance(rid, limit=2 * PAGE_SIZE)
        invariants()
        assert len(log) > progressed, f"nothing could advance (a deadlock): {log[-30:]}"
    assert not (waiting or admitted or running), f"requests never finished: {log[-30:]}"
    return seen


@pytest.mark.parametrize(("fit", "order"), [("peak", "backfill"), ("peak", "fifo"), ("sum", "backfill")])
def test_no_deadlock_when_requests_do_not_advance_together(monkeypatch, fit, order):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(max_num_pages=16, fit=fit, order=order)

    seen = _drive(kv, random.Random(SEED), ops=2500)

    assert seen["finished"] > 60, f"seed {SEED}: too few requests ran to the end: {seen}"
    if order == "backfill":
        assert seen["backfilled"] > 5, f"seed {SEED}: nothing was ever backfilled: {seen}"
    if fit == "peak":
        assert seen["deferred"] > 0, f"seed {SEED}: the guard never fired, so nothing was shown: {seen}"
    _finish(kv)


@pytest.mark.parametrize("seed", range(1, 6))
def test_peak_and_backfill_run_to_the_end_on_other_seeds(monkeypatch, seed):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(max_num_pages=random.Random(seed).choice([14, 20, 30]), fit="peak", order="backfill")

    _drive(kv, random.Random(SEED + seed), ops=1200)

    _finish(kv)


# ── TP ──────────────────────────────────────────────────────────────────


def test_every_rank_refuses_the_same_grants():
    """Rank 0 admits at readiness; rank 1 learns of a request at its first step. Their
    pools answer every step alike, which is what keeps the ranks in step. Not run on real TP."""
    leader = _manager(max_num_pages=21, fit="peak", order="fifo", rank=0, world_size=2)
    follower = _manager(max_num_pages=21, fit="peak", order="fifo", rank=1, world_size=2)
    for kv in (leader, follower):
        for rid, prompt, max_tokens in (
            ("c", 4 * PAGE_SIZE, 2 * PAGE_SIZE),
            ("a", PAGE_SIZE, 9 * PAGE_SIZE),
            ("b", PAGE_SIZE, 9 * PAGE_SIZE),
        ):
            kv.ingest_request(rid, _request(list(range(hash(rid) % 997, hash(rid) % 997 + prompt)), max_tokens))
    for rid in ("c", "a", "b"):
        assert _ready(leader, rid).ready

    def advance(kv, rid, span, walk):
        step = KVStep(segments=(Segment(rid, "main", span),))
        ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)
        outcome = kv.admit(step, ctx)
        if outcome.ok:
            kv.commit(step, ctx)
        return None if outcome.ok else type(outcome.reason).__name__

    script = [("c", 4 * PAGE_SIZE, PREFILL), ("a", PAGE_SIZE, PREFILL), ("b", PAGE_SIZE, PREFILL)]
    script += [(rid, PAGE_SIZE, DECODE) for _ in range(12) for rid in ("a", "b")]
    verdicts = [
        (advance(leader, *step), advance(follower, *step)) for step in script
    ]

    assert all(a == b for a, b in verdicts), verdicts
    assert any(a is not None for a, _ in verdicts), "the guard never fired on either rank"
