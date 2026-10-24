"""Prefix reuse over a recurrent pool: the state a prompt reached at a page boundary, restored
for a later prompt that shares the pages before it.

A recurrent layer can't skip tokens the way attention can: its state after token n is all it
keeps of tokens 0..n. So a KV hit serves a recurrent layer only where a snapshot of that state
exists, keyed by the KV chain's key at the boundary. A hit the pool can't restore would skip
tokens the recurrent layers never saw, and the request would decode from the wrong state.

CPU-only: nothing here launches a kernel.
"""

from __future__ import annotations

import logging

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources.base import CachedPrefix, Resource
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnSpec, LinearAttnVariant
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import Segment, StepContext
from mstar.graph.base import GraphNode
from mstar.model.base import KVLabel, PrefixKeys
from mstar.worker.engine_manager import _refuse_unskippable_resources

NODE, WALK, PAGE = "llm", "prefill", 4


class _StubChains:
    """Keys each request's whole pages of ids, as the KV cache's chains do."""

    def __init__(self):
        self.config = type("Config", (), {"page_size": PAGE})()
        self.keys: dict[str, list[bytes]] = {}

    def add(self, rid: str, tokens: list[int]) -> None:
        whole = len(tokens) // PAGE * PAGE
        self.keys[rid] = chain([tokens[at:at + PAGE] for at in range(0, whole, PAGE)])

    def keyed_label(self, rid, node_name, graph_walk):
        return "main" if rid in self.keys else None

    def chain_key(self, rid, label, pages):
        keys = self.keys.get(rid, [])
        return keys[pages - 1] if 0 < pages <= len(keys) else None

    def chain_keys(self, rid, label, pages):
        return self.keys.get(rid, [])[:pages]


def _pool(prefix_slots: int = 2, world_size: int = 1) -> RecurrentStatePool:
    config = RecurrentStateConfig(
        num_layers=2, blocks={"state": RecurrentBlockConfig(shape=(3,), dtype=torch.float32)},
        max_slots=4, prefix_slots=prefix_slots, kv_cache="kv",
    )
    pool = RecurrentStatePool(device=torch.device("cpu"), config=config, world_size=world_size)
    pool.bind_kv(_StubChains())
    pool.enable_prefix_cache(b"a root")
    return pool


def _ctx(rid: str, preplan: bool = False) -> StepContext:
    return StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False, is_preplan=preplan)


def _step(rid: str, span: int) -> RecurrentStep:
    return RecurrentStep(segments=(Segment(rid, "main", span),))


def _snapshot(pool: RecurrentStatePool, rid: str, tokens: list[int], at: int, value: float) -> int:
    """File a snapshot of ``value`` under the chain key of ``tokens`` at boundary ``at``."""
    pool._kv.add(rid, tokens)
    snap = pool._reserve_snapshot()
    pool._blocks["state"][:, snap].fill_(value)
    assert pool._index(snap, pool._kv.chain_key(rid, "main", at // PAGE))
    return snap


def _probe(pool: RecurrentStatePool, rid: str, tokens: list[int], hit: int) -> CachedPrefix:
    """Probe with a KV cache that holds ``hit`` tokens, and return the pool's cap of that answer."""
    pool.ingest_request(rid)
    pool._kv.add(rid, tokens)
    assert pool.resolve_cached_prefix(rid, NODE, WALK, None) is None
    prefix = pool.cap_cached_prefix(rid, NODE, WALK, CachedPrefix("main", hit, hit + 100))
    pool.apply_cached_prefix(rid, NODE, WALK, None, prefix)
    return prefix


def _state(pool: RecurrentStatePool, rid: str) -> torch.Tensor:
    return pool._blocks["state"][:, pool._slots[rid]["main"].index]


A = list(range(20))


def test_a_later_prompt_restores_the_snapshot_at_its_kv_hit():
    pool = _pool()
    _snapshot(pool, "a", A, 8, 1.0)
    _snapshot(pool, "a", A, 16, 2.0)

    assert _probe(pool, "b", A[:16] + [99, 98, 97], hit=16) == CachedPrefix("main", 16, 116)
    step, ctx = _step("b", 3), _ctx("b")
    assert pool.admit(step, ctx).ok
    pool.plan(step, ctx)
    assert pool._slots["b"]["main"].has_state
    assert bool((_state(pool, "b") == 2.0).all()), "the restore copied another state"
    pool.commit(step, ctx)
    assert not pool._snap_refs, "a restored snapshot is still held, so it can never be evicted"


def test_a_stream_that_holds_state_restores_nothing():
    pool = _pool()
    _snapshot(pool, "a", A, 8, 1.0)
    pool.ingest_request("b")
    step, ctx = _step("b", 4), _ctx("b")
    assert pool.admit(step, ctx).ok
    pool.commit(step, ctx)
    pool._kv.add("b", A)
    prefix = pool.cap_cached_prefix("b", NODE, WALK, CachedPrefix("main", 8, 8))
    assert prefix.tokens == 0, "a restore would overwrite the state the stream already built"


def test_apply_raises_for_a_length_the_pool_does_not_hold():
    pool = _pool()
    _snapshot(pool, "a", A, 8, 1.0)
    _probe(pool, "b", A[:12], hit=8)
    with pytest.raises(RuntimeError, match="holds a snapshot of 8 tokens"):
        pool.apply_cached_prefix("b", NODE, WALK, None, CachedPrefix("main", 4, 4))


def test_a_full_set_evicts_the_oldest_snapshot_nothing_holds():
    pool = _pool(prefix_slots=2)
    held = _snapshot(pool, "a", A, 8, 1.0)
    _snapshot(pool, "c", [5] * 20, 8, 2.0)
    # b leases a@8, the oldest, so the next save takes c@8
    assert _probe(pool, "b", A[:12], hit=8).tokens == 8
    assert pool._reserve_snapshot() != held
    assert pool._snap_meta[held] in pool._snap_by_key
    assert len(pool._snap_by_key) == 1



def test_snapshots_stay_off_above_one_rank_and_without_slots():
    assert not _pool(world_size=2).enable_prefix_cache(b"a root")
    assert not _pool(prefix_slots=0).enable_prefix_cache(b"a root")
    assert _pool().enable_prefix_cache(b"a root")


@pytest.mark.parametrize("admitted", [False, True])
def test_removal_gives_back_the_lease_or_the_restore_its_plan_never_ran(admitted):
    pool = _pool()
    _snapshot(pool, "a", A, 8, 1.0)
    _probe(pool, "b", A[:12], hit=8)
    if admitted:
        assert pool.admit(_step("b", 4), _ctx("b")).ok
    pool.remove_request("b")
    held = pool._leases or pool._restores or pool._admit_restores or pool._snap_refs or pool._keyed
    assert not held, "the snapshot is held for good"


def test_a_refused_admit_turns_the_restore_back_into_a_lease():
    pool = _pool()
    _snapshot(pool, "a", A, 8, 1.0)
    _probe(pool, "b", A[:12], hit=8)
    step, ctx = _step("b", 4), _ctx("b")
    assert pool.admit(step, ctx).ok
    pool.rollback_admit(step, ctx)
    assert pool._leases["b"].tokens == 8 and not pool._restores
    assert pool.cap_cached_prefix("b", NODE, WALK, CachedPrefix("main", 8, 8)).tokens == 8
    assert pool._snap_refs == {pool._leases["b"].slot: 1}, "the re-probe took a second reference"


def test_dropping_a_pre_plan_gives_back_only_what_its_pass_took():
    pool = _pool()
    _snapshot(pool, "a", A, 8, 1.0)
    for rid in ("b", "c"):
        _probe(pool, rid, A[:12], hit=8)
    assert pool.admit(_step("b", 4), _ctx("b")).ok
    staged, ctx = _step("c", 4), _ctx("c", preplan=True)
    assert pool.admit(staged, ctx).ok
    pool.plan(staged, ctx)
    pool.clear_preplan()
    assert set(pool._restores) == {("b", "main")}, "the live step lost its restore"
    assert pool._leases["c"].tokens == 8 and "b" not in pool._leases


def test_a_pre_planned_restore_lands_when_its_plan_is_promoted():
    """Staging must leave the slot alone, since the step in flight may still read it, but the
    staged step's addressing already reads the slot as holding state."""
    pool = _pool()
    _snapshot(pool, "a", A, 8, 2.0)
    _probe(pool, "b", A[:12], hit=8)
    step, ahead = _step("b", 4), _ctx("b", preplan=True)
    assert pool.admit(step, ahead).ok
    assert bool(pool.plan(step, ahead)["main"].has_state[0]), "the staged step would read zeros"
    slot = pool._slots["b"]["main"]
    assert not slot.has_state
    pool.plan(step, _ctx("b"))
    assert slot.has_state and slot.tokens == 8
    assert bool((_state(pool, "b") == 2.0).all()), "promotion skipped the restore"


class _StubHit(Resource):
    """A KV cache that holds the first ``length`` tokens of every prompt."""

    def __init__(self, length: int | None):
        self.length = length

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def resolve_cached_prefix(self, rid, node_name, graph_walk, inputs):
        return None if self.length is None else CachedPrefix("main", self.length, self.length)


def _runner(pool: RecurrentStatePool, hit: int | None) -> StepRunner:
    resources = {"kv": _StubHit(hit), "state": pool}
    return StepRunner(resources, node_resources={NODE: list(resources)})


@pytest.mark.parametrize("hit, agreed", [(19, 16), (15, 8), (7, 0)])
def test_the_agreed_prefix_is_the_longest_snapshot_under_the_kv_hit(hit, agreed):
    pool = _pool()
    _snapshot(pool, "a", A, 8, 1.0)
    _snapshot(pool, "a", A, 16, 2.0)
    pool.ingest_request("b")
    pool._kv.add("b", A)
    prefix = _runner(pool, hit).resolve_cached_prefix("b", NODE, WALK, None)
    assert prefix == CachedPrefix("main", agreed, agreed), "a hit past every snapshot skips unrestored tokens"


def test_a_node_without_a_kv_answer_restores_nothing():
    pool = _pool()
    _snapshot(pool, "a", A, 8, 1.0)
    pool.ingest_request("b")
    pool._kv.add("b", A)
    assert _runner(pool, None).resolve_cached_prefix("b", NODE, WALK, None) is None
    assert "b" not in pool._leases


class _StubTransfer:
    """No transfer engine on CPU."""

    def __init__(self, *args, **kwargs):
        pass

    def remove_request(self, request_id):
        pass


class _Keyed:
    """Keys ``main`` on ``kv`` from the prompt walk, decode extending it."""

    def checkpoint_path(self):
        return None

    def preprocess_fingerprint(self):
        return "_Keyed"

    def kv_labels(self):
        return {"kv": {"main": KVLabel(
            growth=(("decode", "decode_loop"),), keys=PrefixKeys("text_inputs", (WALK,), growth=True),
        )}}

    def get_graph_walk_graphs(self):
        return {WALK: GraphNode(name="LLM", input_names=["text_inputs"], outputs=[])}


class _StubGDN(Resource):
    """A linear-attention resource over the pool, which can't skip a prefix itself."""

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError


def _specs(prefix_slots: int) -> dict[str, NodeResourceSpec]:
    state = RecurrentStateConfig(
        num_layers=1, blocks={"state": RecurrentBlockConfig(shape=(3,), dtype=torch.float32)},
        max_slots=4, prefix_slots=prefix_slots, kv_cache="kv",
    )
    return {
        "kv": KVSpec(resource_key="kv", nodes={"LLM"}, config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, max_num_pages=8,
            page_size=PAGE, prefix_cache_salt="a salt",
        )),
        "state": RecurrentStateSpec(resource_key="state", nodes={"LLM"}, config=state),
        "gdn": LinearAttnSpec(resource_key="gdn", nodes={"LLM"}, config=LinearAttnConfig(
            recurrent_state="state", variant=LinearAttnVariant.GDN,
        )),
    }


def _open(monkeypatch, prefix_slots: int = 2, world_size: int = 1) -> Engine:
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    specs = _specs(prefix_slots)
    engine = Engine.__new__(Engine)
    engine._resources = {
        "kv": KVManager(cfg=specs["kv"].config, name="kv", joint_comm_group=None,
                        transfer_engine_info=None, device=torch.device("cpu")),
        "state": RecurrentStatePool(torch.device("cpu"), specs["state"].config, world_size=world_size),
        "gdn": _StubGDN(),
    }
    engine._open_prefix_caches(specs, _Keyed())
    return engine


def test_a_node_whose_pool_keeps_snapshots_opens_its_caches(monkeypatch):
    engine = _open(monkeypatch)
    assert engine._resources["kv"]._index is not None
    assert engine._resources["state"]._kv is engine._resources["kv"]
    assert WALK in engine._keyed_walks["LLM"]


@pytest.mark.parametrize("prefix_slots, world_size", [(0, 1), (2, 2)])
def test_a_node_whose_pool_keeps_no_snapshots_opens_no_cache(monkeypatch, caplog, prefix_slots, world_size):
    with caplog.at_level(logging.INFO):
        engine = _open(monkeypatch, prefix_slots, world_size)
    assert engine._resources["kv"]._index is None, "a KV hit would skip tokens the GDN state never saw"
    assert "LLM" not in engine._keyed_walks
    assert "state on LLM" in caplog.text


def test_a_keyed_node_loads_with_a_recurrent_pool_and_the_resource_over_it():
    specs = _specs(prefix_slots=0)
    _refuse_unskippable_resources(list(specs.values()), _Keyed())


@pytest.mark.parametrize("depends_on", [set(), {"kv"}])
def test_a_keyed_node_still_refuses_a_resource_that_cant_skip_a_prefix(depends_on):
    class _Unskippable(NodeResourceSpec):
        @property
        def resource_class(self):
            return _StubGDN

        def depends_on(self):
            return depends_on

    specs = [*_specs(prefix_slots=2).values(), _Unskippable(resource_key="other", nodes={"LLM"})]
    with pytest.raises(ValueError, match="'other' cannot serve a request"):
        _refuse_unskippable_resources(specs, _Keyed())
