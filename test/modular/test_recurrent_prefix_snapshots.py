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
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources.base import CachedPrefix, EngineResourceInfo, Resource
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVSpec, PagedKVConfig
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnSpec, LinearAttnVariant
from mstar.engine.resources.linear_attn.gdn import GDNManager
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentAddressing, RecurrentStatePool, SnapshotTarget
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.spec import CombinedSplit, NodeResourceSpec
from mstar.engine.resources.step import Segment, StepContext
from mstar.graph.base import GraphNode
from mstar.model.base import KVLabel, PrefixKeys
from mstar.validation.resources import _unskippable_resources

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


def _build(config: RecurrentStateConfig, world_size: int = 1, prefix_cache: bool = True) -> RecurrentStatePool:
    """The pool as the engine builds it, over a KV cache whose `prefix_cache` is ``prefix_cache``."""
    kv = _specs(0)["kv"]
    kv.config.prefix_cache = prefix_cache
    group = type("Group", (), {"world_size": world_size})() if world_size > 1 else None
    return RecurrentStatePool.build(
        RecurrentStateSpec(resource_key="state", nodes={NODE}, config=config),
        EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=group, dependencies={"kv": kv}),
    )


def _pool(prefix_slots: int = 2, world_size: int = 1, prefix_cache: bool = True) -> RecurrentStatePool:
    config = RecurrentStateConfig(
        num_layers=2, blocks={"state": RecurrentBlockConfig(shape=(3,), dtype=torch.float32)},
        max_slots=4, prefix_slots=prefix_slots, kv_cache="kv",
    )
    pool = _build(config, world_size, prefix_cache)
    pool.bind_kv(_StubChains())
    pool.enable_prefix_cache(b"a root")
    return pool


def _ctx(rid: str) -> StepContext:
    return StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False)


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



@pytest.mark.parametrize("closed", [{"world_size": 2}, {"prefix_cache": False}, {"prefix_slots": 0}])
def test_snapshots_stay_off_and_unallocated_where_the_cache_cant_open(closed):
    pool = _pool(**closed)
    assert not pool.enable_prefix_cache(b"a root")
    assert pool.block("state", 0).shape[0] == pool.config.max_slots, "allocated snapshot slots nothing can use"
    assert _pool().enable_prefix_cache(b"a root")


@pytest.mark.parametrize("override, refusal", [
    ({"prefix_slots": -1}, "prefix_slots=-1"),
    # the sink takes one slot, so no request would ever be admitted
    ({"max_slots": 1}, "max_slots=1"),
])
def test_a_slot_override_that_cannot_work_is_refused(override, refusal):
    with pytest.raises(ValueError, match=refusal):
        _specs(0)["state"].apply_yaml_overrides(**override)


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
        "state": _build(specs["state"].config, world_size),
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
    assert not list(_unskippable_resources(list(specs.values()), _Keyed())), (
        "a recurrent pool and the resource over it were refused, though the pool restores from snapshots"
    )


@pytest.mark.parametrize("variant", [LinearAttnVariant.KDA, LinearAttnVariant.MAMBA2])
def test_a_keyed_node_refuses_a_pool_whose_backend_writes_no_snapshots(variant):
    specs = _specs(prefix_slots=2)
    specs["gdn"] = LinearAttnSpec(resource_key="gdn", nodes={"LLM"}, config=LinearAttnConfig(
        recurrent_state="state", variant=variant,
    ))
    problems = list(_unskippable_resources(list(specs.values()), _Keyed()))
    assert problems and "'state' cannot serve a request" in problems[0], problems


@pytest.mark.parametrize("depends_on", [set(), {"kv"}])
def test_a_keyed_node_still_refuses_a_resource_that_cant_skip_a_prefix(depends_on):
    class _Unskippable(NodeResourceSpec):
        @property
        def resource_class(self):
            return _StubGDN

        def depends_on(self):
            return depends_on

    specs = [*_specs(prefix_slots=2).values(), _Unskippable(resource_key="other", nodes={"LLM"})]
    problems = list(_unskippable_resources(specs, _Keyed()))
    assert problems and "'other' cannot serve a request" in problems[0], problems


def _prefill(pool: RecurrentStatePool, rid: str, tokens: list[int], hit: int | None) -> RecurrentStep:
    """Probe and admit a prompt's prefill: a KV cache holding ``hit`` tokens, or none."""
    if hit is None:
        pool.ingest_request(rid)
        pool._kv.add(rid, tokens)
        pool.resolve_cached_prefix(rid, NODE, WALK, None)
        restored = 0
    else:
        restored = _probe(pool, rid, tokens, hit).tokens
    step = _step(rid, len(tokens) - restored)
    assert pool.admit(step, _ctx(rid)).ok
    return step


def _kept(pool: RecurrentStatePool) -> dict[int, int]:
    """Boundary -> offset into the row, for every snapshot slot admit reserved."""
    return {key[2]: write.offset for key, write in pool._snap_writes.items()}


@pytest.mark.parametrize("length, kept", [(19, {16: 16}), (16, {12: 12}), (8, {4: 4}), (4, {})])
def test_a_prefill_from_token_0_keeps_its_last_boundary(length, kept):
    """A page-aligned prompt keeps the page before its end: a resend's KV hit stops there."""
    pool = _pool()
    _prefill(pool, "a", A[:length], hit=None)
    assert _kept(pool) == kept


def test_a_prefill_from_a_restored_boundary_keeps_its_last():
    pool = _pool(prefix_slots=3)
    _snapshot(pool, "a", A, 8, 1.0)
    _prefill(pool, "b", A[:19], hit=8)
    assert _kept(pool) == {16: 8}


@pytest.mark.parametrize("hit", [12, 14])
def test_a_prefill_whose_kv_hit_runs_past_its_snapshot_keeps_the_hit_end_too(hit):
    """Without the KV hit's end, a prefix that many prompts share is never restored, since
    each prompt's last boundary lies past the shared prefix. A hit can end inside a page, at
    an image's end, where no page's key names the state, so the prefill keeps the page boundary
    under the hit's end."""
    pool = _pool(prefix_slots=3)
    _snapshot(pool, "a", A, 4, 1.0)
    _prefill(pool, "b", A[:19], hit=hit)
    assert _kept(pool) == {12: 8, 16: 12}


def test_a_prefill_past_the_largest_bucket_keeps_its_boundaries_on_a_wider_checkpoint_interval():
    """An eager step writes the checkpoint buffer the captured steps share, and one checkpoint per
    page would outgrow it (512 MiB at 32K tokens on Qwen3.5-9B)."""
    pool = _pool(prefix_slots=3)
    pool.build_accelerator_graph_buffers([], max_bs=1, max_seq_len=8)
    _snapshot(pool, "a", list(range(30)), 4, 1.0)
    step = _prefill(pool, "b", list(range(30)), hit=26)
    pool.plan(step, _ctx("b"))
    assert pool.addressing().snapshot_every == 8, "the step checkpoints more often than the shared buffer holds"
    assert _kept(pool) == {20: 16, 28: 24}


def test_a_prefill_keeps_no_boundary_a_snapshot_already_holds():
    pool = _pool(prefix_slots=3)
    _snapshot(pool, "a", A, 16, 1.0)
    free = len(pool._snap_free)
    _prefill(pool, "b", A[:19], hit=None)
    assert not pool._snap_writes and len(pool._snap_free) == free, "a duplicate evicted a snapshot"


def test_a_prefill_hands_its_kept_states_to_the_backend_and_commit_files_them():
    pool = _pool()
    step = _prefill(pool, "a", A[:19], hit=None)
    pool.plan(step, _ctx("a"))
    slot = pool._snap_writes[("a", "main", 16)].slot
    assert pool.addressing().snapshots == (SnapshotTarget(0, 16, slot),)
    assert pool.addressing().snapshot_every == PAGE
    pool.commit(step, _ctx("a"))
    assert pool._snap_by_key[pool._kv.chain_key("a", "main", 4)] == slot
    assert pool._slots["a"]["main"].tokens == 19, "later boundaries would be placed from the wrong length"


@pytest.mark.parametrize("release", ["rollback", "removal"])
def test_a_prefill_that_never_commits_gives_its_snapshot_slots_back(release):
    """A forward that raises gets neither commit nor rollback, only the request's removal.
    Each path returns what admit reserved, or prefix reuse stops once every slot is gone."""
    pool = _pool()
    pool.ingest_request("a")
    pool._kv.add("a", A[:19])
    pool.resolve_cached_prefix("a", NODE, WALK, None)
    step, ctx = _step("a", 19), _ctx("a")
    assert pool.admit(step, ctx).ok
    if release == "rollback":
        pool.rollback_admit(step, ctx)
    else:
        pool.plan(step, ctx)
        pool.remove_request("a")
    assert not pool._snap_writes and len(pool._snap_free) == 2


def _decode(pool: RecurrentStatePool, rid: str, steps: int, value: float = 0.0) -> None:
    """Run ``steps`` one-token steps, each leaving ``value`` in the request's state."""
    for _ in range(steps):
        step, ctx = _step(rid, 1), _ctx(rid)
        assert pool.admit(step, ctx).ok
        pool.plan(step, ctx)
        _state(pool, rid).fill_(value)
        pool.commit(step, ctx)


def _reply(pool: RecurrentStatePool, rid: str = "a") -> None:
    """Prefill a 3-token prompt, then have the KV chain key the reply's ids up to token 16."""
    step = _prefill(pool, rid, A[:3], hit=None)
    pool.commit(step, _ctx(rid))
    pool._kv.add(rid, A[:16])


def test_a_decode_step_onto_a_page_boundary_saves_its_state():
    pool = _pool()
    _reply(pool)
    _decode(pool, "a", 1, value=3.0)
    decode = pool._decode[("a", "main")]
    assert decode.tokens == 4 and decode.key == pool._kv.chain_key("a", "main", 1)
    assert pool._snap_by_key[decode.key] == decode.slot
    assert bool((pool._blocks["state"][:, decode.slot] == 3.0).all()), "the snapshot is not the state at 4"


def test_a_decode_snapshot_is_filed_once_the_chain_keys_its_page():
    pool = _pool()
    _reply(pool)
    pool._kv.add("a", A[:3])  # the reply's ids haven't reached the chain
    _decode(pool, "a", 1)
    assert ("a", "main") in pool._unkeyed and not pool._snap_by_key
    pool._kv.add("a", A[:16])
    _decode(pool, "a", 1)
    assert not pool._unkeyed and pool._decode[("a", "main")].key in pool._snap_by_key


def test_a_decode_that_crosses_a_boundary_replaces_the_previous_decode_snapshot():
    pool = _pool()
    _reply(pool)
    _decode(pool, "a", 1)
    previous = pool._decode[("a", "main")]
    _decode(pool, "a", 4)
    assert previous.key not in pool._snap_by_key, "a reply keeps a snapshot per page and fills the pool"
    assert pool._decode[("a", "main")].tokens == 8
    assert len(pool._snap_free) == 1


def test_a_replaced_decode_snapshot_a_probe_holds_stays_filed():
    pool = _pool()
    _reply(pool)
    _decode(pool, "a", 1)
    previous = pool._decode[("a", "main")]
    assert _probe(pool, "b", A[:8], hit=4).tokens == 4
    _decode(pool, "a", 4)
    assert pool._snap_by_key[previous.key] == previous.slot, "a lease lost the snapshot it holds"


def test_removal_frees_an_unfiled_decode_snapshot_and_keeps_a_filed_one():
    pool = _pool()
    for rid in ("a", "b"):
        _reply(pool, rid)
    pool._kv.add("b", A[:3])
    _decode(pool, "a", 1)
    _decode(pool, "b", 1)
    filed = pool._decode[("a", "main")].key
    pool.remove_request("a")
    pool.remove_request("b")
    assert pool._snap_by_key.keys() == {filed}, "a finished reply left no snapshot for its next turn"
    assert not pool._decode and not pool._unkeyed and len(pool._snap_free) == 1


def test_a_decode_snapshot_the_chain_never_keys_leaves_the_one_it_replaces_filed():
    """The step speculated past a reply's last token can end a page, and its token never
    reaches the chain."""
    pool = _pool()
    _reply(pool)
    _decode(pool, "a", 1)
    previous = pool._decode[("a", "main")]
    pool._kv.add("a", A[:7])
    _decode(pool, "a", 4)
    pool.remove_request("a")
    assert pool._snap_by_key.get(previous.key) == previous.slot, "the reply's snapshot went with the wasted step"
    assert len(pool._snap_free) == 1


def test_a_restored_request_places_its_decode_boundaries_from_the_restored_length():
    pool = _pool(prefix_slots=3)
    _snapshot(pool, "a", A, 8, 1.0)
    step = _prefill(pool, "b", A[:10], hit=8)
    pool.plan(step, _ctx("b"))
    pool.commit(step, _ctx("b"))
    _decode(pool, "b", 2)
    decode = pool._decode[("b", "main")]
    assert decode.tokens == 12 and decode.key == pool._kv.chain_key("b", "main", 3), "keyed by a page it never reached"


def test_a_restore_makes_its_snapshot_the_newest_to_evict():
    pool = _pool(prefix_slots=3)
    restored = _snapshot(pool, "a", A, 8, 1.0)
    older = _snapshot(pool, "c", [5] * 20, 8, 2.0)
    _snapshot(pool, "d", [6] * 20, 8, 3.0)
    step = _prefill(pool, "b", A[:10], hit=8)
    pool.plan(step, _ctx("b"))
    pool.commit(step, _ctx("b"))
    assert pool._reserve_snapshot() == older, "eviction took a snapshot a request just restored"


def test_a_decode_never_evicts_the_snapshot_a_prompt_restored_in_the_same_step():
    pool = _pool(prefix_slots=2)
    _reply(pool, "r")
    restored = _snapshot(pool, "a", A, 8, 1.0)
    _snapshot(pool, "c", [5] * 20, 8, 2.0)
    _probe(pool, "b", A[:10], hit=8)
    step = RecurrentStep(segments=(Segment("b", "main", 2), Segment("r", "main", 1)))
    ctx = StepContext(request_ids=("b", "r"), graph_walk=WALK, slot=0, capture=False)
    assert pool.admit(step, ctx).ok
    pool.plan(step, ctx)
    pool.commit(step, ctx)
    assert pool._snap_meta.get(restored) == pool._kv.chain_key("a", "main", 2), "r's decode evicted b's restore"


class _StubKeyedWalks:
    """The pool's answer for the walks its KV cache keys."""

    def __init__(self, walks):
        self.walks = frozenset(walks)

    def keyed_walks(self, label):
        return self.walks


def _gdn() -> tuple[GDNManager, list[tuple]]:
    """A GDN backend over a pool whose KV cache keys `WALK`, recording each wrapper plan."""
    gdn = GDNManager(
        config=LinearAttnConfig(
            recurrent_state="state", variant=LinearAttnVariant.GDN, splits=(CombinedSplit("mixed", (WALK,)),),
        ),
        geometry=DeltaNetGeometry(num_k_heads=1, num_v_heads=1, head_k_dim=2, head_v_dim=2, conv_kernel_size=2),
        num_layers=1, state_dtype=torch.float32, has_sink=True, device=torch.device("cpu"),
    )
    gdn.bind_dependencies({"state": _StubKeyedWalks({WALK})})
    gdn.build_accelerator_graph_buffers([], max_bs=1, max_seq_len=8 * PAGE)
    plans: list[tuple] = []
    gdn._get_wrapper = lambda *args, **kwargs: SimpleNamespace(plan=lambda *plan: plans.append(plan))
    return gdn, plans


def _plan_gdn(gdn: GDNManager, walk: str, span: int, snapshots=(), captured: bool = False) -> None:
    addressing = RecurrentAddressing(
        slot_indices=torch.zeros(1, dtype=torch.int32), has_state=torch.zeros(1, dtype=torch.bool), num_rows=1,
        host_slot_indices=(0,), host_has_state=(False,), snapshots=snapshots, snapshot_every=PAGE,
    )
    lease = SimpleNamespace(bucket=SimpleNamespace(graph_walk=walk)) if captured else None
    ctx = SimpleNamespace(force_prefill=False, slot_lease=lease, graph_walk=walk)
    gdn._build_plan("main", [Segment("a", "main", span)], addressing, ctx)


def test_a_graph_for_a_walk_the_kv_cache_doesnt_key_writes_no_checkpoints():
    """Such a walk keeps no snapshots (Qwen3.5's `prefill_vision`), so checkpointing every row's
    state there writes a buffer nothing reads."""
    gdn, plans = _gdn()
    _plan_gdn(gdn, "vision", 2 * PAGE, captured=True)
    (_, _, _, _, every, checkpoints), = plans
    assert every == 0 and checkpoints is None, "the graph checkpoints states nothing restores"


@pytest.mark.parametrize("walk", [WALK, "mixed"])
def test_a_graph_for_a_keyed_walk_writes_checkpoints(walk):
    gdn, plans = _gdn()
    _plan_gdn(gdn, walk, 2 * PAGE, captured=True)
    (_, _, _, _, every, checkpoints), = plans
    assert every == PAGE and checkpoints is not None, "the graph can never save a snapshot of this walk"


def test_an_eager_prefill_writes_the_checkpoint_buffer_the_graphs_share():
    """A buffer of its own per GDN layer would allocate the step's checkpoints again on every layer."""
    gdn, plans = _gdn()
    _plan_gdn(gdn, WALK, 2 * PAGE, captured=True)
    _plan_gdn(gdn, WALK, 6 * PAGE, snapshots=(SnapshotTarget(0, 5 * PAGE, 1),))
    assert plans[1][5] is plans[0][5], "the eager step allocated checkpoints of its own"
