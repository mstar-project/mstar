"""Prefix reuse over KDA: a prefill keeps its state at page boundaries in the pool's snapshot
slots, and a later prompt that shares the pages before one restores it and runs on from there.

The KV cache, the recurrent pool and the KDA resource run under a real runner, with the torch
kernel bundle doing one KDA layer's math over a fixed table of token projections. Above one
rank, rank 0 leases the snapshot as it sends the step and the other ranks adopt the same slot.

CPU-only.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.engine.engine import Engine
from mstar.engine.resources.base import EngineResourceInfo, KeyedWalks, build_resource
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.linear_attn.kda import KDAParams
from mstar.engine.resources.recurrent import pool as pool_mod
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.model.base import KVLabel, LabelLayout, PrefixKeys, Span
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker.micro_scheduler import MicroScheduler

NODE, WALK, ROOT = "LLM", "prefill", b"a root"
# a page is the KDA prefill's 64-token chunk, so snapshots sit on chunk ends
PAGE, HEADS, DIM, VOCAB = 64, 2, 16, 97
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _no_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda *args, **kwargs: None)


def _config(tokens: list[int]) -> KVReqConfig:
    """What the preprocess worker sends for a text prompt, keyed by its own code."""
    layout = LabelLayout(spans=(Span("ids", WALK, len(tokens), len(tokens), "text_inputs"),))
    worker = SimpleNamespace(
        _keyed_labels={"kv": {"main": KVLabel(keys=PrefixKeys("text_inputs", (WALK,)))}},
        _prefix_page_sizes={"kv": PAGE}, model=None,
        _page_layout=lambda *args: PreprocessWorkerThread._page_layout(None, *args),
    )
    rows = {"kv": {"main": layout.row()}}
    sent = PreprocessWorkerThread._prefix_keys(
        worker, {"text_inputs": [torch.tensor(tokens)]}, {"kv": {"main": layout}}, rows, None, {},
    )
    config = KVReqConfig()
    config.apply_conductor_config(kv_layout=rows["kv"], **{name: by["kv"] for name, by in sent.items()})
    return config


def _tokens(n: int, seed: int) -> list[int]:
    return [(seed * 7919 + i * 31) % VOCAB for i in range(n)]


class _Node:
    """One rank's KV cache, KDA pool and KDA resource under a real runner. A token's KDA inputs
    are its row of a fixed table, so a prompt's state depends on its tokens alone."""

    def __init__(self, rank: int = 0, world_size: int = 1, prefix_slots: int = 3,
                 speculative_tokens: int = 0):
        group = SimpleNamespace(rank=rank, world_size=world_size) if world_size > 1 else None
        self.kv = KVManager(
            cfg=PagedKVConfig(
                num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=1024, num_qo_heads=2,
                max_num_pages=40, page_size=PAGE, prefix_cache_salt="a salt",
                prefix_cache_tp=world_size > 1,
            ),
            name="kv", joint_comm_group=group, transfer_engine_info=None, device=CPU,
            dtype=torch.float32,
        )
        self.kv.enable_prefix_cache(ROOT, {"main": KeyedWalks((WALK,))})
        geometry = DeltaNetGeometry(HEADS, HEADS, DIM, DIM, conv_kernel_size=4)
        pool_spec = RecurrentStateSpec("state", {NODE}, RecurrentStateConfig(
            num_layers=1, max_slots=5, prefix_slots=prefix_slots, kv_cache="kv",
            blocks=geometry.to_blocks(conv_dtype=torch.float32, speculative_tokens=speculative_tokens),
        ))
        self.pool = RecurrentStatePool.build(pool_spec, EngineResourceInfo(
            device=CPU, joint_comm_group=group, dependencies={"kv": self.kv},
        ))
        self.kda = build_resource(
            LinearAttnSpec("kda", {NODE}, LinearAttnConfig(
                recurrent_state="state", variant=LinearAttnVariant.KDA, gate_lower_bound=-5.0,
            )),
            EngineResourceInfo(device=CPU, joint_comm_group=group, dependencies={"state": pool_spec}),
        )
        self.pool.bind_dependencies({"kv": self.kv})
        self.kda.bind_dependencies({"state": self.pool})
        assert self.pool.enable_prefix_cache(ROOT) == bool(prefix_slots)
        self.runner = StepRunner(
            {"kv": self.kv, "state": self.pool, "kda": self.kda}, node_resources={NODE: ["kv", "state", "kda"]},
        )
        heads = HEADS // world_size
        width = heads * DIM
        gen = torch.Generator().manual_seed(0)
        self.table = torch.randn(VOCAB, 4 * width + heads, generator=gen)
        self.params = KDAParams(
            conv_weight=torch.randn(3 * width, 4, generator=gen) * 0.5,
            A_log=torch.randn(heads, generator=gen) * 0.1, dt_bias=torch.randn(width, generator=gen) * 0.1,
            lower_bound=-5.0, num_heads=heads, head_dim=DIM, scale=DIM ** -0.5,
        )
        self.width, self.heads = width, heads

    def ingest(self, rid: str, tokens: list[int]) -> None:
        self.runner.ingest_request(rid, {"kv": _config(tokens)})

    def probe(self, rid: str, tokens: list[int]) -> int:
        """The engine's probe of a prefill: the tokens every resource agreed to skip."""
        inputs = ARNodeInputs(input_seq_len=len(tokens))
        prefix = self.runner.resolve_cached_prefix(rid, NODE, WALK, inputs)
        self.runner.apply_cached_prefix(rid, NODE, WALK, inputs, prefix)
        return prefix.tokens if prefix is not None else 0

    def run(self, rows: dict[str, list[int]], windows: list[int] | None = None,
            ) -> tuple[torch.Tensor, Callable[[], None]]:
        """One prefill step over ``rows`` (rid -> the tokens it runs), admitted, planned and run;
        its outputs and its commit. ``windows``: the packed offsets the forward cuts windows at."""
        segments = [Segment(rid, "main", len(tokens)) for rid, tokens in rows.items()]
        step = SubmoduleStep(
            steps={"kv": KVStep(), "state": RecurrentStep(), "kda": LinearAttnStep()}, segments=segments,
        )
        step.set_ctx(StepContext(request_ids=tuple(rows), graph_walk=WALK, slot=0, capture=False))
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        out = self._forward([t for tokens in rows.values() for t in tokens], [len(t) for t in rows.values()], windows)
        return out, lambda: self.runner.commit(step)

    def step(self, rows: dict[str, list[int]], windows: list[int] | None = None) -> torch.Tensor:
        out, commit = self.run(rows, windows)
        commit()
        return out

    def prefill(self, rid: str, tokens: list[int]) -> tuple[torch.Tensor, int]:
        """Ingest ``rid``, probe, and run the rest of its prompt: its outputs and the tokens skipped."""
        self.ingest(rid, tokens)
        held = self.probe(rid, tokens)
        return self.step({rid: tokens[held:]}), held

    def _forward(self, tokens: list[int], spans: list[int], windows: list[int] | None) -> torch.Tensor:
        x = self.table[torch.tensor(tokens)]
        qkv, g, beta = x.split([3 * self.width, self.width, self.heads], dim=-1)
        conv, state = self.pool.block("conv", 0), self.pool.block("state", 0)
        if not windows:
            return self.kda.run(qkv, g, beta, conv, state, self.params, label="main")
        cuts = [0, *windows, len(tokens)]
        starts = [sum(spans[:i]) for i in range(len(spans))]
        outs = []
        for a, b in zip(cuts, cuts[1:], strict=False):
            pieces = [(row, min(r0 + n, b) - max(r0, a))
                      for row, (r0, n) in enumerate(zip(starts, spans, strict=True)) if r0 < b and r0 + n > a]
            with self.kda.token_window(pieces, "main"):
                outs.append(self.kda.run(qkv[a:b], g[a:b], beta[a:b], conv, state, self.params, label="main"))
        return torch.cat(outs)

    def slot(self, rid: str) -> int:
        return self.pool._slots[rid]["main"].index

    def state(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pool.block("state", 0)[slot].clone(), self.pool.block("conv", 0)[slot].clone()

    def snapshot(self, rid: str, boundary: int) -> int:
        return self.pool._snap_by_key[self.kv.chain_key(rid, "main", boundary // PAGE)]


A = _tokens(2 * PAGE + 10, seed=1)


def test_a_prefill_keeps_the_state_a_prompt_cut_at_its_last_page_boundary_leaves():
    node, cut = _Node(), _Node()
    node.prefill("a", A)
    cut.prefill("cut", A[:2 * PAGE])
    for kept, cold in zip(node.state(node.snapshot("a", 2 * PAGE)), cut.state(cut.slot("cut")), strict=True):
        assert torch.equal(kept, cold), "the snapshot at the boundary isn't the state a prompt ending there leaves"


def test_a_prompt_sharing_the_pages_restores_the_snapshot_and_runs_on_as_a_cold_run():
    node, cold = _Node(), _Node(prefix_slots=0)
    node.prefill("a", A)
    b = A[:2 * PAGE] + _tokens(30, seed=2)
    out, held = node.prefill("b", b)
    assert held == 2 * PAGE, "b missed the two pages a cached"
    ref, cold_held = cold.prefill("b", b)
    assert cold_held == 0
    # within rounding: the cold run's conv runs over a longer span, so its kernels round otherwise
    torch.testing.assert_close(out, ref[held:], rtol=1e-5, atol=1e-5)
    for warm, base in zip(node.state(node.slot("b")), cold.state(cold.slot("b")), strict=True):
        torch.testing.assert_close(warm, base, rtol=1e-5, atol=1e-5)


def test_a_restore_computes_the_bits_a_cold_run_cut_at_its_boundary_does():
    """A resent prompt restores the snapshot its first send kept and runs the rest as a cold run
    of it does once cut there (a node that keeps that snapshot too): the restore itself rounds
    nothing. Different bits from an uncut run come only from how the tokens were launched."""
    node, cut = _Node(), _Node()
    b = A[:2 * PAGE] + _tokens(30, seed=3)
    node.prefill("first", b)
    out, held = node.prefill("again", b)
    assert held == 2 * PAGE
    ref, _ = cut.prefill("cold", b)
    assert torch.equal(out, ref[held:])
    for warm, base in zip(node.state(node.slot("again")), cut.state(cut.slot("cold")), strict=True):
        assert torch.equal(warm, base), "the restored request computes other bits than a cold run cut there"


def test_windows_that_cut_a_row_inside_a_chunk_compute_what_the_whole_step_does():
    """The second window starts 20 tokens into b, so its head runs alone up to b's chunk end,
    where b's snapshot lands."""
    rows = {"a": _tokens(100, seed=3), "b": _tokens(90, seed=4)}
    whole, cut = _Node(), _Node()
    for node in (whole, cut):
        for rid, tokens in rows.items():
            node.ingest(rid, tokens)
            assert node.probe(rid, tokens) == 0
    ref = whole.step(rows)
    out = cut.step(rows, windows=[120])
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    for rid in rows:
        for node_a, node_b in ((whole.state(whole.slot(rid)), cut.state(cut.slot(rid))),
                               (whole.state(whole.snapshot(rid, PAGE)), cut.state(cut.snapshot(rid, PAGE)))):
            for x, y in zip(node_a, node_b, strict=True):
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)


def test_a_restored_slot_starts_with_no_verify_prefix():
    """A restore copies every block of the snapshot slot, and the kernels write only the state's:
    a slot an earlier snapshot used must not hand its verify prefix to the next request."""
    node = _Node(speculative_tokens=3)
    first = node.pool.config.max_slots
    node.pool.block("spec_len")[first:] = 3
    node.pool.block("spec_prefix", 0)[first:] = 1.0
    node.prefill("a", A)
    node.prefill("b", A[:2 * PAGE] + _tokens(5, seed=5))
    slot = node.slot("b")
    assert int(node.pool.block("spec_len")[slot]) == 0, "b's first verify step would replay a stale prefix"
    assert not node.pool.block("spec_prefix", 0)[slot].any()


# ── above one rank ──────────────────────────────────────────────────────


class _Submodule:
    def split_inputs(self, graph_walk, fwd_info, inputs, start, end):
        return ARNodeSubmodule.split_inputs(self, graph_walk, fwd_info, inputs, start, end)


def _engine(node: _Node) -> Engine:
    engine = Engine.__new__(Engine)
    engine._resources = {"kv": node.kv, "state": node.pool, "kda": node.kda}
    engine._runner = node.runner
    engine._submodules = {NODE: SimpleNamespace(submodule=_Submodule())}
    engine._keyed_walks = {NODE: {WALK}}
    engine._prefix_model = "_Model"
    engine._graph_runtime = SimpleNamespace(
        get_rid_string=lambda rid: f"wire-{rid}", get_rid_handle=lambda wire: wire.removeprefix("wire-"),
    )
    return engine


def _send(leader: _Node, rid: str) -> ScheduleTPNode:
    engine = _engine(leader)
    batch = SimpleNamespace(
        node_name=NODE, request_ids=(rid,),
        step_context=StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False),
    )
    return ScheduleTPNode(
        NODE, WALK, [f"wire-{rid}"], prefix_hits=engine.decide_cached_prefix(batch),
        page_trails=engine.page_trails(NODE),
    )


def _follow(follower: _Node, head: ScheduleTPNode, caught_up: bool = True) -> bool:
    engine = _engine(follower)
    scheduler = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine), parallel_leader_nodes=set(),
    )
    scheduler.register_tp_follow(head)
    return scheduler.settle_tp_follow_delta(caught_up)


@pytest.fixture
def _ranks(monkeypatch):
    # each step carries rank 0's snapshot trail, which the follower checks
    monkeypatch.setattr(pool_mod, "_DEBUG_ASSERTS", True)
    return _Node(rank=0, world_size=2), _Node(rank=1, world_size=2)


def test_a_follower_restores_the_snapshot_rank_0_leased_and_never_looks_one_up(_ranks):
    leader, follower = _ranks
    for node in _ranks:
        node.prefill("a", A)
    b = A[:2 * PAGE] + _tokens(7, seed=6)
    for node in _ranks:
        node.ingest("b", b)
    inputs = ARNodeInputs(input_seq_len=len(b))
    assert follower.runner.resolve_cached_prefix("b", NODE, WALK, inputs).tokens == 0
    assert not follower.pool._leases, "the follower looked a snapshot up for itself"
    head = _send(leader, "b")
    assert head.prefix_hits["wire-b"]["state"] == [leader.snapshot("a", 2 * PAGE), 2 * PAGE, 2 * PAGE]
    assert _follow(follower, head)
    for node in _ranks:
        assert node.probe("b", b) == 2 * PAGE
        node.step({"b": b[2 * PAGE:]})
    assert leader.slot("b") == follower.slot("b")
    assert leader.pool._snap_by_key.keys() == follower.pool._snap_by_key.keys()
    assert not leader.pool._snap_refs and not follower.pool._snap_refs


def test_a_follower_waits_for_its_own_commit_to_file_the_snapshot(_ranks):
    leader, follower = _ranks
    commits = {}
    for node in _ranks:
        node.ingest("a", A)
        node.probe("a", A)
        _, commits[node] = node.run({"a": A})
    commits[leader]()
    b = A[:2 * PAGE] + _tokens(7, seed=7)
    for node in _ranks:
        node.ingest("b", b)
    head = _send(leader, "b")
    assert head.prefix_hits["wire-b"]["state"][0] >= 0
    assert not _follow(follower, head, caught_up=False), "the follower took a snapshot it hasn't filed"
    commits[follower]()
    assert _follow(follower, head, caught_up=False)
    assert follower.pool._leases["b"] == leader.pool._leases["b"]


def test_a_follower_missing_the_snapshot_rank_0_leased_stops():
    leader, follower = _Node(rank=0, world_size=2), _Node(rank=1, world_size=2)
    leader.prefill("a", A)
    b = A[:2 * PAGE] + _tokens(7, seed=9)
    follower.ingest("b", b)
    lease = [leader.snapshot("a", 2 * PAGE), 2 * PAGE, 2 * PAGE]
    assert not follower.pool.adopt_cached_prefix("b", NODE, WALK, lease, caught_up=False)
    with pytest.raises(RuntimeError, match="rank 0 leased snapshot slot"):
        follower.pool.adopt_cached_prefix("b", NODE, WALK, lease, caught_up=True)


def test_a_step_that_wont_run_gives_its_snapshot_back_on_every_rank(_ranks):
    leader, follower = _ranks
    for node in _ranks:
        node.prefill("a", A)
        node.ingest("b", A[:2 * PAGE] + _tokens(3, seed=8))
    head = _send(leader, "b")
    assert _follow(follower, head)
    for node in _ranks:
        assert node.pool._snap_refs
        _engine(node).release_cached_prefix(SimpleNamespace(
            node_name=NODE, request_ids=("b",),
            step_context=StepContext(request_ids=("b",), graph_walk=WALK, slot=0, capture=False),
        ))
        assert not node.pool._snap_refs and not node.pool._leases, "the snapshot stays held for good"
