"""Prefix reuse over KDA: a prefill keeps its state at page boundaries in the pool's snapshot
slots, and a later prompt that shares the pages before one restores it and runs on from there.

The KV cache, the recurrent pool and the KDA resource run under a real runner, with the torch
kernel bundle doing one KDA layer's math over a fixed table of token projections. Above one
rank, rank 0 leases the snapshot as it sends the step and the other ranks adopt the same slot;
a GDN node (Qwen3.5) runs the same lease under chunked prefill, without the forward.

CPU-only.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.communication.wire import decode, encode
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.chunking import ChunkingManagement, ChunkRange
from mstar.engine.engine import Engine
from mstar.engine.resources.base import CachedSnapshot, EngineResourceInfo, KeyedWalks, build_resource
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, KVStep, PagedKVConfig, opening_prefix_caches
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
from mstar.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ChunkingPolicy, InputSeqLenInfo
from mstar.testing_utils.worker import make_scheduler
from mstar.utils.ipc_format import ScheduleTPNode, WorkerMessage, WorkerMessageType
from mstar.worker.batch_builder import fill_token_budget_in_start_order

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
    are its row of a fixed table, so a prompt's state depends on its tokens alone. ``gdn``: the
    pool as a GDN node binds it, and no forward (FlashInfer's kernels need a GPU)."""

    def __init__(self, rank: int = 0, world_size: int = 1, prefix_slots: int = 3,
                 speculative_tokens: int = 0, page: int = PAGE, gdn: bool = False):
        self.rank, self.gdn = rank, gdn
        tp = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(world_size)))
        group = JointGroups(tp_group=tp, sp_group=CommGroup.trivial(), instance=tp) if world_size > 1 else None
        kv_spec = KVSpec("kv", {NODE}, PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=1024, num_qo_heads=2,
            max_num_pages=40, page_size=page, prefix_cache_salt="a salt",
            prefix_cache_tp=world_size > 1,
        ))
        self.kv = KVManager(
            cfg=kv_spec.config, name="kv", joint_comm_group=group, transfer_engine_info=None, device=CPU,
            dtype=torch.float32,
        )
        self.kv.enable_prefix_cache(ROOT, {"main": KeyedWalks((WALK,))})
        geometry = DeltaNetGeometry(HEADS, HEADS, DIM, DIM, conv_kernel_size=4)
        pool_spec = RecurrentStateSpec("state", {NODE}, RecurrentStateConfig(
            num_layers=1, max_slots=5, prefix_slots=prefix_slots, kv_cache="kv",
            blocks=geometry.to_blocks(conv_dtype=torch.float32, speculative_tokens=speculative_tokens),
        ))
        kda_spec = LinearAttnSpec("kda", {NODE}, LinearAttnConfig(
            recurrent_state="state", variant=LinearAttnVariant.GDN if gdn else LinearAttnVariant.KDA,
        ))
        # as the engine decides it: above one rank the snapshots open with the KV cache's prefix_cache_tp
        opened = opening_prefix_caches([kv_spec, pool_spec, kda_spec], {NODE: world_size}, checkpoint_named=False)
        assert opened == ({"kv", "state"} if prefix_slots else set())
        self.pool = RecurrentStatePool.build(pool_spec, EngineResourceInfo(
            device=CPU, joint_comm_group=group, dependencies={"kv": kv_spec}, prefix_caches=opened,
        ))
        if gdn:
            self.pool.bind_dependencies({"kv": self.kv})
            self.runner = StepRunner({"kv": self.kv, "state": self.pool}, node_resources={NODE: ["kv", "state"]})
            return
        self.kda = build_resource(
            kda_spec, EngineResourceInfo(device=CPU, joint_comm_group=group, dependencies={"state": pool_spec}),
        )
        self.kda.bind_layer(lower_bound=-5.0, full_rank_output_gate=False)
        self.pool.bind_dependencies({"kv": self.kv})
        self.kda.bind_dependencies({"state": self.pool})
        assert (self.pool._kv is not None) == bool(prefix_slots)
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

    def declare(self, rows: dict[str, list[int]], after: dict[str, int] | None = None) -> SubmoduleStep:
        """A prefill step over ``rows``, with the context the engine binds to it."""
        segments = [Segment(rid, "main", len(tokens)) for rid, tokens in rows.items()]
        steps = {"kv": KVStep(), "state": RecurrentStep()} | ({} if self.gdn else {"kda": LinearAttnStep()})
        step = SubmoduleStep(steps=steps, segments=segments)
        step.set_ctx(StepContext(
            request_ids=tuple(rows), graph_walk=WALK, slot=0, capture=False, chunk_tokens_after=after or {},
        ))
        return step

    def run(self, rows: dict[str, list[int]], windows: list[int] | None = None,
            after: dict[str, int] | None = None) -> tuple[torch.Tensor, Callable[[], None]]:
        """One prefill step over ``rows`` (rid -> the tokens it runs), admitted, planned and run;
        its outputs and its commit. ``windows``: the packed offsets the forward cuts windows at.
        ``after``: the tokens a chunked row runs after this step, as the engine stamps them."""
        step = self.declare(rows, after)
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        if self.gdn:
            return None, lambda: self.runner.commit(step)
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


def test_a_window_cut_with_no_snapshot_in_it_runs_its_row_as_one_piece():
    """With no snapshot to place, a row a window boundary cuts runs in one launch, chunked from
    the cut, and still computes what the whole step does."""
    rows = {"a": _tokens(100, seed=3), "b": _tokens(90, seed=4)}
    whole, cut = _Node(prefix_slots=0), _Node(prefix_slots=0)
    for node in (whole, cut):
        for rid, tokens in rows.items():
            node.ingest(rid, tokens)
    heads = []
    run = cut.kda.run

    def recording(*args, **kwargs):
        heads.append(cut.kda.current_plan("main").head)
        return run(*args, **kwargs)

    cut.kda.run = recording
    ref = whole.step(rows)
    out = cut.step(rows, windows=[120])
    assert heads == [None, None], "a window split a row that keeps no snapshot"
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    for rid in rows:
        for x, y in zip(whole.state(whole.slot(rid)), cut.state(cut.slot(rid)), strict=True):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)


def test_a_page_off_the_kda_chunk_is_refused_at_load():
    """A snapshot sits on a chunk end, so a page of half a chunk would leave one inside a chunk."""
    with pytest.raises(ValueError, match="page_size to a multiple of 64"):
        _Node(page=PAGE // 2)


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
    engine._resources = dict(node.runner._resources)
    engine._runner = node.runner
    engine._submodules = {NODE: SimpleNamespace(submodule=_Submodule())}
    engine._keyed_walks = {NODE: {WALK}}
    engine._prefix_model = "_Model"
    engine._graph_runtime = SimpleNamespace(
        get_rid_string=lambda rid: f"wire-{rid}", get_rid_handle=lambda wire: wire.removeprefix("wire-"),
    )
    return engine


def _send(leader: _Node, rid: str, chunk: ChunkRange | None = None, engine: Engine | None = None) -> ScheduleTPNode:
    engine = engine or _engine(leader)
    batch = SimpleNamespace(
        node_name=NODE, request_ids=(rid,), chunk_ranges={rid: chunk} if chunk else {},
        step_context=StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False),
    )
    head = ScheduleTPNode(
        NODE, WALK, [f"wire-{rid}"], prefix_hits=engine.decide_cached_prefix(batch),
        page_trails=engine.page_trails(NODE),
    )
    # the followers read rank 0's leases off the wire
    return decode(encode(WorkerMessage(message_type=WorkerMessageType.SCHEDULE_TP, body=head))).body


def _follow(follower: _Node, head: ScheduleTPNode, caught_up: bool = True) -> bool:
    engine = _engine(follower)
    scheduler = make_scheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine), parallel_leader_nodes=set(),
    )
    scheduler.lockstep.enqueue_follow(head)
    return scheduler.lockstep.settle_delta(caught_up)


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
    assert head.prefix_hits["wire-b"]["state"] == CachedSnapshot(leader.snapshot("a", 2 * PAGE), 2 * PAGE, 2 * PAGE)
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
    assert head.prefix_hits["wire-b"]["state"].slot >= 0
    assert not _follow(follower, head, caught_up=False), "the follower took a snapshot it hasn't filed"
    commits[follower]()
    assert _follow(follower, head, caught_up=False)
    assert follower.pool._leases["b"] == leader.pool._leases["b"]


def test_a_follower_missing_the_snapshot_rank_0_leased_stops():
    leader, follower = _Node(rank=0, world_size=2), _Node(rank=1, world_size=2)
    leader.prefill("a", A)
    b = A[:2 * PAGE] + _tokens(7, seed=9)
    follower.ingest("b", b)
    lease = CachedSnapshot(leader.snapshot("a", 2 * PAGE), 2 * PAGE, 2 * PAGE)
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


# a chunk's token budget: Qwen3.5's 2048, scaled to the page
BUDGET = 3 * PAGE


class _ChunkedPrompt(_Submodule):
    """A chunkable prefill as the engine sees it: measured, prepared whole once, cut per chunk."""

    def __init__(self, tokens: list[int]):
        self.tokens = tokens

    def get_input_sequence_len(self, graph_walk, fwd_info, inputs):
        return InputSeqLenInfo(len(self.tokens))

    def prepare_inputs(self, **kwargs):
        return ARNodeInputs(input_ids=torch.tensor(self.tokens), input_seq_len=len(self.tokens))


def _chunking_engine(node: _Node, tokens: list[int]) -> Engine:
    engine = _engine(node)
    engine._submodules[NODE] = SimpleNamespace(
        submodule=_ChunkedPrompt(tokens), resources=dict(node.runner._resources),
    )
    engine._chunking = ChunkingManagement(engine._submodules)
    engine._chunking.set_policies({(NODE, WALK): ChunkingPolicy(walks=(WALK,), chunkable=True)})
    engine._chunk_alignment = {}
    engine._tp_leader_nodes = {NODE} if node.rank == 0 else set()
    return engine


def _prepare(engine: Engine, rid: str, chunk: ChunkRange) -> dict[str, int]:
    """Prepare one chunk of ``rid``'s prompt; the tokens it runs after it, as `Engine` stamps a
    chunked step's context."""
    batch = SimpleNamespace(
        node_name=NODE, per_request_info_wrapped={rid: None}, per_request_input_tensors={},
        final_stream_rids=set(), per_request_input_metadata={}, first_chunk_rids=set(),
    )
    part = engine._prepare_chunk(batch, rid, WALK, chunk)
    assert (part.chunk_start, part.input_seq_len) == (chunk.start, chunk.end - chunk.start)
    return {} if part.is_final_chunk else {rid: part.chunk_total - part.chunk_start - part.input_seq_len}


def _run_chunk(node: _Node, engine: Engine, rid: str, tokens: list[int], chunk: ChunkRange) -> Callable[[], None]:
    """Prepare and run one chunk of ``rid``'s prompt; its commit."""
    after = _prepare(engine, rid, chunk)
    _, commit = node.run({rid: tokens[chunk.start:chunk.end]}, after=after)
    return commit


def _chunks(engine: Engine, rid: str, total: int, budget: int = BUDGET) -> tuple[ChunkRange, ChunkRange]:
    """Rank 0's measure of ``rid`` cut into two chunks by the scheduler's cut, on the node's alignment."""
    info = engine.input_sequence_len(NODE, WALK, None, {}, rid=rid)
    align = engine.chunk_alignment(NODE)
    take, _ = fill_token_budget_in_start_order(
        [rid], {rid: info.seq_len}, {rid}, budget, engine.min_chunk_tokens(budget, align), {}, align,
    )
    end = info.cached_prefix + take[rid]
    return ChunkRange(info.cached_prefix, end, total, end), ChunkRange(end, total, total, total - end)


def _prefill_in_chunks(ranks: tuple[_Node, _Node], engines: dict, rid: str, tokens: list[int],
                       chunks: tuple[ChunkRange, ChunkRange]) -> None:
    """Both chunks on both ranks, in TP async's order: rank 0 sends the second while the first
    runs, and the follower takes it once its own first chunk has committed."""
    leader, follower = ranks
    first, second = chunks
    assert _follow(follower, _send(leader, rid, first, engines[leader]))
    commits = {node: _run_chunk(node, engines[node], rid, tokens, first) for node in ranks}
    head = _send(leader, rid, second, engines[leader])
    for node in ranks:
        commits[node]()
    assert _follow(follower, head)
    assert not head.prefix_hits, "the second chunk carries a lease again"
    for node in ranks:
        _run_chunk(node, engines[node], rid, tokens, second)()


def _agree(ranks: tuple[_Node, _Node], rid: str) -> None:
    leader, follower = ranks
    assert leader.slot(rid) == follower.slot(rid)
    assert leader.pool._snap_by_key == follower.pool._snap_by_key
    assert leader.kv._streams[rid]["main"].page_indices == follower.kv._streams[rid]["main"].page_indices
    assert not leader.pool._snap_refs and not follower.pool._snap_refs


@pytest.fixture
def _gdn_ranks(monkeypatch):
    monkeypatch.setattr(pool_mod, "_DEBUG_ASSERTS", True)
    return _Node(rank=0, world_size=2, gdn=True), _Node(rank=1, world_size=2, gdn=True)


def test_a_chunked_resend_starts_past_rank_0s_snapshot_on_every_rank(_gdn_ranks):
    """A GDN prompt longer than the budget, resent at TP2: rank 0 leases the snapshot as it
    measures the row, so its chunks start past it on every rank, and the lease goes out once."""
    ranks = leader, _ = _gdn_ranks
    for node in ranks:
        node.prefill("a", A)
    b = A[:2 * PAGE] + _tokens(350 - 2 * PAGE, seed=10)
    for node in ranks:
        node.ingest("b", b)
    engines = {node: _chunking_engine(node, b) for node in ranks}
    chunks = _chunks(engines[leader], "b", len(b))

    _prefill_in_chunks(ranks, engines, "b", b, chunks)

    assert chunks[0].start == 2 * PAGE, "b ran a's pages again"
    _agree(ranks, "b")


def test_a_later_chunk_takes_no_lease_after_its_first_declined_one(_gdn_ranks):
    """Measured with nothing cached, then a's pages landed while b waited: b's first chunk
    starts at 0 and declines the hit its lease finds, on every rank. Its second chunk, sent
    before the first commits, must not lease the hit again."""
    ranks = leader, _ = _gdn_ranks
    b = A[:2 * PAGE] + _tokens(350 - 2 * PAGE, seed=11)
    for node in ranks:
        node.ingest("b", b)
    engines = {node: _chunking_engine(node, b) for node in ranks}
    chunks = _chunks(engines[leader], "b", len(b))
    for node in ranks:
        node.prefill("a", A)

    _prefill_in_chunks(ranks, engines, "b", b, chunks)

    assert chunks[0].start == 0
    _agree(ranks, "b")


def _held(node: _Node) -> tuple:
    """What a rank's eviction reads: each page's owners, and the snapshots a lease holds."""
    arena = node.kv._arena
    return arena.owners_of(range(arena.max_num_pages)), dict(node.pool._snap_refs), set(node.pool._leases)


def test_each_chunks_end_is_kept_alike_on_every_rank_and_a_resend_restores_the_last(_gdn_ranks):
    """Above one rank the pool keeps snapshots, so it declares its page as the chunk alignment
    there too: rank 0 cuts whole pages, every rank keeps each chunk's end, and the later chunk's
    makes the earlier one's the first to evict. A resend is leased at the later chunk's end."""
    ranks = leader, follower = _gdn_ranks
    b = _tokens(5 * PAGE + 10, seed=12)
    for node in ranks:
        node.ingest("b", b)
    engines = {node: _chunking_engine(node, b) for node in ranks}
    chunks = _chunks(engines[leader], "b", len(b), budget=3 * PAGE + 20)
    assert chunks[0].end == 3 * PAGE, "rank 0 cut b off a page"

    _prefill_in_chunks(ranks, engines, "b", b, chunks)

    _agree(ranks, "b")
    for node in ranks:
        ends = node.snapshot("b", 3 * PAGE), node.snapshot("b", 5 * PAGE)
        assert [node.pool._snap_meta[snap][1] for snap in ends] == [True, False]
    # in eviction order
    assert list(leader.pool._snap_meta.items()) == list(follower.pool._snap_meta.items())

    c = b[:5 * PAGE] + _tokens(20, seed=13)
    for node in ranks:
        node.ingest("c", c)
    head = _send(leader, "c")
    assert head.prefix_hits["wire-c"]["state"] == CachedSnapshot(leader.snapshot("b", 5 * PAGE), 5 * PAGE, 5 * PAGE)
    assert _follow(follower, head)
    for node in ranks:
        assert node.probe("c", c) == 5 * PAGE
        node.step({"c": c[5 * PAGE:]})
    _agree(ranks, "c")


def test_rank_0_gives_back_what_it_leased_for_a_row_its_step_leaves_out(_gdn_ranks):
    """Rank 0 leases a chunkable row's prefix as it measures the row. Left out of the step, the
    row waits holding nothing the other ranks don't, and the step that takes it leases again."""
    ranks = leader, follower = _gdn_ranks
    for node in ranks:
        node.prefill("a", A)
    b = A[:2 * PAGE] + _tokens(350 - 2 * PAGE, seed=14)
    for node in ranks:
        node.ingest("b", b)
    engines = {node: _chunking_engine(node, b) for node in ranks}
    assert engines[leader].input_sequence_len(NODE, WALK, None, {}, rid="b").cached_prefix == 2 * PAGE
    assert _held(leader) != _held(follower)

    assert engines[leader].give_back_cached_prefix(NODE, "b", WALK)

    assert _held(leader) == _held(follower)
    assert not engines[follower].give_back_cached_prefix(NODE, "b", WALK), "a follower measured a row"
    chunks = _chunks(engines[leader], "b", len(b))
    assert chunks[0].start == 2 * PAGE
    _prefill_in_chunks(ranks, engines, "b", b, chunks)
    _agree(ranks, "b")


def test_a_rebuilds_probe_on_rank_0_holds_nothing_the_other_ranks_dont(_gdn_ranks):
    """A rebuild row is measured through `Engine.unserved_len`, and nothing gives back what that
    probe would lease, so rank 0 takes no lease there: the rebuild's step leases as it is sent."""
    ranks = leader, follower = _gdn_ranks
    for node in ranks:
        node.prefill("a", A)
    b = A[:2 * PAGE] + _tokens(350 - 2 * PAGE, seed=16)
    for node in ranks:
        node.ingest("b", b)
    engines = {node: _chunking_engine(node, b) for node in ranks}
    # as MicroScheduler._measure_unstarted measures a rebuild row that has run no chunk
    engines[leader].unserved_len(NODE, WALK, None, "b", len(b), len(b))
    assert _held(leader) == _held(follower)


def test_a_row_measured_past_pages_no_snapshot_ends_on_waits_holding_nothing(_gdn_ranks):
    """b shares a's first page, where no snapshot sits, so the pool cuts the hit to nothing: rank 0
    measures b from 0 and holds no page for it until its step is sent. That send leases the page
    on every rank, and each keeps the state at the hit's end for the next prompt."""
    ranks = leader, follower = _gdn_ranks
    for node in ranks:
        node.prefill("a", A)
    b = A[:PAGE] + _tokens(350 - PAGE, seed=15)
    for node in ranks:
        node.ingest("b", b)
    engines = {node: _chunking_engine(node, b) for node in ranks}
    chunks = _chunks(engines[leader], "b", len(b))
    assert chunks[0].start == 0
    assert _held(leader) == _held(follower), "rank 0 holds a's page while b waits"

    _prefill_in_chunks(ranks, engines, "b", b, chunks)

    _agree(ranks, "b")
    assert leader.snapshot("b", PAGE) == follower.snapshot("b", PAGE)


def test_a_refused_steps_lease_waits_with_its_row_on_every_rank(_gdn_ranks):
    """A step refused at admit keeps its hit on every rank: the KV pages are its stream's by
    then. Measured again for the retry and left out, the row gives nothing back on rank 0, and
    the step that takes it runs on from the restored snapshot on every rank."""
    ranks = leader, follower = _gdn_ranks
    for node in ranks:
        node.prefill("a", A)
        for i in range(node.pool.num_free_slots):
            node.prefill(f"x{i}", _tokens(9, seed=30 + i))
    b = A[:2 * PAGE] + _tokens(350 - 2 * PAGE, seed=16)
    for node in ranks:
        node.ingest("b", b)
    engines = {node: _chunking_engine(node, b) for node in ranks}
    chunks = _chunks(engines[leader], "b", len(b))
    assert _follow(follower, _send(leader, "b", chunks[0], engines[leader]))
    for node in ranks:
        # no slot is free, so the pool refuses the step once the KV cache has taken the hit;
        # the engine then drops the prepared input, and the retry probes again
        step = node.declare({"b": b[chunks[0].start:chunks[0].end]}, _prepare(engines[node], "b", chunks[0]))
        assert not node.runner.admit(step).ok
        engines[node]._chunking.release("b", NODE)
    held = _held(follower)
    assert _held(leader) == held

    assert engines[leader].input_sequence_len(NODE, WALK, None, {}, rid="b").cached_prefix == 2 * PAGE
    engines[leader].give_back_cached_prefix(NODE, "b", WALK)

    assert _held(leader) == held, "rank 0 gave back a hit the other ranks hold"
    for node in ranks:
        node.pool.remove_request("x0")
    assert _chunks(engines[leader], "b", len(b)) == chunks
    _prefill_in_chunks(ranks, engines, "b", b, chunks)
    _agree(ranks, "b")
    for node in ranks:
        assert node.kv.stored_len("b") == len(b), "b's cached pages are in its stream twice"
