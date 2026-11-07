"""The sampler on a node's resident rows: a resident step reads each row's
scalars off the planner's cells by slot, and its RNG offset and last token off
the sampler's own rows, so a request's draws don't depend on its slot, on being
displaced or offloaded, or on which steps ran resident.

On the CPU: ``gather`` and the captured sampler's ops run eagerly, as a replay
runs what its capture recorded, and the sampling kernel is replaced by one that
records what each row read and returns a token of its seed and offset.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.device_run import RunState  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.resident import TRASH_SLOT, ResidentPlanner, ResidentStream  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.engine.resources.sampler import resident as sampler_resident  # noqa: E402
from mstar.engine.resources.sampler import utils as sampler_utils  # noqa: E402
from mstar.engine.resources.sampler.config import SamplerStep, SamplingReqConfig  # noqa: E402
from mstar.engine.resources.sampler.resource import SamplerResource  # noqa: E402
from mstar.engine.resources.sampler.utils import (  # noqa: E402
    SamplingConfig,
    _rng_offset_stride,
    canonical_sampling_values,
    scalar_fields,
)
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext, SubmoduleStep  # noqa: E402

CPU = torch.device("cpu")
V = 64
WALK = "decode"
STRIDE = _rng_offset_stride("cpu", V)


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


@pytest.fixture(params=[True, False], ids=["ingraph_scatter", "scatter_after_replay"])
def ingraph(request, monkeypatch) -> bool:
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "1" if request.param else "0")
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "1")
    return request.param


@pytest.fixture
def drawn(monkeypatch):
    """Per draw, each row's (temperature, top_k, top_p, seed, offset, penalty); a
    token of the row's seed and offset, so a token says which draw it was."""
    calls: list[list[tuple]] = []

    def record(temperature, top_k, top_p, seed, offset, penalty):
        rows = zip(temperature.tolist(), top_k.tolist(), top_p.tolist(), seed.tolist(), offset.tolist(),
                   penalty.tolist(), strict=True)
        calls.append(list(rows))
        return (seed * 7 + offset // STRIDE) % V

    def captured(logits, temperature, top_k, top_p, seed, offset, rep_penalty=None, **kwargs):
        return record(temperature, top_k, top_p, seed, offset, rep_penalty)

    def eager(logits, *, temperature, top_k, top_p, seed, rand_offset, repetition_penalty=1.0, **kwargs):
        n = logits.shape[0]

        def rows(x):
            x = torch.as_tensor(x).reshape(-1)
            return x.expand(n) if x.numel() == 1 else x

        return record(*(rows(x) for x in (temperature, top_k, top_p, seed, rand_offset, repetition_penalty)))

    monkeypatch.setattr(sampler_utils, "sample_accelerator_graphable_gpu", captured)
    monkeypatch.setattr(sampler_utils, "sample_tokens", eager)
    return calls


def _cfg(seed: int, temperature: float = 0.7, top_k: int = 20, top_p: float = 0.9, penalty: float = 1.0):
    cfg = SamplingReqConfig(temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=penalty)
    cfg.apply_conductor_config(seed=seed)
    return cfg


def _expected(cfg: SamplingReqConfig) -> tuple:
    t, k, p, _ = canonical_sampling_values(cfg.temperature, cfg.top_k, cfg.top_p, cfg.min_p)
    return (pytest.approx(t), k, pytest.approx(p), cfg._seed, pytest.approx(cfg.repetition_penalty))


class _Node:
    """A planner over a KV pool and a sampler, driven as the engine drives them."""

    def __init__(self, max_bs: int, with_kv: bool = True, planner: bool = True, history: int = 0):
        self.kv = None
        resources = {}
        if with_kv:
            self.kv = KVManager(
                cfg=PagedKVConfig(
                    num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=512, max_num_pages=64, page_size=4,
                ),
                name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
            )
            self.kv._cg_max_seq_len = 64
            resources["kv"] = self.kv
        self.sampler = SamplerResource(vocab_size=V, enable_repetion_penalty=True, device=CPU,
                                       max_repetition_window=history)
        resources["sampler"] = self.sampler
        # as warmup does: the runners claim their buffers, then the planner binds
        self.sampler.build_accelerator_graph_buffers(
            [SimpleNamespace(slot=0), SimpleNamespace(slot=1)], max_bs=max_bs, max_seq_len=1,
        )
        walk = {WALK: {key: (ResidentStream("main", reads=False),) for key in resources}}
        self.planner = ResidentPlanner("n", walk, resources, CPU, max_bs=max_bs) if planner else None
        self.bufs = self.sampler._cg_buffers
        self.captured: dict = {}
        self.tokens: dict[int, list[int]] = {}

    def ingest(self, rid: int, cfg: SamplingReqConfig, prompt: int = 3) -> None:
        self.sampler.ingest_request(rid, cfg)
        if self.bufs.ingraph_scatter:
            assert self.sampler.reserve(rid, "n", WALK, None) is None
        if self.kv is not None:
            self.kv.ingest_request(rid, KVReqConfig())
            self.host([rid], [prompt], walk="prefill")

    def host(self, rids, spans, walk="prefill"):
        """A host-planned eager step (a prefill or mixed step): the sampler draws eagerly."""
        segments = tuple(Segment(r, "main", n) for r, n in zip(rids, spans, strict=True))
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
        smp = SamplerStep()
        if self.kv is not None:
            kv = KVStep(segments=segments)
            assert self.kv.admit(kv, ctx).ok
            self.kv.plan(kv, ctx)
        self.sampler.plan(smp, ctx)
        out = self.sampler.sample(list(rids), torch.zeros(len(rids), V))
        if self.kv is not None:
            self.kv.commit(kv, ctx)
        self.sampler.commit(smp, ctx)
        if self.planner is not None:
            self.planner.commit(SubmoduleStep(steps={}, segments=list(segments)), ctx)
        for rid, token in zip(rids, out.tolist(), strict=True):
            self.tokens.setdefault(rid, []).append(token)
        return out

    def decode(
        self, rids, slot: int = 0, bs: int | None = None, preplan: bool = True, kept=None, replays: int = 1,
        draws: int = 1,
    ):
        """One captured decode step: capture once per (bucket, slot), then pre-plan,
        promote, read the loop-back tokens, flush, replay and commit. A step
        that repeats replays ``replays`` times, each drawing ``draws`` times."""
        bs = bs or len(rids)
        padded = tuple(rids) + tuple(-(i + 1) for i in range(bs - len(rids)))
        bucket = BucketKey(graph_walk=WALK, bs=bs, num_tokens=bs)
        lease = SlotLease(slot=slot, bucket=bucket)
        segments = tuple(Segment(r, "main", 1) for r in padded)
        kv = KVStep(segments=segments) if self.kv is not None else None
        smp = SamplerStep(kept_rids=kept)
        step = SubmoduleStep(steps={"sampler": smp, **({"kv": kv} if kv else {})}, segments=list(segments))
        ctx = StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=slot, capture=False, slot_lease=lease)
        ctx.set_padded_rids(padded)
        if kv is not None:
            assert self.kv.admit(kv, ctx).ok
        key = (bucket, slot)
        if key not in self.captured:
            for rid in padded[len(rids):]:
                # as the runner's capture rows are
                if rid not in self.sampler._sampler._sampling_config:
                    self.sampler.ingest_capture_row(rid)
            cap = StepContext(request_ids=padded, graph_walk=WALK, slot=slot, capture=True, slot_lease=lease)
            if kv is not None:
                cap.plan_results["kv"] = self.kv.plan(kv, cap)
            self.sampler.plan(smp, cap)
            step.set_ctx(cap)
            rec = self.planner.capture_step(bucket, slot, step) if self.planner is not None else None
            if rec is not None:
                self.planner.gather(rec)
            # the sampler the captured forward sampled with: its views are what a replay reads
            self.captured[key] = (rec, self.sampler._cg_sampler)
        step.set_ctx(ctx)
        rec, graph_sampler = self.captured[key]
        if self.planner is not None:
            ctx.resident = self.planner
        if preplan:
            ctx.is_preplan = True
            if self.planner is not None:
                self.planner.plan(step, ctx)
            if kv is not None:
                self.kv.plan(kv, ctx)
            self.sampler.plan(smp, ctx)
            ctx.is_preplan = False
        if self.planner is not None:
            self.planner.plan(step, ctx)
        if kv is not None:
            self.kv.plan(kv, ctx)
        self.sampler.plan(smp, ctx)
        assert self.sampler._cg_sampler is graph_sampler
        if self.planner is not None:
            self.planner.flush(ctx.resident_plan)
        n = len(rids)
        for _ in range(replays):
            if self.planner is not None:
                self.planner.gather(rec)  # the gather recorded before every forward
            for _ in range(draws):
                out = graph_sampler.sample(
                    list(padded), torch.zeros(bs, V), apply_penalty=self.sampler._apply_penalty_this_step,
                )
            if self.planner is not None:
                self.planner.commit_in_graph(rec)
            for rid, token in zip(rids, out[:n].tolist(), strict=True):
                if kept is None or rid in kept:
                    self.tokens.setdefault(rid, []).append(token)
        if kv is not None:
            self.kv.commit(kv, ctx)
        self.sampler.commit(smp, ctx)
        if self.planner is not None:
            self.planner.commit(step, ctx)
        return SimpleNamespace(ctx=ctx, out=out)


def _offset(node: _Node, rid: int) -> int:
    return int(node.bufs.rows.offset[node.bufs.rows.slot_of(rid)])


# ---- the cells ----

def test_scalars_are_read_by_slot_after_a_displacement_and_a_reload(drawn, ingraph):
    """Six requests over four slots (B=2): each pair displaces the oldest pair,
    and an offload gives a slot up. Every row reads its own scalars."""
    node = _Node(max_bs=2)
    cfgs = {rid: _cfg(seed=rid, temperature=0.1 * rid, top_k=rid, top_p=0.5 + 0.05 * rid, penalty=1 + 0.1 * rid)
            for rid in range(1, 7)}
    for rid, cfg in cfgs.items():
        node.ingest(rid, cfg)
    displaced = []
    take = node.planner._displace

    def spy(n, settled):
        out = take(n, settled)
        displaced.append(n)
        return out

    node.planner._displace = spy
    order = [(1, 2), (3, 4), (5, 6), (1, 2), (3, 4), (5, 6), (1, 2)]
    for i, rids in enumerate(order):
        drawn.clear()
        node.decode(list(rids), slot=i % 2)
        for j, rid in enumerate(rids):
            assert drawn[-1][j][:4] == _expected(cfgs[rid])[:4]
            assert drawn[-1][j][5] == _expected(cfgs[rid])[4]
    assert sum(displaced) >= 6
    # an offload gives the slot up; the reload is written whole
    node.planner.release(1)
    drawn.clear()
    node.decode([1, 2], slot=1)
    assert [row[:4] for row in drawn[-1]] == [_expected(cfgs[r])[:4] for r in (1, 2)]


def test_a_config_update_rewrites_its_row_and_the_next_sample_reads_it(drawn, ingraph):
    node = _Node(max_bs=2)
    node.ingest(1, _cfg(seed=1, temperature=0.5))
    node.ingest(2, _cfg(seed=2, temperature=0.6))
    node.decode([1, 2])
    node.decode([1, 2], slot=1)
    new = SamplingConfig(temperature=0.3, top_k=5, top_p=0.8)
    new.set_seed(9)
    node.bufs.update_request_config(2, new)
    drawn.clear()
    s = node.decode([1, 2])
    # the one row, written whole by the step after the update
    if ingraph:
        assert s.ctx.resident_plan.taken == {2}
    else:
        assert node.sampler._cells is None
    assert drawn[-1][0][0] == pytest.approx(0.5)
    assert drawn[-1][1][:4] == (pytest.approx(0.3), 5, pytest.approx(0.8), 9)
    # and no config version restages a resident step's scalars
    calls = []
    real = node.bufs.gather_static
    node.bufs.gather_static = lambda *a, **k: (calls.append(a), real(*a, **k))
    node.decode([1, 2], slot=1)
    assert bool(calls) is not ingraph


def test_padding_rows_read_the_defaults_and_leave_the_trash_row(drawn, ingraph):
    node = _Node(max_bs=4)
    node.ingest(1, _cfg(seed=3, temperature=0.4, top_k=7, top_p=0.6, penalty=1.3))
    drawn.clear()
    node.decode([1], bs=4)
    real, *padding = drawn[-1]
    assert real[:4] == _expected(_cfg(seed=3, temperature=0.4, top_k=7, top_p=0.6))[:4]
    defaults = {f.name: f.default for f in scalar_fields()}
    for row in padding:
        assert row == (defaults["temperature"], defaults["top_k"], defaults["top_p"], defaults["seed"],
                       row[4], defaults["rep_penalty"])
    # the planner's trash row stays zero; the sampler's trash row took the padding draws
    assert not node.planner.store.view(node.planner.max_slots, -1)[TRASH_SLOT].any()
    assert node.bufs.rows.slot_of(1) != sampler_resident.SAMPLER_TRASH_ROW


# ---- the offset and the tokens ----

def test_the_offset_advances_in_graph_through_displacements_offloads_and_host_steps(drawn, ingraph):
    """Each draw of a request is at the next offset whatever its slot, whether
    it was displaced or offloaded, and across eager steps between."""
    node = _Node(max_bs=2)
    for rid in range(1, 6):
        node.ingest(rid, _cfg(seed=rid))
    draws = {rid: 1 for rid in range(1, 6)}  # the prefill drew once
    plan = [((1, 2), "C"), ((3, 4), "C"), ((5, 1), "C"), ((2, 3), "E"), ((4, 5), "C"), ((1, 2), "C"),
            ((3, 4), "C"), ((5, 2), "C")]
    for i, (rids, mode) in enumerate(plan):
        if i == 4:
            node.planner.release(4)  # offloaded, then reloaded by the next step it runs
        if mode == "C":
            node.decode(list(rids), slot=i % 2)
        else:
            node.host(list(rids), [1, 1], walk="mixed")
        for rid in rids:
            draws[rid] += 1
            assert _offset(node, rid) == draws[rid] * STRIDE
    # every draw of a request was at the next offset: no offset twice, none skipped
    for rid, tokens in node.tokens.items():
        assert tokens == [(rid * 7 + k) % V for k in range(len(tokens))]


def test_tokens_do_not_depend_on_the_slot_or_the_path(drawn, ingraph):
    """The same requests through a resident node that displaces every step and
    through one with no planner draw the same tokens."""
    runs = []
    for planner in (True, False):
        node = _Node(max_bs=2, planner=planner)
        for rid in range(1, 6):
            node.ingest(rid, _cfg(seed=100 + rid))
        for i, rids in enumerate([(1, 2), (3, 4), (5, 1), (2, 3), (4, 5), (1, 2), (3, 5)]):
            node.decode(list(rids), slot=i % 2)
        runs.append(node.tokens)
    assert runs[0] == runs[1]


@pytest.mark.parametrize("restore", [None, frozenset({1, 2})])
def test_a_dropped_draw_keeps_its_offset(drawn, ingraph, restore):
    node = _Node(max_bs=2)
    node.ingest(1, _cfg(seed=1))
    node.ingest(2, _cfg(seed=2))
    node.decode([1, 2])
    node.decode([1, 2], slot=1, kept=frozenset({1}))
    assert _offset(node, 1) == 3 * STRIDE
    assert _offset(node, 2) == 2 * STRIDE
    node.decode([1, 2])
    assert _offset(node, 2) == 3 * STRIDE
    # Both default and explicit all-kept restore the same capture's writes.
    node.decode([1, 2], slot=1, kept=restore)
    assert _offset(node, 1) == 5 * STRIDE
    assert _offset(node, 2) == 4 * STRIDE


def test_a_recycled_row_whose_first_step_is_resident_starts_at_zero(drawn, ingraph):
    """A request that decodes here first (its prefill ran elsewhere) takes a
    sampler row another request drew on; its first draw is at offset 0."""
    node = _Node(max_bs=2, with_kv=False)
    node.ingest(1, _cfg(seed=1))
    for i in range(3):
        node.decode([1], slot=i % 2)
    row = node.bufs.rows.slot_of(1)
    node.planner.release(1)
    node.sampler.remove_request(1)
    node.ingest(2, _cfg(seed=2))
    assert node.bufs.rows.slot_of(2) == row
    drawn.clear()
    node.decode([2])
    assert drawn[-1][0][4] == 0
    assert node.tokens[2] == [(2 * 7) % V]


def test_each_replay_reads_sampler_identity_and_contents_from_resident_cells(drawn, monkeypatch):
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "1")
    node = _Node(max_bs=3, with_kv=False, history=4)
    for rid in (1, 2, 3):
        node.ingest(rid, _cfg(seed=rid, penalty=1.3))
    node.decode([1, 2], bs=3)
    # Initialize a new owner row without changing this captured batch's plan.
    # Every later replay must gather its RNG/history identity from the binding.
    joining = SubmoduleStep({"sampler": SamplerStep()}, segments=[Segment(3, "main", 1)])
    joining.set_ctx(StepContext((3,), WALK, 0, False, resident=node.planner, repeats=4,
                               slot_lease=SlotLease(0, BucketKey(WALK, 3, 3))))
    assert StepRunner({"sampler": node.sampler}).admit_rows(joining).ok
    snapshot = node.planner.prepare_rows(joining, joining.ctx)
    node.planner.flush(snapshot)
    rec, sampler = node.captured[(BucketKey(WALK, 3, 3), 0)]
    owner = node.sampler._rows
    masks = []
    original = sampler_utils.sample_accelerator_graphable_gpu

    def inspect(*args, **kwargs):
        masks.append(kwargs["seen_tokens"].clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(sampler_utils, "sample_accelerator_graphable_gpu", inspect)
    # Only device-side resident membership changes between these reads. The
    # same prepared sampler/gather must stop reading and writing the old owner.
    state = RunState(3, None, (), CPU, node.planner.drop_rows(BucketKey(WALK, 3, 3), 0),
                     independent_rows=True)
    state.stage(2, [False, False], 4)
    for owners, ready in (((1, 2), (True, True)), ((1, 2), (False, True)),
                          ((1, 2), (True, True)), ((3,), (True,))):
        rids = tuple(rid for rid, enabled in zip(owners, ready, strict=True) if enabled)
        rows = owner.indices((1, 2, 3))
        before_offsets = owner.offset.index_select(0, rows).clone()
        before_tokens, before_counts = owner.history.read(rows)
        before_masks = owner.read_seen(rows)
        slots = [node.planner.slot_of(rid) for rid in owners]
        state.dropping.rows.copy_(torch.tensor([*slots, *([TRASH_SLOT] * (3 - len(slots)))]))
        state.eligible.copy_(torch.tensor([*ready, *([False] * (3 - len(ready)))]))
        state.select_rows()
        state.head()
        node.planner.gather(rec)
        out = sampler.sample([1, 2, -1], torch.zeros(3, V), apply_penalty=True)
        for rid in rids:
            i = owners.index(rid)
            assert drawn[-1][i][3:5] == (rid, before_offsets[rid - 1].item())
            torch.testing.assert_close(masks[-1][i], before_masks[rid - 1])
            assert owner.offset[owner.slot_of(rid)].item() == before_offsets[rid - 1].item() + STRIDE
            assert owner.history.count[owner.slot_of(rid)].item() == before_counts[rid - 1].item() + 1
            assert owner.read_seen(owner.indices([rid]))[0, out[i]].item()
        for rid in set((1, 2, 3)) - set(rids):
            row = owner.indices([rid])
            assert owner.offset[row].item() == before_offsets[rid - 1].item()
            tokens, counts = owner.history.read(row)
            assert counts.item() == before_counts[rid - 1].item()
            torch.testing.assert_close(tokens[0], before_tokens[rid - 1])
            torch.testing.assert_close(owner.read_seen(row)[0], before_masks[rid - 1])
    node.planner.release(3)
    node.planner.unpin(tuple(snapshot.slots.tolist()))


def test_preplanning_a_successor_preserves_the_current_identity_and_kept_rows(drawn, monkeypatch):
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "1")
    node = _Node(max_bs=2, with_kv=False)
    for rid in (1, 2, 3):
        node.ingest(rid, _cfg(seed=rid))
    node.decode([1, 2])
    node.decode([2, 3], slot=1, kept=frozenset({3}))
    before = {rid: _offset(node, rid) for rid in (1, 2, 3)}
    real_flush = node.planner.flush

    def preplan_after_flush(plan):
        real_flush(plan)
        # A successor may be prepared while this replay is outstanding.
        monkeypatch.setattr(node.planner, "flush", real_flush)
        ctx = StepContext((2, 3), WALK, 1, False, is_preplan=True,
                          slot_lease=SlotLease(1, BucketKey(WALK, 2, 2)))
        ctx.resident = node.planner
        smp = SamplerStep(kept_rids=frozenset({3}))
        step = SubmoduleStep(steps={"sampler": smp}, segments=[Segment(rid, "main", 1) for rid in (2, 3)])
        node.planner.plan(step, ctx)
        node.sampler.plan(smp, ctx)

    monkeypatch.setattr(node.planner, "flush", preplan_after_flush)
    node.decode([1, 2])
    assert [_offset(node, rid) for rid in (1, 2, 3)] == [before[1] + STRIDE, before[2] + STRIDE, before[3]]
    node.decode([2, 3], slot=1)
    assert [_offset(node, rid) for rid in (1, 2, 3)] == [before[1] + STRIDE, before[2] + 2 * STRIDE, before[3] + STRIDE]


# ---- what a resident step stages ----

def test_a_steady_batch_uploads_no_rows_and_restages_nothing(drawn, monkeypatch):
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "1")
    node = _Node(max_bs=2)
    node.ingest(1, _cfg(seed=1))
    node.ingest(2, _cfg(seed=2))
    node.decode([1, 2])
    node.decode([1, 2], slot=1)
    copies, statics, dynamics = [], [], []
    stager = node.sampler._cells._stager
    real = stager.copy_
    monkeypatch.setattr(stager, "copy_", lambda *a, **k: (copies.append(1), real(*a, **k)))
    monkeypatch.setattr(node.bufs, "gather_static", lambda *a, **k: statics.append(1))
    monkeypatch.setattr(node.bufs, "gather_dynamic", lambda *a, **k: dynamics.append(1))
    node.decode([1, 2])
    node.decode([1, 2], slot=1)
    assert copies == statics == dynamics == []
    node.ingest(3, _cfg(seed=3))
    node.decode([3, 2])
    assert copies == statics == dynamics == []
    assert drawn[-1][0][3] == 3


def test_a_node_without_a_planner_keeps_host_staging(drawn, ingraph):
    node = _Node(max_bs=2, planner=False)
    assert node.sampler._cells is None and node.bufs.cells is None
    node.ingest(1, _cfg(seed=1, temperature=0.3))
    staged = []
    real = node.bufs.gather_static
    node.bufs.gather_static = lambda *a, **k: (staged.append(1), real(*a, **k))
    drawn.clear()
    node.decode([1])
    assert staged and drawn[-1][0][0] == pytest.approx(0.3)


def test_host_steps_of_a_resident_node_keep_host_staging(drawn, ingraph):
    """A mixed or prefill capture of a resident node isn't resident: its sampler
    stages its scalars on the host, and the scalar rows its graphs read are
    not the ones a resident graph fills."""
    node = _Node(max_bs=2)
    node.ingest(1, _cfg(seed=1, temperature=0.3))
    node.decode([1])
    resident = node.sampler._cg_sampler
    host = node.bufs.sampler_for(1, 0)
    assert (resident is not host) is ingraph
    assert (resident.temperature_buf.data_ptr() != host.temperature_buf.data_ptr()) is ingraph


def test_other_owners_keep_their_cells(drawn, ingraph):
    """The sampler's cells sit beside the KV pool's: each table its own columns,
    the pool's rows as its stream."""
    node = _Node(max_bs=2)
    for rid in (1, 2, 3):
        node.ingest(rid, _cfg(seed=rid), prompt=5)
    for i, rids in enumerate([(1, 2), (3, 1), (2, 3), (1, 2)]):
        node.decode(list(rids), slot=i % 2)
    tables = node.kv._resident_tables["main"]
    cols = [t.col for t in node.sampler._cells._tables] if ingraph else []
    if ingraph:
        assert cols == list(range(cols[0], cols[0] + len(cols)))
    assert tables.kv_len.col not in cols and not set(cols) & set(range(tables.block_table.col, node.planner._stride))
    for rid in (1, 2):
        slot = node.planner.slot_of(rid)
        stream = node.kv._streams[rid]["main"]
        assert int(tables.kv_len.view[slot, 0]) == stream.stored_len
        n = -(-stream.stored_len // 4)
        assert tables.block_table.view[slot, :n].tolist() == stream.page_indices[:n]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the cell gather kernel needs CUDA")
def test_the_cell_kernel_matches_the_torch_ops_under_capture():
    """Every field a spec can enable, each dtype, a padding row: the captured
    kernel writes what the CPU path does."""
    fields = (*scalar_fields(True, True, True, True, True, True),
              sampler_utils.ScalarField("sampler_row", torch.long, 0))
    sizes = tuple(torch.empty((), dtype=f.dtype).element_size() for f in fields)
    enc = [sampler_resident._encoder(f.dtype) for f in fields]
    defaults = [e(f.default) for e, f in zip(enc, fields, strict=True)]
    stride, col, slots_n, bs = len(fields) + 3, 2, 9, 6
    rng = np.random.default_rng(0)
    store = torch.zeros(slots_n * stride, dtype=torch.long)
    for slot in range(1, slots_n):
        for j, (f, e) in enumerate(zip(fields, enc, strict=True)):
            value = (rng.random() if f.dtype == torch.float32 else bool(rng.integers(0, 2)) if f.dtype == torch.bool
                     else int(rng.integers(-2**40, 2**40)) if f.dtype == torch.long else int(rng.integers(0, 1000)))
            store[slot * stride + col + j] = e(value)
    slots = torch.tensor([4, 0, 8, 1, 0, 3])
    want = sampler_resident._ScalarRow(fields, sizes, bs, CPU)
    sampler_resident.gather_cells(store, slots, stride, col, want, fields, sizes, defaults, torch.tensor(defaults))
    got = sampler_resident._ScalarRow(fields, sizes, bs, torch.device("cuda"))
    store_d, slots_d, cells_d = store.cuda(), slots.cuda(), torch.tensor(defaults).cuda()
    sampler_resident.gather_cells(store_d, slots_d, stride, col, got, fields, sizes, defaults, cells_d)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampler_resident.gather_cells(store_d, slots_d, stride, col, got, fields, sizes, defaults, cells_d)
    got.row.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(got.row.cpu(), want.row)
    for f in fields:
        assert want.views[f.name][1].item() == f.default


# ---- steps that repeat ----

def test_a_penalised_run_of_four_draws_as_four_steps_do(drawn, ingraph):
    """A run replays its draws on one plan: the RNG offset and the seen-token
    mask move in the graph, so four replays give the tokens, offsets and mask
    four one-replay steps do, and a live penalty no longer holds a step to one."""
    seen = []
    for replays in (1, 4):
        node = _Node(max_bs=2, with_kv=False)
        node.ingest(1, _cfg(seed=1, penalty=1.3))
        smp_ctx = StepContext(request_ids=(1,), graph_walk=WALK, slot=0, capture=False)
        assert node.sampler.max_repeats(SamplerStep(), smp_ctx) > 4
        for _ in range(4 // replays):
            node.decode([1], replays=replays)
        mask = node.sampler._rows.read_seen(node.sampler._rows.indices([1]))[0]
        seen.append((node.tokens[1], _offset(node, 1), mask.nonzero().flatten().tolist()))
    assert seen[0] == seen[1]
    assert len(set(seen[0][0])) > 1 and seen[0][1] == 4 * STRIDE and seen[0][2]


def test_a_loop_back_free_sampler_drawing_fifteen_times_a_replay_may_repeat(drawn, monkeypatch):
    """A second sampler of a node, one no loop-back reads (a code predictor's),
    moves its own offsets in the graph too: its walk may repeat, and each of
    its 15 draws a replay, in each of 4 replays, is fresh."""
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "1")
    node = _Node(max_bs=2, with_kv=False)
    assert node.sampler.repeat_refusal(WALK) is None
    node.ingest(1, _cfg(seed=1))
    node.decode([1], replays=4, draws=15)
    offsets = [row[0][4] for row in drawn]
    assert offsets == [i * STRIDE for i in range(60)]
    assert _offset(node, 1) == 60 * STRIDE


@pytest.mark.parametrize("unsafe", ["reset", "prompt", "cached_prefix", "filters", "dropped_draw"])
def test_joining_sampler_requires_initialized_owner_and_matching_capture(unsafe, monkeypatch, drawn):
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "1")
    node = _Node(max_bs=2, with_kv=False)
    node.ingest(1, _cfg(11))
    node.decode([1], bs=2)
    node.ingest(2, _cfg(12))
    sampler = node.sampler
    lease = SlotLease(0, BucketKey(WALK, 2, 2))
    ctx = StepContext((2,), WALK, 0, False, slot_lease=lease, resident=node.planner, repeats=4)
    step = SamplerStep()
    if unsafe != "reset":
        sampler._rows.sync()
    if unsafe == "prompt":
        step = SamplerStep(prefill_tracked_tokens={2: torch.tensor([3])})
    elif unsafe == "cached_prefix":
        sampler._cached_prefix[2] = {0: torch.tensor([3])}
    elif unsafe == "filters":
        step = SamplerStep(apply_filters=False)
    elif unsafe == "dropped_draw":
        sampler._cells.stage_keep(lease.bucket, lease.slot, (1,), frozenset())
    with pytest.raises(ValueError):
        sampler.resident_binding().check_horizon(step, ctx)
    sampler._rows.sync()
    sampler._cached_prefix.clear()
    sampler._cells.stage_keep(lease.bucket, lease.slot, (1,), None)
    sampler.resident_binding().check_horizon(SamplerStep(), ctx)
