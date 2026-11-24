"""The shared sampler's ring gives the model's RAS tokens across steps."""

from __future__ import annotations

import math
import time
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from mstar.communication.wire import decode, encode
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.resident import TRASH_SLOT, ResidentPlanner
from mstar.engine.resources.sampler import resident as sampler_resident
from mstar.engine.resources.sampler import resource as sampler_resource
from mstar.engine.resources.sampler import utils as sampler_utils
from mstar.engine.resources.sampler.config import (
    RASSamplingReqConfig,
    SamplerSpec,
    SamplerStep,
    SamplingReqConfig,
)
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sampler.utils import (
    GenerationHistory,
    SamplerBuffers,
    SamplingConfig,
    samples_greedy,
    scalar_fields,
)
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext, SubmoduleStep
from mstar.model.cosyvoice3.components.sampling import ras_sample_plain as old_ras_sample
from mstar.model.cosyvoice3.config import LM_SAMPLER
from mstar.model.cosyvoice3.cosyvoice3_model import CosyVoice3Model
from mstar.model.cosyvoice3.submodules import LMSubmodule

CPU = torch.device("cpu")


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def remove_request(self, request_id):
        del request_id

    def cleanup(self):
        pass


def _resource(window: int) -> SamplerResource:
    return SamplerResource(
        vocab_size=None, enable_repetition_penalty=False, device=torch.device("cpu"),
        max_repetition_window=window, enable_blocked_tokens=True, enable_ras=True,
    )


def _config(seed: int, window: int, tau: float, temperature: float, top_p: float) -> RASSamplingReqConfig:
    cfg = RASSamplingReqConfig(
        temperature=temperature, top_k=7, top_p=top_p,
        repetition_window=window, ras_mode=True, ras_tau=tau,
    )
    cfg.apply_conductor_config(seed=seed, resource_key="lm_sampler")
    return cfg


def test_ras_request_settings_are_model_scoped_and_cross_typed_wire():
    public_fields = {field.name for field in fields(SamplingReqConfig)}
    assert "ras_mode" not in public_fields and "ras_tau" not in public_fields
    cfg = _config(7, 25, 0.6, 1.0, 0.0)
    decoded = decode(encode(cfg))
    assert isinstance(decoded, RASSamplingReqConfig)
    assert decoded == cfg and decoded.ras_mode and decoded.ras_tau == 0.6


def test_ras_resource_matches_model_draws_across_mixed_batch_steps():
    width = 4
    rows = {
        "a": _config(123456789, 4, 0.25, 1.0, 0.8),
        "b": _config(987654321, 2, 0.5, 0.7, 0.0),
        "c": _config(42, 0, 0.0, 0.0, 0.9),
    }
    resource = _resource(width)
    for rid, cfg in rows.items():
        resource.ingest_request(rid, cfg)
    history = {rid: torch.full((width,), -1, dtype=torch.long) for rid in rows}
    counts = dict.fromkeys(rows, 0)
    generator = torch.Generator().manual_seed(345)

    for batch in (["a"], ["a", "b"], ["b", "a", "c"], ["c", "b", "a"], ["a", "c"]):
        logits = torch.randn(len(batch), 11, generator=generator) * 2
        blocked = torch.zeros_like(logits, dtype=torch.bool)
        # A forced EOS row keeps only one id; the model does this at prefill's
        # terminal token. Other rows exercise the minimum-length mask.
        if "c" in batch:
            blocked[batch.index("c"), :] = True
            blocked[batch.index("c"), 9] = False
        if "b" in batch:
            blocked[batch.index("b"), 10] = True
        legacy = old_ras_sample(
            logits, seed=torch.tensor([rows[rid].seed for rid in batch]),
            step=torch.tensor([counts[rid] for rid in batch]),
            history=torch.stack([history[rid] for rid in batch]),
            temperature=torch.tensor([rows[rid].temperature for rid in batch]),
            top_p=torch.tensor([rows[rid].top_p for rid in batch]),
            top_k=torch.tensor([rows[rid].top_k for rid in batch]),
            window=torch.tensor([rows[rid].repetition_window for rid in batch]),
            repeat_threshold=torch.tensor([
                rows[rid].repetition_window * rows[rid].ras_tau for rid in batch
            ]),
            blocked=blocked,
        )
        actual = resource.sample(batch, logits, blocked)
        assert torch.equal(actual, legacy)
        for rid, token in zip(batch, legacy, strict=True):
            history[rid] = torch.cat((history[rid][1:], token.reshape(1)))
            counts[rid] += 1

    assert resource._history is not None
    assert all(resource._history._rid_to_slot[rid] > 0 for rid in rows)


def test_ras_resource_keeps_a_requests_tokens_across_batch_sizes():
    only, mixed = _resource(3), _resource(3)
    for resource in (only, mixed):
        resource.ingest_request("a", _config(7, 3, 0.34, 1.0, 0.8))
    mixed.ingest_request("b", _config(8, 3, 0.34, 1.0, 0.8))
    generator = torch.Generator().manual_seed(891)
    for _ in range(6):
        a_logits = torch.randn(1, 17, generator=generator)
        b_logits = torch.randn(1, 17, generator=generator)
        a = only.sample(["a"], a_logits)
        both = mixed.sample(["b", "a"], torch.cat((b_logits, a_logits)))
        assert a.item() == both[1].item()


def test_ras_threshold_keeps_python_product_rounding():
    # The model computed float(window) * float(tau), then staged one float32.
    # Multiplying a float32 window by a float32 tau rounds 25 * 0.6 slightly
    # above 15 and misses the redraw at exactly 15 repeated tokens.
    eager, captured = _resource(25), _resource(25)
    captured.build_accelerator_graph_buffers(slots=[], max_bs=1, max_seq_len=0)
    for resource in (eager, captured):
        resource.ingest_request("a", _config(7, 25, 0.6, 1.0, 0.0))
    buffers = captured._cg_buffers
    assert buffers is not None
    logits = torch.tensor([[5.0, 0.0]])
    for step in range(16):
        expected = eager.sample(["a"], logits)
        buffers.gather_static(["a"], 1, 0)
        buffers.gather_dynamic(["a"], 1, 0)
        assert captured._history is not None
        captured._history.gather_step(["a"], 1)
        captured._cg_sampler = buffers.sampler_for(1, 0)
        actual = captured.sample(["a"], logits)
        captured._history.scatter_step()
        captured._cg_sampler = None
        assert actual.item() == expected.item() == (1 if step == 15 else 0)


def test_ras_resource_eager_and_captured_row_emulation_match():
    eager, captured = _resource(4), _resource(4)
    captured.build_accelerator_graph_buffers(slots=[], max_bs=3, max_seq_len=0)
    for resource in (eager, captured):
        resource.ingest_request("a", _config(100, 4, 0.25, 1.0, 0.8))
        resource.ingest_request("b", _config(200, 2, 0.5, 0.0, 0.0))
    buffers = captured._cg_buffers
    assert buffers is not None
    generator = torch.Generator().manual_seed(125)
    for batch in (["a"], ["b", "a"], ["a", "b"], ["b"]):
        logits = torch.randn(len(batch), 15, generator=generator)
        blocked = torch.zeros_like(logits, dtype=torch.bool)
        if "b" in batch:
            blocked[batch.index("b"), 14] = True
        expected = eager.sample(batch, logits, blocked)
        buffers.gather_static(batch, 3, 0)
        buffers.gather_dynamic(batch, 3, 0)
        assert captured._history is not None
        captured._history.gather_step(batch, 3)
        captured._cg_sampler = buffers.sampler_for(3, 0)
        padded_logits = torch.cat((logits, torch.zeros(3 - len(batch), 15)))
        padded_blocked = torch.cat((blocked, torch.zeros(3 - len(batch), 15, dtype=torch.bool)))
        actual = captured.sample(batch, padded_logits, padded_blocked)
        captured._history.scatter_step()
        captured._cg_sampler = None
        assert torch.equal(actual[:len(batch)], expected)


# ---------------------------------------------------------------------------
# ResidentPlanner + SamplerCells + 6-field RAS optimization
# ---------------------------------------------------------------------------


def test_ras_fields_for_omits_unused_penalty_fields():
    history = GenerationHistory(10, CPU)
    fields_ras = SamplerBuffers.fields_for(
        history=history, enable_ras=True, enable_repetition_penalty=False,
    )
    assert tuple(f.name for f in fields_ras) == (
        "temperature", "top_k", "top_p", "seed", "window", "ras_threshold",
    )
    # Non-RAS defaults remain unchanged (all 5 base + 2 windowed fields).
    fields_std = SamplerBuffers.fields_for(history=history, enable_ras=False)
    assert tuple(f.name for f in fields_std) == (
        "temperature", "top_k", "top_p", "seed", "rep_penalty", "window", "window_penalty",
    )
    assert scalar_fields() == tuple(f for f in fields_std if f.name not in ("window", "window_penalty"))

    res = _resource(10)
    planner = ResidentPlanner("lm", {"decode"}, {"sampler": res}, CPU, max_bs=4)
    assert tuple(f.name for f in res._cells.fields) == (
        "temperature", "top_k", "top_p", "seed", "window", "ras_threshold",
    )
    res.build_accelerator_graph_buffers([SimpleNamespace(slot=0)], max_bs=4, max_seq_len=1)
    bufs = res._cg_buffers
    assert bufs is not None
    assert bufs.rep_penalty is None and bufs.window_penalty is None
    assert bufs.ras_threshold is not None and bufs.window is not None
    sliced = bufs.slice_for_bs(2, 0)
    assert sliced["rep_penalty_buf"] is None
    assert sliced["history"].window_penalty is None
    assert sliced["ras_threshold_buf"] is not None

    # Verify gather_cells on CPU reads the 6 RAS fields and TRASH_SLOT defaults accurately.
    sizes = tuple(torch.empty((), dtype=f.dtype).element_size() for f in fields_ras)
    enc = [sampler_resident._encoder(f.dtype) for f in fields_ras]
    defaults = [e(f.default) for e, f in zip(enc, fields_ras, strict=True)]
    stride, col, bs = len(fields_ras) + 2, 1, 3
    store = torch.zeros(4 * stride, dtype=torch.long)
    sample_vals = {
        "temperature": 0.75, "top_k": 25, "top_p": 0.8,
        "seed": 2**45 + 7, "window": 10, "ras_threshold": 1.0,
    }
    for j, (f, e) in enumerate(zip(fields_ras, enc, strict=True)):
        store[2 * stride + col + j] = e(sample_vals[f.name])
    row = sampler_resident._ScalarRow(fields_ras, sizes, bs, CPU)
    sampler_resident.gather_cells(
        store, torch.tensor([2, TRASH_SLOT, 2]), stride, col, row,
        fields_ras, sizes, defaults, torch.tensor(defaults),
    )
    for f in fields_ras:
        assert row.views[f.name][0].item() == pytest.approx(sample_vals[f.name])
        assert row.views[f.name][1].item() == f.default
        assert row.views[f.name][2].item() == pytest.approx(sample_vals[f.name])


class _RASResidentNode:
    """Drives SamplerResource(enable_ras=True) alongside ResidentPlanner + KVManager."""

    def __init__(self, max_bs: int = 2, max_window: int = 4, num_kv_pages: int = 16, monkeypatch=None):
        if monkeypatch is not None:
            monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
        self.kv = KVManager(
            cfg=PagedKVConfig(
                num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=128,
                max_num_pages=num_kv_pages, page_size=4,
            ),
            name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
        )
        self.kv._cg_max_seq_len = 64
        self.sampler = SamplerResource(
            vocab_size=None, enable_repetition_penalty=False, device=CPU,
            max_repetition_window=max_window, enable_blocked_tokens=True, enable_ras=True,
        )
        self.planner = ResidentPlanner(
            "lm", {"decode"}, {"kv": self.kv, "sampler": self.sampler}, CPU, max_bs=max_bs,
        )
        self.sampler.build_accelerator_graph_buffers(
            [SimpleNamespace(slot=0), SimpleNamespace(slot=1)], max_bs=max_bs, max_seq_len=1,
        )
        self.captured: dict = {}

    def ingest(self, rid, cfg: RASSamplingReqConfig, prompt: int = 2) -> None:
        self.sampler.ingest_request(rid, cfg)
        self.kv.ingest_request(rid, KVReqConfig())
        self.host([rid], [prompt], torch.randn(1, 13))

    def host(self, rids, spans, logits: torch.Tensor, blocked: torch.Tensor | None = None,
             kept=None, walk: str = "prefill") -> torch.Tensor:
        segments = tuple(Segment(r, "main", n) for r, n in zip(rids, spans, strict=True))
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
        kv = KVStep(segments=segments)
        smp = SamplerStep(kept_rids=kept)
        assert self.kv.admit(kv, ctx).ok
        self.kv.plan(kv, ctx)
        self.sampler.plan(smp, ctx)
        out = self.sampler.sample(list(rids), logits, blocked_tokens=blocked)
        self.kv.commit(kv, ctx)
        self.sampler.commit(smp, ctx)
        self.planner.commit(SubmoduleStep(steps={}, segments=list(segments)), ctx)
        return out

    def decode(self, rids, logits: torch.Tensor, blocked: torch.Tensor | None = None,
               slot: int = 0, bs: int | None = None, preplan: bool = True,
               kept=None, rebuild_rids=frozenset(), chunk_tokens_after=None) -> tuple[torch.Tensor, StepContext]:
        bs = bs or len(rids)
        vocab = logits.shape[-1]
        padded = tuple(rids) + tuple(-(i + 1) for i in range(bs - len(rids)))
        bucket = BucketKey(graph_walk="decode", bs=bs, num_tokens=bs)
        lease = SlotLease(slot=slot, bucket=bucket)
        segments = tuple(Segment(r, "main", 1) for r in padded)
        kv = KVStep(segments=segments)
        smp = SamplerStep(kept_rids=kept)
        step = SubmoduleStep(steps={"sampler": smp, "kv": kv}, segments=list(segments))
        ctx = StepContext(
            request_ids=tuple(rids), graph_walk="decode", slot=slot, capture=False,
            slot_lease=lease, rebuild_rids=frozenset(rebuild_rids),
            chunk_tokens_after=dict(chunk_tokens_after or {}),
        )
        ctx.set_padded_rids(padded)
        assert self.kv.admit(kv, ctx).ok
        key = (bucket, slot)
        if key not in self.captured:
            for rid in padded[len(rids):]:
                if rid not in self.sampler._sampler._sampling_config:
                    self.sampler.ingest_capture_row(rid)
            cap = StepContext(request_ids=padded, graph_walk="decode", slot=slot, capture=True, slot_lease=lease)
            self.kv.plan(kv, cap)
            self.sampler.plan(smp, cap)
            rec = self.planner.capture_step(bucket, slot, step)
            self.planner.gather(rec)
            self.captured[key] = (rec, self.sampler._cg_sampler)
        rec, graph_sampler = self.captured[key]
        ctx.resident = self.planner
        if preplan:
            ctx.is_preplan = True
            self.planner.plan(step, ctx)
            self.kv.plan(kv, ctx)
            self.sampler.plan(smp, ctx)
            ctx.is_preplan = False
        self.planner.plan(step, ctx)
        self.kv.plan(kv, ctx)
        self.sampler.plan(smp, ctx)
        assert self.sampler._cg_sampler is graph_sampler
        self.planner.flush(ctx.resident_plan)
        for owner in self.planner._owners:
            if owner is self.sampler:
                self.sampler._cells.gather(rec)
            else:
                owner.resident_gather(rec)
        if len(rids) < bs:
            logits = torch.cat((logits, torch.zeros(bs - len(rids), vocab, dtype=logits.dtype)))
            if blocked is not None:
                blocked = torch.cat((blocked, torch.zeros(bs - len(rids), vocab, dtype=torch.bool)))
        out = self.sampler.sample(list(padded), logits, blocked_tokens=blocked)
        self.planner.commit_in_graph(rec)
        self.kv.commit(kv, ctx)
        self.sampler.commit(smp, ctx)
        self.planner.commit(step, ctx)
        return out[:len(rids)], ctx


def test_ras_resident_planner_parity_with_displacements_offload_preplan_and_updates(monkeypatch):
    width, vocab = 4, 13
    node = _RASResidentNode(max_bs=2, max_window=width, num_kv_pages=64, monkeypatch=monkeypatch)
    cfgs = {
        1: _config(101, 4, 0.25, 1.0, 0.8),
        2: _config(102, 2, 0.50, 0.7, 0.0),
        3: _config(103, 0, 0.00, 0.0, 0.9),
        4: _config(104, 4, 0.25, 0.9, 1.0),
        5: _config(105, 3, 0.34, 1.0, 0.5),
        6: _config(106, 1, 1.00, 1.2, 0.8),
    }
    history = {rid: torch.full((width,), -1, dtype=torch.long) for rid in cfgs}
    counts = dict.fromkeys(cfgs, 0)
    gen = torch.Generator().manual_seed(2026)

    def ref_step(rids, logits, blocked):
        legacy = old_ras_sample(
            logits,
            seed=torch.tensor([cfgs[r].seed for r in rids]),
            step=torch.tensor([counts[r] for r in rids]),
            history=torch.stack([history[r] for r in rids]),
            temperature=torch.tensor([cfgs[r].temperature for r in rids]),
            top_p=torch.tensor([cfgs[r].top_p for r in rids]),
            top_k=torch.tensor([cfgs[r].top_k for r in rids]),
            window=torch.tensor([cfgs[r].repetition_window for r in rids]),
            repeat_threshold=torch.tensor([cfgs[r].repetition_window * cfgs[r].ras_tau for r in rids]),
            blocked=blocked,
        )
        for rid, tok in zip(rids, legacy, strict=True):
            history[rid] = torch.cat((history[rid][1:], tok.reshape(1)))
            counts[rid] += 1
        return legacy

    # Ingest requests (prefill eager step) and verify parity with reference.
    for rid, cfg in cfgs.items():
        node.sampler.ingest_request(rid, cfg)
        node.kv.ingest_request(rid, KVReqConfig())
        logits = torch.randn(1, vocab, generator=gen) * 2
        blocked = torch.zeros(1, vocab, dtype=torch.bool)
        blocked[0, 0] = True
        actual = node.host([rid], [2], logits, blocked=blocked)
        expected = ref_step([rid], logits, blocked)
        assert torch.equal(actual, expected)

    displaced = []
    take = node.planner._displace
    node.planner._displace = lambda n, settled: (displaced.append(n), take(n, settled))[1]

    # Exercise displacements across 6 requests, offload, padding rows, and slot alternation.
    batches = [
        ([1, 2], 2, 0),
        ([3, 4], 2, 1),
        ([5, 6], 2, 0),
        ([1, 2], 2, 1),
        ([3], 2, 0),      # padding row in bs=2 bucket
        ([4, 5], 2, 1),
        ([6, 1], 2, 0),
    ]
    for step_i, (rids, bs, slot) in enumerate(batches):
        if step_i == 3:
            node.planner.release(1)  # offload + reload on next step
        logits = torch.randn(len(rids), vocab, generator=gen) * 2
        blocked = torch.zeros(len(rids), vocab, dtype=torch.bool)
        if 3 in rids:
            blocked[rids.index(3), :] = True
            blocked[rids.index(3), 7] = False
        actual, _ = node.decode(rids, logits, blocked=blocked, slot=slot, bs=bs)
        expected = ref_step(rids, logits, blocked)
        assert torch.equal(actual, expected)
    assert sum(displaced) >= 2

    # Pre-plan discard: pre-plan for [1, 2], then execute a different batch [3, 4] inline.
    pre_step = SubmoduleStep(
        steps={"sampler": SamplerStep(), "kv": KVStep(segments=(Segment(1, "main", 1), Segment(2, "main", 1)))},
        segments=[Segment(1, "main", 1), Segment(2, "main", 1)],
    )
    pre_ctx = StepContext(
        request_ids=(1, 2), graph_walk="decode", slot=0, capture=False,
        slot_lease=SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=2, num_tokens=2)),
        is_preplan=True, resident=node.planner,
    )
    pre_ctx.set_padded_rids((1, 2))
    node.planner.plan(pre_step, pre_ctx)
    node.sampler.plan(SamplerStep(), pre_ctx)
    assert node.sampler._preplanned
    logits = torch.randn(2, vocab, generator=gen)
    actual, _ = node.decode([3, 4], logits, blocked=None, slot=0, bs=2, preplan=False)
    assert not node.sampler._preplanned
    expected = ref_step([3, 4], logits, None)
    assert torch.equal(actual, expected)

    # Mid-request config update on resident row (marks row stale and applies new ras_tau/temperature).
    new_req_cfg = RASSamplingReqConfig(
        temperature=0.0, top_k=5, top_p=0.5, repetition_window=4, ras_mode=True, ras_tau=0.0,
    )
    new_req_cfg.apply_conductor_config(seed=999, resource_key=LM_SAMPLER)
    cfgs[2] = new_req_cfg
    node.sampler.ingest_request(2, new_req_cfg)
    logits = torch.randn(2, vocab, generator=gen) * 2
    actual, ctx = node.decode([1, 2], logits, blocked=None, slot=1, bs=2)
    assert 2 in ctx.resident_plan.taken
    assert torch.equal(actual, ref_step([1, 2], logits, None))
    # Trash row (slot 0) in GenerationHistory must remain clean despite padding rows.
    assert node.sampler._history.count[0].item() == 0
    assert (node.sampler._history.tokens[0] == -1).all()


# ---------------------------------------------------------------------------
# kept_rids, rebuild_rids, chunk_tokens_after, slot reuse, and isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["eager", "captured", "resident"])
def test_ras_kept_rids_rebuild_and_chunk_tokens_after(mode, monkeypatch):
    width, vocab = 4, 11
    cfg_a = _config(11, 4, 0.25, 1.0, 0.8)
    cfg_b = _config(22, 4, 0.25, 1.0, 0.8)
    if mode == "resident":
        node = _RASResidentNode(max_bs=2, max_window=width, num_kv_pages=8, monkeypatch=monkeypatch)
        node.sampler.ingest_request("a", cfg_a)
        node.sampler.ingest_request("b", cfg_b)
        node.kv.ingest_request("a", KVReqConfig())
        node.kv.ingest_request("b", KVReqConfig())
        res = node.sampler
    else:
        res = _resource(width)
        if mode == "captured":
            res.build_accelerator_graph_buffers([SimpleNamespace(slot=0)], max_bs=2, max_seq_len=1)
        res.ingest_request("a", cfg_a)
        res.ingest_request("b", cfg_b)

    def run_step(rids, logits, kept=None, rebuild=frozenset(), chunk_after=None):
        if mode == "resident":
            return node.decode(rids, logits, kept=kept, rebuild_rids=rebuild, chunk_tokens_after=chunk_after)[0]
        lease = (
            SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=2, num_tokens=2))
            if mode == "captured" else None
        )
        ctx = StepContext(
            request_ids=tuple(rids), graph_walk="decode", slot=0, capture=False,
            slot_lease=lease, rebuild_rids=frozenset(rebuild), chunk_tokens_after=dict(chunk_after or {}),
        )
        if mode == "captured":
            ctx.set_padded_rids(tuple(rids))
        smp = SamplerStep(kept_rids=kept)
        res.plan(smp, ctx)
        out = res.sample(list(rids), logits)
        res.commit(smp, ctx)
        return out

    # Step 1: keep both
    run_step(["a", "b"], torch.tensor([[5.0] + [0.0] * (vocab - 1), [0.0, 5.0] + [0.0] * (vocab - 2)]))
    _, tok_before, cnt_before = res._history.read(["a", "b"])
    assert cnt_before.tolist() == [1, 1]

    # Step 2: drop "b" via kept_rids={"a"}
    run_step(["a", "b"], torch.randn(2, vocab), kept=frozenset({"a"}))
    _, tok_after_kept, cnt_after_kept = res._history.read(["a", "b"])
    assert cnt_after_kept.tolist() == [2, 1]
    assert torch.equal(tok_after_kept[1], tok_before[1])

    # Step 3: drop "a" via rebuild_rids={"a"} and "b" via chunk_tokens_after={"b": 1}
    run_step(["a", "b"], torch.randn(2, vocab), rebuild=frozenset({"a"}), chunk_after={"b": 1})
    _, tok_after_drop_all, cnt_after_drop_all = res._history.read(["a", "b"])
    assert cnt_after_drop_all.tolist() == [2, 1]
    assert torch.equal(tok_after_drop_all[0], tok_after_kept[0])
    assert torch.equal(tok_after_drop_all[1], tok_before[1])

    # Step 4: drop all via empty kept_rids=frozenset()
    run_step(["a", "b"], torch.randn(2, vocab), kept=frozenset())
    _, _, cnt_after_empty = res._history.read(["a", "b"])
    assert cnt_after_empty.tolist() == [2, 1]


@pytest.mark.parametrize("mode", ["eager", "captured"])
def test_ras_slot_reuse_and_active_reingest_isolation(mode):
    width, vocab = 4, 9
    res, fresh = _resource(width), _resource(width)
    if mode == "captured":
        res.build_accelerator_graph_buffers([SimpleNamespace(slot=0)], max_bs=2, max_seq_len=1)
        fresh.build_accelerator_graph_buffers([SimpleNamespace(slot=0)], max_bs=2, max_seq_len=1)

    def step_one(r, rid, logits):
        if mode == "eager":
            return r.sample([rid], logits)
        ctx = StepContext(
            request_ids=(rid,), graph_walk="decode", slot=0, capture=False,
            slot_lease=SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=2, num_tokens=2)),
        )
        ctx.set_padded_rids((rid, -1))
        smp = SamplerStep()
        r.plan(smp, ctx)
        padded = torch.cat((logits, torch.zeros(1, logits.shape[-1])))
        out = r.sample([rid, -1], padded)[:1]
        r.commit(smp, ctx)
        return out

    cfg1 = _config(7, 4, 0.25, 0.0, 0.0)
    res.ingest_request("r1", cfg1)
    slot_r1 = res._history._rid_to_slot["r1"]
    # Wrap ring (> width=4 steps) with repeated token 0 to fill r1's slot with 0s and redraws.
    hot0 = torch.tensor([[10.0] + [0.0] * (vocab - 1)])
    for _ in range(6):
        step_one(res, "r1", hot0)
    assert res._history.count[slot_r1].item() == 6

    # Active config-only re-ingest must preserve existing history while applying the new config.
    cfg1_updated = _config(7, 4, 0.5, 0.0, 0.0)
    res.ingest_request("r1", cfg1_updated)
    assert res._history._rid_to_slot["r1"] == slot_r1
    assert res._history.count[slot_r1].item() == 6

    # remove_request -> ingest_request("r2") reuses the freed slot and resets count=0, tokens=-1.
    res.remove_request("r1")
    cfg2 = _config(7, 4, 0.25, 0.0, 0.0)
    res.ingest_request("r2", cfg2)
    fresh.ingest_request("r2", cfg2)
    assert res._history._rid_to_slot["r2"] == slot_r1
    for step_idx in range(6):
        tok_recycled = step_one(res, "r2", hot0)
        tok_fresh = step_one(fresh, "r2", hot0)
        assert tok_recycled.item() == tok_fresh.item()
        if step_idx == 0:
            # First step must NOT redraw from r1's old history (token 0 wins).
            assert tok_recycled.item() == 0

    # Slot 0 (trash row) must remain clean even when capture/unregistered rows run through gather_step/scatter_step.
    assert res._history.count[0].item() == 0
    assert (res._history.tokens[0] == -1).all()
    if mode == "captured":
        res._history.gather_step(["unregistered_ghost"], 2)
        res._history.step_count[:1].fill_(99)
        res._history.step_tokens[:1].fill_(5)
        res._history.scatter_step()
        assert res._history.count[0].item() == 0
        assert (res._history.tokens[0] == -1).all()


# ---------------------------------------------------------------------------
# Edge cases: temperature=0 / 1e-12, bf16/fp16, top_p/top_k bounds, blocked=None, lean/greedy
# ---------------------------------------------------------------------------


def test_ras_edge_cases_dtype_tiny_temp_bounds_blocked_none_and_greedy_classification():
    # samples_greedy & lean_capture classification
    greedy_ras_cfg = SamplingConfig(
        temperature=0.0, top_k=25, top_p=0.0, repetition_window=0, ras_mode=True, ras_tau=0.0,
    )
    assert samples_greedy(greedy_ras_cfg) is False
    res_greedy_cap = SamplerResource(
        vocab_size=None, enable_repetition_penalty=False, device=CPU,
        max_repetition_window=10, enable_blocked_tokens=True, enable_ras=True, greedy_capture=True,
    )
    assert res_greedy_cap.lean_capture is False

    # temperature=0.0: greedy first draw, stochastic redraw when repeated >= window * tau
    res = _resource(4)
    cfg_zero = _config(77, 4, 0.25, 0.0, 0.0)
    res.ingest_request("z", cfg_zero)
    logits = torch.tensor([[10.0, 5.0, 1.0, 0.5]])
    assert res.sample(["z"], logits).item() == 0  # first draw greedy argmax=0
    second = res.sample(["z"], logits).item()     # count=1, repeat=1 >= 4*0.25=1 -> stochastic redraw != 0
    assert second != 0

    # tiny positive temperature (< GREEDY_TEMPERATURE = 1e-5) matches temperature=0.0 without NaN
    res_tiny = _resource(4)
    cfg_tiny = _config(77, 4, 0.25, 1e-12, 0.0)
    res_tiny.ingest_request("z", cfg_tiny)
    assert res_tiny.sample(["z"], logits).item() == 0
    assert res_tiny.sample(["z"], logits).item() == second

    # bfloat16 and float16 logits match float32-upcast reference without NaN
    for dtype in (torch.bfloat16, torch.float16):
        res_dt = _resource(4)
        cfg_dt = _config(55, 4, 0.5, 0.8, 0.8)
        res_dt.ingest_request("d", cfg_dt)
        raw_logits = torch.randn(1, 17, generator=torch.Generator().manual_seed(12)).to(dtype)
        actual = res_dt.sample(["d"], raw_logits, blocked_tokens=None)
        expected = old_ras_sample(
            raw_logits.float(),
            seed=torch.tensor([cfg_dt.seed]),
            step=torch.tensor([0]),
            history=torch.full((1, 4), -1, dtype=torch.long),
            temperature=torch.tensor([cfg_dt.temperature]),
            top_p=torch.tensor([cfg_dt.top_p]),
            top_k=torch.tensor([cfg_dt.top_k]),
            window=torch.tensor([cfg_dt.repetition_window]),
            repeat_threshold=torch.tensor([cfg_dt.repetition_window * cfg_dt.ras_tau]),
            blocked=None,
        )
        assert torch.equal(actual, expected)

    # top_k=1 and top_k=128 (with top_p=0.0 and top_p=1.0), partial & wrapped ring, blocked=None
    res_edge = _resource(3)
    c1 = RASSamplingReqConfig(temperature=1.0, top_k=1, top_p=0.0, repetition_window=0, ras_mode=True, ras_tau=0.0)
    c2 = RASSamplingReqConfig(temperature=0.9, top_k=128, top_p=1.0, repetition_window=3, ras_mode=True, ras_tau=0.34)
    c1.apply_conductor_config(seed=10, resource_key=LM_SAMPLER)
    c2.apply_conductor_config(seed=20, resource_key=LM_SAMPLER)
    res_edge.ingest_request("c1", c1)
    res_edge.ingest_request("c2", c2)
    hist_c2 = torch.full((3,), -1, dtype=torch.long)
    gen = torch.Generator().manual_seed(99)
    for step_i in range(6):  # covers partial ring (step 0..2) and wrapped ring (step 3..5)
        lg = torch.randn(2, 150, generator=gen)
        got = res_edge.sample(["c1", "c2"], lg, blocked_tokens=None)
        exp = old_ras_sample(
            lg,
            seed=torch.tensor([c1.seed, c2.seed]),
            step=torch.tensor([step_i, step_i]),
            history=torch.stack([torch.full((3,), -1, dtype=torch.long), hist_c2]),
            temperature=torch.tensor([c1.temperature, c2.temperature]),
            top_p=torch.tensor([c1.top_p, c2.top_p]),
            top_k=torch.tensor([c1.top_k, c2.top_k]),
            window=torch.tensor([0, 3]),
            repeat_threshold=torch.tensor([0.0, 3 * 0.34]),
            blocked=None,
        )
        assert torch.equal(got, exp)
        hist_c2 = torch.cat((hist_c2[1:], exp[1:2]))


# ---------------------------------------------------------------------------
# Full validation refusal matrix & CosyVoice3 integration
# ---------------------------------------------------------------------------


def test_ras_validation_refusal_matrix_and_cosyvoice3_integration(tmp_path, monkeypatch):
    # 1. SamplerSpec invariants
    for bad_spec in (
        dict(enable_repetition_penalty=True),
        dict(enable_min_p=True),
        dict(enable_logprobs=True),
        dict(enable_processed_logprobs=True),
        dict(enable_top_p_first=True),
        dict(min_tokens_stop_ids=(1,)),
        dict(enable_repetition_penalty=False, max_repetition_window=-1),
        dict(enable_repetition_penalty=False, max_repetition_window=True),
    ):
        kw = {"resource_key": "s", "nodes": {"n"}, "vocab_size": None, "enable_ras": True}
        if "enable_repetition_penalty" not in bad_spec:
            kw["enable_repetition_penalty"] = False
        kw.update(bad_spec)
        with pytest.raises(ValueError):
            SamplerSpec(**kw)

    spec = SamplerSpec(
        resource_key=LM_SAMPLER, nodes={"llm"}, vocab_size=None,
        enable_repetition_penalty=False, max_repetition_window=10,
        enable_blocked_tokens=True, enable_ras=True,
    )
    with pytest.raises(ValueError, match="enable_logprobs"):
        spec.apply_yaml_overrides(enable_logprobs=True)

    # 2. SamplerResource.__init__ invariants
    with pytest.raises(ValueError, match="enable_ras"):
        SamplerResource(vocab_size=None, enable_repetition_penalty=True, device=CPU, enable_ras=True)
    with pytest.raises(ValueError, match="enable_ras"):
        SamplerResource(
            vocab_size=None, enable_repetition_penalty=False, device=CPU,
            enable_ras=True, min_tokens_stop_ids=(1,),
        )

    # 3. SamplingReqConfig / RASSamplingReqConfig.validate(spec=None) & validate(spec)
    plain_spec = SamplerSpec(resource_key="s", nodes={"n"}, vocab_size=100, enable_repetition_penalty=False)
    for bad_window in (-1, True, 1.5, None):
        with pytest.raises(ValueError):
            SamplingReqConfig(repetition_window=bad_window).validate()
        with pytest.raises(ValueError):
            RASSamplingReqConfig(top_k=25, repetition_window=bad_window).validate()

    bad_ras_cases = [
        dict(top_k=0),
        dict(top_k=-1),
        dict(top_k=129),
        dict(top_k=True),
        dict(top_k=None),
        dict(top_k=25, ras_tau=-0.1),
        dict(top_k=25, ras_tau=1.1),
        dict(top_k=25, ras_tau=math.nan),
        dict(top_k=25, ras_tau=math.inf),
        dict(top_k=25, ras_tau=True),
        dict(top_k=25, repetition_penalty=1.2),
        dict(top_k=25, min_p=0.1),
        dict(top_k=25, return_logprobs=True),
        dict(top_k=25, min_tokens=1),
        dict(top_k=25, top_p_first=True),
        dict(top_k=25, top_p_min_keep=2),
        dict(top_k=25, top_p_min_keep=True),
        dict(top_k=25, ras_mode=False),
    ]
    for kwargs in bad_ras_cases:
        with pytest.raises(ValueError):
            RASSamplingReqConfig(**kwargs).validate(None)
        with pytest.raises(ValueError):
            RASSamplingReqConfig(**kwargs).validate(spec)

    with pytest.raises(ValueError, match="history"):
        RASSamplingReqConfig(top_k=25, repetition_window=11).validate(spec)
    with pytest.raises(ValueError, match="RAS"):
        RASSamplingReqConfig(top_k=25, repetition_window=0).validate(plain_spec)

    # 4. Direct SamplerResource.ingest_request bypass refusal (and no stale state left behind)
    res = _resource(10)
    direct_bad = [
        None,
        SamplingReqConfig(),
        RASSamplingReqConfig(top_k=0),
        RASSamplingReqConfig(top_k=129),
        RASSamplingReqConfig(top_k=25, ras_tau=math.nan),
        RASSamplingReqConfig(top_k=25, ras_tau=-0.1),
        RASSamplingReqConfig(top_k=25, ras_tau=1.1),
        RASSamplingReqConfig(top_k=25, temperature=-1.0),
        RASSamplingReqConfig(top_k=25, temperature=math.nan),
        RASSamplingReqConfig(top_k=25, top_p=-0.1),
        RASSamplingReqConfig(top_k=25, top_p=1.1),
        RASSamplingReqConfig(top_k=25, repetition_penalty=1.2),
        RASSamplingReqConfig(top_k=25, min_p=0.1),
        RASSamplingReqConfig(top_k=25, return_logprobs=True),
        RASSamplingReqConfig(top_k=25, min_tokens=1),
        RASSamplingReqConfig(top_k=25, top_p_first=True),
        RASSamplingReqConfig(top_k=25, top_p_min_keep=2),
        RASSamplingReqConfig(top_k=25, repetition_window=-1),
        RASSamplingReqConfig(top_k=25, repetition_window=11),
    ]
    for idx, bad_cfg in enumerate(direct_bad):
        rid = f"bad_{idx}"
        with pytest.raises(ValueError):
            res.ingest_request(rid, bad_cfg)
        assert rid not in res._sampler._sampling_config
        assert rid not in res._history._rid_to_slot

    non_ras_res = SamplerResource(vocab_size=16, enable_repetition_penalty=False, device=CPU)
    with pytest.raises(ValueError, match="RAS"):
        non_ras_res.ingest_request("r_bad", RASSamplingReqConfig(top_k=5))
    assert "r_bad" not in non_ras_res._sampler._sampling_config

    # 5. CosyVoice3Model -> get_request_resource_configs -> apply_conductor_config -> validate -> ingest
    monkeypatch.setattr(CosyVoice3Model, "_build_text_tokenizer", lambda self: SimpleNamespace())
    model = CosyVoice3Model(model_path_hf=str(tmp_path))
    model_specs = {s.resource_key: s for s in model.get_node_resources()}
    req_cfgs = model.get_request_resource_configs(
        ["text"], ["audio"], {"temperature": 0.7, "top_p": 0.0, "top_k": 15, "ras_window": 6, "ras_tau": 0.2},
    )
    ras_cfg = req_cfgs[LM_SAMPLER]
    assert isinstance(ras_cfg, RASSamplingReqConfig)
    ras_cfg.apply_conductor_config(seed=12345, resource_key=LM_SAMPLER)
    assert ras_cfg.seed == 12345  # direct request seed preserved (not derive_seed)
    ras_cfg.validate(model_specs[LM_SAMPLER])
    cv_sampler = SamplerResource.build(
        model_specs[LM_SAMPLER], SimpleNamespace(device=CPU, joint_comm_group=None),
    )
    cv_sampler.ingest_request("cv_ok", ras_cfg)

    # 6. LMSubmodule caches the [1, V] vocab ids tensor per (device, vocab_size).
    lm_sub = LMSubmodule(None, model.config)
    v_ids_1 = lm_sub._vocab_ids_for(model.config.lm.speech_vocab_size, CPU)
    v_ids_2 = lm_sub._vocab_ids_for(model.config.lm.speech_vocab_size, CPU)
    assert v_ids_1 is v_ids_2
    assert v_ids_1.shape == (1, model.config.lm.speech_vocab_size)


# ---------------------------------------------------------------------------
# CPU dispatch overhead & op-count regression checks
# ---------------------------------------------------------------------------


def test_ras_cpu_dispatch_overhead_and_op_counts(monkeypatch):
    # 1. Captured RAS steps (both host-planned and resident) perform zero offset gather/scatter.
    res = _resource(4)
    res.build_accelerator_graph_buffers([SimpleNamespace(slot=0)], max_bs=4, max_seq_len=1)
    res.ingest_request("a", _config(7, 4, 0.25, 1.0, 0.8))
    res.ingest_request("b", _config(8, 4, 0.25, 1.0, 0.8))

    offset_ops: list[str] = []
    real_gather = res._cg_buffers.offset.gather
    real_scatter = res._cg_buffers.offset.scatter
    monkeypatch.setattr(
        res._cg_buffers.offset, "gather",
        lambda *a, **k: (offset_ops.append("gather"), real_gather(*a, **k))[1],
    )
    monkeypatch.setattr(
        res._cg_buffers.offset, "scatter",
        lambda *a, **k: (offset_ops.append("scatter"), real_scatter(*a, **k))[1],
    )

    hist_copies: list[int] = []
    real_hist_copy = res._history._stager.copy_
    monkeypatch.setattr(
        res._history._stager, "copy_",
        lambda *a, **k: (hist_copies.append(1), real_hist_copy(*a, **k))[1],
    )

    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=2, num_tokens=2))
    logits = torch.randn(2, 13)
    for _ in range(4):
        ctx = StepContext(request_ids=("a", "b"), graph_walk="decode", slot=0, capture=False, slot_lease=lease)
        ctx.set_padded_rids(("a", "b"))
        res.plan(SamplerStep(), ctx)
        res.sample(["a", "b"], logits)
        res.commit(SamplerStep(), ctx)

    assert offset_ops == []
    # Steady-batch GenerationHistory.gather_step issues only 1 stager copy on the first step, 0 thereafter.
    assert len(hist_copies) == 1

    # Batch/registration change invalidates GenerationHistory.gather_step cache cleanly.
    res.ingest_request("c", _config(9, 4, 0.25, 1.0, 0.8))
    ctx_c = StepContext(request_ids=("a", "c"), graph_walk="decode", slot=0, capture=False, slot_lease=lease)
    ctx_c.set_padded_rids(("a", "c"))
    res.plan(SamplerStep(), ctx_c)
    res.sample(["a", "c"], logits)
    res.commit(SamplerStep(), ctx_c)
    assert len(hist_copies) == 2

    # Resident RAS steps also perform zero offset gather/scatter and skip steady-batch cell staging.
    rnode = _RASResidentNode(max_bs=2, max_window=4, num_kv_pages=64, monkeypatch=monkeypatch)
    rnode.ingest("ra", _config(11, 4, 0.25, 1.0, 0.8))
    rnode.ingest("rb", _config(12, 4, 0.25, 1.0, 0.8))
    r_offset_ops: list[str] = []
    real_rg = rnode.sampler._cg_buffers.offset.gather
    real_rs = rnode.sampler._cg_buffers.offset.scatter
    monkeypatch.setattr(
        rnode.sampler._cg_buffers.offset, "gather",
        lambda *a, **k: (r_offset_ops.append("gather"), real_rg(*a, **k))[1],
    )
    monkeypatch.setattr(
        rnode.sampler._cg_buffers.offset, "scatter",
        lambda *a, **k: (r_offset_ops.append("scatter"), real_rs(*a, **k))[1],
    )
    for step_i in range(3):
        rnode.decode(["ra", "rb"], logits, slot=step_i % 2, bs=2)
    assert r_offset_ops == []

    # 2. Eager _sample_ras issues zero torch.tensor(..., device=...) or zeros_like allocations.
    eager = _resource(4)
    eager.ingest_request("a", _config(7, 4, 0.25, 1.0, 0.8))
    eager.ingest_request("b", _config(8, 4, 0.25, 1.0, 0.8))
    blocking_tensor_calls = []
    zeros_like_calls = []
    real_tensor = torch.tensor
    real_zeros_like = torch.zeros_like
    monkeypatch.setattr(
        torch, "tensor",
        lambda *a, **k: (
            blocking_tensor_calls.append((a, k)) if k.get("device") is not None else None,
            real_tensor(*a, **k),
        )[1],
    )
    monkeypatch.setattr(
        torch, "zeros_like",
        lambda *a, **k: (zeros_like_calls.append(a), real_zeros_like(*a, **k))[1],
    )
    eager.sample(["a", "b"], logits, blocked_tokens=None)
    assert blocking_tensor_calls == []
    assert zeros_like_calls == []

    # 3. Non-RAS SamplerResource pays zero RAS allocations/calls.
    with monkeypatch.context() as mp:
        mp.setattr(sampler_utils, "sample_tokens", lambda *, logits, **k: torch.argmax(logits, dim=-1))
        std_res = SamplerResource(vocab_size=16, enable_repetition_penalty=False, device=CPU)
        std_res.build_accelerator_graph_buffers([SimpleNamespace(slot=0)], max_bs=2, max_seq_len=1)
        std_res.ingest_request("s1", SamplingReqConfig(temperature=0.7, top_k=5, top_p=0.9))
        assert std_res._history is None and std_res._ras_stager is None
        assert std_res._cg_buffers.ras_threshold is None
        ras_calls = []
        mp.setattr(sampler_resource, "ras_sample", lambda *a, **k: ras_calls.append(1))
        std_ctx = StepContext(request_ids=("s1",), graph_walk="decode", slot=0, capture=False)
        std_res.plan(SamplerStep(), std_ctx)
        std_res.sample(["s1"], torch.randn(1, 16))
        std_res.commit(SamplerStep(), std_ctx)
        assert ras_calls == []

    # 4. Wall-clock CPU sanity check for bs=1, 4, 32 (bounded dispatch latency).
    for bs in (1, 4, 32):
        rids = [f"r{i}" for i in range(bs)]
        bench = _resource(10)
        bench.build_accelerator_graph_buffers([SimpleNamespace(slot=0)], max_bs=bs, max_seq_len=1)
        for i, rid in enumerate(rids):
            bench.ingest_request(rid, _config(100 + i, 10, 0.1, 1.0, 0.8))
        b_lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=bs, num_tokens=bs))
        b_logits = torch.randn(bs, 64)
        b_ctx = StepContext(request_ids=tuple(rids), graph_walk="decode", slot=0, capture=False, slot_lease=b_lease)
        b_ctx.set_padded_rids(tuple(rids))
        t0 = time.perf_counter()
        for _ in range(50):
            bench.plan(SamplerStep(), b_ctx)
            bench.sample(rids, b_logits)
            bench.commit(SamplerStep(), b_ctx)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"CPU RAS dispatch at bs={bs} took {elapsed:.2f}s for 50 steps"
