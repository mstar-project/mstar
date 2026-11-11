"""A draft model one token behind its target (`DrafterNeeds.lag`): the one declaration and where the
engine carries it.

The engine copies the drafter's lag into a verify step's KV writes of its layer set, positions and
attention binding (`drafter_step`, from `engine_step`, so a capture and each step it replays carry the
same), and no other walk lags. Startup finds where it goes (`draft_lag`). Each verify step sinks the
drafter's row 0 unless the target accepted every draft last step, and sets that flag for the next
(`draft_verify_step`), all on the device.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.draft.drafter import (
    DRAFT_KEEP_FIRST,
    Drafter,
    DrafterNeeds,
    TargetOutput,
    VerifyNode,
    VerifyResult,
    draft_verify_step,
    verify_writes,
)
from mstar.engine.resources.draft.node import DrafterShare, DraftLag, drafter_step
from mstar.engine.resources.draft.startup import draft_lag
from mstar.engine.resources.kv.config import KVLayerSet, KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.step import Lag, Segment, SubmoduleStep

NODE, WALK, K = "LLM", "decode", 3
DRAFT = KVLayerSet("kv", "draft", 2, 2, 8)
LAYOUT = Lag(1, (2, 1, 1))
LAG = DraftLag(WALK, LAYOUT, "kv", "draft", "rope", "draft_attn")


def lag_step(step, walk, lag):
    """``step`` with only a lag for its drafter's share (`drafter_step`)."""
    share = DrafterShare(
        verify_walk=WALK, sampler_key="sampler", num_draft_tokens=3, prompt_walks=frozenset({"prefill"}),
        prompt_writes=frozenset(), verify_writes=frozenset(), prompt_steps={}, verify_steps={}, lag=lag,
    )
    return drafter_step(step, walk, share)


def _step() -> SubmoduleStep:
    return SubmoduleStep(segments=[Segment(1, "main", K + 1)], steps={
        "kv": KVStep(), "rope": PositionStep(), "attn": AttentionStep(verify=True),
        "draft_attn": AttentionStep(verify=True), "sampler": SamplerStep(),
    })


def test_the_verify_walk_carries_the_one_lag_and_no_other_walk_does():
    lagged = lag_step(_step(), WALK, LAG)
    assert lagged.steps["kv"].lag == {"draft": LAYOUT}
    assert lagged.steps["rope"].lag == LAYOUT and lagged.steps["draft_attn"].lag == LAYOUT
    # the target's own attention is untouched
    assert lagged.steps["attn"].lag is None
    prompt = lag_step(_step(), "prefill", LAG)
    assert prompt.steps["kv"].lag == {} and prompt.steps["rope"].lag is None and prompt.steps["draft_attn"].lag is None
    assert lag_step(_step(), WALK, None).steps["kv"].lag == {}


def test_a_model_never_declares_a_lag_and_a_verify_step_carries_every_step_it_rides_on():
    step = _step()
    declared = SubmoduleStep(segments=step.segments, steps={**step.steps, "rope": PositionStep(lag=LAYOUT)})
    with pytest.raises(RuntimeError, match="only the engine sets"):
        lag_step(declared, "prefill", LAG)
    missing = SubmoduleStep(segments=step.segments, steps={k: v for k, v in step.steps.items() if k != "draft_attn"})
    with pytest.raises(RuntimeError, match="no 'draft_attn' step"):
        lag_step(missing, WALK, LAG)


def test_only_a_verify_step_stores_a_lagged_drafters_keep_flag():
    """A prompt step writes every prompt row unlagged, so it stores the carry and not the keep flag;
    a verify step stores both. (The engine's share is built from the same two functions, which
    the draft model's share test pins on a built engine.)"""
    from mstar.engine.resources.draft.drafter import DRAFT_CARRY, CarrySpec, carry_writes

    needs = DrafterNeeds(kv=DRAFT, lag=LAYOUT, carry=CarrySpec(4, torch.float32))
    share = DrafterShare(
        verify_walk=WALK, sampler_key="sampler", num_draft_tokens=K, prompt_walks=frozenset({"prefill"}),
        prompt_writes=carry_writes(needs), verify_writes=verify_writes(needs), prompt_steps={}, verify_steps={},
        lag=LAG,
    )
    prompt = drafter_step(SubmoduleStep(segments=[Segment(1, "main", 5)], steps={"sampler": SamplerStep()}),
                          "prefill", share)
    verify = drafter_step(_step(), WALK, share)
    assert prompt.steps["sampler"].loopback_writes == {DRAFT_CARRY}
    assert verify.steps["sampler"].loopback_writes == {DRAFT_CARRY, DRAFT_KEEP_FIRST}


def _specs(binding: bool = True, positions: int = 1) -> dict:
    pool = PagedKVConfig(num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64, max_num_pages=8, page_size=4)
    pool.join(DRAFT)
    specs = {"kv": KVSpec("kv", {NODE}, pool),
             "attn": AttentionSpec("attn", {NODE}, AttentionConfig(kv_cache="kv"))}
    if binding:
        specs["draft_attn"] = AttentionSpec("draft_attn", {NODE}, AttentionConfig(kv_cache="kv", layer_set="draft"))
    for i in range(positions):
        specs[f"rope{i or ''}"] = PositionSpec(f"rope{i or ''}", {NODE}, PositionConfig(kv_cache="kv"))
    return specs


def _decl(**needs) -> VerifyNode:
    return VerifyNode(NODE, WALK, SimpleNamespace(num_draft_tokens=K, block=K + 1), DrafterNeeds(**needs))


def test_startup_finds_where_the_lag_goes_or_says_why_it_cant():
    assert draft_lag(_decl(), _specs()) == (None, [])
    assert draft_lag(_decl(kv=DRAFT, lag=LAYOUT), _specs()) == (LAG, [])
    for decl, specs, why in [
        (_decl(kv=DRAFT, lag=Lag(2, (2, 1, 1))), _specs(), "lags its target by one"),
        (_decl(kv=DRAFT, lag=Lag(1, (2, 1))), _specs(), "don't split the verify block of 4"),
        (_decl(lag=LAYOUT), _specs(), "joins no layers"),
        (_decl(kv=DRAFT, lag=LAYOUT), _specs(binding=False), "AttentionConfig\\(layer_set='draft'\\)"),
        (_decl(kv=DRAFT, lag=LAYOUT), _specs(positions=2), "declares 2"),
    ]:
        lag, problems = draft_lag(decl, specs)
        assert lag is None and len(problems) == 1
        assert re.search(why, problems[0]), problems[0]
    assert DRAFT_KEEP_FIRST in verify_writes(DrafterNeeds(kv=DRAFT, lag=LAYOUT))
    assert DRAFT_KEEP_FIRST not in verify_writes(DrafterNeeds(kv=DRAFT))


class _Sampler:
    """The verifying sampler's share of a step, with per-row loop-back state."""
    verify_draft_tokens = K

    def __init__(self, keep: list[int], accepted: list[int]):
        self.state = {DRAFT_KEEP_FIRST: torch.tensor(keep).view(-1, 1)}
        self.accepted = torch.tensor(accepted)

    def loopback_state(self, name, rids):
        return self.state[name]

    def store_loopback_state(self, name, rids, values):
        self.state[name] = values.clone()

    def agree(self, drafts):
        return drafts

    def verify(self, rids, logits, drafts):
        n = drafts.shape[0]
        return VerifyResult(torch.zeros(n, K + 1, dtype=torch.long), self.accepted)


class _KV:
    def __init__(self):
        self.sunk = None

    def sink_lagged_rows(self, keep):
        self.sunk = keep.clone()


class _Drafter(Drafter):
    def __init__(self, kv):
        self.needs = DrafterNeeds(kv=DRAFT, lag=LAYOUT)
        self.kv = kv
        self.saw = None

    def propose(self, inputs):
        # the select ran before any of the drafter's writes
        self.saw = self.kv.sunk
        return torch.zeros(inputs.input_ids.shape[0], K, dtype=torch.long)


def test_each_verify_step_sinks_row_0_unless_every_draft_was_accepted_and_sets_the_next_flag():
    kv = _KV()
    sampler, drafter = _Sampler(keep=[1, 0, 0], accepted=[K, 1, 0]), _Drafter(kv)
    draft_verify_step(sampler, [1, 2, 3], torch.zeros(3, dtype=torch.long),
                      lambda block: TargetOutput(torch.zeros(block.shape[0], 5)), drafter, {"kv": kv})
    assert drafter.saw.tolist() == [True, False, False]
    assert sampler.state[DRAFT_KEEP_FIRST].view(-1).tolist() == [1, 0, 0]


def test_each_fa3_pass_attends_through_its_own_extent_host_planned_or_resident():
    """A lagged pass's FA3 wrapper (`FA3PagedWrapper(plan_span=, kv_extent=)`): its own queries a row over
    the verify block's pages, through ``kv_extent`` tokens past the row's stored length; host-planned from
    the KV plan's block lengths, or resident from the device length."""
    from mstar.engine.resources.attn.fa3_paged import FA3PagedWrapper
    from mstar.engine.resources.kv.plan import SequenceView, build_paged_indptrs, lagged_pass_views

    page, stored = 4, [16, 9]
    views = [SequenceView(rid, "main", list(range(10 * i + 1, 10 * i + 1 + -(-(n + K + 1) // page))), n + K + 1,
                          K + 1) for i, (rid, n) in enumerate(zip([1, 2], stored, strict=True))]
    indptrs = build_paged_indptrs(views, page)
    for p, (queries, extent) in enumerate(zip(LAYOUT.passes, (1, 2, 3), strict=True)):
        assert LAYOUT.extent(p) == extent
        wrapper = FA3PagedWrapper(page_size=page, table_width=16, device=torch.device("cpu"), span=queries,
                                  plan_span=K + 1, kv_extent=extent)
        wrapper.plan(**indptrs.to_kwargs_dict())
        assert wrapper._seqused.tolist() == [n + extent for n in stored]
        assert wrapper._qo_indptr_buf.tolist() == [0, queries, 2 * queries]
        # resident: the same lengths off the device's
        wrapper._seqused = torch.zeros(2, dtype=torch.int32)
        wrapper._page_table = torch.zeros(2, wrapper.width, dtype=torch.int32)
        wrapper.gather_resident(torch.zeros(8, 32, dtype=torch.int32), torch.tensor([1, 2]),
                                torch.tensor(stored, dtype=torch.int32), 32, 0)
        assert wrapper._seqused.tolist() == [n + extent for n in stored]
        # FlashInfer's host plan of the pass: the same extent, its own queries
        lagged = lagged_pass_views(views, K + 1, extent, queries, page)
        assert [v.length for v in lagged] == [n + extent for n in stored]
        assert [v.to_compute for v in lagged] == [queries, queries]
        assert all(len(v.page_idxs) == -(-v.length // page) for v in lagged)


def test_a_step_carries_exactly_the_declared_lag_at_capture_and_after():
    from mstar.engine.resources.draft.node import check_lag

    check_lag(lag_step(_step(), WALK, LAG), WALK, LAG)
    check_lag(_step(), "prefill", LAG)
    check_lag(_step(), WALK, None)
    with pytest.raises(RuntimeError, match="'kv' lags None"):
        check_lag(_step(), WALK, LAG)
    other = SubmoduleStep(segments=_step().segments, steps={**lag_step(_step(), WALK, LAG).steps,
                                                            "attn": AttentionStep(verify=True, lag=LAYOUT)})
    with pytest.raises(RuntimeError, match="'attn' lags"):
        check_lag(other, WALK, LAG)


def test_an_asynchronous_lagged_drafter_on_flashinfer_is_refused_for_what_it_needs():
    from mstar.engine.resources.draft.config import DraftMode
    from mstar.engine.resources.draft.startup import resolve_mode

    specs = _specs()
    specs["draft_attn"] = AttentionSpec("draft_attn", {NODE}, AttentionConfig(
        kv_cache="kv", layer_set="draft", verify_backend="flashinfer"))
    decl = _decl(kv=DRAFT, lag=LAYOUT)._replace(config=SimpleNamespace(
        num_draft_tokens=K, block=K + 1, mode=DraftMode.ASYNCHRONOUS))
    with pytest.raises(ValueError, match="lagged drafter's passes are planned on the host"):
        resolve_mode(decl, specs, True)


def _paged_attend(q, k_pages, v_pages, table, seqused, qo, page):
    """FA3's paged causal attention on the CPU: each row's queries are the last of its ``seqused`` keys
    (bottom-right aligned), its keys read through its page table row."""
    out = []
    for i in range(len(seqused)):
        n, (a, b) = int(seqused[i]), (int(qo[i]), int(qo[i + 1]))
        pages = table[i, : -(-n // page)].long()
        keys = k_pages[pages].reshape(-1, k_pages.shape[-1])[:n]
        values = v_pages[pages].reshape(-1, v_pages.shape[-1])[:n]
        scores = q[a:b] @ keys.T
        rows = torch.arange(b - a)[:, None]
        scores = scores.masked_fill(torch.arange(n)[None, :] > n - (b - a) + rows, float("-inf"))
        out.append(scores.softmax(-1) @ values)
    return torch.cat(out)


def test_each_pass_attends_as_dense_causal_attention_over_the_lagged_history():
    """A draft model's verify block, one row behind the target's and run in passes: each pass's FA3 plan
    (page table, lengths, query offsets) gives each query exactly the keys up to its own position, as dense
    causal attention over the lagged history does."""
    from mstar.engine.resources.attn.fa3_paged import FA3PagedWrapper
    from mstar.engine.resources.kv.plan import SequenceView, build_paged_indptrs

    page, dim, block = 4, 8, K + 1
    gen = torch.Generator().manual_seed(0)
    stored = [9, 16, 3]
    # each row's draft-model history: positions 0 .. L + k - 1 once the block's passes have written them
    history = [(torch.randn(n + block, dim, generator=gen), torch.randn(n + block, dim, generator=gen)) for n in stored]
    k_pages, v_pages, views, first = torch.zeros(64, page, dim), torch.zeros(64, page, dim), [], 1
    for i, (n, (keys, values)) in enumerate(zip(stored, history, strict=True)):
        count = -(-(n + block) // page)
        pages = list(range(first, first + count))
        first += count
        for j in range(n + block):
            k_pages[pages[j // page], j % page], v_pages[pages[j // page], j % page] = keys[j], values[j]
        views.append(SequenceView(i, "main", pages, n + block, block))
    indptrs = build_paged_indptrs(views, page)
    for p, (start, queries) in enumerate(zip(LAYOUT.starts, LAYOUT.passes, strict=True)):
        wrapper = FA3PagedWrapper(page_size=page, table_width=16, device=torch.device("cpu"), span=queries,
                                  plan_span=block, kv_extent=LAYOUT.extent(p))
        wrapper.plan(**indptrs.to_kwargs_dict())
        q = torch.randn(len(stored) * queries, dim, generator=gen)
        got = _paged_attend(q, k_pages, v_pages, wrapper._page_table, wrapper._seqused, wrapper._qo_indptr_buf, page)
        want = []
        for i, (n, (keys, values)) in enumerate(zip(stored, history, strict=True)):
            for r in range(queries):
                at = n - LAYOUT.rows + start + r  # the query's lagged position
                scores = q[i * queries + r] @ keys[: at + 1].T
                want.append(scores.softmax(-1) @ values[: at + 1])
        torch.testing.assert_close(got, torch.stack(want))
