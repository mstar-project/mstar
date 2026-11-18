"""Asynchronous mixed prompt-plus-verify steps on resident tables: the harness.

A mixed step's verify rows run ``k + 1`` tokens a row at lengths only the device knows until their results are
recorded, as a verify step's do (`test_spec_resident.py`); its prompt rows run at lengths the host knows. In
asynchronous mode both kinds go into one captured step: the verify rows read their write slots, positions and
attention lengths off the resident tables, the prompt rows read theirs from the host plan, and the graph advances
the verify rows' length cells by ``accepted + 1``.

The real KV and position resources and the real planner on the CPU, driven as a replay drives them, with the KV
debug checks on. The planner gives the verify rows slots and leaves the prompt rows on the trash row
(`ResidentPlanner.mixed_walks`), and the gathers skip that row (`ResidentBuffers.mixed`).
"""
from __future__ import annotations

import random
import sys

import pytest
import torch

sys.path.insert(0, ".")

from test_resident_planning import _stub_transfer  # noqa: E402,F401
from test_spec_resident import PAGE, SPAN, WALK, _check_step, _Node  # noqa: E402

from mstar.engine.resources import resident as resident_mod  # noqa: E402
from mstar.engine.resources.base import CGSlotKey  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVReqConfig, KVStep  # noqa: E402
from mstar.engine.resources.position.config import PositionStep  # noqa: E402
from mstar.engine.resources.resident import ResidentPlanner  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext, SubmoduleStep  # noqa: E402

K = SPAN - 1
MIXED = "mixed"
PREFILL = "prefill"
CPU = torch.device("cpu")

@pytest.fixture(autouse=True)
def _debug_asserts(monkeypatch):
    # every write and read checked against the pool's own bookkeeping, and the planner's flush read back
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    monkeypatch.setattr(resident_mod, "_DEBUG_ASSERTS", True)


class _MixedNode(_Node):
    """`_Node` whose planner takes the mixed walk's steps, as the engine binds a verifying node's mixed walk."""

    def __init__(self, prompts: dict[int, int], synchronous: bool = False, max_bs: int = 8):
        super().__init__(prompts, synchronous=synchronous, max_bs=max_bs)
        self.planner = ResidentPlanner("n", {WALK}, self.res, CPU, max_bs=max_bs, max_span=SPAN)
        self.planner.mixed_walks = frozenset({MIXED})
        self.planner.pending = self.ledger.pending
        self.planner.unrecorded = self.ledger.unrecorded

    def mixed_step(
        self, rids, prompts: dict[int, int], accepted: dict[int, int], slot: int, bs: int, tokens: int, attend=None,
    ):
        """One mixed step: ``rids``' verify rows, then ``prompts``' rows (new requests of that many tokens), then
        padding rows to ``bs``, in a bucket of ``tokens``; the take-back first, the verify rows' advance in the
        graph, and the step's verify rows alone entered in the ledger. ``attend(captured, ctx)`` runs where an
        attention resource plans and gathers, after the planner's gather."""
        res, planner = self.res, self.planner
        self.take_back(rids)
        for rid in prompts:
            res["kv"].ingest_request(rid, KVReqConfig())
            res["rope"].ingest_request(rid, None)
        rows = (*rids, *prompts)
        padded = rows + tuple(-(i + 1) for i in range(bs - len(rows)))
        walks = {**dict.fromkeys(rids, WALK), **dict.fromkeys(prompts, PREFILL)}
        bucket = BucketKey(graph_walk=MIXED, bs=bs, num_tokens=tokens)
        lease = SlotLease(slot=slot, bucket=bucket)
        ctx = StepContext(request_ids=rows, graph_walk=MIXED, slot=slot, capture=False, slot_lease=lease,
                          request_walks=walks)
        ctx.set_padded_rids(padded)
        segments = (
            *(Segment(r, "main", SPAN) for r in rids), *(Segment(r, "main", n) for r, n in prompts.items()),
            *(Segment(r, "main", 0) for r in padded[len(rows):]),
        )
        kv, rope = KVStep(segments=segments), PositionStep(segments=segments)
        step = SubmoduleStep(steps={"kv": kv}, segments=list(segments))
        assert res["kv"].admit(kv, ctx).ok
        if (bucket, slot) not in self.captured:
            self.captured.add((bucket, slot))
            first = StepContext(request_ids=padded, graph_walk=MIXED, slot=slot, capture=True, slot_lease=lease,
                                request_walks=walks)
            first.plan_results["kv"] = res["kv"].plan(kv, first)
            res["rope"].plan(rope, first)
        captured = planner.capture_step(bucket, slot, step)
        ctx.resident = planner
        planner.plan(step, ctx)
        ctx.plan_results["kv"] = res["kv"].plan(kv, ctx)
        res["rope"].plan(rope, ctx)
        planner.flush(ctx.resident_plan)
        planner.gather(captured)
        if attend is not None:
            attend(captured, ctx)
        state = res["kv"]._static_plan_state(slot, ("main", 0), tokens)
        positions = res["rope"]._static_pos_ids[CGSlotKey(bucket=bucket, slot=slot, label="main")]
        n = len(rids) * SPAN + sum(prompts.values())
        out = {
            "pages": state.token_to_page[:n].tolist(), "offsets": state.token_to_cache[:n].tolist(),
            "positions": positions[:n].tolist(),
        }
        # what the verify writes in the graph: a verify row keeps accepted + 1, a prompt row its whole span
        advance = [accepted[r] + 1 for r in rids] + list(prompts.values()) + [0] * (bs - len(rows))
        captured.buffers.advance.copy_(torch.tensor(advance))
        planner.commit_in_graph(captured)
        res["kv"].commit(kv, ctx)
        res["rope"].commit(rope, ctx)
        planner.commit(step, ctx)
        seq = self.ledger.committed(rids, {"kv": "main", "rope": "main"}, [SPAN] * len(rids))
        for rid, length in prompts.items():
            self.true_len[rid] = length
        return out, seq, tuple(rids), [accepted[r] for r in rids]


def _check_mixed(node: _MixedNode, out: dict, rids, prompts: dict[int, int]) -> None:
    """The verify rows write and read at their true lengths, as on a verify step; each prompt row at its own
    tokens from zero, on its own pages."""
    verify = {key: values[: len(rids) * SPAN] for key, values in out.items()}
    _check_step(node, type("Out", (), verify), rids)
    t = len(rids) * SPAN
    for rid, n in prompts.items():
        pages = node.stream(rid).page_indices
        for j in range(n):
            assert out["pages"][t] == pages[j // PAGE], (rid, j)
            assert out["offsets"][t] == j % PAGE, (rid, j)
            assert out["positions"][t] == j, (rid, j)
            t += 1


def _verify_rounds(node: _MixedNode, rids, rounds: int, rng: random.Random, held, slot0: int = 0):
    for t in range(rounds):
        accepted = {rid: rng.randint(0, K) for rid in rids}
        out = node.step(rids, accepted, slot=(slot0 + t) % 2, bs=8)
        _check_step(node, out, rids)
        for rid in rids:
            node.true_len[rid] += accepted[rid] + 1
            assert node.device_len(rid) == node.true_len[rid]
        if held is not None:
            node.record(held)
        held = out
    return held


def test_an_async_mixed_steps_verify_rows_run_at_device_lengths_and_its_prompt_rows_at_host_ones():
    """Verify steps with random acceptance and asynchronous recording (a step's result lands after the next
    admit), a mixed step whose verify rows hold unapplied results while two prompts join, then verify steps over
    every row: each step writes and reads at the true lengths, and once everything is applied the host holds
    them."""
    rng = random.Random(5)
    node = _MixedNode({1: 5, 2: 3, 3: 10})
    rids = [1, 2, 3]
    held = _verify_rounds(node, rids, 3, rng, None)
    prompts = {4: 6, 5: 1}
    accepted = {rid: rng.randint(0, K) for rid in rids}
    out, seq, step_rids, kept = node.mixed_step(rids, prompts, accepted, slot=1, bs=8, tokens=8 * SPAN + 16)
    _check_mixed(node, out, rids, prompts)
    for rid in rids:
        node.true_len[rid] += accepted[rid] + 1
        assert node.device_len(rid) == node.true_len[rid]
    node.record(held)
    held = type("Held", (), {"seq": seq, "rids": step_rids, "accepted": kept})
    rows = [*rids, *prompts]
    held = _verify_rounds(node, rows, 4, rng, held, slot0=0)
    node.record(held)
    node.take_back(rows)
    node.ledger.committed(None)
    node.take_back(rows)
    for rid in rows:
        assert node.stream(rid).stored_len == node.true_len[rid]
        assert node.res["rope"].position(rid, "main") == node.true_len[rid]
        assert not node.ledger.pending(rid)


def test_a_mixed_step_over_rows_with_unrecorded_results_keeps_their_slots_and_device_lengths():
    """A verify row whose last result isn't recorded when a mixed step admits it holds its slot, and the step's
    rewrite of a pending row leaves its length cell to the device; the prompt rows take fresh slots."""
    rng = random.Random(7)
    node = _MixedNode({1: 9, 2: 4})
    rids = [1, 2]
    held = _verify_rounds(node, rids, 2, rng, None)
    slots = {rid: node.planner.slot_of(rid) for rid in rids}
    assert all(node.ledger.unrecorded(rid) for rid in rids)
    accepted = {rid: rng.randint(0, K) for rid in rids}
    out, _, _, _ = node.mixed_step(rids, {7: 3}, accepted, slot=0, bs=4, tokens=4 * SPAN + 16)
    _check_mixed(node, out, rids, {7: 3})
    assert {rid: node.planner.slot_of(rid) for rid in rids} == slots
    del held


@pytest.mark.parametrize("hide", [0, 1], ids=["target", "dspark"])
def test_a_mixed_steps_ragged_attention_reads_its_verify_rows_off_the_tables_and_its_prompt_rows_off_the_host(hide):
    """FA3's ragged wrapper (`FA3RaggedWrapper`) under a mixed capture: the host plans every row from the KV
    plan, a verify row at its reserved length, and the resident gather rewrites the verify rows' pages and
    lengths from the tables, at their true lengths. The prompt row and the padding row keep the host's. The
    step stays host-planned (`CapturedResidentStep.host_planned`), for its prompt rows."""
    from mstar.engine.resources.attn.fa3_paged import FA3RaggedWrapper
    from mstar.engine.resources.attn.flashinfer import FlashInferManager
    from mstar.engine.resources.kv.plan import build_paged_indptrs, hide_tail_views

    node = _MixedNode({1: 5, 2: 9})
    rids, bs, tokens = [1, 2], 4, 4 * SPAN + 16
    held = None
    for t in range(2):
        # nothing accepted: each held result leaves the reserved length k past the true one
        out = node.step(rids, dict.fromkeys(rids, 0), slot=t % 2, bs=bs)
        for rid in rids:
            node.true_len[rid] += 1
        if held is not None:
            node.record(held)
        held = out
    bucket = BucketKey(graph_walk=MIXED, bs=bs, num_tokens=tokens)
    width = node.res["kv"].config.block_table_width(PAGE)
    wrapper = FA3RaggedWrapper(page_size=PAGE, table_width=width, device=CPU, batch_size=bs, max_tokens=tokens,
                               accelerator_graph=True, causal=hide == 0, verify_extent=SPAN - hide)
    manager = FlashInferManager.__new__(FlashInferManager)
    manager._kv_cache_name, manager._lagged_cg, manager._mixed_cg = "kv", {}, {(bucket, 0): wrapper}
    true = {rid: node.true_len[rid] for rid in rids}
    seen = {}

    def attend(captured, ctx):
        ((_, kv_out),) = ctx.plan_results["kv"].items()
        rows = [ctx.walk_of(rid) == WALK for rid in ctx.request_ids]
        views = hide_tail_views(kv_out.views, hide, PAGE, rows) if hide else kv_out.views
        wrapper.plan(**build_paged_indptrs(views, PAGE).to_kwargs_dict())
        seen["host"] = wrapper._seqused.tolist()
        manager.resident_gather(captured)
        seen["device"], seen["table"] = wrapper._seqused.tolist(), wrapper._page_table.clone()
        seen["host_planned"] = captured.host_planned

    node.mixed_step(rids, {7: 6}, dict.fromkeys(rids, 0), slot=0, bs=bs, tokens=tokens, attend=attend)
    assert seen["device"] == [true[1] + SPAN - hide, true[2] + SPAN - hide, 6, 0]
    assert seen["host"][2:] == [6, 0] and seen["host"][:2] != seen["device"][:2], "the host held the true lengths"
    for i, rid in enumerate([*rids, 7]):
        pages = node.stream(rid).page_indices
        n = -(-seen["device"][i] // PAGE)
        assert seen["table"][i, :n].tolist() == list(pages[:n]), rid
    assert seen["host_planned"]
