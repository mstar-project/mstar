"""Speculative decoding's bookkeeping, driven the way a serving step drives it,
on CPU tensors.

The real KV manager, AITER's plan staging, the recurrent pool, the GDN verify
wrapper and the speculative resource run their admit/plan/commit; a toy target
and a toy draft stand in for the layers. Both read their context back out of a
fake paged cache through the device-side addressing the resource writes (KV
write slots, the block table, per-row KV lengths), and a toy recurrent kernel
starts each row from the slot ``num_accepted`` names. So a wrong position, page,
length or start slot changes what the toy models see, and the test sees it:

* greedy speculation emits exactly what plain greedy decoding does, for k = 1..4,
  whether the host learns each step's acceptance before the next step is
  planned or one step late (the async worker), with page crossings, padding
  rows, a stop id and a token budget;
* a draft that is right except at known positions is accepted exactly up to
  the first of them, every step (the draft's own KV chain is addressed right);
* at every admit the host's KV length covers what the step writes, and the
  host gives back exactly what it over-counted (reserve, then trim);
* each request holds k + 1 recurrent slots and resumes from the one its last
  step accepted up to (slot re-pointing without a copy).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources.attn.aiter import (  # noqa: E402
    _TILE_TOKENS,
    _image_numel,
    _PlanImage,
    kernel_rows,
    stage_plan,
)
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.kv.plan import SINK_PAGE  # noqa: E402
from mstar.engine.resources.linear_attn.gdn_kernels import GDNKernels  # noqa: E402
from mstar.engine.resources.linear_attn.wrappers import GDNVerifyWrapper  # noqa: E402
from mstar.engine.resources.recurrent.config import (  # noqa: E402
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import SINK_SLOT, RecurrentStatePool  # noqa: E402
from mstar.engine.resources.spec_decode.config import SpecDecodeStep, SpecPhase  # noqa: E402
from mstar.engine.resources.spec_decode.resource import SpecDecodeResource  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "kv_prefix_decode_helpers", pathlib.Path(__file__).with_name("test_kv_prefix_decode.py"),
)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

CPU = torch.device("cpu")
PAGE = 4
VOCAB = 97
EOS = VOCAB - 1
NUM_PAGES = 512
PAD_COLS = -(-_TILE_TOKENS // PAGE)


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(H.manager_mod, "KVTransferManager", H._StubTransfer)


def next_token(seq) -> int:
    """The toy target: a deterministic function of the whole prefix, never EOS
    unless the test plants it."""
    acc = len(seq)
    for i, t in enumerate(seq):
        acc = (acc * 31 + (int(t) + 7) * (i + 3)) % 1_000_003
    return acc % (VOCAB - 1)


def greedy(prompt: list[int], budget: int, stop_at: int | None = None) -> list[int]:
    """Plain greedy decoding: the prompt's token, then ``budget`` more, cut
    after the first token at position ``stop_at`` (made EOS)."""
    seq, out = list(prompt), []
    for _ in range(budget + 1):
        t = EOS if stop_at is not None and len(seq) == stop_at else next_token(seq)
        out.append(t)
        seq.append(t)
        if t == EOS:
            break
    return out


class ToyState:
    """A recurrent state per slot: a running hash of the tokens it has seen."""

    @staticmethod
    def step(state: float, token: int) -> float:
        return (state * 1.000_37 + token + 1) % 10_007.0


class _NoKernels(GDNKernels):
    qk_l2norm = False

    def chunked_prefill(self):
        raise NotImplementedError

    def captured_prefill(self, bs, num_tokens):
        raise NotImplementedError

    def decode(self, bs):
        raise NotImplementedError

    def verify(self):
        return None


class Harness:
    """One node's resources and a fake two-layer token cache (layer 0 the
    target's, layer 1 the draft head's)."""

    def __init__(self, k: int, *, stop_at: int | None = None, wrong: frozenset[int] = frozenset()):
        self.k, self.t = k, k + 1
        self.stop_at = stop_at
        # absolute token positions the toy draft guesses wrong
        self.wrong = wrong
        self.kv = KVManager(
            cfg=PagedKVConfig(
                num_layers=2, num_kv_heads=1, head_dim=8, max_seq_len=1024,
                max_num_pages=NUM_PAGES, page_size=PAGE,
            ),
            name="kv", joint_comm_group=None, transfer_engine_info=None,
            device=CPU, dtype=torch.float32,
        )
        self.kv._cg_max_seq_len = 256
        self.spec = SpecDecodeResource(
            device=CPU, num_draft_tokens=k, max_requests=8, page_size=PAGE,
            kv_key="kv", attn_key="attn", stop_token_ids=(EOS,),
        )
        self.pool = RecurrentStatePool(CPU, RecurrentStateConfig(
            num_layers=1,
            blocks={
                "state": RecurrentBlockConfig(shape=(1,), dtype=torch.float64),
                "conv": RecurrentBlockConfig(shape=(2, 3), dtype=torch.float32, speculative_window=True),
            },
            max_slots=1 + 8 * (k + 1),
            spec_tokens=k,
        ))
        self.gdn: dict = {}
        self.cache = torch.full((NUM_PAGES, PAGE, 2), -1, dtype=torch.long)
        self.slot = 0
        self.max_seen_writes = 0

    # one step's resources, as the runner sweeps them

    def _ctx(self, rids, padded, walk, bucket=None):
        self.slot ^= 1
        lease = None if bucket is None else SlotLease(slot=self.slot, bucket=bucket)
        ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=self.slot, capture=False,
                          slot_lease=lease)
        if padded is not None:
            ctx.set_padded_rids(tuple(padded))
        return ctx

    def _attn(self, kv_out, lease):
        rows = kernel_rows(kv_out.cpu_indptrs, PAGE)
        num_rows = rows.qo_indptr.shape[0] - 1
        width = NUM_PAGES // 4 + PAD_COLS if lease is not None else int(np.diff(rows.kv_indptr).max()) + PAD_COLS
        image = _PlanImage(_image_numel(num_rows, width), CPU)
        return stage_plan(image, rows, width, PAD_COLS, self.t, 1024)

    def prefill(self, prompts: dict[int, list[int]], budgets: dict[int, int]):
        rids = list(prompts)
        for rid in rids:
            self.kv.ingest_request(rid, KVReqConfig())
            self.spec.ingest_request(rid)
            self.pool.ingest_request(rid)
        segs = tuple(Segment(rid, "main", len(prompts[rid])) for rid in rids)
        ctx = self._ctx(rids, None, "prefill_text")
        kv_step = KVStep(segments=segs, commit_adjust={(r, "main"): self.spec.seed_lookahead for r in rids})
        spec_step = SpecDecodeStep(segments=segs, phase=SpecPhase.SEED, budgets=budgets)
        pool_step = RecurrentStep(segments=segs)
        assert self.kv.admit(kv_step, ctx).ok and self.pool.admit(pool_step, ctx).ok
        assert self.spec.admit(spec_step, ctx).ok
        ctx.plan_results["kv"] = self.kv.plan(kv_step, ctx)
        ctx.plan_results["gdn_state"] = self.pool.plan(pool_step, ctx)
        self.spec.plan(spec_step, ctx)
        # the forward: target tokens and draft-layer tokens at prompt positions,
        # each row's recurrent state over its prompt, then the seed
        state = self.pool.block("state", 0)
        first, drafts = [], []
        for i, rid in enumerate(rids):
            prompt = prompts[rid]
            t0 = EOS if self.stop_at == len(prompt) else next_token(prompt)
            view = ctx.plan_results["kv"]["main"].views[i]
            for p, tok in enumerate(prompt):
                page = view.page_idxs[p // PAGE]
                self.cache[page, p % PAGE, 0] = tok
                self.cache[page, p % PAGE, 1] = (prompt + [t0])[p + 1]
            h = 0.0
            for tok in prompt:
                h = ToyState.step(h, tok)
            state[self.pool._slots[rid]["main"].index, 0] = h
            first.append(t0)
            drafts.append(self._draft_guess(prompt + [t0], len(prompt) + 1))
        self.spec.seed(torch.tensor(first), torch.tensor(drafts))
        self.kv.commit(kv_step, ctx)
        self.pool.commit(pool_step, ctx)
        self.spec.commit(spec_step, ctx)
        return first

    def _draft_guess(self, seq: list[int], position: int) -> int:
        """The toy draft's guess for the token at ``position``, from ``seq``
        (the tokens before it); wrong where the test says."""
        true = EOS if self.stop_at == position else next_token(seq[:position])
        return (true + 1) % (VOCAB - 1) if position in self.wrong else true

    def verify(self, rids: list[int], pad: int = 0):
        """One verify step and its draft passes; returns (tokens, accepted, done)
        rows for the real requests, and checks the reservation invariant."""
        padded = rids + [-(i + 1) for i in range(pad)]
        t = self.t
        retract = {r: self.spec.pending_retract(r) for r in rids if self.spec.pending_retract(r)}
        segs = tuple(Segment(r, "main", t) for r in padded)
        bucket = BucketKey("decode", len(padded), len(padded) * t) if pad else None
        ctx = self._ctx(rids, padded if pad else None, "decode", bucket)
        lease = ctx.slot_lease
        kv_step = KVStep(segments=segs, commit_adjust={(r, "main"): -g for r, g in retract.items()})
        pool_step = RecurrentStep(segments=segs)
        spec_step = SpecDecodeStep(segments=segs, phase=SpecPhase.VERIFY, retract=retract)
        assert self.kv.admit(kv_step, ctx).ok and self.pool.admit(pool_step, ctx).ok
        assert self.spec.admit(spec_step, ctx).ok
        ctx.plan_results["kv"] = kv_out = self.kv.plan(kv_step, ctx)
        ctx.plan_results["attn"] = {"main": self._attn(kv_out["main"], lease)}
        ctx.plan_results["gdn_state"] = addressing = self.pool.plan(pool_step, ctx)
        plan = self.spec.plan(spec_step, ctx)
        wrapper = self.gdn.setdefault(lease is not None, GDNVerifyWrapper(
            CPU, pad_slot_id=-1, tokens_per_row=t, bs=len(padded) if pad else None,
            accelerator_graph=lease is not None, null_slot_id=SINK_SLOT, kernels=_NoKernels(),
        ))
        wrapper.plan(addressing["main"].host_slot_groups, plan.num_accepted)

        ids, positions = self.spec.verify_inputs()
        b = len(padded)
        base = positions.view(b, t)[:, 0]
        views = kv_out["main"].views
        for i in range(len(rids)):
            # the host's optimistic length covers every write of this step
            assert views[i].length >= int(base[i]) + 2 * self.k, (views[i].length, int(base[i]))
            assert len(views[i].page_idxs) * PAGE >= int(base[i]) + 2 * self.k
        self._write(plan, ids, layer=0)
        self._recurrent(wrapper, ids, b)
        targets = self._target(plan, b, positions)
        out = self.spec.accept_tokens(targets)
        drafted = out.target.reshape(-1)
        keep = self.spec.draft_slots()
        for j in range(self.k):
            where = self.spec.draft_address(j)
            self._write_flat(where.slots, drafted, layer=1)
            query = where.positions[keep] if j == 0 else where.positions
            guess = self._draft(plan, b, query, where.kv_lens)
            drafted = self.spec.record_draft(j, torch.nn.functional.one_hot(guess, VOCAB).float())
        self.kv.commit(kv_step, ctx)
        self.pool.commit(pool_step, ctx)
        self.spec.commit(spec_step, ctx)
        n = len(rids)
        return out.tokens[:n], out.accepted[:n], out.done[:n]

    def current_state(self, rid: int) -> float:
        """The recurrent state the request's next step starts from."""
        group = self.pool._slots[rid]["main"].group
        accepted = int(self.spec._num_acc[self.spec._row_of[rid]])
        return float(self.pool.block("state", 0)[group[accepted - 1], 0])

    def _write(self, plan, tokens, layer):
        pages, offsets = plan.write_pages, plan.write_offsets
        real = pages != SINK_PAGE
        self.cache[pages[real], offsets[real], layer] = tokens[real]

    def _write_flat(self, slots, tokens, layer):
        """A draft pass's KV write, one flat slot per token."""
        pages, offsets = torch.div(slots, PAGE, rounding_mode="floor"), slots % PAGE
        real = pages != SINK_PAGE
        self.cache[pages[real], offsets[real], layer] = tokens[real]

    def _read(self, plan, row: int, upto: int, layer: int) -> list[int]:
        """Tokens of ``layer`` at positions ``0 .. upto - 1`` of a row, through
        the block table the step's attention reads."""
        table = plan.block_table[row]
        return [int(self.cache[int(table[p // PAGE]), p % PAGE, layer]) for p in range(upto)]

    def _target(self, plan, b, positions) -> torch.Tensor:
        """Query i of row r sees keys 0 .. kv_len - (k + 1) + i, causal from
        the bottom right as the kernel aligns it."""
        t = self.t
        lens = plan.kv_lens
        out = torch.zeros(b * t, dtype=torch.long)
        for r in range(b):
            if not bool(plan.image.real[r]):
                continue
            for i in range(t):
                seen = int(lens[r]) - t + i + 1
                ctx_tokens = self._read(plan, r, seen, layer=0)
                assert -1 not in ctx_tokens, f"row {r} attends an unwritten key"
                out[r * t + i] = EOS if self.stop_at == seen else next_token(ctx_tokens)
        return out

    def _draft(self, plan, b, positions, kv_lens) -> torch.Tensor:
        """A row's one query, at position q, reads the draft layer's keys 0 ..
        q (the token after each) and guesses the token at q + 2."""
        out = torch.zeros(b, dtype=torch.long)
        for r in range(b):
            if not bool(plan.image.real[r]):
                continue
            seen = int(kv_lens[r])
            q = int(positions[r])
            assert seen == q + 1, "a draft query must see its own position and none after"
            seq = self._read(plan, r, 1, layer=0) + self._read(plan, r, seen, layer=1)
            out[r] = self._draft_guess(seq, q + 2) if -1 not in seq else 0
        return out

    def _recurrent(self, wrapper: GDNVerifyWrapper, ids, b):
        """The verify kernel's contract on the toy state: start from the
        accepted slot, keep the state after every token."""
        plan = wrapper._plan_state
        state = self.pool.block("state", 0)
        for r in range(b):
            row = plan.slot_map[r]
            if int(row[0]) < 0:
                continue
            h = float(state[int(row[int(plan.num_accepted[r]) - 1]), 0])
            for i in range(self.t):
                h = ToyState.step(h, int(ids[r * self.t + i]))
                state[int(row[i]), 0] = h


def run(h: Harness, prompts: dict[int, list[int]], budget: int, lag: int, pad: int = 0):
    """Generate to completion; ``lag`` steps between a step and the host
    learning what it accepted. Returns each request's emitted tokens and the
    accepted count of every step."""
    rids = list(prompts)
    first = h.prefill(prompts, {r: budget for r in rids})
    emitted = {r: [first[i]] for i, r in enumerate(rids)}
    live = [r for i, r in enumerate(rids) if first[i] != EOS]
    pending: list[tuple[list[int], list[int]]] = []
    history: dict[int, list[int]] = {r: [] for r in rids}
    for _ in range(4 * budget + 8):
        if not live:
            break
        tokens, accepted, done = h.verify(live, pad=pad)
        pending.append((list(live), accepted.tolist()))
        if len(pending) > lag:
            h.spec.learn(*pending.pop(0))
        still = []
        for i, rid in enumerate(live):
            row = [int(x) for x in tokens[i] if int(x) >= 0]
            emitted[rid] += row
            history[rid].append(int(accepted[i]))
            if not bool(done[i]):
                still.append(rid)
                # it has run every committed token but the last emitted one
                expect = 0.0
                for tok in prompts[rid] + emitted[rid][:-1]:
                    expect = ToyState.step(expect, tok)
                assert h.current_state(rid) == pytest.approx(expect), rid
        live = still
    assert not live, "generation never finished"
    return emitted, history


PROMPTS = {0: [3, 1, 4, 1, 5, 9, 2], 1: [2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4]}


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("lag", [0, 1])
def test_greedy_speculation_reproduces_greedy(k, lag):
    budget = 23
    h = Harness(k, wrong=frozenset(range(9, 200, 5)))
    emitted, history = run(h, PROMPTS, budget, lag)
    for rid, prompt in PROMPTS.items():
        assert emitted[rid] == greedy(prompt, budget), rid
        assert all(1 <= a <= k + 1 for a in history[rid])


@pytest.mark.parametrize("k", [1, 3, 4])
def test_padding_rows_leave_real_rows_alone(k):
    budget = 17
    h = Harness(k, wrong=frozenset(range(12, 200, 4)))
    emitted, _ = run(h, PROMPTS, budget, lag=1, pad=2)
    for rid, prompt in PROMPTS.items():
        assert emitted[rid] == greedy(prompt, budget), rid


@pytest.mark.parametrize("k", [2, 3, 4])
def test_a_perfect_draft_is_accepted_whole(k):
    h = Harness(k)
    budget = 6 * (k + 1)
    _, history = run(h, {0: PROMPTS[0]}, budget, lag=1)
    # The prompt's step drafts one token (the rest of the first row repeat
    # it), so the first verify takes at most two; every later step but the
    # last (cut by the budget) takes all k + 1.
    assert history[0][0] <= 2
    assert history[0][1:-1] == [k + 1] * (len(history[0]) - 2)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_acceptance_stops_at_the_first_wrong_draft(k):
    wrong = frozenset({11, 12, 17, 23, 24, 25, 31})
    h = Harness(k, wrong=wrong)
    prompt = PROMPTS[0]
    _, history = run(h, {0: prompt}, 30, lag=1)
    # the first verify's x_0 (the prompt's token) sits at len(prompt); its
    # drafts past the first repeat it (see the perfect-draft test)
    base = len(prompt) + history[0][0]
    for accepted in history[0][1:-1]:
        drafts = [base + i for i in range(1, k + 1)]
        expect = 1
        for pos in drafts:
            if pos in wrong:
                break
            expect += 1
        assert accepted == expect, (base, accepted, expect)
        base += accepted


@pytest.mark.parametrize("k", [1, 3])
def test_a_stop_id_inside_a_step_ends_it_there(k):
    prompt = PROMPTS[1]
    stop_at = len(prompt) + 6
    h = Harness(k, stop_at=stop_at)
    emitted, _ = run(h, {1: prompt}, 40, lag=1)
    assert emitted[1] == greedy(prompt, 40, stop_at=stop_at)
    assert emitted[1][-1] == EOS


@pytest.mark.parametrize("k", [2, 4])
def test_host_gives_back_exactly_what_it_overcounted(k):
    h = Harness(k, wrong=frozenset(range(10, 300, 3)))
    prompt = PROMPTS[0]
    rid = 0
    h.prefill({rid: prompt}, {rid: 40})
    stream = h.kv._streams[rid]["main"]
    assert stream.stored_len == len(prompt) + k - 1
    true_len = len(prompt)
    for _ in range(6):
        _, accepted, _ = h.verify([rid])
        true_len += int(accepted[0])
        # nothing learned yet: the step committed its whole row
        assert stream.stored_len == true_len + (k - 1) + (k + 1 - int(accepted[0]))
        h.spec.learn([rid], [int(accepted[0])])
        assert h.spec.pending_retract(rid) == k + 1 - int(accepted[0])
        # the next verify gives it back in its commit, then over-counts anew
    _, accepted, _ = h.verify([rid])
    true_len += int(accepted[0])
    assert h.spec.pending_retract(rid) == 0
    assert stream.stored_len == true_len + (k - 1) + (k + 1 - int(accepted[0]))


def test_draft_passes_attend_one_query_a_row():
    """Pass 0 writes every verify position and attends slot n alone; a later
    pass runs one token a row, at the position after it. Both reach the
    attention with each row's keys up to and including its query."""
    k, rows, heads, kv_heads, dim = 2, 2, 2, 1, 8
    t = k + 1
    spec = SpecDecodeResource(device=CPU, num_draft_tokens=k, max_requests=rows, page_size=PAGE,
                              kv_key="kv", attn_key="attn", stop_token_ids=(EOS,))
    for r in range(rows):
        spec.ingest_request(r)
    image = spec._new_image(rows)
    image.rows.copy_(torch.tensor([spec._row_of[r] for r in range(rows)], dtype=torch.int32))
    image.real.fill_(1)
    width = 16
    table = (1 + torch.arange(rows * width, dtype=torch.int32)).view(rows, width)
    from mstar.engine.resources.spec_decode.resource import SpecDecodePlan

    spec._current = SpecDecodePlan(
        phase=SpecPhase.VERIFY, num_rows=rows, image=image, block_table=table,
        kv_lens=torch.zeros(rows, dtype=torch.int32), write_pages=torch.zeros(rows * t, dtype=torch.long),
        write_offsets=torch.zeros(rows * t, dtype=torch.long),
    )
    regs = slice(1, rows + 1)
    spec._seq_len[regs] = torch.tensor([10, 21], dtype=torch.int32)
    spec._budget[regs] = 1 << 20
    # row 0 accepts everything, row 1 refuses its first draft
    spec._tokens[:, regs] = torch.tensor([[1, 2], [3, 4], [5, 6]])
    ids, _ = spec.verify_inputs()
    spec.accept_tokens(torch.tensor([3, 5, 9, 8, 8, 8]))

    calls = []

    class Attn:
        def attend_rows(self, q, layer, kv_lens):
            calls.append((q.clone(), kv_lens.tolist()))
            return q

    pages, slots_per_page = 32, PAGE
    layer = torch.zeros(pages, 2, slots_per_page, kv_heads, dim)
    spec._kv = type("KV", (), {"layer_view": lambda self, idx: layer})()
    spec._attn = Attn()
    cos, sin = torch.ones(64, 2), torch.zeros(64, 2)
    width_qkv = (2 * heads + 2 * kv_heads) * dim
    gen = torch.Generator().manual_seed(0)
    q_w, k_w = torch.zeros(dim), torch.zeros(dim)
    args = (q_w, k_w, cos, sin, heads, kv_heads, dim, 1e-6, True, 0)

    full = torch.randn(rows * t, width_qkv, generator=gen)
    out0, gate0 = spec.attend_draft(0, full, *args)
    assert out0.shape == (rows, heads, dim) and gate0.shape == (rows, heads, dim)
    # row 0's last accepted slot is 2, row 1's is 0
    assert spec.draft_slots().tolist() == [2, 3]
    assert calls[0][1] == [13, 22]
    pos_pages = [int(table[0, p // PAGE]) for p in (10, 11, 12)]
    written = layer[pos_pages, 0, [10 % PAGE, 11 % PAGE, 12 % PAGE]]
    assert written.abs().sum(dim=(1, 2)).min() > 0, "pass 0 writes every verify position of a row"

    one = torch.randn(rows, width_qkv, generator=gen)
    out1, _ = spec.attend_draft(1, one, *args)
    assert out1.shape == (rows, heads, dim)
    assert calls[1][1] == [14, 23]
    where = spec.draft_address(1)
    assert where.positions.tolist() == [13, 22]
    assert where.slots.tolist() == [int(table[r, p // PAGE]) * PAGE + p % PAGE for r, p in enumerate((13, 22))]


def test_attend_rows_reads_one_query_a_row_off_a_verify_plan(monkeypatch):
    from mstar.engine.resources.attn import aiter

    manager = aiter.AiterPagedAttentionManager.__new__(aiter.AiterPagedAttentionManager)
    manager._device = CPU
    manager._row_starts = None
    manager._reserve_row_starts(3)
    cu, lens = torch.tensor([0, 3, 6, 9], dtype=torch.int32), torch.zeros(3, dtype=torch.int32)
    table = torch.zeros(3, 8, dtype=torch.int32)
    manager._current_plans = {"main": aiter._Staged(cu, lens, table, 3, 1024, 9)}
    seen = {}
    monkeypatch.setattr(manager, "_unified", lambda q, plan, layer: seen.update(plan=plan) or q)
    kv_lens = torch.tensor([7, 8, 9], dtype=torch.int32)
    manager.attend_rows(torch.zeros(3, 2, 4), None, kv_lens)
    plan = seen["plan"]
    assert plan.cu_seqlens_q.tolist() == [0, 1, 2, 3]
    assert plan.seqused_k is kv_lens and plan.max_seqlen_q == 1 and plan.num_tokens == 3
    assert plan.block_table is table and plan.max_seqlen_k == 1024


def test_an_fp8_head_keeps_a_clear_argmax(monkeypatch):
    from mstar.engine.resources.spec_decode import draft_head as module

    class Platform:
        name = "test"

        def fp8_dtype(self):
            return torch.float8_e4m3fnuz

    monkeypatch.setattr(module, "current_platform", Platform())
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(256, 64, generator=gen, dtype=torch.bfloat16) * 0.02
    head = module.Fp8DraftHead(weight)
    assert head.weight.dtype == torch.float8_e4m3fnuz
    # rows along a row of the head: its own dot product stands well clear of the rest
    x = weight[[3, 200]].float() + 0.001 * torch.randn(2, 64, generator=gen)
    logits = head(x.to(torch.bfloat16))
    assert logits.shape == (2, 256) and logits.dtype == torch.bfloat16
    assert logits.argmax(dim=-1).tolist() == [3, 200]


def test_requests_hold_slot_groups_and_return_them():
    k = 3
    h = Harness(k)
    free = h.pool.num_free_slots
    h.prefill({0: PROMPTS[0], 1: PROMPTS[1]}, {0: 4, 1: 4})
    groups = [h.pool._slots[r]["main"].group for r in (0, 1)]
    assert all(len(g) == k + 1 for g in groups)
    assert len(set(groups[0]) | set(groups[1])) == 2 * (k + 1)
    assert h.pool.num_free_slots == free - 2 * (k + 1)
    # the conv window holds k more columns; the state block is as declared
    assert h.pool.block("conv", 0).shape[-1] == 3 + k
    assert h.pool.block("state", 0).shape[-1] == 1
    for r in (0, 1):
        h.pool.remove_request(r)
        h.spec.remove_request(r)
    assert h.pool.num_free_slots == free
    assert h.spec.pending_retract(0) == 0 and not h.spec._row_of


def _filtered(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    """The sampler's distribution: top-k, then top-p over the renormalised rest."""
    probs = (logits.double() / temperature).softmax(-1)
    order = probs.argsort(descending=True)
    keep = torch.zeros_like(probs, dtype=torch.bool)
    if top_k > 0:
        keep[order[:top_k]] = True
    else:
        keep[:] = True
    probs = torch.where(keep, probs, 0.0)
    probs = probs / probs.sum()
    sorted_p = probs[order]
    mass_before = sorted_p.cumsum(0) - sorted_p
    cut = torch.zeros_like(keep)
    cut[order] = mass_before < top_p
    probs = torch.where(cut & keep, probs, 0.0)
    return probs / probs.sum()


def _reference_draw(logits, temperature, top_k, top_p, seed, offset):
    """`draw_tokens` on CPU: each row's filtered distribution, drawn with a
    generator seeded from its seed and offset (argmax at temperature 0)."""
    out = torch.empty(logits.shape[0], dtype=torch.long)
    for i in range(logits.shape[0]):
        t = float(temperature[i])
        if t == 0:
            out[i] = logits[i].argmax()
            continue
        k = int(top_k[i])
        probs = _filtered(logits[i], t, float(top_p[i]), k if k < logits.shape[1] else 0)
        gen = torch.Generator().manual_seed(int(seed[i]) * 1_000_003 + int(offset[i]) * 7919 + i)
        out[i] = torch.multinomial(probs.float(), 1, generator=gen)[0]
    return out


@pytest.mark.parametrize("params", [(1.0, 1.0, 0), (0.7, 0.9, 0), (1.3, 1.0, 5), (0.8, 0.6, 4)])
def test_sampled_verification_follows_the_target_distribution(params, monkeypatch):
    """k = 1 against fixed logits: the first emitted token is a draw from the
    target's filtered distribution at position 0, whether or not the draft is
    accepted, and the token after an accepted draft one from position 1's."""
    from mstar.engine.resources.spec_decode import resource as spec_mod

    monkeypatch.setattr(spec_mod.sampling, "draw_tokens", _reference_draw)
    torch.manual_seed(0)
    temperature, top_p, top_k = params
    rows, vocab, draws = 400, 12, 30
    spec = SpecDecodeResource(device=CPU, num_draft_tokens=1, max_requests=rows, page_size=PAGE,
                              kv_key="kv", attn_key="attn", sampling=True)
    for r in range(rows):
        spec.ingest_request(r)
    image = spec._new_image(rows)
    image.rows.copy_(torch.tensor([spec._row_of[r] for r in range(rows)], dtype=torch.int32))
    image.real.fill_(1)
    table = torch.arange(1, rows * 4 + 1, dtype=torch.int32).view(rows, 4)
    state = torch.zeros(rows * 2, dtype=torch.long)
    from mstar.engine.resources.spec_decode.resource import SpecDecodePlan

    spec._current = SpecDecodePlan(
        phase=SpecPhase.VERIFY, num_rows=rows, image=image, block_table=table,
        kv_lens=torch.zeros(rows, dtype=torch.int32), write_pages=state, write_offsets=state.clone(),
    )
    regs = slice(1, rows + 1)
    spec._temperature[regs], spec._top_p[regs], spec._top_k[regs] = temperature, top_p, top_k
    spec._seed[regs] = torch.arange(rows)
    spec._budget[regs] = 1 << 20
    logits = torch.randn(2, vocab) * 2
    p0, p1 = (_filtered(logits[i], temperature, top_p, top_k) for i in range(2))
    draft = int(p0.argmax())
    first, after = torch.zeros(vocab), torch.zeros(vocab)
    for _ in range(draws):
        spec._tokens[:, regs] = torch.tensor([[3], [draft]])
        spec._seq_len[regs] = 10
        spec.verify_inputs()
        out = spec.accept(logits.repeat(rows, 1), sample=True)
        first += torch.bincount(out.tokens[:, 0], minlength=vocab)
        accepted = out.accepted == 2
        after += torch.bincount(out.tokens[accepted, 1], minlength=vocab)
    n = rows * draws
    tv_first = 0.5 * (first / n - p0).abs().sum()
    tv_after = 0.5 * (after / after.sum() - p1).abs().sum()
    assert tv_first < 0.02, (tv_first, first / n, p0)
    assert tv_after < 0.03, (tv_after, after / after.sum(), p1)
    # filtered-out tokens are never drawn
    assert not first[p0 == 0].any() and not after[p1 == 0].any()
    # the draft is accepted as often as the target would have drawn it
    assert abs(float(accepted.float().mean()) - float(p0[draft])) < 0.1
